from __future__ import annotations

import hashlib
import heapq
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from edgerag.core.deps import require_optional
from edgerag.core.utils import atomic_write_json, safe_mkdir


def iter_kilt_pages(ks_json: Path) -> Iterator[Dict[str, Any]]:
    with open(ks_json, "rb") as fb:
        first_non_ws = b""
        while True:
            ch = fb.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_ws = ch
                break
    if first_non_ws == b"[":
        ijson = require_optional("ijson", "ijson")
        print("[kb] Detected JSON-array KILT knowledge source; using ijson streaming parser")
        with open(ks_json, "rb") as f:
            for page in ijson.items(f, "item"):
                if isinstance(page, dict):
                    yield page
    else:
        print("[kb] Detected JSONL/NDJSON KILT knowledge source; using line-by-line parser")
        with open(ks_json, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    page = json.loads(line)
                except Exception as e:
                    print(f"[kb][warn] Could not parse knowledge-source line {line_number}: {e}")
                    continue
                if isinstance(page, dict):
                    yield page


def sqlite_passage_count(sqlite_path: Path) -> int:
    if not sqlite_path.exists():
        return 0
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM passages")
        row = cur.fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def validate_existing_sqlite(sqlite_path: Path, label: str) -> bool:
    if not sqlite_path.exists():
        return False
    try:
        count = sqlite_passage_count(sqlite_path)
        if count > 0:
            print(f"[{label}] Reusing existing SQLite KB: {sqlite_path} ({count:,} passages)")
            return True
        print(f"[{label}][warn] Existing SQLite KB is empty; rebuilding: {sqlite_path}")
    except Exception as e:
        print(f"[{label}][warn] Existing SQLite KB is invalid ({e}); rebuilding: {sqlite_path}")
    sqlite_path.unlink(missing_ok=True)
    return False


def build_kilt_sqlite(ks_json: Path, sqlite_path: Path) -> None:
    safe_mkdir(sqlite_path.parent)
    if validate_existing_sqlite(sqlite_path, "kb"):
        return
    print(f"[kb] Building SQLite KB at {sqlite_path} ...")
    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("CREATE TABLE IF NOT EXISTS passages (pid TEXT PRIMARY KEY, wikipedia_id INTEGER, paragraph_id INTEGER, title TEXT, text TEXT)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_wid ON passages(wikipedia_id)")
    conn.commit()

    batch: List[Tuple[str, int, int, str, str]] = []
    batch_size = 2000
    for page_num, page in enumerate(iter_kilt_pages(ks_json), start=1):
        try:
            wid = int(page.get("wikipedia_id"))
        except Exception:
            continue
        title = page.get("wikipedia_title", "") or ""
        paragraphs = page.get("text", []) or []
        if not isinstance(paragraphs, list):
            continue
        for pid_idx, para in enumerate(paragraphs):
            if not isinstance(para, str) or not para.strip():
                continue
            pid = f"{wid}:{pid_idx}"
            batch.append((pid, wid, pid_idx, title, para.strip()))
            if len(batch) >= batch_size:
                cur.executemany(
                    "INSERT OR REPLACE INTO passages(pid, wikipedia_id, paragraph_id, title, text) VALUES (?,?,?,?,?)",
                    batch,
                )
                conn.commit()
                batch.clear()
        if page_num % 1000 == 0:
            print(f"[kb] Processed lines/pages: {page_num:,}")
    if batch:
        cur.executemany(
            "INSERT OR REPLACE INTO passages(pid, wikipedia_id, paragraph_id, title, text) VALUES (?,?,?,?,?)",
            batch,
        )
        conn.commit()
    total = sqlite_passage_count(sqlite_path)
    conn.close()
    print(f"[kb] Done. passages={total:,}")


def _hash64(seed: int, s: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b"|")
    h.update(s.encode("utf-8"))
    return int.from_bytes(h.digest(), "big", signed=False)


def build_reduced_kilt_sqlite(
    ks_json: Path,
    sqlite_path: Path,
    keep_pids: Sequence[str],
    random_paragraphs: int,
    seed: int,
    *,
    max_pages_debug: Optional[int] = None,
) -> None:
    safe_mkdir(sqlite_path.parent)
    if validate_existing_sqlite(sqlite_path, "kb"):
        return
    keep_set = set(str(x) for x in keep_pids)
    print(f"[kb] Building REDUCED SQLite KB at {sqlite_path} (gold={len(keep_set):,}, random={random_paragraphs:,}) ...")

    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("CREATE TABLE IF NOT EXISTS passages (pid TEXT PRIMARY KEY, wikipedia_id INTEGER, paragraph_id INTEGER, title TEXT, text TEXT)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_wid ON passages(wikipedia_id)")
    conn.commit()

    seen_gold: set[str] = set()
    heap: List[Tuple[int, str, int, int, str, str]] = []
    batch: List[Tuple[str, int, int, str, str]] = []
    batch_size = 2000

    for page_i, page in enumerate(iter_kilt_pages(ks_json), start=1):
        if max_pages_debug is not None and page_i > max_pages_debug:
            break
        try:
            wid = int(page.get("wikipedia_id"))
        except Exception:
            continue
        title = page.get("wikipedia_title", "") or ""
        paragraphs = page.get("text", []) or []
        if not isinstance(paragraphs, list):
            continue
        for pid_idx, para in enumerate(paragraphs):
            if not isinstance(para, str) or not para.strip():
                continue
            pid = f"{wid}:{pid_idx}"
            text = para.strip()
            if pid in keep_set:
                seen_gold.add(pid)
                batch.append((pid, wid, pid_idx, title, text))
                if len(batch) >= batch_size:
                    cur.executemany(
                        "INSERT OR REPLACE INTO passages(pid, wikipedia_id, paragraph_id, title, text) VALUES (?,?,?,?,?)",
                        batch,
                    )
                    conn.commit()
                    batch.clear()
                continue
            if random_paragraphs <= 0:
                continue
            h = _hash64(seed, pid)
            item = (-int(h), pid, wid, pid_idx, title, text)
            if len(heap) < random_paragraphs:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
        if page_i % 1000 == 0:
            print(f"[kb] Processed lines/pages: {page_i:,} | gold_seen={len(seen_gold):,} | heap={len(heap):,}")

    if batch:
        cur.executemany(
            "INSERT OR REPLACE INTO passages(pid, wikipedia_id, paragraph_id, title, text) VALUES (?,?,?,?,?)",
            batch,
        )
        conn.commit()
    random_rows = [(pid, wid, pid_idx, title, text) for (_, pid, wid, pid_idx, title, text) in heap]
    if random_rows:
        cur.executemany(
            "INSERT OR REPLACE INTO passages(pid, wikipedia_id, paragraph_id, title, text) VALUES (?,?,?,?,?)",
            random_rows,
        )
        conn.commit()
    total = sqlite_passage_count(sqlite_path)
    conn.close()

    gold_path = sqlite_path.with_suffix(".gold_pids.txt")
    rand_path = sqlite_path.with_suffix(".random_pids.txt")
    meta_path = sqlite_path.with_suffix(".meta.json")
    with open(gold_path, "w", encoding="utf-8") as f:
        for pid in sorted(keep_set):
            f.write(pid + "\n")
    with open(rand_path, "w", encoding="utf-8") as f:
        for (_, pid, *_rest) in sorted(heap, reverse=True):
            f.write(pid + "\n")
    missing = sorted(keep_set.difference(seen_gold))
    atomic_write_json(meta_path, {
        "kb_mode": "gold_plus_random",
        "seed": int(seed),
        "gold_count": int(len(keep_set)),
        "gold_seen_in_ks": int(len(seen_gold)),
        "gold_missing": int(len(missing)),
        "random_paragraphs": int(random_paragraphs),
        "random_selected": int(len(heap)),
        "note": "Random set selected by min-hash sampling over paragraph IDs.",
    })
    if missing:
        print(f"[kb][warn] {len(missing)} gold paragraphs were not found in the knowledge source. Example: {missing[:5]}")
    print(f"[kb] Reduced KB done. total_passages={total:,}")


def fetch_passages(sqlite_path: Path, pids: Sequence[str]) -> List[Tuple[str, str]]:
    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    out: List[Tuple[str, str]] = []
    for pid in pids:
        cur.execute("SELECT text FROM passages WHERE pid=?", (pid,))
        row = cur.fetchone()
        out.append((pid, row[0] if row else ""))
    conn.close()
    return out
