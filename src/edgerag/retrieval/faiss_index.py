from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from edgerag.core.deps import require_optional
from edgerag.core.utils import atomic_write_json, safe_mkdir
from edgerag.data.kb import sqlite_passage_count
from edgerag.retrieval.embeddings import embed_texts


def build_faiss_index(
    ollama: Any,
    embed_model: str,
    sqlite_kb: Path,
    index_dir: Path,
    batch_size: int = 64,
    limit_passages: Optional[int] = None,
) -> Tuple[Path, Path]:
    faiss = require_optional("faiss", "faiss-cpu")
    safe_mkdir(index_dir)
    safe_name = embed_model.replace(":", "_").replace("/", "_")
    index_path = index_dir / f"index_{safe_name}.faiss"
    ids_path = index_dir / f"index_{safe_name}.ids.txt"
    meta_path = index_dir / f"index_{safe_name}.meta.json"
    ids_tmp_path = ids_path.with_suffix(ids_path.suffix + ".tmp")
    if index_path.exists() and ids_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            n = int(meta.get("passages", 0))
            if n > 0 and ids_path.stat().st_size > 0:
                print(f"[faiss] Reusing existing index for embedder={embed_model} ({n:,} passages)")
                return index_path, ids_path
            print(f"[faiss][warn] Existing index for embedder={embed_model} is empty; rebuilding")
        except Exception as e:
            print(f"[faiss][warn] Could not validate existing index for embedder={embed_model}: {e}; rebuilding")
        for p in [index_path, ids_path, meta_path, ids_tmp_path]:
            p.unlink(missing_ok=True)

    print(f"[faiss] Building index for embedder={embed_model} ...")
    probe = embed_texts(ollama, embed_model, ["dimension probe"], keep_alive="30m")
    dim = int(probe.shape[1])
    index = faiss.IndexFlatIP(dim)

    conn = sqlite3.connect(str(sqlite_kb))
    cur = conn.cursor()
    total_rows = sqlite_passage_count(sqlite_kb)
    target_rows = min(total_rows, int(limit_passages)) if limit_passages is not None else total_rows
    cur.execute("SELECT pid, text FROM passages")
    batch_pids: List[str] = []
    batch_texts: List[str] = []
    n_total = 0
    batch_no = 0
    start_t = time.time()
    last_report_t = start_t
    try:
        with open(ids_tmp_path, "w", encoding="utf-8") as idf:
            for pid, text in cur:
                if limit_passages is not None and n_total >= limit_passages:
                    break
                text = (text or "").strip()
                if not text:
                    continue
                batch_pids.append(pid)
                batch_texts.append(text)
                if len(batch_texts) >= batch_size:
                    batch_no += 1
                    t0 = time.time()
                    vecs = embed_texts(ollama, embed_model, batch_texts, keep_alive="30m")
                    embed_dt = time.time() - t0
                    index.add(vecs)
                    for p in batch_pids:
                        idf.write(p + "\n")
                    n_total += len(batch_pids)
                    batch_pids.clear()
                    batch_texts.clear()
                    now = time.time()
                    if batch_no == 1 or batch_no % 10 == 0 or (now - last_report_t) >= 30.0:
                        rate = n_total / max(now - start_t, 1e-9)
                        pct = (100.0 * n_total / max(target_rows, 1)) if target_rows else 0.0
                        print(f"[faiss][progress] emb={embed_model} | batch={batch_no} | indexed={n_total:,}/{target_rows:,} ({pct:.1f}%) | last_batch={embed_dt:.2f}s | avg_rate={rate:.1f} passages/s")
                        last_report_t = now
            if batch_texts:
                batch_no += 1
                t0 = time.time()
                vecs = embed_texts(ollama, embed_model, batch_texts, keep_alive="30m")
                embed_dt = time.time() - t0
                index.add(vecs)
                for p in batch_pids:
                    idf.write(p + "\n")
                n_total += len(batch_pids)
                print(f"[faiss][progress] emb={embed_model} | batch={batch_no} | indexed={n_total:,}/{target_rows:,} | last_batch={embed_dt:.2f}s")
            idf.flush()
            os.fsync(idf.fileno())
        os.replace(ids_tmp_path, ids_path)
        faiss.write_index(index, str(index_path))
        atomic_write_json(meta_path, {"embed_model": embed_model, "dim": dim, "passages": n_total})
        print(f"[faiss] Saved index: {index_path} ({n_total:,} passages, dim={dim}, elapsed={(time.time()-start_t)/60.0:.1f}m)")
        return index_path, ids_path
    except Exception:
        ids_tmp_path.unlink(missing_ok=True)
        raise
    finally:
        conn.close()


def load_faiss_index(index_path: Path, ids_path: Path) -> Tuple[Any, List[str]]:
    faiss = require_optional("faiss", "faiss-cpu")
    index = faiss.read_index(str(index_path))
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    return index, ids


def retrieve(
    ollama: Any,
    embed_model: str,
    index: Any,
    ids: Sequence[str],
    query: str,
    top_k: int,
) -> Tuple[List[str], List[float]]:
    qv = embed_texts(ollama, embed_model, [query], keep_alive="5m")
    scores, idxs = index.search(qv, top_k)
    idxs_list = idxs[0].tolist()
    scores_list = scores[0].tolist()
    out_ids = [ids[i] for i in idxs_list if 0 <= i < len(ids)]
    return out_ids, scores_list
