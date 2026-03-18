from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict


def now_iso() -> str:
    """Return a stable local timestamp used in result records."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, obj: Any) -> None:
    safe_mkdir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sanitize_for_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(text))[:160]


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_text_atomic(path: Path, text: str) -> None:
    safe_mkdir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl_debug(path: Path, obj: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
