from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from edgerag.core.utils import atomic_write_json, sha1_str


@dataclass(frozen=True)
class RunKey:
    pipeline: str
    generator: str
    embedder: str
    top_k: int
    context_length: int

    def to_str(self) -> str:
        return sha1_str(f"{self.pipeline}|{self.generator}|{self.embedder}|{self.top_k}|{self.context_length}")


def load_resume_state(state_path: Path) -> Dict[str, int]:
    if not state_path.exists():
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in obj.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out


def save_resume_state(state_path: Path, state: Dict[str, int]) -> None:
    atomic_write_json(state_path, state)
