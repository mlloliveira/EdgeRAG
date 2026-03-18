from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PHASE1_MAX_QUESTIONS = 500
PHASE1_SUBSET_MODE = "random"
PHASE1_SEED = 123
PHASE1_KB_MODE = "gold_plus_random"
PHASE1_KB_RANDOM_PARAGRAPHS = 100_000
PHASE1_TOP_KS = "5,10"

DEFAULT_EMBEDDERS = [
    "qwen3-embedding:4b",
    "all-MiniLM-L6-v2",
    "nomic-embed-text:v1.5",
]

DEFAULT_GENERATORS = [
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3:8b",
    "qwen3:14b",
    "qwen3:30b",
]


def load_config(path: Optional[str | Path]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config JSON must contain a top-level object")
    return cfg


def cfg_model_names(cfg: Dict[str, Any], key_names: Sequence[str], default: Sequence[str]) -> List[str]:
    for key in key_names:
        if key not in cfg:
            continue
        raw = cfg[key]
        if not isinstance(raw, list):
            continue
        out: List[str] = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and item.get("name"):
                out.append(str(item["name"]))
        if out:
            return out
    return list(default)


def cfg_value(args: Any, cfg: Dict[str, Any], arg_name: str, cfg_keys: Sequence[str], default: Any) -> Any:
    value = getattr(args, arg_name)
    if value is not None:
        return value
    for key in cfg_keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def parse_int_list(raw: Any) -> List[int]:
    if isinstance(raw, list):
        return [int(x) for x in raw]
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]
