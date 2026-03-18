from __future__ import annotations

from pathlib import Path
from typing import Optional

from edgerag.core.config import load_config

DEFAULT_CONFIG_PATH = Path("configs/phase1.json")
DEFAULT_RESULTS_PATH = Path("results_phase1/results.jsonl")
BASE_ANALYSIS_OUTDIR = Path("artifacts/paper/sample_outputs/base_analysis")
SAS_ANALYSIS_OUTDIR = Path("artifacts/paper/sample_outputs/sas_analysis")


def resolve_config_path(config_path: Optional[Path]) -> Path:
    if config_path is not None:
        return config_path
    return DEFAULT_CONFIG_PATH


def resolve_results_path(results_path: Optional[Path], config_path: Optional[Path]) -> Path:
    if results_path is not None:
        return results_path
    cfg = load_config(resolve_config_path(config_path))
    results_dir = cfg.get("results_dir") if isinstance(cfg, dict) else None
    if results_dir:
        return Path(results_dir) / "results.jsonl"
    return DEFAULT_RESULTS_PATH
