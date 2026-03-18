from __future__ import annotations

import json
from pathlib import Path

from edgerag.analysis.common import resolve_results_path
from edgerag.analysis.base import build_parser as build_base_parser
from edgerag.analysis.sas import build_parser as build_sas_parser


def test_resolve_results_path_from_config(tmp_path: Path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"results_dir": "results_phase1"}), encoding="utf-8")
    assert resolve_results_path(None, cfg) == Path("results_phase1") / "results.jsonl"


def test_base_parser_defaults_match_public_paths():
    parser = build_base_parser()
    args = parser.parse_args([])
    assert args.config == Path("configs/phase1.json")
    assert args.outdir == Path("artifacts/paper/sample_outputs/base_analysis")


def test_sas_parser_defaults_match_public_paths():
    parser = build_sas_parser()
    args = parser.parse_args([])
    assert args.config == Path("configs/phase1.json")
    assert args.outdir == Path("artifacts/paper/sample_outputs/sas_analysis")
