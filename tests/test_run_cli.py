from __future__ import annotations

import json
from pathlib import Path

from edgerag.cli.run import build_parser
from edgerag.pipelines.runner import run


class FakeOllama:
    pass


def test_run_parser_defaults_match_public_config():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.config == "configs/phase1.json"
    assert args.dry_run is False


def test_dry_run_without_local_kilt_data_writes_metadata(tmp_path: Path):
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "indices"
    results_dir = tmp_path / "results"

    run(
        ollama=FakeOllama(),
        data_dir=data_dir,
        index_root_dir=index_dir,
        results_dir=results_dir,
        split="dev",
        max_questions=500,
        subset_mode="random",
        seed=123,
        kb_mode="gold_plus_random",
        kb_random_paragraphs=100000,
        kb_max_pages_debug=None,
        time_budget_hours=2.0,
        grace_minutes=20.0,
        embedders=["e1", "e2"],
        generators=["g1"],
        top_ks=[5, 10],
        context_lengths=[4096],
        keep_embedders=True,
        limit_passages_for_debug=None,
        dry_run=True,
        verbose_stream=False,
        first_token_timeout_s=300.0,
        stream_timeout_s=600.0,
        avoid_redundant_p0_p1=True,
    )

    metadata_path = results_dir / "run_metadata.json"
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "dry_run_resolved"
    assert payload["local_data_present"] is False
    assert payload["question_count"] == 500
    assert payload["kb_mode"] == "gold_plus_random"
    assert payload["generator_count"] == 1
    assert payload["embedder_count"] == 2
