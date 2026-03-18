from __future__ import annotations

from pathlib import Path

from edgerag.core.runtime import build_runtime_metadata


def test_build_runtime_metadata_contains_elapsed_and_paths(tmp_path: Path):
    payload = build_runtime_metadata(
        start_ts=0.0,
        budget_s=3600.0,
        grace_s=300.0,
        results_path=tmp_path / "results.jsonl",
        resume_path=tmp_path / "resume_state.json",
        stream_dir=tmp_path / "live_streams",
        phase="running",
        completed_keys=4,
        extra={"split": "dev"},
    )
    assert payload["phase"] == "running"
    assert payload["completed_run_keys"] == 4
    assert payload["results_path"].endswith("results.jsonl")
    assert payload["resume_path"].endswith("resume_state.json")
    assert payload["live_stream_dir"].endswith("live_streams")
