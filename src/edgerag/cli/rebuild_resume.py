from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from edgerag.core.config import cfg_value, load_config
from edgerag.core.resume import RunKey
from edgerag.data.kilt import ensure_kilt_nq, load_kilt_nq_examples


def reconstruct_resume_state(qid_to_idx: Dict[str, int], rows: Iterable[dict]) -> tuple[Dict[str, int], int, int, int]:
    resume_state: Dict[str, int] = {}
    bad_rows = 0
    used_rows = 0
    skipped_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            bad_rows += 1
            continue
        qid = str(row.get("question_id", ""))
        if qid not in qid_to_idx:
            skipped_rows += 1
            continue
        pipeline = str(row.get("pipeline", ""))
        generator = str(row.get("generator", ""))
        embedder = str(row.get("embedder", ""))
        top_k = int(row.get("top_k", 0))
        context_length = int(row.get("context_length", 0))
        idx = qid_to_idx[qid]
        key = RunKey(
            pipeline=pipeline,
            generator=generator,
            embedder=embedder,
            top_k=top_k,
            context_length=context_length,
        )
        key_hash = key.to_str()
        resume_state[key_hash] = max(resume_state.get(key_hash, 0), idx + 1)
        used_rows += 1
    return resume_state, bad_rows, used_rows, skipped_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild resume_state.json from results.jsonl using the configured question subset")
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.json"), help="Stable config path")
    parser.add_argument("--results", type=Path, default=None, help="Path to results.jsonl; defaults to config results_dir/results.jsonl")
    parser.add_argument("--resume", type=Path, default=None, help="Path to resume_state.json; defaults to config results_dir/resume_state.json")
    parser.add_argument("--split", type=str, default=None, choices=["dev", "train"], help="Override KILT split")
    parser.add_argument("--max_questions", type=int, default=None, help="Override question count")
    parser.add_argument("--subset_mode", type=str, default=None, choices=["head", "random"], help="Override subset mode")
    parser.add_argument("--seed", type=int, default=None, help="Override subset seed")
    parser.add_argument("--no_backup", action="store_true", help="Do not create a .bak copy before overwriting")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    data_dir = Path(cfg_value(args, cfg, "data_dir", ["data_dir"], "data"))
    results_dir = Path(cfg_value(args, cfg, "results_dir", ["results_dir"], "results"))
    split = cfg_value(args, cfg, "split", ["split"], "dev")
    max_questions = int(cfg_value(args, cfg, "max_questions", ["max_questions"], 500))
    subset_mode = cfg_value(args, cfg, "subset_mode", ["subset_mode"], "random")
    seed = int(cfg_value(args, cfg, "seed", ["subset_seed", "seed"], 123))
    results_path = args.results or (results_dir / "results.jsonl")
    resume_path = args.resume or (results_dir / "resume_state.json")

    if not results_path.exists():
        raise FileNotFoundError(f"Could not find results file: {results_path}")

    nq_path, _ = ensure_kilt_nq(data_dir, split=split)
    examples = load_kilt_nq_examples(nq_path, max_questions=max_questions, subset_mode=subset_mode, seed=seed)
    qid_to_idx = {str(ex.qid): i for i, ex in enumerate(examples)}

    def row_iter():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    yield None

    resume_state, bad_rows, used_rows, skipped_rows = reconstruct_resume_state(qid_to_idx, row_iter())

    if resume_path.exists() and not args.no_backup:
        backup_path = resume_path.with_suffix(".json.bak")
        shutil.copy2(resume_path, backup_path)
        print(f"[info] Backed up old resume file to: {backup_path}")

    tmp_path = resume_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(resume_state, f, indent=2, ensure_ascii=False)
    tmp_path.replace(resume_path)

    print(f"[info] Loaded {len(examples)} examples from subset")
    print(f"[info] Used {used_rows} rows from results.jsonl")
    print(f"[info] Ignored {bad_rows} malformed rows")
    print(f"[info] Skipped {skipped_rows} rows whose qid was not in current subset")
    print(f"[ok] Rebuilt resume file: {resume_path}")


if __name__ == "__main__":
    main()
