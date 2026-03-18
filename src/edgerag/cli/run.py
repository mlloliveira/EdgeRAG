from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from edgerag.core.config import (
    DEFAULT_EMBEDDERS,
    DEFAULT_GENERATORS,
    PHASE1_KB_MODE,
    PHASE1_KB_RANDOM_PARAGRAPHS,
    PHASE1_MAX_QUESTIONS,
    PHASE1_SEED,
    PHASE1_SUBSET_MODE,
    PHASE1_TOP_KS,
    cfg_model_names,
    cfg_value,
    load_config,
    parse_int_list,
)
from edgerag.core.utils import safe_mkdir
from edgerag.llm.ollama_client import OllamaClient
from edgerag.pipelines.runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the preserved EdgeRAG local-RAG sweep on reduced KILT-NQ")
    parser.add_argument("--config", default="configs/phase1.json", help="Stable experiment config path")
    parser.add_argument("--data_dir", default=None, help="Directory to store datasets and KB")
    parser.add_argument("--index_dir", default=None, help="Directory to store FAISS indices")
    parser.add_argument("--results_dir", default=None, help="Directory to store results and resume state")
    parser.add_argument("--split", default=None, choices=["dev", "train"], help="KILT-NQ split")
    parser.add_argument("--max_questions", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--subset_mode", default=None, choices=["head", "random"], help="Question subset selection mode")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for deterministic sampling")
    parser.add_argument("--kb_mode", default=None, choices=["full", "gold_plus_random"], help="Knowledge-base mode")
    parser.add_argument("--kb_random_paragraphs", type=int, default=None, help="Random paragraph count for reduced KB")
    parser.add_argument("--kb_max_pages_debug", type=int, default=None, help="DEBUG ONLY: stop reading KB after N pages")
    parser.add_argument("--time_budget_hours", type=float, default=None, help="Run budget before graceful stop")
    parser.add_argument("--grace_minutes", type=float, default=None, help="Grace window after budget expires")
    parser.add_argument("--ollama_url", default=None, help="Ollama base URL")
    parser.add_argument("--keep_embedders", action="store_true", help="Do not delete embedders pulled by the script")
    parser.add_argument("--limit_passages_for_debug", type=int, default=None, help="Index only first N passages")
    parser.add_argument("--embedders_json", default=None, help="Optional JSON file listing embedding models")
    parser.add_argument("--generators_json", default=None, help="Optional JSON file listing generator models")
    parser.add_argument("--top_ks", default=None, help="Comma-separated top-k values")
    parser.add_argument("--context_lengths", default=None, help="Comma-separated context lengths")
    parser.add_argument("--dry_run", action="store_true", help="Resolve the experiment plan without running Ollama or FAISS. If local KILT files are absent, dry-run will not download them.")
    parser.add_argument("--verbose_stream", action="store_true", default=None, help="Write streamed token logs")
    parser.add_argument("--first_token_timeout_s", type=float, default=None, help="First-token watchdog timeout")
    parser.add_argument("--stream_timeout_s", type=float, default=None, help="Whole-stream watchdog timeout")
    parser.add_argument("--avoid_redundant_p0_p1", action="store_true", default=None, help="Run shared P0 and generator-only P1 once")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    data_dir = Path(cfg_value(args, cfg, "data_dir", ["data_dir"], "data"))
    index_dir = Path(cfg_value(args, cfg, "index_dir", ["index_dir"], "indices"))
    results_dir = Path(cfg_value(args, cfg, "results_dir", ["results_dir"], "results"))
    safe_mkdir(data_dir)
    safe_mkdir(index_dir)
    safe_mkdir(results_dir)

    embedders = cfg_model_names(cfg, ["embed_models", "embedders"], DEFAULT_EMBEDDERS)
    if args.embedders_json:
        with open(args.embedders_json, "r", encoding="utf-8") as f:
            embedders = json.load(f)
    generators = cfg_model_names(cfg, ["generator_models", "generators"], DEFAULT_GENERATORS)
    if args.generators_json:
        with open(args.generators_json, "r", encoding="utf-8") as f:
            generators = json.load(f)

    top_ks = parse_int_list(cfg_value(args, cfg, "top_ks", ["top_k_values", "top_ks"], PHASE1_TOP_KS))
    context_lengths = parse_int_list(cfg_value(args, cfg, "context_lengths", ["context_lengths"], [4096]))

    ollama_url = cfg_value(args, cfg, "ollama_url", ["ollama_url"], "http://localhost:11434")
    verbose_stream = bool(cfg_value(args, cfg, "verbose_stream", ["verbose_stream"], False))
    first_token_timeout_s = float(cfg_value(args, cfg, "first_token_timeout_s", ["first_token_timeout_s"], 300.0))
    stream_timeout_s = float(cfg_value(args, cfg, "stream_timeout_s", ["stream_timeout_s"], 600.0))
    avoid_redundant_p0_p1 = bool(cfg_value(args, cfg, "avoid_redundant_p0_p1", ["avoid_redundant_p0_p1"], True))
    split = cfg_value(args, cfg, "split", ["split"], "dev")
    subset_mode = cfg_value(args, cfg, "subset_mode", ["subset_mode"], PHASE1_SUBSET_MODE)
    seed = int(cfg_value(args, cfg, "seed", ["subset_seed", "seed"], PHASE1_SEED))
    kb_mode = cfg_value(args, cfg, "kb_mode", ["kb_mode"], PHASE1_KB_MODE)
    kb_random_paragraphs = int(cfg_value(args, cfg, "kb_random_paragraphs", ["kb_random_passages", "kb_random_paragraphs"], PHASE1_KB_RANDOM_PARAGRAPHS))
    time_budget_hours = float(cfg_value(args, cfg, "time_budget_hours", ["time_limit_hours", "time_budget_hours"], 2.0))
    grace_minutes = float(cfg_value(args, cfg, "grace_minutes", ["grace_minutes"], 20.0))
    max_q = cfg_value(args, cfg, "max_questions", ["max_questions"], PHASE1_MAX_QUESTIONS)
    if max_q is not None and int(max_q) <= 0:
        max_q = None

    ollama = OllamaClient(base_url=ollama_url, timeout=600, retries=6, retry_backoff_s=5.0, verbose=True)
    if not args.dry_run:
        print(f"[ollama] Preflight check against {ollama_url} ...")
        ollama.wait_until_ready(timeout_s=120.0)

    run(
        ollama=ollama,
        data_dir=data_dir,
        index_root_dir=index_dir,
        results_dir=results_dir,
        split=split,
        max_questions=max_q,
        subset_mode=subset_mode,
        seed=seed,
        kb_mode=kb_mode,
        kb_random_paragraphs=kb_random_paragraphs,
        kb_max_pages_debug=args.kb_max_pages_debug,
        time_budget_hours=time_budget_hours,
        grace_minutes=grace_minutes,
        embedders=embedders,
        generators=generators,
        top_ks=top_ks,
        context_lengths=context_lengths,
        keep_embedders=args.keep_embedders,
        limit_passages_for_debug=args.limit_passages_for_debug,
        dry_run=args.dry_run,
        verbose_stream=verbose_stream,
        first_token_timeout_s=first_token_timeout_s,
        stream_timeout_s=stream_timeout_s,
        avoid_redundant_p0_p1=avoid_redundant_p0_p1,
    )


if __name__ == "__main__":
    main()
