from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from edgerag.core.utils import append_jsonl, now_iso


def append_failure_record(
    results_path: Path,
    *,
    kb_mode: str,
    kb_tag: str,
    kb_random_paragraphs: int,
    subset_mode: str,
    seed: int,
    pipeline: str,
    generator: str,
    embedder: str,
    top_k: int,
    context_length: int,
    ex: Any,
    query_used: str,
    retrieved_ids: List[str],
    retrieved_scores: List[float],
    answer: str,
    rewrite_time_s: float,
    retrieval_time_s: float,
    gen_time_s: float,
    error_stage: str,
    error_message: str,
) -> None:
    record: Dict[str, Any] = {
        "timestamp": now_iso(),
        "status": "failed",
        "error_stage": error_stage,
        "error_message": error_message,
        "kb_mode": kb_mode,
        "kb_tag": kb_tag,
        "kb_random_paragraphs": int(kb_random_paragraphs) if kb_mode != "full" else 0,
        "subset_mode": subset_mode,
        "subset_seed": int(seed),
        "pipeline": pipeline,
        "generator": generator,
        "embedder": embedder,
        "top_k": int(top_k),
        "context_length": int(context_length),
        "question_id": ex.qid,
        "question": ex.question,
        "query_used": query_used,
        "retrieved_ids": retrieved_ids,
        "retrieved_scores": retrieved_scores,
        "answer": answer,
        "gold_answers": ex.gold_answers,
        "gold_provenance": ex.gold_provenance,
        "timings": {
            "rewrite_s": rewrite_time_s,
            "retrieve_s": retrieval_time_s,
            "generate_s": gen_time_s,
            "total_s": rewrite_time_s + retrieval_time_s + gen_time_s,
        },
        "metrics": {
            "em": 0.0,
            "f1": 0.0,
            "recall_at_k": 0.0,
            "r_precision": 0.0,
            "kilt_em_hit": 0.0,
            "kilt_f1_hit": 0.0,
            "kilt_em_all": 0.0,
            "kilt_f1_all": 0.0,
        },
    }
    append_jsonl(results_path, record)
