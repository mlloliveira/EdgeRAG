from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from edgerag.core.metrics import compute_exact_match, compute_f1, r_precision, recall_at_k
from edgerag.core.prompts import ANSWER_SYSTEM, BASELINE_SYSTEM, REWRITE_SYSTEM, format_rag_prompt
from edgerag.core.results import append_failure_record
from edgerag.core.resume import RunKey, load_resume_state, save_resume_state
from edgerag.core.runtime import should_stop, write_runtime_metadata
from edgerag.core.utils import append_jsonl, now_iso, safe_mkdir, sanitize_for_filename
from edgerag.data.kilt import ensure_kilt_nq, load_kilt_nq_examples, resolve_kilt_nq_paths
from edgerag.data.kb import (
    build_kilt_sqlite,
    build_reduced_kilt_sqlite,
    fetch_passages,
    sqlite_passage_count,
    validate_existing_sqlite,
)
from edgerag.llm.ollama_client import OllamaError, OllamaClient, make_stream_callback
from edgerag.retrieval.embeddings import is_hf_embedder
from edgerag.retrieval.faiss_index import build_faiss_index, load_faiss_index, retrieve


def run(
    *,
    ollama: OllamaClient,
    data_dir: Path,
    index_root_dir: Path,
    results_dir: Path,
    split: str,
    max_questions: Optional[int],
    subset_mode: str,
    seed: int,
    kb_mode: str,
    kb_random_paragraphs: int,
    kb_max_pages_debug: Optional[int],
    time_budget_hours: float,
    grace_minutes: float,
    embedders: Sequence[str],
    generators: Sequence[str],
    top_ks: Sequence[int],
    context_lengths: Sequence[int],
    keep_embedders: bool,
    limit_passages_for_debug: Optional[int],
    dry_run: bool = False,
    verbose_stream: bool = False,
    first_token_timeout_s: float = 300.0,
    stream_timeout_s: float = 600.0,
    avoid_redundant_p0_p1: bool = True,
) -> None:
    safe_mkdir(data_dir)
    safe_mkdir(results_dir)
    safe_mkdir(index_root_dir)
    results_path = results_dir / "results.jsonl"
    resume_path = results_dir / "resume_state.json"
    runtime_metadata_path = results_dir / "run_metadata.json"
    stream_dir = results_dir / "live_streams"
    safe_mkdir(stream_dir)

    print(f"[data] Generators={len(generators)} | Embedders={len(embedders)} | top_ks={list(top_ks)} | context_lengths={list(context_lengths)}")
    print(f"[data] verbose_stream={verbose_stream} | live_stream_dir={stream_dir}")
    print(f"[data] first_token_timeout_s={first_token_timeout_s} | stream_timeout_s={stream_timeout_s}")

    kb_mode = (kb_mode or "full").strip().lower()
    if kb_mode not in {"full", "gold_plus_random"}:
        raise ValueError("kb_mode must be 'full' or 'gold_plus_random'")

    start_ts = time.time()
    budget_s = time_budget_hours * 3600.0
    grace_s = grace_minutes * 60.0
    resume: Dict[str, int] = {}
    resolved_kb_tag: Optional[str] = None
    resolved_question_count: Optional[int] = None

    def save_runtime_metadata(phase: str, **extra: Any) -> None:
        write_runtime_metadata(
            runtime_metadata_path,
            start_ts=start_ts,
            budget_s=budget_s,
            grace_s=grace_s,
            results_path=results_path,
            resume_path=resume_path,
            stream_dir=stream_dir,
            phase=phase,
            completed_keys=len(resume),
            extra={
                "split": split,
                "subset_mode": subset_mode,
                "seed": int(seed),
                "kb_mode": kb_mode,
                "kb_tag": resolved_kb_tag,
                "question_count": resolved_question_count,
                "generator_count": len(generators),
                "embedder_count": len(embedders),
                **extra,
            },
        )

    nq_path_expected, ks_path_expected = resolve_kilt_nq_paths(data_dir, split=split)

    if dry_run and (not nq_path_expected.exists() or not ks_path_expected.exists()):
        planned_questions = int(max_questions) if max_questions is not None else None
        planned_q = planned_questions if planned_questions is not None else "unknown"
        kb_tag = "full" if kb_mode == "full" else f"goldrand_q{planned_q}_r{int(kb_random_paragraphs)}_s{int(seed)}"
        resolved_kb_tag = kb_tag
        resolved_question_count = planned_questions
        sqlite_path = data_dir / ("kilt_knowledgesource.sqlite" if kb_mode == "full" else f"kilt_knowledgesource_{kb_tag}.sqlite")
        index_dir = index_root_dir / kb_tag
        safe_mkdir(index_dir)
        print("[dry-run] Resolved experiment plan without downloading KILT or building the KB/index.")
        print(f"[dry-run] expected_nq_path={nq_path_expected}")
        print(f"[dry-run] expected_ks_path={ks_path_expected}")
        print(f"[dry-run] local_data_present=False | planned_questions={planned_q}")
        print(f"[dry-run] split={split} kb_mode={kb_mode} kb_tag={kb_tag}")
        print(f"[dry-run] embedders={list(embedders)}")
        print(f"[dry-run] generators={list(generators)}")
        print(f"[dry-run] top_ks={list(top_ks)} context_lengths={list(context_lengths)}")
        print(f"[dry-run] avoid_redundant_p0_p1={avoid_redundant_p0_p1}")
        print(f"[dry-run] sqlite_path={sqlite_path}")
        print(f"[dry-run] index_dir={index_dir}")
        save_runtime_metadata(
            "dry_run_resolved",
            question_count=planned_questions,
            kb_tag=kb_tag,
            expected_nq_path=str(nq_path_expected),
            expected_ks_path=str(ks_path_expected),
            local_data_present=False,
            dry_run_note="Dry-run skipped KILT download and KB/FAISS construction because the local data files were absent.",
        )
        return

    nq_path, ks_path = ensure_kilt_nq(data_dir, split=split)
    examples = load_kilt_nq_examples(nq_path, max_questions=max_questions, subset_mode=subset_mode, seed=seed)
    print(f"[data] Loaded {len(examples)} examples from {nq_path.name} (subset_mode={subset_mode}, seed={seed})")

    gold_pids: List[str] = []
    for ex in examples:
        gold_pids.extend(ex.gold_provenance)
    gold_pids = list(dict.fromkeys(gold_pids))
    print(f"[data] Unique gold provenance paragraph ids: {len(gold_pids):,}")

    if kb_mode == "full":
        kb_tag = "full"
        sqlite_path = data_dir / "kilt_knowledgesource.sqlite"
    else:
        kb_tag = f"goldrand_q{len(examples)}_r{int(kb_random_paragraphs)}_s{int(seed)}"
        sqlite_path = data_dir / f"kilt_knowledgesource_{kb_tag}.sqlite"
    resolved_kb_tag = kb_tag
    resolved_question_count = len(examples)

    index_dir = index_root_dir / kb_tag
    safe_mkdir(index_dir)

    if dry_run:
        kb_passages = sqlite_passage_count(sqlite_path) if sqlite_path.exists() else 0
        print("[dry-run] Resolved experiment plan using local KILT files only (no Ollama, KB build, or FAISS execution).")
        print(f"[dry-run] split={split} questions={len(examples)} kb_mode={kb_mode} kb_tag={kb_tag}")
        print(f"[dry-run] embedders={list(embedders)}")
        print(f"[dry-run] generators={list(generators)}")
        print(f"[dry-run] top_ks={list(top_ks)} context_lengths={list(context_lengths)}")
        print(f"[dry-run] avoid_redundant_p0_p1={avoid_redundant_p0_p1}")
        print(f"[dry-run] sqlite_path={sqlite_path} | existing_passage_count={kb_passages}")
        print(f"[dry-run] index_dir={index_dir}")
        save_runtime_metadata(
            "dry_run_resolved",
            question_count=len(examples),
            kb_tag=kb_tag,
            expected_nq_path=str(nq_path),
            expected_ks_path=str(ks_path),
            local_data_present=True,
            kb_sqlite_exists=sqlite_path.exists(),
            kb_existing_passage_count=kb_passages,
        )
        return

    if kb_mode == "full":
        if not validate_existing_sqlite(sqlite_path, "kb"):
            build_kilt_sqlite(ks_path, sqlite_path)
    else:
        if not validate_existing_sqlite(sqlite_path, "kb"):
            build_reduced_kilt_sqlite(
                ks_json=ks_path,
                sqlite_path=sqlite_path,
                keep_pids=gold_pids,
                random_paragraphs=int(kb_random_paragraphs),
                seed=int(seed),
                max_pages_debug=kb_max_pages_debug,
            )
    print(f"[kb] Using KB sqlite: {sqlite_path}")
    print(f"[kb] Current passage count: {sqlite_passage_count(sqlite_path):,}")

    pulled_by_script: set[str] = set()
    for emb in embedders:
        print(f"[embed] Starting embedder: {emb}")
        if is_hf_embedder(emb):
            print(f"[embed] {emb} will be served by sentence-transformers (no Ollama pull needed)")
        else:
            if not ollama.model_exists(emb):
                print(f"[ollama] pulling embedder: {emb}")
                ollama.pull(emb)
                pulled_by_script.add(emb)
            else:
                print(f"[ollama] embedder already available locally: {emb}")
        print(f"[embed] Building/loading FAISS for: {emb}")
        try:
            build_faiss_index(
                ollama,
                emb,
                sqlite_path,
                index_dir,
                batch_size=64,
                limit_passages=limit_passages_for_debug,
            )
        except OllamaError as e:
            print(f"[fatal] Ollama failed while building index for embedder={emb}: {e}")
            print("[fatal] Tip: restart `ollama serve`, then rerun. Healthy completed indices will be reused.")
            save_runtime_metadata("fatal_index_build_failure", last_error_stage="index_build", embedder=emb)
            return
    indices: Dict[str, Tuple[Any, List[str]]] = {}
    for emb in embedders:
        print(f"[embed] Loading FAISS index into memory: {emb}")
        safe_name = emb.replace(":", "_").replace("/", "_")
        indices[emb] = load_faiss_index(index_dir / f"index_{safe_name}.faiss", index_dir / f"index_{safe_name}.ids.txt")
    print(f"[embed] Loaded {len(indices)} FAISS indices into memory")

    resume = load_resume_state(resume_path)
    if resume and (not results_path.exists() or results_path.stat().st_size == 0):
        print("[resume][warn] Resume checkpoints exist but results.jsonl is missing/empty. Resetting resume state to avoid skipping work.")
        resume = {}
        save_resume_state(resume_path, resume)
    print(f"[resume] Loaded {len(resume)} checkpoints")
    print(f"[resume] Results file exists={results_path.exists()} size={(results_path.stat().st_size if results_path.exists() else 0)} bytes")

    save_runtime_metadata("starting", kb_tag=kb_tag, question_count=len(examples))

    def cleanup_embedders() -> None:
        if keep_embedders:
            return
        for emb_to_delete in list(pulled_by_script):
            print(f"[ollama] deleting embedder (pulled by script): {emb_to_delete}")
            try:
                ollama.delete(emb_to_delete)
            except Exception as e:
                print(f"[warn] could not delete embedder {emb_to_delete}: {e}")

    def execute_config(
        *,
        pipeline: str,
        gen: str,
        emb: str,
        top_k: int,
        ctx_len: int,
        keep_alive: str,
        index: Any = None,
        ids: Optional[List[str]] = None,
    ) -> bool:
        print(f"[run] generator={gen} | embedder={emb} | top_k={top_k} | context_length={ctx_len} | pipeline={pipeline}")
        key = RunKey(pipeline=pipeline, generator=gen, embedder=emb, top_k=top_k, context_length=int(ctx_len))
        key_hash = key.to_str()
        start_i = resume.get(key_hash, 0)
        print(f"[resume] start_i={start_i} / {len(examples)} for key={key_hash}")
        if start_i >= len(examples):
            print("[resume] All questions already completed for this configuration; skipping")
            return False
        if pipeline in {"P0", "P2", "P3"} and (index is None or ids is None):
            raise ValueError(f"Pipeline {pipeline} requires retrieval index and ids")

        for i in range(start_i, len(examples)):
            if should_stop(start_ts, budget_s, grace_s):
                print("[time] Budget reached. Saving resume state and stopping.")
                resume[key_hash] = i
                save_resume_state(resume_path, resume)
                save_runtime_metadata("budget_reached", current_run_key=key_hash, current_question_index=i)
                return True
            ex = examples[i]
            if i == start_i or (i + 1) % 10 == 0:
                print(f"[progress] {pipeline} | gen={gen} | emb={emb} | top_k={top_k} | ctx={ctx_len} | q={i+1}/{len(examples)}")

            retrieved_ids: List[str] = []
            retrieved_scores: List[float] = []
            passages: List[Tuple[str, str]] = []
            retrieval_time_s = 0.0
            query_used = ex.question
            rewrite_time_s = 0.0

            if pipeline in {"P0", "P2", "P3"}:
                if pipeline == "P3":
                    t0 = time.time()
                    stream_file = stream_dir / f"rewrite_{sanitize_for_filename(gen)}_{sanitize_for_filename(emb)}_q{i+1}.txt"
                    stream_cb = make_stream_callback(verbose_stream, f"rewrite gen={gen} q={i+1}", stream_file)
                    try:
                        rewritten = ollama.generate(
                            gen,
                            prompt=ex.question,
                            system=REWRITE_SYSTEM,
                            options={"temperature": 0.0},
                            keep_alive=keep_alive,
                            stream=True,
                            on_token=stream_cb,
                            retries=1,
                            debug_label=f"rewrite gen={gen} q={i+1}",
                            debug_file=stream_file if verbose_stream else None,
                            heartbeat_s=5.0,
                            first_token_timeout_s=first_token_timeout_s,
                            stream_timeout_s=stream_timeout_s,
                        ).strip()
                        if verbose_stream:
                            print()
                    except OllamaError as e:
                        if verbose_stream:
                            print()
                        print(f"[warn] rewrite failed for gen={gen} q={i+1}: {e}")
                        append_failure_record(
                            results_path,
                            kb_mode=kb_mode,
                            kb_tag=kb_tag,
                            kb_random_paragraphs=kb_random_paragraphs,
                            subset_mode=subset_mode,
                            seed=seed,
                            pipeline=pipeline,
                            generator=gen,
                            embedder=emb,
                            top_k=top_k,
                            context_length=ctx_len,
                            ex=ex,
                            query_used=query_used,
                            retrieved_ids=[],
                            retrieved_scores=[],
                            answer="",
                            rewrite_time_s=time.time() - t0,
                            retrieval_time_s=0.0,
                            gen_time_s=0.0,
                            error_stage="rewrite",
                            error_message=str(e),
                        )
                        resume[key_hash] = i + 1
                        save_resume_state(resume_path, resume)
                        save_runtime_metadata("running", current_run_key=key_hash, current_question_index=i + 1, last_error_stage="rewrite")
                        continue
                    rewrite_time_s = time.time() - t0
                    if rewritten:
                        query_used = rewritten

                t0 = time.time()
                try:
                    retrieved_ids, retrieved_scores = retrieve(ollama, emb, index, ids, query_used, top_k)
                except OllamaError as e:
                    print(f"[warn] retrieval failed for gen={gen} emb={emb} q={i+1}: {e}")
                    append_failure_record(
                        results_path,
                        kb_mode=kb_mode,
                        kb_tag=kb_tag,
                        kb_random_paragraphs=kb_random_paragraphs,
                        subset_mode=subset_mode,
                        seed=seed,
                        pipeline=pipeline,
                        generator=gen,
                        embedder=emb,
                        top_k=top_k,
                        context_length=ctx_len,
                        ex=ex,
                        query_used=query_used,
                        retrieved_ids=[],
                        retrieved_scores=[],
                        answer="",
                        rewrite_time_s=rewrite_time_s,
                        retrieval_time_s=time.time() - t0,
                        gen_time_s=0.0,
                        error_stage="retrieve",
                        error_message=str(e),
                    )
                    resume[key_hash] = i + 1
                    save_resume_state(resume_path, resume)
                    save_runtime_metadata("running", current_run_key=key_hash, current_question_index=i + 1, last_error_stage="retrieve")
                    continue
                retrieval_time_s = time.time() - t0
                passages = fetch_passages(sqlite_path, retrieved_ids)

            answer = ""
            gen_time_s = 0.0
            if pipeline in {"P1", "P2", "P3"}:
                if pipeline == "P1":
                    prompt_used = f"Question: {ex.question}\nAnswer:"
                    system_used = BASELINE_SYSTEM
                else:
                    prompt_used = format_rag_prompt(ex.question, passages)
                    system_used = ANSWER_SYSTEM
                t0 = time.time()
                stream_label = f"answer gen={gen} pipeline={pipeline} q={i+1}"
                stream_file = stream_dir / f"answer_{sanitize_for_filename(gen)}_{pipeline}_{sanitize_for_filename(emb)}_q{i+1}.txt"
                stream_cb = make_stream_callback(verbose_stream, stream_label, stream_file)
                try:
                    answer = ollama.generate(
                        gen,
                        prompt=prompt_used,
                        system=system_used,
                        options={"temperature": 0.0, "top_p": 1.0, "num_ctx": int(ctx_len)},
                        keep_alive=keep_alive,
                        stream=True,
                        on_token=stream_cb,
                        retries=1,
                        debug_label=stream_label,
                        debug_file=stream_file if verbose_stream else None,
                        heartbeat_s=5.0,
                        first_token_timeout_s=first_token_timeout_s,
                        stream_timeout_s=stream_timeout_s,
                    ).strip()
                    if verbose_stream:
                        print()
                except OllamaError as e:
                    if verbose_stream:
                        print()
                    print(f"[warn] generation failed for gen={gen} q={i+1}: {e}")
                    append_failure_record(
                        results_path,
                        kb_mode=kb_mode,
                        kb_tag=kb_tag,
                        kb_random_paragraphs=kb_random_paragraphs,
                        subset_mode=subset_mode,
                        seed=seed,
                        pipeline=pipeline,
                        generator=gen,
                        embedder=emb,
                        top_k=top_k,
                        context_length=ctx_len,
                        ex=ex,
                        query_used=query_used,
                        retrieved_ids=retrieved_ids,
                        retrieved_scores=retrieved_scores,
                        answer="",
                        rewrite_time_s=rewrite_time_s,
                        retrieval_time_s=retrieval_time_s,
                        gen_time_s=time.time() - t0,
                        error_stage="generate",
                        error_message=str(e),
                    )
                    resume[key_hash] = i + 1
                    save_resume_state(resume_path, resume)
                    save_runtime_metadata("running", current_run_key=key_hash, current_question_index=i + 1, last_error_stage="generate")
                    continue
                gen_time_s = time.time() - t0

            em = compute_exact_match(answer, ex.gold_answers)
            f1 = compute_f1(answer, ex.gold_answers)
            rec = recall_at_k(retrieved_ids, ex.gold_provenance, k=top_k) if retrieved_ids else 0.0
            rprec = r_precision(retrieved_ids, ex.gold_provenance) if retrieved_ids else 0.0
            gold_set = set(ex.gold_provenance)
            retrieved_set = set(retrieved_ids)
            prov_hit = 1.0 if gold_set and gold_set.intersection(retrieved_set) else 0.0
            prov_all = 1.0 if gold_set and gold_set.issubset(retrieved_set) else 0.0

            record = {
                "timestamp": now_iso(),
                "status": "ok",
                "error_stage": None,
                "error_message": None,
                "kb_mode": kb_mode,
                "kb_tag": kb_tag,
                "kb_random_paragraphs": int(kb_random_paragraphs) if kb_mode != "full" else 0,
                "subset_mode": subset_mode,
                "subset_seed": int(seed),
                "pipeline": pipeline,
                "generator": gen,
                "embedder": emb,
                "top_k": int(top_k),
                "context_length": int(ctx_len),
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
                    "em": em,
                    "f1": f1,
                    "recall_at_k": rec,
                    "r_precision": rprec,
                    "kilt_em_hit": float(em) * prov_hit,
                    "kilt_f1_hit": float(f1) * prov_hit,
                    "kilt_em_all": float(em) * prov_all,
                    "kilt_f1_all": float(f1) * prov_all,
                },
            }
            append_jsonl(results_path, record)
            resume[key_hash] = i + 1
            save_resume_state(resume_path, resume)
            save_runtime_metadata("running", current_run_key=key_hash, current_question_index=i + 1)

        save_resume_state(resume_path, resume)
        save_runtime_metadata("configuration_complete", current_run_key=key_hash, current_question_index=len(examples))
        return False

    def generator_has_pending_work(gen: str) -> bool:
        if avoid_redundant_p0_p1:
            for ctx_len in context_lengths:
                key = RunKey(pipeline="P1", generator=gen, embedder="__none__", top_k=0, context_length=int(ctx_len))
                if resume.get(key.to_str(), 0) < len(examples):
                    return True
            for emb in embedders:
                for top_k in top_ks:
                    for ctx_len in context_lengths:
                        for pipeline in ["P2", "P3"]:
                            key = RunKey(pipeline=pipeline, generator=gen, embedder=emb, top_k=int(top_k), context_length=int(ctx_len))
                            if resume.get(key.to_str(), 0) < len(examples):
                                return True
            return False
        for emb in embedders:
            for top_k in top_ks:
                for ctx_len in context_lengths:
                    for pipeline in ["P0", "P1", "P2", "P3"]:
                        key = RunKey(pipeline=pipeline, generator=gen, embedder=emb, top_k=int(top_k), context_length=int(ctx_len))
                        if resume.get(key.to_str(), 0) < len(examples):
                            return True
        return False

    print("[run] Starting generator-centric evaluation loop")
    if avoid_redundant_p0_p1:
        print("[run] avoid_redundant_p0_p1=True | P0 will run once per (embedder, top_k, context_length); P1 once per (generator, context_length)")
        shared_p0_gen = "__shared_p0__"
        for emb in embedders:
            index, ids = indices[emb]
            for top_k in top_ks:
                for ctx_len in context_lengths:
                    if execute_config(pipeline="P0", gen=shared_p0_gen, emb=emb, top_k=int(top_k), ctx_len=int(ctx_len), keep_alive="5m", index=index, ids=ids):
                        save_runtime_metadata("stopped_early")
                        cleanup_embedders()
                        return
    else:
        print("[run] avoid_redundant_p0_p1=False | running legacy fully repeated loop")

    for gen in generators:
        print(f"[gen] Starting generator: {gen}")
        if not generator_has_pending_work(gen):
            print(f"[gen] All generator-dependent experiments already completed for: {gen}; skipping model check/pull")
            continue
        gen_pulled = False
        if not ollama.model_exists(gen):
            print(f"[ollama] pulling generator: {gen}")
            ollama.pull(gen)
            gen_pulled = True
        else:
            print(f"[ollama] generator already available locally: {gen}")
        keep_alive = "30m"
        try:
            if avoid_redundant_p0_p1:
                for ctx_len in context_lengths:
                    if execute_config(pipeline="P1", gen=gen, emb="__none__", top_k=0, ctx_len=int(ctx_len), keep_alive=keep_alive):
                        save_runtime_metadata("stopped_early")
                        cleanup_embedders()
                        return
                for emb in embedders:
                    index, ids = indices[emb]
                    for top_k in top_ks:
                        for ctx_len in context_lengths:
                            for pipeline in ["P2", "P3"]:
                                if execute_config(pipeline=pipeline, gen=gen, emb=emb, top_k=int(top_k), ctx_len=int(ctx_len), keep_alive=keep_alive, index=index, ids=ids):
                                    save_runtime_metadata("stopped_early")
                                    cleanup_embedders()
                                    return
            else:
                for emb in embedders:
                    index, ids = indices[emb]
                    for top_k in top_ks:
                        for ctx_len in context_lengths:
                            for pipeline in ["P0", "P1", "P2", "P3"]:
                                if execute_config(pipeline=pipeline, gen=gen, emb=emb, top_k=int(top_k), ctx_len=int(ctx_len), keep_alive=keep_alive, index=index, ids=ids):
                                    save_runtime_metadata("stopped_early")
                                    cleanup_embedders()
                                    return
        finally:
            try:
                ollama.generate(gen, prompt="", system="", options={"temperature": 0.0}, keep_alive="0")
            except Exception:
                pass
            if gen_pulled:
                print(f"[ollama] deleting generator (pulled by script): {gen}")
                try:
                    ollama.delete(gen)
                except Exception as e:
                    print(f"[warn] could not delete generator {gen}: {e}")
        print(f"[gen] Completed generator: {gen}")

    cleanup_embedders()
    save_runtime_metadata("completed")
    print("[done] Experiment completed for all configured generators.")
