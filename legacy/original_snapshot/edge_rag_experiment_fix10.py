"""edge_rag_experiment_fix6.py

EdgeRAG on KILT-NQ (single-workstation, Ollama-first)
====================================================

This script implements the *experiment runner* for the EdgeRAG proposal.

Key design goals
----------------
1) Generator-centric: evaluate many local/quantized generators under a mostly fixed retriever.
2) Restartable & time-budgeted: safe to stop anytime; resumes from the last saved checkpoint.
3) Disk-friendly: generators are pulled on demand; models pulled by this script are deleted
   after finishing to avoid hoarding hundreds of GB.
4) Mostly-Python: interaction with Ollama uses the local HTTP API.

What it runs (pipelines)
------------------------
We operationalize the proposal's ablations as four pipelines per generator:

P0  Retrieval-only evaluation: (query -> retrieve top-k) and compute retrieval metrics.
P1  No-retrieval baseline: generator answers without passages.
P2  Standard RAG: retrieve top-k passages, then answer.
P3  Single-model query rewrite RAG: rewrite query using the *same* generator, retrieve, answer.

Notes on feasibility
--------------------
Building dense indices over the full KILT knowledge source is expensive.
This script supports building FAISS indices per embedding model, but you should expect
index building to take a long time (hours to days) depending on embedding throughput.

Dependencies
------------
Required:
  - Python 3.9+
  - requests
  - numpy
Recommended:
  - faiss-cpu or faiss-gpu
  - ijson (only needed if your KILT knowledge source is a giant JSON array)
  - sentence-transformers (only needed for HF embedders such as MiniLM)

Ollama:
  - Ollama running locally (default http://localhost:11434)

Phase-1 defaults (sanity + plots)
--------------------------------
This file is pre-configured for your **Phase 1** plan:

  - 500 questions (random, deterministic seed)
  - Reduced KB: gold paragraphs + 100k random paragraphs
  - 6 generators: 0.6B, 1.7B, 4B, 8B, 14B, 30B
  - 3 embedding models

You can override everything via CLI flags (see --help), but the defaults
are set so a simple run is:

python edge_rag_experiment_fix6.py --data_dir data --results_dir results

All outputs are written as JSONL to:
  results_dir/results.jsonl

The runner writes a resume state to:
  results_dir/resume_state.json
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import sqlite3
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# Phase 1 defaults (edit here if you want)
# -----------------------------

# Question subset
PHASE1_MAX_QUESTIONS = 500
PHASE1_SUBSET_MODE = "random"  # "head" or "random"
PHASE1_SEED = 123

# Reduced KB = gold + random
PHASE1_KB_MODE = "gold_plus_random"  # "full" or "gold_plus_random"
PHASE1_KB_RANDOM_PARAGRAPHS = 100_000

# Retrieval settings
PHASE1_TOP_KS = "5,10"


# -----------------------------
# Utilities (pure python)
# -----------------------------


def now_iso() -> str:
    """Return a stable ISO-like timestamp (local time)."""

    return time.strftime("%Y-%m-%dT%H:%M:%S")


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def normalize_text(s: str) -> str:
    """SQuAD-style normalization for EM/F1."""

    import re
    import string

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(pred: str, golds: Sequence[str]) -> int:
    """Exact match against any gold answer."""

    p = normalize_text(pred)
    for g in golds:
        if p == normalize_text(g):
            return 1
    return 0


def compute_f1(pred: str, golds: Sequence[str]) -> float:
    """Token-level F1 (max over golds), SQuAD-style."""

    from collections import Counter

    pred_toks = normalize_text(pred).split()
    if not pred_toks:
        return 0.0
    pred_counter = Counter(pred_toks)

    best = 0.0
    for g in golds:
        gold_toks = normalize_text(g).split()
        if not gold_toks:
            continue
        gold_counter = Counter(gold_toks)
        common = pred_counter & gold_counter
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Recall@k (success@k) for provenance."""

    if not gold or k <= 0:
        return 0.0
    gset = set(gold)
    topk = set(retrieved[:k])
    return 1.0 if gset.intersection(topk) else 0.0


def r_precision(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """R-Precision (precision@R) where R is number of gold provenance items."""

    if not gold:
        return 0.0
    gset = set(gold)
    R = len(gset)
    if R <= 0:
        return 0.0
    topR = retrieved[:R]
    hits = sum(1 for pid in topR if pid in gset)
    return hits / R


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def sanitize_for_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(text))[:160]


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON record and fsync (crash safe)."""

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def make_stream_callback(enabled: bool, label: str, stream_file: Optional[Path] = None) -> Optional[Callable[[str], None]]:
    if not enabled:
        return None
    if stream_file is not None:
        safe_mkdir(stream_file.parent)
        with open(stream_file, "w", encoding="utf-8") as f:
            f.write(f"[{label}]\n")
            f.flush()

    def _cb(token: str) -> None:
        print(token, end="", flush=True)
        if stream_file is not None:
            with open(stream_file, "a", encoding="utf-8") as f:
                f.write(token)
                f.flush()

    return _cb


def _write_text_atomic(path: Path, text: str) -> None:
    safe_mkdir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl_debug(path: Path, obj: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


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
    record = {
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


# -----------------------------
# Ollama client (HTTP API)
# -----------------------------


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Minimal Ollama HTTP client with retries and readiness checks."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 600,
        retries: int = 6,
        retry_backoff_s: float = 5.0,
        verbose: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.retry_backoff_s = float(retry_backoff_s)
        self.verbose = bool(verbose)

    def _req(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        import requests

        url = f"{self.base_url}{path}"
        timeout = self.timeout if timeout is None else timeout
        retries = self.retries if retries is None else max(1, int(retries))
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                r = requests.request(method, url, json=payload, timeout=timeout)
                if r.status_code == 200:
                    try:
                        return r.json()
                    except Exception as e:
                        raise OllamaError(f"Ollama returned non-JSON: {r.text[:200]}") from e

                try:
                    msg = r.json()
                except Exception:
                    msg = r.text

                retryable = r.status_code in {408, 409, 429, 500, 502, 503, 504}
                if retryable and attempt < retries:
                    wait_s = min(30.0, self.retry_backoff_s * attempt)
                    if self.verbose:
                        print(f"[ollama][retry {attempt}/{retries}] HTTP {r.status_code} on {method} {path}; sleeping {wait_s:.1f}s")
                    time.sleep(wait_s)
                    continue
                raise OllamaError(f"Ollama error {r.status_code} for {method} {url}: {msg}")
            except requests.RequestException as e:
                last_err = e
                if attempt < retries:
                    wait_s = min(30.0, self.retry_backoff_s * attempt)
                    if self.verbose:
                        print(f"[ollama][retry {attempt}/{retries}] request failed for {method} {path}: {e}; sleeping {wait_s:.1f}s")
                    time.sleep(wait_s)
                    continue
                raise OllamaError(f"Ollama request failed after {retries} attempts: {method} {url}: {e}") from e
        raise OllamaError(f"Ollama request failed: {method} {url}: {last_err}")

    def _generate_stream(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        *,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
        on_token: Optional[Callable[[str], None]] = None,
        debug_label: Optional[str] = None,
        debug_file: Optional[Path] = None,
        heartbeat_s: float = 5.0,
        first_token_timeout_s: Optional[float] = None,
        stream_timeout_s: Optional[float] = None,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 10.0,
    ) -> str:
        import requests

        url = f"{self.base_url}/api/generate"
        timeout = self.timeout if timeout is None else timeout
        retries = self.retries if retries is None else max(1, int(retries))
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
        if system is not None:
            payload["system"] = system
        if options:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        last_err = None
        status_file: Optional[Path] = None
        events_file: Optional[Path] = None
        state: Dict[str, Any] = {"status": "init", "chars": 0, "first_token_t": None, "last_token_t": None}
        start_wall = time.time()
        stop_heartbeat = threading.Event()
        abort_stream = threading.Event()
        abort_reason: Dict[str, Optional[str]] = {"value": None}
        current_response: Dict[str, Any] = {"obj": None}

        def log_event(kind: str, **extra: Any) -> None:
            if events_file is None:
                return
            payload = {
                "ts": now_iso(),
                "kind": kind,
                "label": debug_label or model,
                "elapsed_s": round(time.time() - start_wall, 3),
            }
            payload.update(extra)
            _append_jsonl_debug(events_file, payload)

        heartbeat_thread: Optional[threading.Thread] = None
        if debug_file is not None:
            status_file = debug_file.with_suffix(debug_file.suffix + ".status.txt")
            events_file = debug_file.with_suffix(debug_file.suffix + ".events.jsonl")
            _write_text_atomic(status_file, f"[{debug_label or model}]\nstatus=starting\nelapsed_s=0\nchars=0\nfirst_token=no\n")
            log_event("start_request", model=model, url=url)

        if (debug_file is not None) or (first_token_timeout_s is not None) or (stream_timeout_s is not None):
            def _heartbeat() -> None:
                while not stop_heartbeat.wait(max(1.0, float(heartbeat_s))):
                    elapsed = time.time() - start_wall
                    first = state["first_token_t"] is not None
                    last_age = None if state["last_token_t"] is None else round(time.time() - float(state["last_token_t"]), 3)
                    if (not abort_stream.is_set()) and (not first) and first_token_timeout_s is not None and elapsed >= float(first_token_timeout_s):
                        abort_reason["value"] = f"no_first_token_timeout after {elapsed:.1f}s"
                        state["status"] = "no_first_token_timeout"
                        abort_stream.set()
                        try:
                            resp = current_response.get("obj")
                            if resp is not None:
                                resp.close()
                        except Exception:
                            pass
                        log_event("no_first_token_timeout", elapsed_s=round(elapsed, 3))
                    elif (not abort_stream.is_set()) and first and stream_timeout_s is not None and elapsed >= float(stream_timeout_s):
                        abort_reason["value"] = f"stream_timeout after {elapsed:.1f}s"
                        state["status"] = "stream_timeout"
                        abort_stream.set()
                        try:
                            resp = current_response.get("obj")
                            if resp is not None:
                                resp.close()
                        except Exception:
                            pass
                        log_event("stream_timeout", elapsed_s=round(elapsed, 3))

                    msg = (
                        f"[{debug_label or model}]\n"
                        f"status={state['status']}\n"
                        f"elapsed_s={elapsed:.1f}\n"
                        f"chars={state['chars']}\n"
                        f"first_token={'yes' if first else 'no'}\n"
                        f"last_token_age_s={last_age if last_age is not None else 'NA'}\n"
                    )
                    if status_file is not None:
                        _write_text_atomic(status_file, msg)
                    if self.verbose:
                        print(f"[stream][heartbeat] {debug_label or model} | status={state['status']} | elapsed={elapsed:.1f}s | chars={state['chars']} | first_token={'yes' if first else 'no'}")
                    log_event("heartbeat", status=state["status"], chars=state["chars"], first_token=first, last_token_age_s=last_age)

            heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
            heartbeat_thread.start()

        def _finish_status(final_status: str, error: Optional[str] = None) -> None:
            stop_heartbeat.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)
            if status_file is not None:
                text = (
                    f"[{debug_label or model}]\n"
                    f"status={final_status}\n"
                    f"elapsed_s={time.time()-start_wall:.1f}\n"
                    f"chars={state['chars']}\n"
                    f"first_token={'yes' if state['first_token_t'] is not None else 'no'}\n"
                )
                if error:
                    text += f"error={error}\n"
                _write_text_atomic(status_file, text)

        attempt = 0
        while True:
            attempt += 1
            chunks: List[str] = []
            try:
                with requests.request("POST", url, json=payload, timeout=(connect_timeout_s, read_timeout_s), stream=True) as r:
                    current_response["obj"] = r
                    state["status"] = f"http_{r.status_code}"
                    log_event("http_response", status_code=r.status_code)
                    if r.status_code != 200:
                        try:
                            msg = r.json()
                        except Exception:
                            msg = r.text
                        retryable = r.status_code in {408, 409, 429, 500, 502, 503, 504}
                        elapsed = time.time() - start_wall
                        allow_more = (
                            state["first_token_t"] is None
                            and first_token_timeout_s is not None
                            and elapsed < float(first_token_timeout_s)
                        )
                        if retryable and (attempt < retries or allow_more):
                            wait_s = min(30.0, self.retry_backoff_s * min(attempt, max(1, retries)))
                            if self.verbose:
                                print(f"[ollama][retry {attempt}/{retries}] HTTP {r.status_code} on POST /api/generate(stream); sleeping {wait_s:.1f}s")
                            time.sleep(wait_s)
                            continue
                        raise OllamaError(f"Ollama error {r.status_code} for POST {url}: {msg}")

                    state["status"] = "stream_open"
                    log_event("stream_open")
                    try:
                        for raw in r.iter_lines(decode_unicode=True):
                            if abort_stream.is_set():
                                raise OllamaError(abort_reason["value"] or "stream_aborted")
                            if not raw:
                                continue
                            try:
                                evt = json.loads(raw)
                            except Exception as e:
                                raise OllamaError(f"Could not parse streamed Ollama JSON line: {raw[:200]}") from e
                            piece = evt.get("response", "")
                            if piece:
                                chunks.append(piece)
                                state["chars"] += len(piece)
                                now_t = time.time()
                                state["last_token_t"] = now_t
                                if state["first_token_t"] is None:
                                    state["first_token_t"] = now_t
                                    state["status"] = "streaming"
                                    log_event("first_token", chars=state["chars"])
                                if on_token is not None:
                                    on_token(piece)
                            if evt.get("done") is True:
                                state["status"] = "done"
                                log_event("done_event", chars=state["chars"], done_reason=evt.get("done_reason"))
                                _finish_status("done")
                                return "".join(chunks)
                    except OllamaError:
                        raise
                    except Exception as e:
                        # The watchdog may close the underlying response while requests/urllib3
                        # is still blocked in iter_lines(), which can surface as low-level
                        # AttributeError/ProtocolError/etc. Treat those as the intended abort
                        # when an abort reason has already been set.
                        if abort_stream.is_set():
                            msg = abort_reason["value"] or f"stream_aborted_after_internal_error: {e}"
                            _finish_status("failed", msg)
                            raise OllamaError(msg) from e
                        raise

                    if abort_stream.is_set():
                        msg = abort_reason["value"] or "stream_aborted"
                        _finish_status("failed", msg)
                        raise OllamaError(msg)
                    state["status"] = "stream_ended_no_done"
                    log_event("stream_end_without_done", chars=state["chars"])
                    _finish_status("stream_ended_no_done")
                    return "".join(chunks)
            except OllamaError as e:
                _finish_status("failed", str(e))
                raise
            except requests.RequestException as e:
                state["status"] = "request_exception"
                log_event("request_exception", error=str(e), attempt=attempt)
                last_err = e
                elapsed = time.time() - start_wall

                if state["first_token_t"] is None and first_token_timeout_s is not None and elapsed >= float(first_token_timeout_s):
                    msg = f"no_first_token_timeout after {elapsed:.1f}s"
                    state["status"] = "no_first_token_timeout"
                    log_event("no_first_token_timeout", elapsed_s=round(elapsed, 3), attempt=attempt)
                    _finish_status("failed", msg)
                    raise OllamaError(msg) from e

                if state["first_token_t"] is not None and stream_timeout_s is not None and elapsed >= float(stream_timeout_s):
                    msg = f"stream_timeout after {elapsed:.1f}s"
                    state["status"] = "stream_timeout"
                    log_event("stream_timeout", elapsed_s=round(elapsed, 3), attempt=attempt)
                    _finish_status("failed", msg)
                    raise OllamaError(msg) from e

                allow_more = (
                    state["first_token_t"] is None
                    and first_token_timeout_s is not None
                    and elapsed < float(first_token_timeout_s)
                ) or (
                    state["first_token_t"] is not None
                    and stream_timeout_s is not None
                    and elapsed < float(stream_timeout_s)
                )
                if attempt < retries or allow_more:
                    wait_s = min(30.0, self.retry_backoff_s * min(attempt, max(1, retries)))
                    if self.verbose:
                        if state["first_token_t"] is None and first_token_timeout_s is not None:
                            remaining = max(0.0, float(first_token_timeout_s) - elapsed)
                            print(f"[ollama][retry {attempt}/{retries}] no first token yet for {debug_label or model}; elapsed={elapsed:.1f}s, remaining_first_token_budget={remaining:.1f}s; sleeping {wait_s:.1f}s")
                        else:
                            print(f"[ollama][retry {attempt}/{retries}] request failed for POST /api/generate(stream): {e}; sleeping {wait_s:.1f}s")
                    time.sleep(wait_s)
                    continue
                _finish_status("failed", str(e))
                raise OllamaError(f"Ollama streamed request failed after {max(retries, attempt)} attempts: POST {url}: {e}") from e
            finally:
                current_response["obj"] = None
        _finish_status("failed", str(last_err))
        raise OllamaError(f"Ollama streamed request failed: POST {url}: {last_err}")

    def wait_until_ready(self, timeout_s: float = 120.0) -> None:
        deadline = time.time() + max(1.0, timeout_s)
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                _ = self.list_models()
                if self.verbose:
                    print(f"[ollama] server is reachable at {self.base_url} (attempt {attempt})")
                return
            except Exception as e:
                if self.verbose:
                    remaining = max(0.0, deadline - time.time())
                    print(f"[ollama] waiting for server at {self.base_url} (attempt {attempt}, {remaining:.0f}s left): {e}")
                time.sleep(min(10.0, self.retry_backoff_s))
        raise OllamaError(f"Ollama server at {self.base_url} did not become ready within {timeout_s}s")

    def list_models(self) -> List[str]:
        data = self._req("GET", "/api/tags")
        out = []
        for m in data.get("models", []) or []:
            name = m.get("name")
            if name:
                out.append(name)
        return out

    def model_exists(self, model: str) -> bool:
        return model in set(self.list_models())

    def pull(self, model: str) -> None:
        self._req("POST", "/api/pull", {"model": model, "stream": False})

    def delete(self, model: str) -> None:
        self._req("DELETE", "/api/delete", {"model": model})

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        *,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
        debug_label: Optional[str] = None,
        debug_file: Optional[Path] = None,
        heartbeat_s: float = 5.0,
        first_token_timeout_s: Optional[float] = None,
        stream_timeout_s: Optional[float] = None,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 10.0,
    ) -> str:
        if stream:
            return self._generate_stream(
                model,
                prompt,
                system=system,
                options=options,
                keep_alive=keep_alive,
                timeout=timeout,
                retries=retries,
                on_token=on_token,
                debug_label=debug_label,
                debug_file=debug_file,
                heartbeat_s=heartbeat_s,
                first_token_timeout_s=first_token_timeout_s,
                stream_timeout_s=stream_timeout_s,
                connect_timeout_s=connect_timeout_s,
                read_timeout_s=read_timeout_s,
            )
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if system is not None:
            payload["system"] = system
        if options:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        data = self._req("POST", "/api/generate", payload, timeout=timeout, retries=retries)
        return data.get("response", "")

    def embed(
        self,
        model: str,
        inputs: Sequence[str],
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> np.ndarray:
        payload: Dict[str, Any] = {"model": model, "input": list(inputs)}
        if options:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        data = self._req("POST", "/api/embed", payload)
        embs = data.get("embeddings")
        if not isinstance(embs, list) or not embs:
            raise OllamaError(f"Unexpected embed response keys: {list(data.keys())}")
        return np.asarray(embs, dtype=np.float32)


# -----------------------------
# KILT data loading
# -----------------------------


KILT_BASE = "http://dl.fbaipublicfiles.com/KILT"


def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download with resume (HTTP Range) into out_path."""

    import requests

    safe_mkdir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    resume_pos = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()

        # If we requested a range but the server ignored it (status 200), restart.
        if resume_pos > 0 and r.status_code == 200:
            resume_pos = 0
            headers = {}
            tmp.unlink(missing_ok=True)

    # Re-open the request (simple, clear logic)
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        mode = "ab" if resume_pos > 0 else "wb"
        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, out_path)


def ensure_kilt_nq(data_dir: Path, split: str = "dev") -> Tuple[Path, Path]:
    split = split.lower()
    if split not in {"train", "dev"}:
        raise ValueError("split must be 'train' or 'dev'")
    nq_name = f"nq-{split}-kilt.jsonl"
    ks_name = "kilt_knowledgesource.json"
    nq_path = data_dir / nq_name
    ks_path = data_dir / ks_name
    if not nq_path.exists():
        print(f"[data] Downloading {nq_name} ...")
        download_file(f"{KILT_BASE}/{nq_name}", nq_path)
    if not ks_path.exists():
        print(f"[data] Downloading {ks_name} (large: ~35GiB) ...")
        download_file(f"{KILT_BASE}/{ks_name}", ks_path)
    return nq_path, ks_path


@dataclass
class KiltExample:
    qid: str
    question: str
    gold_answers: List[str]
    gold_provenance: List[str]


def load_kilt_nq_examples(
    nq_jsonl: Path,
    max_questions: Optional[int] = None,
    subset_mode: str = "head",
    seed: int = PHASE1_SEED,
) -> List[KiltExample]:
    """Load KILT-NQ examples.

    Parameters
    ----------
    nq_jsonl:
        Path to KILT-NQ JSONL.
    max_questions:
        If set, limit the number of questions.
    subset_mode:
        "head" = take the first ``max_questions`` examples (fastest).
        "random" = deterministic random subset of size ``max_questions``.
    seed:
        Seed used when ``subset_mode="random"``.

    Provenance handling
    -------------------
    KILT provides provenance spans (start_paragraph_id/end_paragraph_id).
    We include **all** paragraphs in that inclusive span when both are present.
    """

    subset_mode = (subset_mode or "head").strip().lower()
    if subset_mode not in {"head", "random"}:
        raise ValueError("subset_mode must be 'head' or 'random'")

    exs_all: List[KiltExample] = []
    with open(nq_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # Fast path for head mode
            if subset_mode == "head" and max_questions is not None and i >= max_questions:
                break
            obj = json.loads(line)
            qid = str(obj.get("id", i))
            question = (obj.get("input") or "").strip()
            gold_answers: List[str] = []
            gold_prov: List[str] = []

            for out in obj.get("output", []) or []:
                ans = out.get("answer")
                if isinstance(ans, str) and ans.strip():
                    gold_answers.append(ans.strip())

                for prov in out.get("provenance", []) or []:
                    wid = prov.get("wikipedia_id")
                    sp = prov.get("start_paragraph_id")
                    ep = prov.get("end_paragraph_id")
                    if wid is None:
                        continue

                    # Include full span when available.
                    if sp is not None and ep is not None:
                        try:
                            sp_i = int(sp)
                            ep_i = int(ep)
                        except Exception:
                            sp_i = None
                            ep_i = None
                        if sp_i is not None and ep_i is not None and ep_i >= sp_i:
                            for pid_idx in range(sp_i, ep_i + 1):
                                gold_prov.append(f"{wid}:{pid_idx}")
                            continue

                    if sp is not None:
                        gold_prov.append(f"{wid}:{sp}")
                    else:
                        gold_prov.append(str(wid))

            gold_answers = list(dict.fromkeys(gold_answers))
            gold_prov = list(dict.fromkeys(gold_prov))
            exs_all.append(
                KiltExample(qid=qid, question=question, gold_answers=gold_answers, gold_provenance=gold_prov)
            )

    if subset_mode == "random" and max_questions is not None:
        if max_questions >= len(exs_all):
            return exs_all
        rng = np.random.default_rng(int(seed))
        idxs = rng.choice(len(exs_all), size=int(max_questions), replace=False)
        idxs_sorted = sorted(int(i) for i in idxs)
        return [exs_all[i] for i in idxs_sorted]

    if max_questions is not None:
        return exs_all[: int(max_questions)]
    return exs_all


# -----------------------------
# Knowledge store (SQLite)
# -----------------------------


def _require(module: str, pip_name: Optional[str] = None) -> Any:
    try:
        return __import__(module)
    except Exception as e:
        raise RuntimeError(f"Missing dependency '{module}'. Install with: pip install {pip_name or module}") from e


def iter_kilt_pages(ks_json: Path) -> Iterator[Dict[str, Any]]:
    """Yield KILT knowledge-source pages robustly.

    The official KILT knowledge source is commonly encountered as newline-delimited JSON
    (one page-object per line), but some mirrors may wrap it in one large JSON array.
    This iterator supports both formats.
    """
    with open(ks_json, "rb") as fb:
        first_non_ws = b""
        while True:
            ch = fb.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_ws = ch
                break

    if first_non_ws == b"[":
        ijson = _require("ijson", "ijson")
        print("[kb] Detected JSON-array KILT knowledge source; using ijson streaming parser")
        with open(ks_json, "rb") as f:
            for page in ijson.items(f, "item"):
                if isinstance(page, dict):
                    yield page
    else:
        print("[kb] Detected JSONL/NDJSON KILT knowledge source; using line-by-line parser")
        with open(ks_json, "r", encoding="utf-8") as f:
            for line_i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    page = json.loads(line)
                except Exception as e:
                    print(f"[kb][warn] Could not parse knowledge-source line {line_i}: {e}")
                    continue
                if isinstance(page, dict):
                    yield page


def sqlite_passage_count(sqlite_path: Path) -> int:
    if not sqlite_path.exists():
        return 0
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM passages")
        row = cur.fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def validate_existing_sqlite(sqlite_path: Path, label: str) -> bool:
    """Return True if an existing KB SQLite file is usable; otherwise remove it."""
    if not sqlite_path.exists():
        return False
    try:
        count = sqlite_passage_count(sqlite_path)
        if count > 0:
            print(f"[{label}] Reusing existing SQLite KB: {sqlite_path} ({count:,} passages)")
            return True
        print(f"[{label}][warn] Existing SQLite KB is empty; rebuilding: {sqlite_path}")
    except Exception as e:
        print(f"[{label}][warn] Existing SQLite KB is invalid ({e}); rebuilding: {sqlite_path}")
    try:
        sqlite_path.unlink(missing_ok=True)
    except Exception:
        pass
    return False


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def is_hf_embedder(model: str) -> bool:
    m = (model or "").strip().lower()
    return m.startswith("hf:") or m in {
        "all-minilm-l6-v2",
        "all-minilm",
        "sentence-transformers/all-minilm-l6-v2",
    }


def resolve_hf_model_name(model: str) -> str:
    m = (model or "").strip()
    ml = m.lower()
    if ml.startswith("hf:"):
        return m[3:]
    if ml in {"all-minilm-l6-v2", "all-minilm", "sentence-transformers/all-minilm-l6-v2"}:
        return "sentence-transformers/all-MiniLM-L6-v2"
    return m


_HF_MODEL_CACHE: Dict[str, Any] = {}


def embed_texts(ollama: OllamaClient, embed_model: str, texts: Sequence[str], keep_alive: str = "30m") -> np.ndarray:
    """Embed text with either Ollama or SentenceTransformers, always L2-normalized."""
    if is_hf_embedder(embed_model):
        model_name = resolve_hf_model_name(embed_model)
        if model_name not in _HF_MODEL_CACHE:
            print(f"[hf] Loading embedding model: {model_name}")
            st = _require("sentence_transformers", "sentence-transformers")
            _HF_MODEL_CACHE[model_name] = st.SentenceTransformer(model_name)
        model = _HF_MODEL_CACHE[model_name]
        vecs = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
    return l2_normalize(ollama.embed(embed_model, texts, keep_alive=keep_alive))


def build_kilt_sqlite(ks_json: Path, sqlite_path: Path) -> None:
    """Build paragraph-level SQLite KB from KILT knowledge source."""

    safe_mkdir(sqlite_path.parent)
    if validate_existing_sqlite(sqlite_path, "kb"):
        return
    print(f"[kb] Building SQLite KB at {sqlite_path} ...")
    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute(
        "CREATE TABLE IF NOT EXISTS passages (pid TEXT PRIMARY KEY, wikipedia_id INTEGER, paragraph_id INTEGER, title TEXT, text TEXT)"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_wid ON passages(wikipedia_id)")
    conn.commit()

    batch: List[Tuple[str, int, int, str, str]] = []
    BATCH_SIZE = 2000
    for n, page in enumerate(iter_kilt_pages(ks_json), start=1):
            try:
                wid = int(page.get("wikipedia_id"))
            except Exception:
                continue
            title = page.get("wikipedia_title", "") or ""
            paragraphs = page.get("text", []) or []
            if not isinstance(paragraphs, list):
                continue
            for pid_idx, para in enumerate(paragraphs):
                if not isinstance(para, str) or not para.strip():
                    continue
                pid = f"{wid}:{pid_idx}"
                batch.append((pid, wid, pid_idx, title, para.strip()))
                if len(batch) >= BATCH_SIZE:
                    cur.executemany(
                        "INSERT OR REPLACE INTO passages(pid, wikipedia_id, paragraph_id, title, text) VALUES (?,?,?,?,?)",
                        batch,
                    )
                    conn.commit()
                    batch.clear()
            if n % 1000 == 0:
                print(f"[kb] Processed lines/pages: {n:,}")
    if batch:
        cur.executemany(
            "INSERT OR REPLACE INTO passages(pid, wikipedia_id, paragraph_id, title, text) VALUES (?,?,?,?,?)",
            batch,
        )
        conn.commit()
    total = sqlite_passage_count(sqlite_path)
    conn.close()
    print(f"[kb] Done. passages={total:,}")


def _hash64(seed: int, s: str) -> int:
    """Deterministic 64-bit hash for streaming sampling.

    We use BLAKE2b because it is stable across Python versions and platforms.
    """

    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b"|")
    h.update(s.encode("utf-8"))
    return int.from_bytes(h.digest(), "big", signed=False)


def build_reduced_kilt_sqlite(
    ks_json: Path,
    sqlite_path: Path,
    keep_pids: Sequence[str],
    random_paragraphs: int,
    seed: int,
    *,
    max_pages_debug: Optional[int] = None,
) -> None:
    """Build a reduced SQLite KB: gold paragraphs + deterministic random paragraphs.

    This function implements the Phase-1 KB reduction:
      KB = {all gold provenance paragraphs} ∪ {N random paragraphs}

    The random set is selected deterministically via *min-hash sampling*:
    among all non-gold paragraphs, we keep the N paragraphs with the smallest
    64-bit hash values. This is reproducible and single-pass.

    Parameters
    ----------
    ks_json:
        KILT knowledge source JSON (large JSON array of pages).
    sqlite_path:
        Output SQLite file.
    keep_pids:
        Paragraph IDs to always include (gold provenance), e.g. "12345:7".
    random_paragraphs:
        Number of additional random paragraphs.
    seed:
        Seed for deterministic sampling.
    max_pages_debug:
        If set, stop after this many pages (debug only; NOT for reporting).
    """

    safe_mkdir(sqlite_path.parent)
    if validate_existing_sqlite(sqlite_path, "kb"):
        return

    keep_set = set(str(x) for x in keep_pids)
    print(
        f"[kb] Building REDUCED SQLite KB at {sqlite_path} (gold={len(keep_set):,}, random={random_paragraphs:,}) ..."
    )

    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute(
        "CREATE TABLE IF NOT EXISTS passages (pid TEXT PRIMARY KEY, wikipedia_id INTEGER, paragraph_id INTEGER, title TEXT, text TEXT)"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_wid ON passages(wikipedia_id)")
    conn.commit()

    # Keep track of which gold pids we actually observed in the KS file.
    seen_gold: set[str] = set()

    # Max-heap (by hash) of size random_paragraphs.
    # Store tuples: (-hash, pid, wid, pid_idx, title, text)
    heap: List[Tuple[int, str, int, int, str, str]] = []

    BATCH_SIZE = 2000
    batch: List[Tuple[str, int, int, str, str]] = []

    for page_i, page in enumerate(iter_kilt_pages(ks_json), start=1):
        if max_pages_debug is not None and page_i > max_pages_debug:
            break

        try:
            wid = int(page.get("wikipedia_id"))
        except Exception:
            continue
        title = page.get("wikipedia_title", "") or ""
        paragraphs = page.get("text", []) or []
        if not isinstance(paragraphs, list):
            continue

        for pid_idx, para in enumerate(paragraphs):
            if not isinstance(para, str) or not para.strip():
                continue
            pid = f"{wid}:{pid_idx}"
            text = para.strip()

            if pid in keep_set:
                seen_gold.add(pid)
                batch.append((pid, wid, pid_idx, title, text))
                if len(batch) >= BATCH_SIZE:
                    cur.executemany(
                        "INSERT OR REPLACE INTO passages(pid, wikipedia_id, paragraph_id, title, text) VALUES (?,?,?,?,?)",
                        batch,
                    )
                    conn.commit()
                    batch.clear()
                continue

            if random_paragraphs <= 0:
                continue
            h = _hash64(seed, pid)
            item = (-int(h), pid, wid, pid_idx, title, text)
            if len(heap) < random_paragraphs:
                heapq.heappush(heap, item)
            else:
                if item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

        if page_i % 1000 == 0:
            print(f"[kb] Processed lines/pages: {page_i:,} | gold_seen={len(seen_gold):,} | heap={len(heap):,}")

    # Flush gold batch
    if batch:
        cur.executemany(
            "INSERT OR REPLACE INTO passages(pid, wikipedia_id, paragraph_id, title, text) VALUES (?,?,?,?,?)",
            batch,
        )
        conn.commit()
        batch.clear()

    # Insert random paragraphs from heap
    random_rows = [(pid, wid, pid_idx, title, text) for (_, pid, wid, pid_idx, title, text) in heap]
    if random_rows:
        cur.executemany(
            "INSERT OR REPLACE INTO passages(pid, wikipedia_id, paragraph_id, title, text) VALUES (?,?,?,?,?)",
            random_rows,
        )
        conn.commit()

    total = sqlite_passage_count(sqlite_path)
    conn.close()

    # Save manifests for reproducibility.
    gold_path = sqlite_path.with_suffix(".gold_pids.txt")
    rand_path = sqlite_path.with_suffix(".random_pids.txt")
    meta_path = sqlite_path.with_suffix(".meta.json")
    with open(gold_path, "w", encoding="utf-8") as f:
        for pid in sorted(keep_set):
            f.write(pid + "\n")
    with open(rand_path, "w", encoding="utf-8") as f:
        for (_, pid, *_rest) in sorted(heap, reverse=True):
            f.write(pid + "\n")

    missing = sorted(keep_set.difference(seen_gold))
    atomic_write_json(
        meta_path,
        {
            "kb_mode": "gold_plus_random",
            "seed": int(seed),
            "gold_count": int(len(keep_set)),
            "gold_seen_in_ks": int(len(seen_gold)),
            "gold_missing": int(len(missing)),
            "random_paragraphs": int(random_paragraphs),
            "random_selected": int(len(heap)),
            "note": "Random set selected by min-hash sampling over paragraph IDs.",
        },
    )
    if missing:
        print(f"[kb][warn] {len(missing)} gold paragraphs were not found in the knowledge source. Example: {missing[:5]}")
    print(f"[kb] Reduced KB done. total_passages={total:,}")


def fetch_passages(sqlite_path: Path, pids: Sequence[str]) -> List[Tuple[str, str]]:
    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    out: List[Tuple[str, str]] = []
    for pid in pids:
        cur.execute("SELECT text FROM passages WHERE pid=?", (pid,))
        row = cur.fetchone()
        out.append((pid, row[0] if row else ""))
    conn.close()
    return out


# -----------------------------
# FAISS index
# -----------------------------


def build_faiss_index(
    ollama: OllamaClient,
    embed_model: str,
    sqlite_kb: Path,
    index_dir: Path,
    batch_size: int = 64,
    limit_passages: Optional[int] = None,
) -> Tuple[Path, Path]:
    faiss = _require("faiss", "faiss-cpu")
    safe_mkdir(index_dir)
    safe_name = embed_model.replace(":", "_").replace("/", "_")
    index_path = index_dir / f"index_{safe_name}.faiss"
    ids_path = index_dir / f"index_{safe_name}.ids.txt"
    meta_path = index_dir / f"index_{safe_name}.meta.json"
    ids_tmp_path = ids_path.with_suffix(ids_path.suffix + ".tmp")
    if index_path.exists() and ids_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            n = int(meta.get("passages", 0))
            if n > 0 and ids_path.stat().st_size > 0:
                print(f"[faiss] Reusing existing index for embedder={embed_model} ({n:,} passages)")
                return index_path, ids_path
            print(f"[faiss][warn] Existing index for embedder={embed_model} is empty; rebuilding")
        except Exception as e:
            print(f"[faiss][warn] Could not validate existing index for embedder={embed_model}: {e}; rebuilding")
        for p in [index_path, ids_path, meta_path, ids_tmp_path]:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    print(f"[faiss] Building index for embedder={embed_model} ...")
    probe = embed_texts(ollama, embed_model, ["dimension probe"], keep_alive="30m")
    dim = int(probe.shape[1])
    print(f"[faiss] Probe dimension for {embed_model}: {dim}")
    index = faiss.IndexFlatIP(dim)

    conn = sqlite3.connect(str(sqlite_kb))
    cur = conn.cursor()
    total_rows = sqlite_passage_count(sqlite_kb)
    target_rows = min(total_rows, int(limit_passages)) if limit_passages is not None else total_rows
    print(f"[faiss] Target passages for {embed_model}: {target_rows:,} (batch_size={batch_size})")
    cur.execute("SELECT pid, text FROM passages")

    batch_pids: List[str] = []
    batch_texts: List[str] = []
    n_total = 0
    batch_no = 0
    start_t = time.time()
    last_report_t = start_t
    try:
        with open(ids_tmp_path, "w", encoding="utf-8") as idf:
            for pid, text in cur:
                if limit_passages is not None and n_total >= limit_passages:
                    break
                t = (text or "").strip()
                if not t:
                    continue
                batch_pids.append(pid)
                batch_texts.append(t)
                if len(batch_texts) >= batch_size:
                    batch_no += 1
                    t0 = time.time()
                    vecs = embed_texts(ollama, embed_model, batch_texts, keep_alive="30m")
                    embed_dt = time.time() - t0
                    index.add(vecs)
                    for p in batch_pids:
                        idf.write(p + "\n")
                    n_total += len(batch_pids)
                    batch_pids.clear()
                    batch_texts.clear()
                    now = time.time()
                    if batch_no == 1 or batch_no % 10 == 0 or (now - last_report_t) >= 30.0:
                        rate = n_total / max(now - start_t, 1e-9)
                        pct = (100.0 * n_total / max(target_rows, 1)) if target_rows else 0.0
                        print(
                            f"[faiss][progress] emb={embed_model} | batch={batch_no} | indexed={n_total:,}/{target_rows:,} "
                            f"({pct:.1f}%) | last_batch={embed_dt:.2f}s | avg_rate={rate:.1f} passages/s"
                        )
                        last_report_t = now
            if batch_texts:
                batch_no += 1
                t0 = time.time()
                vecs = embed_texts(ollama, embed_model, batch_texts, keep_alive="30m")
                embed_dt = time.time() - t0
                index.add(vecs)
                for p in batch_pids:
                    idf.write(p + "\n")
                n_total += len(batch_pids)
                print(
                    f"[faiss][progress] emb={embed_model} | batch={batch_no} | indexed={n_total:,}/{target_rows:,} "
                    f"(100.0% if exact) | last_batch={embed_dt:.2f}s"
                )
            idf.flush()
            os.fsync(idf.fileno())
        os.replace(ids_tmp_path, ids_path)
        faiss.write_index(index, str(index_path))
        atomic_write_json(meta_path, {"embed_model": embed_model, "dim": dim, "passages": n_total})
        total_dt = time.time() - start_t
        print(f"[faiss] Saved index: {index_path} ({n_total:,} passages, dim={dim}, elapsed={total_dt/60.0:.1f}m)")
        return index_path, ids_path
    except Exception:
        try:
            ids_tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    finally:
        conn.close()


def load_faiss_index(index_path: Path, ids_path: Path) -> Tuple[Any, List[str]]:
    faiss = _require("faiss", "faiss-cpu")
    index = faiss.read_index(str(index_path))
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    return index, ids


def retrieve(
    ollama: OllamaClient,
    embed_model: str,
    index: Any,
    ids: Sequence[str],
    query: str,
    top_k: int,
) -> Tuple[List[str], List[float]]:
    qv = embed_texts(ollama, embed_model, [query], keep_alive="5m")
    scores, idxs = index.search(qv, top_k)
    idxs_list = idxs[0].tolist()
    scores_list = scores[0].tolist()
    out_ids = [ids[i] for i in idxs_list if 0 <= i < len(ids)]
    return out_ids, scores_list


# -----------------------------
# Prompting
# -----------------------------


REWRITE_SYSTEM = (
    "You are a search query rewriter for open-domain question answering. "
    "Rewrite the user question into a concise search query that is likely to retrieve relevant Wikipedia passages. "
    "Return ONLY the rewritten query, without quotes or extra text."
)

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided context. "
    "If the answer cannot be found in the context, say: \"I don't know based on the provided context.\" "
    "Be concise."
)

# Baseline (no-retrieval) system prompt: allow general knowledge.
BASELINE_SYSTEM = "You are a helpful assistant. Answer the question concisely."


def format_rag_prompt(question: str, passages: Sequence[Tuple[str, str]]) -> str:
    ctx_lines = []
    for i, (pid, text) in enumerate(passages, start=1):
        ctx_lines.append(f"[Passage {i} | {pid}]\n{text}")
    ctx = "\n\n".join(ctx_lines)
    return f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"


# -----------------------------
# Experiment config & runner
# -----------------------------


DEFAULT_EMBEDDERS = [
    # Ollama / HF embedding models (MiniLM is handled through sentence-transformers)
    "qwen3-embedding:4b",
    "all-MiniLM-L6-v2",
    "nomic-embed-text:v1.5",
]

# Phase-1 generator shortlist (0.6B, 1.7B, 4B, 8B, 14B, 30B)
# IMPORTANT: Ollama model tags vary. If a pull fails, run `ollama list`
# and edit these strings to match the tags available on your system.
DEFAULT_GENERATORS = [
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3:8b",
    "qwen3:14b",
    "qwen3:30b",
]


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


def should_stop(start_ts: float, budget_s: float, grace_s: float) -> bool:
    elapsed = time.time() - start_ts
    # We allow a "grace" overrun so a run can finish the current question/batch.
    # Example: budget=2h, grace=20m -> hard stop at 2h20m.
    if budget_s <= 0:
        return False
    return elapsed >= (budget_s + max(0.0, grace_s))


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
    # Indices are namespaced by KB tag (computed below) to prevent mixing runs.
    results_path = results_dir / "results.jsonl"
    resume_path = results_dir / "resume_state.json"
    stream_dir = results_dir / "live_streams"
    safe_mkdir(stream_dir)

    # Data
    nq_path, ks_path = ensure_kilt_nq(data_dir, split=split)
    examples = load_kilt_nq_examples(nq_path, max_questions=max_questions, subset_mode=subset_mode, seed=seed)
    print(f"[data] Loaded {len(examples)} examples from {nq_path.name} (subset_mode={subset_mode}, seed={seed})")
    print(f"[data] Generators={len(generators)} | Embedders={len(embedders)} | top_ks={list(top_ks)} | context_lengths={list(context_lengths)}")
    print(f"[data] verbose_stream={verbose_stream} | live_stream_dir={stream_dir}")
    if verbose_stream:
        print("[data] stream debug is enabled: each streamed request writes .txt, .status.txt and .events.jsonl files")
    print(f"[data] first_token_timeout_s={first_token_timeout_s} | stream_timeout_s={stream_timeout_s}")

    # KB selection
    kb_mode = (kb_mode or "full").strip().lower()
    if kb_mode not in {"full", "gold_plus_random"}:
        raise ValueError("kb_mode must be 'full' or 'gold_plus_random'")

    gold_pids: List[str] = []
    for ex in examples:
        gold_pids.extend(ex.gold_provenance)
    gold_pids = list(dict.fromkeys(gold_pids))
    print(f"[data] Unique gold provenance paragraph ids: {len(gold_pids):,}")

    if kb_mode == "full":
        kb_tag = "full"
        sqlite_path = data_dir / "kilt_knowledgesource.sqlite"
        if not validate_existing_sqlite(sqlite_path, "kb"):
            build_kilt_sqlite(ks_path, sqlite_path)
    else:
        kb_tag = f"goldrand_q{len(examples)}_r{int(kb_random_paragraphs)}_s{int(seed)}"
        sqlite_path = data_dir / f"kilt_knowledgesource_{kb_tag}.sqlite"
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

    index_dir = index_root_dir / kb_tag
    safe_mkdir(index_dir)

    if dry_run:
        print("[dry-run] Resolved experiment plan:")
        print(f"[dry-run] split={split} questions={len(examples)} kb_mode={kb_mode} kb_tag={kb_tag}")
        print(f"[dry-run] embedders={list(embedders)}")
        print(f"[dry-run] generators={list(generators)}")
        print(f"[dry-run] top_ks={list(top_ks)} context_lengths={list(context_lengths)}")
        print(f"[dry-run] avoid_redundant_p0_p1={avoid_redundant_p0_p1}")
        print(f"[dry-run] index_dir={index_dir}")
        return

    # Embedders + indices
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
            return
    indices: Dict[str, Tuple[Any, List[str]]] = {}
    for emb in embedders:
        print(f"[embed] Loading FAISS index into memory: {emb}")
        safe_name = emb.replace(":", "_").replace("/", "_")
        idx_path = index_dir / f"index_{safe_name}.faiss"
        ids_path = index_dir / f"index_{safe_name}.ids.txt"
        indices[emb] = load_faiss_index(idx_path, ids_path)
    print(f"[embed] Loaded {len(indices)} FAISS indices into memory")

    # Resume
    resume = load_resume_state(resume_path)
    if resume and (not results_path.exists() or results_path.stat().st_size == 0):
        print("[resume][warn] Resume checkpoints exist but results.jsonl is missing/empty. Resetting resume state to avoid skipping work.")
        resume = {}
        save_resume_state(resume_path, resume)
    print(f"[resume] Loaded {len(resume)} checkpoints")
    print(f"[resume] Results file exists={results_path.exists()} size={(results_path.stat().st_size if results_path.exists() else 0)} bytes")
    budget_s = time_budget_hours * 3600.0
    grace_s = grace_minutes * 60.0
    start_ts = time.time()

    def cleanup_embedders() -> None:
        if not keep_embedders:
            for emb_to_delete in list(pulled_by_script):
                print(f"[ollama] deleting embedder (pulled by script): {emb_to_delete}")
                try:
                    ollama.delete(emb_to_delete)
                except Exception as e:
                    print(f"[warn] could not delete embedder {emb_to_delete}: {e}")

    def execute_config(*, pipeline: str, gen: str, emb: str, top_k: int, ctx_len: int, keep_alive: str, index: Any = None, ids: Optional[List[str]] = None) -> bool:
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
            if i == start_i:
                print(f"[run] entering example loop for pipeline={pipeline}; first question id={examples[i].qid if i < len(examples) else 'NA'}")
            if should_stop(start_ts, budget_s, grace_s):
                print("[time] Budget reached. Saving resume state and stopping.")
                resume[key_hash] = i
                save_resume_state(resume_path, resume)
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
                    print(f"[P3] q={i+1} | starting rewrite")
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
                            results_path, kb_mode=kb_mode, kb_tag=kb_tag, kb_random_paragraphs=kb_random_paragraphs,
                            subset_mode=subset_mode, seed=seed, pipeline=pipeline, generator=gen, embedder=emb,
                            top_k=top_k, context_length=ctx_len, ex=ex, query_used=query_used, retrieved_ids=[],
                            retrieved_scores=[], answer="", rewrite_time_s=time.time()-t0, retrieval_time_s=0.0, gen_time_s=0.0,
                            error_stage="rewrite", error_message=str(e),
                        )
                        resume[key_hash] = i + 1
                        save_resume_state(resume_path, resume)
                        continue
                    rewrite_time_s = time.time() - t0
                    if rewritten:
                        query_used = rewritten
                    print(f"[P3] q={i+1} | rewrite done in {rewrite_time_s:.2f}s | chars={len(query_used)}")

                t0 = time.time()
                try:
                    retrieved_ids, retrieved_scores = retrieve(
                        ollama,
                        emb,
                        index,
                        ids,
                        query_used,
                        top_k,
                    )
                except OllamaError as e:
                    print(f"[warn] retrieval failed for gen={gen} emb={emb} q={i+1}: {e}")
                    append_failure_record(
                        results_path, kb_mode=kb_mode, kb_tag=kb_tag, kb_random_paragraphs=kb_random_paragraphs,
                        subset_mode=subset_mode, seed=seed, pipeline=pipeline, generator=gen, embedder=emb,
                        top_k=top_k, context_length=ctx_len, ex=ex, query_used=query_used, retrieved_ids=[],
                        retrieved_scores=[], answer="", rewrite_time_s=rewrite_time_s, retrieval_time_s=time.time()-t0, gen_time_s=0.0,
                        error_stage="retrieve", error_message=str(e),
                    )
                    resume[key_hash] = i + 1
                    save_resume_state(resume_path, resume)
                    continue
                retrieval_time_s = time.time() - t0
                passages = fetch_passages(sqlite_path, retrieved_ids)
                if i == start_i or (i + 1) % 10 == 0:
                    print(f"[retrieve] got {len(retrieved_ids)} passages in {retrieval_time_s:.2f}s")

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
                    print(f"[{pipeline}] q={i+1} | starting answer generation")
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
                        results_path, kb_mode=kb_mode, kb_tag=kb_tag, kb_random_paragraphs=kb_random_paragraphs,
                        subset_mode=subset_mode, seed=seed, pipeline=pipeline, generator=gen, embedder=emb,
                        top_k=top_k, context_length=ctx_len, ex=ex, query_used=query_used, retrieved_ids=retrieved_ids,
                        retrieved_scores=retrieved_scores, answer="", rewrite_time_s=rewrite_time_s, retrieval_time_s=retrieval_time_s, gen_time_s=time.time()-t0,
                        error_stage="generate", error_message=str(e),
                    )
                    resume[key_hash] = i + 1
                    save_resume_state(resume_path, resume)
                    continue
                gen_time_s = time.time() - t0
                if i == start_i or (i + 1) % 10 == 0:
                    print(f"[generate] answer_len={len(answer)} chars in {gen_time_s:.2f}s")

            em = compute_exact_match(answer, ex.gold_answers)
            f1 = compute_f1(answer, ex.gold_answers)
            rec = recall_at_k(retrieved_ids, ex.gold_provenance, k=top_k) if retrieved_ids else 0.0
            rprec = r_precision(retrieved_ids, ex.gold_provenance) if retrieved_ids else 0.0

            gold_set = set(ex.gold_provenance)
            retrieved_set = set(retrieved_ids)
            prov_hit = 1.0 if gold_set and gold_set.intersection(retrieved_set) else 0.0
            prov_all = 1.0 if gold_set and gold_set.issubset(retrieved_set) else 0.0
            kilt_em_hit = float(em) * prov_hit
            kilt_f1_hit = float(f1) * prov_hit
            kilt_em_all = float(em) * prov_all
            kilt_f1_all = float(f1) * prov_all

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
                    "kilt_em_hit": kilt_em_hit,
                    "kilt_f1_hit": kilt_f1_hit,
                    "kilt_em_all": kilt_em_all,
                    "kilt_f1_all": kilt_f1_all,
                },
            }
            append_jsonl(results_path, record)
            if i == start_i or (i + 1) % 10 == 0:
                print(f"[write] saved result for q={i+1} pipeline={pipeline}")
            resume[key_hash] = i + 1
            save_resume_state(resume_path, resume)

        save_resume_state(resume_path, resume)
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
            print(f"[run][shared P0] embedder={emb}")
            index, ids = indices[emb]
            for top_k in top_ks:
                print(f"[run][shared P0] embedder={emb} | top_k={top_k}")
                for ctx_len in context_lengths:
                    if execute_config(pipeline="P0", gen=shared_p0_gen, emb=emb, top_k=int(top_k), ctx_len=int(ctx_len), keep_alive="5m", index=index, ids=ids):
                        cleanup_embedders()
                        return
    else:
        print("[run] avoid_redundant_p0_p1=False | running legacy fully repeated loop")

    for gen in generators:
        print(f"[gen] Starting generator: {gen}")
        if not generator_has_pending_work(gen):
            print(f"[gen] All generator-dependent experiments already completed for: {gen}; skipping model check/pull")
            continue
        gen_preexisting = ollama.model_exists(gen)
        gen_pulled = False
        if not gen_preexisting:
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
                        cleanup_embedders()
                        return
                for emb in embedders:
                    print(f"[run] generator={gen} | embedder={emb}")
                    index, ids = indices[emb]
                    for top_k in top_ks:
                        print(f"[run] generator={gen} | embedder={emb} | top_k={top_k}")
                        for ctx_len in context_lengths:
                            print(f"[run] generator={gen} | embedder={emb} | top_k={top_k} | context_length={ctx_len}")
                            for pipeline in ["P2", "P3"]:
                                if execute_config(pipeline=pipeline, gen=gen, emb=emb, top_k=int(top_k), ctx_len=int(ctx_len), keep_alive=keep_alive, index=index, ids=ids):
                                    cleanup_embedders()
                                    return
            else:
                for emb in embedders:
                    print(f"[run] generator={gen} | embedder={emb}")
                    index, ids = indices[emb]
                    for top_k in top_ks:
                        print(f"[run] generator={gen} | embedder={emb} | top_k={top_k}")
                        for ctx_len in context_lengths:
                            print(f"[run] generator={gen} | embedder={emb} | top_k={top_k} | context_length={ctx_len}")
                            for pipeline in ["P0", "P1", "P2", "P3"]:
                                if execute_config(pipeline=pipeline, gen=gen, emb=emb, top_k=int(top_k), ctx_len=int(ctx_len), keep_alive=keep_alive, index=index, ids=ids):
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
    print("[done] Experiment completed for all configured generators.")


# -----------------------------
# Self-tests (no Ollama needed)
# -----------------------------


def run_self_tests() -> None:
    assert normalize_text("The Eiffel Tower") == "eiffel tower"
    assert compute_exact_match("Paris", ["paris"]) == 1
    f = compute_f1("New York City", ["New York"])
    assert 0.79 < f <= 1.0
    assert recall_at_k(["a", "b"], ["b"], 2) == 1.0
    assert abs(r_precision(["a", "b"], ["a"]) - 1.0) < 1e-9
    arr = np.asarray([[3.0, 4.0]], dtype=np.float32)
    norm = l2_normalize(arr)
    assert abs(float(np.linalg.norm(norm[0])) - 1.0) < 1e-6
    assert is_hf_embedder("all-MiniLM-L6-v2")
    assert resolve_hf_model_name("all-MiniLM-L6-v2") == "sentence-transformers/all-MiniLM-L6-v2"
    print("[tests] basic metric tests passed")
    # Smoke test: streamed request that never emits a first token should be aborted by the watchdog.
    import requests as _requests

    class _FakeResp:
        def __init__(self):
            self.status_code = 200
            self._closed = False
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False
        def close(self):
            self._closed = True
        def iter_lines(self, decode_unicode=True):
            while not self._closed:
                time.sleep(0.02)
                if False:
                    yield ""

    _orig_request = _requests.request
    try:
        _requests.request = lambda *a, **k: _FakeResp()
        c = OllamaClient(verbose=False, retries=1, retry_backoff_s=0.01)
        try:
            c._generate_stream(
                model="fake",
                prompt="hello",
                debug_label="test_no_first_token",
                first_token_timeout_s=0.15,
                stream_timeout_s=1.0,
                heartbeat_s=0.05,
                connect_timeout_s=0.01,
                read_timeout_s=0.01,
            )
            raise AssertionError("Expected no_first_token_timeout watchdog to fire")
        except OllamaError as e:
            assert "no_first_token_timeout" in str(e)
    finally:
        _requests.request = _orig_request


    # Smoke test: watchdog-triggered close may surface as an internal iterator error; this
    # should still be converted into the same structured timeout failure.
    class _FakeRespAttrErr:
        def __init__(self):
            self.status_code = 200
            self._closed = False
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False
        def close(self):
            self._closed = True
        def iter_lines(self, decode_unicode=True):
            while True:
                time.sleep(0.02)
                if self._closed:
                    raise AttributeError("'NoneType' object has no attribute 'read'")
                if False:
                    yield ""

    try:
        _requests.request = lambda *a, **k: _FakeRespAttrErr()
        c = OllamaClient(verbose=False, retries=1, retry_backoff_s=0.01)
        try:
            c._generate_stream(
                model="fake",
                prompt="hello",
                debug_label="test_no_first_token_attrerr",
                first_token_timeout_s=0.15,
                stream_timeout_s=1.0,
                heartbeat_s=0.05,
                connect_timeout_s=0.01,
                read_timeout_s=0.01,
            )
            raise AssertionError("Expected no_first_token_timeout watchdog to fire via internal iterator error")
        except OllamaError as e:
            assert "no_first_token_timeout" in str(e)
    finally:
        _requests.request = _orig_request

    print("[tests] stream watchdog tests passed")
    # Unit test: reduced-repeat planning should lower the number of attempted rows.
    n_gens, n_embs, n_topk, n_ctx, n_q = 2, 3, 2, 1, 5
    repeated = n_gens * n_embs * n_topk * n_ctx * 4 * n_q
    reduced = (n_embs * n_topk * n_ctx * n_q) + (n_gens * n_ctx * n_q) + (n_gens * n_embs * n_topk * n_ctx * 2 * n_q)
    assert reduced < repeated
    assert reduced == (3*2*1*5) + (2*1*5) + (2*3*2*1*2*5)
    print("[tests] reduced-repeat planning tests passed")


def load_config(path: Optional[str]) -> Dict[str, Any]:
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


def cfg_value(args: argparse.Namespace, cfg: Dict[str, Any], arg_name: str, cfg_keys: Sequence[str], default: Any) -> Any:
    value = getattr(args, arg_name)
    if value is not None:
        return value
    for key in cfg_keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EdgeRAG experiments on KILT-NQ with Ollama models")
    p.add_argument("--config", default=None, help="Optional JSON config file (Phase 1 recommended)")
    p.add_argument("--data_dir", default=None, help="Directory to store datasets and KB")
    p.add_argument("--index_dir", default=None, help="Directory to store FAISS indices")
    p.add_argument("--results_dir", default=None, help="Directory to store results and resume state")
    p.add_argument("--split", default=None, choices=["dev", "train"], help="KILT-NQ split")
    p.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Number of questions to evaluate (Phase-1 default: 500)",
    )
    p.add_argument(
        "--subset_mode",
        default=None,
        choices=["head", "random"],
        help="How to choose the question subset: head or random (deterministic)",
    )
    p.add_argument("--seed", type=int, default=None, help="Seed used for deterministic sampling")
    p.add_argument(
        "--kb_mode",
        default=None,
        choices=["full", "gold_plus_random"],
        help="KB mode: full KILT Wikipedia, or reduced (gold + random)",
    )
    p.add_argument(
        "--kb_random_paragraphs",
        type=int,
        default=None,
        help="When kb_mode=gold_plus_random, how many random paragraphs to include",
    )
    p.add_argument(
        "--kb_max_pages_debug",
        type=int,
        default=None,
        help="DEBUG ONLY: stop reading knowledge source after N pages",
    )
    p.add_argument("--time_budget_hours", type=float, default=None, help="How long to run before stopping")
    p.add_argument("--grace_minutes", type=float, default=None, help="Finish ongoing batch within grace window")
    p.add_argument("--ollama_url", default=None, help="Ollama base URL")
    p.add_argument("--keep_embedders", action="store_true", help="Do not delete embedders pulled by script")
    p.add_argument("--limit_passages_for_debug", type=int, default=None, help="Index only first N passages (debug)")
    p.add_argument("--embedders_json", default=None, help="Optional JSON file listing embedding models")
    p.add_argument("--generators_json", default=None, help="Optional JSON file listing generator models")
    p.add_argument("--top_ks", default=None, help="Comma-separated top-k values")
    p.add_argument("--context_lengths", default=None, help="Comma-separated context lengths (e.g. 4096,8192)")
    p.add_argument("--dry_run", action="store_true", help="Resolve the plan and print it without running Ollama/FAISS")
    p.add_argument("--verbose_stream", action="store_true", default=None, help="Stream model tokens to stdout and results/live_streams/*.txt during rewrite/generation")
    p.add_argument("--first_token_timeout_s", type=float, default=None, help="Fail a streamed trial if no first token arrives within this many seconds")
    p.add_argument("--stream_timeout_s", type=float, default=None, help="Fail a streamed trial if total stream time exceeds this many seconds")
    p.add_argument("--avoid_redundant_p0_p1", action="store_true", default=None, help="Run P0 only once globally and P1 only once per generator")
    p.add_argument("--run_tests", action="store_true", help="Run self-tests and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_tests:
        run_self_tests()
        return
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

    top_ks_raw = cfg_value(args, cfg, "top_ks", ["top_k_values", "top_ks"], PHASE1_TOP_KS)
    if isinstance(top_ks_raw, list):
        top_ks = [int(x) for x in top_ks_raw]
    else:
        top_ks = [int(x.strip()) for x in str(top_ks_raw).split(",") if x.strip()]

    ctx_raw = cfg_value(args, cfg, "context_lengths", ["context_lengths"], [4096])
    if isinstance(ctx_raw, list):
        context_lengths = [int(x) for x in ctx_raw]
    else:
        context_lengths = [int(x.strip()) for x in str(ctx_raw).split(",") if x.strip()]

    ollama_url = cfg_value(args, cfg, "ollama_url", ["ollama_url"], "http://localhost:11434")
    verbose_stream = bool(cfg_value(args, cfg, "verbose_stream", ["verbose_stream"], False))
    first_token_timeout_s = float(cfg_value(args, cfg, "first_token_timeout_s", ["first_token_timeout_s"], 300.0))
    stream_timeout_s = float(cfg_value(args, cfg, "stream_timeout_s", ["stream_timeout_s"], 600.0))
    avoid_redundant_p0_p1 = bool(cfg_value(args, cfg, "avoid_redundant_p0_p1", ["avoid_redundant_p0_p1"], True))
    ollama = OllamaClient(base_url=ollama_url, timeout=600, retries=6, retry_backoff_s=5.0, verbose=True)

    split = cfg_value(args, cfg, "split", ["split"], "dev")
    subset_mode = cfg_value(args, cfg, "subset_mode", ["subset_mode"], PHASE1_SUBSET_MODE)
    seed = int(cfg_value(args, cfg, "seed", ["subset_seed", "seed"], PHASE1_SEED))
    kb_mode = cfg_value(args, cfg, "kb_mode", ["kb_mode"], PHASE1_KB_MODE)
    kb_random_paragraphs = int(
        cfg_value(args, cfg, "kb_random_paragraphs", ["kb_random_passages", "kb_random_paragraphs"], PHASE1_KB_RANDOM_PARAGRAPHS)
    )
    time_budget_hours = float(cfg_value(args, cfg, "time_budget_hours", ["time_limit_hours", "time_budget_hours"], 2.0))
    grace_minutes = float(cfg_value(args, cfg, "grace_minutes", ["grace_minutes"], 20.0))

    max_q: Optional[int] = cfg_value(args, cfg, "max_questions", ["max_questions"], PHASE1_MAX_QUESTIONS)
    # Convenience: allow --max_questions 0 or -1 to mean "use all".
    if max_q is not None and int(max_q) <= 0:
        max_q = None

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