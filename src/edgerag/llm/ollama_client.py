from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from edgerag.core.utils import append_jsonl_debug, now_iso, safe_mkdir, write_text_atomic


class OllamaError(RuntimeError):
    pass


def make_stream_callback(
    enabled: bool,
    label: str,
    stream_file: Optional[Path] = None,
) -> Optional[Callable[[str], None]]:
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


class OllamaClient:
    """Minimal Ollama HTTP client with retries and watchdog-aware streaming."""

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
                response = requests.request(method, url, json=payload, timeout=timeout)
                if response.status_code == 200:
                    try:
                        return response.json()
                    except Exception as e:
                        raise OllamaError(f"Ollama returned non-JSON: {response.text[:200]}") from e
                try:
                    msg = response.json()
                except Exception:
                    msg = response.text
                retryable = response.status_code in {408, 409, 429, 500, 502, 503, 504}
                if retryable and attempt < retries:
                    wait_s = min(30.0, self.retry_backoff_s * attempt)
                    if self.verbose:
                        print(f"[ollama][retry {attempt}/{retries}] HTTP {response.status_code} on {method} {path}; sleeping {wait_s:.1f}s")
                    time.sleep(wait_s)
                    continue
                raise OllamaError(f"Ollama error {response.status_code} for {method} {url}: {msg}")
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
        retries = self.retries if retries is None else max(1, int(retries))
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
        if system is not None:
            payload["system"] = system
        if options:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

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
            payload_local = {
                "ts": now_iso(),
                "kind": kind,
                "label": debug_label or model,
                "elapsed_s": round(time.time() - start_wall, 3),
            }
            payload_local.update(extra)
            append_jsonl_debug(events_file, payload_local)

        heartbeat_thread: Optional[threading.Thread] = None
        if debug_file is not None:
            status_file = debug_file.with_suffix(debug_file.suffix + ".status.txt")
            events_file = debug_file.with_suffix(debug_file.suffix + ".events.jsonl")
            write_text_atomic(status_file, f"[{debug_label or model}]\nstatus=starting\nelapsed_s=0\nchars=0\nfirst_token=no\n")
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
                        write_text_atomic(status_file, msg)
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
                write_text_atomic(status_file, text)

        attempt = 0
        while True:
            attempt += 1
            chunks: List[str] = []
            try:
                with requests.request("POST", url, json=payload, timeout=(connect_timeout_s, read_timeout_s), stream=True) as response:
                    current_response["obj"] = response
                    state["status"] = f"http_{response.status_code}"
                    log_event("http_response", status_code=response.status_code)
                    if response.status_code != 200:
                        try:
                            msg = response.json()
                        except Exception:
                            msg = response.text
                        retryable = response.status_code in {408, 409, 429, 500, 502, 503, 504}
                        elapsed = time.time() - start_wall
                        allow_more = state["first_token_t"] is None and first_token_timeout_s is not None and elapsed < float(first_token_timeout_s)
                        if retryable and (attempt < retries or allow_more):
                            wait_s = min(30.0, self.retry_backoff_s * min(attempt, max(1, retries)))
                            if self.verbose:
                                print(f"[ollama][retry {attempt}/{retries}] HTTP {response.status_code} on POST /api/generate(stream); sleeping {wait_s:.1f}s")
                            time.sleep(wait_s)
                            continue
                        raise OllamaError(f"Ollama error {response.status_code} for POST {url}: {msg}")
                    state["status"] = "stream_open"
                    log_event("stream_open")
                    try:
                        for raw in response.iter_lines(decode_unicode=True):
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
                elapsed = time.time() - start_wall
                if state["first_token_t"] is None and first_token_timeout_s is not None and elapsed >= float(first_token_timeout_s):
                    msg = f"no_first_token_timeout after {elapsed:.1f}s"
                    _finish_status("failed", msg)
                    raise OllamaError(msg) from e
                if state["first_token_t"] is not None and stream_timeout_s is not None and elapsed >= float(stream_timeout_s):
                    msg = f"stream_timeout after {elapsed:.1f}s"
                    _finish_status("failed", msg)
                    raise OllamaError(msg) from e
                allow_more = (
                    state["first_token_t"] is None and first_token_timeout_s is not None and elapsed < float(first_token_timeout_s)
                ) or (
                    state["first_token_t"] is not None and stream_timeout_s is not None and elapsed < float(stream_timeout_s)
                )
                if attempt < retries or allow_more:
                    wait_s = min(30.0, self.retry_backoff_s * min(attempt, max(1, retries)))
                    if self.verbose:
                        print(f"[ollama][retry {attempt}/{retries}] request failed for POST /api/generate(stream): {e}; sleeping {wait_s:.1f}s")
                    time.sleep(wait_s)
                    continue
                _finish_status("failed", str(e))
                raise OllamaError(f"Ollama streamed request failed after {max(retries, attempt)} attempts: POST {url}: {e}") from e
            finally:
                current_response["obj"] = None

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
        for model in data.get("models", []) or []:
            name = model.get("name")
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
