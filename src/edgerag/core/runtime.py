from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def should_stop(start_ts: float, budget_s: float, grace_s: float) -> bool:
    if budget_s <= 0:
        return False
    import time
    elapsed = time.time() - start_ts
    return elapsed >= (budget_s + max(0.0, grace_s))


@dataclass(frozen=True)
class RuntimeSession:
    start_ts: float
    end_ts: float
    files: List[str]

    @property
    def duration_s(self) -> float:
        return self.end_ts - self.start_ts


def get_file_timestamp(path: Path, time_field: str) -> float:
    stat = path.stat()
    if time_field == "created":
        return stat.st_ctime
    if time_field == "modified":
        return stat.st_mtime
    raise ValueError(f"Unsupported time field: {time_field}")


def collect_files(folders: Sequence[Path], time_field: str) -> List[Tuple[float, Path]]:
    files: List[Tuple[float, Path]] = []
    for folder in folders:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                files.append((get_file_timestamp(path, time_field), path))
    return files


def split_into_sessions(sorted_files: Sequence[Tuple[float, Path]], gap_threshold_seconds: int) -> List[RuntimeSession]:
    if not sorted_files:
        return []
    grouped: List[List[Tuple[float, Path]]] = [[sorted_files[0]]]
    for prev, curr in zip(sorted_files[:-1], sorted_files[1:]):
        if curr[0] - prev[0] > gap_threshold_seconds:
            grouped.append([curr])
        else:
            grouped[-1].append(curr)
    return [
        RuntimeSession(
            start_ts=session[0][0],
            end_ts=session[-1][0],
            files=[str(path) for _, path in session],
        )
        for session in grouped
    ]


def format_seconds(seconds: float) -> str:
    td = timedelta(seconds=round(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours}h {minutes}m {secs}s"


def dt_str(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")



def build_runtime_metadata(*,
    start_ts: float,
    budget_s: float,
    grace_s: float,
    results_path: Path,
    resume_path: Path,
    stream_dir: Path,
    phase: str,
    completed_keys: int,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "phase": phase,
        "started_at": dt_str(start_ts),
        "updated_at": dt_str(time.time()),
        "elapsed_runtime_seconds": max(0.0, time.time() - start_ts),
        "budget_seconds": float(budget_s),
        "grace_seconds": float(grace_s),
        "results_path": str(results_path),
        "resume_path": str(resume_path),
        "live_stream_dir": str(stream_dir),
        "completed_run_keys": int(completed_keys),
    }
    if extra:
        payload.update(extra)
    return payload


def write_runtime_metadata(path: Path, **kwargs: Any) -> None:
    from edgerag.core.utils import atomic_write_json

    atomic_write_json(path, build_runtime_metadata(**kwargs))
