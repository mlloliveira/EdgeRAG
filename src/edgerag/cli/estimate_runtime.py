from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from edgerag.core.runtime import collect_files, dt_str, format_seconds, split_into_sessions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate cumulative GPU runtime from live stream file timestamps")
    parser.add_argument("--results_dir", type=Path, default=Path("results_phase1"), help="Results directory containing live_streams folders")
    parser.add_argument("--folders", nargs="*", default=None, help="Optional explicit folder list; defaults to results_dir/live_streams and live_streams_0")
    parser.add_argument("--gap_threshold_seconds", type=int, default=600, help="Gap that starts a new session")
    parser.add_argument("--time_field", choices=["modified", "created"], default="modified", help="Filesystem timestamp to use")
    parser.add_argument("--output_json", type=Path, default=None, help="Optional output JSON path; defaults to results_dir/runtime_estimate.json")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    folders = [Path(p) for p in args.folders] if args.folders else [args.results_dir / "live_streams", args.results_dir / "live_streams_0"]
    files = collect_files(folders, args.time_field)
    if not files:
        print("No files found.")
        return
    files.sort(key=lambda x: x[0])
    sessions = split_into_sessions(files, args.gap_threshold_seconds)
    total_runtime_seconds = sum(session.duration_s for session in sessions)

    print("=" * 80)
    print(f"Time field used      : {args.time_field}")
    print(f"Gap threshold        : {args.gap_threshold_seconds} seconds")
    print(f"Total files analyzed : {len(files)}")
    print(f"Detected sessions    : {len(sessions)}")
    print("=" * 80)
    payload = {
        "time_field": args.time_field,
        "gap_threshold_seconds": args.gap_threshold_seconds,
        "total_files": len(files),
        "session_count": len(sessions),
        "total_runtime_seconds": total_runtime_seconds,
        "sessions": [],
    }
    for i, session in enumerate(sessions, start=1):
        print(f"\nSession {i}")
        print(f"  Files in session : {len(session.files)}")
        print(f"  Start time       : {dt_str(session.start_ts)}")
        print(f"  End time         : {dt_str(session.end_ts)}")
        print(f"  Duration         : {format_seconds(session.duration_s)}")
        print(f"  First file       : {session.files[0]}")
        print(f"  Last file        : {session.files[-1]}")
        payload["sessions"].append(
            {
                "index": i,
                "files": session.files,
                "start_time": dt_str(session.start_ts),
                "end_time": dt_str(session.end_ts),
                "duration_seconds": session.duration_s,
            }
        )
    print("\n" + "=" * 80)
    print("TOTAL GPU RUNTIME ESTIMATE")
    print("=" * 80)
    print(f"Total runtime: {format_seconds(total_runtime_seconds)}")
    print(f"Total hours  : {total_runtime_seconds / 3600:.3f}")
    print(f"Total minutes: {total_runtime_seconds / 60:.1f}")

    output_json = args.output_json or (args.results_dir / "runtime_estimate.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved runtime summary to: {output_json}")


if __name__ == "__main__":
    main()
