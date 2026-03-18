from pathlib import Path
from datetime import datetime, timedelta
import math

# =========================
# CONFIG
# =========================
FOLDERS = [
    Path("results_phase1/live_streams"),
    Path("results_phase1/live_streams_0"),
]

# Gap threshold X in seconds.
# If the difference between two consecutive files is > X,
# we assume the GPU stopped and a new run started.
GAP_THRESHOLD_SECONDS = 600  # 10 minutes

# Choose which timestamp to use:
# "modified" -> last modified time
# "created"  -> creation time on Windows
TIME_FIELD = "modified"  # change to "created" if you prefer


# =========================
# HELPERS
# =========================
def get_file_timestamp(path: Path, time_field: str) -> float:
    """
    Return the timestamp (seconds since epoch) for a file.

    On Windows:
      - st_ctime is usually creation time
      - st_mtime is modification time

    On Unix/Linux/macOS:
      - st_ctime is metadata-change time, not true creation time
      - st_mtime is modification time

    So "modified" is the safest cross-platform option.
    """
    stat = path.stat()

    if time_field == "created":
        return stat.st_ctime
    elif time_field == "modified":
        return stat.st_mtime
    else:
        raise ValueError(f"Unsupported TIME_FIELD: {time_field}")


def collect_files(folders, time_field):
    """
    Collect all real files from the given folders and return a list of:
        (timestamp, Path)
    """
    files = []

    for folder in folders:
        if not folder.exists():
            print(f"[warn] Folder does not exist: {folder}")
            continue

        for p in folder.rglob("*"):
            if p.is_file():
                try:
                    ts = get_file_timestamp(p, time_field)
                    files.append((ts, p))
                except Exception as e:
                    print(f"[warn] Could not read timestamp for {p}: {e}")

    return files


def split_into_sessions(sorted_files, gap_threshold_seconds):
    """
    Split the sorted files into sessions.

    A new session starts whenever:
        T_(i+1) - T_i > gap_threshold_seconds
    """
    if not sorted_files:
        return []

    sessions = []
    current_session = [sorted_files[0]]

    for prev, curr in zip(sorted_files[:-1], sorted_files[1:]):
        prev_ts, _ = prev
        curr_ts, _ = curr
        gap = curr_ts - prev_ts

        if gap > gap_threshold_seconds:
            sessions.append(current_session)
            current_session = [curr]
        else:
            current_session.append(curr)

    sessions.append(current_session)
    return sessions


def format_seconds(seconds: float) -> str:
    td = timedelta(seconds=round(seconds))
    total_seconds = int(td.total_seconds())

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours}h {minutes}m {secs}s"


def dt_str(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# =========================
# MAIN
# =========================
def main():
    files = collect_files(FOLDERS, TIME_FIELD)

    if not files:
        print("No files found.")
        return

    # Sort globally across both folders
    files.sort(key=lambda x: x[0])

    sessions = split_into_sessions(files, GAP_THRESHOLD_SECONDS)

    total_runtime_seconds = 0.0

    print("=" * 80)
    print(f"Time field used      : {TIME_FIELD}")
    print(f"Gap threshold        : {GAP_THRESHOLD_SECONDS} seconds")
    print(f"Total files analyzed : {len(files)}")
    print(f"Detected sessions    : {len(sessions)}")
    print("=" * 80)

    for i, session in enumerate(sessions, start=1):
        start_ts, start_file = session[0]
        end_ts, end_file = session[-1]

        runtime = end_ts - start_ts
        total_runtime_seconds += runtime

        print(f"\nSession {i}")
        print(f"  Files in session : {len(session)}")
        print(f"  Start time       : {dt_str(start_ts)}")
        print(f"  End time         : {dt_str(end_ts)}")
        print(f"  Duration         : {format_seconds(runtime)}")
        print(f"  First file       : {start_file}")
        print(f"  Last file        : {end_file}")

        if len(session) > 1:
            # Optional: show the largest internal gap within the session
            max_gap = max(session[j + 1][0] - session[j][0] for j in range(len(session) - 1))
            print(f"  Max internal gap : {max_gap:.1f} seconds")

    print("\n" + "=" * 80)
    print("TOTAL GPU RUNTIME ESTIMATE")
    print("=" * 80)
    print(f"Total runtime: {format_seconds(total_runtime_seconds)}")
    print(f"Total hours  : {total_runtime_seconds / 3600:.3f}")
    print(f"Total minutes: {total_runtime_seconds / 60:.1f}")


if __name__ == "__main__":
    main()