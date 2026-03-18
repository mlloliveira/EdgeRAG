from pathlib import Path

from edgerag.core.runtime import split_into_sessions


def test_split_into_sessions_creates_new_session_on_gap():
    files = [
        (0.0, Path("a")),
        (10.0, Path("b")),
        (1000.0, Path("c")),
    ]
    sessions = split_into_sessions(files, 100)
    assert len(sessions) == 2
    assert sessions[0].files == ["a", "b"]
    assert sessions[1].files == ["c"]
