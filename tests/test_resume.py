from edgerag.cli.rebuild_resume import reconstruct_resume_state
from edgerag.core.resume import RunKey


def test_reconstruct_resume_state_tracks_max_index():
    qid_to_idx = {"q1": 0, "q2": 1}
    rows = [
        {"question_id": "q1", "pipeline": "P2", "generator": "g", "embedder": "e", "top_k": 5, "context_length": 4096},
        {"question_id": "q2", "pipeline": "P2", "generator": "g", "embedder": "e", "top_k": 5, "context_length": 4096},
    ]
    state, bad_rows, used_rows, skipped_rows = reconstruct_resume_state(qid_to_idx, rows)
    key = RunKey("P2", "g", "e", 5, 4096).to_str()
    assert state[key] == 2
    assert bad_rows == 0 and used_rows == 2 and skipped_rows == 0
