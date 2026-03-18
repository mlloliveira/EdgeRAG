import numpy as np
import pandas as pd

from edgerag.analysis import base as base_module
from edgerag.analysis.sas import compute_sas_scores


class FakeModel:
    def predict(self, pairs, batch_size=32, show_progress_bar=True):
        out = []
        for a, b in pairs:
            if a.lower() == b.lower():
                out.append(3.0)
            elif a.lower() in b.lower() or b.lower() in a.lower():
                out.append(1.0)
            else:
                out.append(-2.0)
        return np.array(out, dtype=np.float32)


def test_compute_sas_scores_with_fake_model():
    raw = pd.DataFrame([
        {
            "pipeline": "P1", "generator": "g1", "generator_label": "G1", "embedder": "__none__", "embedder_label": "__none__",
            "top_k": 0, "context_length": 4096, "status_norm": "ok", "answer": "<think>x</think> Paris",
            "gold_answers": ["Paris"], "retrieved_ids": [], "gold_provenance": []
        }
    ])
    scored = compute_sas_scores(base_module, raw, fake_model=FakeModel(), progress_bar=False)
    assert scored.loc[0, "metric_sas"] > 0.9
