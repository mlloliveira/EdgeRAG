from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from edgerag.core.deps import require_optional

_HF_MODEL_CACHE: Dict[str, Any] = {}


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


def embed_texts(ollama: Any, embed_model: str, texts: Sequence[str], keep_alive: str = "30m") -> np.ndarray:
    if is_hf_embedder(embed_model):
        model_name = resolve_hf_model_name(embed_model)
        if model_name not in _HF_MODEL_CACHE:
            print(f"[hf] Loading embedding model: {model_name}")
            st = require_optional("sentence_transformers", "sentence-transformers")
            _HF_MODEL_CACHE[model_name] = st.SentenceTransformer(model_name)
        model = _HF_MODEL_CACHE[model_name]
        vecs = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
    return l2_normalize(ollama.embed(embed_model, texts, keep_alive=keep_alive))
