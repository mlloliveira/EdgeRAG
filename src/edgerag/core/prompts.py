from __future__ import annotations

from typing import Sequence, Tuple

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

BASELINE_SYSTEM = "You are a helpful assistant. Answer the question concisely."


def format_rag_prompt(question: str, passages: Sequence[Tuple[str, str]]) -> str:
    ctx_lines = []
    for i, (pid, text) in enumerate(passages, start=1):
        ctx_lines.append(f"[Passage {i} | {pid}]\n{text}")
    ctx = "\n\n".join(ctx_lines)
    return f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
