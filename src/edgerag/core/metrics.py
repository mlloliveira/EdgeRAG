from __future__ import annotations

from typing import Any, List, Sequence


def normalize_text(s: str) -> str:
    """SQuAD-style normalization used by both runner and analysis."""
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
    p = normalize_text(pred)
    for g in golds:
        if p == normalize_text(g):
            return 1
    return 0


def compute_f1(pred: str, golds: Sequence[str]) -> float:
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
    if not gold or k <= 0:
        return 0.0
    gset = set(gold)
    topk = set(retrieved[:k])
    return 1.0 if gset.intersection(topk) else 0.0


def r_precision(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    if not gold:
        return 0.0
    gset = set(gold)
    r_value = len(gset)
    if r_value <= 0:
        return 0.0
    top_r = retrieved[:r_value]
    hits = sum(1 for pid in top_r if pid in gset)
    return hits / r_value


def strip_think_prefix_for_scoring(answer: Any) -> Any:
    """Strip one or more leading <think>...</think> blocks before scoring."""
    if not isinstance(answer, str):
        return answer
    cleaned = answer
    lower = cleaned.lower()
    close_tag = "</think>"
    while True:
        start = lower.find("<think")
        end = lower.find(close_tag)
        if start == -1 or end == -1 or start > end:
            break
        cleaned = cleaned[end + len(close_tag):].lstrip()
        lower = cleaned.lower()
    return cleaned
