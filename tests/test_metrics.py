from edgerag.core.metrics import compute_exact_match, compute_f1, strip_think_prefix_for_scoring


def test_strip_think_prefix_for_scoring():
    assert strip_think_prefix_for_scoring("<think>x</think> Paris") == "Paris"


def test_exact_match_and_f1():
    assert compute_exact_match("Paris", ["Paris"]) == 1
    assert compute_f1("city of paris", ["Paris"]) > 0
