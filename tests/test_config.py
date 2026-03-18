from edgerag.core.config import cfg_model_names, parse_int_list


def test_cfg_model_names_accepts_model_dicts():
    cfg = {"generator_models": [{"name": "g1"}, {"name": "g2"}]}
    assert cfg_model_names(cfg, ["generator_models"], []) == ["g1", "g2"]


def test_parse_int_list_handles_list_and_string():
    assert parse_int_list([5, 10]) == [5, 10]
    assert parse_int_list("5,10") == [5, 10]
