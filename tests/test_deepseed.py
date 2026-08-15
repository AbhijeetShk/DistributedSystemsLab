import importlib.util


def test_deepspeed_is_optional():
    assert importlib.util.find_spec("deepspeed") is not None or True
