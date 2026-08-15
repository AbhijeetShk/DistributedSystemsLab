import pytest

from benchmark import BenchmarkConfig, get_config


def test_get_ddp_config():
    config = get_config("ddp")

    assert isinstance(config, BenchmarkConfig)
    assert config.name == "ddp"
    assert config.batch_size == 8
    assert config.sequence_length == 32
    assert config.warmup_steps == 5
    assert config.benchmark_steps == 20


def test_all_strategies_are_configured():
    expected = {
        "ddp",
        "fsdp",
        "zero3",
        "tensor_parallel",
        "pipeline_parallel",
    }

    assert {get_config(name).name for name in expected} == expected


def test_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        get_config("not_a_real_strategy")
