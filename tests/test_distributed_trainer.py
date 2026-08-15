import pytest
import torch

from models import GPTModel, TransformerConfig
from trainer import (
    DistributedTrainerConfig,
    build_distributed_model,
)


def create_model():
    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
    )

    return GPTModel(config)


def test_single_strategy_moves_model_to_device():
    model = create_model()

    context = type(
        "Context",
        (),
        {
            "device": torch.device("cpu"),
        },
    )()

    result = build_distributed_model(
        model=model,
        context=context,
        strategy="single",
    )

    assert next(result.parameters()).device.type == "cpu"


def test_unknown_strategy_rejected():
    model = create_model()

    context = type(
        "Context",
        (),
        {
            "device": torch.device("cpu"),
        },
    )()

    with pytest.raises(ValueError, match="Unsupported"):
        build_distributed_model(
            model=model,
            context=context,
            strategy="unknown",
        )


def test_config_defaults():
    config = DistributedTrainerConfig()

    assert config.strategy == "ddp"
    assert config.learning_rate == 3e-4
