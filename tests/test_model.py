import torch

from models import GPTModel, TransformerConfig


def test_model_forward():
    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
    )

    model = GPTModel(config)

    input_ids = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))

    logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 16, 100)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_invalid_attention_configuration():
    try:
        TransformerConfig(hidden_size=127, num_heads=8)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid configuration to fail")


def test_model_backward():
    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
    )

    model = GPTModel(config)

    input_ids = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))

    _, loss = model(input_ids, targets)
    loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert model.blocks[0].attention.qkv.weight.grad is not None
    assert model.blocks[0].mlp[0].weight.grad is not None


def test_weight_tying():
    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
    )

    model = GPTModel(config)

    assert model.lm_head.weight is model.token_embedding.weight
