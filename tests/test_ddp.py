import torch
from torch.utils.data import TensorDataset

from ds_distributed.data import create_distributed_dataloader
from ds_distributed.ddp import wrap_model_ddp
from models import GPTModel, TransformerConfig


def test_ddp_requires_process_group():
    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
    )

    model = GPTModel(config)

    if torch.distributed.is_initialized():
        return

    try:
        wrap_model_ddp(
            model,
            torch.device("cpu"),
        )
    except RuntimeError as exc:
        assert "process group" in str(exc)
    else:
        raise AssertionError("Expected DDP initialization to require a process group")


def test_distributed_sampler_shards_dataset():
    dataset = TensorDataset(
        torch.arange(16),
        torch.arange(16),
    )

    loader_rank0 = create_distributed_dataloader(
        dataset,
        batch_size=2,
        rank=0,
        world_size=2,
        shuffle=False,
    )

    loader_rank1 = create_distributed_dataloader(
        dataset,
        batch_size=2,
        rank=1,
        world_size=2,
        shuffle=False,
    )

    rank0 = list(loader_rank0.sampler)
    rank1 = list(loader_rank1.sampler)

    assert rank0 == [0, 2, 4, 6, 8, 10, 12, 14]
    assert rank1 == [1, 3, 5, 7, 9, 11, 13, 15]
