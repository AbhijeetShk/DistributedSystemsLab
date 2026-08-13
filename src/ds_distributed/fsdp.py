import torch
from torch import nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    ShardingStrategy,
)


def wrap_model_fsdp(
    model: nn.Module,
    device: torch.device,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
) -> FullyShardedDataParallel:
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed process group must be initialized before wrapping the model with FSDP."
        )

    model = model.to(device)

    return FullyShardedDataParallel(
        model,
        sharding_strategy=sharding_strategy,
        device_id=device if device.type == "cuda" else None,
    )
