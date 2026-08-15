from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from ds_distributed import wrap_model_fsdp
from ds_distributed.runtime import DistributedContext

from .trainer import Trainer


@dataclass(frozen=True)
class DistributedTrainerConfig:
    strategy: str = "ddp"
    learning_rate: float = 3e-4


def build_distributed_model(
    model: nn.Module,
    context: DistributedContext,
    strategy: str,
) -> nn.Module:
    if strategy == "single":
        return model.to(context.device)

    if strategy == "ddp":
        model = model.to(context.device)

        return DistributedDataParallel(
            model,
            device_ids=([context.device.index] if context.device.type == "cuda" else None),
        )

    if strategy == "fsdp":
        return wrap_model_fsdp(
            model,
            context.device,
        )

    raise ValueError(f"Unsupported distributed strategy: {strategy}")


def create_distributed_trainer(
    model: nn.Module,
    context: DistributedContext,
    config: DistributedTrainerConfig,
) -> Trainer:
    model = build_distributed_model(
        model=model,
        context=context,
        strategy=config.strategy,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    return Trainer(
        model=model,
        optimizer=optimizer,
        device=context.device,
    )
