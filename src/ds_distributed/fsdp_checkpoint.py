from pathlib import Path

import torch
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel,
    StateDictType,
)


def save_fsdp_checkpoint(
    model: FullyShardedDataParallel,
    path: str | Path,
) -> None:
    if not isinstance(model, FullyShardedDataParallel):
        raise TypeError("save_fsdp_checkpoint expects a FullyShardedDataParallel model")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )

    with FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config,
    ):
        state_dict = model.state_dict()

    if torch.distributed.get_rank() == 0:
        torch.save(
            state_dict,
            path,
        )
