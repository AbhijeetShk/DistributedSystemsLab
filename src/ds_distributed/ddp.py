import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def wrap_model_ddp(
    model: nn.Module,
    device: torch.device,
) -> DistributedDataParallel:
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed process group must be initialized before wrapping the model with DDP."
        )

    model = model.to(device)

    if device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
        )

    return DistributedDataParallel(model)
