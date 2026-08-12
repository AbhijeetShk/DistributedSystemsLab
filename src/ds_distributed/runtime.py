import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def initialize_distributed(
    backend: str | None = None,
) -> DistributedContext:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    if dist.is_initialized():
        return get_distributed_context()

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def get_distributed_context() -> DistributedContext:
    if not dist.is_initialized():
        return DistributedContext(
            rank=0,
            world_size=1,
            local_rank=0,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(__import__("os").environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
