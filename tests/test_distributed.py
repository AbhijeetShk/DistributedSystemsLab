import torch
import torch.distributed as dist

from ds_distributed import (
    DistributedContext,
    get_distributed_context,
)


def test_default_distributed_context():
    if dist.is_initialized():
        return

    context = get_distributed_context()

    assert isinstance(context, DistributedContext)
    assert context.rank == 0
    assert context.world_size == 1
    assert context.local_rank == 0
    assert context.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert context.is_main_process
