import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from ds_distributed.tensor_parallel import TensorParallelMLP

WORLD_SIZE = 2
HIDDEN_SIZE = 32
INTERMEDIATE_SIZE = 64
BATCH_SIZE = 4


def worker(rank: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29505",
        rank=rank,
        world_size=WORLD_SIZE,
    )

    torch.manual_seed(42)

    reference_up = nn.Linear(
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
    )

    reference_down = nn.Linear(
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
    )

    reference_activation = nn.GELU()

    tp_mlp = TensorParallelMLP(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    )

    with torch.no_grad():
        intermediate_per_rank = INTERMEDIATE_SIZE // WORLD_SIZE

        start = rank * intermediate_per_rank
        end = start + intermediate_per_rank

        tp_mlp.up_projection.weight.copy_(reference_up.weight[start:end])

        tp_mlp.up_projection.bias.copy_(reference_up.bias[start:end])

        tp_mlp.down_projection.weight.copy_(reference_down.weight[:, start:end])

        tp_mlp.down_projection.bias.copy_(reference_down.bias)

    x = torch.randn(
        BATCH_SIZE,
        HIDDEN_SIZE,
    )

    reference_output = reference_down(reference_activation(reference_up(x)))

    tp_output = tp_mlp(x)

    max_error = (reference_output - tp_output).abs().max().item()

    print(
        f"rank={rank} tp_output_shape={tuple(tp_output.shape)} max_error={max_error:.8f}",
        flush=True,
    )

    assert torch.allclose(
        reference_output,
        tp_output,
        atol=1e-6,
        rtol=1e-5,
    )

    dist.barrier()

    if rank == 0:
        print(
            "tensor_parallel_mlp_matches_reference=True",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(
        worker,
        nprocs=WORLD_SIZE,
    )
