import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ds_distributed.tensor_parallel import RowParallelLinear

WORLD_SIZE = 2
INPUT_SIZE = 64
OUTPUT_SIZE = 32


def worker(rank: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29504",
        rank=rank,
        world_size=WORLD_SIZE,
    )

    torch.manual_seed(42)

    layer = RowParallelLinear(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
    )

    local_input_size = INPUT_SIZE // WORLD_SIZE

    x = torch.randn(
        2,
        local_input_size,
    )

    output = layer(x)

    print(
        f"rank={rank} "
        f"weight_shape={tuple(layer.weight.shape)} "
        f"input_shape={tuple(x.shape)} "
        f"output_shape={tuple(output.shape)}",
        flush=True,
    )

    assert layer.weight.shape == (
        OUTPUT_SIZE,
        local_input_size,
    )

    assert x.shape == (
        2,
        local_input_size,
    )

    assert output.shape == (
        2,
        OUTPUT_SIZE,
    )

    dist.barrier()

    if rank == 0:
        print(
            f"world_size={WORLD_SIZE} "
            f"global_input_size={INPUT_SIZE} "
            f"local_input_size={local_input_size} "
            f"output_size={OUTPUT_SIZE}",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(
        worker,
        nprocs=WORLD_SIZE,
    )
