import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ds_distributed.tensor_parallel import ColumnParallelLinear

WORLD_SIZE = 2
INPUT_SIZE = 32
OUTPUT_SIZE = 64


def worker(rank: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29503",
        rank=rank,
        world_size=WORLD_SIZE,
    )

    torch.manual_seed(42)

    layer = ColumnParallelLinear(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
    )

    x = torch.randn(2, INPUT_SIZE)

    output = layer(x)

    print(
        f"rank={rank} weight_shape={tuple(layer.weight.shape)} output_shape={tuple(output.shape)}",
        flush=True,
    )

    expected_local_output = OUTPUT_SIZE // WORLD_SIZE

    assert layer.weight.shape == (
        expected_local_output,
        INPUT_SIZE,
    )

    assert output.shape == (
        2,
        expected_local_output,
    )

    dist.barrier()

    if rank == 0:
        print(
            f"world_size={WORLD_SIZE} "
            f"global_output_size={OUTPUT_SIZE} "
            f"local_output_size={expected_local_output}",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(
        worker,
        nprocs=WORLD_SIZE,
    )
