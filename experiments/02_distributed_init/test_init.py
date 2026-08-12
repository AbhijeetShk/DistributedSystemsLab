import torch.distributed as dist

from ds_distributed import cleanup_distributed, initialize_distributed


def main() -> None:
    context = initialize_distributed()

    print(
        f"rank={context.rank} "
        f"local_rank={context.local_rank} "
        f"world_size={context.world_size} "
        f"device={context.device}"
    )

    dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
