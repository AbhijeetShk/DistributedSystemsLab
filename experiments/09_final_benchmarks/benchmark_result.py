from benchmark import benchmark_steps
from benchmark.config import get_config


def main() -> None:
    config = get_config("ddp")

    def step() -> int:
        total = sum(range(100_000))

        assert total > 0

        return config.batch_size

    result = benchmark_steps(
        name=config.name,
        step_fn=step,
        steps=config.benchmark_steps,
        world_size=1,
    )

    print(
        f"name={result.name} "
        f"steps={result.steps} "
        f"samples={result.samples} "
        f"elapsed={result.elapsed_seconds:.4f}s "
        f"samples/sec={result.samples_per_second:.2f} "
        f"peak_memory={result.peak_memory_mb:.2f}MB "
        f"world_size={result.world_size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
