from benchmark import benchmark_steps

STEP_SAMPLES = 8
STEPS = 10


def main() -> None:
    state = {"step": 0}

    def step() -> int:
        state["step"] += 1

        total = 0

        for value in range(100_000):
            total += value

        assert total > 0

        return STEP_SAMPLES

    result = benchmark_steps(
        name="synthetic",
        step_fn=step,
        steps=STEPS,
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
