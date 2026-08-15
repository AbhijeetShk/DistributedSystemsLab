import time
from collections.abc import Callable

import torch

from .metrics import BenchmarkResult


def benchmark_steps(
    name: str,
    step_fn: Callable[[], int],
    *,
    steps: int,
    world_size: int,
) -> BenchmarkResult:
    """Benchmark a fixed number of training steps.

    step_fn must execute one complete training step and return
    the number of samples processed by that step.
    """

    if steps <= 0:
        raise ValueError("steps must be positive.")

    if world_size <= 0:
        raise ValueError("world_size must be positive.")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    samples = 0

    for _ in range(steps):
        samples += step_fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_memory_mb = 0.0

    elapsed_seconds = time.perf_counter() - start

    return BenchmarkResult(
        name=name,
        steps=steps,
        samples=samples,
        elapsed_seconds=elapsed_seconds,
        peak_memory_mb=peak_memory_mb,
        world_size=world_size,
    )
