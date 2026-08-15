from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    batch_size: int
    sequence_length: int
    warmup_steps: int
    benchmark_steps: int


CONFIGS = {
    "ddp": BenchmarkConfig(
        name="ddp",
        batch_size=8,
        sequence_length=32,
        warmup_steps=5,
        benchmark_steps=20,
    ),
    "fsdp": BenchmarkConfig(
        name="fsdp",
        batch_size=8,
        sequence_length=32,
        warmup_steps=5,
        benchmark_steps=20,
    ),
    "zero3": BenchmarkConfig(
        name="zero3",
        batch_size=8,
        sequence_length=32,
        warmup_steps=5,
        benchmark_steps=20,
    ),
    "tensor_parallel": BenchmarkConfig(
        name="tensor_parallel",
        batch_size=8,
        sequence_length=32,
        warmup_steps=5,
        benchmark_steps=20,
    ),
    "pipeline_parallel": BenchmarkConfig(
        name="pipeline_parallel",
        batch_size=8,
        sequence_length=32,
        warmup_steps=5,
        benchmark_steps=20,
    ),
}


def get_config(name: str) -> BenchmarkConfig:
    try:
        return CONFIGS[name]
    except KeyError as exc:
        available = ", ".join(CONFIGS)
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}") from exc
