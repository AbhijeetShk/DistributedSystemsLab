from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    steps: int
    samples: int
    elapsed_seconds: float
    peak_memory_mb: float
    world_size: int

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.elapsed_seconds
