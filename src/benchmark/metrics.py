import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    steps: int
    samples: int
    elapsed_seconds: float
    peak_memory_mb: float
    world_size: int

    @property
    def samples_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0

        return self.samples / self.elapsed_seconds

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "samples_per_second": self.samples_per_second,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
