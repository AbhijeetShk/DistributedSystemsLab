import json

import pytest

from benchmark import BenchmarkResult, benchmark_steps


def test_samples_per_second():
    result = BenchmarkResult(
        name="test",
        steps=10,
        samples=100,
        elapsed_seconds=5.0,
        peak_memory_mb=256.0,
        world_size=2,
    )

    assert result.samples_per_second == 20.0


def test_result_to_dict():
    result = BenchmarkResult(
        name="ddp",
        steps=10,
        samples=100,
        elapsed_seconds=5.0,
        peak_memory_mb=256.0,
        world_size=2,
    )

    data = result.to_dict()

    assert data["name"] == "ddp"
    assert data["steps"] == 10
    assert data["samples"] == 100
    assert data["world_size"] == 2
    assert data["samples_per_second"] == 20.0


def test_result_save(tmp_path):
    result = BenchmarkResult(
        name="ddp",
        steps=2,
        samples=16,
        elapsed_seconds=1.0,
        peak_memory_mb=128.0,
        world_size=2,
    )

    path = tmp_path / "result.json"

    result.save(path)

    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["name"] == "ddp"
    assert data["samples_per_second"] == 16.0


def test_benchmark_steps():
    state = {"steps": 0}

    def step() -> int:
        state["steps"] += 1
        return 4

    result = benchmark_steps(
        name="test",
        step_fn=step,
        steps=5,
        world_size=1,
    )

    assert state["steps"] == 5
    assert result.steps == 5
    assert result.samples == 20
    assert result.elapsed_seconds >= 0.0


@pytest.mark.parametrize(
    "steps",
    [0, -1],
)
def test_benchmark_requires_positive_steps(steps):
    with pytest.raises(ValueError):
        benchmark_steps(
            name="test",
            step_fn=lambda: 1,
            steps=steps,
            world_size=1,
        )


def test_benchmark_requires_positive_world_size():
    with pytest.raises(ValueError):
        benchmark_steps(
            name="test",
            step_fn=lambda: 1,
            steps=1,
            world_size=0,
        )
