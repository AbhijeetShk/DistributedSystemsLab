from benchmark import BenchmarkResult


def test_samples_per_second():
    result = BenchmarkResult(
        steps=10,
        samples=100,
        elapsed_seconds=5.0,
        peak_memory_mb=256.0,
        world_size=2,
    )

    assert result.samples_per_second == 20.0
