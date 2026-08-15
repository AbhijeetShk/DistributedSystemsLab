from profiling import create_profiler, export_trace


def test_create_profiler():
    profiler = create_profiler(
        "tmp/profile",
    )

    assert profiler is not None


def test_export_trace(tmp_path):
    profiler = create_profiler(tmp_path)

    with profiler:
        pass

    output = export_trace(
        profiler,
        tmp_path,
    )

    assert output.exists()
    assert output.name == "trace.json"
