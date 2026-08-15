from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


def create_profiler(
    output_dir: str | Path,
    *,
    record_shapes: bool = True,
    profile_memory: bool = True,
) -> profile:
    """Create a PyTorch profiler for CPU and CUDA workloads."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]

    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    return profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=False,
    )


def export_trace(
    profiler: profile,
    output_dir: str | Path,
    filename: str = "trace.json",
) -> Path:
    """Export a profiler trace that can be inspected later."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename

    profiler.export_chrome_trace(str(path))

    return path
