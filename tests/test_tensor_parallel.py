import torch

from ds_distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)


def test_column_parallel_requires_process_group():
    if torch.distributed.is_initialized():
        return

    try:
        ColumnParallelLinear(
            input_size=16,
            output_size=32,
        )
    except RuntimeError as exc:
        assert "process group" in str(exc)
    else:
        raise AssertionError("Expected tensor parallel layer to require a process group")


def test_column_parallel_requires_divisible_output_size():
    if not torch.distributed.is_initialized():
        return

    world_size = torch.distributed.get_world_size()

    invalid_output_size = world_size * 4 + 1

    try:
        ColumnParallelLinear(
            input_size=16,
            output_size=invalid_output_size,
        )
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected output_size divisibility validation to fail")


def test_row_parallel_requires_process_group():
    if torch.distributed.is_initialized():
        return

    try:
        RowParallelLinear(
            input_size=32,
            output_size=16,
        )
    except RuntimeError as exc:
        assert "Distributed process group" in str(exc)
    else:
        raise AssertionError("Expected tensor parallel layer to require a process group")


def test_row_parallel_requires_divisible_input_size():
    if not torch.distributed.is_initialized():
        return

    world_size = torch.distributed.get_world_size()

    invalid_input_size = world_size * 4 + 1

    try:
        RowParallelLinear(
            input_size=invalid_input_size,
            output_size=16,
        )
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected input_size divisibility validation to fail")
