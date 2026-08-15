import torch

from ds_distributed.tensor_parallel import ColumnParallelLinear


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
