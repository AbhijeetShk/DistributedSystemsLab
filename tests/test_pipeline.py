import torch

from ds_distributed.pipeline import build_pipeline_stage


def test_pipeline_requires_process_group():
    if torch.distributed.is_initialized():
        return

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.Linear(16, 8),
    )

    example_input = (torch.randn(2, 8),)

    try:
        build_pipeline_stage(
            model=model,
            example_input=example_input,
            stage_index=0,
            num_stages=2,
            device=torch.device("cpu"),
        )
    except RuntimeError as exc:
        assert "process group" in str(exc)
    else:
        raise AssertionError("Expected pipeline construction to require a process group")
