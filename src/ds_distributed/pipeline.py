import torch
from torch import nn
from torch.distributed.pipelining import (
    PipelineStage,
    ScheduleGPipe,
    SplitPoint,
    pipeline,
)


def build_pipeline_stage(
    model: nn.Module,
    example_input: tuple[torch.Tensor, ...],
    stage_index: int,
    num_stages: int,
    device: torch.device,
):
    if not torch.distributed.is_initialized():
        raise RuntimeError("Distributed process group must be initialized.")

    if torch.distributed.get_world_size() != num_stages:
        raise ValueError("Pipeline stage count must match distributed world size.")

    pipe = pipeline(
        module=model,
        mb_args=example_input,
        split_spec={
            "blocks.1": SplitPoint.BEGINNING,
        },
    )

    return pipe.build_stage(
        stage_index,
        device,
    )


def create_gpipe_schedule(
    stage: PipelineStage,
    num_microbatches: int,
    loss_fn,
):
    return ScheduleGPipe(
        stage,
        n_microbatches=num_microbatches,
        loss_fn=loss_fn,
    )
