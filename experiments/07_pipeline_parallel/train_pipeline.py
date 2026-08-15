import torch
import torch.distributed as dist

from ds_distributed import (
    build_pipeline_stage,
    cleanup_distributed,
    create_gpipe_schedule,
    initialize_distributed,
)
from models import GPTModel, TransformerConfig

NUM_MICROBATCHES = 4


def loss_fn(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        outputs.reshape(-1, outputs.size(-1)),
        targets.reshape(-1),
    )


def main() -> None:
    context = initialize_distributed()

    if context.world_size != 2:
        raise RuntimeError("This experiment requires exactly 2 pipeline stages.")

    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
    )

    model = GPTModel(config)

    example_input = (
        torch.randint(
            0,
            config.vocab_size,
            (2, config.max_seq_len),
        ),
    )

    stage = build_pipeline_stage(
        model=model,
        example_input=example_input,
        stage_index=context.rank,
        num_stages=context.world_size,
        device=context.device,
    )

    schedule = create_gpipe_schedule(
        stage=stage,
        num_microbatches=NUM_MICROBATCHES,
        loss_fn=loss_fn,
    )

    if context.rank == 0:
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (8, config.max_seq_len),
            device=context.device,
        )

        targets = torch.randint(
            0,
            config.vocab_size,
            (8, config.max_seq_len),
            device=context.device,
        )

        schedule.step(
            input_ids,
            target=targets,
        )

        print(
            "pipeline_training_completed=True",
            flush=True,
        )

    else:
        schedule.step()

    dist.barrier()

    if context.is_main_process:
        print(
            f"world_size={context.world_size} num_microbatches={NUM_MICROBATCHES}",
            flush=True,
        )

    cleanup_distributed()


if __name__ == "__main__":
    main()
