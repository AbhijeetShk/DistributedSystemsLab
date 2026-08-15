import torch
from torch.utils.data import TensorDataset

from ds_distributed import (
    cleanup_distributed,
    initialize_distributed,
)
from models import GPTModel, TransformerConfig
from trainer import (
    DistributedTrainerConfig,
    create_distributed_trainer,
)

BATCH_SIZE = 8
EPOCHS = 3


def main() -> None:
    context = initialize_distributed()

    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
    )

    model = GPTModel(config)

    trainer_config = DistributedTrainerConfig(
        strategy="ddp",
        learning_rate=3e-4,
    )

    trainer = create_distributed_trainer(
        model=model,
        context=context,
        config=trainer_config,
    )

    input_ids = torch.randint(
        0,
        100,
        (128, 32),
    )

    targets = torch.randint(
        0,
        100,
        (128, 32),
    )

    dataset = TensorDataset(
        input_ids,
        targets,
    )

    from ds_distributed.data import (
        create_distributed_dataloader,
    )

    dataloader = create_distributed_dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        rank=context.rank,
        world_size=context.world_size,
    )

    for epoch in range(EPOCHS):
        metrics = trainer.train_epoch(
            dataloader,
            epoch=epoch,
        )

        if context.is_main_process:
            print(
                f"epoch={epoch + 1} loss={metrics.loss:.4f} lr={metrics.learning_rate:.2e}",
                flush=True,
            )

    cleanup_distributed()


if __name__ == "__main__":
    main()
