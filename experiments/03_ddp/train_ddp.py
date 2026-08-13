import torch
from torch.utils.data import TensorDataset

from ds_distributed import (
    cleanup_distributed,
    initialize_distributed,
)
from ds_distributed.data import create_distributed_dataloader
from ds_distributed.ddp import wrap_model_ddp
from models import GPTModel, TransformerConfig
from trainer import Trainer


def main() -> None:
    context = initialize_distributed()

    torch.manual_seed(42)

    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
    )

    model = GPTModel(config)

    model = wrap_model_ddp(
        model,
        context.device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=context.device,
    )

    input_ids = torch.randint(
        0,
        100,
        (64, 32),
    )

    targets = torch.randint(
        0,
        100,
        (64, 32),
    )

    dataset = TensorDataset(
        input_ids,
        targets,
    )

    dataloader = create_distributed_dataloader(
        dataset=dataset,
        batch_size=8,
        rank=context.rank,
        world_size=context.world_size,
    )

    for epoch in range(3):
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
