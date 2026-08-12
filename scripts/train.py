import torch
from torch.utils.data import DataLoader, TensorDataset

from models import GPTModel, TransformerConfig
from trainer import Trainer


def main() -> None:
    torch.manual_seed(42)

    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
    )

    model = GPTModel(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    input_ids = torch.randint(0, 100, (64, 32))
    targets = torch.randint(0, 100, (64, 32))

    dataset = TensorDataset(input_ids, targets)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
    )

    for epoch in range(3):
        metrics = trainer.train_epoch(dataloader)

        print(f"epoch={epoch + 1} loss={metrics.loss:.4f} lr={metrics.learning_rate:.2e}")


if __name__ == "__main__":
    main()
