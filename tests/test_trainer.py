import torch
from torch.utils.data import DataLoader, TensorDataset

from models import GPTModel, TransformerConfig
from trainer import Trainer, load_checkpoint


def create_model():
    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
    )

    return GPTModel(config)


def test_train_step_updates_parameters():
    torch.manual_seed(42)

    model = create_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
    )

    input_ids = torch.randint(0, 100, (4, 16))
    targets = torch.randint(0, 100, (4, 16))

    before = model.blocks[0].attention.qkv.weight.detach().clone()

    metrics = trainer.train_step(input_ids, targets)

    after = model.blocks[0].attention.qkv.weight.detach()

    assert torch.isfinite(torch.tensor(metrics.loss))
    assert not torch.equal(before, after)


def test_train_epoch():
    torch.manual_seed(42)

    model = create_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
    )

    input_ids = torch.randint(0, 100, (8, 16))
    targets = torch.randint(0, 100, (8, 16))

    dataset = TensorDataset(input_ids, targets)
    dataloader = DataLoader(dataset, batch_size=2)

    metrics = trainer.train_epoch(dataloader)

    assert torch.isfinite(torch.tensor(metrics.loss))
    assert metrics.loss > 0


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(42)

    model = create_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
    )

    input_ids = torch.randint(0, 100, (4, 16))
    targets = torch.randint(0, 100, (4, 16))

    metrics = trainer.train_step(
        input_ids,
        targets,
    )

    checkpoint_path = tmp_path / "checkpoint.pt"

    trainer.save_checkpoint(
        path=checkpoint_path,
        epoch=1,
        step=1,
        loss=metrics.loss,
    )

    restored_model = create_model()

    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=3e-4,
    )

    checkpoint = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
    )

    assert checkpoint["epoch"] == 1
    assert checkpoint["step"] == 1
    assert checkpoint["loss"] == metrics.loss

    for original, restored in zip(
        model.parameters(),
        restored_model.parameters(),
        strict=True,
    ):
        assert torch.equal(original, restored)
