import torch
from torch.utils.data import TensorDataset

from ds_distributed import initialize_deepspeed
from models import GPTModel, TransformerConfig


def main() -> None:
    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
    )

    model = GPTModel(config)

    optimizer_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    engine, _, _ = initialize_deepspeed(
        model=model,
        model_parameters=optimizer_parameters,
        config_path="configs/deepspeed/zero3.json",
    )

    input_ids = torch.randint(
        0,
        100,
        (128, 32),
        device=engine.device,
    )

    targets = torch.randint(
        0,
        100,
        (128, 32),
        device=engine.device,
    )

    dataset = TensorDataset(
        input_ids,
        targets,
    )

    for input_ids, targets in dataset:
        input_ids = input_ids.unsqueeze(0)
        targets = targets.unsqueeze(0)

        logits, loss = engine(
            input_ids,
            targets,
        )

        engine.backward(loss)
        engine.step()

    if engine.global_rank == 0:
        print("DeepSpeed ZeRO-3 training completed.")


if __name__ == "__main__":
    main()
