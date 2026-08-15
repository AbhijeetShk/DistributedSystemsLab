import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ds_distributed import cleanup_distributed, initialize_distributed
from models import GPTModel, TransformerConfig
from profiling import create_profiler, export_trace

PROFILE_DIR = "artifacts/profiles/ddp"


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

    model = GPTModel(config).to(context.device)

    model = DistributedDataParallel(
        model,
        device_ids=([context.device.index] if context.device.type == "cuda" else None),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    input_ids = torch.randint(
        0,
        100,
        (8, 32),
        device=context.device,
    )

    targets = torch.randint(
        0,
        100,
        (8, 32),
        device=context.device,
    )

    profiler = create_profiler(
        PROFILE_DIR,
    )

    with profiler:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)

            _, loss = model(
                input_ids,
                targets,
            )

            loss.backward()
            optimizer.step()

            profiler.step()

    if context.is_main_process:
        trace_path = export_trace(
            profiler,
            PROFILE_DIR,
        )

        print(
            f"profile_trace={trace_path}",
            flush=True,
        )

        print(
            profiler.key_averages().table(
                sort_by="cpu_time_total",
                row_limit=15,
            ),
            flush=True,
        )

    dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
