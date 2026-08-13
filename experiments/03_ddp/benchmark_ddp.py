import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import TensorDataset

from benchmark import BenchmarkResult
from ds_distributed import cleanup_distributed, initialize_distributed
from ds_distributed.data import create_distributed_dataloader
from models import GPTModel, TransformerConfig


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
        device_ids=[context.device.index] if context.device.type == "cuda" else None,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
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

    dataloader = create_distributed_dataloader(
        dataset=dataset,
        batch_size=8,
        rank=context.rank,
        world_size=context.world_size,
    )

    warmup_steps = 3
    benchmark_steps = 10

    model.train()

    for step, (input_ids, targets) in enumerate(dataloader):
        if step >= warmup_steps:
            break

        optimizer.zero_grad(set_to_none=True)

        input_ids = input_ids.to(context.device)
        targets = targets.to(context.device)

        _, loss = model(input_ids, targets)

        loss.backward()
        optimizer.step()

    if context.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    dist.barrier()

    start = time.perf_counter()

    samples = 0

    for step, (input_ids, targets) in enumerate(dataloader):
        if step >= benchmark_steps:
            break

        optimizer.zero_grad(set_to_none=True)

        input_ids = input_ids.to(context.device)
        targets = targets.to(context.device)

        _, loss = model(input_ids, targets)

        loss.backward()
        optimizer.step()

        samples += input_ids.size(0)

    if context.device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    peak_memory_mb = 0.0

    if context.device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    result = BenchmarkResult(
        steps=benchmark_steps,
        samples=samples,
        elapsed_seconds=elapsed,
        peak_memory_mb=peak_memory_mb,
        world_size=context.world_size,
    )

    if context.is_main_process:
        print(
            f"steps={result.steps} "
            f"samples={result.samples} "
            f"time={result.elapsed_seconds:.4f}s "
            f"samples/sec={result.samples_per_second:.2f} "
            f"peak_memory={result.peak_memory_mb:.2f}MB "
            f"world_size={result.world_size}",
            flush=True,
        )

    cleanup_distributed()


if __name__ == "__main__":
    main()
