import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

from models import GPTModel, TransformerConfig


def worker(rank: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29502",
        rank=rank,
        world_size=2,
    )

    torch.manual_seed(42)

    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=16,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
    )

    model = GPTModel(config)
    model = DistributedDataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    input_ids = torch.randint(0, 100, (16, 16))
    targets = torch.randint(0, 100, (16, 16))

    dataset = TensorDataset(input_ids, targets)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=2,
        rank=rank,
        shuffle=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
    )

    for input_ids, targets in dataloader:
        optimizer.zero_grad(set_to_none=True)

        _, loss = model(input_ids, targets)

        loss.backward()
        optimizer.step()

    reference = next(model.parameters()).detach().clone()

    dist.broadcast(reference, src=0)

    synchronized = torch.equal(
        next(model.parameters()).detach(),
        reference,
    )

    result = torch.tensor(
        int(synchronized),
        dtype=torch.int,
    )

    dist.all_reduce(result, op=dist.ReduceOp.MIN)

    if rank == 0:
        print(f"parameters_synchronized={bool(result.item())}")

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=2)
