from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def create_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool = True,
) -> DataLoader:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
    )
