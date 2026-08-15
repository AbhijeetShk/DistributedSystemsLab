import torch
import torch.distributed as dist
from torch import nn


class ColumnParallelLinear(nn.Module):
    """Linear layer whose output features are partitioned across ranks."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ) -> None:
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError("Distributed process group must be initialized.")

        world_size = dist.get_world_size()

        if output_size % world_size != 0:
            raise ValueError("output_size must be divisible by world_size.")

        self.world_size = world_size
        self.input_size = input_size
        self.output_size = output_size
        self.output_size_per_partition = output_size // world_size

        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                input_size,
            )
        )

        self.bias = nn.Parameter(
            torch.zeros(
                self.output_size_per_partition,
            )
        )

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.matmul(
                x,
                self.weight.t(),
            )
            + self.bias
        )
