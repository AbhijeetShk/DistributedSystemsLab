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


class RowParallelLinear(nn.Module):
    """Linear layer whose input features are partitioned across ranks."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ) -> None:
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError("Distributed process group must be initialized.")

        world_size = dist.get_world_size()

        if input_size % world_size != 0:
            raise ValueError("input_size must be divisible by world_size.")

        self.world_size = world_size
        self.input_size = input_size
        self.output_size = output_size
        self.input_size_per_partition = input_size // world_size

        self.weight = nn.Parameter(
            torch.empty(
                output_size,
                self.input_size_per_partition,
            )
        )

        self.bias = nn.Parameter(torch.zeros(output_size))

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.input_size_per_partition:
            raise ValueError(
                "Input feature dimension must match the local tensor-parallel partition."
            )

        output = torch.matmul(
            x,
            self.weight.t(),
        )

        dist.all_reduce(
            output,
            op=dist.ReduceOp.SUM,
        )

        return output + self.bias
