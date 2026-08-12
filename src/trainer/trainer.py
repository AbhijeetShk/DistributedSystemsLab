from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from .checkpoint import save_checkpoint


@dataclass
class TrainMetrics:
    loss: float
    learning_rate: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        device: str | torch.device = "cpu",
        grad_clip_norm: float | None = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm

    def train_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> TrainMetrics:
        self.model.train()

        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        _, loss = self.model(input_ids, targets)

        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip_norm,
            )

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return TrainMetrics(
            loss=loss.item(),
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )

    def train_epoch(
        self,
        dataloader,
    ) -> TrainMetrics:
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc="Training")

        for input_ids, targets in progress:
            metrics = self.train_step(input_ids, targets)

            total_loss += metrics.loss
            num_batches += 1

            progress.set_postfix(
                loss=f"{metrics.loss:.4f}",
                lr=f"{metrics.learning_rate:.2e}",
            )

        if num_batches == 0:
            raise ValueError("Training dataloader is empty")

        return TrainMetrics(
            loss=total_loss / num_batches,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        step: int,
        loss: float,
    ) -> None:
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            step=step,
            loss=loss,
        )
