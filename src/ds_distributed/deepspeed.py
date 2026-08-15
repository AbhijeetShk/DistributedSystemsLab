from pathlib import Path

import deepspeed
from torch import nn


def initialize_deepspeed(
    model: nn.Module,
    model_parameters,
    config_path: str | Path,
):
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        config=str(config_path),
    )

    return engine, optimizer, scheduler
