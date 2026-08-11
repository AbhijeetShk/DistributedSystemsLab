import torch
from torch import nn

from .attention import CausalSelfAttention
from .config import TransformerConfig


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)

        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.hidden_size,
        )

        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))

        self.norm = nn.LayerNorm(config.hidden_size)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, targets=None):
        _, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.config.max_seq_len}")

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        )

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.norm(x))

        if targets is None:
            return logits

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

        return logits, loss
