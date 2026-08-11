import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        mask = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        return self.proj(output)
