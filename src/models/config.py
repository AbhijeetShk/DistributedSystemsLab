from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 32_000
    max_seq_len: int = 1024
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
