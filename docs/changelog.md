# Changelog

## Unreleased

### Model

- Added minimal GPT-style decoder-only Transformer.
- Added causal self-attention using PyTorch SDPA.
- Added weight-tied token embedding and language-model head.
- Added configuration validation.

### Validation

- Forward-pass shape and finite-loss tests.
- Backward/gradient-flow test.
- Weight-tying test.
- CPU execution smoke test.
- Five-step optimization smoke test successfully reduced loss from
  84.87 to 73.53 on a fixed synthetic batch.

### Next

- Implement single-GPU training engine.
- Add checkpointing and training metrics.