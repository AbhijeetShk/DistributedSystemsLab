## Current Status

### v0.1 — Minimal GPT Model

Implemented a minimal GPT-style decoder-only Transformer using PyTorch's
scaled dot-product attention primitive.

Validated:

- Forward pass produces expected logits and loss shapes.
- Backward pass produces gradients through attention, MLP, and embeddings.
- Input/output embedding weights are tied.
- CPU execution works.
- A short optimization smoke test successfully reduces loss over 5 steps.

Example smoke test:

```text
step=0 loss=84.8695
step=1 loss=82.0960
step=2 loss=79.2812
step=3 loss=76.4263
step=4 loss=73.5322

The decreasing loss confirms that the model participates correctly in the
training loop; this is a functional smoke test, not a model-quality benchmark.