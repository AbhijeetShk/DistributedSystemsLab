# Changelog

## Unreleased

### v0.1 — Single GPU

Built the initial training system around a minimal GPT-style decoder-only
Transformer. Added causal self-attention using PyTorch SDPA, weight tying
between the token embedding and language-model head, configuration validation,
the training loop, metrics, and checkpoint save/load.

The model was validated through forward, backward, weight-tying, and CPU smoke
tests. A fixed five-step optimization run reduced loss from 84.87 to 73.53.
This is only a functional check that gradients and optimization behave
correctly; it is not a model-quality benchmark.

### v0.2 — Distributed Runtime

Added the distributed runtime responsible for process-group initialization,
rank and world-size discovery, local-rank handling, device selection, and
cleanup. The runtime supports the backend appropriate to the environment,
with Gloo used for local CPU development and NCCL intended for CUDA execution.

Multi-process Gloo experiments verified that independent processes can form a
single distributed job and communicate through collectives.

### v0.3 — DDP

Added DistributedDataParallel integration and distributed data loading. The
DDP experiment validates model replication, process-local batches, gradient
synchronization, and multi-process execution.

A small benchmark and profiling experiment were added so DDP can be measured
rather than treated only as an API integration.

### v0.4 — FSDP

Added Fully Sharded Data Parallel integration and distributed checkpoint
handling. Tests cover the wrapper and checkpoint path while the dedicated
experiment provides the multi-process execution boundary.

The implementation intentionally uses PyTorch FSDP rather than reproducing its
internal sharding machinery.

### v0.5 — ZeRO

Added DeepSpeed integration for ZeRO-style optimizer and model-state
partitioning. The lab treats DeepSpeed as an external systems boundary rather
than attempting to reproduce the ZeRO implementation.

The purpose of this stage is to understand where state partitioning enters the
training system and how it differs from ordinary data parallelism.

### v0.6 — Tensor Parallelism

Added column-parallel and row-parallel linear layers using explicit tensor
partitioning and collective communication. The two primitives were composed
into a tensor-parallel MLP and checked against the corresponding non-parallel
execution.

The implementation is intentionally small enough for the partitioning and
communication paths to remain visible.

### v0.7 — Pipeline Parallelism

Added pipeline-parallel execution using PyTorch's distributed pipelining API
and GPipe scheduling. The experiment focuses on stage partitioning,
microbatches, and pipeline execution rather than implementing another
scheduler.

### v0.8 — Performance

Added distributed profiling and benchmark infrastructure. Profiling exports
execution traces for inspecting compute, communication, synchronization, and
idle time. The benchmark layer provides a common representation for elapsed
time, throughput, samples processed, peak memory, and world size.

### v0.9 — Final Benchmarks

Added a common benchmark configuration for comparing the distributed strategies
under the same workload definition. The final benchmark pass is intended to
run on CUDA/NCCL hardware so that throughput and memory measurements reflect
the environment these systems are designed for.

Local macOS/Gloo results are kept as correctness evidence rather than presented
as GPU performance numbers.

### v1.0 — Capstone

Added the unified distributed training entry point and composed the parts of
the system that naturally share the same training abstraction. Tensor and
pipeline parallelism remain explicit execution strategies where hiding their
differences would make the system harder to understand.

The implementation phase is complete. The remaining work is experimental:
run the final benchmark suite on CUDA/NCCL hardware and record what the
different strategies actually change.