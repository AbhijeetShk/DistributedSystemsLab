# Distributed Systems Lab

Experiments in distributed training and AI systems.

A small distributed training system built to understand what actually changes
when a Transformer stops fitting on one device.

The goal is not to reimplement PyTorch, DeepSpeed, or Megatron.

The goal is to isolate the important mechanisms, implement the parts worth
understanding, use mature primitives where they already exist, and measure
the consequences.

## What is here

```text
Single GPU
    |
    v
Distributed Runtime
    |
    +-- DDP
    |
    +-- FSDP
    |
    +-- ZeRO / DeepSpeed
    |
    +-- Tensor Parallelism
    |     +-- Column Parallel
    |     +-- Row Parallel
    |
    +-- Pipeline Parallelism
              |
              v
       Profiling + Benchmarks
              |
              v
           Capstone
```

The workload is a small GPT-style decoder-only Transformer.

It is deliberately small. The interesting object here is the execution system,
not the model.

## Design

A few rules drive the project:

- Prefer mechanisms over abstractions.
- Don't implement a framework that already exists.
- Don't add a feature because a distributed-training checklist contains it.
- Keep the training core independent of the execution strategy where possible.
- Make distributed behavior observable.
- Treat correctness and measurement as first-class parts of the implementation.

This means PyTorch primitives are used deliberately:

- `DistributedDataParallel` for DDP
- `FullyShardedDataParallel` for FSDP
- DeepSpeed for ZeRO
- `torch.distributed.pipelining` for pipeline execution
- PyTorch profiler for execution traces

The tensor-parallel layers are implemented directly because the partitioning
and collective communication are the point of the exercise.

## Repository

```text
distributed-systems-ai-lab/
|
+-- src/
|   +-- models/
|   +-- trainer/
|   +-- ds_distributed/
|   +-- benchmark/
|   +-- profiling/
|
+-- experiments/
|   +-- 01_foundations/
|   +-- 02_distributed_init/
|   +-- 03_ddp/
|   +-- 04_fsdp/
|   +-- 05_deepspeed/
|   +-- 06_tensor_parallel/
|   +-- 07_pipeline_parallel/
|   +-- 08_performance/
|   +-- 09_final_benchmarks/
|   +-- 10_capstone/
|
+-- tests/
+-- docs/
+-- README.md
```

## Experiments

| Stage | Question |
|---|---|
| Foundations | What does the training system actually need? |
| Distributed init | How do independent processes become one job? |
| DDP | What does data parallelism cost? |
| FSDP | What changes when model state is sharded? |
| ZeRO | Where can optimizer and model state be partitioned? |
| Tensor Parallel | What happens when one layer spans devices? |
| Pipeline Parallel | What happens when the model itself is split into stages? |
| Performance | Where does time go? |
| Final Benchmarks | What do the strategies actually buy? |
| Capstone | Can the pieces compose into one training system? |

## Validation

The repository has unit tests for the core components and multi-process
experiments for distributed behavior.

```bash
make check
pytest -v
```

Distributed experiments are intentionally separate from the normal test suite.
This keeps ordinary development fast while making process-level behavior
explicit.

## Performance

The final comparison is intended to answer:

```text
How much memory does each strategy save?
What communication does it introduce?
What happens to step time?
What happens to throughput?
Where does scaling stop paying for itself?
```

The benchmark harness records:

- step time
- samples / second
- token throughput
- peak CUDA memory
- world size

The profiler provides the execution-level view.

GPU results are not fabricated into the repository. They will be generated on
a CUDA/NCCL environment and recorded with the workload and configuration used.

## Documentation

- [Architecture](docs/architecture.md)
- [Distributed Strategies](docs/distributed-strategies.md)
- [Benchmarks](docs/benchmarks.md)
- [Changelog](docs/changelog.md)
- [Decisions](docs/decisions.md)
- [Roadmap](docs/roadmap.md)

## Why this exists

Large-model systems are often described in terms of their APIs.

This lab is about the layer underneath:

```text
parameters
    |
    v
memory
    |
    v
communication
    |
    v
synchronization
    |
    v
compute
    |
    v
throughput
```

If changing the parallelism strategy changes those quantities, the system
should make that visible.

That is the lab.