# Architecture

The lab evolves from a single-process training engine into a distributed
training system.

The architecture is intentionally layered. Distributed execution should change
how work is executed without requiring the core training logic to be rewritten
for every parallelism strategy.

## v0.1 — Single GPU

```text
                    ┌──────────────────┐
                    │      Config      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Trainer      │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
          DataLoader       Model        Optimizer
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                       Training Step
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
               Checkpoint          Metrics
```

The first version deliberately has no distributed concepts.

The trainer owns the training step, while the model, data, optimizer,
checkpointing, and metrics remain separate concerns.

## v0.2 — Distributed Runtime

The first distributed layer is process management rather than parallelism.

```text
                     ┌────────────────────┐
                     │ DistributedContext │
                     └─────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
            rank           world_size       local_rank
                               │
                               ▼
                        Process Group
                               │
                         ┌─────┴─────┐
                         ▼           ▼
                       Gloo        NCCL
```

The runtime exposes the state required by the rest of the system:

- rank
- world size
- local rank
- device

The training code should not need to know how the process group was created.

## v0.3 — DDP

DDP is the first real distributed training baseline.

```text
                         Dataset
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
             Process 0             Process 1
                 │                     │
             Model copy             Model copy
                 │                     │
          Forward / backward    Forward / backward
                 │                     │
                 └──────────┬──────────┘
                            ▼
                         AllReduce
                            │
                            ▼
                     synchronized
                       gradients
```

Each process owns a model replica and a different portion of the data.

The important system boundary is the gradient synchronization step.

DDP answers the baseline question:

> What does it cost to replicate the model and distribute only the data?

## v0.4 — FSDP

DDP keeps a complete copy of the model state on every worker.

FSDP changes that assumption.

```text
                         Model State
                              │
                  ┌───────────┼───────────┐
                  ▼           ▼           ▼
                Rank 0      Rank 1      Rank N
                shard       shard       shard
```

Parameters, gradients, and optimizer state can be sharded across workers.

Parameters are gathered when required for computation and resharded afterward.

The resulting tradeoff is the important part:

```text
less memory
    ↕
more communication
```

Checkpointing also becomes a distributed systems concern rather than simply
writing a model file.

## v0.5 — ZeRO

ZeRO extends state partitioning through DeepSpeed.

The lab uses DeepSpeed rather than reproducing its internals.

```text
                     Training Loop
                          │
                          ▼
                    DeepSpeed Engine
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
         optimizer      gradients   parameters
           state         state        state
             │            │            │
             └────────────┴────────────┘
                     partitioned
```

The useful question here is not how many lines are required to reproduce
DeepSpeed.

It is where the framework boundary sits and what state partitioning changes.

## v0.6 — Tensor Parallelism

Tensor parallelism changes the unit of parallelism again.

The model is no longer simply replicated or sharded between training steps.
Individual operations are partitioned across devices.

### Column Parallel

```text
                         W
                    ┌────┴────┐
                    ▼         ▼
                   W0         W1
                 Rank 0     Rank 1
                    │         │
                   XW0       XW1
```

The output dimension is partitioned across workers.

### Row Parallel

```text
                         X
                    ┌────┴────┐
                    ▼         ▼
                   X0         X1
                    │         │
                   X0W0      X1W1
                    │         │
                    └────┬────┘
                         ▼
                      AllReduce
                         │
                         ▼
                         Y
```

The input dimension is partitioned and the partial results are reduced.

Together they form a common Transformer pattern:

```text
Column Parallel
       │
      GELU
       │
Row Parallel
       │
   AllReduce
```

The interesting part is not the linear layer.

It is where the communication appears when a single operation is split across
devices.

## v0.7 — Pipeline Parallelism

Pipeline parallelism partitions the model by depth rather than by individual
operations.

```text
GPU 0                         GPU 1

┌─────────────┐              ┌─────────────┐
│   Stage 0   │─────────────▶│   Stage 1   │
└─────────────┘              └─────────────┘
      │                            │
 microbatch 0                  microbatch 0
 microbatch 1                  microbatch 1
 microbatch 2                  microbatch 2
```

Multiple microbatches allow different stages to make progress concurrently.

This introduces another systems cost:

```text
pipeline bubbles
```

The lab uses PyTorch's pipeline API and GPipe scheduling rather than
implementing another scheduler.

## v0.8 — Performance

Once the mechanisms exist, implementation is no longer the interesting
question.

The system is instrumented to expose:

```text
compute
communication
synchronization
memory
idle time
```

The benchmark layer answers:

> How fast did the complete training step run?

The profiler answers:

> Where did that time go?

These are deliberately separate questions.

## v0.9 — Final Benchmarks

All strategies are brought under a common benchmark definition.

```text
             Common Workload
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
      DDP          FSDP         ZeRO
       │            │            │
       ├────────────┼────────────┤
       ▼            ▼            ▼
       TP           PP       Single GPU
       │            │            │
       └────────────┼────────────┘
                    ▼
              Measurements
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Throughput  Memory   Step Time
```

The comparison is only meaningful if the workload and measurement procedure
are held constant.

Performance numbers are therefore collected separately from correctness
experiments.

## v1.0 — Capstone

The capstone is not another parallelism implementation.

It is the point where the earlier layers meet:

```text
                    Config
                      │
                      ▼
                 Distributed
                   Runtime
                      │
                      ▼
                   Trainer
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
         DDP         FSDP       Single GPU
          │           │           │
          └───────────┼───────────┘
                      ▼
                  Benchmark
                      │
                      ▼
                  Profiling
```

Tensor and pipeline parallelism remain explicit execution strategies rather
than being forced into an abstraction that hides their differences.

The final system should make those differences easier to see, not harder.

## Boundary

The project intentionally stops before becoming another distributed training
framework.

PyTorch and DeepSpeed already solve the production problem.

This lab is concerned with the layer underneath:

```text
What is replicated?
What is sharded?
What is communicated?
When is it communicated?
What waits?
What scales?
What stops scaling?
```

Those are the questions the architecture is built to expose.