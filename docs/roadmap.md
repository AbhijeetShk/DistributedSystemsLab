# Roadmap

The implementation path is intentionally linear: start with a working
single-device trainer, establish distributed execution, then introduce
different ways of partitioning data, model state, computation, and execution.

## Completed

The core implementation path is complete.

```text
Single GPU
    ↓
Distributed Runtime
    ↓
DDP
    ↓
FSDP
    ↓
ZeRO
    ↓
Tensor Parallelism
    ↓
Pipeline Parallelism
    ↓
Profiling
    ↓
Benchmarking
    ↓
Capstone
```

Each stage exists because it changes a different part of the systems problem.
DDP changes data ownership, FSDP and ZeRO change state ownership, tensor
parallelism changes how individual operations execute, and pipeline
parallelism changes how the model's computation is scheduled.

## Validation

The implementation has been validated locally through unit tests and
multi-process Gloo experiments.

The CUDA path was then exercised on a Tesla T4. Profiling and workload sweeps
were used to identify a regime where model compute and memory became
substantial rather than dominated by framework overhead.

A roughly 303M-parameter model with batch size 4 and sequence length 512 is now
the locked workload for future multi-GPU measurements.

A complete multi-GPU comparison is not included because the available CUDA
environment contains one physical GPU. Simulating additional ranks on that
device would measure contention rather than distributed scaling.

## What Comes Next

The next feature should come from an observation, not a checklist.

If the system exposes a question I cannot answer, build the experiment.

If it doesn't, don't add the feature.

Some directions are interesting for that reason: communication/computation
overlap, activation checkpointing under sharding, sequence parallelism,
mixture-of-experts and expert parallelism, distributed inference, continuous
batching, fault tolerance, and multi-node scaling.

None of these are committed features. They become worth implementing when the
existing experiments give a reason to investigate them.

## Frontier

The longer-term direction is to move from reproducing known parallelism
patterns toward understanding the systems problems appearing in frontier AI
infrastructure.

That means looking beyond individual techniques and asking how memory,
communication, scheduling, model architecture, and hardware interact when
the scale changes.

The roadmap should therefore remain deliberately incomplete.

A useful system should change its roadmap when the experiments reveal
something unexpected.