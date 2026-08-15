# Roadmap

The implementation path is intentionally linear: start with a working
single-device trainer, establish distributed execution, then introduce
different ways of partitioning data, model state, computation, and execution.

## Completed

The core path is complete.

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

## Final Validation

The remaining benchmark work is experimental rather than architectural. The
completed implementations need to be run under a CUDA/NCCL environment using
the common benchmark configuration.

The comparison should measure throughput, step time, peak memory, and scaling
behavior across DDP, FSDP, ZeRO, tensor parallelism, and pipeline parallelism.
The profiler should then be used to explain the differences rather than simply
report them.

The important result is not which strategy "wins". It is understanding why the
result changes with model size, batch size, device count, and communication
pattern.

## What Comes Next

The next feature should come from an observation, not a checklist.

If the system exposes a question I cannot answer, build the experiment.

If it doesn't, don't add the feature.

Some directions are interesting for that reason: communication/computation
overlap, activation checkpointing under sharding, sequence parallelism,
mixture-of-experts and expert parallelism, distributed inference, continuous
batching, fault tolerance, and multi-node scaling.

None of these are committed features yet. They become worth implementing when
the existing experiments give a reason to investigate them.

## Frontier

The longer-term direction is to move from reproducing known parallelism
patterns toward understanding the systems problems appearing in frontier AI
infrastructure.

That means looking beyond individual techniques and asking how memory,
communication, scheduling, model architecture, and hardware interact when the
scale changes.

The roadmap should therefore remain deliberately incomplete.

A useful system should change its roadmap when the experiments reveal something
unexpected.