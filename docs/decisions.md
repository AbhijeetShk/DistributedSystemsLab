# Engineering Decisions

This file records decisions that shape the lab.

The purpose is not to document every implementation detail. It is to preserve
the reasoning behind choices that would otherwise disappear as the code grows.

## Minimal Model

The model is intentionally small relative to the systems it is meant to
represent.

A distributed systems lab does not need a frontier-scale language model to
demonstrate distributed training mechanics. The implementation should remain
small enough that process behavior, communication, memory, and failures can
be inspected directly.

The model is therefore a workload, not the subject of the project.

For CUDA benchmarking, the workload was deliberately scaled beyond the tiny
development configuration until GPU compute and memory became meaningful.
The benchmark model is still small by modern training standards, but large
enough to expose the systems behavior being studied.

## Use Mature Distributed Primitives

The lab does not attempt to reproduce production implementations of DDP, FSDP,
DeepSpeed, or pipeline scheduling.

Where a mature implementation already exists, use it.

The useful questions are:

- What abstraction does it expose?
- What state does it own?
- What communication does it introduce?
- What assumptions does it make?
- What does the user give up in exchange for the optimization?

Reimplementing thousands of lines to arrive at an inferior copy would obscure
those questions.

## Implement the Mechanism When the Mechanism Is the Point

Tensor parallelism is treated differently.

The column-parallel and row-parallel layers are implemented directly because
the partitioning and collective communication are exactly what the experiment
is trying to understand.

The implementation stays small enough that the communication path can be read
directly.

## Keep the Training Core Stable

The training loop should not become:

```text
if ddp:
    ...
elif fsdp:
    ...
elif tensor_parallel:
    ...
elif pipeline_parallel:
    ...
```

just to make every strategy look identical.

DDP and FSDP can naturally sit around the same training abstraction.

Tensor and pipeline parallelism alter the execution model itself and therefore
remain explicit where that makes the system easier to understand.

The abstraction should follow the mechanism, not hide it.

## Distributed Runtime Is Its Own Layer

Process-group initialization, rank handling, device selection, and cleanup
belong to the distributed runtime.

The trainer should receive a distributed context rather than reconstructing
one.

This keeps:

```text
process management
```

separate from:

```text
training logic
```

and makes local execution possible without pretending that distributed state
does not exist.

## Correctness Before Performance

A distributed implementation that is fast but occasionally wrong is not useful.

The development order is therefore:

```text
single process
      ↓
unit tests
      ↓
multi-process correctness
      ↓
profiling
      ↓
benchmarking
```

Performance claims come last.

## Local Gloo Is a Correctness Environment

The development machine is macOS, so local distributed experiments use Gloo
where CUDA/NCCL is unavailable.

That is useful for validating:

- process-group initialization
- ranks
- world size
- collectives
- distributed data partitioning
- basic synchronization

It is not treated as a substitute for GPU performance testing.

CUDA/NCCL measurements belong in the final benchmark environment.

## Benchmark Complete Steps

The benchmark measures complete training steps rather than isolated kernels.

A distributed strategy can make an individual operation faster while making the
overall step slower because of communication or synchronization.

The quantity that matters is therefore:

```text
time to perform useful training work
```

The profiler is used afterward when the aggregate result needs explanation.

## Don't Optimize Before the Bottleneck Exists

The project does not add:

- custom communication overlap
- custom memory allocators
- custom schedulers
- elaborate asynchronous execution
- framework-level abstractions

without an observed bottleneck that justifies them.

An optimization without a measurement is a hypothesis.

It should be treated as one.

## Checkpointing Is Part of Distributed Training

Checkpointing is not treated as a final serialization detail.

Once model state is partitioned, the representation of that state changes.

Therefore checkpoint save/load behavior is tested as part of the distributed
training system.

## Keep Experiments Reproducible

Experiments should expose:

```text
configuration
launcher
world size
workload
measurement procedure
```

Results without those details are difficult to interpret.

The final GPU benchmark should therefore record the environment alongside the
measurements.

## Avoid Feature-Checklist Development

The roadmap is not:

```text
implement every feature used by modern LLM infrastructure
```

A feature earns its place when it exposes a meaningful systems question.

For example:

```text
Tensor Parallelism
        ↓
Where does communication enter a Transformer layer?

FSDP
        ↓
How much memory can state sharding recover, and what does it cost?

Pipeline Parallelism
        ↓
What happens when computation is split across sequential stages?

Profiling
        ↓
Where does the time actually go?
```

This keeps the project focused.

## What Comes Next

After the core implementation is complete, new work should be driven by
observations from the system rather than by expanding the feature list.

A useful next experiment should answer a question that the current system
cannot yet answer.

That is a stronger criterion than adding another version number.