# Benchmarks

The benchmark layer exists to make the distributed strategies comparable.

A benchmark is useful only when the workload, measurement procedure, and
environment are explicit. Otherwise a throughput number is mostly a property
of the benchmark setup.

## What Is Measured

The benchmark records elapsed training time, samples processed, samples per
second, peak CUDA memory, and world size.

For language-model workloads, token throughput can be derived from samples per
second and sequence length.

```text
samples / second × sequence length
```

Step time is derived from the measured training interval:

```text
elapsed time / measured steps
```

The profiler provides a second view of the same execution. Instead of asking
how fast the complete step was, it helps explain where that time went.

## Common Workload

The final benchmark configuration currently uses the same basic workload
definition across strategies:

```text
batch size       = 8
sequence length  = 32
warmup steps     = 5
benchmark steps  = 20
```

The model configuration is kept fixed while comparing parallelism strategies.

The exact hardware, software versions, world size, and launch configuration
should be recorded with the final results.

## Warmup

Initial iterations are not included in the measured interval.

This avoids treating startup effects, lazy initialization, memory allocation,
and other one-time costs as steady-state training performance.

The benchmark therefore follows:

```text
initialization
      ↓
warmup
      ↓
measurement
```

## Metrics

### Throughput

The primary throughput metric is samples per second:

```text
samples / elapsed seconds
```

For the language-model workload, token throughput is:

```text
samples / second × sequence length
```

Higher throughput is better, but only within the same workload and hardware
configuration.

### Step Time

Average time per measured training step.

Lower is better.

This is particularly useful when comparing communication-heavy strategies,
because a higher throughput number alone does not explain where the time is
being spent.

### Peak Memory

Peak CUDA memory allocated during the measured execution.

Memory is one of the main reasons to move beyond DDP, so this metric is as
important as throughput.

A strategy that uses less memory but introduces substantial communication may
still be the better choice when the alternative cannot fit the workload.

### Scaling

When multiple world sizes are available, scaling should be measured rather
than assumed.

A simple throughput scaling comparison is:

```text
distributed throughput / single-device throughput
```

Ideal scaling would increase proportionally with the number of devices.

The gap between ideal and observed scaling is where communication,
synchronization, imbalance, and other overheads become visible.

## Profiling

The profiler is not intended to replace the benchmark.

The benchmark tells us that something changed.

The profiler helps answer why.

For example, a lower throughput result might come from:

```text
more collective communication
synchronization stalls
memory movement
pipeline bubbles
load imbalance
```

The DDP profiling experiment exports a trace that can be inspected at the
operator and execution level.

## Environment

The final performance measurements should be collected on Linux with CUDA and
NCCL.

The local macOS/Gloo experiments are useful for validating distributed
correctness and process behavior, but they are not used as GPU performance
evidence.

The final result should record at least:

```text
GPU model
number of GPUs
PyTorch version
CUDA version
NCCL version
world size
batch size
sequence length
model configuration
```

Without this information, the numbers are difficult to reproduce or interpret.

## Final Comparison

The intended comparison is:

| Strategy | Step time | Samples/s | Tokens/s | Peak memory |
|---|---:|---:|---:|---:|
| Single GPU | — | — | — | — |
| DDP | — | — | — | — |
| FSDP | — | — | — | — |
| ZeRO-3 | — | — | — | — |
| Tensor Parallel | — | — | — | — |
| Pipeline Parallel | — | — | — | — |

These values are intentionally left empty until the CUDA/NCCL experiments are
run.

The point of the benchmark is to discover the tradeoffs, not to manufacture a
ranking beforehand.

## Questions

The final experiment should leave us with more than a table.

The useful questions are whether memory savings justify additional
communication, when DDP stops scaling efficiently, where tensor parallelism
starts paying for its collectives, how pipeline bubbles affect utilization,
and whether the different strategies behave differently as the workload grows.

If the measurements answer those questions, the benchmark has done its job.