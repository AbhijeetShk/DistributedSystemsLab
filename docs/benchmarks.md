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

## Workload Selection

The first CUDA workload was intentionally small:

```text
hidden size     = 128
layers          = 2
heads           = 4
batch size      = 8
sequence length = 32
```

It was useful for validating the CUDA path, but profiling showed that the
workload was dominated by CPU-side framework overhead and kernel launches.
The GPU execution time was only a small fraction of the measured CPU time.

The workload was therefore increased progressively rather than chosen
arbitrarily.

The resulting model-size sweep was:

```text
85.5M parameters    2.16 GB peak memory
302.9M parameters   6.49 GB peak memory
473.0M parameters   9.29 GB peak memory
```

The 302.9M-parameter configuration was selected as the working benchmark
configuration. It provides substantial compute and memory pressure on the
available T4 while leaving enough headroom for the rest of the system.

## Locked Workload

```text
hidden size     = 1024
layers          = 24
attention heads = 16
parameters      ≈ 303M

batch size      = 4
sequence length = 512

warmup steps    = 5
benchmark steps = 20
```

The workload is fixed before comparing execution strategies. It should not be
changed to make an individual strategy look better.

## CUDA Environment

The CUDA experiments were run on:

```text
GPU       = NVIDIA Tesla T4
Memory    = 14.6 GB
PyTorch   = 2.11.0+cu128
CUDA      = 12.8
GPUs      = 1
```

The local development environment uses macOS and Gloo for distributed
correctness experiments. Those runs validate process groups, collectives,
synchronization, and distributed data handling, but are not treated as GPU
performance measurements.

## T4 Measurements

The workload exploration produced:

| Model | Parameters | Batch | Sequence | Step Time | Tokens/s | Peak Memory |
|---|---:|---:|---:|---:|---:|---:|
| Small | 3.3M | 4 | 256 | 16.05 ms | — | 0.13 GB |
| Medium | 25.4M | 4 | 256 | 63.13 ms | — | 0.59 GB |
| 85M | 85.5M | 4 | 512 | 412.5 ms | 4,965 | 2.16 GB |
| 303M | 302.9M | 4 | 512 | 1,409.7 ms | 1,453 | 6.49 GB |
| 473M | 473.0M | 4 | 512 | 2,009.6 ms | 1,019 | 9.29 GB |

These measurements were used to select the benchmark workload rather than to
rank distributed strategies.

The 303M configuration is the locked workload for future multi-GPU runs.

## Profiling

The profiler was used to understand why the original small workload was
unsuitable.

For the 128-hidden, two-layer model:

```text
Self CPU time     = 230.985 ms
Self CUDA time    =   5.606 ms
```

A large amount of CPU-side time was associated with kernel launches and
runtime module loading.

The useful observation is not the absolute number. It is that the workload
was operating in the wrong execution regime for studying distributed
performance.

A distributed strategy cannot compensate for a workload whose dominant cost
is framework and launch overhead.

## Distributed Results

A complete multi-GPU performance comparison was not performed.

The available CUDA environment contained one physical GPU. Launching multiple
distributed processes on the same T4 would measure contention rather than
multi-GPU scaling, so those numbers would not represent the behavior this lab
is intended to study.

The following strategies are implemented and have correctness coverage:

```text
DDP
FSDP
ZeRO / DeepSpeed
Tensor Parallelism
Pipeline Parallelism
```

Their multi-GPU performance remains an open experiment rather than an
unmeasured claim.

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

The table is intentionally empty. The required multi-GPU hardware was not
available, and no predicted results are substituted for measurements.

## Questions

The eventual multi-GPU experiment should answer more than which strategy is
fastest.

The useful questions are whether memory savings justify additional
communication, when DDP stops scaling efficiently, where tensor parallelism
starts paying for its collectives, how pipeline bubbles affect utilization,
and how those tradeoffs change as the workload grows.

The benchmark is successful if it makes those tradeoffs visible.