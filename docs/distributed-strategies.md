# Distributed Strategies

The strategies in this lab solve different problems. Treating them as
interchangeable "ways to use multiple GPUs" hides the important part.

The useful distinction is what each strategy chooses to partition and what
communication that partitioning introduces.

## DDP

Distributed Data Parallelism replicates the model on every worker and
partitions the input data.

Each process performs its own forward and backward pass, then gradients are
synchronized through collective communication.

```text
                  Model
                 /     \
                /       \
             Rank 0    Rank 1
               |          |
            Batch 0     Batch 1
               |          |
           Backward    Backward
                \        /
                 \      /
                 AllReduce
                     |
                     ↓
             synchronized model
```

DDP is the baseline because it changes the least. The model remains intact on
every device and only the data and resulting gradients are distributed.

That makes it simple, but the full model, gradients, and optimizer state still
need to fit on every worker.

## FSDP

FSDP changes the ownership of model state.

Instead of keeping a complete copy of parameters, gradients, and optimizer
state on every worker, the state is sharded across the data-parallel group.

```text
                 Model State
                      |
          ┌───────────┼───────────┐
          ↓           ↓           ↓
        Rank 0      Rank 1      Rank 2
        shard       shard       shard
```

Parameters are gathered when required for computation and can be resharded
afterward.

The fundamental tradeoff becomes clear:

```text
                 less memory
                     ↕
              more communication
```

FSDP is therefore not simply "DDP but faster". It changes the memory boundary
of the training system.

## ZeRO

ZeRO takes state partitioning further through DeepSpeed.

The stages progressively partition optimizer states, gradients, and parameters.
At the highest stage, the model parameters themselves are partitioned across
workers.

The lab uses DeepSpeed rather than reproducing the implementation because the
interesting boundary is the partitioning strategy, not rebuilding a production
training engine.

The useful comparison with FSDP is therefore empirical: how do different
sharding strategies affect memory, communication, and throughput under the
same workload?

## Tensor Parallelism

Tensor parallelism partitions individual operations across devices.

This is fundamentally different from DDP and FSDP. The model is not merely
replicated or sharded between steps; the computation itself is distributed.

### Column Parallel

A weight matrix is partitioned along its output dimension.

```text
                     W
                ┌────┴────┐
                ↓         ↓
               W0         W1
             Rank 0     Rank 1
                |         |
               XW0       XW1
```

Each device computes a portion of the output.

### Row Parallel

The input is partitioned and each device produces a partial result.

```text
                     X
                ┌────┴────┐
                ↓         ↓
               X0         X1
                |         |
               X0W0      X1W1
                \         /
                 \       /
                  AllReduce
                     |
                     ↓
                     Y
```

The two primitives compose naturally into the structure commonly used for
Transformer MLPs:

```text
Column Parallel
       ↓
      GELU
       ↓
Row Parallel
       ↓
   AllReduce
```

The interesting property is the location of communication. Splitting a layer
creates communication that does not exist in a single-device execution.

## Pipeline Parallelism

Pipeline parallelism partitions the model by depth.

Different devices own different stages of the model, while microbatches move
through those stages.

```text
GPU 0                    GPU 1

Stage 0 ───────────────→ Stage 1
   |                         |
   ├─ microbatch 0 ─────────→|
   ├─ microbatch 1 ─────────→|
   └─ microbatch 2 ─────────→|
```

Pipeline execution can keep multiple stages busy, but it introduces pipeline
bubbles and stage-to-stage communication.

The number of stages, number of microbatches, and balance of computation
between stages therefore matter.

The lab uses PyTorch's pipeline API and GPipe scheduling rather than
implementing another scheduling framework.

## The Tradeoff

The strategies can be viewed through the resource they choose to distribute.

| Strategy | What is partitioned? | Main communication | Main reason to use it |
|---|---|---|---|
| DDP | Data | Gradient synchronization | Simple scaling |
| FSDP | Model state | Parameter/gradient collectives | Reduce memory |
| ZeRO | Training state | Parameter/gradient communication | Reduce memory |
| Tensor Parallel | Operations | Layer collectives | Scale individual layers |
| Pipeline Parallel | Model depth | Stage communication | Scale model depth |

These strategies can also be combined.

A large training system may use data, tensor, and pipeline parallelism together
rather than choosing exactly one.

That is outside the scope of the current implementation, but it is where the
individual mechanisms become a larger systems problem.

## What the Lab Is Actually Trying to Learn

The useful questions are not:

```text
Which strategy is best?
Which framework is fastest?
Which technique is newest?
```

They are:

```text
What is replicated?
What is sharded?
Where does communication appear?
When does synchronization happen?
How much memory is recovered?
What becomes the bottleneck?
What stops scaling?
```

Those answers should come from the experiments rather than from the names of
the techniques.