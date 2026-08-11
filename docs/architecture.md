# Architecture

The lab evolves from a single-process training engine into a distributed
training system.

The architecture is intentionally layered. Distributed execution should
change how work is executed without requiring the core training logic to
be rewritten for every parallelism strategy.

## v0.1 — Single GPU

```text
                    ┌──────────────────┐
                    │     Config       │
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