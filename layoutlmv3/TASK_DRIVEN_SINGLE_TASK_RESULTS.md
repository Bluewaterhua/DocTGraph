# Single-Task Task-Driven Results (`audit200`)

This note summarizes the single-task ablation runs used to test whether the
task-driven router becomes more effective when each run is optimized with only
one task loss.

## Setup

- Dataset: `contract_synth_safe_v2_audit200`
- Shared baseline: `router_mode = none`
- Task-driven variant: `router_mode = mask`
- Each run keeps only one task loss active and sets the other two task weights
  to `0.0`
- Reported numbers below use the best observed epoch under the primary metric
  of the active task, not necessarily the checkpoint selected by `val_loss`

## Primary Metrics

| Task | Metric | `none` | `mask` | Delta |
| --- | --- | ---: | ---: | ---: |
| Entity consolidation | F1 | 0.4557 | 0.6500 | +0.1943 |
| Semantic linking | Macro-F1 | 0.4641 | 0.5834 | +0.1193 |
| Attribute canonicalization | Type Macro-F1 | 0.5905 | 0.7314 | +0.1409 |

## Normalization Auxiliary Metric

| Task | Metric | `none` | `mask` | Delta |
| --- | --- | ---: | ---: | ---: |
| Attribute canonicalization | Value Exact Match | 0.5461 | 0.6028 | +0.0567 |

## Takeaway

Under single-task supervision, the task-driven routing variant consistently
outperforms the shared-routing baseline on all three KG-oriented stages. This
supports the hypothesis that the fully joint objective can partly obscure the
benefit of task-adaptive routing.

## Train10k Entity-Only

The `train10k` follow-up was run for the entity consolidation task only, using
the same single-task setup and comparing `none` against `mask_subgraph`.

| Dataset | Task | Metric | `none` | `mask_subgraph` | Delta |
| --- | --- | --- | ---: | ---: | ---: |
| `train10k` | Entity consolidation | Best F1 | 0.7890 | 0.7957 | +0.0067 |

Notes:

- `none`: best entity F1 at epoch `2`
- `mask_subgraph`: best entity F1 at epoch `2`
- If selection follows `val_loss`, `mask_subgraph` would pick epoch `3`, which
  is slightly worse on the active task metric
