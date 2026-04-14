# Audit500 Key Comparison Results

This file records the key `audit500` runs and compares them with `audit200` under the KG-oriented task formulation.

## Planned Runs

1. `mask` (legacy multiplicative gate)
2. `none`
3. `no_dom`
4. `mask_edgeattn` (task-specific edge attention)

## Metrics

- Task 1: Entity Consolidation
- `coref_pairwise_f1`
- Task 2: Semantic Linking
- `relation_macro_f1`
- `relation_micro_f1`
- Task 3: Attribute Canonicalization
- `norm_type_macro_f1`
- `norm_type_accuracy`
- `norm_value_exact_match`
- `norm_value_exact_match_recoverable`
- `norm_surface_format_accuracy_unrecoverable`

## Results

### mask

- Config: `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit500_mask.json`
- Output: `layoutlmv3/outputs/audit500_mask`
- Best epoch: `3`
- Best checkpoint: `layoutlmv3/outputs/audit500_mask/epoch_03.pt`
- `val_loss = 0.7096`
- `coref_pairwise_f1 = 0.7273`
- `relation_macro_f1 = 0.6453`
- `relation_micro_f1 = 0.9679`
- `norm_type_macro_f1 = 0.7784`
- `norm_type_accuracy = 0.7479`
- `norm_value_exact_match = 0.6148`
- `norm_value_exact_match_recoverable = 0.6201`
- `norm_surface_format_accuracy_unrecoverable = 1.0000`
- Compared with `audit200 mask`:
  - coref is slightly lower (`0.7273` vs `0.7755`)
  - relation is much stronger (`0.6453` vs `0.4493`)
  - normalization is slightly stronger (`0.6201` vs `0.5899`)
- Status: completed

### mask_edgeattn

- Config: `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit500_mask_edgeattn.json`
- Output: `layoutlmv3/outputs/audit500_mask_edgeattn`
- Historical best iteration (task-adaptive message passing):
  - `val_loss = 0.4934`
  - `coref_pairwise_f1 = 0.7477`
  - `relation_macro_f1 = 0.6647`
  - `relation_micro_f1 = 0.9971`
  - `norm_type_macro_f1 = 0.9131`
  - `norm_type_accuracy = 0.8669`
  - `norm_value_exact_match = 0.6961`
  - `norm_value_exact_match_recoverable = 0.7020`
- Current iteration (adds explicit task-edge/task-node type bias):
  - Best epoch: `3`
  - Best checkpoint: `layoutlmv3/outputs/audit500_mask_edgeattn/epoch_03.pt`
  - `val_loss = 0.4671`
  - `coref_pairwise_f1 = 0.7143`
  - `relation_macro_f1 = 0.6608`
  - `relation_micro_f1 = 0.9912`
  - `norm_type_macro_f1 = 0.9006`
  - `norm_type_accuracy = 0.8725`
  - `norm_value_exact_match = 0.6891`
  - `norm_value_exact_match_recoverable = 0.6949`
- `norm_surface_format_accuracy_unrecoverable = 1.0000`
- Compared with legacy `audit500 mask`:
  - historical best iteration is stronger on all main metrics
  - current explicit-bias iteration is weaker on task metrics but lower on validation loss
- Compared with `audit500 none`:
  - historical best iteration is slightly stronger on all three main tasks
  - current explicit-bias iteration is weaker on coref, nearly tied on relation, and slightly stronger on normalization
- Compared with the previous `mask_edgeattn` iteration:
  - historical best iteration remains the strongest task-adaptive routing result so far
  - adding explicit task-edge/task-node type bias did not improve over that version
- Status: completed

### none

- Config: `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit500_none.json`
- Output: `layoutlmv3/outputs/audit500_none`
- Best epoch: `3`
- Best checkpoint: `layoutlmv3/outputs/audit500_none/epoch_03.pt`
- `val_loss = 0.6069`
- `coref_pairwise_f1 = 0.7440`
- `relation_macro_f1 = 0.6628`
- `relation_micro_f1 = 0.9942`
- `norm_type_macro_f1 = 0.8715`
- `norm_type_accuracy = 0.8193`
- `norm_value_exact_match = 0.6681`
- `norm_value_exact_match_recoverable = 0.6737`
- `norm_surface_format_accuracy_unrecoverable = 1.0000`
- Compared with `audit200 none`:
  - coref is stronger (`0.7440` vs `0.6724`)
  - relation is much stronger (`0.6628` vs `0.4615`)
  - normalization is much stronger (`0.6737` vs `0.4784`)
- Compared with `audit500 mask`:
  - coref is slightly stronger (`0.7440` vs `0.7273`)
  - relation is slightly stronger (`0.6628` vs `0.6453`)
  - normalization is clearly stronger (`0.6737` vs `0.6201`)
- Status: completed

### no_dom

- Config: `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit500_no_dom.json`
- Output: `layoutlmv3/outputs/audit500_no_dom`
- Best epoch: `3`
- Best checkpoint: `layoutlmv3/outputs/audit500_no_dom/epoch_03.pt`
- `val_loss = 1.4531`
- `coref_pairwise_f1 = 0.6701`
- `relation_macro_f1 = 0.3563`
- `relation_micro_f1 = 0.5938`
- `norm_type_macro_f1 = 0.5289`
- `norm_type_accuracy = 0.5658`
- `norm_value_exact_match = 0.4454`
- `norm_value_exact_match_recoverable = 0.4492`
- `norm_surface_format_accuracy_unrecoverable = 0.1667`
- Compared with `audit200 no_dom`:
  - coref is stronger (`0.6701` vs `0.5846`)
  - relation partially recovers but remains much weaker than structured runs (`0.3563` vs `0.0400`)
  - normalization is only modestly stronger (`0.4492` vs `0.4101`)
- Compared with `audit500 mask`:
  - coref drops (`0.6701` vs `0.7273`)
  - relation drops sharply (`0.3563` vs `0.6453`)
  - normalization drops sharply (`0.4492` vs `0.6201`)
- Compared with `audit500 none`:
  - coref drops (`0.6701` vs `0.7440`)
  - relation drops sharply (`0.3563` vs `0.6628`)
  - normalization drops sharply (`0.4492` vs `0.6737`)
- Status: completed

## Audit200 Reference

- `mask`
  - `coref_pairwise_f1 = 0.7755`
  - `relation_macro_f1 = 0.4493`
  - `relation_micro_f1 = 0.6737`
  - `norm_type_macro_f1 = 0.7500`
  - `norm_value_exact_match_recoverable = 0.5899`
- `none`
  - `coref_pairwise_f1 = 0.6724`
  - `relation_macro_f1 = 0.4615`
  - `relation_micro_f1 = 0.6923`
  - `norm_type_macro_f1 = 0.5869`
  - `norm_value_exact_match_recoverable = 0.4784`
- `no_dom`
  - `coref_pairwise_f1 = 0.5846`
  - `relation_macro_f1 = 0.0400`
  - `relation_micro_f1 = 0.0698`
  - `norm_type_macro_f1 = 0.4424`
  - `norm_value_exact_match_recoverable = 0.4101`

## Summary

- `audit500` is materially more stable than `audit200`, especially on relation extraction and normalization.
- The legacy multiplicative `mask` is no longer the best task-driven variant and should be treated as an abandoned implementation.
- The best current task-adaptive setting on `audit500` is the task-adaptive message passing variant of `mask_edgeattn`.
- Adding explicit task-edge/task-node type bias on top of that did not further help.
- DOM-derived structure remains important. Even on `audit500`, removing parent and same-parent edges causes a large drop in relation extraction and hurts coreference and normalization as well.

## Audit500 Analysis Notes

- Under the KG-oriented formulation, the three tasks should be read as:
  - Task 1: entity consolidation
  - Task 2: semantic linking
  - Task 3: attribute canonicalization
- The biggest shift from `audit200` to `audit500` is that semantic linking stops looking noisy and becomes consistently high for both `mask` and `none`.
- Replacing the legacy hand-written node gate with task-adaptive message passing improves both the method definition and the empirical results. The strongest version no longer suppresses node representations with a fixed prior and instead conditions edge weighting, message transformation, and update transformation on the task.
- Explicitly adding task-edge and task-node type bias on top of that did not improve results, which suggests the softer learned task adapters already capture most of the useful task preference.
- The current evidence for the method is therefore asymmetric:
  - the graph structure prior is strongly supported
  - the revised task-conditioned graph reasoning is also supported on `audit500`
- For `10k`, the safest comparison set is now:
  1. `mask_edgeattn`
  2. `none`
  3. `no_dom`

## Prompt Pilot

- Goal: test task prompts at the encoder stage instead of graph-side pruning or masking.
- Configs:
  - `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit500_prompt_pilot.json`
  - `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit500_none_pilot.json`
- Shared pilot setup:
  - `1` epoch
  - `8` train steps
  - `20` validation documents
- `prompt` pilot:
  - `val_loss = 2.5432`
  - `entity_consolidation_f1 = 0.0000`
  - `semantic_linking_macro_f1 = 0.0000`
  - `attribute_type_macro_f1 = 0.0809`
  - `attribute_value_exact_match_recoverable = 0.2711`
  - `kg_stage_macro_score = 0.0904`
- `none` pilot:
  - `val_loss = 2.5299`
  - `entity_consolidation_f1 = 0.0000`
  - `semantic_linking_macro_f1 = 0.0000`
  - `attribute_type_macro_f1 = 0.0809`
  - `attribute_value_exact_match_recoverable = 0.2711`
  - `kg_stage_macro_score = 0.0904`
- Observation:
  - Under the same short pilot budget, encoder-side task prompts did not show any measurable gain over `none`.
  - This pilot only validates the implementation path; it is too short to support a conclusion about the full method.

## Recommendation For 10k

- Use `audit500 mask_edgeattn` as the current leading configuration.
- Keep `audit500 none` as the main shared-routing comparison.
- Keep `audit500 no_dom` as the structure ablation.
- Drop the legacy multiplicative gate from the main story.
- For the main story, define the method as task-adaptive graph reasoning on a shared full graph for three KG construction stages, rather than task-specific pruning.

## Full Graph + Task Subgraph (Typed Graph Regime)

- Note:
  - The following runs are under the updated typed graph builder (type-constrained spatial edges and narrowed semantic-link candidates).
  - They are directly comparable with each other, but not directly comparable with earlier pre-change audit500 records.

### `mask_subgraph`

- Config: `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit500_mask_subgraph.json`
- Output: `layoutlmv3/outputs/audit500_mask_subgraph`
- Best epoch: `3`
- `val_loss = 0.5528`
- `entity_consolidation_f1 = 0.7016`
- `semantic_linking_macro_f1 = 0.6174`
- `semantic_linking_micro_f1 = 0.9263`
- `attribute_type_macro_f1 = 0.8800`
- `attribute_value_exact_match = 0.7003`
- `attribute_value_exact_match_recoverable = 0.7062`
- `kg_stage_macro_score = 0.6750`

### `none` (rerun under the same typed graph regime)

- Config: `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit500_none.json`
- Output: `layoutlmv3/outputs/audit500_none`
- Best epoch: `3`
- `val_loss = 0.6678`
- `entity_consolidation_f1 = 0.7349`
- `semantic_linking_macro_f1 = 0.6667`
- `semantic_linking_micro_f1 = 1.0000`
- `attribute_type_macro_f1 = 0.8313`
- `attribute_value_exact_match = 0.6275`
- `attribute_value_exact_match_recoverable = 0.6328`
- `kg_stage_macro_score = 0.6781`

### Comparison Summary

- `mask_subgraph`:
  - better `val_loss` and clearly better attribute canonicalization (`0.7062` vs `0.6328` recoverable exact match)
- `none`:
  - better entity consolidation and semantic linking
  - slightly better overall `kg_stage_macro_score` (`0.6781` vs `0.6750`)

## Train10k

- Dataset:
  - `synthetic_contract_ds/contract_synth_safe_v2_train10k`
  - `9000` train docs / `1000` val docs
- Note:
  - These runs use the current typed graph builder regime.

### `mask_subgraph`

- Config: `layoutlmv3/configs/task_driven_layoutlmv3_gnn_train10k_mask_subgraph.json`
- Output: `layoutlmv3/outputs/train10k_mask_subgraph`
- Best epoch: `3`
- `val_loss = 0.3602`
- `entity_consolidation_f1 = 0.7826`
- `semantic_linking_macro_f1 = 0.6667`
- `semantic_linking_micro_f1 = 1.0000`
- `attribute_type_macro_f1 = 0.8284`
- `attribute_value_exact_match = 0.6922`
- `attribute_value_exact_match_recoverable = 0.7029`
- `kg_stage_macro_score = 0.7174`

### `none`

- Config: `layoutlmv3/configs/task_driven_layoutlmv3_gnn_train10k_none.json`
- Output: `layoutlmv3/outputs/train10k_none`
- Best epoch: `3`
- `val_loss = 0.3521`
- `entity_consolidation_f1 = 0.7879`
- `semantic_linking_macro_f1 = 0.6667`
- `semantic_linking_micro_f1 = 1.0000`
- `attribute_type_macro_f1 = 0.8726`
- `attribute_value_exact_match = 0.6921`
- `attribute_value_exact_match_recoverable = 0.7028`
- `kg_stage_macro_score = 0.7191`

### Train10k Summary

- On `10k`, `mask_subgraph` and `none` are effectively tied.
- `none` is slightly better on `val_loss`, entity consolidation, attribute type classification, and overall `kg_stage_macro_score`.
- `semantic_linking_macro_f1` is identical (`0.6667`).
- `attribute_value_exact_match_recoverable` is also effectively identical (`0.7029` vs `0.7028`).
