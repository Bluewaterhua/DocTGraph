# Audit200 Ablation Results

This file records the small-scale ablation results for the `audit200` suite.

## Run Order

1. `mask`
2. `token`
3. `none`
4. `no_dom`

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

- Config: `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit200_mask.json`
- Output: `layoutlmv3/outputs/audit200_mask`
- Best epoch: `3`
- Best checkpoint: `layoutlmv3/outputs/audit200_mask/epoch_03.pt`
- `val_loss = 0.8921`
- `coref_pairwise_f1 = 0.7755`
- `relation_macro_f1 = 0.4493`
- `relation_micro_f1 = 0.6737`
- `norm_type_macro_f1 = 0.7500`
- `norm_type_accuracy = 0.7128`
- `norm_value_exact_match = 0.5816`
- `norm_value_exact_match_recoverable = 0.5899`
- `norm_surface_format_accuracy_unrecoverable = 1.0000`
- Status: completed

### token

- Config: `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit200_token.json`
- Output: `layoutlmv3/outputs/audit200_token`
- Best epoch: `3`
- Best checkpoint: `layoutlmv3/outputs/audit200_token/epoch_03.pt`
- `val_loss = 1.2390`
- `coref_pairwise_f1 = 0.6882`
- `relation_macro_f1 = 0.4502`
- `relation_micro_f1 = 0.6771`
- `norm_type_macro_f1 = 0.5309`
- `norm_type_accuracy = 0.6277`
- `norm_value_exact_match = 0.4716`
- `norm_value_exact_match_recoverable = 0.4784`
- `norm_surface_format_accuracy_unrecoverable = 1.0000`
- Status: completed

### none

- Config: `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit200_none.json`
- Output: `layoutlmv3/outputs/audit200_none`
- Best epoch: `3`
- Best checkpoint: `layoutlmv3/outputs/audit200_none/epoch_03.pt`
- `val_loss = 1.2242`
- `coref_pairwise_f1 = 0.6724`
- `relation_macro_f1 = 0.4615`
- `relation_micro_f1 = 0.6923`
- `norm_type_macro_f1 = 0.5869`
- `norm_type_accuracy = 0.6135`
- `norm_value_exact_match = 0.4716`
- `norm_value_exact_match_recoverable = 0.4784`
- `norm_surface_format_accuracy_unrecoverable = 1.0000`
- Status: completed

### no_dom

- Config: `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit200_no_dom.json`
- Output: `layoutlmv3/outputs/audit200_no_dom`
- Best epoch: `3`
- Best checkpoint: `layoutlmv3/outputs/audit200_no_dom/epoch_03.pt`
- `val_loss = 1.5836`
- `coref_pairwise_f1 = 0.5846`
- `relation_macro_f1 = 0.0400`
- `relation_micro_f1 = 0.0698`
- `norm_type_macro_f1 = 0.4424`
- `norm_type_accuracy = 0.5426`
- `norm_value_exact_match = 0.4043`
- `norm_value_exact_match_recoverable = 0.4101`
- `norm_surface_format_accuracy_unrecoverable = 0.2500`
- Status: completed

## Summary

- `mask` is the strongest setting overall by validation loss, coreference F1, and normalization performance.
- `token` and `none` are close on semantic linking, but both are clearly weaker than `mask` on entity consolidation and attribute canonicalization.
- `none` slightly exceeds `mask` on relation macro/micro F1 in this small run, so the current evidence for task-adaptive graph reasoning mainly comes from entity consolidation and attribute canonicalization rather than semantic linking.
- `no_dom` collapses relation performance and significantly hurts all three tasks, which strongly supports the importance of DOM-derived structural edges.

## Audit200 Analysis Notes

- In the later project direction, `mask` should be interpreted as task-adaptive graph reasoning on a shared full graph, not as hard task-specific pruning.
- The small-scale `audit200` result still shows the clearest gains on entity consolidation and attribute canonicalization.
- The difference between `token` and `none` is small on semantic linking in the current `audit200` run, so the routing advantage is not yet uniformly established for Task 2.
- Removing DOM-derived parent and same-parent edges causes a severe drop in semantic linking and also hurts entity consolidation and attribute canonicalization, showing that structural context is not optional in this pipeline.
- In the current implementation, removing DOM-derived edges does not leave the model with text only. The model still retains text, image, bounding boxes, spatial KNN edges, and explicit `ref -> object` edges. What is lost is the hierarchical and container-level structure prior. In effect, the model degrades from a structure-aware document graph to a multimodal spatial-neighborhood graph.
- For Task 3, the recoverable exact match metric is more informative than raw exact match because masked bank accounts introduce unrecoverable samples by construction.
- The unrecoverable formatting score should be treated as an auxiliary metric only; it does not mean hidden values are recovered.

## Audit500 Plan

- Dataset status: `not generated yet`
- Expected next step:
  - generate or restore `synthetic_contract_ds/contract_synth_safe_v2_audit500`
  - run the key comparison subset on `audit500`
- Recommended `audit500` subset:
  1. `mask`
  2. `none`
  3. `no_dom`

Rationale:

- `mask` tests whether the current best small-scale setting remains strongest.
- `none` checks whether the routing benefit holds beyond the smallest audit split.
- `no_dom` checks whether the structural-edge dependency remains strong at a slightly larger scale.

Expected outcome:

- `audit500` should be more stable than `audit200`, especially for relation extraction, because the current task 2 differences between `mask` and `none` are still noisy at the smallest scale.

## Prompt Light Pilot

- Goal: test a lightweight prompt-conditioned direction without running the encoder three times.
- Shared setup:
  - dataset: `audit200`
  - `1` epoch
  - `24` train steps
  - `20` validation documents
- Configs:
  - `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit200_prompt_light_pilot.json`
  - `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit200_none_pilot.json`
- `prompt_light`:
  - `val_loss = 2.4081`
  - `entity_consolidation_f1 = 0.0000`
  - `semantic_linking_macro_f1 = 0.0000`
  - `attribute_type_macro_f1 = 0.0807`
  - `attribute_value_exact_match_recoverable = 0.3094`
  - `kg_stage_macro_score = 0.1031`
- `none`:
  - `val_loss = 2.4043`
  - `entity_consolidation_f1 = 0.0000`
  - `semantic_linking_macro_f1 = 0.0000`
  - `attribute_type_macro_f1 = 0.0807`
  - `attribute_value_exact_match_recoverable = 0.3094`
  - `kg_stage_macro_score = 0.1031`
- Observation:
  - Lightweight prompt conditioning is computationally cheap again.
  - Under a matched pilot budget, it does not show any measurable gain over `none`.

## Typed Graph Builder Trial (audit200 mask)

- Change:
  - Added type-constrained `spatial_knn` edges.
  - Restricted semantic linking block sources to `p_*` and `cap_*` blocks (plus all `ref` nodes).
- Run:
  - Config: `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit200_mask.json`
  - Log: `layoutlmv3/outputs/audit200_mask_typed_graph.log`
- Best epoch (`3`) vs previous `audit200 mask` recheck:
  - `val_loss`: `0.8834` vs `0.9480` (better)
  - `entity_consolidation_f1`: `0.6813` vs `0.7143` (worse)
  - `semantic_linking_macro_f1`: `0.4684` vs `0.4530` (better)
  - `attribute_value_exact_match_recoverable`: `0.7086` vs `0.5468` (better, large gain)
  - `kg_stage_macro_score`: `0.6194` vs `0.5713` (better)
- Observation:
  - Type-aware graph construction improves overall KG-stage score and significantly improves attribute canonicalization.
  - Entity consolidation dropped slightly, which suggests Task 1 may need a dedicated mention-focused edge supplement.

## True Subgraph Trial (audit200 mask_subgraph)

- Change:
  - Kept one global graph per document.
  - Added task-specific edge masks (`task_edge_masks`) in `graph_builder`.
  - During GNN propagation, each task uses only its own subgraph:
    - Task 1: entity consolidation subgraph
    - Task 2: semantic linking subgraph
    - Task 3: attribute canonicalization subgraph
  - Router mode: `mask_subgraph` (task-adaptive message passing + task subgraph).
- Config:
  - `layoutlmv3/configs/task_driven_layoutlmv3_gnn_audit200_mask_subgraph.json`
- Output:
  - `layoutlmv3/outputs/audit200_mask_subgraph`
- Best epoch (`3`) metrics:
  - `val_loss = 0.9075`
  - `entity_consolidation_f1 = 0.8000`
  - `semantic_linking_macro_f1 = 0.5010`
  - `semantic_linking_micro_f1 = 0.7485`
  - `attribute_type_macro_f1 = 0.7921`
  - `attribute_value_exact_match = 0.6454`
  - `attribute_value_exact_match_recoverable = 0.6547`
  - `kg_stage_macro_score = 0.6519`
- Observation:
  - This is better than the current typed-graph-only `mask` trial on all three KG stages.
  - The subgraph mechanism improves both relation quality and entity consolidation while keeping normalization strong.
