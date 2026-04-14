# LayoutLMv3 Small-Scale Ablations

This file defines the small-scale ablation suite for the KG-oriented document graph model.

## Goal

The `audit200` suite is used to verify mechanism-level hypotheses before scaling to larger training sets.

The two main questions are:

1. Does task-adaptive graph reasoning improve KG construction stages beyond a shared graph reasoner?
2. Does DOM-derived structure help beyond pure spatial graph connections?

## Experiment Groups

### Group A: Task Routing

- `task_driven_layoutlmv3_gnn_audit200_mask.json`
  - main method
  - task-adaptive graph reasoning
- `task_driven_layoutlmv3_gnn_audit200_token.json`
  - replace mask routing with additive task token
- `task_driven_layoutlmv3_gnn_audit200_none.json`
  - shared graph reasoner across all KG construction stages

Interpretation:

- `mask > token > none` supports the task-adaptive graph reasoning hypothesis.
- `token > none` supports the value of task identity alone.
- `mask > token` supports the stronger claim that task-conditioned graph reasoning matters beyond a task label.

### Group B: Structure Input

- `task_driven_layoutlmv3_gnn_audit200_no_dom.json`
  - remove DOM-derived parent and same-parent edges
  - keep spatial KNN edges and `ref -> object` edges

Interpretation:

- `mask > no_dom` supports the usefulness of DOM-derived structural context.

## Primary Metrics

Task 1: Entity Consolidation

- `coref_pairwise_f1`

Task 2: Semantic Linking

- `relation_macro_f1`
- `relation_micro_f1`

Task 3: Attribute Canonicalization

- `norm_type_macro_f1`
- `norm_value_exact_match_recoverable`

Auxiliary Task 3 metrics:

- `norm_type_accuracy`
- `norm_value_exact_match`
- `norm_surface_format_accuracy_unrecoverable`

## Recommended Run Order

1. `mask`
2. `token`
3. `none`
4. `no_dom`

This keeps the reference run first and makes ablation comparisons easier to read.
