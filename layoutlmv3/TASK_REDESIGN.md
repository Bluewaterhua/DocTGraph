# Task Redesign For KG Construction

This note reframes the current three-task setup from the perspective of knowledge graph construction rather than "three independent classifiers".

## Core Goal

The real system goal is:

- extract structured semantic units from documents so they can be inserted into a knowledge graph reliably.

This means the pipeline should support three graph-construction stages:

1. graph node identity resolution
2. graph edge construction
3. graph attribute normalization

The task split should follow these stages.

## Problem With The Current Split

The current task definitions are:

1. `coref`
2. `relation`
3. `normalization`

These are directionally correct, but in the current implementation all three tasks are reduced to very similar supervision forms:

- `coref`: mention-pair classification
- `relation`: candidate edge classification
- `normalization`: value type classification plus rule-based normalization

As a result, the model can often learn a strong shared representation and behave like `none`, because the tasks are not separated at the level of graph-construction function.

## Proposed Task Split

### Task 1: Entity Consolidation

Goal:

- decide which document mentions should be consolidated into the same KG entity.

Input emphasis:

- mention surface form
- nearby descriptive context
- document-level identity consistency

Output:

- entity cluster or pairwise consolidation decision

Current label source:

- `nodes[*].entity_id`
- `labels.coref.entities`

This is the KG node identity stage.

### Task 2: Semantic Linking

Goal:

- decide which graph edges should be created between document nodes or extracted entities.

Input emphasis:

- DOM hierarchy
- reference triggers
- object anchoring
- layout containment and structural neighborhood

Output:

- semantic edge labels such as `refer_to`, `caption_of`, `contains`

Current label source:

- `labels.relations`

This is the KG edge construction stage.

### Task 3: Attribute Canonicalization

Goal:

- convert raw value mentions into canonical attributes suitable for KG storage.

Input emphasis:

- value surface form
- local field context
- unit and schema cues

Output:

- attribute type
- canonical value

Current label source:

- `labels.normalization`
- `nodes[*].norm_type`
- `nodes[*].norm_value`

This is the KG attribute stage.

## What Actually Changes

This redesign does not require inventing fake tasks.

It changes how the tasks are interpreted and reported:

- Task 1 is no longer just "pair classification"; it is entity consolidation for KG nodes.
- Task 2 is no longer just "relation extraction"; it is semantic linking for KG edges.
- Task 3 is no longer just "type classification"; it is attribute canonicalization for KG properties.

## Why This Split Is More Meaningful

These three tasks correspond to different graph-construction objects:

- node identity
- edge structure
- attribute value

That difference is stronger and more honest than saying the tasks differ only because they have different classifier heads.

This also gives a better motivation for task-adaptive graph reasoning:

- entity consolidation needs identity consistency cues
- semantic linking needs structural connectivity cues
- attribute canonicalization needs value-focused local interpretation

The key point is not that each task must see less information.
The key point is that each task may need a different inference bias over the same full graph.

## Recommended Experimental Framing

Keep the current model framing:

- shared full graph
- shared encoder
- task-adaptive graph reasoning

Do not frame the method as hard pruning or reduced visibility.

Instead frame it as:

- shared-information multi-task reasoning with task-adaptive message passing

## Immediate Code Implications

Short-term:

- no label format changes are strictly required
- method text and experiment text should switch to the new task interpretation

Medium-term:

- Task 1 could move from pairwise mention classification toward cluster-aware decoding or entity-level evaluation only
- Task 2 could be expanded to additional KG edge types if needed
- Task 3 should remain a two-stage pipeline:
  - type prediction
  - canonical value generation

## Decision Boundary

If later multi-seed experiments show that task-adaptive routing is not consistently better than `none`, the paper should not force the claim.

In that case the safe contribution stack is:

1. structure-aware document graph construction
2. KG-oriented task formulation
3. attribute canonicalization setup
4. task-adaptive graph reasoning as an exploratory module
