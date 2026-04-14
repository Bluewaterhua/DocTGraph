# Synthetic Contract Dataset

This directory contains the synthetic single-page contract dataset generator used for DocTGraph-style document graph construction experiments.

## Files

- `generate_docs_v2.py`: generate synthetic single-page contract HTML files and task labels.
- `render_bbox_v2.py`: render HTML into page images and export normalized `nodes.json`.
- `config_v2_audit200.json`: small-scale audit configuration for debugging and validation.
- `config_v2_train10k.json`: large-scale configuration for the formal experiment set.
- `template_contract_v2.html`: HTML layout template used by the generator.

## Output Structure

Each generated document folder contains:

- `doc.html`: rendered contract source.
- `image_p1.jpg`: rendered single-page image.
- `nodes.json`: node-level document graph annotations.
- `labels.json`: task labels for coreference, relation extraction, and normalization.
- `meta.json`: generation metadata.

## Node Types

- `block`: ordinary layout-semantic unit such as title, paragraph, caption, footer, or section header.
- `mention`: one textual mention of an entity in the document.
- `ref`: a structural reference trigger such as `表1`, `图1`, or similar pointing expressions.
- `value`: a normalizable value mention such as date, money, phone number, tax number, bank account, or email.
- `object`: a structural object such as a table or figure that can be referenced by text.

## Field Definitions

- `node_id`: unique node identifier.
- `kind`: node type, one of `block`, `mention`, `ref`, `value`, `object`.
- `page_idx`: page index. In the current dataset this is always `1`.
- `bbox`: normalized bounding box in `[x0, y0, x1, y1]`, scaled to `0-1000`.
- `text`: textual content of the node.
- `parent_element_id`: parent layout block ID for local structural context.
- `element_id`: element ID if the node itself is a layout element.
- `mention_id`: mention ID when `kind == "mention"`.
- `value_id`: value ID when `kind == "value"`.
- `object_id`: object ID when `kind == "object"`.
- `ref_id`: reference ID when `kind == "ref"`.
- `rel`: relation type stored on the node itself, mainly used by reference nodes, such as `refer_to`.
- `target_obj`: target object ID pointed to by a reference node.
- `entity_id`: canonical entity ID used for entity alignment and coreference.
- `norm_type`: normalization type, such as `datetime`, `money`, `phone`, `tax_no`, `bank_account`, or `email`.
- `norm_value`: normalized canonical value.
- `tag`: reserved field for future extensions.

## Why Many Fields Are `null`

`nodes.json` uses a unified schema for all node types. This means most fields are type-specific and are only meaningful for some node kinds.

Examples:

- `mention` nodes usually have `mention_id` and `entity_id`, but `value_id`, `object_id`, and `ref_id` are `null`.
- `value` nodes usually have `value_id`, `norm_type`, and `norm_value`, but `mention_id` and `object_id` are `null`.
- `ref` nodes usually have `ref_id`, `rel`, and `target_obj`, but `entity_id` and `norm_value` are `null`.
- `object` nodes usually have `object_id`, while most other task-specific fields are `null`.

This is expected and does not indicate a problem. Validity should be judged together with `kind`.

## Task Mapping

### 1. Entity Alignment and Disambiguation

Primary node type:

- `mention`

Key fields:

- `text`
- `bbox`
- `parent_element_id`
- `entity_id`

Interpretation:

- A `mention` node represents one surface form appearing in the document.
- `entity_id` gives the canonical entity that this mention belongs to.
- Different mention nodes may share the same `entity_id`, which means they refer to the same entity.

Example:

- `甲方`
- `宁岫澈拓系统集成股份有限公司`
- `该公司`

These may all map to the same `entity_id`, such as `entA_000001`.

### 2. Structured Relation Extraction

Primary node types:

- `ref`
- `object`
- `block`

Key fields:

- `ref_id`
- `rel`
- `target_obj`
- `object_id`
- `text`

Interpretation:

- A `ref` node marks where a reference is triggered in text.
- `target_obj` indicates which structural object is being referenced.
- `labels.json` stores the complete relation triples.

Typical supported relations:

- `refer_to`
- `caption_of`

Example:

- In `付款计划如表1所示`, the string `表1` is a `ref` node.
- That node points to a table `object` node through `target_obj`.

### 3. Type Normalization

Primary node type:

- `value`

Key fields:

- `text`
- `value_id`
- `norm_type`
- `norm_value`

Interpretation:

- A `value` node stores one raw value mention in the document.
- `norm_type` indicates the semantic type of the value.
- `norm_value` stores the canonical normalized form.

Examples:

- `2025年5月9日 -> 2025-05-09`
- `650,300.00元 -> CNY:650300.00`
- `178 0451 6026 -> 17804516026`

This is normalization of heterogeneous surface forms into a unified representation, not base conversion.

## labels.json

`labels.json` contains supervision for the three core tasks.

- `coref.entities`
  - entity clusters for entity alignment and coreference.
- `relations`
  - document graph relations such as `refer_to` and `caption_of`.
- `normalization`
  - canonical normalization targets for each value node.

## Design Choice

The current dataset is intentionally restricted to single-page contracts.

Reason:

- single-page layout is much easier to validate and debug;
- relation boundaries are clearer;
- it reduces annotation noise from cross-page references;
- it is sufficient for the current three core tasks:
  - entity alignment and disambiguation,
  - structured relation extraction,
  - type normalization.

The recommended workflow is:

1. validate with `config_v2_audit200.json`
2. inspect generated samples and label consistency
3. scale to `config_v2_train10k.json` for formal experiments

## Current Experiment Status

Current project state as of `2026-03-03`:

- The synthetic contract dataset v2 pipeline has been scaffolded but formal model training has not started yet.
- Data generation and rendering code for the v2 single-page contract setup already exists in this repo.
- LayoutLMv3-side training framework files have also been added under `layoutlmv3/`, but they still need the first actual experiment run.

Current focus:

- first run an audit-scale dataset build to verify end-to-end outputs;
- then launch the first training run on the formal train split;
- only after that start evaluating task behavior and model errors.

Important project areas:

- `synthetic_contract_ds/generate_docs_v2.py`: synthetic HTML document generation and label export.
- `synthetic_contract_ds/render_bbox_v2.py`: HTML-to-image rendering and `nodes.json` export.
- `synthetic_contract_ds/corpus_v2.py`: v2 synthetic content and schema logic.
- `synthetic_contract_ds/gen_v2.py`: generation helpers and orchestration utilities.
- `synthetic_contract_ds/template_contract_v2.html`: contract page template.
- `layoutlmv3/train.py`: training entry point to inspect before starting the first run.
- `layoutlmv3/import_from_hf.py`: Hugging Face import/bootstrap helper.

Git working tree notes observed during recovery:

- there are uncommitted changes in both `synthetic_contract_ds/` and `layoutlmv3/`;
- this means the active work is recoverable from the workspace even if a Codex chat thread cannot be resumed;
- if chat history is lost, re-open the repo and recover context from this file plus `git status`.

## Resume Checklist

If a chat thread is lost or Codex cannot resume the previous session, use this checklist:

1. run `git status` and confirm the current uncommitted files;
2. read this `README.md` first;
3. inspect `synthetic_contract_ds/config_v2_audit200.json` and `synthetic_contract_ds/config_v2_train10k.json`;
4. confirm whether the audit dataset has already been regenerated;
5. inspect `layoutlmv3/train.py` before launching the first training command;
6. record the exact command, config, and output directory used for the next run.

Recommended minimal run log to keep after each experiment:

- date and time;
- command line used;
- config file used;
- output directory;
- dataset version;
- whether the run reached training, validation, or failed during preprocessing.

## Safe Exit And Session Retention

Codex session state and project state are different:

- project state is your real source of truth and is stored in files in this repo;
- chat session state may fail to resume even when login is still valid.

Observed failure mode in PyCharm on `2026-03-03`:

- local authentication was still present;
- the failed step was ACP `session/resume`;
- the server returned `no rollout found for thread id ...`;
- this means a previous chat thread may disappear even though the workspace files are intact.

Recommended way to leave the workspace:

1. finish the current message and wait for Codex to stop generating;
2. save any files you changed in the IDE;
3. append a short status note to this file if you completed a meaningful step;
4. if the change matters, create a git commit or at least verify `git status`;
5. only then close the chat tab or the IDE.

Recommended way to come back later:

1. open the project;
2. do not rely on the old chat tab being resumable;
3. open a new Codex chat if needed;
4. ask Codex to recover context from `synthetic_contract_ds/README.md` and `git status`.

What not to rely on:

- locking the computer and returning later does not guarantee the same chat thread will resume;
- TUN/VPN reconnects, network changes, IDE restarts, plugin restarts, or machine sleep may break thread recovery;
- "still logged in" does not mean "old thread can still be resumed".

Practical rule:

- if you need the work to be recoverable, write the state into the repo, not only into the chat window.
