"""Qianfan-VLM evaluation entry point.

Modes (per codex sign-off, 2026-05-07):

  * ``--diagnose_only``       — cache/graph candidate health, no scoring.
  * ``--mode raw_partial``    — VLM raw output, only coref B^3 + schema EM.
                                Refer/relation NOT scored (gated on objects[]).
  * ``--mode refiner_partial``— VLM-built graph + refiner coref head, only
                                B^3. Schema falls back to VLM raw fields.
  * ``--mode gold_upper_partial`` — gold-node graph + refiner coref head;
                                Gold-node upper bound for B^3.
  * ``--mode raw / refiner / gold_upper`` — full pipeline including
                                refer/relation. NOT YET IMPLEMENTED until
                                the upgraded extractor schema lands.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .extractor.schema import SCHEMA_FIELDS
from .qianfan_pipeline import (
    align_pred_mentions_to_gold_entities,
    align_pred_mentions_to_gold_entities_with_role_resolver,
    build_gold_role_map,
    filter_entity_like_mentions,
    graph_health,
    load_vlm_cache,
    parsed_health,
    raw_predictions,
    refiner_predictions_partial,
    schema_eq_semantic,
    schema_eq_strict,
    tax_no_diagnostics,
    vlm_to_graph,
)


# ---------------------------------------------------------------------------
# Doc enumeration / page count
# ---------------------------------------------------------------------------


def _list_doc_dirs(dataset: Path, limit: Optional[int]) -> List[Path]:
    docs = sorted(p for p in dataset.iterdir()
                  if p.is_dir() and p.name.startswith("doc_"))
    if limit is not None:
        docs = docs[:limit]
    return docs


def _page_count(doc_dir: Path) -> int:
    nodes_path = doc_dir / "nodes.json"
    if nodes_path.exists():
        try:
            data = json.loads(nodes_path.read_text(encoding="utf-8"))
            return int(data.get("document", {}).get("page_count")
                       or len(list(doc_dir.glob("image_p*.jpg"))) or 1)
        except json.JSONDecodeError:
            pass
    return max(len(list(doc_dir.glob("image_p*.jpg"))), 1)


# ---------------------------------------------------------------------------
# Diagnose mode
# ---------------------------------------------------------------------------


def _aggregate(per_doc: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_docs = len(per_doc)
    n_cache_missing = sum(1 for d in per_doc if d.get("_cache_missing"))
    n_cache_error = sum(1 for d in per_doc if d.get("parsed", {}).get("_error"))
    keys_to_sum = (
        "n_fields", "n_mentions", "n_values", "n_refs", "n_objects",
        "bbox_bad_total", "bbox_missing_total", "bbox_placeholder_total",
        "bbox_total",
        "refs_with_target_page", "refs_with_target_anchor",
    )
    sums = {k: 0 for k in keys_to_sum}
    n_ok = 0
    for d in per_doc:
        ph = d.get("parsed_health") or {}
        if ph.get("_error"):
            continue
        n_ok += 1
        for k in keys_to_sum:
            sums[k] += int(ph.get(k) or 0)

    graph_keys = (
        "n_nodes", "n_edges", "n_rel_candidate_edges",
        "n_ref_candidate_edges", "n_caption_candidate_edges",
        "n_contains_candidate_edges",
    )
    g_sums = {k: 0 for k in graph_keys}
    n_zero_rel_cand = 0
    n_zero_mention = 0
    for d in per_doc:
        gh = d.get("graph_health") or {}
        if gh.get("_error"):
            continue
        for k in graph_keys:
            g_sums[k] += int(gh.get(k) or 0)
        if gh.get("n_rel_candidate_edges", 0) == 0:
            n_zero_rel_cand += 1
        node_counts = gh.get("node_kind_counts") or {}
        if not node_counts.get("mention"):
            n_zero_mention += 1

    def _mean(total: int) -> Optional[float]:
        return (total / n_ok) if n_ok else None

    placeholder_rate = (
        sums["bbox_placeholder_total"] / sums["bbox_total"]
        if sums["bbox_total"] else None
    )
    missing_rate = (
        sums["bbox_missing_total"] / sums["bbox_total"]
        if sums["bbox_total"] else None
    )
    bbox_bad_rate = (
        sums["bbox_bad_total"] / sums["bbox_total"]
        if sums["bbox_total"] else None
    )
    refs_target_rate = (
        sums["refs_with_target_page"] / sums["n_refs"]
        if sums["n_refs"] else None
    )
    refs_anchor_rate = (
        sums["refs_with_target_anchor"] / sums["n_refs"]
        if sums["n_refs"] else None
    )

    # ---- gating: what can the evaluator actually score? ----
    has_objects = sums["n_objects"] > 0
    has_mentions = sums["n_mentions"] > 0
    has_rel_candidates = g_sums["n_rel_candidate_edges"] > 0
    has_fields = sums["n_fields"] > 0

    can_eval_coref = has_mentions
    can_eval_schema = has_fields
    can_eval_refer = has_objects and has_rel_candidates

    reasons: List[str] = []
    if not has_objects:
        reasons.append("no object nodes")
    if not has_rel_candidates:
        reasons.append("ref_candidate_edges=0")
    if not has_mentions:
        reasons.append("no mentions")
    if not has_fields:
        reasons.append("no fields")

    return {
        "n_docs": n_docs,
        "n_ok": n_ok,
        "n_cache_missing": n_cache_missing,
        "n_cache_error": n_cache_error,
        "n_docs_with_zero_rel_candidate": n_zero_rel_cand,
        "n_docs_with_zero_mentions": n_zero_mention,
        "totals": {**sums, **g_sums},
        "means_per_doc": {
            "n_fields": _mean(sums["n_fields"]),
            "n_mentions": _mean(sums["n_mentions"]),
            "n_values": _mean(sums["n_values"]),
            "n_refs": _mean(sums["n_refs"]),
            "n_objects": _mean(sums["n_objects"]),
            "n_rel_candidate_edges": _mean(g_sums["n_rel_candidate_edges"]),
            "n_ref_candidate_edges": _mean(g_sums["n_ref_candidate_edges"]),
        },
        "bbox_bad_rate": bbox_bad_rate,
        "bbox_missing_rate": missing_rate,
        "bbox_placeholder_rate": placeholder_rate,
        # legacy alias kept for any downstream readers; equals bad_rate.
        "placeholder_bbox_rate": bbox_bad_rate,
        "refs_with_target_page_rate": refs_target_rate,
        "refs_with_target_anchor_rate": refs_anchor_rate,
        "candidate_health": {
            "can_eval_coref": can_eval_coref,
            "can_eval_schema": can_eval_schema,
            "can_eval_refer": can_eval_refer,
            "reasons": reasons,
        },
    }


def diagnose(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"[diagnose] dataset not found: {dataset}", file=sys.stderr)
        return 2

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"[diagnose] cache_dir not found: {cache_dir}", file=sys.stderr)
        return 2

    docs = _list_doc_dirs(dataset, args.limit_docs)
    if not docs:
        print(f"[diagnose] no doc_* under {dataset}", file=sys.stderr)
        return 2

    per_doc: List[Dict[str, Any]] = []
    for i, d in enumerate(docs):
        parsed = load_vlm_cache(cache_dir, d.name, args.backend_name)
        record: Dict[str, Any] = {"doc_id": d.name}
        if parsed is None:
            record["_cache_missing"] = True
            record["parsed_health"] = {"_error": "missing"}
            record["graph_health"] = {"_error": "no_graph"}
            per_doc.append(record)
            continue
        record["parsed"] = {
            "_error": parsed.get("_error"),
            "_truncated": parsed.get("_truncated", False),
        }
        record["parsed_health"] = parsed_health(parsed)
        page_count = _page_count(d)
        graph = vlm_to_graph(parsed, page_count, d.name)
        record["graph_health"] = graph_health(graph) if graph is not None \
            else {"_error": "no_graph"}
        per_doc.append(record)
        if (i + 1) % 25 == 0:
            print(f"[diagnose] {i+1}/{len(docs)}", flush=True)

    summary = _aggregate(per_doc)
    out = {
        "dataset": str(dataset),
        "cache_dir": str(cache_dir),
        "backend_name": args.backend_name,
        "summary": summary,
        "per_doc": per_doc,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    # ---- console summary ----
    s = summary
    ch = s["candidate_health"]
    print("\n=== Qianfan diagnose ===")
    print(f"  n_docs={s['n_docs']}  ok={s['n_ok']}  "
          f"missing={s['n_cache_missing']}  cache_error={s['n_cache_error']}")
    print(f"  totals: {s['totals']}")
    print(f"  means/doc: {s['means_per_doc']}")
    print(f"  bbox_missing_rate={s['bbox_missing_rate']}  "
          f"bbox_placeholder_rate={s['bbox_placeholder_rate']}  "
          f"(bbox_bad_rate={s['bbox_bad_rate']})")
    print(f"  refs_with_target_page_rate={s['refs_with_target_page_rate']}")
    print(f"  refs_with_target_anchor_rate={s['refs_with_target_anchor_rate']}")
    print(f"  n_docs_with_zero_rel_candidate={s['n_docs_with_zero_rel_candidate']}")
    print(f"  n_docs_with_zero_mentions={s['n_docs_with_zero_mentions']}")
    print(f"  candidate_health: coref={ch['can_eval_coref']}  "
          f"schema={ch['can_eval_schema']}  refer={ch['can_eval_refer']}")
    if ch["reasons"]:
        print(f"  reasons: {ch['reasons']}")
    if not ch["can_eval_refer"]:
        print(
            "  [gate] object extraction missing; "
            "relation/refiner evaluation is not valid yet. "
            "Upgrade extractor schema (objects[]) before running --mode refiner.",
            flush=True,
        )
    print(f"\nFull report -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# Stubs for raw / refiner / gold_upper (full pipeline)
# ---------------------------------------------------------------------------


def _stub(mode: str) -> int:
    print(
        f"[FATAL] --mode {mode} is not implemented yet. "
        f"Run --diagnose_only first; then implement the mode after the "
        f"VLM extractor schema upgrade is signed off.",
        file=sys.stderr,
    )
    return 3


# ---------------------------------------------------------------------------
# Partial evaluators: coref B^3 + schema EM
# ---------------------------------------------------------------------------


def _gold_mentions_for_doc(doc_dir: Path) -> Tuple[
        List[Dict[str, Any]],
        List[List[int]],
        List[Optional[str]],
        List[Dict[str, Any]],
]:
    """Return (mentions, clusters_in_mention_pos, entity_id_per_pos, entities).

    * ``mentions[i]``  = {node_id, text, page_idx}
    * ``clusters``     = list of mention-position lists; singletons appended
                          for mentions not in any entity (kept for legacy
                          callers; entity-level B³ uses entity_id_per_pos).
    * ``entity_id_per_pos[i]`` = entity_id of mention i, or None if it
                          isn't part of any gold entity (so the entity-
                          level aligner can ignore it).
    * ``entities`` = labels.coref.entities list (with canonical etc.) so
                          the aligner can match pred mentions to canonicals.
    """
    nodes_json = json.loads((doc_dir / "nodes.json").read_text(encoding="utf-8"))
    labels = json.loads((doc_dir / "labels.json").read_text(encoding="utf-8"))
    mentions: List[Dict[str, Any]] = []
    node_id_to_pos: Dict[str, int] = {}
    for n in nodes_json["nodes"]:
        if n.get("kind") != "mention":
            continue
        node_id_to_pos[n["node_id"]] = len(mentions)
        mentions.append({
            "node_id": n["node_id"],
            "text": n.get("text", ""),
            "page_idx": int(n.get("page_idx", 1)),
        })
    grouped: set = set()
    clusters: List[List[int]] = []
    entity_id_per_pos: List[Optional[str]] = [None] * len(mentions)
    entities = (labels.get("coref") or {}).get("entities") or []
    for entity in entities:
        members: List[int] = []
        for mid in entity["mentions"]:
            pos = node_id_to_pos.get(mid)
            if pos is not None:
                members.append(pos)
                grouped.add(pos)
                entity_id_per_pos[pos] = entity.get("entity_id")
        if members:
            clusters.append(members)
    for pos in range(len(mentions)):
        if pos not in grouped:
            clusters.append([pos])
    return mentions, clusters, entity_id_per_pos, entities


def _gold_schema_for_doc(doc_dir: Path) -> Dict[str, Optional[str]]:
    """Map schema_field -> normalised gold value (or None if absent).

    * Most fields: gold ``v_<field>`` value node (norm_value or text).
      ``v_bank_acct`` aliases ``bank_account``.
    * partyA / partyB (codex 2026-05-08 fix): there is no ``v_partyA``
      node in the synthetic v3 dataset. Read from ``labels.json``
      ``coref.entities``: prefer the entity whose ``entity_id`` starts
      with ``entA``/``entB`` (matching the renderer's convention), with
      positional fallback (entities[0]/[1]) and a final fallback to the
      mention text of ``m_partyA_cover``/``m_partyB_cover``. Without
      this fix, partyA/partyB were always counted as gold-null and
      every Qianfan answer became a false-mismatch.
    """
    nodes_json = json.loads((doc_dir / "nodes.json").read_text(encoding="utf-8"))
    out: Dict[str, Optional[str]] = {f: None for f in SCHEMA_FIELDS}
    mention_text_by_id: Dict[str, str] = {}
    for n in nodes_json["nodes"]:
        nid = n.get("node_id", "")
        if n.get("kind") == "mention":
            mention_text_by_id[nid] = n.get("text", "")
        if not nid.startswith("v_"):
            continue
        field = nid[2:]
        if field == "bank_acct":
            field = "bank_account"
        if field in out and out[field] is None:
            out[field] = n.get("norm_value") or n.get("text")

    if "partyA" in out or "partyB" in out:
        labels_path = doc_dir / "labels.json"
        if labels_path.exists():
            try:
                labels = json.loads(labels_path.read_text(encoding="utf-8"))
                entities = (labels.get("coref") or {}).get("entities") or []
            except json.JSONDecodeError:
                entities = []
            ent_a: Optional[Dict[str, Any]] = None
            ent_b: Optional[Dict[str, Any]] = None
            for e in entities:
                eid = e.get("entity_id", "")
                if eid.startswith("entA") and ent_a is None:
                    ent_a = e
                elif eid.startswith("entB") and ent_b is None:
                    ent_b = e
            if ent_a is None and entities:
                ent_a = entities[0]
            if ent_b is None and len(entities) >= 2:
                ent_b = entities[1]
            if out.get("partyA") is None and ent_a is not None:
                out["partyA"] = (
                    ent_a.get("canonical")
                    or mention_text_by_id.get("m_partyA_cover")
                    or None
                )
            if out.get("partyB") is None and ent_b is not None:
                out["partyB"] = (
                    ent_b.get("canonical")
                    or mention_text_by_id.get("m_partyB_cover")
                    or None
                )
        # Final fallback to cover mention text if labels.json missing.
        if out.get("partyA") is None:
            out["partyA"] = mention_text_by_id.get("m_partyA_cover")
        if out.get("partyB") is None:
            out["partyB"] = mention_text_by_id.get("m_partyB_cover")
    return out


def _pred_cluster_id_by_parsed_pos(
    pred_clusters: List[List[int]],
) -> Dict[int, int]:
    """Map parsed['mentions'] index -> cluster id.

    Works for both ``raw_predictions`` (clusters indexed over parsed
    mentions directly) and ``refiner_predictions_partial`` (clusters
    indexed over the filtered mention_idx, which is built in the same
    order as parsed['mentions'], so the k-th refiner cluster index ==
    k-th parsed mention).
    """
    out: Dict[int, int] = {}
    for cid, members in enumerate(pred_clusters):
        for m in members:
            out[m] = cid
    return out


def _entity_level_pred_assignment(
    gold_mention_to_entity: List[Optional[str]],
    pred_entity_like: List[Dict[str, Any]],
    pred_to_gold_entity: List[Optional[str]],
    pred_cluster_id_by_parsed_pos: Dict[int, int],
) -> List[Optional[int]]:
    """For each gold mention pos, the pred cluster id implied by the alignment.

    Codex 2026-05-09 finding: previous majority-vote algorithm collapsed
    fragmented pred clusters — every gold mention of an entity inherited
    the *single* most-voted pred cluster id, so K pred singletons all
    aligned to one gold entity scored as one perfect cluster (fake B^3
    F1 = 1.0). Replaced with **insertion-order 1-to-1 pairing**:

      * Build per-entity pred queue: pred mentions aligned to entity E,
        in pred-mention insertion order, carrying their cluster ids.
      * Walk gold mentions in gold-insertion order. For each gold mention
        with entity E, pop the head of E's queue and inherit that cluster
        id. Empty queue -> None (becomes a unique singleton in B^3).
      * Pred mentions left over (more pred than gold for an entity) are
        silently ignored — they're outside the gold universe so B^3 does
        not see them directly; they only matter if they appear in some
        OTHER entity's gold cluster, which they cannot if the aligner is
        consistent.

    Behavior on the 4 canonical scenarios:

      * 5 gold + 5 pred all in cluster 0:    all 5 gold inherit cluster 0
                                              -> B^3 = 1.0 (correct, perfect merge)
      * 5 gold + 5 pred in 5 distinct clusters: 5 gold get 5 distinct
                                              cluster ids -> B^3 punishes
                                              fragmentation (correct)
      * 5 gold + 1 pred in cluster 0:         1 gold gets cluster 0; 4
                                              gold get None (singletons)
                                              -> B^3 ~0.2 recall (correct)
      * 1 gold + 5 pred in cluster 0:         1 gold gets cluster 0; 4
                                              extra pred ignored
                                              -> B^3 = 1.0 (correct)
    """
    # Per-entity FIFO of pred cluster ids in pred insertion order.
    entity_to_pred_queue: Dict[str, List[int]] = {}
    for k, eid in enumerate(pred_to_gold_entity):
        if eid is None:
            continue
        parsed_idx = pred_entity_like[k]["idx"]
        cluster_id = pred_cluster_id_by_parsed_pos.get(parsed_idx)
        if cluster_id is None:
            continue
        entity_to_pred_queue.setdefault(eid, []).append(cluster_id)

    out: List[Optional[int]] = []
    head: Dict[str, int] = {}  # next index to pop per entity
    for eid in gold_mention_to_entity:
        if eid is None:
            out.append(None)
            continue
        queue = entity_to_pred_queue.get(eid)
        if not queue:
            out.append(None)
            continue
        i = head.get(eid, 0)
        if i >= len(queue):
            out.append(None)
        else:
            out.append(queue[i])
            head[eid] = i + 1
    return out


def _b3_aligned(gold_clusters: List[List[int]],
                pred_cluster_id_by_gold_pos: List[Optional[int]],
                ) -> Dict[str, Any]:
    """B^3 over the gold-mention positions.

    A gold mention with no predicted alignment is assigned a unique
    singleton (uuid-style int counter) so it neither merges nor splits
    other entities artificially.
    """
    n = sum(len(c) for c in gold_clusters)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_mentions": 0,
                "n_aligned": 0, "n_unaligned": 0}
    # Build idx -> gold cluster set
    idx_to_gold: Dict[int, set] = {}
    for members in gold_clusters:
        s = set(members)
        for m in members:
            idx_to_gold[m] = s
    # Build idx -> pred cluster set, using fresh int ids for unaligned.
    pred_id_to_members: Dict[Any, List[int]] = {}
    fresh = -1
    n_unaligned = 0
    for pos in range(n):
        cid = pred_cluster_id_by_gold_pos[pos]
        if cid is None:
            cid = fresh
            fresh -= 1
            n_unaligned += 1
        pred_id_to_members.setdefault(cid, []).append(pos)
    idx_to_pred: Dict[int, set] = {}
    for members in pred_id_to_members.values():
        s = set(members)
        for m in members:
            idx_to_pred[m] = s
    p_sum = 0.0
    r_sum = 0.0
    for k in range(n):
        P = idx_to_pred[k]
        G = idx_to_gold[k]
        inter = len(P & G)
        p_sum += inter / max(len(P), 1)
        r_sum += inter / max(len(G), 1)
    P = p_sum / n
    R = r_sum / n
    F = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
    return {
        "precision": P, "recall": R, "f1": F,
        "n_mentions": n,
        "n_aligned": n - n_unaligned,
        "n_unaligned": n_unaligned,
    }


def _schema_em(gold: Dict[str, Optional[str]],
               pred: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """Dual schema EM: strict (headline) + semantic (diagnostic).

    Codex 2026-05-08 spec:
      * strict   = field-aware deterministic normalize + equality. The
                   number that goes into the paper main table.
      * semantic = looser equivalence on date/money (relative/absolute
                   tolerance per field); other fields fall back to
                   strict so we never silently flatter Qianfan.
      * diagnostics.tax_no holds confusable_match + edit ratio per doc;
                   NOT used to relax EM, surfaced so we can later
                   decide whether OCR is the dominant error mode.
    """
    n_field = len(gold)
    n_match_strict = 0
    n_match_semantic = 0
    n_pred = 0
    n_present = 0
    per_field: Dict[str, Dict[str, Any]] = {}
    diagnostics: Dict[str, Any] = {}
    for f, gv in gold.items():
        pv = pred.get(f)
        match_s = schema_eq_strict(f, gv, pv)
        match_sem = schema_eq_semantic(f, gv, pv)
        per_field[f] = {
            "gold": gv,
            "pred": pv,
            "strict_match": match_s,
            "semantic_match": match_sem,
        }
        if gv:
            n_present += 1
        if pv is not None and pv != "":
            n_pred += 1
        if match_s:
            n_match_strict += 1
        if match_sem:
            n_match_semantic += 1
        if f == "tax_no":
            diagnostics["tax_no"] = tax_no_diagnostics(gv, pv)
    return {
        "strict": {
            "n_fields": n_field,
            "n_present_gold": n_present,
            "n_pred_nonempty": n_pred,
            "n_exact_match": n_match_strict,
            "em_rate": n_match_strict / max(n_field, 1),
            "em_rate_present_only":
                n_match_strict / max(n_present, 1),
        },
        "semantic": {
            "n_fields": n_field,
            "n_present_gold": n_present,
            "n_pred_nonempty": n_pred,
            "n_exact_match": n_match_semantic,
            "em_rate": n_match_semantic / max(n_field, 1),
            "em_rate_present_only":
                n_match_semantic / max(n_present, 1),
        },
        "per_field": per_field,
        "diagnostics": diagnostics,
    }


def _b3_triple(
    gold_clusters: List[List[int]],
    pred_assignment: List[Optional[int]],
    pred_entity_like: List[Dict[str, Any]],
    pred_to_gold_entity: List[Optional[str]],
) -> Dict[str, Any]:
    """Triple-flavor B^3 reporting (codex 2026-05-08 spec).

      * ``main``         -- unaligned gold mentions become unique
                            singletons. THIS is the headline B^3 that
                            goes into the paper main table.
      * ``aligned_only`` -- restrict gold to mentions that have an
                            entity-aligned pred. Diagnostic only:
                            exposes how the refiner does on the subset
                            it can score, but excluding missed mentions
                            silently flatters recall, so we tag it.
      * ``alignment``    -- gold/pred coverage stats so a reader can
                            tell whether ``aligned_only`` was computed
                            on a tiny slice of the doc.
    """
    n_total = sum(len(c) for c in gold_clusters)
    main = _b3_aligned(gold_clusters, pred_assignment)

    aligned_pos_set = {
        pos for pos, cid in enumerate(pred_assignment) if cid is not None
    }
    if aligned_pos_set:
        # Compress gold -> aligned-only positions, preserving cluster structure.
        old_to_new: Dict[int, int] = {}
        for cluster in gold_clusters:
            for old_pos in cluster:
                if old_pos in aligned_pos_set and old_pos not in old_to_new:
                    old_to_new[old_pos] = len(old_to_new)
        new_clusters: List[List[int]] = []
        for cluster in gold_clusters:
            new_cluster = [old_to_new[p] for p in cluster if p in old_to_new]
            if new_cluster:
                new_clusters.append(new_cluster)
        ordered_old = sorted(old_to_new, key=old_to_new.get)
        new_pred = [pred_assignment[op] for op in ordered_old]
        aligned_only = _b3_aligned(new_clusters, new_pred)
    else:
        aligned_only = {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "n_mentions": 0, "n_aligned": 0, "n_unaligned": 0,
        }
    aligned_only["_diagnostic_only"] = (
        "excludes gold mentions Qianfan didn't extract; not for paper main table"
    )

    n_pred_total = len(pred_entity_like)
    n_pred_aligned = sum(1 for e in pred_to_gold_entity if e is not None)
    return {
        "main": main,
        "aligned_only": aligned_only,
        "alignment": {
            "n_gold_total": n_total,
            "n_gold_aligned": len(aligned_pos_set),
            "n_gold_unaligned": n_total - len(aligned_pos_set),
            "gold_alignment_rate":
                (len(aligned_pos_set) / n_total) if n_total else None,
            "n_pred_entity_like": n_pred_total,
            "n_pred_aligned": n_pred_aligned,
            "pred_alignment_rate":
                (n_pred_aligned / n_pred_total) if n_pred_total else None,
        },
    }


def _aggregate_partial(per_doc: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Doc-level aggregation for partial mode.

    Reports B^3 mention-weighted under both "main" (with-singletons) and
    "aligned_only" (diagnostic) per codex 2026-05-08, alongside dual
    schema EM (strict headline + semantic diagnostic).
    """
    n_docs = len(per_doc)
    n_skipped = 0

    # B^3 main + aligned_only (mention-weighted)
    main_p = main_r = 0.0
    main_n = 0
    aln_p = aln_r = 0.0
    aln_n = 0
    n_gold_total = 0
    n_gold_aligned = 0
    n_pred_entity_like_total = 0
    n_pred_aligned_total = 0

    # Schema strict + semantic
    n_match_strict = 0
    n_match_semantic = 0
    n_field_total = 0
    n_present_total = 0
    n_pred_nonempty = 0
    n_docs_schema_scored = 0

    for d in per_doc:
        if d.get("_skip"):
            n_skipped += 1
            continue
        b3 = d.get("b3") or {}
        m = b3.get("main") or {}
        a = b3.get("aligned_only") or {}
        align = b3.get("alignment") or {}
        nm = m.get("n_mentions", 0)
        main_p += m.get("precision", 0.0) * nm
        main_r += m.get("recall", 0.0) * nm
        main_n += nm
        nma = a.get("n_mentions", 0)
        aln_p += a.get("precision", 0.0) * nma
        aln_r += a.get("recall", 0.0) * nma
        aln_n += nma
        n_gold_total += align.get("n_gold_total", 0)
        n_gold_aligned += align.get("n_gold_aligned", 0)
        n_pred_entity_like_total += align.get("n_pred_entity_like", 0)
        n_pred_aligned_total += align.get("n_pred_aligned", 0)

        # Codex 2026-05-09 fix: per-doc schema records may carry
        # _not_scored=True (gold_upper). Skip those from EM accumulation
        # so summary doesn't dilute "0 matches over 0 fields" into "0%".
        s = d.get("schema") or {}
        if s.get("_not_scored"):
            continue
        n_docs_schema_scored += 1
        s_strict = s.get("strict") or {}
        s_semantic = s.get("semantic") or {}
        n_match_strict += s_strict.get("n_exact_match", 0)
        n_match_semantic += s_semantic.get("n_exact_match", 0)
        n_field_total += s_strict.get("n_fields", 0)
        n_present_total += s_strict.get("n_present_gold", 0)
        n_pred_nonempty += s_strict.get("n_pred_nonempty", 0)

    def _f1(p: float, r: float) -> float:
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    main_P = main_p / max(main_n, 1)
    main_R = main_r / max(main_n, 1)
    aln_P = aln_p / max(aln_n, 1)
    aln_R = aln_r / max(aln_n, 1)

    if n_docs_schema_scored == 0:
        # Codex 2026-05-09 fix: gold_upper-only run -> no schema scored.
        # Don't synthesize a 0% number; mark explicitly.
        schema_summary: Dict[str, Any] = {
            "_not_scored": True,
            "reason": "no doc had schema scored (e.g. gold_upper_partial)",
            "n_docs_schema_scored": 0,
        }
    else:
        schema_summary = {
            "n_docs_schema_scored": n_docs_schema_scored,
            "strict": {
                "em_rate": n_match_strict / max(n_field_total, 1),
                "em_rate_present_only":
                    n_match_strict / max(n_present_total, 1),
                "n_fields_total": n_field_total,
                "n_present_gold": n_present_total,
                "n_pred_nonempty": n_pred_nonempty,
                "n_exact_match": n_match_strict,
            },
            "semantic": {
                "em_rate": n_match_semantic / max(n_field_total, 1),
                "em_rate_present_only":
                    n_match_semantic / max(n_present_total, 1),
                "n_fields_total": n_field_total,
                "n_present_gold": n_present_total,
                "n_pred_nonempty": n_pred_nonempty,
                "n_exact_match": n_match_semantic,
                "_diagnostic_only":
                    "looser equivalence (date/money tolerant); "
                    "headline metric is strict",
            },
        }

    return {
        "n_docs": n_docs,
        "n_skipped": n_skipped,
        "b3_mention_weighted": {
            "main": {
                "precision": main_P, "recall": main_R,
                "f1": _f1(main_P, main_R),
                "n_mentions": main_n,
            },
            "aligned_only": {
                "precision": aln_P, "recall": aln_R,
                "f1": _f1(aln_P, aln_R),
                "n_mentions": aln_n,
                "_diagnostic_only":
                    "excludes gold mentions Qianfan didn't extract; "
                    "not for paper main table",
            },
            "alignment": {
                "n_gold_total": n_gold_total,
                "n_gold_aligned": n_gold_aligned,
                "n_gold_unaligned": n_gold_total - n_gold_aligned,
                "gold_alignment_rate":
                    (n_gold_aligned / n_gold_total)
                    if n_gold_total else None,
                "n_pred_entity_like": n_pred_entity_like_total,
                "n_pred_aligned": n_pred_aligned_total,
                "pred_alignment_rate":
                    (n_pred_aligned_total / n_pred_entity_like_total)
                    if n_pred_entity_like_total else None,
            },
        },
        "schema": schema_summary,
    }


def run_partial(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    cache_dir = Path(args.cache_dir)
    if not dataset.exists():
        print(f"[partial] dataset not found: {dataset}", file=sys.stderr)
        return 2
    # gold_upper does not consume VLM cache (codex 2026-05-09 fix); other
    # modes still require it.
    if args.mode != "gold_upper_partial" and not cache_dir.exists():
        print(f"[partial] cache_dir not found: {cache_dir}", file=sys.stderr)
        return 2
    docs = _list_doc_dirs(dataset, args.limit_docs)
    if not docs:
        print(f"[partial] no doc_* under {dataset}", file=sys.stderr)
        return 2

    # ---- load refiner if needed ----
    model = None
    device = None
    if args.mode in ("refiner_partial", "gold_upper_partial"):
        if args.refiner_ckpt is None:
            print("[FATAL] --refiner_ckpt required for "
                  f"--mode {args.mode}", file=sys.stderr)
            return 2
        from .refiner.model import DocKGRefiner
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = DocKGRefiner()
        state = torch.load(args.refiner_ckpt, map_location=device,
                           weights_only=False)
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            print(f"[FATAL] failed to load refiner ckpt: {exc}",
                  file=sys.stderr)
            return 2
        model.to(device)
        model.eval()

    per_doc: List[Dict[str, Any]] = []
    threshold = args.threshold
    for i, d in enumerate(docs):
        record: Dict[str, Any] = {"doc_id": d.name, "threshold": threshold}
        try:
            (gold_mentions, gold_clusters,
             gold_mention_to_entity, gold_entities) = _gold_mentions_for_doc(d)
            gold_schema = _gold_schema_for_doc(d)
        except Exception as exc:  # noqa: BLE001
            record["_skip"] = True
            record["_error"] = f"gold_load_failed: {exc!r}"
            per_doc.append(record)
            continue

        # gold_upper does NOT consume VLM cache (codex 2026-05-09 fix);
        # only raw_partial / refiner_partial need pred mentions from
        # Qianfan output.
        parsed: Optional[Dict[str, Any]] = None
        if args.mode in ("raw_partial", "refiner_partial"):
            parsed = load_vlm_cache(cache_dir, d.name, args.backend_name)
            if parsed is None or parsed.get("_error"):
                record["_skip"] = True
                record["_error"] = "vlm_cache_missing_or_error"
                per_doc.append(record)
                continue

        if args.mode == "raw_partial":
            pred = raw_predictions(parsed)
            pred_clusters = pred["coref_clusters"]
            schema_pred = pred["schema"]
        elif args.mode == "refiner_partial":
            page_count = _page_count(d)
            graph = vlm_to_graph(parsed, page_count, d.name)
            if graph is None:
                record["_skip"] = True
                record["_error"] = "vlm_to_graph_failed"
                per_doc.append(record)
                continue
            r = refiner_predictions_partial(model, graph, device, threshold)
            # refiner_predictions_partial returns clusters indexed over the
            # filtered mention_idx, which preserves parsed['mentions'] order
            # because vlm_to_graph adds mention nodes in input order. So a
            # cluster member k maps to parsed['mentions'][k] directly.
            pred_clusters = r["coref_clusters"]
            # Schema falls back to raw VLM (--disable_schema_refine)
            schema_pred = raw_predictions(parsed)["schema"]
        else:  # gold_upper_partial
            from .coref_full_eval import coref_full_eval
            single = coref_full_eval(
                model, [d], device, threshold=threshold,
            )
            pd0 = single["per_doc"][0] if single.get("per_doc") else None
            if pd0 is None:
                record["_skip"] = True
                record["_error"] = "no_mentions_in_gold"
                per_doc.append(record)
                continue
            # Gold-upper uses gold mentions directly -> all aligned by definition.
            n_g = pd0["n_mentions"]
            record["b3"] = {
                "main": {
                    "precision": pd0["b3_precision"],
                    "recall":    pd0["b3_recall"],
                    "f1":        pd0["b3_f1"],
                    "n_mentions": n_g,
                },
                "aligned_only": {
                    "precision": pd0["b3_precision"],
                    "recall":    pd0["b3_recall"],
                    "f1":        pd0["b3_f1"],
                    "n_mentions": n_g,
                    "_diagnostic_only":
                        "trivially equal to main in gold_upper mode "
                        "(no Qianfan extraction step)",
                },
                "alignment": {
                    "n_gold_total": n_g,
                    "n_gold_aligned": n_g,
                    "n_gold_unaligned": 0,
                    "gold_alignment_rate": 1.0 if n_g else None,
                    "n_pred_entity_like": n_g,
                    "n_pred_aligned": n_g,
                    "pred_alignment_rate": 1.0 if n_g else None,
                },
            }
            # Codex 2026-05-09 fix: gold_upper does not score schema.
            # Emitting a real EM=0 (with empty pred) was misleading;
            # mark explicitly as not_scored so console / aggregator
            # distinguish "0 of N" from "n/a".
            record["schema"] = {
                "_not_scored": True,
                "reason": "gold_upper_partial does not score schema",
            }
            per_doc.append(record)
            if (i + 1) % 10 == 0:
                print(f"[gold_upper_partial] {i+1}/{len(docs)}", flush=True)
            continue

        # ---- entity-level alignment for raw_partial / refiner_partial ----
        pred_entity_like = filter_entity_like_mentions(parsed)
        if args.use_role_resolver:
            gold_role_map = build_gold_role_map(gold_entities)
            pred_to_gold_entity, _align_info = (
                align_pred_mentions_to_gold_entities_with_role_resolver(
                    pred_entity_like, gold_entities, gold_role_map,
                )
            )
        else:
            pred_to_gold_entity, _align_info = align_pred_mentions_to_gold_entities(
                pred_entity_like, gold_entities,
            )
        pred_cluster_by_pos = _pred_cluster_id_by_parsed_pos(pred_clusters)
        pred_assignment = _entity_level_pred_assignment(
            gold_mention_to_entity,
            pred_entity_like,
            pred_to_gold_entity,
            pred_cluster_by_pos,
        )
        record["b3"] = _b3_triple(
            gold_clusters, pred_assignment,
            pred_entity_like, pred_to_gold_entity,
        )
        record["schema"] = _schema_em(gold_schema, schema_pred)
        # Per-doc breakdown so codex can sanity-check the aligner.
        record["pred_entity_like_debug"] = {
            "n_all_mentions": len(parsed.get("mentions") or []),
            "n_entity_like": len(pred_entity_like),
            "use_role_resolver": bool(args.use_role_resolver),
            "alignment_info": _align_info,
            "alignments": [
                {
                    "text": pm["text"],
                    "entity_hint": pm["entity_hint"],
                    "gold_entity_id": pred_to_gold_entity[k],
                }
                for k, pm in enumerate(pred_entity_like)
            ],
        }
        per_doc.append(record)
        if (i + 1) % 10 == 0:
            print(f"[{args.mode}] {i+1}/{len(docs)}", flush=True)

    summary = _aggregate_partial(per_doc)
    out = {
        "mode": args.mode,
        "dataset": str(dataset),
        "cache_dir": str(cache_dir),
        "backend_name": args.backend_name,
        "threshold": threshold,
        "summary": summary,
        "per_doc": per_doc,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    s = summary
    print(f"\n=== Qianfan {args.mode} ===")
    b3 = s["b3_mention_weighted"]
    main = b3["main"]
    aln = b3["aligned_only"]
    align = b3["alignment"]
    sc = s["schema"]
    print(f"  n_docs={s['n_docs']} skipped={s['n_skipped']}")
    print(f"  B^3 main:         F1={main['f1']:.4f}  "
          f"P={main['precision']:.3f} R={main['recall']:.3f}  "
          f"n_mentions={main['n_mentions']}")
    print(f"  B^3 aligned_only: F1={aln['f1']:.4f}  "
          f"P={aln['precision']:.3f} R={aln['recall']:.3f}  "
          f"n_mentions={aln['n_mentions']}  [diagnostic]")
    print(f"  Alignment: gold {align['n_gold_aligned']}/{align['n_gold_total']} "
          f"({align['gold_alignment_rate']})  "
          f"pred entity-like {align['n_pred_aligned']}/{align['n_pred_entity_like']} "
          f"({align['pred_alignment_rate']})")
    if sc.get("_not_scored"):
        # Codex 2026-05-09 fix: distinguish "not measured" from "0%".
        print(f"  Schema: not_scored ({sc.get('reason', 'gold_upper mode')})")
        sc_sem = None  # silence the second print
    else:
        sc_strict = sc["strict"]
        sc_sem = sc["semantic"]
        print(f"  Schema strict:    EM={sc_strict['em_rate']:.4f}  "
              f"EM(present)={sc_strict['em_rate_present_only']:.4f}  "
              f"({sc_strict['n_exact_match']}/{sc_strict['n_fields_total']})")
    if sc_sem is not None:
        print(f"  Schema semantic:  EM={sc_sem['em_rate']:.4f}  "
              f"EM(present)={sc_sem['em_rate_present_only']:.4f}  "
              f"({sc_sem['n_exact_match']}/{sc_sem['n_fields_total']})  [diagnostic]")
    print(f"\nFull report -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qianfan-VLM eval (diagnose stage)")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--cache_dir", required=True, type=Path,
                   help="Directory of cached VLM JSONs (e.g. "
                        "dockg_refiner/outputs/vlm_cache/qianfan-llamacpp_seed0)")
    p.add_argument("--backend_name", required=True,
                   help="Backend tag used in the cache filename "
                        "(e.g. qianfan-llamacpp_seed0)")
    p.add_argument("--limit_docs", type=int, default=None)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--diagnose_only", action="store_true",
                   help="Only run cache/graph health diagnostics; do NOT "
                        "load a refiner checkpoint or score predictions.")
    p.add_argument("--mode", choices=[
        "raw", "refiner", "gold_upper",
        "raw_partial", "refiner_partial", "gold_upper_partial",
    ], default=None,
                   help="Scoring mode. *_partial only reports coref B^3 + "
                        "schema EM (no refer/relation).")
    p.add_argument("--refiner_ckpt", type=Path, default=None,
                   help="Refiner best.pt (used by --mode refiner / gold_upper).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Coref decision threshold for --mode refiner / gold_upper.")
    p.add_argument("--disable_schema_refine", action="store_true",
                   help="Skip the schema head when scoring; recommended "
                        "during smoke runs (codex 2026-05-07).")
    p.add_argument("--use_role_resolver", action="store_true",
                   help="Anaphora-aware alignment: resolve 甲方/乙方/第三方 "
                        "to gold partyA/partyB/partyC entity_id; use pred "
                        "entity_hint as fallback when surface match misses. "
                        "Default OFF — only enable for the anaphora 1-doc "
                        "A/B path (codex 2026-05-09).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.diagnose_only:
        return diagnose(args)
    if args.mode is None:
        print("[FATAL] specify --diagnose_only or --mode "
              "{raw|refiner|gold_upper|raw_partial|refiner_partial|gold_upper_partial}",
              file=sys.stderr)
        return 2
    if args.mode in ("raw_partial", "refiner_partial", "gold_upper_partial"):
        return run_partial(args)
    return _stub(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
