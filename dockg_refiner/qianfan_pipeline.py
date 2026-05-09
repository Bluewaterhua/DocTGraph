"""Qianfan-VLM → DocGraph → RefinerBatch → predictions pipeline.

Public surface (2026-05-07):

  * ``load_vlm_cache(cache_dir, doc_id, backend_name)``        — find/load JSON
  * ``parsed_health(parsed)`` / ``graph_health(graph)``         — diagnose
  * ``vlm_to_graph(parsed, page_count, doc_id)``               — build_graph_from_llm_only
  * ``raw_predictions(parsed)``                                — coref/schema from raw VLM
  * ``build_partial_batch_from_graph(graph)``                  — coref-only batch (no labels)
  * ``refiner_predictions_partial(model, graph, threshold)``   — coref/schema after refiner

The full ``--mode raw / refiner / gold_upper`` pipeline (including
relation extraction) is gated on the upgraded extractor schema, see
``qianfan_eval._stub`` for the gating message.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .data.dataset import REL_LABEL_IDX
from .graph.builder import (
    DocGraph,
    Edge,
    Node,
    build_graph_from_llm_only,
)
from .refiner.model import (
    EDGE_TYPE_TO_IDX,
    NODE_KIND_TO_IDX,
    RefinerBatch,
)


# ---------------------------------------------------------------------------
# VLM cache loading
# ---------------------------------------------------------------------------


def load_vlm_cache(cache_dir: Path, doc_id: str,
                   backend_name: str) -> Optional[Dict[str, Any]]:
    """Return the parsed JSON for ``doc_id`` from ``cache_dir`` or ``None``.

    The cache filename pattern is::

        {doc_id}__{backend_name}__{prompt_hash}.json

    We accept any prompt_hash (so re-runs after a prompt upgrade can be
    diagnosed alongside the legacy cache).
    """
    pattern = f"{doc_id}__{backend_name}__*.json"
    matches = sorted(Path(cache_dir).glob(pattern))
    if not matches:
        return None
    # Prefer the most recently modified file when several prompt_hashes
    # coexist on disk.
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Diagnostics: raw VLM output
# ---------------------------------------------------------------------------


_PLACEHOLDER_BBOX = [0, 0, 10, 10]


def _is_placeholder_bbox(bbox: Optional[List[int]]) -> bool:
    """True if bbox is missing or equals the builder's fallback."""
    return not bbox or bbox == _PLACEHOLDER_BBOX


def _bbox_stats(items: List[Dict[str, Any]],
                bbox_key: str = "bbox") -> Tuple[int, int, int]:
    """Return (n_total, n_missing, n_placeholder_exact) for a list of items.

    ``bbox_key`` lets callers point at ``source_bbox`` for ``refs[]``.
    """
    if not items:
        return 0, 0, 0
    n_missing = 0
    n_placeholder = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        bb = it.get(bbox_key)
        if bb is None:
            n_missing += 1
        elif bb == _PLACEHOLDER_BBOX:
            n_placeholder += 1
    return len(items), n_missing, n_placeholder


def parsed_health(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Quick per-doc counts. Operates only on the raw VLM JSON, no builder.

    bbox accounting (2026-05-07 codex review): scored separately per
    surface kind because the schema disagrees on which key carries the
    bounding box -- ``mentions[].bbox`` / ``values[].bbox`` /
    ``refs[].source_bbox`` / ``objects[].bbox``. The previous "single
    placeholder_rate" lumped ``refs`` into the bbox channel under the
    wrong key and counted them all as missing.
    """
    if parsed is None or parsed.get("_error"):
        return {"_error": parsed.get("_error", "missing")
                if parsed else "missing"}
    fields = parsed.get("fields") or {}
    mentions = parsed.get("mentions") or []
    values = parsed.get("values") or []
    refs = parsed.get("refs") or []
    objects = parsed.get("objects") or []

    bb_mentions = _bbox_stats(mentions, "bbox")
    bb_values = _bbox_stats(values, "bbox")
    bb_refs = _bbox_stats(refs, "source_bbox")
    bb_objects = _bbox_stats(objects, "bbox")

    def _bad(triple: Tuple[int, int, int]) -> int:
        return triple[1] + triple[2]   # missing + exact-placeholder

    bbox_bad = (
        _bad(bb_mentions) + _bad(bb_values)
        + _bad(bb_refs) + _bad(bb_objects)
    )
    bbox_missing = (bb_mentions[1] + bb_values[1]
                    + bb_refs[1] + bb_objects[1])
    bbox_placeholder = (bb_mentions[2] + bb_values[2]
                        + bb_refs[2] + bb_objects[2])
    bbox_total = bb_mentions[0] + bb_values[0] + bb_refs[0] + bb_objects[0]

    refs_with_target_page = sum(
        1 for r in refs if isinstance(r, dict)
        and r.get("guess_target_page") is not None
    )
    refs_with_target_anchor = sum(
        1 for r in refs if isinstance(r, dict)
        and r.get("guess_target_anchor") is not None
    )
    return {
        "n_fields": len(fields),
        "n_mentions": len(mentions),
        "n_values": len(values),
        "n_refs": len(refs),
        "n_objects": len(objects),
        # ---- bbox health, broken down by kind ----
        # Codex 2026-05-07: split bad rate into missing vs placeholder.
        # Missing (None / null) is acceptable noise; exact placeholder
        # [0,0,10,10] is the only thing we hard-fail on.
        "bbox_bad_total": bbox_bad,
        "bbox_missing_total": bbox_missing,
        "bbox_placeholder_total": bbox_placeholder,
        "bbox_total": bbox_total,
        "bbox_bad_rate":
            (bbox_bad / bbox_total) if bbox_total else None,
        "bbox_missing_rate":
            (bbox_missing / bbox_total) if bbox_total else None,
        "bbox_placeholder_rate":
            (bbox_placeholder / bbox_total) if bbox_total else None,
        "bbox_per_kind": {
            "mentions": {"n": bb_mentions[0],
                         "missing": bb_mentions[1],
                         "placeholder": bb_mentions[2]},
            "values":   {"n": bb_values[0],
                         "missing": bb_values[1],
                         "placeholder": bb_values[2]},
            "refs":     {"n": bb_refs[0],
                         "missing": bb_refs[1],
                         "placeholder": bb_refs[2],
                         "_bbox_key": "source_bbox"},
            "objects":  {"n": bb_objects[0],
                         "missing": bb_objects[1],
                         "placeholder": bb_objects[2]},
        },
        # ---- ref target-side annotations ----
        "refs_with_target_page": refs_with_target_page,
        "refs_with_target_page_rate":
            (refs_with_target_page / len(refs)) if refs else None,
        "refs_with_target_anchor": refs_with_target_anchor,
        "refs_with_target_anchor_rate":
            (refs_with_target_anchor / len(refs)) if refs else None,
        "_truncated": parsed.get("_truncated", False),
    }


# ---------------------------------------------------------------------------
# Diagnostics: post-builder graph
# ---------------------------------------------------------------------------


def vlm_to_graph(parsed: Dict[str, Any], page_count: int,
                 doc_id: str) -> Optional[DocGraph]:
    """Wrap ``build_graph_from_llm_only`` with an error guard."""
    if parsed is None or parsed.get("_error"):
        return None
    return build_graph_from_llm_only(doc_id, parsed, page_count)


def graph_health(graph: DocGraph) -> Dict[str, Any]:
    """Count nodes and edges by kind/etype in the post-builder graph."""
    if graph is None:
        return {"_error": "no_graph"}
    node_kind_counts: Dict[str, int] = {}
    for n in graph.nodes:
        node_kind_counts[n.kind] = node_kind_counts.get(n.kind, 0) + 1
    edge_etype_counts: Dict[str, int] = {}
    for e in graph.edges:
        edge_etype_counts[e.etype] = edge_etype_counts.get(e.etype, 0) + 1
    n_rel_candidate_edges = sum(
        edge_etype_counts.get(et, 0)
        for et in ("ref_candidate", "caption_candidate", "contains_candidate")
    )
    return {
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "node_kind_counts": node_kind_counts,
        "edge_etype_counts": edge_etype_counts,
        "n_rel_candidate_edges": n_rel_candidate_edges,
        "n_ref_candidate_edges": edge_etype_counts.get("ref_candidate", 0),
        "n_caption_candidate_edges": edge_etype_counts.get("caption_candidate", 0),
        "n_contains_candidate_edges": edge_etype_counts.get("contains_candidate", 0),
    }


# ---------------------------------------------------------------------------
# Raw VLM predictions: coref via entity_hint + schema via fields[]
# ---------------------------------------------------------------------------


_PUNCT_RE = re.compile(r"[\s\-_,.;:'\"/\\()\[\]{}<>+\*\|，。；：、（）【】《》　]+")


def _norm_text(text: Optional[str]) -> str:
    """Surface-normalise for entity_hint clustering (lower + strip punct).

    Kept as the *strict* default for callers that aren't field-aware (e.g.
    mention text alignment). Field-specific normalizers below are used by
    schema EM where each field has a different equivalence relation.
    """
    if not text:
        return ""
    return _PUNCT_RE.sub("", str(text)).lower()


# ---------------------------------------------------------------------------
# Field-aware normalizers (codex 2026-05-08)
# ---------------------------------------------------------------------------
# Two layers per field:
#   * STRICT  -> deterministic, surface-form-aware normalize used by
#                schema_em_strict. Headline metric. Conservative on
#                purpose; if Qianfan output a different surface form
#                (date format, money unit, OCR confusable), it counts as
#                a mismatch.
#   * SEMANTIC -> looser equivalence used by schema_em_semantic.
#                Diagnostic only -- not the headline. date / money have
#                custom equivalence relations; everything else falls
#                back to strict so we never silently flatter Qianfan.
# tax_no is intentionally identical between strict and semantic per
# 2026-05-08 user spec (we report confusable_match / edit_ratio as
# *separate* diagnostic fields rather than relaxing the EM bar).


def _digits_only(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\D", "", str(text))


def _strip_punct_upper(text: Optional[str]) -> str:
    if not text:
        return ""
    return _PUNCT_RE.sub("", str(text)).upper()


def _strip_punct_keep_case(text: Optional[str]) -> str:
    if not text:
        return ""
    return _PUNCT_RE.sub("", str(text))


def _strip_whitespace_lower(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", str(text)).lower()


def _norm_money_strict(text: Optional[str]) -> str:
    """Strict money: drop punct/whitespace/currency markers, keep digits + . + 万亿千 markers.

    Ensures '¥1,572,946.00' -> '1572946.00' and '5.41万元' -> '5.41万'. So
    different surface forms with the same digits collapse, but '5.41万'
    still differs from '54100' (we don't expand units in strict).
    """
    if not text:
        return ""
    s = str(text).upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", "").replace("，", "")
    # Drop currency markers but keep unit markers (so 万 vs 万元 collapse).
    s = re.sub(r"(¥|￥|CNY|RMB)", "", s)
    s = s.replace("元", "")
    return s


# --- semantic equivalence helpers -------------------------------------------

def _date_canonical(text: Optional[str]) -> Optional[str]:
    """YYYYMMDD if parseable as a complete date, else None.

    Conservative: requires 8 contiguous digits forming a valid Y/M/D. A
    partial date like '2019年7月' (only 6 digits) returns None and falls
    back to strict equality so we don't claim equivalence when one side
    is incomplete.
    """
    if not text:
        return None
    digits = _digits_only(text)
    if len(digits) < 8:
        return None
    cand = digits[:8]
    try:
        y, m, d = int(cand[0:4]), int(cand[4:6]), int(cand[6:8])
    except ValueError:
        return None
    if not (1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}{m:02d}{d:02d}"


_MONEY_UNIT_MULTIPLIER: Dict[str, Decimal] = {
    "亿": Decimal("100000000"),
    "万": Decimal("10000"),
    "千": Decimal("1000"),
}


def _money_parse(text: Optional[str]) -> Optional[Tuple[Decimal, bool]]:
    """Parse to (yuan_amount, used_abbreviation) or None.

    Recognises ¥/CNY/RMB/元/万/万元/千/亿. ``used_abbreviation`` flags
    whether the source text used 万/亿/千 (so the caller can apply a 1%
    relative tolerance instead of 1-yuan absolute).
    """
    if not text:
        return None
    s = str(text).upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", "").replace("，", "")
    s = re.sub(r"(¥|￥|CNY|RMB)", "", s)
    s = s.replace("元", "")
    # Strip remaining punctuation that callers commonly leave attached
    # to the currency tag (e.g. "CNY:1572946.00" -> drop the colon)
    # without touching digits, decimal point, or unit markers.
    s = re.sub(r"[^\d.万亿千]", "", s)
    used_abbrev = False
    multiplier = Decimal("1")
    for unit, mult in _MONEY_UNIT_MULTIPLIER.items():
        if unit in s:
            used_abbrev = True
            multiplier = mult
            s = s.replace(unit, "")
            break
    if not re.fullmatch(r"\d+(\.\d+)?", s):
        return None
    try:
        return (Decimal(s) * multiplier, used_abbrev)
    except InvalidOperation:
        return None


def _money_semantic_equal(gold: Optional[str], pred: Optional[str]) -> bool:
    """Per 2026-05-08 user spec.

    * Both sides parse and neither used an abbreviation: |gold-pred| <= 1 元
    * At least one side used 万/亿/千: |gold-pred| / max(|gold|,|pred|) <= 1%
    * Either side fails to parse: fall back to strict normalized string equality
    """
    g_parsed = _money_parse(gold)
    p_parsed = _money_parse(pred)
    if g_parsed is None or p_parsed is None:
        gn, pn = _norm_money_strict(gold), _norm_money_strict(pred)
        return bool(gn) and gn == pn
    g_val, g_abbrev = g_parsed
    p_val, p_abbrev = p_parsed
    abs_diff = abs(g_val - p_val)
    if g_abbrev or p_abbrev:
        denom = max(abs(g_val), abs(p_val))
        if denom == 0:
            return abs_diff == 0
        return (abs_diff / denom) <= Decimal("0.01")
    return abs_diff <= Decimal("1")


# --- tax_no diagnostics (NOT used to relax EM) ------------------------------

# Per 2026-05-08 user spec: tax_no strict and semantic both use upper +
# strip-punct + strict equality. Confusables and edit-distance are surfaced
# as separate diagnostic fields so codex can decide later whether to
# relax. We do NOT silently grant matches via these.

_TAX_CONFUSABLE_MAP = str.maketrans({
    "O": "0", "Q": "0",
    "I": "1", "L": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
})


def _tax_confusable_canonical(text: Optional[str]) -> str:
    if not text:
        return ""
    return _strip_punct_upper(text).translate(_TAX_CONFUSABLE_MAP)


def _levenshtein(a: str, b: str) -> int:
    """Pure-Python Levenshtein distance (small strings only)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def tax_no_diagnostics(gold: Optional[str], pred: Optional[str]) -> Dict[str, Any]:
    """Diagnostic info for tax_no comparison; never feeds into EM.

    Returns confusable_match (after O/0, I/1, S/5, B/8, ... map), the raw
    edit distance and the edit_ratio = 1 - dist / max(len). Useful for
    deciding whether OCR is the dominant error mode.
    """
    g = _strip_punct_upper(gold)
    p = _strip_punct_upper(pred)
    if not g or not p:
        return {
            "confusable_match": False,
            "edit_distance": None,
            "edit_ratio": None,
        }
    g_conf = g.translate(_TAX_CONFUSABLE_MAP)
    p_conf = p.translate(_TAX_CONFUSABLE_MAP)
    dist = _levenshtein(g, p)
    max_len = max(len(g), len(p))
    return {
        "confusable_match": g_conf == p_conf,
        "edit_distance": dist,
        "edit_ratio": 1.0 - dist / max_len,
    }


# --- per-field strict normalizers + dispatch table --------------------------

# Strict: produces a canonical string; equality means strict EM match.
_STRICT_NORM_BY_FIELD: Dict[str, Callable[[Optional[str]], str]] = {
    "contract_id":  _strip_punct_upper,
    "sign_date":    _digits_only,
    "start_date":   _digits_only,
    "end_date":     _digits_only,
    "total_money":  _norm_money_strict,
    "phone":        _digits_only,
    "email":        _strip_whitespace_lower,
    "tax_no":       _strip_punct_upper,
    "bank_account": _digits_only,
    "partyA":       _strip_punct_keep_case,
    "partyB":       _strip_punct_keep_case,
}


def schema_eq_strict(field: str, gold: Optional[str], pred: Optional[str]) -> bool:
    """Strict EM equality for a schema field (the headline metric)."""
    norm = _STRICT_NORM_BY_FIELD.get(field, _norm_text)
    g, p = norm(gold), norm(pred)
    return bool(g) and g == p


def schema_eq_semantic(field: str, gold: Optional[str], pred: Optional[str]) -> bool:
    """Semantic equality (looser; date/money custom, others fall back to strict).

    Diagnostic only -- never the headline metric per codex 2026-05-08.
    """
    if field in ("sign_date", "start_date", "end_date"):
        gc = _date_canonical(gold)
        pc = _date_canonical(pred)
        if gc is None or pc is None:
            return schema_eq_strict(field, gold, pred)
        return gc == pc
    if field == "total_money":
        return _money_semantic_equal(gold, pred)
    return schema_eq_strict(field, gold, pred)


# ---------------------------------------------------------------------------
# Entity-like mention filter and gold-entity alignment (codex 2026-05-08)
# ---------------------------------------------------------------------------

# Filter applies ONLY to coref evaluation mention universe -- schema/
# relation pipelines see all mentions. The hard-coded anaphora regex is
# forward-looking: current Qianfan cache does not emit "甲方/乙方" mentions
# (only the cover company names), so on the current cache it's a no-op,
# but once the P1 prompt upgrade lands the eval needs no further change.

_ENTITY_LIKE_HINT_SET = {
    "公司名", "甲方", "乙方", "第三方",
    "partyA", "partyB", "company",
}

_ANAPHORA_RE = re.compile(
    r"(甲方|乙方|双方|对方|本公司|我司|贵司|该公司|上述公司|第三方|第三方机构)"
)


def is_entity_like(mention: Dict[str, Any]) -> bool:
    """True if a Qianfan mention should participate in coref evaluation."""
    if not isinstance(mention, dict):
        return False
    hint = mention.get("entity_hint")
    if hint and hint in _ENTITY_LIKE_HINT_SET:
        return True
    text = mention.get("text") or ""
    return bool(_ANAPHORA_RE.search(text))


def filter_entity_like_mentions(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of {idx, text, entity_hint, page_idx} for entity-like mentions.

    ``idx`` is the position in ``parsed['mentions']`` so the caller can
    map back to predicted cluster ids from raw_predictions /
    refiner_predictions_partial.
    """
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(parsed.get("mentions") or []):
        if not isinstance(m, dict):
            continue
        if not is_entity_like(m):
            continue
        out.append({
            "idx": i,
            "text": m.get("text") or "",
            "entity_hint": m.get("entity_hint"),
            "page_idx": int(m.get("page_idx", 1)),
        })
    return out


def _company_normalize(text: Optional[str]) -> str:
    """Strict company normalize for canonical/alias matching (no stemming)."""
    return _strip_punct_keep_case(text)


def align_pred_mentions_to_gold_entities(
    pred_entity_like: List[Dict[str, Any]],
    gold_entities: List[Dict[str, Any]],
) -> Tuple[List[Optional[str]], Dict[str, Any]]:
    """For each entity-like pred mention, return best gold ``entity_id`` (or None).

    ``gold_entities`` items: ``{"entity_id", "canonical", "aliases"?}``.

    Scoring (highest wins, must clear threshold):
      * normalized text equals canonical or any alias  -> score 1.0
      * normalized text contains canonical (or vice versa) -> 0.7
      * Levenshtein ratio against canonical -> ratio (must be >= 0.85)

    Anaphora ("甲方/乙方/...") cannot be resolved by text similarity
    alone -- they're left unaligned (None). When P1 prompt upgrade
    lands, the eval may need an upstream "甲方→entA, 乙方→entB" rule
    based on document role; not added now to keep this change pure
    P0 (no prompt-side coupling).
    """
    norm_canonicals: List[Tuple[str, str, List[str]]] = []
    for e in gold_entities:
        cano = _company_normalize(e.get("canonical"))
        aliases = [
            _company_normalize(a) for a in (e.get("aliases") or [])
            if a
        ]
        norm_canonicals.append((e.get("entity_id", ""), cano, aliases))

    assignments: List[Optional[str]] = []
    n_aligned = 0
    n_anaphora_unresolved = 0
    for pm in pred_entity_like:
        text = pm.get("text") or ""
        if _ANAPHORA_RE.fullmatch(text) or _ANAPHORA_RE.fullmatch(text.strip()):
            # Pure anaphora -- can't decide from text alone.
            assignments.append(None)
            n_anaphora_unresolved += 1
            continue
        norm_text = _company_normalize(text)
        if not norm_text:
            assignments.append(None)
            continue
        best_eid: Optional[str] = None
        best_score = 0.0
        for eid, cano, aliases in norm_canonicals:
            if not cano:
                continue
            # exact / alias
            if norm_text == cano or norm_text in aliases:
                score = 1.0
            elif norm_text in cano or cano in norm_text:
                score = 0.7
            else:
                # Levenshtein ratio
                dist = _levenshtein(norm_text, cano)
                ml = max(len(norm_text), len(cano))
                ratio = (1.0 - dist / ml) if ml else 0.0
                score = ratio if ratio >= 0.85 else 0.0
            if score > best_score:
                best_score = score
                best_eid = eid
        if best_score > 0:
            assignments.append(best_eid)
            n_aligned += 1
        else:
            assignments.append(None)
    info = {
        "n_pred_entity_like": len(pred_entity_like),
        "n_pred_aligned": n_aligned,
        "n_pred_anaphora_unresolved": n_anaphora_unresolved,
        "alignment_rate":
            (n_aligned / len(pred_entity_like)) if pred_entity_like else 0.0,
    }
    return assignments, info


# ---------------------------------------------------------------------------
# Role resolver (codex 2026-05-09, anaphora 1-doc A/B only)
# ---------------------------------------------------------------------------

# Unambiguous role tokens — fullmatch only. partyA/partyB exist in every
# 2-party contract; partyC exists only when the doc has a third party
# entity (e.g. doc_000003's "岚州泽洛咨询中心").
_ROLE_ANAPHORA_FULL: Dict[str, str] = {
    "甲方": "partyA",
    "乙方": "partyB",
    "第三方": "partyC",
    "第三方机构": "partyC",
}

# Ambiguous indexicals — speaker / addressee / "the company" without an
# explicit role label. These should NOT be silently bucketed into either
# party because that risks fake-good B^3 (matching the wrong entity).
_AMBIGUOUS_ANAPHORA: set = {
    "双方", "对方", "我司", "本公司", "贵司", "该公司", "上述公司",
}

# entity_hint values that the anaphora prompt can emit, mapped to role.
_HINT_TO_ROLE: Dict[str, str] = {
    "partyA": "partyA",
    "partyB": "partyB",
    "partyC": "partyC",
    "甲方": "partyA",
    "乙方": "partyB",
    "第三方": "partyC",
}


def build_gold_role_map(
    gold_entities: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Map gold ``entity_id -> 'partyA'|'partyB'|'partyC'``.

    Heuristic (matches synthetic v3 renderer):
      * holds ``m_partyA_cover`` OR ``entity_id`` starts with ``entA`` -> partyA
      * holds ``m_partyB_cover`` OR ``entity_id`` starts with ``entB`` -> partyB
      * everything else -> partyC

    Built once per doc; cheap. Used only when ``--use_role_resolver`` is on.
    """
    role_map: Dict[str, str] = {}
    for e in gold_entities or []:
        eid = e.get("entity_id", "")
        ms = e.get("mentions") or []
        if "m_partyA_cover" in ms or eid.startswith("entA"):
            role_map[eid] = "partyA"
        elif "m_partyB_cover" in ms or eid.startswith("entB"):
            role_map[eid] = "partyB"
        else:
            role_map[eid] = "partyC"
    return role_map


def align_pred_mentions_to_gold_entities_with_role_resolver(
    pred_entity_like: List[Dict[str, Any]],
    gold_entities: List[Dict[str, Any]],
    gold_role_map: Dict[str, str],
) -> Tuple[List[Optional[str]], Dict[str, Any]]:
    """Anaphora-aware variant of ``align_pred_mentions_to_gold_entities``.

    Tiers (first hit wins):

      T1 — surface match (sole tier in P0): canonical / contains / Lev>=0.85.
      T2 — role anaphora: text fullmatch in {甲方, 乙方, 第三方, 第三方机构}
           -> role -> entity_id. If the role isn't represented in this doc
           (e.g. partyC missing), the mention is left unaligned.
      T3 — entity_hint fallback: pred mention's ``entity_hint`` in
           ``_HINT_TO_ROLE`` -> role -> entity_id. Fires only when surface
           and anaphora tiers both miss.
      Ambiguous indexicals (双方/对方/我司/本公司/贵司/该公司/上述公司):
           always None, counted under ``n_pred_anaphora_ambiguous``.

    Returns (assignments, info). ``info`` carries per-tier counters so
    codex can sanity-check why each pred mention landed where it did.
    """
    norm_canonicals: List[Tuple[str, str, List[str]]] = []
    for e in gold_entities or []:
        cano = _company_normalize(e.get("canonical"))
        aliases = [
            _company_normalize(a) for a in (e.get("aliases") or []) if a
        ]
        norm_canonicals.append((e.get("entity_id", ""), cano, aliases))

    role_to_eid: Dict[str, str] = {}
    for eid, role in (gold_role_map or {}).items():
        role_to_eid.setdefault(role, eid)

    assignments: List[Optional[str]] = []
    n_aligned = 0
    n_via_surface = 0
    n_via_anaphora = 0
    n_via_hint = 0
    n_anaphora_ambiguous = 0
    n_anaphora_role_missing = 0

    for pm in pred_entity_like:
        text = (pm.get("text") or "").strip()
        norm_text = _company_normalize(text)

        # T1: surface match
        best_eid: Optional[str] = None
        best_score = 0.0
        if norm_text:
            for eid, cano, aliases in norm_canonicals:
                if not cano:
                    continue
                if norm_text == cano or norm_text in aliases:
                    score = 1.0
                elif norm_text in cano or cano in norm_text:
                    score = 0.7
                else:
                    dist = _levenshtein(norm_text, cano)
                    ml = max(len(norm_text), len(cano))
                    ratio = (1.0 - dist / ml) if ml else 0.0
                    score = ratio if ratio >= 0.85 else 0.0
                if score > best_score:
                    best_score = score
                    best_eid = eid
        if best_score > 0:
            assignments.append(best_eid)
            n_aligned += 1
            n_via_surface += 1
            continue

        # T2: role anaphora (fullmatch)
        if text in _ROLE_ANAPHORA_FULL:
            role = _ROLE_ANAPHORA_FULL[text]
            eid = role_to_eid.get(role)
            if eid is not None:
                assignments.append(eid)
                n_aligned += 1
                n_via_anaphora += 1
            else:
                assignments.append(None)
                n_anaphora_role_missing += 1
            continue

        # Ambiguous indexicals — never auto-resolve
        if text in _AMBIGUOUS_ANAPHORA:
            assignments.append(None)
            n_anaphora_ambiguous += 1
            continue

        # T3: entity_hint fallback (only when surface failed)
        hint = pm.get("entity_hint")
        if hint:
            role = _HINT_TO_ROLE.get(hint)
            if role:
                eid = role_to_eid.get(role)
                if eid is not None:
                    assignments.append(eid)
                    n_aligned += 1
                    n_via_hint += 1
                    continue

        assignments.append(None)

    info = {
        "n_pred_entity_like": len(pred_entity_like),
        "n_pred_aligned": n_aligned,
        "n_aligned_via_surface": n_via_surface,
        "n_aligned_via_anaphora_role": n_via_anaphora,
        "n_aligned_via_hint_only": n_via_hint,
        "n_pred_anaphora_ambiguous": n_anaphora_ambiguous,
        "n_pred_anaphora_role_missing_in_doc": n_anaphora_role_missing,
        "alignment_rate":
            (n_aligned / len(pred_entity_like)) if pred_entity_like else 0.0,
        "role_resolver": True,
    }
    return assignments, info


def raw_predictions(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Score raw VLM output as a gold-shaped pred dict.

    Coref: cluster ``mentions[]`` by ``(entity_hint, normalised text)`` —
    same hint AND same surface text -> same cluster. Mentions without
    entity_hint become singletons.

    Schema: ``fields[k].raw`` -> normalised text. Used by ``schema_em``.

    Relations: NOT YET (gated on objects[]). The pred dict carries
    ``relations: []`` so the evaluator can still run partial metrics.
    """
    if not parsed or parsed.get("_error"):
        return {"coref_clusters": [], "schema": {}, "relations": []}
    mentions = parsed.get("mentions") or []
    # Cluster by (hint, normalised text). We use BOTH because hints alone
    # collapse all "公司" mentions into one bucket, which destroys B^3.
    cluster_key_to_members: Dict[Tuple[Optional[str], str], List[int]] = {}
    cluster_texts: List[str] = []
    for i, m in enumerate(mentions):
        if not isinstance(m, dict):
            continue
        text = _norm_text(m.get("text"))
        hint = m.get("entity_hint")
        key = (hint, text) if text else (hint, f"_unk_{i}")
        cluster_key_to_members.setdefault(key, []).append(i)
        cluster_texts.append(str(m.get("text") or ""))
    coref_clusters = list(cluster_key_to_members.values())

    fields = parsed.get("fields") or {}
    schema_pred: Dict[str, Optional[str]] = {}
    for k, payload in fields.items():
        if isinstance(payload, dict):
            schema_pred[k] = payload.get("norm") or payload.get("raw")
        else:
            schema_pred[k] = None

    return {
        "coref_clusters": coref_clusters,           # list of lists of mention indices
        "mentions_text": cluster_texts,             # parallel raw text (for debug)
        "schema": schema_pred,
        "relations": [],
    }


# ---------------------------------------------------------------------------
# Refiner-side: partial batch with no gold labels (coref only)
# ---------------------------------------------------------------------------


def _placeholder_schema_slots(num_schema_fields: int = 11
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Empty schema slot tensors so RefinerBatch validates."""
    nodes = torch.full((num_schema_fields, 1), -1, dtype=torch.long)
    targets = torch.full((num_schema_fields,), -1, dtype=torch.long)
    return nodes, targets


def build_partial_batch_from_graph(graph: DocGraph,
                                   num_schema_fields: int = 11
                                   ) -> Tuple[RefinerBatch, List[int]]:
    """Build a ``RefinerBatch`` for inference (no gold labels).

    Returns ``(batch, mention_node_indices)``. ``mention_node_indices``
    holds the position (0..N-1) of each ``kind=='mention'`` node so the
    caller can map coref-pair predictions back to the input graph.

    NOTE: this is the *partial* batch — no relation candidates are
    enumerated, no schema slot supervision is built. The full batch
    (relation + schema legal mask, etc.) is gated on the upgraded VLM
    schema and lives in a future ``build_batch_from_graph``.
    """
    n = len(graph.nodes)
    if n == 0:
        empty = RefinerBatch(
            node_text=[],
            node_kind=torch.zeros(0, dtype=torch.long),
            node_bbox=torch.zeros(0, 4, dtype=torch.float),
            node_page=torch.zeros(0, dtype=torch.long),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_type=torch.zeros(0, dtype=torch.long),
            coref_pairs=torch.zeros(0, 2, dtype=torch.long),
            coref_labels=torch.zeros(0, dtype=torch.long),
            rel_edges=torch.zeros(0, 2, dtype=torch.long),
            rel_labels=torch.zeros(0, dtype=torch.long),
            schema_slot_nodes=_placeholder_schema_slots(num_schema_fields)[0],
            schema_slot_targets=_placeholder_schema_slots(num_schema_fields)[1],
        )
        return empty, []

    node_id_to_idx = {node.node_id: i for i, node in enumerate(graph.nodes)}
    node_text = [node.text or "" for node in graph.nodes]
    node_kind = torch.tensor(
        [NODE_KIND_TO_IDX[node.kind] for node in graph.nodes], dtype=torch.long
    )
    node_bbox = torch.tensor([node.bbox for node in graph.nodes], dtype=torch.float)
    node_page = torch.tensor([node.page_idx for node in graph.nodes], dtype=torch.long)

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_types: List[int] = []
    for edge in graph.edges:
        if edge.src not in node_id_to_idx or edge.dst not in node_id_to_idx:
            continue
        if edge.etype not in EDGE_TYPE_TO_IDX:
            continue
        edge_src.append(node_id_to_idx[edge.src])
        edge_dst.append(node_id_to_idx[edge.dst])
        edge_types.append(EDGE_TYPE_TO_IDX[edge.etype])
    edge_index = (
        torch.tensor([edge_src, edge_dst], dtype=torch.long)
        if edge_src else torch.zeros(2, 0, dtype=torch.long)
    )
    edge_type = (
        torch.tensor(edge_types, dtype=torch.long)
        if edge_types else torch.zeros(0, dtype=torch.long)
    )

    schema_nodes, schema_targets = _placeholder_schema_slots(num_schema_fields)
    batch = RefinerBatch(
        node_text=node_text,
        node_kind=node_kind,
        node_bbox=node_bbox,
        node_page=node_page,
        edge_index=edge_index,
        edge_type=edge_type,
        coref_pairs=torch.zeros(0, 2, dtype=torch.long),  # filled by caller per-doc
        coref_labels=torch.zeros(0, dtype=torch.long),
        rel_edges=torch.zeros(0, 2, dtype=torch.long),
        rel_labels=torch.zeros(0, dtype=torch.long),
        schema_slot_nodes=schema_nodes,
        schema_slot_targets=schema_targets,
    )
    mention_indices = [i for i, node in enumerate(graph.nodes)
                       if node.kind == "mention"]
    return batch, mention_indices


def _batch_to(batch: RefinerBatch, device) -> RefinerBatch:
    """Move tensor fields of a RefinerBatch to ``device`` (text stays as list)."""
    fields = (
        "node_kind", "node_bbox", "node_page", "edge_index", "edge_type",
        "coref_pairs", "coref_labels", "coref_pair_source",
        "rel_edges", "rel_labels", "schema_slot_nodes",
        "schema_slot_targets",
    )
    new_kwargs: Dict[str, Any] = {}
    for f in fields:
        v = getattr(batch, f, None)
        if v is None:
            continue
        new_kwargs[f] = v.to(device, non_blocking=True)
    new_kwargs["node_text"] = batch.node_text
    return type(batch)(**new_kwargs)


def _union_find(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    out: Dict[int, List[int]] = {}
    for k in range(n):
        out.setdefault(find(k), []).append(k)
    return list(out.values())


def refiner_predictions_partial(model: torch.nn.Module,
                                graph: DocGraph,
                                device,
                                threshold: float = 0.5,
                                ) -> Dict[str, Any]:
    """Run refiner *coref head only* over an LLM-built graph.

    Returns the same shape as ``raw_predictions`` so the evaluator is
    symmetric. ``schema`` here is empty -- partial refiner does not
    score schema (codex 2026-05-07: --disable_schema_refine for smoke).
    Use ``raw_predictions(parsed)["schema"]`` for the schema row when
    reporting partial metrics.
    """
    model.eval()
    batch, mention_idx = build_partial_batch_from_graph(graph)
    if not mention_idx:
        return {"coref_clusters": [], "schema": {}, "relations": []}
    n = len(mention_idx)
    if n < 2:
        return {
            "coref_clusters": [[0]],
            "mentions_text": [graph.nodes[mention_idx[0]].text or ""],
            "schema": {},
            "relations": [],
        }

    batch = _batch_to(batch, device)
    pair_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_t = torch.tensor(
        [[mention_idx[i], mention_idx[j]] for (i, j) in pair_list],
        dtype=torch.long, device=device,
    )
    with torch.no_grad():
        h = model.encode(batch)
        logits = model.coref_logits(h, pair_t, batch=batch)
        probs = torch.sigmoid(logits).detach().cpu().tolist()

    edges_pos = [pair_list[k] for k, p in enumerate(probs) if p >= threshold]
    clusters = _union_find(n, edges_pos)
    return {
        "coref_clusters": clusters,                                # positions in mention_idx
        "mentions_text": [graph.nodes[mi].text or "" for mi in mention_idx],
        "schema": {},
        "relations": [],
        "n_pairs_scored": len(pair_list),
        "n_pairs_pos": len(edges_pos),
    }


# ---------------------------------------------------------------------------
# Stubs for the full refiner path (relations + schema)
# ---------------------------------------------------------------------------


def build_batch_from_graph(graph: DocGraph, num_schema_fields: int = 11):
    raise NotImplementedError(
        "build_batch_from_graph (full version with rel candidates) is a "
        "stub; implement after Qianfan re-extract with objects[] schema "
        "is verified."
    )


def refine(model, batch, graph: DocGraph,
           threshold: float = 0.5,
           disable_schema: bool = False) -> Dict[str, Any]:
    raise NotImplementedError(
        "refine() (relations + optional schema) is a stub; implement "
        "after Qianfan re-extract with objects[] schema is verified."
    )
