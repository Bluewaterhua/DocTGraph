"""Regression tests for the three codex 2026-05-09 evaluator findings.

Run via:
    .venv\\Scripts\\python.exe -m pytest dockg_refiner/tests/test_evaluator_p0.py -v
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import unittest
from pathlib import Path

from dockg_refiner.qianfan_eval import (
    _aggregate_partial,
    _b3_aligned,
    _entity_level_pred_assignment,
    run_partial,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DS = REPO_ROOT / "synthetic_contract_ds" / "contract_synth_safe_v3_smoke"


class TestB3PairingNoCollapse(unittest.TestCase):
    """Finding 1: pred fragmentation must not score B^3 F1 = 1.0."""

    def test_two_pred_singletons_do_not_collapse_to_one_cluster(self):
        # Gold: one entity entA with 2 mentions at gold positions 0 and 1.
        gold_clusters = [[0, 1]]
        gold_mention_to_entity = ["entA", "entA"]

        # Pred: 2 entity-like mentions both alignable to entA, but each
        # in its own pred singleton cluster.
        pred_entity_like = [
            {"idx": 0, "text": "alpha"},
            {"idx": 1, "text": "beta"},
        ]
        pred_to_gold_entity = ["entA", "entA"]
        pred_cluster_id_by_parsed_pos = {0: 0, 1: 1}  # two singletons

        assignment = _entity_level_pred_assignment(
            gold_mention_to_entity,
            pred_entity_like,
            pred_to_gold_entity,
            pred_cluster_id_by_parsed_pos,
        )

        # Insertion-order pairing: gold pos 0 -> pred cluster 0,
        # gold pos 1 -> pred cluster 1. NOT both -> 0 (no majority collapse).
        self.assertEqual(assignment, [0, 1])

        b3 = _b3_aligned(gold_clusters, assignment)
        # Per-mention: P=1/1=1.0 (each pred singleton has size 1), R=1/2=0.5
        # (gold cluster size 2, intersection 1). F1 = 0.667. NOT 1.0.
        self.assertLess(
            b3["f1"], 0.99,
            f"fragmented pred singletons should not score F1=1.0, got {b3}",
        )
        self.assertAlmostEqual(b3["f1"], 2 / 3, places=3)

    def test_perfect_merge_still_scores_1(self):
        """Sanity: 2 pred mentions in the same cluster still score 1.0."""
        gold_clusters = [[0, 1]]
        gold_mention_to_entity = ["entA", "entA"]
        pred_entity_like = [
            {"idx": 0, "text": "alpha"},
            {"idx": 1, "text": "beta"},
        ]
        pred_to_gold_entity = ["entA", "entA"]
        pred_cluster_id_by_parsed_pos = {0: 0, 1: 0}  # both in cluster 0

        assignment = _entity_level_pred_assignment(
            gold_mention_to_entity,
            pred_entity_like,
            pred_to_gold_entity,
            pred_cluster_id_by_parsed_pos,
        )
        self.assertEqual(assignment, [0, 0])
        b3 = _b3_aligned(gold_clusters, assignment)
        self.assertAlmostEqual(b3["f1"], 1.0, places=4)


class TestGoldUpperBypassesVlmCache(unittest.TestCase):
    """Finding 2: gold_upper_partial must not depend on VLM cache."""

    def test_gold_upper_skips_cache_dir_existence_check(self):
        if not SMOKE_DS.exists():
            self.skipTest(f"smoke dataset missing: {SMOKE_DS}")

        args = argparse.Namespace(
            dataset=SMOKE_DS,
            cache_dir=Path("nonexistent_path_xyzz_12345"),
            backend_name="qianfan-llamacpp_seed0",
            limit_docs=1,
            mode="gold_upper_partial",
            # Fake ckpt path so run_partial fails AFTER the cache_dir
            # gating, proving the gating worked.
            refiner_ckpt=Path("nonexistent_ckpt.pt"),
            threshold=0.5,
            disable_schema_refine=True,
            use_role_resolver=False,
            diagnose_only=False,
            out=Path("__test_out_should_not_be_written.json"),
        )

        buf_err = io.StringIO()
        buf_out = io.StringIO()
        with contextlib.redirect_stderr(buf_err), \
             contextlib.redirect_stdout(buf_out):
            try:
                run_partial(args)
            except Exception:  # noqa: BLE001
                # Expected: torch.load on the fake ckpt path raises.
                # We don't care about ckpt load failures; we only assert
                # that the cache_dir-not-found early return did NOT fire.
                pass

        stderr = buf_err.getvalue()
        self.assertNotIn(
            "cache_dir not found", stderr,
            "gold_upper_partial should not bail out on missing cache_dir; "
            f"stderr=<{stderr.strip()}>",
        )


class TestGoldUpperSchemaNotScored(unittest.TestCase):
    """Finding 3: gold_upper schema must be marked not_scored, not 0%."""

    def test_aggregate_marks_schema_not_scored_when_all_docs_skip(self):
        per_doc = [
            {
                "doc_id": "d1",
                "b3": {
                    "main": {
                        "precision": 1.0, "recall": 1.0, "f1": 1.0,
                        "n_mentions": 4,
                    },
                    "aligned_only": {
                        "precision": 1.0, "recall": 1.0, "f1": 1.0,
                        "n_mentions": 4,
                    },
                    "alignment": {
                        "n_gold_total": 4, "n_gold_aligned": 4,
                        "n_pred_entity_like": 4, "n_pred_aligned": 4,
                    },
                },
                "schema": {
                    "_not_scored": True,
                    "reason": "gold_upper_partial does not score schema",
                },
            },
        ]
        summary = _aggregate_partial(per_doc)
        sc = summary["schema"]
        self.assertTrue(
            sc.get("_not_scored"),
            f"schema should be marked _not_scored, got: {sc}",
        )
        # Crucially: no fabricated em_rate / strict / semantic blocks
        self.assertNotIn("strict", sc)
        self.assertNotIn("semantic", sc)

    def test_aggregate_skips_not_scored_records_in_mixed_run(self):
        """Mixed: 1 real-scored doc + 1 not_scored doc.

        The not_scored doc must be excluded from EM averages so its
        zero-but-fake numbers don't dilute the real score.
        """
        per_doc = [
            {
                "doc_id": "d_scored",
                "b3": {
                    "main": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_mentions": 1},
                    "aligned_only": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_mentions": 1},
                    "alignment": {"n_gold_total": 1, "n_gold_aligned": 1,
                                  "n_pred_entity_like": 1, "n_pred_aligned": 1},
                },
                "schema": {
                    "strict": {
                        "n_fields": 11, "n_present_gold": 10,
                        "n_pred_nonempty": 10, "n_exact_match": 8,
                    },
                    "semantic": {"n_exact_match": 9},
                },
            },
            {
                "doc_id": "d_unscored",
                "b3": {
                    "main": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_mentions": 1},
                    "aligned_only": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_mentions": 1},
                    "alignment": {"n_gold_total": 1, "n_gold_aligned": 1,
                                  "n_pred_entity_like": 1, "n_pred_aligned": 1},
                },
                "schema": {"_not_scored": True, "reason": "gold_upper"},
            },
        ]
        summary = _aggregate_partial(per_doc)
        sc = summary["schema"]
        self.assertFalse(sc.get("_not_scored"))
        self.assertEqual(sc["n_docs_schema_scored"], 1)
        # n_fields_total should be 11 (only the scored doc), not 22.
        self.assertEqual(sc["strict"]["n_fields_total"], 11)
        self.assertEqual(sc["strict"]["n_exact_match"], 8)


if __name__ == "__main__":
    unittest.main()
