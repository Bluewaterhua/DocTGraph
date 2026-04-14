from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback for lightweight environments.
    np = None
    linear_sum_assignment = None

from layoutlmv3.normalization import (
    canonicalize_surface_form,
    is_recoverable_value,
    normalize_value,
)


def binary_counts_from_logits(logits, labels) -> tuple[int, int, int]:
    preds = (logits.sigmoid() >= 0.5).long()
    gold = labels.long()
    tp = int(((preds == 1) & (gold == 1)).sum().item())
    fp = int(((preds == 1) & (gold == 0)).sum().item())
    fn = int(((preds == 0) & (gold == 1)).sum().item())
    return tp, fp, fn


def binary_f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1_from_predictions(
    preds: Sequence[int],
    labels: Sequence[int],
    num_classes: int,
    ignore_index: int | None = None,
) -> float:
    class_f1 = []
    for class_id in range(num_classes):
        if ignore_index is not None and class_id == ignore_index:
            continue
        tp = sum(1 for pred, gold in zip(preds, labels) if pred == class_id and gold == class_id)
        fp = sum(1 for pred, gold in zip(preds, labels) if pred == class_id and gold != class_id)
        fn = sum(1 for pred, gold in zip(preds, labels) if pred != class_id and gold == class_id)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            class_f1.append(0.0)
        else:
            class_f1.append(2 * precision * recall / (precision + recall))
    return sum(class_f1) / len(class_f1) if class_f1 else 0.0


def _sanitize_metric_key(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(text)).strip("_") or "unknown"


def _page_count_bucket(page_count: int | None) -> str:
    if page_count is None:
        return "unknown"
    if page_count <= 1:
        return "1"
    if page_count <= 3:
        return "2_3"
    return "4_plus"


def _sample_bucket_labels(sample: Dict) -> Dict[str, str]:
    meta = sample.get("meta", {})
    return {
        "noise_level": _sanitize_metric_key(meta.get("noise_level", "unknown")),
        "coref_difficulty": _sanitize_metric_key(meta.get("coref_difficulty", "unknown")),
        "ref_difficulty": _sanitize_metric_key(meta.get("ref_difficulty", "unknown")),
        "layout_profile": _sanitize_metric_key(meta.get("layout_profile", "unknown")),
        "page_count_bucket": _page_count_bucket(meta.get("page_count")),
    }


def _mention_ids_from_sample(sample: Dict) -> List[str]:
    mention_ids = []
    for node in sample["nodes"]:
        if node["kind"] != "mention":
            continue
        mention_ids.append(node.get("mention_id") or node["node_id"])
    return mention_ids


def _gold_clusters_from_sample(sample: Dict) -> List[set[str]]:
    mention_universe = set(_mention_ids_from_sample(sample))
    clusters = []
    covered = set()
    for entity in sample["labels"]["coref"]["entities"]:
        cluster = {mention_id for mention_id in entity["mentions"] if mention_id in mention_universe}
        if cluster:
            clusters.append(cluster)
            covered.update(cluster)
    for mention_id in sorted(mention_universe - covered):
        clusters.append({mention_id})
    return clusters


def _predicted_clusters_from_pairs(sample: Dict, pairs: List[Tuple[int, int, int]], logits) -> List[set[str]]:
    mention_nodes = {
        idx: (node.get("mention_id") or node["node_id"])
        for idx, node in enumerate(sample["nodes"])
        if node["kind"] == "mention"
    }
    mentions = list(mention_nodes.values())
    parent = {mention_id: mention_id for mention_id in mentions}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pred_flags = (logits.sigmoid() >= 0.5).tolist()
    for (idx_a, idx_b, _), keep in zip(pairs, pred_flags):
        if not keep:
            continue
        mention_a = mention_nodes.get(idx_a)
        mention_b = mention_nodes.get(idx_b)
        if mention_a is None or mention_b is None:
            continue
        union(mention_a, mention_b)

    grouped = defaultdict(set)
    for mention_id in mentions:
        grouped[find(mention_id)].add(mention_id)
    return list(grouped.values())


def _cluster_map(clusters: Iterable[set[str]]) -> Dict[str, set[str]]:
    mapping = {}
    for cluster in clusters:
        for mention_id in cluster:
            mapping[mention_id] = cluster
    return mapping


def _muc_num_den(reference: List[set[str]], response: List[set[str]]) -> tuple[float, float]:
    response_lookup = _cluster_map(response)
    numerator = 0.0
    denominator = 0.0
    for cluster in reference:
        if len(cluster) <= 1:
            continue
        partitions = {
            tuple(sorted(response_lookup.get(mention_id, {mention_id})))
            for mention_id in cluster
        }
        numerator += len(cluster) - len(partitions)
        denominator += len(cluster) - 1
    return numerator, denominator


def _bcub_num_den(gold_clusters: List[set[str]], pred_clusters: List[set[str]]) -> tuple[float, float, float]:
    gold_map = _cluster_map(gold_clusters)
    pred_map = _cluster_map(pred_clusters)
    mentions = sorted(set(gold_map) | set(pred_map))
    precision_sum = 0.0
    recall_sum = 0.0
    for mention_id in mentions:
        gold_cluster = gold_map.get(mention_id, {mention_id})
        pred_cluster = pred_map.get(mention_id, {mention_id})
        overlap = len(gold_cluster & pred_cluster)
        precision_sum += overlap / max(len(pred_cluster), 1)
        recall_sum += overlap / max(len(gold_cluster), 1)
    return precision_sum, recall_sum, float(len(mentions))


def _greedy_alignment(similarity: List[List[float]]) -> float:
    flat = []
    for row_idx, row in enumerate(similarity):
        for col_idx, value in enumerate(row):
            flat.append((value, row_idx, col_idx))
    flat.sort(reverse=True)
    used_rows = set()
    used_cols = set()
    total = 0.0
    for value, row_idx, col_idx in flat:
        if row_idx in used_rows or col_idx in used_cols:
            continue
        used_rows.add(row_idx)
        used_cols.add(col_idx)
        total += value
    return total


def _ceafe_similarity(gold_clusters: List[set[str]], pred_clusters: List[set[str]]) -> tuple[float, int, int]:
    if not gold_clusters or not pred_clusters:
        return 0.0, len(gold_clusters), len(pred_clusters)
    similarity = [
        [
            (2.0 * len(gold_cluster & pred_cluster)) / max(len(gold_cluster) + len(pred_cluster), 1)
            for pred_cluster in pred_clusters
        ]
        for gold_cluster in gold_clusters
    ]
    if linear_sum_assignment is not None and np is not None:
        cost = -np.asarray(similarity, dtype=float)
        row_ind, col_ind = linear_sum_assignment(cost)
        score = float(sum(similarity[row_idx][col_idx] for row_idx, col_idx in zip(row_ind, col_ind)))
    else:
        score = _greedy_alignment(similarity)
    return score, len(gold_clusters), len(pred_clusters)


def _lea_num_den(reference: List[set[str]], response: List[set[str]]) -> tuple[float, float]:
    response_lookup = _cluster_map(response)
    numerator = 0.0
    denominator = 0.0
    for cluster in reference:
        importance = float(len(cluster))
        denominator += importance
        if len(cluster) <= 1:
            numerator += importance
            continue
        overlaps = Counter()
        for mention_id in cluster:
            response_cluster = response_lookup.get(mention_id, {mention_id})
            overlaps[tuple(sorted(response_cluster))] += 1
        correct_links = sum(count * (count - 1) / 2.0 for count in overlaps.values())
        total_links = len(cluster) * (len(cluster) - 1) / 2.0
        numerator += importance * (correct_links / max(total_links, 1.0))
    return numerator, denominator


def _empty_coref_stats() -> Dict[str, float]:
    return {
        "pair_tp": 0.0,
        "pair_fp": 0.0,
        "pair_fn": 0.0,
        "muc_precision_num": 0.0,
        "muc_precision_den": 0.0,
        "muc_recall_num": 0.0,
        "muc_recall_den": 0.0,
        "b3_precision_sum": 0.0,
        "b3_recall_sum": 0.0,
        "b3_mentions": 0.0,
        "ceafe_similarity": 0.0,
        "ceafe_gold_total": 0.0,
        "ceafe_pred_total": 0.0,
        "lea_precision_num": 0.0,
        "lea_precision_den": 0.0,
        "lea_recall_num": 0.0,
        "lea_recall_den": 0.0,
    }


def _coref_metrics_from_stats(stats: Dict[str, float]) -> Dict[str, float]:
    muc_precision = stats["muc_precision_num"] / max(stats["muc_precision_den"], 1.0)
    muc_recall = stats["muc_recall_num"] / max(stats["muc_recall_den"], 1.0)
    b3_precision = stats["b3_precision_sum"] / max(stats["b3_mentions"], 1.0)
    b3_recall = stats["b3_recall_sum"] / max(stats["b3_mentions"], 1.0)
    ceafe_precision = stats["ceafe_similarity"] / max(stats["ceafe_pred_total"], 1.0)
    ceafe_recall = stats["ceafe_similarity"] / max(stats["ceafe_gold_total"], 1.0)
    lea_precision = stats["lea_precision_num"] / max(stats["lea_precision_den"], 1.0)
    lea_recall = stats["lea_recall_num"] / max(stats["lea_recall_den"], 1.0)
    return {
        "pairwise_f1": binary_f1_from_counts(
            int(stats["pair_tp"]),
            int(stats["pair_fp"]),
            int(stats["pair_fn"]),
        ),
        "muc_f1": _f1_from_pr(muc_precision, muc_recall),
        "b3_f1": _f1_from_pr(b3_precision, b3_recall),
        "ceafe_f1": _f1_from_pr(ceafe_precision, ceafe_recall),
        "lea_f1": _f1_from_pr(lea_precision, lea_recall),
    }


def _f1_from_pr(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class DocumentTaskEvaluator:
    def __init__(self, relation_label_count: int, norm_label_count: int, norm_id_to_type: Dict[int, str]) -> None:
        self.relation_label_count = relation_label_count
        self.norm_label_count = norm_label_count
        self.norm_id_to_type = norm_id_to_type

        self.coref_stats = _empty_coref_stats()
        self.coref_bucket_stats = defaultdict(_empty_coref_stats)

        self.relation_preds: List[int] = []
        self.relation_labels: List[int] = []
        self.relation_bucket_scores = defaultdict(lambda: {"preds": [], "labels": []})
        self.relation_page_scores = defaultdict(lambda: {"preds": [], "labels": []})

        self.norm_preds: List[int] = []
        self.norm_labels: List[int] = []
        self.norm_bucket_scores = defaultdict(lambda: {"preds": [], "labels": []})

        self.norm_value_exact = 0
        self.norm_value_total = 0
        self.recoverable_exact = 0
        self.recoverable_total = 0
        self.unrecoverable_format_exact = 0
        self.unrecoverable_total = 0
        self.norm_type_value_stats = defaultdict(
            lambda: {
                "exact": 0,
                "total": 0,
                "recoverable_exact": 0,
                "recoverable_total": 0,
            }
        )
        self.norm_bucket_value_stats = defaultdict(
            lambda: {
                "exact": 0,
                "total": 0,
                "recoverable_exact": 0,
                "recoverable_total": 0,
            }
        )

    def update(self, sample: Dict, entity_out: Dict, relation_out: Dict, attr_out: Dict) -> None:
        bucket_labels = _sample_bucket_labels(sample)
        self._update_coref(sample, entity_out, bucket_labels)
        self._update_relation(sample, relation_out, bucket_labels)
        self._update_norm(sample, attr_out, bucket_labels)

    def _update_coref(self, sample: Dict, entity_out: Dict, bucket_labels: Dict[str, str]) -> None:
        if entity_out["logits"] is None:
            return
        tp, fp, fn = binary_counts_from_logits(entity_out["logits"], entity_out["labels"])
        self.coref_stats["pair_tp"] += tp
        self.coref_stats["pair_fp"] += fp
        self.coref_stats["pair_fn"] += fn

        pred_clusters = _predicted_clusters_from_pairs(sample, entity_out["pairs"], entity_out["logits"])
        gold_clusters = _gold_clusters_from_sample(sample)
        self._add_coref_cluster_stats(self.coref_stats, gold_clusters, pred_clusters)

        for field_name, field_value in bucket_labels.items():
            bucket_key = f"{field_name}__{field_value}"
            bucket_stats = self.coref_bucket_stats[bucket_key]
            bucket_stats["pair_tp"] += tp
            bucket_stats["pair_fp"] += fp
            bucket_stats["pair_fn"] += fn
            self._add_coref_cluster_stats(bucket_stats, gold_clusters, pred_clusters)

    def _add_coref_cluster_stats(
        self,
        stats: Dict[str, float],
        gold_clusters: List[set[str]],
        pred_clusters: List[set[str]],
    ) -> None:
        muc_recall_num, muc_recall_den = _muc_num_den(gold_clusters, pred_clusters)
        muc_precision_num, muc_precision_den = _muc_num_den(pred_clusters, gold_clusters)
        b3_precision_sum, b3_recall_sum, b3_mentions = _bcub_num_den(gold_clusters, pred_clusters)
        ceafe_similarity, ceafe_gold_total, ceafe_pred_total = _ceafe_similarity(gold_clusters, pred_clusters)
        lea_recall_num, lea_recall_den = _lea_num_den(gold_clusters, pred_clusters)
        lea_precision_num, lea_precision_den = _lea_num_den(pred_clusters, gold_clusters)

        stats["muc_precision_num"] += muc_precision_num
        stats["muc_precision_den"] += muc_precision_den
        stats["muc_recall_num"] += muc_recall_num
        stats["muc_recall_den"] += muc_recall_den
        stats["b3_precision_sum"] += b3_precision_sum
        stats["b3_recall_sum"] += b3_recall_sum
        stats["b3_mentions"] += b3_mentions
        stats["ceafe_similarity"] += ceafe_similarity
        stats["ceafe_gold_total"] += ceafe_gold_total
        stats["ceafe_pred_total"] += ceafe_pred_total
        stats["lea_precision_num"] += lea_precision_num
        stats["lea_precision_den"] += lea_precision_den
        stats["lea_recall_num"] += lea_recall_num
        stats["lea_recall_den"] += lea_recall_den

    def _update_relation(self, sample: Dict, relation_out: Dict, bucket_labels: Dict[str, str]) -> None:
        if relation_out["logits"] is None:
            return
        preds = relation_out["logits"].argmax(dim=-1).tolist()
        labels = relation_out["labels"].tolist()
        self.relation_preds.extend(preds)
        self.relation_labels.extend(labels)

        for field_name, field_value in bucket_labels.items():
            bucket_key = f"{field_name}__{field_value}"
            self.relation_bucket_scores[bucket_key]["preds"].extend(preds)
            self.relation_bucket_scores[bucket_key]["labels"].extend(labels)

        for (src_idx, dst_idx, _), pred, gold in zip(relation_out["candidates"], preds, labels):
            src_page = sample["nodes"][src_idx].get("page_idx")
            dst_page = sample["nodes"][dst_idx].get("page_idx")
            page_bucket = "same_page" if src_page == dst_page else "cross_page"
            self.relation_page_scores[page_bucket]["preds"].append(pred)
            self.relation_page_scores[page_bucket]["labels"].append(gold)

    def _update_norm(self, sample: Dict, attr_out: Dict, bucket_labels: Dict[str, str]) -> None:
        if attr_out["logits"] is None:
            return
        preds = attr_out["logits"].argmax(dim=-1).tolist()
        labels = attr_out["labels"].tolist()
        self.norm_preds.extend(preds)
        self.norm_labels.extend(labels)

        for field_name, field_value in bucket_labels.items():
            bucket_key = f"{field_name}__{field_value}"
            self.norm_bucket_scores[bucket_key]["preds"].extend(preds)
            self.norm_bucket_scores[bucket_key]["labels"].extend(labels)

        for pred_type_id, target in zip(preds, attr_out["targets"]):
            pred_type = self.norm_id_to_type[pred_type_id]
            gold_type = target["norm_type"]
            pred_norm = normalize_value(target["raw_text"], pred_type)
            gold_norm = target["norm_value"]

            self.norm_value_total += 1
            self.norm_type_value_stats[gold_type]["total"] += 1
            if pred_norm == gold_norm:
                self.norm_value_exact += 1
                self.norm_type_value_stats[gold_type]["exact"] += 1

            if is_recoverable_value(target["raw_text"], gold_type):
                self.recoverable_total += 1
                self.norm_type_value_stats[gold_type]["recoverable_total"] += 1
                if pred_norm == gold_norm:
                    self.recoverable_exact += 1
                    self.norm_type_value_stats[gold_type]["recoverable_exact"] += 1
            else:
                self.unrecoverable_total += 1
                pred_surface = canonicalize_surface_form(target["raw_text"], pred_type)
                gold_surface = canonicalize_surface_form(target["raw_text"], gold_type)
                if pred_surface is not None and pred_surface == gold_surface:
                    self.unrecoverable_format_exact += 1

            for field_name, field_value in bucket_labels.items():
                bucket_key = f"{field_name}__{field_value}"
                bucket_stats = self.norm_bucket_value_stats[bucket_key]
                bucket_stats["total"] += 1
                if pred_norm == gold_norm:
                    bucket_stats["exact"] += 1
                if is_recoverable_value(target["raw_text"], gold_type):
                    bucket_stats["recoverable_total"] += 1
                    if pred_norm == gold_norm:
                        bucket_stats["recoverable_exact"] += 1

    def metrics(self) -> Dict[str, float]:
        relation_micro_tp = sum(
            1
            for pred, gold in zip(self.relation_preds, self.relation_labels)
            if pred == gold and gold != 0
        )
        relation_micro_fp = sum(
            1
            for pred, gold in zip(self.relation_preds, self.relation_labels)
            if pred != gold and pred != 0
        )
        relation_micro_fn = sum(
            1
            for pred, gold in zip(self.relation_preds, self.relation_labels)
            if pred != gold and gold != 0
        )

        metrics = {
            "coref_pairwise_f1": _coref_metrics_from_stats(self.coref_stats)["pairwise_f1"],
            "coref_muc_f1": _coref_metrics_from_stats(self.coref_stats)["muc_f1"],
            "coref_b3_f1": _coref_metrics_from_stats(self.coref_stats)["b3_f1"],
            "coref_ceafe_f1": _coref_metrics_from_stats(self.coref_stats)["ceafe_f1"],
            "coref_lea_f1": _coref_metrics_from_stats(self.coref_stats)["lea_f1"],
            "relation_macro_f1": macro_f1_from_predictions(
                self.relation_preds,
                self.relation_labels,
                num_classes=self.relation_label_count,
                ignore_index=0,
            ) if self.relation_labels else 0.0,
            "relation_micro_f1": binary_f1_from_counts(
                relation_micro_tp,
                relation_micro_fp,
                relation_micro_fn,
            ) if self.relation_labels else 0.0,
            "norm_type_macro_f1": macro_f1_from_predictions(
                self.norm_preds,
                self.norm_labels,
                num_classes=self.norm_label_count,
                ignore_index=None,
            ) if self.norm_labels else 0.0,
            "norm_type_accuracy": (
                sum(int(pred == gold) for pred, gold in zip(self.norm_preds, self.norm_labels)) / len(self.norm_labels)
                if self.norm_labels
                else 0.0
            ),
            "norm_value_exact_match": self.norm_value_exact / self.norm_value_total if self.norm_value_total else 0.0,
            "norm_value_exact_match_recoverable": (
                self.recoverable_exact / self.recoverable_total if self.recoverable_total else 0.0
            ),
            "norm_surface_format_accuracy_unrecoverable": (
                self.unrecoverable_format_exact / self.unrecoverable_total if self.unrecoverable_total else 0.0
            ),
        }

        entity_consolidation_f1 = metrics["coref_pairwise_f1"]
        semantic_linking_macro_f1 = metrics["relation_macro_f1"]
        semantic_linking_micro_f1 = metrics["relation_micro_f1"]
        attribute_type_macro_f1 = metrics["norm_type_macro_f1"]
        attribute_value_exact_match = metrics["norm_value_exact_match"]
        attribute_value_exact_match_recoverable = metrics["norm_value_exact_match_recoverable"]
        attribute_surface_accuracy_unrecoverable = metrics["norm_surface_format_accuracy_unrecoverable"]
        metrics.update(
            {
                "entity_consolidation_f1": entity_consolidation_f1,
                "semantic_linking_macro_f1": semantic_linking_macro_f1,
                "semantic_linking_micro_f1": semantic_linking_micro_f1,
                "attribute_type_macro_f1": attribute_type_macro_f1,
                "attribute_value_exact_match": attribute_value_exact_match,
                "attribute_value_exact_match_recoverable": attribute_value_exact_match_recoverable,
                "attribute_surface_accuracy_unrecoverable": attribute_surface_accuracy_unrecoverable,
                "kg_stage_macro_score": (
                    entity_consolidation_f1
                    + semantic_linking_macro_f1
                    + attribute_value_exact_match_recoverable
                )
                / 3.0,
            }
        )

        metrics.update(self._relation_breakdown_metrics())
        metrics.update(self._norm_breakdown_metrics())
        metrics.update(self._bucket_metrics())
        return metrics

    def _relation_breakdown_metrics(self) -> Dict[str, float]:
        metrics = {}
        label_names = {
            1: "refer_to",
            2: "caption_of",
            3: "contains",
        }
        for label_id, label_name in label_names.items():
            tp = sum(
                1 for pred, gold in zip(self.relation_preds, self.relation_labels) if pred == label_id and gold == label_id
            )
            fp = sum(
                1 for pred, gold in zip(self.relation_preds, self.relation_labels) if pred == label_id and gold != label_id
            )
            fn = sum(
                1 for pred, gold in zip(self.relation_preds, self.relation_labels) if pred != label_id and gold == label_id
            )
            metrics[f"relation_f1__{label_name}"] = binary_f1_from_counts(tp, fp, fn)

        for bucket_name, bucket_scores in self.relation_page_scores.items():
            metrics[f"relation_macro_f1__{bucket_name}"] = macro_f1_from_predictions(
                bucket_scores["preds"],
                bucket_scores["labels"],
                num_classes=self.relation_label_count,
                ignore_index=0,
            ) if bucket_scores["labels"] else 0.0
            tp = sum(
                1
                for pred, gold in zip(bucket_scores["preds"], bucket_scores["labels"])
                if pred == gold and gold != 0
            )
            fp = sum(
                1
                for pred, gold in zip(bucket_scores["preds"], bucket_scores["labels"])
                if pred != gold and pred != 0
            )
            fn = sum(
                1
                for pred, gold in zip(bucket_scores["preds"], bucket_scores["labels"])
                if pred != gold and gold != 0
            )
            metrics[f"relation_micro_f1__{bucket_name}"] = binary_f1_from_counts(tp, fp, fn)
        return metrics

    def _norm_breakdown_metrics(self) -> Dict[str, float]:
        metrics = {}
        id_to_type = {label_id: label_name for label_id, label_name in self.norm_id_to_type.items()}
        for label_id, label_name in id_to_type.items():
            tp = sum(1 for pred, gold in zip(self.norm_preds, self.norm_labels) if pred == label_id and gold == label_id)
            fp = sum(1 for pred, gold in zip(self.norm_preds, self.norm_labels) if pred == label_id and gold != label_id)
            fn = sum(1 for pred, gold in zip(self.norm_preds, self.norm_labels) if pred != label_id and gold == label_id)
            metrics[f"norm_type_f1__{label_name}"] = binary_f1_from_counts(tp, fp, fn)

        for norm_type, stats in self.norm_type_value_stats.items():
            metrics[f"norm_value_exact_match__{norm_type}"] = stats["exact"] / max(stats["total"], 1)
            metrics[f"norm_value_exact_match_recoverable__{norm_type}"] = (
                stats["recoverable_exact"] / max(stats["recoverable_total"], 1)
                if stats["recoverable_total"]
                else 0.0
            )
        return metrics

    def _bucket_metrics(self) -> Dict[str, float]:
        metrics = {}
        for bucket_name, stats in self.coref_bucket_stats.items():
            coref_metrics = _coref_metrics_from_stats(stats)
            metrics[f"coref_pairwise_f1__{bucket_name}"] = coref_metrics["pairwise_f1"]
            metrics[f"coref_muc_f1__{bucket_name}"] = coref_metrics["muc_f1"]
            metrics[f"coref_b3_f1__{bucket_name}"] = coref_metrics["b3_f1"]
            metrics[f"coref_ceafe_f1__{bucket_name}"] = coref_metrics["ceafe_f1"]
            metrics[f"coref_lea_f1__{bucket_name}"] = coref_metrics["lea_f1"]

        for bucket_name, bucket_scores in self.relation_bucket_scores.items():
            metrics[f"relation_macro_f1__{bucket_name}"] = macro_f1_from_predictions(
                bucket_scores["preds"],
                bucket_scores["labels"],
                num_classes=self.relation_label_count,
                ignore_index=0,
            ) if bucket_scores["labels"] else 0.0
            tp = sum(
                1
                for pred, gold in zip(bucket_scores["preds"], bucket_scores["labels"])
                if pred == gold and gold != 0
            )
            fp = sum(
                1
                for pred, gold in zip(bucket_scores["preds"], bucket_scores["labels"])
                if pred != gold and pred != 0
            )
            fn = sum(
                1
                for pred, gold in zip(bucket_scores["preds"], bucket_scores["labels"])
                if pred != gold and gold != 0
            )
            metrics[f"relation_micro_f1__{bucket_name}"] = binary_f1_from_counts(tp, fp, fn)

        for bucket_name, bucket_scores in self.norm_bucket_scores.items():
            metrics[f"norm_type_macro_f1__{bucket_name}"] = macro_f1_from_predictions(
                bucket_scores["preds"],
                bucket_scores["labels"],
                num_classes=self.norm_label_count,
                ignore_index=None,
            ) if bucket_scores["labels"] else 0.0
            value_stats = self.norm_bucket_value_stats[bucket_name]
            metrics[f"norm_value_exact_match__{bucket_name}"] = value_stats["exact"] / max(value_stats["total"], 1)
            metrics[f"norm_value_exact_match_recoverable__{bucket_name}"] = (
                value_stats["recoverable_exact"] / max(value_stats["recoverable_total"], 1)
                if value_stats["recoverable_total"]
                else 0.0
            )
        return metrics
