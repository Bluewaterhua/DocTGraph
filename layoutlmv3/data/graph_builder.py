from __future__ import annotations

import math
from typing import Dict, List, Tuple


NODE_KIND_TO_ID = {
    "block": 0,
    "mention": 1,
    "ref": 2,
    "value": 3,
    "object": 4,
}

RELATION_TO_ID = {
    "no_relation": 0,
    "refer_to": 1,
    "caption_of": 2,
    "contains": 3,
}

NORM_TYPE_TO_ID = {
    "contract_id": 0,
    "datetime": 1,
    "money": 2,
    "phone": 3,
    "tax_no": 4,
    "bank_account": 5,
    "email": 6,
}

EDGE_TYPE_TO_ID = {
    "parent_to_child": 0,
    "child_to_parent": 1,
    "spatial_knn": 2,
    "ref_to_object": 3,
    "same_parent": 4,
}

SPATIAL_ALLOWED_KIND_PAIRS = {
    ("block", "block"),
    ("block", "mention"),
    ("mention", "block"),
    ("mention", "mention"),
    ("block", "value"),
    ("value", "block"),
    ("value", "value"),
    ("block", "ref"),
    ("ref", "block"),
    ("ref", "object"),
    ("object", "ref"),
    ("block", "object"),
    ("object", "block"),
}

RELATION_BLOCK_PREFIXES = ("p_", "cap_")

TASK_EDGE_ALLOWED_TYPES = {
    "entity_consolidation": {"parent_to_child", "child_to_parent", "same_parent", "spatial_knn"},
    "semantic_linking": {"parent_to_child", "child_to_parent", "same_parent", "spatial_knn", "ref_to_object"},
    "attribute_canonicalization": {"parent_to_child", "child_to_parent", "same_parent", "spatial_knn"},
}

TASK_EDGE_ALLOWED_KINDS = {
    "entity_consolidation": {"mention", "block"},
    "semantic_linking": {"block", "ref", "object"},
    "attribute_canonicalization": {"value", "block"},
}


def _bbox_center(node: Dict) -> Tuple[float, float]:
    x0, y0, x1, y1 = node["bbox"]
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def build_graph(
    nodes: List[Dict],
    labels: Dict,
    knn_k: int = 4,
    use_parent_edges: bool = True,
    use_same_parent_edges: bool = True,
    use_ref_edges: bool = True,
) -> Dict:
    node_id_to_index = {node["node_id"]: idx for idx, node in enumerate(nodes)}
    edge_pairs: List[Tuple[int, int, int]] = []

    parent_to_children: Dict[int, List[int]] = {}
    if use_parent_edges or use_same_parent_edges:
        for idx, node in enumerate(nodes):
            parent_id = node.get("parent_element_id")
            if parent_id and parent_id in node_id_to_index:
                parent_idx = node_id_to_index[parent_id]
                if use_parent_edges:
                    edge_pairs.append((parent_idx, idx, EDGE_TYPE_TO_ID["parent_to_child"]))
                    edge_pairs.append((idx, parent_idx, EDGE_TYPE_TO_ID["child_to_parent"]))
                parent_to_children.setdefault(parent_idx, []).append(idx)

    if use_same_parent_edges:
        for _, children in parent_to_children.items():
            for src_pos in range(len(children)):
                for dst_pos in range(len(children)):
                    if src_pos == dst_pos:
                        continue
                    edge_pairs.append((children[src_pos], children[dst_pos], EDGE_TYPE_TO_ID["same_parent"]))

    centers = [_bbox_center(node) for node in nodes]
    for src_idx, src_center in enumerate(centers):
        src_kind = nodes[src_idx]["kind"]
        distances = []
        for dst_idx, dst_center in enumerate(centers):
            if src_idx == dst_idx:
                continue
            dst_kind = nodes[dst_idx]["kind"]
            if (src_kind, dst_kind) not in SPATIAL_ALLOWED_KIND_PAIRS:
                continue
            dist = math.dist(src_center, dst_center)
            distances.append((dist, dst_idx))
        distances.sort(key=lambda item: item[0])
        for _, dst_idx in distances[:knn_k]:
            edge_pairs.append((src_idx, dst_idx, EDGE_TYPE_TO_ID["spatial_knn"]))

    if use_ref_edges:
        for node in nodes:
            if node["kind"] == "ref" and node.get("target_obj") in node_id_to_index:
                src_idx = node_id_to_index[node["node_id"]]
                dst_idx = node_id_to_index[node["target_obj"]]
                edge_pairs.append((src_idx, dst_idx, EDGE_TYPE_TO_ID["ref_to_object"]))

    seen_edges = set()
    inv_edge_type = {v: k for k, v in EDGE_TYPE_TO_ID.items()}
    edge_index = [[], []]
    edge_type = []
    edge_meta: List[Tuple[str, str, str]] = []
    for src_idx, dst_idx, type_id in edge_pairs:
        key = (src_idx, dst_idx, type_id)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edge_index[0].append(src_idx)
        edge_index[1].append(dst_idx)
        edge_type.append(type_id)
        edge_meta.append((nodes[src_idx]["kind"], nodes[dst_idx]["kind"], inv_edge_type[type_id]))

    task_edge_masks: Dict[str, List[bool]] = {}
    for task_name, allowed_types in TASK_EDGE_ALLOWED_TYPES.items():
        allowed_kinds = TASK_EDGE_ALLOWED_KINDS[task_name]
        mask = []
        for src_kind, dst_kind, edge_type_name in edge_meta:
            keep_type = edge_type_name in allowed_types
            keep_kinds = (src_kind in allowed_kinds) and (dst_kind in allowed_kinds)
            mask.append(keep_type and keep_kinds)
        if not any(mask):
            mask = [True] * len(edge_meta)
        task_edge_masks[task_name] = mask

    relation_lookup = {}
    for item in labels["relations"]:
        relation_lookup[(item["h"], item["t"])] = RELATION_TO_ID[item["r"]]

    mention_indices = [idx for idx, node in enumerate(nodes) if node["kind"] == "mention"]
    entity_consolidation_pairs = []
    for left in range(len(mention_indices)):
        for right in range(left + 1, len(mention_indices)):
            idx_a = mention_indices[left]
            idx_b = mention_indices[right]
            same_entity = int(nodes[idx_a].get("entity_id") == nodes[idx_b].get("entity_id"))
            entity_consolidation_pairs.append((idx_a, idx_b, same_entity))

    semantic_link_candidates = []
    for src_idx, src_node in enumerate(nodes):
        if src_node["kind"] == "block":
            if not src_node["node_id"].startswith(RELATION_BLOCK_PREFIXES):
                continue
        elif src_node["kind"] != "ref":
            continue
        for dst_idx, dst_node in enumerate(nodes):
            if dst_node["kind"] != "object":
                continue
            label_id = relation_lookup.get((src_node["node_id"], dst_node["node_id"]), 0)
            semantic_link_candidates.append((src_idx, dst_idx, label_id))

    attribute_canonicalization_targets = []
    norm_lookup = {item["value_id"]: item for item in labels["normalization"]}
    for idx, node in enumerate(nodes):
        if node["kind"] != "value":
            continue
        target = norm_lookup.get(node["node_id"])
        if target is None:
            continue
        attribute_canonicalization_targets.append(
            {
                "node_idx": idx,
                "norm_type_id": NORM_TYPE_TO_ID[target["type"]],
                "norm_type": target["type"],
                "norm_value": target["norm"],
                "raw_text": node["text"],
            }
        )

    return {
        "node_type_ids": [NODE_KIND_TO_ID[node["kind"]] for node in nodes],
        "edge_index": edge_index,
        "edge_type": edge_type,
        "task_edge_masks": task_edge_masks,
        "entity_consolidation_pairs": entity_consolidation_pairs,
        "semantic_link_candidates": semantic_link_candidates,
        "attribute_canonicalization_targets": attribute_canonicalization_targets,
        # Backward-compatible aliases for older training code or checkpoints.
        "coref_pairs": entity_consolidation_pairs,
        "relation_candidates": semantic_link_candidates,
        "normalization_targets": attribute_canonicalization_targets,
    }
