from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMv3Model

from layoutlmv3.data.graph_builder import EDGE_TYPE_TO_ID, NORM_TYPE_TO_ID, RELATION_TO_ID
from layoutlmv3.models.gnn_layers import EdgeAwareGraphLayer, build_gnn_layer


TASK_TO_ID = {
    "entity_consolidation": 0,
    "semantic_linking": 1,
    "attribute_canonicalization": 2,
}

TASK_NAME_ALIASES = {
    "coref": "entity_consolidation",
    "relation": "semantic_linking",
    "normalization": "attribute_canonicalization",
}


def canonical_task_name(task_name: str) -> str:
    return TASK_NAME_ALIASES.get(task_name, task_name)

class TaskConditioner(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.task_embedding = nn.Embedding(len(TASK_TO_ID), hidden_size)

    def get_task_embedding(self, task_name: str) -> torch.Tensor:
        return self.task_embedding.weight[TASK_TO_ID[canonical_task_name(task_name)]]

    def condition_nodes(
        self,
        node_embeddings: torch.Tensor,
        task_name: str,
        router_mode: str = "mask",
    ) -> torch.Tensor:
        if router_mode == "none":
            return node_embeddings
        task_embed = self.get_task_embedding(task_name)
        if router_mode == "token":
            return node_embeddings + task_embed.unsqueeze(0)
        if router_mode in {"mask", "subgraph", "mask_subgraph"}:
            return node_embeddings
        if router_mode not in {"mask", "subgraph", "mask_subgraph"}:
            raise ValueError(f"Unsupported router_mode: {router_mode}")
        return node_embeddings


class EntityConsolidationHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, pairs: List[Tuple[int, int, int]]) -> Dict:
        if not pairs:
            return {
                "loss": node_embeddings.new_tensor(0.0),
                "logits": None,
                "labels": None,
                "pairs": pairs,
            }
        feats = []
        labels = []
        for idx_a, idx_b, label in pairs:
            emb_a = node_embeddings[idx_a]
            emb_b = node_embeddings[idx_b]
            feats.append(torch.cat([emb_a, emb_b, torch.abs(emb_a - emb_b), emb_a * emb_b], dim=-1))
            labels.append(label)
        feat_tensor = torch.stack(feats)
        label_tensor = torch.tensor(labels, dtype=torch.float, device=node_embeddings.device)
        logits = self.scorer(feat_tensor).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, label_tensor)
        return {"loss": loss, "logits": logits, "labels": label_tensor, "pairs": pairs}


class SemanticLinkingHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(RELATION_TO_ID)),
        )

    def forward(self, node_embeddings: torch.Tensor, candidates: List[Tuple[int, int, int]]) -> Dict:
        if not candidates:
            return {
                "loss": node_embeddings.new_tensor(0.0),
                "logits": None,
                "labels": None,
                "candidates": candidates,
            }
        feats = []
        labels = []
        for src_idx, dst_idx, label in candidates:
            src = node_embeddings[src_idx]
            dst = node_embeddings[dst_idx]
            feats.append(torch.cat([src, dst, torch.abs(src - dst), src * dst], dim=-1))
            labels.append(label)
        feat_tensor = torch.stack(feats)
        label_tensor = torch.tensor(labels, dtype=torch.long, device=node_embeddings.device)
        logits = self.classifier(feat_tensor)
        loss = F.cross_entropy(logits, label_tensor)
        return {"loss": loss, "logits": logits, "labels": label_tensor, "candidates": candidates}


class AttributeCanonicalizationHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(NORM_TYPE_TO_ID)),
        )

    def forward(self, node_embeddings: torch.Tensor, targets: List[Dict]) -> Dict:
        if not targets:
            return {"loss": node_embeddings.new_tensor(0.0), "logits": None, "labels": None}
        node_indices = [item["node_idx"] for item in targets]
        label_tensor = torch.tensor(
            [item["norm_type_id"] for item in targets],
            dtype=torch.long,
            device=node_embeddings.device,
        )
        logits = self.type_classifier(node_embeddings[node_indices])
        loss = F.cross_entropy(logits, label_tensor)
        return {
            "loss": loss,
            "logits": logits,
            "labels": label_tensor,
            "targets": targets,
        }


class TaskDrivenLayoutLMv3GNN(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        gnn_layers: int = 2,
        dropout: float = 0.1,
        task_loss_weights: Dict[str, float] | None = None,
        router_mode: str = "mask",
        gnn_type: str = "hgt",
        gnn_num_heads: int = 4,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = LayoutLMv3Model.from_pretrained(model_name_or_path)
        hidden_size = self.encoder.config.hidden_size
        self.router_mode = router_mode
        self.gnn_type = gnn_type
        self.task_loss_weights = task_loss_weights or {
            "entity_consolidation": 1.0,
            "semantic_linking": 1.0,
            "attribute_canonicalization": 1.0,
        }
        for legacy_name, canonical_name in TASK_NAME_ALIASES.items():
            if legacy_name in self.task_loss_weights and canonical_name not in self.task_loss_weights:
                self.task_loss_weights[canonical_name] = self.task_loss_weights[legacy_name]
        self.node_type_embedding = nn.Embedding(5, hidden_size)
        self.task_conditioner = TaskConditioner(hidden_size=hidden_size)
        self.gnn_layers = nn.ModuleList(
            [
                build_gnn_layer(
                    gnn_type=gnn_type,
                    hidden_size=hidden_size,
                    num_edge_types=len(EDGE_TYPE_TO_ID),
                    dropout=dropout,
                    num_heads=gnn_num_heads,
                )
                for _ in range(gnn_layers)
            ]
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.entity_consolidation_head = EntityConsolidationHead(hidden_size, dropout)
        self.semantic_linking_head = SemanticLinkingHead(hidden_size, dropout)
        self.attribute_canonicalization_head = AttributeCanonicalizationHead(hidden_size, dropout)

    def _pool_tokens_to_nodes(
        self,
        token_embeddings: torch.Tensor,
        word_id_map: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        hidden_size = token_embeddings.size(-1)
        pooled = torch.zeros(num_nodes, hidden_size, device=token_embeddings.device)
        counts = torch.zeros(num_nodes, device=token_embeddings.device)

        for token_idx, word_id in enumerate(word_id_map.tolist()):
            if word_id < 0 or word_id >= num_nodes:
                continue
            pooled[word_id] += token_embeddings[token_idx]
            counts[word_id] += 1.0

        counts = counts.clamp_min(1.0).unsqueeze(-1)
        return pooled / counts

    def _encode_sample(self, batch: Dict, sample_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"][sample_idx].unsqueeze(0)
        attention_mask = batch["attention_mask"][sample_idx].unsqueeze(0)
        bbox = batch["bbox"][sample_idx].unsqueeze(0)
        pixel_values = batch["pixel_values"][sample_idx].unsqueeze(0)
        word_id_map = batch["word_id_maps"][sample_idx]
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
        )
        sample = batch["samples"][sample_idx]
        token_embeddings = outputs.last_hidden_state.squeeze(0)
        node_embeddings = self._pool_tokens_to_nodes(
            token_embeddings=token_embeddings,
            word_id_map=word_id_map,
            num_nodes=len(sample["nodes"]),
        )
        node_type_ids = torch.tensor(sample["graph"]["node_type_ids"], dtype=torch.long, device=token_embeddings.device)
        return node_embeddings + self.node_type_embedding(node_type_ids)

    def _run_gnn(self, task_embeddings: torch.Tensor, sample: Dict, task_name: str) -> torch.Tensor:
        edge_index = torch.tensor(sample["graph"]["edge_index"], dtype=torch.long, device=task_embeddings.device)
        edge_type = torch.tensor(sample["graph"]["edge_type"], dtype=torch.long, device=task_embeddings.device)
        if self.router_mode in {"subgraph", "mask_subgraph"}:
            task_masks = sample["graph"].get("task_edge_masks")
            if task_masks and task_name in task_masks:
                task_mask = torch.tensor(task_masks[task_name], dtype=torch.bool, device=task_embeddings.device)
                if task_mask.numel() == edge_type.numel() and task_mask.any():
                    edge_index = edge_index[:, task_mask]
                    edge_type = edge_type[task_mask]
        hidden = task_embeddings
        task_embedding = None
        if self.router_mode in {"mask", "mask_subgraph"}:
            task_embedding = self.task_conditioner.get_task_embedding(task_name)
        for layer in self.gnn_layers:
            hidden = layer(
                hidden,
                edge_index,
                edge_type,
                task_embedding=task_embedding,
            )
        return hidden

    def forward(self, batch: Dict) -> Dict:
        total_loss = None
        task_outputs = {
            "entity_consolidation": [],
            "semantic_linking": [],
            "attribute_canonicalization": [],
        }

        for sample_idx, sample in enumerate(batch["samples"]):
            shared_node_embeddings = self._encode_sample(batch, sample_idx)
            entity_node_embeddings = shared_node_embeddings
            linking_node_embeddings = shared_node_embeddings
            attr_node_embeddings = shared_node_embeddings
            entity_hidden = self._run_gnn(
                self.task_conditioner.condition_nodes(entity_node_embeddings, "entity_consolidation", self.router_mode),
                sample,
                "entity_consolidation",
            )
            linking_hidden = self._run_gnn(
                self.task_conditioner.condition_nodes(linking_node_embeddings, "semantic_linking", self.router_mode),
                sample,
                "semantic_linking",
            )
            attr_hidden = self._run_gnn(
                self.task_conditioner.condition_nodes(attr_node_embeddings, "attribute_canonicalization", self.router_mode),
                sample,
                "attribute_canonicalization",
            )

            entity_pairs = sample["graph"].get("entity_consolidation_pairs", sample["graph"]["coref_pairs"])
            semantic_links = sample["graph"].get("semantic_link_candidates", sample["graph"]["relation_candidates"])
            attribute_targets = sample["graph"].get(
                "attribute_canonicalization_targets",
                sample["graph"]["normalization_targets"],
            )

            entity_out = self.entity_consolidation_head(entity_hidden, entity_pairs)
            linking_out = self.semantic_linking_head(linking_hidden, semantic_links)
            attr_out = self.attribute_canonicalization_head(attr_hidden, attribute_targets)

            sample_loss = (
                self.task_loss_weights["entity_consolidation"] * entity_out["loss"]
                + self.task_loss_weights["semantic_linking"] * linking_out["loss"]
                + self.task_loss_weights["attribute_canonicalization"] * attr_out["loss"]
            )
            total_loss = sample_loss if total_loss is None else total_loss + sample_loss

            task_outputs["entity_consolidation"].append(entity_out)
            task_outputs["semantic_linking"].append(linking_out)
            task_outputs["attribute_canonicalization"].append(attr_out)

        if total_loss is None:
            total_loss = torch.tensor(0.0, device=batch["input_ids"].device)
        task_outputs["coref"] = task_outputs["entity_consolidation"]
        task_outputs["relation"] = task_outputs["semantic_linking"]
        task_outputs["normalization"] = task_outputs["attribute_canonicalization"]
        return {"loss": total_loss / max(len(batch["samples"]), 1), "tasks": task_outputs}
