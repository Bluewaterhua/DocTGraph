"""Graph layer variants for the task-driven LayoutLMv3 GNN.

Three layers are exposed so that ablations can switch via the ``gnn_type``
config field without touching the main model:

* ``edge_aware`` - the original Doc2TGraph layer (edge embeddings + FiLM).
  Kept for backward compatibility.
* ``gatv2``     - GATv2 (Brody et al., 2022) with edge-type embeddings added
  to the attention key. A solid GNN baseline.
* ``hgt``       - a lightweight Heterogeneous Graph Transformer layer
  (per-edge-type attention bias, multi-head softmax over neighbours, FiLM
  task conditioning). This is the default and is the "best local module"
  we use for the final end-to-end system.

All layers expose the same signature:

    layer(node_embeddings, edge_index, edge_type, task_embedding=None)
    -> torch.Tensor  # same shape as node_embeddings

and compute messages on the directed edges given by ``edge_index`` (shape
``[2, E]``) and the integer ``edge_type`` vector of shape ``[E]``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Scatter helpers (no torch_scatter dependency)
# ---------------------------------------------------------------------------


def _scatter_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Per-destination softmax over neighbours.

    Args:
        src: ``[E, H]`` attention logits, one per edge per head.
        index: ``[E]`` destination node index for each edge.
        num_nodes: total number of nodes.

    Returns:
        ``[E, H]`` normalized attention weights so that, for any destination
        node ``v`` and head ``h``, ``sum_{e: dst(e)=v} out[e, h] == 1``.
    """
    if src.numel() == 0:
        return src
    num_heads = src.size(-1)
    index_exp = index.unsqueeze(-1).expand(-1, num_heads)
    max_vals = torch.full(
        (num_nodes, num_heads), float("-inf"), device=src.device, dtype=src.dtype
    )
    max_vals.scatter_reduce_(0, index_exp, src, reduce="amax", include_self=False)
    max_vals = torch.where(torch.isinf(max_vals), torch.zeros_like(max_vals), max_vals)
    shifted = src - max_vals[index]
    exp = shifted.exp()
    sums = torch.zeros(num_nodes, num_heads, device=src.device, dtype=src.dtype)
    sums.scatter_add_(0, index_exp, exp)
    return exp / sums[index].clamp_min(1e-12)


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Sum messages into destination nodes. ``src`` has shape ``[E, ...]``."""
    out_shape = (num_nodes,) + tuple(src.shape[1:])
    out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    index_exp = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, index_exp, src)
    return out


# ---------------------------------------------------------------------------
# Legacy edge-aware layer (kept so we can cite the original design in the
# ablation table). Identical semantics to the previous implementation.
# ---------------------------------------------------------------------------


class EdgeAwareGraphLayer(nn.Module):
    def __init__(self, hidden_size: int, num_edge_types: int, dropout: float) -> None:
        super().__init__()
        self.edge_embedding = nn.Embedding(num_edge_types, hidden_size)
        self.message_proj = nn.Linear(hidden_size, hidden_size)
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.edge_scale = nn.Parameter(torch.tensor(0.5))
        self.message_adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
        )
        self.update_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.update_adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return node_embeddings
        src = edge_index[0]
        dst = edge_index[1]
        edge_repr = self.edge_embedding(edge_type)
        messages = self.message_proj(node_embeddings[src] + edge_repr)
        if task_embedding is None:
            edge_weight = torch.ones(messages.size(0), 1, device=node_embeddings.device)
        else:
            task_repr = task_embedding.unsqueeze(0).expand(messages.size(0), -1)
            attention_input = torch.cat(
                [node_embeddings[src], node_embeddings[dst], edge_repr, task_repr],
                dim=-1,
            )
            edge_delta = torch.tanh(self.edge_attention(attention_input))
            edge_weight = 1.0 + self.edge_scale * edge_delta
            msg_gamma, msg_beta = self.message_adapter(task_embedding).chunk(2, dim=-1)
            messages = messages * (1.0 + 0.5 * torch.tanh(msg_gamma)).unsqueeze(0)
            messages = messages + msg_beta.unsqueeze(0)
        messages = messages * edge_weight

        aggregated = torch.zeros_like(node_embeddings)
        aggregated.index_add_(0, dst, messages)

        degree = torch.zeros(node_embeddings.size(0), device=node_embeddings.device)
        degree.index_add_(0, dst, edge_weight.squeeze(-1))
        aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)

        if task_embedding is not None:
            upd_gamma, upd_beta = self.update_adapter(task_embedding).chunk(2, dim=-1)
            aggregated = aggregated * (1.0 + 0.5 * torch.tanh(upd_gamma)).unsqueeze(0)
            aggregated = aggregated + upd_beta.unsqueeze(0)

        updated = self.update_proj(torch.cat([node_embeddings, aggregated], dim=-1))
        return self.norm(node_embeddings + updated)


# ---------------------------------------------------------------------------
# GATv2 with edge-type keys (Brody et al. 2022, "How attentive are GATs?").
# ---------------------------------------------------------------------------


class GATv2GraphLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_edge_types: int,
        dropout: float,
        num_heads: int = 4,
        use_task_film: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.lin_src = nn.Linear(hidden_size, hidden_size)
        self.lin_dst = nn.Linear(hidden_size, hidden_size)
        self.lin_edge = nn.Embedding(num_edge_types, hidden_size)
        self.att = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.att)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.use_task_film = use_task_film
        if use_task_film:
            self.task_film = nn.Linear(hidden_size, hidden_size * 2)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_nodes = node_embeddings.size(0)
        if edge_index.numel() == 0:
            return node_embeddings
        src = edge_index[0]
        dst = edge_index[1]

        x_src = self.lin_src(node_embeddings).view(num_nodes, self.num_heads, self.head_dim)
        x_dst = self.lin_dst(node_embeddings).view(num_nodes, self.num_heads, self.head_dim)
        e = self.lin_edge(edge_type).view(-1, self.num_heads, self.head_dim)

        combined = F.leaky_relu(x_src[src] + x_dst[dst] + e, negative_slope=0.2)
        # attention logits: [E, H]
        logits = (combined * self.att.unsqueeze(0)).sum(dim=-1)
        attn = _scatter_softmax(logits, dst, num_nodes)  # [E, H]
        attn = self.dropout(attn)

        # messages: [E, H, D]
        msg = x_src[src] * attn.unsqueeze(-1)
        aggregated = _scatter_sum(msg, dst, num_nodes)  # [N, H, D]
        aggregated = aggregated.reshape(num_nodes, self.hidden_size)
        aggregated = self.out_proj(aggregated)

        if self.use_task_film and task_embedding is not None:
            gamma, beta = self.task_film(task_embedding).chunk(2, dim=-1)
            aggregated = aggregated * (1.0 + 0.5 * torch.tanh(gamma)).unsqueeze(0)
            aggregated = aggregated + beta.unsqueeze(0)

        return self.norm(node_embeddings + aggregated)


# ---------------------------------------------------------------------------
# Heterogeneous Graph Transformer (HGT-lite).
#
# Based on Hu et al. 2020 (WWW). We use a simplified version that:
#   - projects Q / K / V with shared weights across edge types (keeps the
#     parameter count modest given LayoutLMv3's 768-d embeddings);
#   - adds a per-edge-type bias on the key vector and a scalar attention
#     bias, giving the layer enough capacity to model heterogeneity;
#   - applies multi-head scaled dot-product attention with a
#     per-destination softmax;
#   - optionally modulates messages with a task-conditioned FiLM, so it
#     still integrates with the task-driven router.
# ---------------------------------------------------------------------------


class HGTGraphLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_edge_types: int,
        dropout: float,
        num_heads: int = 4,
        use_task_film: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.num_edge_types = num_edge_types

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Per-edge-type additive biases on K/V and a scalar attention bias.
        self.edge_k_bias = nn.Embedding(num_edge_types, hidden_size)
        self.edge_v_bias = nn.Embedding(num_edge_types, hidden_size)
        self.edge_scalar_bias = nn.Embedding(num_edge_types, num_heads)
        # Per-edge-type gating scalar (learnable, initialized to 1) - acts
        # as a soft edge-type importance prior and is the reason the layer
        # can down-weight unhelpful edge types at inference.
        self.edge_gate = nn.Embedding(num_edge_types, num_heads)
        nn.init.constant_(self.edge_gate.weight, 1.0)
        nn.init.zeros_(self.edge_k_bias.weight)
        nn.init.zeros_(self.edge_v_bias.weight)
        nn.init.zeros_(self.edge_scalar_bias.weight)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

        self.use_task_film = use_task_film
        if use_task_film:
            self.task_film = nn.Linear(hidden_size, hidden_size * 2)

        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_nodes = node_embeddings.size(0)
        if edge_index.numel() == 0:
            return self.norm2(node_embeddings + self.ffn(node_embeddings))

        src = edge_index[0]
        dst = edge_index[1]

        q = self.q_proj(node_embeddings).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(node_embeddings).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(node_embeddings).view(num_nodes, self.num_heads, self.head_dim)

        # Edge-conditioned key / value / scalar bias.
        e_k = self.edge_k_bias(edge_type).view(-1, self.num_heads, self.head_dim)
        e_v = self.edge_v_bias(edge_type).view(-1, self.num_heads, self.head_dim)
        e_scalar = self.edge_scalar_bias(edge_type)          # [E, H]
        e_gate = self.edge_gate(edge_type)                   # [E, H]

        # [E, H]
        attn_logits = (q[dst] * (k[src] + e_k)).sum(dim=-1) * self.scale + e_scalar
        attn_logits = attn_logits + e_gate.log().clamp(min=-10.0, max=10.0)
        attn = _scatter_softmax(attn_logits, dst, num_nodes)
        attn = self.dropout(attn)

        # [E, H, D]
        msg = (v[src] + e_v) * attn.unsqueeze(-1)
        aggregated = _scatter_sum(msg, dst, num_nodes).reshape(num_nodes, self.hidden_size)
        aggregated = self.out_proj(aggregated)

        if self.use_task_film and task_embedding is not None:
            gamma, beta = self.task_film(task_embedding).chunk(2, dim=-1)
            aggregated = aggregated * (1.0 + 0.5 * torch.tanh(gamma)).unsqueeze(0)
            aggregated = aggregated + beta.unsqueeze(0)

        x = self.norm1(node_embeddings + aggregated)
        x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_gnn_layer(
    gnn_type: str,
    hidden_size: int,
    num_edge_types: int,
    dropout: float,
    num_heads: int = 4,
) -> nn.Module:
    """Return a graph layer module according to ``gnn_type``."""
    gnn_type = (gnn_type or "hgt").lower()
    if gnn_type == "edge_aware":
        return EdgeAwareGraphLayer(hidden_size, num_edge_types, dropout)
    if gnn_type == "gatv2":
        return GATv2GraphLayer(hidden_size, num_edge_types, dropout, num_heads=num_heads)
    if gnn_type == "hgt":
        return HGTGraphLayer(hidden_size, num_edge_types, dropout, num_heads=num_heads)
    raise ValueError(f"Unknown gnn_type={gnn_type!r}. Choose from edge_aware / gatv2 / hgt.")
