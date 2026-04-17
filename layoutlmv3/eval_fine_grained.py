"""Fine-grained evaluation over an existing checkpoint.

Dumps:
  * overall metrics (same as training-time evaluation),
  * per-relation-type F1 (refer_to / caption_of / contains),
  * per-norm-type value EM and recoverable EM (7 types),
  * bucketed metrics keyed by meta fields (noise_level, coref_difficulty,
    ref_difficulty, layout_profile, page_count_bucket),
  * MUC / B-cubed / CEAFe / LEA for coreference,
  * same_page vs cross_page relation breakdown.

Example usage (on the target server, after copying the checkpoint):

    python -m layoutlmv3.eval_fine_grained \
        --config layoutlmv3/configs/task_driven_layoutlmv3_gnn_train10k_mask_subgraph_hgt.json \
        --checkpoint layoutlmv3/outputs/train10k_mask_subgraph_hgt/seed_42/best.pt \
        --out layoutlmv3/outputs/train10k_mask_subgraph_hgt/seed_42/fine_grained.json \
        --split val

``--split`` may be ``val`` (default, uses the same val split produced by the
config + seed) or ``all`` to evaluate over the entire dataset_root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from layoutlmv3.data.collator import HtmlGraphCollator
from layoutlmv3.data.dataset import HtmlGraphDataset
from layoutlmv3.data.graph_builder import NORM_TYPE_TO_ID, RELATION_TO_ID
from layoutlmv3.evaluation import DocumentTaskEvaluator
from layoutlmv3.models.task_driven_layoutlmv3_gnn import TaskDrivenLayoutLMv3GNN
from layoutlmv3.train import (
    NORM_ID_TO_TYPE,
    evaluate,
    make_dataloader,
    move_batch_to_device,
    split_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=["val", "all"], default="val")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed used to re-derive the val split; defaults to config['seed'] or 42.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    dataset = HtmlGraphDataset(
        dataset_root=cfg["dataset_root"],
        graph_knn_k=cfg["graph_knn_k"],
        use_parent_edges=cfg.get("use_parent_edges", True),
        use_same_parent_edges=cfg.get("use_same_parent_edges", True),
        use_ref_edges=cfg.get("use_ref_edges", True),
    )
    collator = HtmlGraphCollator(
        model_name_or_path=cfg["model_name_or_path"],
        max_length=cfg["max_length"],
    )

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    if args.split == "val":
        _, val_indices = split_indices(
            dataset=dataset,
            val_ratio=cfg.get("val_ratio", 0.1),
            seed=seed,
            split_mode=cfg.get("split_mode", "random"),
            split_field=cfg.get("split_field"),
            val_values=cfg.get("val_values"),
        )
    else:
        val_indices = list(range(len(dataset)))
    loader = make_dataloader(
        dataset,
        val_indices,
        collator,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskDrivenLayoutLMv3GNN(
        model_name_or_path=cfg["model_name_or_path"],
        gnn_layers=cfg["gnn_layers"],
        dropout=cfg["dropout"],
        task_loss_weights=cfg.get("task_loss_weights"),
        router_mode=cfg.get("router_mode", "mask"),
        gnn_type=cfg.get("gnn_type", "hgt"),
        gnn_num_heads=cfg.get("gnn_num_heads", 4),
        freeze_encoder=cfg.get("freeze_encoder", False),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing:
        print(f"warn_missing_keys count={len(missing)}", flush=True)
    if unexpected:
        print(f"warn_unexpected_keys count={len(unexpected)}", flush=True)

    metrics = evaluate(model, loader, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_docs": len(val_indices),
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"fine_grained_metrics_written path={out_path} num_docs={len(val_indices)}", flush=True)

    # Pretty-print the most useful breakdowns.
    print("=== overall ===", flush=True)
    for key in (
        "kg_stage_macro_score",
        "entity_consolidation_f1",
        "coref_muc_f1",
        "coref_b3_f1",
        "coref_ceafe_f1",
        "coref_lea_f1",
        "semantic_linking_macro_f1",
        "semantic_linking_micro_f1",
        "attribute_type_macro_f1",
        "attribute_value_exact_match",
        "attribute_value_exact_match_recoverable",
        "attribute_surface_accuracy_unrecoverable",
    ):
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}", flush=True)

    print("=== relation per-type F1 ===", flush=True)
    for key, value in sorted(metrics.items()):
        if key.startswith("relation_f1_"):
            print(f"  {key}: {value:.4f}", flush=True)

    print("=== norm per-type ===", flush=True)
    for key, value in sorted(metrics.items()):
        if key.startswith("norm_value_em_") or key.startswith("norm_type_f1_"):
            print(f"  {key}: {value:.4f}", flush=True)

    print("=== bucketed metrics (noise/coref/ref/layout/page) ===", flush=True)
    for key, value in sorted(metrics.items()):
        if "__" in key and any(
            key.startswith(prefix)
            for prefix in (
                "coref_",
                "relation_",
                "norm_value_",
                "norm_type_",
            )
        ):
            print(f"  {key}: {value:.4f}", flush=True)


if __name__ == "__main__":
    main()
