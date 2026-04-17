"""Training entry point for the task-driven LayoutLMv3 + GNN model.

Upgrades over the previous script:

* ``--seeds 42,123,314`` runs a full training trajectory per seed and writes
  results under ``output_dir/seed_<N>/``; ``summary.json`` at the parent
  directory aggregates mean / std of the primary metrics.
* Early stopping based on ``early_stopping_metric`` (default
  ``kg_stage_macro_score``, higher-is-better). ``--patience`` controls how
  many epochs without improvement we wait.
* Only the best checkpoint (``best.pt``) is kept, plus a running
  ``last.pt``; per-epoch metrics are appended to ``metrics.jsonl``.
* Supports ``gnn_type`` / ``gnn_num_heads`` / ``freeze_encoder`` in the
  config so that ablations can reuse this script unmodified.
* Optional gradient clipping via ``grad_clip`` and mixed precision via
  ``amp: true``.

The CLI remains backward compatible: a single ``--config path.json`` call
still works and, if the config does not set ``seeds``, falls back to the
legacy ``seed`` field.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from statistics import mean

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from layoutlmv3.data.collator import HtmlGraphCollator
from layoutlmv3.data.dataset import HtmlGraphDataset
from layoutlmv3.data.graph_builder import NORM_TYPE_TO_ID, RELATION_TO_ID
from layoutlmv3.evaluation import DocumentTaskEvaluator
from layoutlmv3.models.task_driven_layoutlmv3_gnn import TaskDrivenLayoutLMv3GNN


NORM_ID_TO_TYPE = {
    0: "contract_id",
    1: "datetime",
    2: "money",
    3: "phone",
    4: "tax_no",
    5: "bank_account",
    6: "email",
}

# Metrics we aggregate across seeds in summary.json.
SUMMARY_METRIC_KEYS = (
    "val_loss",
    "kg_stage_macro_score",
    "entity_consolidation_f1",
    "semantic_linking_macro_f1",
    "semantic_linking_micro_f1",
    "attribute_type_macro_f1",
    "attribute_value_exact_match",
    "attribute_value_exact_match_recoverable",
    "attribute_surface_accuracy_unrecoverable",
    "coref_muc_f1",
    "coref_b3_f1",
    "coref_ceafe_f1",
    "coref_lea_f1",
)


def load_config(config_path: str) -> dict:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    default_config = (
        Path(__file__).resolve().parent / "configs" / "task_driven_layoutlmv3_gnn.json"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated list of seeds (overrides config['seeds'] / config['seed']).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (overrides config['patience']).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override config['num_epochs'].",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build model, run a single train step and a single eval step, then exit.",
    )
    return parser.parse_args()


def split_indices(
    dataset: HtmlGraphDataset,
    val_ratio: float,
    seed: int,
    split_mode: str = "random",
    split_field: str | None = None,
    val_values: list[str] | None = None,
) -> tuple[list[int], list[int]]:
    size = len(dataset)
    indices = list(range(size))
    if split_mode in {"holdout", "metadata_holdout"}:
        if not split_field:
            raise ValueError("split_field is required when split_mode is 'holdout'")
        if not val_values:
            raise ValueError("val_values is required when split_mode is 'holdout'")
        normalized_val_values = {str(value) for value in val_values}
        val_indices = []
        train_indices = []
        for idx, meta in enumerate(dataset.doc_metas):
            meta_value = str(meta.get(split_field, ""))
            if meta_value in normalized_val_values:
                val_indices.append(idx)
            else:
                train_indices.append(idx)
        if not train_indices or not val_indices:
            raise ValueError(
                f"Holdout split produced an empty partition for split_field={split_field} val_values={val_values}"
            )
        return train_indices, val_indices

    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(size * val_ratio)) if size > 1 and val_ratio > 0 else 0
    if val_size >= size:
        val_size = size - 1
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def make_dataloader(dataset, indices, collator, batch_size, num_workers, shuffle):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    for key in ("input_ids", "attention_mask", "bbox", "pixel_values", "word_id_maps"):
        batch[key] = batch[key].to(device)
    if "task_prompt_encodings" in batch:
        for task_encoding in batch["task_prompt_encodings"].values():
            for key in ("input_ids", "attention_mask", "bbox", "pixel_values", "word_id_maps"):
                task_encoding[key] = task_encoding[key].to(device)
    return batch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: TaskDrivenLayoutLMv3GNN, dataloader: DataLoader, device: torch.device) -> dict:
    model.eval()
    losses = []
    evaluator = DocumentTaskEvaluator(
        relation_label_count=len(RELATION_TO_ID),
        norm_label_count=len(NORM_TYPE_TO_ID),
        norm_id_to_type=NORM_ID_TO_TYPE,
    )
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        losses.append(outputs["loss"].item())
        for sample_idx, sample in enumerate(batch["samples"]):
            evaluator.update(
                sample=sample,
                entity_out=outputs["tasks"]["entity_consolidation"][sample_idx],
                relation_out=outputs["tasks"]["semantic_linking"][sample_idx],
                attr_out=outputs["tasks"]["attribute_canonicalization"][sample_idx],
            )
    metrics = {"val_loss": mean(losses) if losses else 0.0}
    metrics.update(evaluator.metrics())
    model.train()
    return metrics


def _dump_metrics(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _save_best_checkpoint(
    model: torch.nn.Module, output_dir: Path, epoch: int, metrics: dict, cfg: dict
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "best.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )
    return path


def train_one_seed(cfg: dict, seed: int, seed_output_dir: Path, dry_run: bool = False) -> dict:
    """Run a full training trajectory for a single seed and return the best metrics."""
    set_global_seed(seed)

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

    train_indices, val_indices = split_indices(
        dataset=dataset,
        val_ratio=cfg.get("val_ratio", 0.1),
        seed=seed,
        split_mode=cfg.get("split_mode", "random"),
        split_field=cfg.get("split_field"),
        val_values=cfg.get("val_values"),
    )
    train_loader = make_dataloader(
        dataset,
        train_indices,
        collator,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
    )
    val_loader = make_dataloader(
        dataset,
        val_indices,
        collator,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
    ) if val_indices else None

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

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    grad_clip = float(cfg.get("grad_clip", 0.0))

    metric_name = cfg.get("early_stopping_metric", "kg_stage_macro_score")
    metric_mode = cfg.get("early_stopping_mode", "max")
    patience = int(cfg.get("patience", 3))
    max_epochs = int(cfg["num_epochs"])

    best_score = float("-inf") if metric_mode == "max" else float("inf")
    best_metrics: dict = {}
    best_epoch = -1
    epochs_without_improve = 0

    print(
        " ".join(
            [
                f"seed={seed}",
                f"device={device}",
                f"train_docs={len(train_indices)}",
                f"val_docs={len(val_indices)}",
                f"split_mode={cfg.get('split_mode', 'random')}",
                f"router_mode={cfg.get('router_mode', 'mask')}",
                f"gnn_type={cfg.get('gnn_type', 'hgt')}",
                f"gnn_layers={cfg['gnn_layers']}",
                f"freeze_encoder={cfg.get('freeze_encoder', False)}",
            ]
        ),
        flush=True,
    )

    model.train()
    metrics_path = seed_output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        epoch_losses = []
        total_steps = len(train_loader)
        print(f"epoch_start epoch={epoch}/{max_epochs} total_steps={total_steps}", flush=True)
        for step, batch in enumerate(train_loader, 1):
            step_start = time.time()
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(batch)
                scaler.scale(outputs["loss"]).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch)
                outputs["loss"].backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            loss_value = outputs["loss"].item()
            epoch_losses.append(loss_value)
            if step == 1 or step % max(1, total_steps // 20) == 0 or step == total_steps:
                print(
                    " ".join(
                        [
                            f"train_step epoch={epoch}",
                            f"step={step}/{total_steps}",
                            f"loss={loss_value:.4f}",
                            f"elapsed={time.time() - step_start:.1f}s",
                        ]
                    ),
                    flush=True,
                )
            if dry_run:
                break

        train_loss = mean(epoch_losses) if epoch_losses else 0.0
        print(
            f"epoch_train_done epoch={epoch} train_loss={train_loss:.4f} "
            f"elapsed={time.time() - epoch_start:.1f}s",
            flush=True,
        )

        metrics = {"epoch": epoch, "train_loss": train_loss, "seed": seed}
        if val_loader is not None:
            val_start = time.time()
            val_metrics = evaluate(model, val_loader, device)
            metrics.update(val_metrics)
            print(
                " ".join(
                    [
                        f"eval_done epoch={epoch}",
                        f"val_loss={val_metrics['val_loss']:.4f}",
                        f"kg_stage_macro={val_metrics['kg_stage_macro_score']:.4f}",
                        f"ent_f1={val_metrics['entity_consolidation_f1']:.4f}",
                        f"link_macro={val_metrics['semantic_linking_macro_f1']:.4f}",
                        f"norm_rec_em={val_metrics['attribute_value_exact_match_recoverable']:.4f}",
                        f"elapsed={time.time() - val_start:.1f}s",
                    ]
                ),
                flush=True,
            )

            score = val_metrics.get(metric_name, 0.0)
            improved = (score > best_score) if metric_mode == "max" else (score < best_score)
            if improved:
                best_score = score
                best_epoch = epoch
                best_metrics = dict(metrics)
                epochs_without_improve = 0
                _save_best_checkpoint(model, seed_output_dir, epoch, metrics, cfg)
                print(
                    f"best_checkpoint_updated seed={seed} epoch={epoch} "
                    f"{metric_name}={score:.4f}",
                    flush=True,
                )
            else:
                epochs_without_improve += 1
                print(
                    f"no_improvement seed={seed} epoch={epoch} "
                    f"{metric_name}={score:.4f} patience={epochs_without_improve}/{patience}",
                    flush=True,
                )

        _dump_metrics(metrics_path, metrics)

        if dry_run:
            break
        if epochs_without_improve >= patience:
            print(f"early_stopping seed={seed} epoch={epoch}", flush=True)
            break

    # Save final state so the run is resumable / inspectable.
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics, "config": cfg},
        seed_output_dir / "last.pt",
    )
    # Free model memory before moving to next seed.
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
    }


def aggregate_summary(per_seed: list[dict]) -> dict:
    summary = {"per_seed": per_seed, "aggregate": {}}
    for key in SUMMARY_METRIC_KEYS:
        values = [float(item["best_metrics"].get(key, 0.0)) for item in per_seed if item["best_metrics"]]
        if not values:
            continue
        summary["aggregate"][key] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
            "values": values,
        }
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.max_epochs is not None:
        cfg["num_epochs"] = args.max_epochs
    if args.patience is not None:
        cfg["patience"] = args.patience
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    elif "seeds" in cfg:
        seeds = [int(s) for s in cfg["seeds"]]
    else:
        seeds = [int(cfg.get("seed", 42))]

    # Reasonable defaults for the new training loop.
    cfg.setdefault("num_epochs", 15)
    cfg.setdefault("patience", 3)
    cfg.setdefault("early_stopping_metric", "kg_stage_macro_score")
    cfg.setdefault("early_stopping_mode", "max")
    cfg.setdefault("gnn_type", "hgt")
    cfg.setdefault("gnn_num_heads", 4)

    output_root = Path(cfg.get("output_dir", ROOT_DIR / "layoutlmv3" / "outputs"))
    output_root.mkdir(parents=True, exist_ok=True)
    # Persist the effective config for reproducibility.
    (output_root / "effective_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    results = []
    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== seed {seed} -> {seed_dir} ===", flush=True)
        result = train_one_seed(cfg, seed, seed_dir, dry_run=args.dry_run)
        results.append(result)
        if args.dry_run:
            break

    summary = aggregate_summary(results)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary_written path={summary_path}", flush=True)

    if summary["aggregate"]:
        print("aggregate_metrics:", flush=True)
        for key, stats in summary["aggregate"].items():
            print(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f} (n={stats['n']})", flush=True)


if __name__ == "__main__":
    main()
