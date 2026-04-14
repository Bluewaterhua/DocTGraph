from __future__ import annotations

import argparse
import json
import random
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


def load_config(config_path: str) -> dict:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    default_config = (
        Path(__file__).resolve().parent / "configs" / "task_driven_layoutlmv3_gnn.json"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config))
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


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


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


def save_checkpoint(model: TaskDrivenLayoutLMv3GNN, output_dir: Path, epoch: int, metrics: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"epoch_{epoch:02d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    print(f"checkpoint_saved path={checkpoint_path}", flush=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

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
        seed=cfg.get("seed", 42),
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
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    output_dir = Path(cfg.get("output_dir", ROOT_DIR / "layoutlmv3" / "outputs"))
    max_train_steps = cfg.get("max_train_steps")

    print(
        " ".join(
            [
                f"device={device}",
                f"train_docs={len(train_indices)}",
                f"val_docs={len(val_indices)}",
                f"split_mode={cfg.get('split_mode', 'random')}",
                f"router_mode={cfg.get('router_mode', 'mask')}",
                f"use_parent_edges={cfg.get('use_parent_edges', True)}",
                f"use_same_parent_edges={cfg.get('use_same_parent_edges', True)}",
                f"use_ref_edges={cfg.get('use_ref_edges', True)}",
            ]
        ),
        flush=True,
    )

    model.train()
    best_val_loss = float("inf")
    for epoch in range(1, cfg["num_epochs"] + 1):
        epoch_start = time.time()
        epoch_losses = []
        total_steps = len(train_loader)
        print(f"epoch_start epoch={epoch}/{cfg['num_epochs']} total_steps={total_steps}", flush=True)
        for step, batch in enumerate(train_loader, 1):
            step_start = time.time()
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(batch)
            outputs["loss"].backward()
            optimizer.step()

            loss_value = outputs["loss"].item()
            epoch_losses.append(loss_value)
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

            if max_train_steps and step >= max_train_steps:
                break

        train_loss = mean(epoch_losses) if epoch_losses else 0.0
        metrics = {"train_loss": train_loss}
        print(
            f"epoch_train_done epoch={epoch} train_loss={train_loss:.4f} elapsed={time.time() - epoch_start:.1f}s",
            flush=True,
        )

        if val_loader is not None:
            val_start = time.time()
            print(f"eval_start epoch={epoch} val_batches={len(val_loader)}", flush=True)
            val_metrics = evaluate(model, val_loader, device)
            metrics.update(val_metrics)
            print(
                " ".join(
                    [
                        f"eval_done epoch={epoch}",
                        f"train_loss={train_loss:.4f}",
                        f"val_loss={val_metrics['val_loss']:.4f}",
                        f"entity_consolidation_f1={val_metrics['entity_consolidation_f1']:.4f}",
                        f"semantic_linking_macro_f1={val_metrics['semantic_linking_macro_f1']:.4f}",
                        f"semantic_linking_micro_f1={val_metrics['semantic_linking_micro_f1']:.4f}",
                        f"attribute_type_macro_f1={val_metrics['attribute_type_macro_f1']:.4f}",
                        f"attribute_value_exact_match={val_metrics['attribute_value_exact_match']:.4f}",
                        f"attribute_value_exact_match_recoverable={val_metrics['attribute_value_exact_match_recoverable']:.4f}",
                        f"attribute_surface_accuracy_unrecoverable={val_metrics['attribute_surface_accuracy_unrecoverable']:.4f}",
                        f"kg_stage_macro_score={val_metrics['kg_stage_macro_score']:.4f}",
                        f"elapsed={time.time() - val_start:.1f}s",
                    ]
                ),
                flush=True,
            )

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                save_checkpoint(model, output_dir, epoch, metrics)
        else:
            save_checkpoint(model, output_dir, epoch, metrics)


if __name__ == "__main__":
    main()
