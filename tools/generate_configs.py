"""Generate the full set of ablation configs we need on the target GPU box.

Running this script writes ~30 config JSON files under
``layoutlmv3/configs/generated/`` without touching the hand-written ones.
Each config embeds:

  * seeds = [42, 123, 314]    (paper-grade multi-seed)
  * num_epochs = 15           (with early stopping patience=3)
  * grad_clip = 1.0
  * amp = true                (RTX 5000 Ada has enough VRAM for bf16/fp16)

The groups of configs emitted:

1. **main**: HGT + router x {none, mask, subgraph, mask_subgraph} on train10k.
2. **gnn_backbone**: router=mask_subgraph x gnn in {edge_aware, gatv2, hgt}
   on train10k (isolates the GNN contribution).
3. **gnn_layers**: router=mask_subgraph, gnn=hgt, layers in {0,1,2,3,4}.
4. **knn_k**: router=mask_subgraph, gnn=hgt, graph_knn_k in {2,4,8,16}.
5. **edge_ablation**: drop one edge family at a time
   (no_parent, no_same_parent, no_spatial, no_ref).
6. **encoder**: freeze_encoder in {true, false} for hgt+mask_subgraph.
7. **scale**: audit200, audit500, train10k (same hparams) for the scaling
   curve.

Every config points at the local Windows paths that already exist in the
workspace. Adjust DATASET_ROOT / MODEL_PATH if deploying to Linux.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "layoutlmv3" / "configs" / "generated"
OUTPUT_ROOT_REL = "layoutlmv3/outputs"

DATASETS = {
    "audit200": "synthetic_contract_ds/contract_synth_safe_v2_audit200",
    "audit500": "synthetic_contract_ds/contract_synth_safe_v2_audit500",
    "train10k": "synthetic_contract_ds/contract_synth_safe_v2_train10k",
}
MODEL_PATH = "layoutlmv3/hf_models/layoutlmv3-base"


def base_config(dataset_key: str, tag: str) -> dict:
    """Build a fully-resolved config pointing at repo-relative paths."""
    return {
        "dataset_root": str(REPO_ROOT / DATASETS[dataset_key]).replace("\\", "/"),
        "model_name_or_path": str(REPO_ROOT / MODEL_PATH).replace("\\", "/"),
        "output_dir": str(REPO_ROOT / OUTPUT_ROOT_REL / tag).replace("\\", "/"),
        "router_mode": "mask_subgraph",
        "gnn_type": "hgt",
        "gnn_num_heads": 4,
        "batch_size": 2,
        "num_epochs": 15,
        "patience": 3,
        "early_stopping_metric": "kg_stage_macro_score",
        "early_stopping_mode": "max",
        "val_ratio": 0.1,
        "seed": 42,
        "seeds": [42, 123, 314],
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "amp": True,
        "max_length": 512,
        "num_workers": 2,
        "graph_knn_k": 4,
        "use_parent_edges": True,
        "use_same_parent_edges": True,
        "use_ref_edges": True,
        "gnn_layers": 2,
        "dropout": 0.1,
        "freeze_encoder": False,
        "task_loss_weights": {
            "entity_consolidation": 1.0,
            "semantic_linking": 1.0,
            "attribute_canonicalization": 1.0,
        },
    }


def dump(cfg: dict, name: str) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def emit_main() -> list[Path]:
    """Router ablation on train10k, HGT backbone, 3 seeds."""
    paths = []
    for router in ("none", "mask", "subgraph", "mask_subgraph"):
        cfg = base_config("train10k", f"train10k_hgt_{router}")
        cfg["router_mode"] = router
        paths.append(dump(cfg, f"train10k_hgt_{router}"))
    return paths


def emit_gnn_backbone() -> list[Path]:
    paths = []
    for gnn in ("edge_aware", "gatv2", "hgt"):
        cfg = base_config("train10k", f"train10k_gnn_{gnn}_mask_subgraph")
        cfg["gnn_type"] = gnn
        paths.append(dump(cfg, f"train10k_gnn_{gnn}"))
    return paths


def emit_gnn_layers() -> list[Path]:
    paths = []
    for n_layers in (0, 1, 2, 3, 4):
        cfg = base_config("train10k", f"train10k_hgt_layers{n_layers}")
        cfg["gnn_layers"] = n_layers
        paths.append(dump(cfg, f"train10k_hgt_layers{n_layers}"))
    return paths


def emit_knn_k() -> list[Path]:
    paths = []
    for k in (2, 4, 8, 16):
        cfg = base_config("train10k", f"train10k_hgt_knn{k}")
        cfg["graph_knn_k"] = k
        paths.append(dump(cfg, f"train10k_hgt_knn{k}"))
    return paths


def emit_edge_ablation() -> list[Path]:
    paths = []
    variants = {
        "no_parent":       {"use_parent_edges": False},
        "no_same_parent":  {"use_same_parent_edges": False},
        "no_spatial":      {"graph_knn_k": 0},
        "no_ref":          {"use_ref_edges": False},
    }
    for name, overrides in variants.items():
        cfg = base_config("train10k", f"train10k_hgt_{name}")
        cfg.update(overrides)
        paths.append(dump(cfg, f"train10k_hgt_{name}"))
    return paths


def emit_encoder() -> list[Path]:
    paths = []
    for freeze in (True, False):
        tag = "frozen" if freeze else "finetune"
        cfg = base_config("train10k", f"train10k_hgt_encoder_{tag}")
        cfg["freeze_encoder"] = freeze
        paths.append(dump(cfg, f"train10k_hgt_encoder_{tag}"))
    return paths


def emit_scale() -> list[Path]:
    paths = []
    for key in ("audit200", "audit500", "train10k"):
        cfg = base_config(key, f"{key}_hgt_mask_subgraph")
        paths.append(dump(cfg, f"{key}_hgt_mask_subgraph"))
    return paths


def main() -> None:
    groups = {
        "main (router ablation)": emit_main(),
        "gnn_backbone": emit_gnn_backbone(),
        "gnn_layers": emit_gnn_layers(),
        "knn_k": emit_knn_k(),
        "edge_ablation": emit_edge_ablation(),
        "encoder": emit_encoder(),
        "scale": emit_scale(),
    }
    total = 0
    for group, paths in groups.items():
        print(f"[{group}] -> {len(paths)} configs")
        for p in paths:
            print(f"  {p.relative_to(REPO_ROOT)}")
        total += len(paths)
    print(f"Total configs: {total}")
    print(f"Config dir: {CONFIG_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
