"""End-to-end smoke test.

Goals:
  1. Verify the data loader can read the target dataset root.
  2. Verify LayoutLMv3 weights are on disk and load correctly.
  3. Verify each GNN variant (edge_aware / gatv2 / hgt) runs one full
     forward+backward pass on 2 documents.
  4. Verify the evaluator produces the expected metric keys.
  5. Exit with a non-zero code if anything fails.

Run it with, e.g.:

    python -m layoutlmv3.smoke_test \
        --dataset-root synthetic_contract_ds/contract_synth_safe_v2_audit200 \
        --model-path layoutlmv3/hf_models/layoutlmv3-base

All arguments are optional; reasonable defaults are derived from the
workspace layout. No config file or checkpoint is required.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
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
from layoutlmv3.train import NORM_ID_TO_TYPE, move_batch_to_device


GNN_VARIANTS = ("edge_aware", "gatv2", "hgt")
ROUTER_MODES = ("none", "token", "mask", "subgraph", "mask_subgraph")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=str(
            ROOT_DIR
            / "synthetic_contract_ds"
            / "contract_synth_safe_v2_audit200"
        ),
    )
    parser.add_argument(
        "--model-path",
        default=str(ROOT_DIR / "layoutlmv3" / "hf_models" / "layoutlmv3-base"),
    )
    parser.add_argument("--num-docs", type=int, default=2)
    parser.add_argument("--gnn", choices=("all", *GNN_VARIANTS), default="all")
    parser.add_argument("--router", choices=("all", *ROUTER_MODES), default="mask_subgraph")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def _pick_device(force_cpu: bool) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _load_dataset(dataset_root: str, model_path: str, num_docs: int):
    dataset = HtmlGraphDataset(
        dataset_root=dataset_root,
        graph_knn_k=4,
        use_parent_edges=True,
        use_same_parent_edges=True,
        use_ref_edges=True,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset at {dataset_root} is empty.")
    collator = HtmlGraphCollator(model_name_or_path=model_path, max_length=256)
    docs = [dataset[i] for i in range(min(num_docs, len(dataset)))]
    batch = collator(docs)
    return dataset, batch


def _run_variant(batch, model_path: str, gnn_type: str, router_mode: str, device: torch.device):
    model = TaskDrivenLayoutLMv3GNN(
        model_name_or_path=model_path,
        gnn_layers=2,
        dropout=0.1,
        router_mode=router_mode,
        gnn_type=gnn_type,
        gnn_num_heads=4,
    ).to(device)
    model.train()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    batch_on_device = move_batch_to_device({k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}, device)

    start = time.time()
    out = model(batch_on_device)
    loss = out["loss"]
    loss.backward()
    optim.step()
    optim.zero_grad()
    fwd_bwd = time.time() - start

    # Run one eval pass through the evaluator to confirm metric plumbing.
    model.eval()
    with torch.no_grad():
        eval_batch = move_batch_to_device({k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}, device)
        eval_out = model(eval_batch)
    evaluator = DocumentTaskEvaluator(
        relation_label_count=len(RELATION_TO_ID),
        norm_label_count=len(NORM_TYPE_TO_ID),
        norm_id_to_type=NORM_ID_TO_TYPE,
    )
    for idx, sample in enumerate(eval_batch["samples"]):
        evaluator.update(
            sample=sample,
            entity_out=eval_out["tasks"]["entity_consolidation"][idx],
            relation_out=eval_out["tasks"]["semantic_linking"][idx],
            attr_out=eval_out["tasks"]["attribute_canonicalization"][idx],
        )
    metrics = evaluator.metrics()

    # Free memory between variants.
    del model, optim, out, eval_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return loss.item(), fwd_bwd, metrics


def main() -> int:
    args = parse_args()
    device = _pick_device(args.cpu)
    print(f"device={device}", flush=True)
    print(f"dataset_root={args.dataset_root}", flush=True)
    print(f"model_path={args.model_path}", flush=True)

    if not Path(args.dataset_root).exists():
        print(f"FAIL dataset_root does not exist: {args.dataset_root}", flush=True)
        return 1
    if not Path(args.model_path).exists():
        print(f"FAIL model_path does not exist: {args.model_path}", flush=True)
        return 1

    try:
        dataset, batch = _load_dataset(args.dataset_root, args.model_path, args.num_docs)
    except Exception:
        print("FAIL dataset / collator failed to load:", flush=True)
        traceback.print_exc()
        return 1
    print(f"OK  dataset_len={len(dataset)} batch_docs={len(batch['samples'])}", flush=True)

    variants = GNN_VARIANTS if args.gnn == "all" else (args.gnn,)
    routers = ROUTER_MODES if args.router == "all" else (args.router,)

    failures = []
    for gnn_type in variants:
        for router_mode in routers:
            label = f"gnn={gnn_type} router={router_mode}"
            try:
                loss, dt, metrics = _run_variant(batch, args.model_path, gnn_type, router_mode, device)
                required_keys = (
                    "kg_stage_macro_score",
                    "entity_consolidation_f1",
                    "semantic_linking_macro_f1",
                    "attribute_value_exact_match",
                )
                missing = [k for k in required_keys if k not in metrics]
                if missing:
                    raise RuntimeError(f"missing metric keys: {missing}")
                print(
                    f"OK  {label} loss={loss:.4f} fwd_bwd_s={dt:.2f} "
                    f"kg_stage_macro={metrics['kg_stage_macro_score']:.4f}",
                    flush=True,
                )
            except Exception as exc:
                failures.append((label, str(exc)))
                print(f"FAIL {label}: {exc}", flush=True)
                traceback.print_exc()

    if failures:
        print(f"=== SMOKE TEST FAILED: {len(failures)} variants broken ===", flush=True)
        for label, msg in failures:
            print(f"  - {label}: {msg}", flush=True)
        return 1
    print("=== SMOKE TEST PASSED ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
