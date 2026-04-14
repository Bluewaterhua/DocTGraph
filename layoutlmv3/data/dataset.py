from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from PIL import Image
from torch.utils.data import Dataset

from .graph_builder import build_graph


class HtmlGraphDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        graph_knn_k: int = 4,
        use_parent_edges: bool = True,
        use_same_parent_edges: bool = True,
        use_ref_edges: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.doc_dirs = sorted(self.dataset_root.glob("doc_*"))
        self.doc_metas = []
        self.graph_knn_k = graph_knn_k
        self.use_parent_edges = use_parent_edges
        self.use_same_parent_edges = use_same_parent_edges
        self.use_ref_edges = use_ref_edges
        if not self.doc_dirs:
            raise FileNotFoundError(f"No doc_* folders found in {self.dataset_root}")
        for doc_dir in self.doc_dirs:
            meta_path = doc_dir / "meta.json"
            if meta_path.exists():
                self.doc_metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
            else:
                self.doc_metas.append({})

    def __len__(self) -> int:
        return len(self.doc_dirs)

    def __getitem__(self, index: int) -> Dict:
        doc_dir = self.doc_dirs[index]
        nodes_json = json.loads((doc_dir / "nodes.json").read_text(encoding="utf-8"))
        labels = json.loads((doc_dir / "labels.json").read_text(encoding="utf-8"))
        meta = json.loads((doc_dir / "meta.json").read_text(encoding="utf-8"))
        image = Image.open(doc_dir / "image_p1.jpg").convert("RGB")

        nodes: List[Dict] = nodes_json["nodes"]
        graph = build_graph(
            nodes,
            labels,
            knn_k=self.graph_knn_k,
            use_parent_edges=self.use_parent_edges,
            use_same_parent_edges=self.use_same_parent_edges,
            use_ref_edges=self.use_ref_edges,
        )

        return {
            "doc_id": meta["doc_id"],
            "image": image,
            "node_texts": [node["text"] if node["text"] else "[EMPTY]" for node in nodes],
            "node_boxes": [node["bbox"] for node in nodes],
            "nodes": nodes,
            "graph": graph,
            "labels": labels,
            "meta": meta,
        }
