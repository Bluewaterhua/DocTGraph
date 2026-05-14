from __future__ import annotations

from typing import Dict, List

import torch
from transformers import LayoutLMv3Processor

# 将数据集输入给layoutLMv3进行编码
'''
nodes.json 里的节点
    ↓
node_texts + node_boxes + image
    ↓
LayoutLMv3Processor
    ↓
LayoutLMv3 tokens
    ↓
LayoutLMv3 输出 token embeddings
    ↓
按 word_id_maps 池化回 node embeddings
    ↓
这些 node embeddings 再去和 graph_builder 里的监督目标对齐
'''
class HtmlGraphCollator:
    def __init__(self, model_name_or_path: str, max_length: int = 512) -> None:
        self.processor = LayoutLMv3Processor.from_pretrained(model_name_or_path, apply_ocr=False)
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict:
        images = [item["image"] for item in batch]
        texts = [item["node_texts"] for item in batch]
        boxes = [item["node_boxes"] for item in batch]
        # 传输给layoutLMv3 images texts boxes 进行编码
        encoding = self.processor(
            images=images,
            text=texts,
            boxes=boxes,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )

        word_id_maps = []
        for batch_idx in range(len(batch)):
            word_ids = encoding.word_ids(batch_index=batch_idx)
            word_id_maps.append([-1 if item is None else item for item in word_ids])


        encoding["word_id_maps"] = torch.tensor(word_id_maps, dtype=torch.long)
        encoding["samples"] = batch
        return encoding
