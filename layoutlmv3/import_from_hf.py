import os
# 【关键】必须放在最前面，先设置镜像，再导入 transformers
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, AutoModel

MODEL_ID = "microsoft/layoutlmv3-base"
LOCAL_DIR = "./hf_models/layoutlmv3-base"

print(f"正在从镜像站 {os.environ['HF_ENDPOINT']} 下载...")

processor = LayoutLMv3Processor.from_pretrained(MODEL_ID)
processor.save_pretrained(LOCAL_DIR)

base_model = AutoModel.from_pretrained(MODEL_ID)
base_model.save_pretrained(LOCAL_DIR)

print("下载完成！")