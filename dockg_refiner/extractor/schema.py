"""Schema definitions and JSON-Schema generators for the LLM extractor."""
from __future__ import annotations

from typing import Any, Dict, List


# Target fields that the LLM must extract per document.
SCHEMA_FIELDS: List[str] = [
    "contract_id",
    "sign_date",
    "start_date",
    "end_date",
    "total_money",
    "phone",
    "email",
    "tax_no",
    "bank_account",
    "partyA",
    "partyB",
]


# Relation label set matches the synthetic v3 label space.
RELATION_LABELS: List[str] = ["refer_to", "caption_of", "contains"]


# Node kinds emitted by the renderer -> valid node types in our HGT graph.
NODE_KINDS: List[str] = ["block", "object", "mention", "value", "ref"]


def build_extraction_prompt(schema_fields: List[str] | None = None) -> str:
    """System prompt for the VLM/LLM coarse extractor.

    Returns a JSON-structured-output instruction asking the model to
    fill in the schema + list every entity mention / value with its
    page index and bbox (normalized 0-1000).

    2026-05-07: prompt upgraded to require ``objects[]`` (containers
    that anchor refer/contains/caption relations) and richer ``refs[]``
    with both ``guess_target_page`` and ``guess_target_anchor``. Without
    objects the refiner has no relation candidate edges to score.
    """
    fields = schema_fields or SCHEMA_FIELDS
    field_list = "\n".join(f"  - {field}" for field in fields)
    return (
        "你是合同文档抽取器。请从多页合同图像中抽取以下字段并返回严格 JSON：\n"
        f"字段：\n{field_list}\n\n"
        "同时请列出每个关键提及（公司名 / 金额 / 日期 / 电话 / 邮箱 / 税号 /\n"
        "银行账号）的 (page_idx, bbox[x0,y0,x1,y1]) （bbox 归一化到 0-1000）。\n"
        "并请列出文中出现的可被引用的『对象/容器』(objects)，例如：\n"
        "  - 表格 (table)、图 (figure)、附录 (appendix)、条款块 (clause)、章节 (section)。\n"
        "  对每个 object 给出 id、kind、page_idx、bbox、anchor (例如 '附录A' /\n"
        "  '表3' / '第5条'，若无显式锚点则为 null)、title (object 的可读标题，\n"
        "  例如 '项目验收标准' / '付款一览表'，若无可写 null)，并尽量列出\n"
        "  contained_mention_ids / contained_value_ids（即 mentions[].id 与\n"
        "  values[].id 中落在该 object bbox 内或语义上隶属该 object 的元素）。\n"
        "最后请列出文中出现的跨段/跨页引用（trigger 文本 + 源段落位置 + 目标页码 +\n"
        "目标锚点）。\n"
        "输出 JSON 结构：\n"
        "{\n"
        '  "fields": { <schema_field>: { "raw": str, "norm": str, '
        '"page_idx": int, "bbox": [x0,y0,x1,y1] } },\n'
        '  "mentions": [ { "id": str, "text": str, "entity_hint": str|null, '
        '"page_idx": int, "bbox": [...] } ],\n'
        '  "values":   [ { "id": str, "text": str, "type": str, "norm": str, '
        '"page_idx": int, "bbox": [...] } ],\n'
        '  "objects":  [ { "id": str, "kind": str, "anchor": str|null, '
        '"title": str|null, "page_idx": int, "bbox": [...]|null, '
        '"contained_mention_ids": [str], "contained_value_ids": [str] } ],\n'
        '  "refs":     [ { "trigger": str, "source_page": int, '
        '"source_bbox": [...], '
        '"guess_target_page": int|null, '
        '"guess_target_anchor": str|null } ]\n'
        "}\n"
        "硬性要求：\n"
        "  - mentions/values/objects 必须含 id 字段，refs.guess_target_anchor\n"
        "    若指向某 object 应等于该 object.anchor 或 object.id；\n"
        "  - bbox 用真实坐标（0-1000 归一化），无法识别时必须写 null，\n"
        "    严禁伪造占位坐标如 [0,0,10,10]；整个字段无法识别可整体省略；\n"
        "  - 严格 JSON，不要输出多余文本。"
    )


def build_extraction_prompt_anaphora(schema_fields: List[str] | None = None) -> str:
    """Anaphora-aware variant of the coarse extractor prompt.

    Codex 2026-05-09 (anaphora 1-doc A/B): the default prompt only asks
    for canonical company mentions. To give the coref refiner something
    to merge, this variant explicitly demands every occurrence of role
    anaphora (甲方/乙方/双方/我司/...) and short-form aliases (e.g. "泽瀚"
    referring back to 青泽泽瀚系统集成有限公司), each as its own mention,
    with a role-typed entity_hint.

    Strictly additive: schema fields, values, objects, refs sections are
    unchanged. Only the mention-coverage paragraph is extended.

    NOTE: this is gated to a separate cache directory
    (vlm_cache_anaphora_1doc/...) so the legacy default cache stays
    untouched until the A/B passes the codex acceptance gates.
    """
    fields = schema_fields or SCHEMA_FIELDS
    field_list = "\n".join(f"  - {field}" for field in fields)
    return (
        "你是合同文档抽取器。请从多页合同图像中抽取以下字段并返回严格 JSON：\n"
        f"字段：\n{field_list}\n\n"
        "mentions 必须涵盖以下三类，文中每出现一次记录一条（重复出现也要重复列出）：\n\n"
        "  (a) 实体提及：\n"
        "      - 公司全称（如『雾川骐烁软件科技有限责任公司』）\n"
        "      - 公司简称（如『泽瀚』、『晨骐』等明确指代某个公司的缩写）\n"
        "      - 人名 / 机构名\n\n"
        "  (b) 角色指代（必须全部抓取，不可跳过、不可合并）：\n"
        "      甲方 / 乙方 / 双方 / 对方 / 本公司 / 我司 / 贵司 /\n"
        "      该公司 / 上述公司 / 第三方 / 第三方机构\n\n"
        "  (c) 数值类：金额 / 日期 / 电话 / 邮箱 / 税号 / 银行账号\n\n"
        "对 (a)(b) 类 mention 必须填写 entity_hint：\n"
        "  - \"partyA\"  指代甲方对应的实体\n"
        "  - \"partyB\"  指代乙方对应的实体\n"
        "  - \"partyC\"  指代第三方对应的实体\n"
        "  - \"company\" 公司名但角色不明\n"
        "  - null       其它（如人名、机构名）\n"
        "对 (c) 类 mention，entity_hint 写 null。\n\n"
        "并请列出文中出现的可被引用的『对象/容器』(objects)，例如：\n"
        "  - 表格 (table)、图 (figure)、附录 (appendix)、条款块 (clause)、章节 (section)。\n"
        "  对每个 object 给出 id、kind、page_idx、bbox、anchor (例如 '附录A' /\n"
        "  '表3' / '第5条'，若无显式锚点则为 null)、title (object 的可读标题，\n"
        "  例如 '项目验收标准' / '付款一览表'，若无可写 null)，并尽量列出\n"
        "  contained_mention_ids / contained_value_ids（即 mentions[].id 与\n"
        "  values[].id 中落在该 object bbox 内或语义上隶属该 object 的元素）。\n"
        "最后请列出文中出现的跨段/跨页引用（trigger 文本 + 源段落位置 + 目标页码 +\n"
        "目标锚点）。\n"
        "输出 JSON 结构：\n"
        "{\n"
        '  "fields": { <schema_field>: { "raw": str, "norm": str, '
        '"page_idx": int, "bbox": [x0,y0,x1,y1] } },\n'
        '  "mentions": [ { "id": str, "text": str, "entity_hint": str|null, '
        '"page_idx": int, "bbox": [...] } ],\n'
        '  "values":   [ { "id": str, "text": str, "type": str, "norm": str, '
        '"page_idx": int, "bbox": [...] } ],\n'
        '  "objects":  [ { "id": str, "kind": str, "anchor": str|null, '
        '"title": str|null, "page_idx": int, "bbox": [...]|null, '
        '"contained_mention_ids": [str], "contained_value_ids": [str] } ],\n'
        '  "refs":     [ { "trigger": str, "source_page": int, '
        '"source_bbox": [...], '
        '"guess_target_page": int|null, '
        '"guess_target_anchor": str|null } ]\n'
        "}\n"
        "硬性要求：\n"
        "  - mentions/values/objects 必须含 id 字段，refs.guess_target_anchor\n"
        "    若指向某 object 应等于该 object.anchor 或 object.id；\n"
        "  - 角色指代 mention（甲方/乙方/...）即使 bbox 难以精确定位，\n"
        "    也必须列出，bbox 可写 null；\n"
        "  - bbox 用真实坐标（0-1000 归一化），无法识别时必须写 null，\n"
        "    严禁伪造占位坐标如 [0,0,10,10]；\n"
        "  - 严格 JSON，不要输出多余文本。"
    )


def schema_embedding_ids(schema_fields: List[str] | None = None) -> Dict[str, int]:
    fields = schema_fields or SCHEMA_FIELDS
    return {field: idx for idx, field in enumerate(fields)}
