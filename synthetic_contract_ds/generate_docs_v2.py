import json
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from jinja2 import Template

from corpus_v2 import sample_boilerplate, sample_clause_paragraphs
from gen_v2 import (
    build_aliases,
    date_norm,
    date_raw_variant,
    gen_address,
    gen_bank_account,
    gen_contact_name,
    gen_email,
    gen_phone,
    gen_tax_no,
    institution_name,
    money_norm,
    money_raw_variant,
    safe_company_name,
)


def weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    items = list(weights.items())
    total = sum(float(prob) for _, prob in items)
    threshold = rng.random() * total
    acc = 0.0
    for key, prob in items:
        acc += float(prob)
        if threshold <= acc:
            return key
    return items[-1][0]


def load_blacklist(cfg: dict, base_dir: Path) -> List[str]:
    ent_cfg = cfg.get("entities", {})
    if not ent_cfg.get("use_blacklist_file", False):
        return []
    path = base_dir / ent_cfg.get("blacklist_file", "blacklist.txt")
    if not path.exists():
        return []
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            terms.append(text)
    return terms


def random_contract_id(rng: random.Random, idx: int) -> str:
    return f"SYN-CT-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}-{idx:06d}"


def pick_alias(rng: random.Random, aliases: Sequence[str], difficulty: str) -> str:
    if difficulty == "easy":
        pool = aliases[:3]
    elif difficulty == "mid":
        pool = aliases[:6]
    else:
        pool = aliases
    return rng.choice(list(pool))


def sanitize_aliases(aliases: Sequence[str], role: str) -> List[str]:
    blocked = {"双方", "贵方"}
    if role == "甲方":
        blocked.update({"乙方", "受托方"})
    else:
        blocked.update({"甲方", "委托方"})
    cleaned = [alias for alias in aliases if alias and alias not in blocked]
    return cleaned or list(aliases)


def make_refer_span(ref_id: str, target_obj: str, text: str) -> str:
    return (
        f'<span data-ref-id="{ref_id}" '
        f'data-rel="refer_to" '
        f'data-target-obj="{target_obj}">{text}</span>'
    )


def build_pay_table(
    rng: random.Random,
    total_amount: int,
    table_idx: int,
    value_registry: Dict[str, Dict[str, str]],
) -> Tuple[List[str], str]:
    num_rows = rng.randint(2, 3)
    ratios = [50, 50] if num_rows == 2 else [30, 40, 30]
    headers = ["期次", "触发条件", "比例", "金额", "计划日期"]
    conditions = ["合同签署", "阶段验收通过", "最终验收通过"]

    base_date = datetime(rng.randint(2019, 2025), rng.randint(1, 12), rng.randint(1, 28))
    rows = []
    for idx in range(num_rows):
        amount = int(round(total_amount * ratios[idx] / 100.0))
        pay_date = base_date + timedelta(days=15 * (idx + 1))
        money_id = f"v_tbl{table_idx}_pay_money_{idx + 1}"
        date_id = f"v_tbl{table_idx}_pay_date_{idx + 1}"
        money_raw = money_raw_variant(rng, amount)
        money_value = money_norm(amount)
        date_raw = date_raw_variant(rng, pay_date)
        date_value = date_norm(pay_date)

        value_registry[money_id] = {
            "value_id": money_id,
            "type": "money",
            "raw": money_raw,
            "norm": money_value,
        }
        value_registry[date_id] = {
            "value_id": date_id,
            "type": "datetime",
            "raw": date_raw,
            "norm": date_value,
        }

        money_cell = (
            f'<span data-value-id="{money_id}" '
            f'data-norm-type="money" '
            f'data-norm-value="{money_value}">{money_raw}</span>'
        )
        date_cell = (
            f'<span data-value-id="{date_id}" '
            f'data-norm-type="datetime" '
            f'data-norm-value="{date_value}">{date_raw}</span>'
        )
        rows.append(
            "<tr>"
            f"<td>{idx + 1}</td>"
            f"<td>{conditions[min(idx, len(conditions) - 1)]}</td>"
            f"<td>{ratios[idx]}%</td>"
            f"<td>{money_cell}</td>"
            f"<td>{date_cell}</td>"
            "</tr>"
        )
    return headers, "\n".join(rows)


def build_deliver_table(rng: random.Random, table_idx: int) -> Tuple[List[str], str]:
    headers = ["序号", "交付物", "形式", "说明"]
    items = [
        ("实施方案", "文档", "V1.0"),
        ("配置清单", "文档", "含关键参数"),
        ("测试报告", "文档", "含截图与结论"),
        ("运维手册", "文档", "含应急流程"),
        ("部署脚本", "脚本", "支持重复执行"),
    ]
    rows = []
    for row_idx, item in enumerate(rng.sample(items, k=rng.randint(3, 4)), 1):
        rows.append(
            "<tr>"
            f"<td>{row_idx}</td>"
            f"<td>{item[0]}</td>"
            f"<td>{item[1]}</td>"
            f"<td>{item[2]}</td>"
            "</tr>"
        )
    return headers, "\n".join(rows)


def build_figure_placeholder(rng: random.Random) -> str:
    return rng.choice(
        [
            "（示意）系统部署结构图",
            "（示意）业务流程关系图",
            "（示意）模块依赖关系图",
        ]
    )


def append_unique(items: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
    if item not in items:
        items.append(item)


def validate_labels(
    expected_ids: Sequence[str],
    labels: Dict[str, Any],
) -> None:
    id_set = set(expected_ids)

    for entity in labels["coref"]["entities"]:
        for mention_id in entity["mentions"]:
            if mention_id not in id_set:
                raise ValueError(f"Missing coref mention id: {mention_id}")

    for rel in labels["relations"]:
        if rel["h"] not in id_set:
            raise ValueError(f"Missing relation head id: {rel}")
        if rel["t"] not in id_set:
            raise ValueError(f"Missing relation tail id: {rel}")
        trigger = rel.get("trigger")
        if trigger and trigger not in id_set:
            raise ValueError(f"Missing relation trigger id: {rel}")

    for item in labels["normalization"]:
        if item["value_id"] not in id_set:
            raise ValueError(f"Missing normalization value id: {item}")


def generate_one(
    doc_idx: int,
    cfg: Dict[str, Any],
    out_dir: Path,
    template: Template,
    blacklist: List[str],
) -> None:
    rng = random.Random(cfg["seed"] * 1000003 + doc_idx)
    dist = cfg["distributions"]
    noise_level = weighted_choice(rng, dist["noise_level"])
    coref_diff = weighted_choice(rng, dist["coref_difficulty"])
    layout_profile = weighted_choice(rng, dist["layout_profile"])

    page_count = 1

    party_a = safe_company_name(rng, blacklist)
    party_b = safe_company_name(rng, blacklist)
    ent_a_id = f"entA_{doc_idx:06d}"
    ent_b_id = f"entB_{doc_idx:06d}"
    aliases_a = sanitize_aliases(build_aliases(rng, party_a, "甲方"), "甲方")
    aliases_b = sanitize_aliases(build_aliases(rng, party_b, "乙方"), "乙方")

    party_a_alias_1 = pick_alias(rng, aliases_a, coref_diff)
    party_b_alias_1 = pick_alias(rng, aliases_b, coref_diff)
    party_a_alias_2 = pick_alias(rng, aliases_a, coref_diff)
    party_b_alias_2 = pick_alias(rng, aliases_b, coref_diff)

    distractor_count = rng.randint(*cfg["entities"]["num_distractors"])
    distractors = []
    distractor_mentions: Dict[str, List[str]] = {}
    for idx in range(distractor_count):
        entity_id = f"entD_{doc_idx:06d}_{idx + 1}"
        canonical = institution_name(rng, blacklist)
        aliases = [canonical, canonical[: max(2, min(6, len(canonical)))], "第三方机构", "服务商"]
        distractors.append({"entity_id": entity_id, "canonical": canonical, "aliases": aliases})
        distractor_mentions[entity_id] = []

    mention_counter = 0

    def make_distractor_clause(text: str) -> str:
        nonlocal mention_counter
        if not distractors or rng.random() > 0.55:
            return text
        distractor = rng.choice(distractors)
        mention_counter += 1
        mention_id = f"m_dis_{doc_idx:06d}_{mention_counter}"
        distractor_mentions[distractor["entity_id"]].append(mention_id)
        alias = rng.choice(distractor["aliases"])
        return (
            text
            + "，相关测试与联调可由"
            + f'<span data-mention-id="{mention_id}" '
            + f'data-entity-id="{distractor["entity_id"]}">{alias}</span>'
            + "参与配合。"
        )

    contract_id_raw = random_contract_id(rng, doc_idx)
    contract_id_norm = contract_id_raw
    project_name = f"合同业务文档构建项目{doc_idx:06d}"

    sign_dt = datetime(
        rng.randint(cfg["values"]["date_start_year_range"][0], cfg["values"]["date_start_year_range"][1]),
        rng.randint(1, 12),
        rng.randint(1, 28),
    )
    duration_days = rng.randint(*cfg["values"]["duration_days_range"])
    start_dt = sign_dt
    end_dt = sign_dt + timedelta(days=duration_days)

    sign_date_raw = date_raw_variant(rng, sign_dt)
    sign_date_norm = date_norm(sign_dt)
    start_date_raw = date_raw_variant(rng, start_dt)
    start_date_norm = date_norm(start_dt)
    end_date_raw = date_raw_variant(rng, end_dt)
    end_date_norm = date_norm(end_dt)

    total_amount = rng.randint(*cfg["values"]["money_cny_range"])
    total_money_raw = money_raw_variant(rng, total_amount)
    total_money_norm = money_norm(total_amount)

    phone_raw, phone_norm = gen_phone(rng)
    tax_raw, tax_norm = gen_tax_no(rng)
    acct_raw, acct_norm = gen_bank_account(rng)
    address = gen_address(rng)
    contact_name = gen_contact_name(rng)
    bank_name = rng.choice(
        [
            "清远商业银行",
            "岚海联合银行",
            "东澜城市银行",
            "衍原发展银行",
            "云港汇通银行",
        ]
    )
    email = gen_email(rng, brand2=party_a[2:4])

    value_registry: Dict[str, Dict[str, str]] = {
        "v_contract_id": {
            "value_id": "v_contract_id",
            "type": "contract_id",
            "raw": contract_id_raw,
            "norm": contract_id_norm,
        },
        "v_sign_date": {
            "value_id": "v_sign_date",
            "type": "datetime",
            "raw": sign_date_raw,
            "norm": sign_date_norm,
        },
        "v_total_money": {
            "value_id": "v_total_money",
            "type": "money",
            "raw": total_money_raw,
            "norm": total_money_norm,
        },
        "v_start_date": {
            "value_id": "v_start_date",
            "type": "datetime",
            "raw": start_date_raw,
            "norm": start_date_norm,
        },
        "v_end_date": {
            "value_id": "v_end_date",
            "type": "datetime",
            "raw": end_date_raw,
            "norm": end_date_norm,
        },
        "v_phone": {
            "value_id": "v_phone",
            "type": "phone",
            "raw": phone_raw,
            "norm": phone_norm,
        },
        "v_tax_no": {
            "value_id": "v_tax_no",
            "type": "tax_no",
            "raw": tax_raw,
            "norm": tax_norm,
        },
        "v_bank_acct": {
            "value_id": "v_bank_acct",
            "type": "bank_account",
            "raw": acct_raw,
            "norm": acct_norm,
        },
        "v_email": {
            "value_id": "v_email",
            "type": "email",
            "raw": email,
            "norm": email,
        },
    }

    pay_table_obj_id = "obj_tbl_pay_1"
    pay_table_caption_id = "cap_tbl_pay_1"
    pay_table_headers, pay_table_rows = build_pay_table(rng, total_amount, 1, value_registry)

    deliver_table_enabled = rng.random() < cfg.get("objects", {}).get("include_deliver_table_prob", 0.35)
    deliver_table_obj_id = "obj_tbl_deliver_1"
    deliver_table_caption_id = "cap_tbl_deliver_1"
    deliver_table_headers, deliver_table_rows = build_deliver_table(rng, 2)

    figure_enabled = rng.random() < cfg.get("objects", {}).get("include_figure_prob", 0.7)
    figure_obj_id = "obj_fig_1"
    figure_caption_id = "cap_fig_1"

    if layout_profile == "compact":
        deliver_table_enabled = False
        figure_enabled = False
    elif layout_profile == "balanced":
        if deliver_table_enabled and figure_enabled:
            if rng.random() < 0.5:
                deliver_table_enabled = False
            else:
                figure_enabled = False
    elif deliver_table_enabled and figure_enabled:
        figure_enabled = False

    clause_pool = sample_clause_paragraphs(rng, k=4, money_raw=total_money_raw)
    boiler_count = 1 if layout_profile == "compact" else 2
    boiler_pool = sample_boilerplate(rng, k=boiler_count)

    relations: List[Dict[str, Any]] = []
    blocks: List[Dict[str, Any]] = []
    expected_ids = {
        "v_contract_id",
        "v_sign_date",
        "m_partyA_1",
        "m_partyA_2",
        "m_partyB_1",
        "m_partyB_2",
    }

    blocks.append({"kind": "section", "element_id": "sec_1", "text": "第一条 合同主体与项目概况"})
    expected_ids.add("sec_1")
    blocks.append(
        {
            "kind": "para",
            "element_id": "p_intro_1",
            "html": (
                "经友好协商，"
                f'<span data-mention-id="m_partyA_2" data-entity-id="{ent_a_id}">{party_a_alias_1}</span>'
                "与"
                f'<span data-mention-id="m_partyB_2" data-entity-id="{ent_b_id}">{party_b_alias_1}</span>'
                f"就“{project_name}”相关服务达成本合同。"
            ),
        }
    )
    expected_ids.add("p_intro_1")

    blocks.append({"kind": "section", "element_id": "sec_2", "text": "第二条 服务范围与交付要求"})
    expected_ids.add("sec_2")
    for idx, text in enumerate(clause_pool[:2], 1):
        element_id = f"p_clause_{idx}"
        blocks.append({"kind": "para", "element_id": element_id, "html": make_distractor_clause(text)})
        expected_ids.add(element_id)

    blocks.append({"kind": "section", "element_id": "sec_3", "text": "第三条 费用、期限与联系信息"})
    expected_ids.add("sec_3")

    ref_pay_id = "r_pay_tbl_1"
    pay_para_html = (
        "本合同总金额为"
        f'<span data-value-id="v_total_money" data-norm-type="money" data-norm-value="{total_money_norm}">{total_money_raw}</span>'
        "，付款计划如"
        + make_refer_span(ref_pay_id, pay_table_obj_id, "表1")
        + "所示。"
    )
    blocks.append({"kind": "para", "element_id": "p_pay_1", "html": pay_para_html})
    expected_ids.update({"p_pay_1", ref_pay_id, "v_total_money", pay_table_obj_id})
    append_unique(
        relations,
        {"h": "p_pay_1", "r": "refer_to", "t": pay_table_obj_id, "trigger": ref_pay_id},
    )

    term_html = (
        "服务期限自"
        f'<span data-value-id="v_start_date" data-norm-type="datetime" data-norm-value="{start_date_norm}">{start_date_raw}</span>'
        "起至"
        f'<span data-value-id="v_end_date" data-norm-type="datetime" data-norm-value="{end_date_norm}">{end_date_raw}</span>'
        "止。"
    )
    blocks.append({"kind": "para", "element_id": "p_term_1", "html": term_html})
    expected_ids.update({"p_term_1", "v_start_date", "v_end_date"})

    contact_html = (
        "项目联系人："
        + contact_name
        + "，联系电话："
        + f'<span data-value-id="v_phone" data-norm-type="phone" data-norm-value="{phone_norm}">{phone_raw}</span>'
        + "，电子邮箱："
        + f'<span data-value-id="v_email" data-norm-type="email" data-norm-value="{email}">{email}</span>'
        + "。"
    )
    blocks.append({"kind": "para", "element_id": "p_contact_1", "html": contact_html})
    expected_ids.update({"p_contact_1", "v_phone", "v_email"})

    finance_html = (
        "纳税识别号："
        + f'<span data-value-id="v_tax_no" data-norm-type="tax_no" data-norm-value="{tax_norm}">{tax_raw}</span>'
        + "，开户行："
        + bank_name
        + "，银行账号："
        + f'<span data-value-id="v_bank_acct" data-norm-type="bank_account" data-norm-value="{acct_norm}">{acct_raw}</span>'
        + f"。联系地址：{address}。"
    )
    blocks.append({"kind": "para", "element_id": "p_finance_1", "html": finance_html})
    expected_ids.update({"p_finance_1", "v_tax_no", "v_bank_acct"})

    blocks.append(
        {
            "kind": "table",
            "wrap_id": "tbl_wrap_pay_1",
            "element_id": "tbl_pay_1",
            "object_id": pay_table_obj_id,
            "headers": pay_table_headers,
            "rows_html": pay_table_rows,
            "caption_id": pay_table_caption_id,
            "caption_text": "表1 付款计划",
        }
    )
    expected_ids.update({"tbl_wrap_pay_1", "tbl_pay_1", pay_table_obj_id, pay_table_caption_id})
    append_unique(relations, {"h": pay_table_caption_id, "r": "caption_of", "t": pay_table_obj_id})

    if deliver_table_enabled and layout_profile != "compact":
        ref_deliver_id = "r_deliver_tbl_1"
        blocks.append(
            {
                "kind": "para",
                "element_id": "p_deliver_ref_1",
                "html": "主要交付物及文档边界见" + make_refer_span(ref_deliver_id, deliver_table_obj_id, "表2") + "。",
            }
        )
        blocks.append(
            {
                "kind": "table",
                "wrap_id": "tbl_wrap_deliver_1",
                "element_id": "tbl_deliver_1",
                "object_id": deliver_table_obj_id,
                "headers": deliver_table_headers,
                "rows_html": deliver_table_rows,
                "caption_id": deliver_table_caption_id,
                "caption_text": "表2 交付物清单",
            }
        )
        expected_ids.update(
            {
                "p_deliver_ref_1",
                ref_deliver_id,
                "tbl_wrap_deliver_1",
                "tbl_deliver_1",
                deliver_table_obj_id,
                deliver_table_caption_id,
            }
        )
        append_unique(
            relations,
            {
                "h": "p_deliver_ref_1",
                "r": "refer_to",
                "t": deliver_table_obj_id,
                "trigger": ref_deliver_id,
            },
        )
        append_unique(relations, {"h": deliver_table_caption_id, "r": "caption_of", "t": deliver_table_obj_id})

    if figure_enabled:
        ref_figure_id = "r_figure_1"
        blocks.append(
            {
                "kind": "para",
                "element_id": "p_fig_ref_1",
                "html": "系统部署结构可参见" + make_refer_span(ref_figure_id, figure_obj_id, "图1") + "。",
            }
        )
        blocks.append(
            {
                "kind": "figure",
                "wrap_id": "fig_wrap_1",
                "element_id": "fig_1",
                "object_id": figure_obj_id,
                "fig_text": build_figure_placeholder(rng),
                "caption_id": figure_caption_id,
                "caption_text": "图1 系统部署结构示意图",
            }
        )
        expected_ids.update({"p_fig_ref_1", ref_figure_id, "fig_wrap_1", "fig_1", figure_obj_id, figure_caption_id})
        append_unique(
            relations,
            {"h": "p_fig_ref_1", "r": "refer_to", "t": figure_obj_id, "trigger": ref_figure_id},
        )
        append_unique(relations, {"h": figure_caption_id, "r": "caption_of", "t": figure_obj_id})

    for idx, boiler in enumerate(boiler_pool[:2], 1):
        element_id = f"p_noise_{idx}"
        blocks.append({"kind": "para", "element_id": element_id, "html": boiler})
        expected_ids.add(element_id)

    pages = [
        {
            "page_idx": 1,
            "page_type": "cover",
            "blocks": blocks,
            "include_signature": False,
        }
    ]

    coref_entities = [
        {"entity_id": ent_a_id, "canonical": party_a, "mentions": ["m_partyA_1", "m_partyA_2"]},
        {"entity_id": ent_b_id, "canonical": party_b, "mentions": ["m_partyB_1", "m_partyB_2"]},
    ]
    for distractor in distractors:
        if distractor_mentions[distractor["entity_id"]]:
            coref_entities.append(
                {
                    "entity_id": distractor["entity_id"],
                    "canonical": distractor["canonical"],
                    "mentions": distractor_mentions[distractor["entity_id"]],
                }
            )
            expected_ids.update(distractor_mentions[distractor["entity_id"]])

    normalization = list(value_registry.values())
    expected_ids.update(value_registry.keys())

    html = template.render(
        version="2.1",
        doc_no=f"DOC-{doc_idx:06d}",
        contract_id_raw=contract_id_raw,
        contract_id_norm=contract_id_norm,
        project_name=project_name,
        entA_id=ent_a_id,
        entB_id=ent_b_id,
        partyA_canonical=party_a,
        partyB_canonical=party_b,
        partyA_alias_1=party_a_alias_1,
        partyB_alias_1=party_b_alias_1,
        partyA_alias_2=party_a_alias_2,
        partyB_alias_2=party_b_alias_2,
        sign_date_raw=sign_date_raw,
        sign_date_norm=sign_date_norm,
        page_total=1,
        pages=pages,
    )

    labels = {
        "coref": {"entities": coref_entities, "difficulty": coref_diff},
        "relations": relations,
        "normalization": normalization,
        "ref_difficulty": "intra_page",
    }
    validate_labels(sorted(expected_ids), labels)

    meta = {
        "doc_id": f"doc_{doc_idx:06d}",
        "seed": cfg["seed"] * 1000003 + doc_idx,
        "page_count": page_count,
        "noise_level": noise_level,
        "coref_difficulty": coref_diff,
        "ref_difficulty": "intra_page",
        "layout_profile": layout_profile,
        "fictional_entities": True,
    }

    doc_dir = out_dir / f"doc_{doc_idx:06d}"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "doc.html").write_text(html, encoding="utf-8")
    (doc_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    (doc_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v2_audit200.json")
    args = parser.parse_args()

    base = Path(".")
    cfg_path = base / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    template = Template((base / "template_contract_v2.html").read_text(encoding="utf-8"))
    blacklist = load_blacklist(cfg, base)

    out_dir = base / cfg["dataset_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for doc_idx in range(1, cfg["num_docs"] + 1):
        generate_one(doc_idx, cfg, out_dir, template, blacklist)

    print(f"[OK] Generated dataset at: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
