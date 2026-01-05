import json
import random
import string
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from jinja2 import Template


# -----------------------------
# Utilities
# -----------------------------
def weighted_choice(rng: random.Random, w: dict):
    items = list(w.items())
    total = sum(float(p) for _, p in items)
    r = rng.random() * total
    acc = 0.0
    for k, p in items:
        acc += float(p)
        if r <= acc:
            return k
    return items[-1][0]


def money_norm_cny(amount_int: int) -> str:
    return f"CNY:{amount_int:.2f}"


def date_norm(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def sample_date(rng: random.Random, year_lo: int, year_hi: int) -> datetime:
    y = rng.randint(year_lo, year_hi)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return datetime(y, m, d)


def load_blacklist(cfg: dict, base_dir: Path) -> List[str]:
    ent_cfg = cfg.get("entities", {})
    if not ent_cfg.get("use_blacklist_file", False):
        return []
    f = base_dir / ent_cfg.get("blacklist_file", "blacklist.txt")
    if not f.exists():
        return []
    terms = []
    for line in f.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        terms.append(t)
    return terms


# -----------------------------
# Fictional entity generator
# -----------------------------
FICTIONAL_PLACES = [
    "岚州", "北岭", "东澜", "苏川", "宁岫", "衡原", "云港", "青汀", "皓城", "沧岑",
    "霁川", "澈州", "沐原", "泓城", "岑海"
]
BRAND_SYLLABLES = [
    "星", "澜", "昱", "云", "衡", "曜", "辰", "瀚", "霁", "岚", "骁", "晟", "拓", "澈", "泓", "沐", "珩", "昊", "汀",
    "原"
]
INDUSTRIES = [
    "信息技术", "智能系统", "数据服务", "系统集成", "工程咨询", "软件科技", "通信技术", "网络安全",
    "自动化", "数字科技", "工业服务"
]
ORG_FORMS = ["有限公司", "股份有限公司", "有限责任公司"]


def _rand_brand(rng: random.Random) -> str:
    a = rng.choice(BRAND_SYLLABLES)
    b = rng.choice([x for x in BRAND_SYLLABLES if x != a])
    return a + b


def safe_fictional_company_name(rng: random.Random, blacklist: List[str]) -> str:
    # 生成极低概率撞真实的公司名，不使用任何真实公司池
    for _ in range(400):
        place = rng.choice(FICTIONAL_PLACES)
        brand = _rand_brand(rng)
        ind = rng.choice(INDUSTRIES)
        form = rng.choice(ORG_FORMS)
        name = f"{place}{brand}{ind}{form}"
        if any(bad in name for bad in blacklist):
            continue
        if len(name) < 10:
            continue
        return name
    # 兜底：加随机字母扰动
    suffix = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
    return f"{rng.choice(FICTIONAL_PLACES)}{_rand_brand(rng)}{rng.choice(INDUSTRIES)}{suffix}有限公司"


def build_aliases(canonical: str, role_term: str, rng: random.Random) -> List[str]:
    # 生成别名（全虚构），用于共指链
    short = canonical
    for s in ["股份有限公司", "有限公司", "有限责任公司"]:
        short = short.replace(s, "")
    # 取“品牌双字”（大致：地名2字后面2字）
    brand = short[2:4] if len(short) >= 4 else short
    # 伪英文/伪缩写（避免真实品牌）
    en = f"{brand}Tech"
    abbr = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
    # 角色/代称
    role = role_term
    pronouns = ["本公司", "贵方", "双方"]  # 可控难度时再用
    return [canonical, short, brand, en, abbr, role] + pronouns


# -----------------------------
# Text & table blocks
# -----------------------------
CLAUSE_POOL = [
    "乙方为甲方提供系统集成、软件开发与现场技术支持等技术服务，具体以双方确认的工作说明为准。",
    "服务成果交付后，甲方应在约定期限内组织验收；逾期未提出异议的，视为验收通过。",
    "双方对在履行本合同过程中获知的对方商业秘密负有保密义务，未经许可不得向第三方披露。",
    "因履行本合同发生争议的，双方应友好协商解决；协商不成的，提交有管辖权的人民法院诉讼解决。",
    "任何一方因不可抗力导致不能或暂时不能履行本合同义务的，应及时通知对方并在合理期限内提供证明。"
]
NOISE_POOL = [
    "注：本页所示格式仅用于文本规范展示，相关条款解释以双方签署版本为准。",
    "说明：合同文本中引用的附件、清单或表格仅作为结构示例，实际以签署版本为准。",
    "提示：如涉及税务、登记或备案事项，应按相关规定另行办理。",
    "声明：本合成文档仅用于算法研究与测试，内容不构成任何真实合同或承诺。"
]


def build_noise_para(rng: random.Random) -> str:
    return rng.choice(NOISE_POOL)


def build_clause_paras(rng: random.Random, k_min=2, k_max=4) -> List[str]:
    k = rng.randint(k_min, k_max)
    return rng.sample(CLAUSE_POOL, k=k)


def build_pay_table(rng: random.Random, total_amount: int):
    # 付款计划表：2-3 期；金额/date 都带 value-id + norm
    n = rng.randint(2, 3)
    ratios = [50, 50] if n == 2 else [30, 40, 30]
    headers = ["期次", "触发条件", "比例", "金额", "计划日期"]

    rows = []
    base_date = sample_date(rng, 2018, 2025)
    for i in range(n):
        amt = int(round(total_amount * ratios[i] / 100.0))
        dt = base_date + timedelta(days=15 * (i + 1))

        v_mid = f"v_pay_money_{i + 1}"
        v_did = f"v_pay_date_{i + 1}"

        money_cell = (
            f'<span data-value-id="{v_mid}" data-norm-type="money" data-norm-value="{money_norm_cny(amt)}">'
            f'{amt}元</span>'
        )
        date_cell = (
            f'<span data-value-id="{v_did}" data-norm-type="datetime" data-norm-value="{date_norm(dt)}">'
            f'{date_norm(dt)}</span>'
        )

        cond = ["合同签署", "阶段验收通过", "最终验收通过"][min(i, 2)]
        rows.append(
            f"<tr><td>{i + 1}</td><td>{cond}</td><td>{ratios[i]}%</td><td>{money_cell}</td><td>{date_cell}</td></tr>")

    return headers, "\n".join(rows)


def make_ref_sentence(target_obj_id: str, ref_id: str, table_no: int) -> str:
    return (
        f'付款计划如表<span data-ref-id="{ref_id}" data-rel="refer_to" data-target-obj="{target_obj_id}">{table_no}</span>所示。'
    )


# -----------------------------
# Document generation
# -----------------------------
def random_contract_id(rng: random.Random, idx: int) -> str:
    # 合成编号，不使用真实格式
    return f"SYN-TS-{rng.randint(1000, 9999)}-{idx:06d}"


def pick_alias(rng: random.Random, aliases: List[str], diff: str) -> str:
    if diff == "easy":
        candidates = aliases[0:3]  # canonical/short/brand
    elif diff == "mid":
        candidates = aliases[0:6]  # + en/abbr/role
    else:
        candidates = aliases  # + pronouns
    return rng.choice(candidates)


def generate_one(doc_idx: int, cfg: Dict[str, Any], out_dir: Path, template: Template, blacklist: List[str]):
    rng = random.Random(cfg["seed"] * 1000003 + doc_idx)

    dist = cfg["distributions"]
    page_count = int(weighted_choice(rng, dist["page_count"]))
    noise_level = weighted_choice(rng, dist["noise_level"])
    coref_diff = weighted_choice(rng, dist["coref_difficulty"])
    ref_diff = weighted_choice(rng, dist["ref_difficulty"])

    # Fictional entities
    partyA = safe_fictional_company_name(rng, blacklist)
    partyB = safe_fictional_company_name(rng, blacklist)
    entA_id = f"entA_{doc_idx:06d}"
    entB_id = f"entB_{doc_idx:06d}"
    aliasesA = build_aliases(partyA, "甲方", rng)
    aliasesB = build_aliases(partyB, "乙方", rng)

    partyA_alias_1 = pick_alias(rng, aliasesA, coref_diff)
    partyB_alias_1 = pick_alias(rng, aliasesB, coref_diff)
    partyA_alias_2 = pick_alias(rng, aliasesA, coref_diff)
    partyB_alias_2 = pick_alias(rng, aliasesB, coref_diff)

    # Values
    contract_id_raw = random_contract_id(rng, doc_idx)
    contract_id_norm = contract_id_raw

    project_name = f"合成技术服务项目{doc_idx}"

    sign_dt = sample_date(rng, cfg["values"]["date_start_year_range"][0], cfg["values"]["date_start_year_range"][1])
    dur = rng.randint(cfg["values"]["duration_days_range"][0], cfg["values"]["duration_days_range"][1])
    start_dt = sign_dt
    end_dt = sign_dt + timedelta(days=dur)

    sign_date_raw = date_norm(sign_dt)
    sign_date_norm = date_norm(sign_dt)
    start_date_raw = date_norm(start_dt)
    start_date_norm = date_norm(start_dt)
    end_date_raw = date_norm(end_dt)
    end_date_norm = date_norm(end_dt)

    total_amount = rng.randint(cfg["values"]["money_cny_range"][0], cfg["values"]["money_cny_range"][1])
    total_money_norm = money_norm_cny(total_amount)
    total_money_raw = f"{total_amount}元"

    # Objects & relations anchors
    pay_obj_id = "obj_tbl_pay_1"
    pay_tbl_element_id = "tbl_pay_1"
    pay_tbl_wrap_id = "tbl_pay_wrap_1"
    pay_tbl_caption_id = "cap_tbl_pay_1"
    ref_id = "r_pay_tbl_1"

    headers, pay_rows_html = build_pay_table(rng, total_amount)

    # Paragraph containing refer_to trigger
    p_pay_element_id = "p_pay_1_1"
    p_pay_html = (
            f'本合同总金额为 '
            f'<span data-value-id="v_total_money" data-norm-type="money" data-norm-value="{total_money_norm}">{total_money_raw}</span>。'
            + make_ref_sentence(pay_obj_id, ref_id, 1)
    )

    # Term paragraph
    p_term_element_id = "p_term_1_1"
    p_term_html = (
        '本合同有效期自 '
        f'<span data-value-id="v_start_date" data-norm-type="datetime" data-norm-value="{start_date_norm}">{start_date_raw}</span>'
        ' 起至 '
        f'<span data-value-id="v_end_date" data-norm-type="datetime" data-norm-value="{end_date_norm}">{end_date_raw}</span>'
        ' 止。'
    )

    clause_paras = build_clause_paras(rng)

    noise_n = {"low": 1, "mid": 2, "high": 4}[noise_level]

    # Page plan
    pages = []
    para_counter = 0
    caption_id_used = None
    for pi in range(1, page_count + 1):
        page_type = "cover" if pi == 1 else "clause"
        blocks = []

        blocks.append(
            {"kind": "section", "element_id": f"sec_{pi}_1", "text": "第一条 服务内容" if pi == 1 else "补充条款"})
        para_counter += 1
        blocks.append({"kind": "para", "element_id": f"p_clause_{pi}_{para_counter}",
                       "html": clause_paras[min(pi - 1, len(clause_paras) - 1)]})

        blocks.append({"kind": "section", "element_id": f"sec_{pi}_2", "text": "第二条 费用与支付"})
        blocks.append({"kind": "para", "element_id": p_pay_element_id if pi == 1 else f"{p_pay_element_id}_p{pi}",
                       "html": p_pay_html})

        # Table placement: intra-page or cross-page
        # 决定表格实际出现在哪一页：intra_page -> 1；cross_page -> 2(若有)否则1
        if ref_diff == "cross_page" and page_count >= 2:
            table_page_idx = 2  # 也可以改成 page_count 放最后一页
        else:
            table_page_idx = 1

        if pi == table_page_idx:
            caption_id_used = pay_tbl_caption_id if pi == 1 else f"{pay_tbl_caption_id}_p{pi}"
            if caption_id_used is None:
                caption_id_used = pay_tbl_caption_id  # 理论上不会发生，兜底
            blocks.append({
                "kind": "table",
                "wrap_id": pay_tbl_wrap_id if pi == 1 else f"{pay_tbl_wrap_id}_p{pi}",
                "element_id": pay_tbl_element_id if pi == 1 else f"{pay_tbl_element_id}_p{pi}",
                "object_id": pay_obj_id,
                "headers": headers,
                "rows_html": pay_rows_html,
                "caption_id": caption_id_used,
                "caption_text": "表1 付款计划"
            })

        blocks.append({"kind": "section", "element_id": f"sec_{pi}_3", "text": "第三条 期限与验收"})
        blocks.append({"kind": "para", "element_id": p_term_element_id if pi == 1 else f"{p_term_element_id}_p{pi}",
                       "html": p_term_html})

        # Noise paras
        for j in range(noise_n if pi == 1 else max(1, noise_n - 1)):
            blocks.append({"kind": "para", "element_id": f"p_noise_{pi}_{j + 1}", "html": build_noise_para(rng)})

        pages.append({
            "page_idx": pi,
            "page_type": page_type,
            "blocks": blocks,
            "include_signature": (pi == page_count)
        })




    # Render HTML
    html = template.render(
        version="1.0",
        doc_no=f"DOC-{doc_idx:06d}",
        contract_id_raw=contract_id_raw,
        contract_id_norm=contract_id_norm,
        project_name=project_name,
        entA_id=entA_id,
        entB_id=entB_id,
        partyA_canonical=partyA,
        partyB_canonical=partyB,
        partyA_alias_1=partyA_alias_1,
        partyB_alias_1=partyB_alias_1,
        partyA_alias_2=partyA_alias_2,
        partyB_alias_2=partyB_alias_2,
        sign_date_raw=sign_date_raw,
        sign_date_norm=sign_date_norm,
        page_total=page_count,
        pages=pages
    )

    doc_dir = out_dir / f"doc_{doc_idx:06d}"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "doc.html").write_text(html, encoding="utf-8")



    # labels.json (3 tasks)
    labels = {
        "coref": {
            "entities": [
                {"entity_id": entA_id, "canonical": partyA, "mentions": ["m_partyA_1", "m_partyA_2", "m_partyA_3"]},
                {"entity_id": entB_id, "canonical": partyB, "mentions": ["m_partyB_1", "m_partyB_2", "m_partyB_3"]}
            ],
            "difficulty": coref_diff
        },
        "relations": [
            {"h": p_pay_element_id, "r": "refer_to", "t": pay_obj_id, "trigger": ref_id},
            {"h": caption_id_used, "r": "caption_of", "t": pay_obj_id}
        ],
        "normalization": [
            {"value_id": "v_contract_id", "type": "contract_id", "raw": contract_id_raw, "norm": contract_id_norm},
            {"value_id": "v_sign_date", "type": "datetime", "raw": sign_date_raw, "norm": sign_date_norm},
            {"value_id": "v_total_money", "type": "money", "raw": total_money_raw, "norm": total_money_norm},
            {"value_id": "v_start_date", "type": "datetime", "raw": start_date_raw, "norm": start_date_norm},
            {"value_id": "v_end_date", "type": "datetime", "raw": end_date_raw, "norm": end_date_norm}
        ],
        "ref_difficulty": ref_diff
    }
    (doc_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "doc_id": f"doc_{doc_idx:06d}",
        "seed": cfg["seed"] * 1000003 + doc_idx,
        "page_count": page_count,
        "noise_level": noise_level,
        "coref_difficulty": coref_diff,
        "ref_difficulty": ref_diff,
        "fictional_entities": True
    }
    (doc_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    base = Path(".")
    cfg = json.loads((base / "config_v1_audit200.json").read_text(encoding="utf-8"))
    template = Template((base / "template_contract_v1.html").read_text(encoding="utf-8"))
    blacklist = load_blacklist(cfg, base)

    out_dir = base / cfg["dataset_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, cfg["num_docs"] + 1):
        generate_one(i, cfg, out_dir, template, blacklist)

    print(f"[OK] HTML + labels + meta generated at: {out_dir}")


if __name__ == "__main__":
    main()
