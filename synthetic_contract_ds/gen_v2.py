import random
import string
from datetime import datetime
from typing import List, Tuple


FICTIONAL_PLACES = [
    "\u5c9a\u5dde",
    "\u5317\u5c94",
    "\u4e1c\u6f9c",
    "\u82cf\u5ddd",
    "\u5b81\u6e2f",
    "\u8854\u539f",
    "\u4e91\u6e2f",
    "\u9752\u6cfd",
    "\u6c90\u57ce",
    "\u6f2f\u5c7f",
    "\u96fe\u5ddd",
    "\u6fb3\u5dde",
    "\u6c90\u539f",
    "\u6ced\u57ce",
    "\u5c9a\u6d77",
]

BRAND_SYLLABLES = [
    "\u661f",
    "\u701a",
    "\u6668",
    "\u4e91",
    "\u8854",
    "\u6714",
    "\u8fbe",
    "\u6cfd",
    "\u9701",
    "\u5c9a",
    "\u9a90",
    "\u666f",
    "\u62d3",
    "\u6db5",
    "\u6d1b",
    "\u7426",
    "\u70c1",
    "\u822a",
]

INDUSTRIES = [
    "\u4fe1\u606f\u6280\u672f",
    "\u667a\u80fd\u7cfb\u7edf",
    "\u6570\u636e\u670d\u52a1",
    "\u7cfb\u7edf\u96c6\u6210",
    "\u5de5\u7a0b\u54a8\u8be2",
    "\u8f6f\u4ef6\u79d1\u6280",
    "\u901a\u4fe1\u6280\u672f",
    "\u7f51\u7edc\u5b89\u5168",
    "\u81ea\u52a8\u5316",
    "\u6570\u5b57\u79d1\u6280",
    "\u5de5\u4e1a\u670d\u52a1",
]

ORG_FORMS = [
    "\u6709\u9650\u516c\u53f8",
    "\u80a1\u4efd\u6709\u9650\u516c\u53f8",
    "\u6709\u9650\u8d23\u4efb\u516c\u53f8",
]

INSTITUTION_TYPES = [
    "\u7814\u7a76\u9662",
    "\u68c0\u6d4b\u4e2d\u5fc3",
    "\u8ba4\u8bc1\u4e2d\u5fc3",
    "\u6570\u636e\u4e2d\u5fc3",
    "\u4fe1\u606f\u6240",
    "\u54a8\u8be2\u4e2d\u5fc3",
]

CONTACT_FAMILY = [
    "\u8d75",
    "\u94b1",
    "\u5b59",
    "\u674e",
    "\u5468",
    "\u5434",
    "\u90d1",
    "\u738b",
    "\u51af",
    "\u9648",
    "\u891a",
    "\u536b",
    "\u848b",
    "\u6c88",
    "\u97e9",
    "\u6768",
]

CONTACT_GIVEN = [
    "\u5b50\u6db5",
    "\u6d69\u7136",
    "\u601d\u8fdc",
    "\u660e\u54f2",
    "\u82e5\u6eaa",
    "\u6893\u8f69",
    "\u96e8\u67cf",
    "\u5609\u6021",
    "\u6587\u535a",
    "\u5929\u5b87",
    "\u5b50\u58a8",
    "\u6b23\u59a4",
    "\u4e00\u9e23",
    "\u6c90\u5b87",
]

ROADS = [
    "\u661f\u8f89\u8def",
    "\u4e91\u8857\u5927\u9053",
    "\u5c9a\u5dde\u4e1c\u8def",
    "\u5317\u5c94\u897f\u8857",
    "\u6ced\u57ce\u4e2d\u8def",
    "\u9752\u6cfd\u79d1\u521b\u8def",
    "\u6f9c\u6e56\u8def",
    "\u6d77\u5dde\u8def",
]


def _rand_brand(rng: random.Random) -> str:
    a = rng.choice(BRAND_SYLLABLES)
    b = rng.choice([x for x in BRAND_SYLLABLES if x != a])
    return a + b


def safe_company_name(rng: random.Random, blacklist: List[str]) -> str:
    for _ in range(500):
        name = f"{rng.choice(FICTIONAL_PLACES)}{_rand_brand(rng)}{rng.choice(INDUSTRIES)}{rng.choice(ORG_FORMS)}"
        if any(blocked in name for blocked in blacklist):
            continue
        return name
    suffix = "".join(rng.choice(string.ascii_uppercase) for _ in range(4))
    return f"{rng.choice(FICTIONAL_PLACES)}{_rand_brand(rng)}{rng.choice(INDUSTRIES)}{suffix}\u6709\u9650\u516c\u53f8"


def institution_name(rng: random.Random, blacklist: List[str]) -> str:
    for _ in range(500):
        name = f"{rng.choice(FICTIONAL_PLACES)}{_rand_brand(rng)}{rng.choice(INSTITUTION_TYPES)}"
        if any(blocked in name for blocked in blacklist):
            continue
        return name
    return f"{rng.choice(FICTIONAL_PLACES)}{_rand_brand(rng)}\u7814\u7a76\u9662"


def build_aliases(rng: random.Random, canonical: str, role_term: str) -> List[str]:
    short = canonical
    for suffix in ORG_FORMS:
        short = short.replace(suffix, "")
    brand = short[2:4] if len(short) >= 4 else short
    en = f"{brand}Tech"
    abbr = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
    pronouns = ["\u672c\u516c\u53f8", "\u8d35\u65b9", "\u53cc\u65b9", "\u59d4\u6258\u65b9", "\u53d7\u6258\u65b9"]
    return [canonical, short, brand, en, abbr, role_term] + pronouns


def gen_phone(rng: random.Random) -> Tuple[str, str]:
    norm = "1" + "".join(str(rng.randint(0, 9)) for _ in range(10))
    style = rng.choice(["plain", "spaced", "dashed", "mixed_space", "paren"])
    if style == "plain":
        raw = norm
    elif style == "spaced":
        raw = f"{norm[:3]} {norm[3:7]} {norm[7:]}"
    elif style == "dashed":
        raw = f"{norm[:3]}-{norm[3:7]}-{norm[7:]}"
    elif style == "mixed_space":
        raw = f"{norm[:3]} {norm[3:5]} {norm[5:7]} {norm[7:]}"
    else:
        raw = f"({norm[:3]}) {norm[3:7]}-{norm[7:]}"
    return raw, norm


def gen_tax_no(rng: random.Random) -> Tuple[str, str]:
    norm = "".join(rng.choice(string.digits + string.ascii_uppercase) for _ in range(18))
    style = rng.choice(["plain", "spaced6", "spaced_var", "dashed", "lower"])
    if style == "plain":
        raw = norm
    elif style == "spaced6":
        raw = f"{norm[:6]} {norm[6:12]} {norm[12:]}"
    elif style == "spaced_var":
        raw = f"{norm[:4]} {norm[4:9]} {norm[9:14]} {norm[14:]}"
    elif style == "dashed":
        raw = f"{norm[:6]}-{norm[6:12]}-{norm[12:]}"
    else:
        raw = norm.lower()
    return raw, norm


def gen_bank_account(rng: random.Random) -> Tuple[str, str]:
    norm = "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([16, 18, 19])))
    raw_style = rng.choice(["plain", "spaced", "dashed", "masked", "grouped_mixed"])
    if raw_style == "plain":
        raw = norm
    elif raw_style == "spaced":
        raw = " ".join([norm[i:i + 4] for i in range(0, len(norm), 4)])
    elif raw_style == "dashed":
        raw = "-".join([norm[i:i + 4] for i in range(0, len(norm), 4)])
    elif raw_style == "grouped_mixed":
        groups = []
        start = 0
        while start < len(norm):
            width = 4 if (len(norm) - start) > 5 else rng.choice([3, 4, 5])
            groups.append(norm[start:start + width])
            start += width
        raw = rng.choice([" ", "-"]).join(groups)
    else:
        raw = norm[:4] + "****" + norm[-4:]
    return raw, norm


def gen_address(rng: random.Random) -> str:
    road = rng.choice(ROADS)
    number = rng.randint(1, 999)
    building = rng.randint(1, 30)
    room = rng.randint(101, 2808)
    return f"{rng.choice(FICTIONAL_PLACES)}{road}{number}\u53f7{building}\u5e62{room}\u5ba4"


def gen_contact_name(rng: random.Random) -> str:
    return rng.choice(CONTACT_FAMILY) + rng.choice(CONTACT_GIVEN)


def gen_email(rng: random.Random, brand2: str) -> str:
    dom = rng.choice(["example.com", "sample.org", "demo.cn", "mail.test", "contract.lab"])
    prefix = brand2.lower() if brand2 else "user"
    style = rng.choice(["plain", "dot", "underscore", "suffix_letter"])
    if style == "plain":
        local = prefix + str(rng.randint(10, 9999))
    elif style == "dot":
        local = prefix + "." + str(rng.randint(10, 9999))
    elif style == "underscore":
        local = prefix + "_" + str(rng.randint(10, 9999))
    else:
        local = prefix + str(rng.randint(10, 999)) + rng.choice(string.ascii_lowercase)
    return f"{local}@{dom}"


def money_norm(amount_int: int) -> str:
    return f"CNY:{amount_int:.2f}"


def money_raw_variant(rng: random.Random, amount: int) -> str:
    styles = ["int_yuan", "comma2", "comma0", "wan", "symbol_wan", "symbol_plain", "cn_upper"]
    style = rng.choice(styles)
    if style == "int_yuan":
        return f"{amount}\u5143"
    if style == "comma2":
        return f"{amount:,.2f}\u5143"
    if style == "comma0":
        return f"{amount:,}\u5143"
    if style == "wan":
        return f"{round(amount / 10000, 2)}\u4e07\u5143"
    if style == "symbol_wan":
        return f"\uffe5{round(amount / 10000, 1)}\u4e07"
    if style == "symbol_plain":
        return f"\uffe5{amount:,.2f}"
    return f"\u4eba\u6c11\u5e01{amount}\u5143\u6574"


def date_norm(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def date_raw_variant(rng: random.Random, dt: datetime) -> str:
    style = rng.choice(["dash", "dot", "slash", "cn", "compact", "year_month_only"])
    if style == "dash":
        return dt.strftime("%Y-%m-%d")
    if style == "dot":
        return dt.strftime("%Y.%m.%d")
    if style == "slash":
        return dt.strftime("%Y/%m/%d")
    if style == "cn":
        return f"{dt.year}\u5e74{dt.month}\u6708{dt.day}\u65e5"
    if style == "compact":
        return dt.strftime("%Y%m%d")
    return f"{dt.year}\u5e74{dt.month}\u6708"
