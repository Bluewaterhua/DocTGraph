from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


_DATE_PATTERNS = [
    re.compile(r"^\s*(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\s*$"),
    re.compile(r"^\s*(\d{4})\u5e74(\d{1,2})\u6708(\d{1,2})\u65e5\s*$"),
]

_MASKED_BANK_ACCOUNT_PATTERN = re.compile(r"^\s*\d{4}\*+\d{4}\s*$")


def _quantize_money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def is_recoverable_value(text: str, norm_type: str) -> bool:
    if norm_type == "bank_account" and "*" in text:
        return False
    return True


def canonicalize_surface_form(text: str, norm_type: str) -> str | None:
    if norm_type == "bank_account" and "*" in text:
        digits_and_mask = re.sub(r"[^0-9*]", "", text)
        return digits_and_mask if _MASKED_BANK_ACCOUNT_PATTERN.match(digits_and_mask) else digits_and_mask
    return normalize_value(text, norm_type)


def normalize_datetime(text: str) -> str | None:
    for pattern in _DATE_PATTERNS:
        match = pattern.match(text)
        if match:
            year, month, day = (int(part) for part in match.groups())
            return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def normalize_money(text: str) -> str | None:
    cleaned = text.strip()
    cleaned = cleaned.replace("\u4eba\u6c11\u5e01", "")
    cleaned = cleaned.replace("\uffe5", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("\u5143\u6574", "")
    cleaned = cleaned.replace("\u5143", "")
    cleaned = cleaned.strip()
    multiplier = Decimal("1")
    if cleaned.endswith("\u4e07"):
        multiplier = Decimal("10000")
        cleaned = cleaned[:-1].strip()
    try:
        amount = Decimal(cleaned) * multiplier
    except InvalidOperation:
        return None
    return f"CNY:{_quantize_money(amount)}"


def normalize_phone(text: str) -> str:
    return re.sub(r"\D", "", text)


def normalize_bank_account(text: str) -> str | None:
    if "*" in text:
        return None
    return re.sub(r"\D", "", text)


def normalize_tax_no(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", text).upper()


def normalize_email(text: str) -> str:
    return text.strip().lower()


def normalize_contract_id(text: str) -> str:
    return re.sub(r"\s+", "", text).upper()


def normalize_value(text: str, norm_type: str) -> str | None:
    if norm_type == "datetime":
        return normalize_datetime(text)
    if norm_type == "money":
        return normalize_money(text)
    if norm_type == "phone":
        return normalize_phone(text)
    if norm_type == "bank_account":
        return normalize_bank_account(text)
    if norm_type == "tax_no":
        return normalize_tax_no(text)
    if norm_type == "email":
        return normalize_email(text)
    if norm_type == "contract_id":
        return normalize_contract_id(text)
    return None
