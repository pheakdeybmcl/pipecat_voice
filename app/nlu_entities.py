from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\t\r\n,.;:!?/\\()\[\]{}\"'`~@#$%^&*_+=|<>-]+")


def _normalize_text(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    t = _PUNCT_RE.sub(" ", t)
    t = _SPACE_RE.sub(" ", t).strip()
    return t


def _contains_phrase(haystack: str, phrase: str) -> bool:
    if not haystack or not phrase:
        return False
    return f" {phrase} " in f" {haystack} "


_SERVICE_ALIASES: dict[str, tuple[str, ...]] = {
    "eSIM": (
        "esim",
        "e sim",
        "e-sim",
        "isim",
        "isima",
        "easy sim",
        "i sim",
        "អ៊ីស៊ីម",
        "អេស៊ីម",
        "sim dien tu",
        "sim ao",
        "ईसिम",
        "ई सिम",
    ),
    "SIM": (
        "sim",
        "sim card",
        "physical sim",
        "ស៊ីម",
        "sim vat ly",
        "thẻ sim",
        "फिजिकल सिम",
    ),
    "Roaming": (
        "roaming",
        "data roaming",
        "roam",
        "រ៉ូមីង",
        "chuyen vung",
        "data chuyen vung",
        "रोमिंग",
    ),
    "Wi-Fi": (
        "wifi",
        "wi fi",
        "wi-fi",
        "hotspot",
        "pocket wifi",
        "វ៉ាយហ្វាយ",
        "internet",
        "wifi du lich",
        "वाई फाई",
        "हॉटस्पॉट",
    ),
}

_COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "Singapore": ("singapore", "sg", "សិង្ហបុរី", "sing", "xin ga po", "सिंगापुर"),
    "Thailand": ("thailand", "thai", "ថៃ", "thai lan", "thái lan", "थाईलैंड"),
    "Vietnam": ("vietnam", "viet nam", "វៀតណាម", "việt nam", "वियतनाम"),
    "Cambodia": ("cambodia", "khmer", "កម្ពុជា", "campuchia", "कंबोडिया"),
    "Malaysia": ("malaysia", "ម៉ាឡេស៊ី", "malay", "मलेशिया"),
}

_INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "pricing": (
        "price",
        "cost",
        "how much",
        "rate",
        "tariff",
        "តម្លៃ",
        "ប៉ុន្មាន",
        "gia",
        "bao nhieu",
        "giá",
        "कीमत",
        "कितना",
    ),
    "availability": (
        "available",
        "have",
        "offer",
        "មាន",
        "អាច",
        "co khong",
        "có không",
        "san co",
        "उपलब्ध",
        "मिलेगा",
    ),
    "setup": (
        "activate",
        "setup",
        "onboarding",
        "register",
        "ចុះឈ្មោះ",
        "បើក",
        "kich hoat",
        "kích hoạt",
        "dang ky",
        "đăng ký",
        "एक्टिवेट",
        "रजिस्टर",
    ),
}

_SERVICE_HINTS = (
    "service",
    "plan",
    "package",
    "tariff",
    "esim",
    "sim",
    "roaming",
    "wifi",
    "សេវាកម្ម",
    "គម្រោង",
    "dich vu",
    "gói cước",
    "goi cuoc",
    "ke hoach",
    "सेवा",
    "प्लान",
    "पैकेज",
)


@dataclass
class EntityResolution:
    raw_text: str
    normalized_text: str
    canonical_text: str
    service: Optional[str] = None
    service_confidence: float = 0.0
    country: Optional[str] = None
    country_confidence: float = 0.0
    intent: Optional[str] = None
    intent_confidence: float = 0.0
    confidence: float = 0.0
    needs_confirmation: bool = False
    likely_service_query: bool = False
    inferred_from_context: bool = False
    debug_notes: list[str] = field(default_factory=list)


def _best_alias_match(text_norm: str, aliases_by_name: dict[str, tuple[str, ...]]) -> tuple[Optional[str], float, Optional[str]]:
    best_name: Optional[str] = None
    best_score = 0.0
    best_alias: Optional[str] = None

    for name, aliases in aliases_by_name.items():
        for alias in aliases:
            alias_norm = _normalize_text(alias)
            if not alias_norm:
                continue
            if _contains_phrase(text_norm, alias_norm):
                score = 0.99 if len(alias_norm) > 3 else 0.95
            else:
                # Fuzzy on short n-grams to recover ASR errors like "isima" -> "esim".
                score = 0.0
                words = text_norm.split()
                if not words:
                    continue
                alias_words = alias_norm.split()
                n = len(alias_words)
                for i in range(0, len(words)):
                    cand = " ".join(words[i : i + n]) if n > 0 else words[i]
                    if not cand:
                        continue
                    ratio = SequenceMatcher(a=cand, b=alias_norm).ratio()
                    if ratio > score:
                        score = ratio
            if score > best_score:
                best_name = name
                best_score = score
                best_alias = alias_norm
    return best_name, best_score, best_alias


def _best_intent(text_norm: str) -> tuple[Optional[str], float]:
    best_intent: Optional[str] = None
    best_score = 0.0
    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            pat_norm = _normalize_text(pat)
            if _contains_phrase(text_norm, pat_norm):
                score = 0.95
            else:
                score = 0.0
            if score > best_score:
                best_intent = intent
                best_score = score
    return best_intent, best_score


def resolve_entities(
    text: str,
    *,
    auto_threshold: float = 0.84,
    confirm_threshold: float = 0.66,
) -> EntityResolution:
    raw = (text or "").strip()
    norm = _normalize_text(raw)
    result = EntityResolution(
        raw_text=raw,
        normalized_text=norm,
        canonical_text=raw,
    )
    if not norm:
        return result

    service, service_score, service_alias = _best_alias_match(norm, _SERVICE_ALIASES)
    country, country_score, _ = _best_alias_match(norm, _COUNTRY_ALIASES)
    intent, intent_score = _best_intent(norm)

    if service and service_score >= confirm_threshold:
        result.service = service
        result.service_confidence = service_score
        if service_alias and service_alias != service.lower():
            result.canonical_text = re.sub(
                re.escape(service_alias),
                service,
                norm,
                flags=re.IGNORECASE,
            )
        else:
            result.canonical_text = norm
        result.debug_notes.append(f"service:{service}@{service_score:.2f}")

    if country and country_score >= confirm_threshold:
        result.country = country
        result.country_confidence = country_score
        result.debug_notes.append(f"country:{country}@{country_score:.2f}")

    if intent and intent_score >= 0.9:
        result.intent = intent
        result.intent_confidence = intent_score
        result.debug_notes.append(f"intent:{intent}")

    result.likely_service_query = any(_contains_phrase(norm, _normalize_text(h)) for h in _SERVICE_HINTS)
    if result.service and result.service_confidence < auto_threshold and result.service_confidence >= confirm_threshold:
        result.needs_confirmation = True

    confs = [c for c in (result.service_confidence, result.country_confidence, result.intent_confidence) if c > 0]
    result.confidence = max(confs) if confs else 0.0
    return result
