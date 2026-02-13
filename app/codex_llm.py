from __future__ import annotations

import asyncio
import random
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

from .config import settings

_PROFILE_CACHE: dict[str, dict] = {}
_DEFAULT_SUPPORT_REFUSAL = "I can only help with company services, plans, pricing, or support questions."
_LATIN_RE = re.compile(r"[A-Za-z]")


def _normalize_lang(lang: str) -> str:
    l = (lang or "").strip().lower()
    if l.startswith("km"):
        return "km"
    if l.startswith("vi"):
        return "vi"
    if l.startswith("hi"):
        return "hi"
    return "en"


def _resolve_profile_path(call_lang: str) -> Path:
    raw = settings.company_profile_path_for_lang(call_lang)
    p = Path(raw)
    if not p.is_absolute():
        p = Path(settings.codex_cd) / p
    return p


def _load_company_profile(call_lang: str) -> dict:
    path = _resolve_profile_path(call_lang)
    key = str(path)
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                data = {}
            _PROFILE_CACHE[key] = data
            return data
    except Exception:
        fallback = Path(settings.company_profile_path)
        if not fallback.is_absolute():
            fallback = Path(settings.codex_cd) / fallback
        fb_key = str(fallback)
        if fb_key in _PROFILE_CACHE:
            return _PROFILE_CACHE[fb_key]
        try:
            with open(fallback, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
                _PROFILE_CACHE[fb_key] = data
                return data
        except Exception:
            _PROFILE_CACHE[key] = {}
            return {}


_DEFAULT_LABELS = {
    "company": "Company",
    "tagline": "Tagline",
    "overview": "Overview",
    "services": "Services",
    "add_ons": "Add-ons",
    "plans": "Plans",
    "numbers": "Number pricing",
    "onboarding": "Onboarding",
    "sla": "SLA",
    "support": "Support",
    "compliance": "Compliance",
    "hours": "Hours",
    "email": "Email",
    "phone": "Phone",
    "response_time": "Response time",
    "overage": "Overage",
}


def _profile_labels(profile: dict) -> dict:
    labels = dict(_DEFAULT_LABELS)
    custom = profile.get("labels") if isinstance(profile, dict) else None
    if isinstance(custom, dict):
        for k, v in custom.items():
            if isinstance(k, str) and isinstance(v, str) and k in labels:
                labels[k] = v
    return labels


def _format_company_profile(profile: dict) -> str:
    if not profile:
        return ""
    labels = _profile_labels(profile)
    lines = []
    name = profile.get("company_name")
    if name:
        lines.append(f"{labels['company']}: {name}")
    tagline = profile.get("tagline")
    if tagline:
        lines.append(f"{labels['tagline']}: {tagline}")
    overview = profile.get("overview")
    if overview:
        lines.append(f"{labels['overview']}: {overview}")
    services = profile.get("services") or []
    if services:
        lines.append(f"{labels['services']}:")
        for svc in services:
            if not isinstance(svc, dict):
                continue
            svc_name = svc.get("name")
            desc = svc.get("description")
            if svc_name:
                line = f"- {svc_name}"
                if desc:
                    line += f": {desc}"
                lines.append(line)
            feats = svc.get("features") or []
            for feat in feats:
                if isinstance(feat, str):
                    lines.append(f"- {feat}")
    add_ons = profile.get("add_ons") or []
    if add_ons:
        lines.append(f"{labels['add_ons']}:")
        for item in add_ons:
            if isinstance(item, str):
                lines.append(f"- {item}")
    plans = profile.get("plans") or []
    if plans:
        lines.append(f"{labels['plans']}:")
        for plan in plans:
            if not isinstance(plan, dict):
                continue
            pname = plan.get("name")
            price = plan.get("price")
            includes = plan.get("includes")
            overage = plan.get("overage")
            if pname:
                line = f"- {pname}"
                if price:
                    line += f" ({price})"
                if includes:
                    line += f": {includes}"
                lines.append(line)
                if overage:
                    lines.append(f"- {labels['overage']}: {overage}")
    numbers = profile.get("numbers") or {}
    if isinstance(numbers, dict) and numbers:
        lines.append(f"{labels['numbers']}:")
        for k, v in numbers.items():
            if v:
                label = str(k).replace("_", " ").title()
                lines.append(f"- {label}: {v}")
    onboarding = profile.get("onboarding") or []
    if onboarding:
        lines.append(f"{labels['onboarding']}:")
        for step in onboarding:
            if isinstance(step, str):
                lines.append(f"- {step}")
    sla = profile.get("sla")
    if sla:
        lines.append(f"{labels['sla']}: {sla}")
    support = profile.get("support") or {}
    if isinstance(support, dict):
        hours = support.get("hours")
        email = support.get("email")
        phone = support.get("phone")
        response_time = support.get("response_time")
        if hours or email or phone:
            lines.append(f"{labels['support']}:")
            if hours:
                lines.append(f"- {labels['hours']}: {hours}")
            if email:
                lines.append(f"- {labels['email']}: {email}")
            if phone:
                lines.append(f"- {labels['phone']}: {phone}")
            if response_time:
                lines.append(f"- {labels['response_time']}: {response_time}")
    compliance = profile.get("compliance") or []
    if compliance:
        lines.append(f"{labels['compliance']}:")
        for item in compliance:
            if isinstance(item, str):
                lines.append(f"- {item}")
    return "\n".join(lines).strip()


def _language_name(lang: str) -> str:
    l = (lang or "").strip().lower()
    if l.startswith("km"):
        return "Khmer"
    if l.startswith("vi"):
        return "Vietnamese"
    if l.startswith("hi"):
        return "Hindi"
    return "English"


def _lang_code(lang: str) -> str:
    l = (lang or "").strip().lower()
    if l.startswith("km"):
        return "km"
    if l.startswith("vi"):
        return "vi"
    if l.startswith("hi"):
        return "hi"
    return "en"

_KM_TOKEN_MAP = {
    r"\bSMS\b": "សារ",
    r"\bVoice\b": "សំឡេង",
    r"\bStarter\b": "ស្តាទ័រ",
    r"\bBusiness\b": "ប៊ីសិនណេស",
    r"\bEnterprise\b": "អិនធឺប្រាយស៍",
    r"\bIVR\b": "អាយវីអារ",
}


def _normalize_km_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    for pat, rep in _KM_TOKEN_MAP.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _number_tokens(text: str) -> list[str]:
    return re.findall(r"\$?\d+(?:[.,]\d+)?", text or "")


def _needs_khmer_rewrite(text: str) -> bool:
    return len(_LATIN_RE.findall(text or "")) >= 6


def _system_prompt(call_lang: str = "en") -> str:
    profile = _load_company_profile(call_lang)
    profile_text = _format_company_profile(profile)
    company_name = (profile.get("company_name") or "the company") if isinstance(profile, dict) else "the company"
    reply_language = _language_name(call_lang)
    rules = (
        "You are a professional support agent for the company.\n"
        "Only discuss the company services, plans, pricing, onboarding, support, or account issues.\n"
        "If a request is unrelated, refuse briefly and redirect to company topics.\n"
        "Never provide instructions to modify files, run commands, access systems, or change servers.\n"
        "Do not mention being an AI.\n"
        "Tone: professional, calm, and helpful. Use clear, natural sentences.\n"
        "Treat the internal company reference as facts, not as a script.\n"
        "Do not quote or recite headings, bullet labels, or raw lines from the reference.\n"
        "Paraphrase naturally like live human support, and explain in your own wording.\n"
        "Never answer in list format, bullets, headings, or field-by-field reading.\n"
        "Speak as natural conversation only.\n"
        "When information is long, summarize first and offer details if the caller wants them.\n"
        "Never invent prices, limits, or policy details; only use facts from the internal company reference.\n"
        "Avoid repeating the company name unless the user asks or clarity requires it. Use \"we\" and \"our\" instead.\n"
        "Vary phrasing and avoid repeating the same refusal wording.\n"
        "Only use a brief acknowledgment (e.g., \"Sure,\" \"Of course,\") when the user explicitly asks for help or a request. "
        "Do not start every reply with an acknowledgment.\n"
        "Keep replies concise (1-2 short sentences). Ask a brief follow-up only if needed.\n"
        "Use spoken-style wording in the reply language, not document-style wording.\n"
        f"Reply language is fixed to {reply_language} for this call. Do not switch to another language unless the user explicitly asks to change language.\n"
    )
    prompt = f"{rules}\nInternal company reference (do not quote verbatim):\n{profile_text}\n"
    return prompt.replace("the company", company_name)


_BLOCKED_RE = re.compile(
    r"(delete|remove|rm\b|format|shutdown|reboot|chmod|chown|systemctl|apt|pip install|curl|wget|ssh|sudo|"
    r"docker|kubectl|git|file system|filesystem|edit file|modify file|create file|write file|run command)",
    re.IGNORECASE,
)


def _trim_sentences(text: str, max_sentences: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) <= max_sentences:
        return text.strip()
    return " ".join(parts[:max_sentences]).strip()


_LEADING_LABEL_RE = re.compile(
    r"(^|[.!?]\s+)([A-Za-z\u1780-\u17FF][A-Za-z0-9\u1780-\u17FF\s\-]{1,30}):\s+"
)


def _de_scriptify_reply(text: str) -> str:
    # Collapse multiline/list-like output into natural spoken text.
    t = (text or "").strip()
    if not t:
        return t
    t = t.replace("\r", "\n")
    t = re.sub(r"\n+\s*[-*•]\s*", " ", t)
    t = re.sub(r"\n+", " ", t)
    t = _LEADING_LABEL_RE.sub(r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _humanize_reply(text: str) -> str:
    t = _de_scriptify_reply(text)
    if not t:
        return t
    t = _trim_sentences(t, 2)
    return t


def _should_refuse(text: str) -> bool:
    if not settings.support_only:
        return False
    return bool(_BLOCKED_RE.search(text))


def _refusal_text() -> str:
    raw = settings.support_refusal.strip()
    if not raw:
        return "I can only help with our services, plans, pricing, or support questions."
    # Allow multiple variants separated by |
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    if not parts:
        return "I can only help with our services, plans, pricing, or support questions."
    return random.choice(parts)



def _default_refusal_for_lang(call_lang: str) -> str:
    l = _lang_code(call_lang)
    if l == "km":
        return "ខ្ញុំអាចជួយបានតែសំណួរដែលពាក់ព័ន្ធនឹងសេវាកម្ម គម្រោង តម្លៃ និងជំនួយអតិថិជនរបស់ក្រុមហ៊ុនប៉ុណ្ណោះ។"
    if l == "vi":
        return "Toi chi ho tro cac cau hoi lien quan den dich vu, goi cuoc, bang gia va ho tro khach hang cua cong ty."
    if l == "hi":
        return "Main sirf hamari sevaon, plans, pricing aur support se jude sawalon mein madad kar sakta hoon."
    return "I can only help with our services, plans, pricing, or support questions."


def _refusal_text_for_lang(call_lang: str) -> str:
    raw = settings.support_refusal.strip()
    if not raw or raw == _DEFAULT_SUPPORT_REFUSAL:
        return _default_refusal_for_lang(call_lang)
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    if not parts:
        return _default_refusal_for_lang(call_lang)
    return random.choice(parts)


async def _rewrite_reply_for_khmer(text: str) -> str:
    source = (text or "").strip()
    if not source:
        return source

    async def _rewrite(prompt: str) -> str:
        cmd = settings.codex_cmd or "codex"
        args = [cmd, "exec", "--skip-git-repo-check", "--json"]
        if settings.codex_model:
            args += ["--model", settings.codex_model]
        if settings.codex_profile:
            args += ["--profile", settings.codex_profile]
        args.append(prompt)

        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=settings.codex_cd or os.getcwd(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=min(settings.codex_timeout_sec, 20)
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ""
        except asyncio.CancelledError:
            proc.kill()
            await proc.communicate()
            raise

        if stderr:
            logger.debug("rewrite-km stderr: {}", stderr.decode(errors="ignore").strip())

        out = ""
        for raw_line in stdout.decode(errors="ignore").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("type") == "item.completed":
                item = evt.get("item") or {}
                if item.get("type") == "agent_message":
                    out = (item.get("text", "") or "").strip()
        return out

    first_prompt = (
        "Rewrite the following support message into natural Khmer.\\n"
        "Rules:\\n"
        "- Keep meaning and numbers exactly.\\n"
        "- Do not add new facts.\\n"
        "- Use Khmer script as much as possible.\\n"
        "- Return plain text only.\\n\\n"
        f"Text: {source}\\n"
    )
    rewritten = (await _rewrite(first_prompt)).strip() or source

    if _LATIN_RE.search(rewritten):
        strict_prompt = (
            "Rewrite this support message into Khmer with NO Latin letters at all.\\n"
            "Rules:\\n"
            "- Keep meaning and numbers exactly.\\n"
            "- Transliterate product names into Khmer script.\\n"
            "- Do not add new facts.\\n"
            "- Return plain text only.\\n\\n"
            f"Text: {rewritten}\\n"
        )
        stricter = (await _rewrite(strict_prompt)).strip()
        if stricter:
            rewritten = stricter

    return rewritten or source

async def run_codex(
    prompt: str,
    session_id: Optional[str],
    *,
    call_lang: str = "en",
) -> Tuple[str, Optional[str]]:
    user_text = (prompt or "").strip()
    if not user_text:
        return "", session_id
    if _should_refuse(user_text):
        return _refusal_text_for_lang(call_lang), session_id

    sys_prompt = _system_prompt(call_lang)
    full_prompt = f"{sys_prompt}\nUser: {user_text}\nSupport:"

    cmd = settings.codex_cmd or "codex"
    args = [cmd, "exec", "--skip-git-repo-check", "--json"]
    if settings.codex_model:
        args += ["--model", settings.codex_model]
    if settings.codex_profile:
        args += ["--profile", settings.codex_profile]
    # Use new thread each turn to keep strict system prompt
    args.append(full_prompt)

    logger.info("codex exec: {}", " ".join(args[:-1]))

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=settings.codex_cd or os.getcwd(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=settings.codex_timeout_sec
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise
    except asyncio.CancelledError:
        proc.kill()
        await proc.communicate()
        raise

    if stderr:
        logger.warning("codex stderr: {}", stderr.decode(errors="ignore").strip())

    text = ""
    new_session_id: Optional[str] = session_id
    for raw_line in stdout.decode(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "thread.started" and evt.get("thread_id"):
            new_session_id = evt["thread_id"]
        if evt.get("type") == "item.completed":
            item = evt.get("item") or {}
            if item.get("type") == "agent_message":
                text = item.get("text", "") or ""

    reply = _humanize_reply(text)
    if _lang_code(call_lang) == "km":
        reply = _normalize_km_text(reply)
        if _needs_khmer_rewrite(reply):
            rewritten = _humanize_reply(await _rewrite_reply_for_khmer(reply))
            rewritten = _normalize_km_text(rewritten)
            if _number_tokens(rewritten) == _number_tokens(reply):
                reply = rewritten

    return reply, new_session_id


async def detect_end_call_intent(user_text: str) -> Tuple[bool, float]:
    if not settings.end_call_enabled:
        return False, 0.0
    text = (user_text or "").strip()
    if not text:
        return False, 0.0

    sys_prompt = (
        "You are a classifier that decides if the caller wants to end the call.\n"
        "Return only JSON in one line with keys: end_call (true/false) and confidence (0-1).\n"
        "Be conservative: only return true if the user clearly wants to end the call.\n"
    )
    full_prompt = f"{sys_prompt}\nUser: {text}\nOutput:"

    cmd = settings.codex_cmd or "codex"
    args = [cmd, "exec", "--skip-git-repo-check", "--json"]
    if settings.codex_model:
        args += ["--model", settings.codex_model]
    args.append(full_prompt)

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=settings.codex_cd or os.getcwd(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=settings.end_call_timeout_sec
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return False, 0.0
    except asyncio.CancelledError:
        proc.kill()
        await proc.communicate()
        raise

    if stderr:
        logger.debug("end-call stderr: {}", stderr.decode(errors="ignore").strip())

    result_text = ""
    for raw_line in stdout.decode(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "item.completed":
            item = evt.get("item") or {}
            if item.get("type") == "agent_message":
                result_text = item.get("text", "") or ""

    try:
        data = json.loads(result_text)
        end_call = bool(data.get("end_call"))
        conf = float(data.get("confidence", 0.0))
        return end_call, max(0.0, min(1.0, conf))
    except Exception:
        return False, 0.0
