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

_PROFILE_CACHE: Optional[dict] = None


def _load_company_profile() -> dict:
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE
    path = Path(settings.company_profile_path)
    if not path.is_absolute():
        path = Path(settings.codex_cd) / path
    try:
        with open(path, "r", encoding="utf-8") as f:
            _PROFILE_CACHE = json.load(f)
    except Exception:
        _PROFILE_CACHE = {}
    return _PROFILE_CACHE


def _format_company_profile(profile: dict) -> str:
    if not profile:
        return ""
    lines = []
    name = profile.get("company_name")
    if name:
        lines.append(f"Company: {name}")
    tagline = profile.get("tagline")
    if tagline:
        lines.append(f"Tagline: {tagline}")
    overview = profile.get("overview")
    if overview:
        lines.append(f"Overview: {overview}")
    services = profile.get("services") or []
    if services:
        lines.append("Services:")
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
        lines.append("Add-ons:")
        for item in add_ons:
            if isinstance(item, str):
                lines.append(f"- {item}")
    plans = profile.get("plans") or []
    if plans:
        lines.append("Plans:")
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
                    lines.append(f"- Overage: {overage}")
    numbers = profile.get("numbers") or {}
    if isinstance(numbers, dict) and numbers:
        lines.append("Number pricing:")
        for k, v in numbers.items():
            if v:
                label = str(k).replace("_", " ").title()
                lines.append(f"- {label}: {v}")
    onboarding = profile.get("onboarding") or []
    if onboarding:
        lines.append("Onboarding:")
        for step in onboarding:
            if isinstance(step, str):
                lines.append(f"- {step}")
    sla = profile.get("sla")
    if sla:
        lines.append(f"SLA: {sla}")
    support = profile.get("support") or {}
    if isinstance(support, dict):
        hours = support.get("hours")
        email = support.get("email")
        phone = support.get("phone")
        response_time = support.get("response_time")
        if hours or email or phone:
            lines.append("Support:")
            if hours:
                lines.append(f"- Hours: {hours}")
            if email:
                lines.append(f"- Email: {email}")
            if phone:
                lines.append(f"- Phone: {phone}")
            if response_time:
                lines.append(f"- Response time: {response_time}")
    compliance = profile.get("compliance") or []
    if compliance:
        lines.append("Compliance:")
        for item in compliance:
            if isinstance(item, str):
                lines.append(f"- {item}")
    return "\n".join(lines).strip()


def _system_prompt() -> str:
    profile = _load_company_profile()
    profile_text = _format_company_profile(profile)
    company_name = (profile.get("company_name") or "the company") if isinstance(profile, dict) else "the company"
    rules = (
        "You are a professional support agent for the company.\n"
        "Only discuss the company services, plans, pricing, onboarding, support, or account issues.\n"
        "If a request is unrelated, refuse briefly and redirect to company topics.\n"
        "Never provide instructions to modify files, run commands, access systems, or change servers.\n"
        "Do not mention being an AI.\n"
        "Tone: professional, calm, and helpful. Use clear, natural sentences.\n"
        "Avoid repeating the company name unless the user asks or clarity requires it. Use \"we\" and \"our\" instead.\n"
        "Vary phrasing and avoid repeating the same refusal wording.\n"
        "Only use a brief acknowledgment (e.g., \"Sure,\" \"Of course,\") when the user explicitly asks for help or a request. "
        "Do not start every reply with an acknowledgment.\n"
        "Keep replies concise (1-3 sentences). Ask a brief follow-up only if needed.\n"
    )
    prompt = f"{rules}\nCompany profile:\n{profile_text}\n"
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


def _humanize_reply(text: str) -> str:
    t = text.strip()
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


async def run_codex(prompt: str, session_id: Optional[str]) -> Tuple[str, Optional[str]]:
    user_text = (prompt or "").strip()
    if not user_text:
        return "", session_id
    if _should_refuse(user_text):
        return _refusal_text(), session_id

    sys_prompt = _system_prompt()
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

    return _humanize_reply(text), new_session_id
