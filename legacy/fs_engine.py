# fs_engine.py
from __future__ import annotations
# FreeSWITCH mod_audio_stream WS server with:
# - VAD utterance segmentation
# - STT (faster-whisper)
# - LLM (Ollama OpenAI-compatible /v1)
# - TTS (edge_tts -> wav/mp3)
# - Optional ESL listener to auto-play streamAudio responses

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import unquote
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
import edge_tts
import httpx

# ---------------- Config ----------------
# Load .env (optional) before reading environment variables
def _load_env_file(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        entries: dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].lstrip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key:
                    entries[key] = val
        override = entries.get("FS_ENGINE_ENV_OVERRIDE", "").lower() in ("1", "true", "yes")
        for key, val in entries.items():
            if override or key not in os.environ:
                os.environ[key] = val
    except Exception:
        # Best effort: if parsing fails, fall back to existing env vars.
        return


_load_env_file(os.getenv("FS_ENGINE_ENV", ".env"))

def _load_company_profile(path: str) -> dict:
    if not path:
        return {}
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _format_company_profile(profile: dict) -> str:
    if not profile:
        return ""
    lines: list[str] = []
    name = profile.get("company_name")
    if name:
        lines.append(f"Company: {name}")
    tagline = profile.get("tagline")
    if tagline:
        lines.append(f"Tagline: {tagline}")
    services = profile.get("services") or []
    if services:
        lines.append("Services:")
        for svc in services:
            title = svc.get("name", "Service")
            desc = svc.get("description", "")
            if desc:
                lines.append(f"- {title}: {desc}")
            else:
                lines.append(f"- {title}")
            feats = svc.get("features") or []
            if feats:
                lines.append(f"  Features: {', '.join(feats)}")
    plans = profile.get("plans") or []
    if plans:
        lines.append("Plans:")
        for plan in plans:
            pname = plan.get("name", "Plan")
            price = plan.get("price", "")
            includes = plan.get("includes", "")
            summary = pname
            if price:
                summary += f" ({price})"
            if includes:
                summary += f" — {includes}"
            lines.append(f"- {summary}")
    support = profile.get("support") or {}
    if support:
        lines.append("Support:")
        if support.get("hours"):
            lines.append(f"- Hours: {support['hours']}")
        if support.get("email"):
            lines.append(f"- Email: {support['email']}")
        if support.get("phone"):
            lines.append(f"- Phone: {support['phone']}")
    return "\n".join(lines)


def _compact(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _join_system_messages(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        if (m.get("role") or "").lower() != "system":
            continue
        content = (m.get("content") or "").strip()
        if content:
            parts.append(content)
    joined = "\n\n".join(parts).strip()
    if LLM_FINAL_ONLY and LLM_FINAL_INSTRUCTION:
        if LLM_FINAL_INSTRUCTION not in joined:
            joined = (joined + "\n\n" + LLM_FINAL_INSTRUCTION).strip()
    return joined


def _parse_stop_list(raw: str) -> list[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _render_prompt(messages: list[dict], include_system: bool = True) -> str:
    # Simple prompt format for /api/generate
    lines: list[str] = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        if role == "SYSTEM" and not include_system:
            continue
        content = m.get("content") or ""
        lines.append(f"{role}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _ollama_options() -> dict:
    opts = {
        "num_predict": LLM_MAX_TOKENS,
        "num_ctx": LLM_NUM_CTX,
        "temperature": LLM_TEMPERATURE,
    }
    if LLM_TOP_P is not None:
        opts["top_p"] = LLM_TOP_P
    if LLM_TOP_K is not None:
        opts["top_k"] = LLM_TOP_K
    if LLM_REPEAT_PENALTY is not None:
        opts["repeat_penalty"] = LLM_REPEAT_PENALTY
    if LLM_SEED is not None:
        opts["seed"] = LLM_SEED
    if LLM_NUM_THREAD is not None:
        opts["num_thread"] = LLM_NUM_THREAD
    if LLM_NUM_GPU is not None:
        opts["num_gpu"] = LLM_NUM_GPU
    if LLM_NUM_BATCH is not None:
        opts["num_batch"] = LLM_NUM_BATCH
    return opts


def _split_tts_chunks(text: str, min_chars: int) -> tuple[list[str], str]:
    if not text:
        return [], ""
    chunks: list[str] = []
    last = 0
    abbrev = {
        "mr",
        "mrs",
        "ms",
        "dr",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "u.s",
        "u.k",
        "u.s.a",
        "u.a.e",
    }

    # Only split at sentence boundaries to keep audio smooth.
    for m in re.finditer(r"[.!?]+|\n+", text):
        end = m.end()
        if end - last < min_chars:
            continue

        token = text[last:end].rstrip()
        if not token:
            continue

        boundary = m.group(0)
        is_newline = "\n" in boundary
        if not is_newline:
            # Ensure punctuation is a sentence end (followed by space or end).
            if end < len(text) and not text[end].isspace():
                continue
            # Skip common abbreviations (e.g., "Dr.", "U.S.").
            last_word = re.split(r"\s+", token)[-1].lower().strip("\"'()[]")
            last_word = re.sub(r"[.!?]+$", "", last_word)
            if last_word in abbrev:
                continue
            # Skip decimal numbers like 3.14
            if token and token[-1].isdigit() and end < len(text) and text[end].isdigit():
                continue

        chunk = text[last:end].strip()
        if chunk:
            chunks.append(chunk)
        last = end

    remainder = text[last:]
    return chunks, remainder




def _extract_generate_text(data: dict) -> tuple[str, bool]:
    resp = (data.get("response") or "").strip()
    if resp:
        return resp, False
    thinking = (data.get("thinking") or "").strip()
    if not thinking:
        return "", False
    # Fallback: use the last non-empty line from "thinking" to avoid silence.
    line = ""
    for cand in reversed(thinking.splitlines()):
        cand = cand.strip()
        if cand:
            line = cand
            break
    if not line:
        line = thinking
    line = re.sub(r"^(final|answer)\s*[:\-]\s*", "", line, flags=re.I).strip()
    return line, True


def _should_hangup(text: str) -> bool:
    if not HANGUP_ENABLED:
        return False
    norm = _normalize_text(text)
    if not norm:
        return False
    # Guard against accidental hangup intents.
    if any(p in norm for p in ("dont hang up", "don't hang up", "do not hang up", "not hang up")):
        return False
    for phrase in HANGUP_STRONG_PHRASES:
        if phrase and phrase in norm:
            return True
    words = norm.split()
    if len(words) > HANGUP_MAX_WORDS:
        return False
    for phrase in HANGUP_PHRASES:
        if phrase and phrase in norm:
            return True
    return False


async def _hangup_call(ws: WebSocket, state: SessionState) -> None:
    await handle_barge_in(state)
    await cancel_task(state.llm_task, "llm")
    if HANGUP_TEXT:
        await play_tts_text(ws, state, HANGUP_TEXT)
    if HANGUP_DELAY_MS > 0:
        await asyncio.sleep(HANGUP_DELAY_MS / 1000.0)
    try:
        await _esl_api_cmd(f"uuid_kill {state.session_id} NORMAL_CLEARING")
    except Exception as e:
        LOG.warning("[call %s] hangup failed: %s", state.session_id, e)


def _infer_service_topic(name: str) -> str:
    n = (name or "").lower()
    if any(k in n for k in ("voice", "calling", "call", "ivr", "sip", "pbx")):
        return "voice"
    if any(k in n for k in ("message", "sms", "mms", "text")):
        return "messaging"
    if "esim" in n or "sim" in n:
        return "esim"
    return ""


def _build_company_facts(profile: dict) -> dict:
    if not profile:
        return {"summary": "", "topics": {}}
    topics: dict[str, str] = {}
    lines: list[str] = []
    name = profile.get("company_name")
    if name:
        lines.append(f"Company: {name}")
    services = profile.get("services") or []
    service_names: list[str] = []
    for svc in services:
        svc_name = svc.get("name") if isinstance(svc, dict) else ""
        if svc_name:
            service_names.append(str(svc_name))
        topic = _infer_service_topic(str(svc_name))
        if topic:
            desc = svc.get("description", "") if isinstance(svc, dict) else ""
            feats = svc.get("features") or []
            chunk = [f"{svc_name}: {desc}".strip()]
            if feats:
                chunk.append("Features: " + ", ".join(feats))
            topics[topic] = "\n".join([c for c in chunk if c]).strip()
    if service_names:
        lines.append("Services: " + ", ".join(service_names))
    plans = profile.get("plans") or []
    if plans:
        plan_bits: list[str] = []
        for plan in plans:
            pname = plan.get("name", "Plan")
            price = plan.get("price", "")
            bit = pname
            if price:
                bit += f" ({price})"
            plan_bits.append(bit)
        lines.append("Plans: " + ", ".join(plan_bits))
        topics["pricing"] = "Plans: " + ", ".join(plan_bits)
        topics.setdefault("plans", topics["pricing"])
    support = profile.get("support") or {}
    if support:
        support_bits = []
        if support.get("hours"):
            support_bits.append(f"Hours: {support['hours']}")
        if support.get("email"):
            support_bits.append(f"Email: {support['email']}")
        if support.get("phone"):
            support_bits.append(f"Phone: {support['phone']}")
        if support_bits:
            topics["support"] = "Support: " + "; ".join(support_bits)
            lines.append("Support: " + "; ".join(support_bits))
    return {"summary": "\n".join(lines).strip(), "topics": topics}


def _detect_topics(prompt: str) -> list[str]:
    if not prompt:
        return []
    t = prompt.lower()
    mapping = {
        "voice": ["voice", "call", "calling", "ivr", "sip", "pbx", "phone"],
        "messaging": ["message", "sms", "mms", "text"],
        "esim": ["esim", "sim", "activation", "carrier", "coverage"],
        "pricing": ["price", "pricing", "cost", "plan", "package", "rate"],
        "billing": ["bill", "billing", "invoice", "payment", "refund"],
        "support": ["support", "help", "ticket", "issue", "problem"],
    }
    hits: list[str] = []
    for topic, keys in mapping.items():
        if any(k in t for k in keys):
            hits.append(topic)
    return hits


def _matches_allowed_topics(text: str, topics: list[str]) -> bool:
    if not topics:
        return True
    t = text.lower()
    return any(topic in t for topic in topics)


# Ollama (OpenAI-compatible)
def _normalize_ollama_base(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/v1"):
        return base
    if base.endswith("/api"):
        return base[:-4] + "/v1"
    return base + "/v1"

OLLAMA_BASE_URL = _normalize_ollama_base(
    os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)
OLLAMA_NATIVE_BASE = os.getenv("OLLAMA_NATIVE_BASE", "").strip().rstrip("/")
if not OLLAMA_NATIVE_BASE:
    if OLLAMA_BASE_URL.endswith("/v1"):
        OLLAMA_NATIVE_BASE = OLLAMA_BASE_URL[:-3]
    else:
        OLLAMA_NATIVE_BASE = OLLAMA_BASE_URL.rstrip("/")
OLLAMA_USE_NATIVE_FALLBACK = os.getenv("OLLAMA_USE_NATIVE_FALLBACK", "true").lower() == "true"
OLLAMA_MODE = os.getenv("OLLAMA_MODE", "chat").strip().lower()  # chat | generate
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
LLM_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_MAX_CONCURRENCY", "4")))
LLM_MAX_TOKENS = max(16, int(os.getenv("LLM_MAX_TOKENS", "120")))
LLM_TIMEOUT_SEC = max(3.0, float(os.getenv("LLM_TIMEOUT_SEC", "20")))
LLM_DISABLE_TIMEOUT = os.getenv("LLM_DISABLE_TIMEOUT", "false").lower() == "true"
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
LLM_NUM_CTX = max(256, int(os.getenv("LLM_NUM_CTX", "1024")))
LLM_TOP_P = _env_float("LLM_TOP_P")
LLM_TOP_K = _env_int("LLM_TOP_K")
LLM_REPEAT_PENALTY = _env_float("LLM_REPEAT_PENALTY")
LLM_SEED = _env_int("LLM_SEED")
LLM_NUM_THREAD = _env_int("LLM_NUM_THREAD")
LLM_NUM_GPU = _env_int("LLM_NUM_GPU")
LLM_NUM_BATCH = _env_int("LLM_NUM_BATCH")
LLM_RETRY_ON_EMPTY = os.getenv("LLM_RETRY_ON_EMPTY", "true").lower() == "true"
LLM_FINAL_ONLY = os.getenv("LLM_FINAL_ONLY", "true").lower() in ("1", "true", "yes")
LLM_FINAL_INSTRUCTION = os.getenv(
    "LLM_FINAL_INSTRUCTION",
    "Answer ONLY with the final response. Do not include analysis or thinking.",
).strip()
OLLAMA_STOP_RAW = os.getenv("OLLAMA_STOP", "")
LLM_STREAM = os.getenv("LLM_STREAM", "false").lower() in ("1", "true", "yes")
TTS_STREAM_MIN_CHARS = max(1, int(os.getenv("TTS_STREAM_MIN_CHARS", "20")))
LLM_EMPTY_REPLY = os.getenv(
    "LLM_EMPTY_REPLY",
    "Sorry, I didn't catch that. Please repeat.",
)
if not LLM_EMPTY_REPLY.strip():
    LLM_EMPTY_REPLY = "Please repeat."
LLM_SEMAPHORE = asyncio.Semaphore(LLM_MAX_CONCURRENCY)
llm_client = AsyncOpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)

STT_MAX_CONCURRENCY = max(1, int(os.getenv("STT_MAX_CONCURRENCY", "2")))
STT_SEMAPHORE = asyncio.Semaphore(STT_MAX_CONCURRENCY)

BASE_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are an assistant for this app and a professional call center agent.\n"
    "Respond like a human in a live phone call.\n"
    "\n"
    "Rules:\n"
    "- Match the length of your reply to the user's intent.\n"
    "- If the user gives a short or unclear input, respond briefly or ask a clarifying question.\n"
    "- Do not ask follow-up questions or offer extra help unless it's required to answer.\n"
    "- Do not over-explain unless the user asks for details.\n"
    "- Stop speaking as soon as your point is made.\n"
)
COMPANY_PROFILE_PATH = os.getenv("COMPANY_PROFILE_PATH", "")
COMPANY_PROFILE = _load_company_profile(COMPANY_PROFILE_PATH)
PROFILE_TEXT = _format_company_profile(COMPANY_PROFILE)
PROFILE_MAX_CHARS = int(os.getenv("PROFILE_MAX_CHARS", "800"))
if PROFILE_TEXT and len(PROFILE_TEXT) > PROFILE_MAX_CHARS:
    PROFILE_TEXT = PROFILE_TEXT[:PROFILE_MAX_CHARS].rstrip() + "..."
COMPANY_PROFILE_MODE = os.getenv("COMPANY_PROFILE_MODE", "topic").strip().lower()  # off|summary|topic
COMPANY_FACTS_MAX_CHARS = int(os.getenv("COMPANY_FACTS_MAX_CHARS", "600"))
COMPANY_FACTS_TOPIC_MAX_CHARS = int(os.getenv("COMPANY_FACTS_TOPIC_MAX_CHARS", "400"))
COMPANY_FACTS = _build_company_facts(COMPANY_PROFILE)
_company_name = COMPANY_PROFILE.get("company_name", "").strip() if isinstance(COMPANY_PROFILE, dict) else ""
_company_aliases = []
if isinstance(COMPANY_PROFILE, dict):
    _company_aliases = [a for a in (COMPANY_PROFILE.get("aliases") or []) if isinstance(a, str)]
ENV_COMPANY_ALIASES = [a.strip() for a in os.getenv("COMPANY_ALIASES", "").split(",") if a.strip()]
COMPANY_ALIASES = [a for a in (_company_aliases + ENV_COMPANY_ALIASES) if a]
COMPANY_RULES = (
    "Company-only policy:\n"
    "- Only answer questions related to the company, its services, plans, pricing, support, or account issues.\n"
    "- If a request is unrelated, politely refuse and redirect to company topics.\n"
)
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n\n" + COMPANY_RULES

# Optional greeting (no LLM)
GREETING_ENABLED = os.getenv("GREETING_ENABLED", "false").lower() in ("1", "true", "yes")
GREETING_TEXT = os.getenv("GREETING_TEXT", "").strip()
if not GREETING_TEXT and isinstance(COMPANY_PROFILE, dict):
    GREETING_TEXT = str(COMPANY_PROFILE.get("greeting", "")).strip()
if GREETING_TEXT and _company_name:
    try:
        GREETING_TEXT = GREETING_TEXT.format(company_name=_company_name)
    except Exception:
        pass

ALLOWED_TOPICS = [t.strip().lower() for t in os.getenv("ALLOWED_TOPICS", "").split(",") if t.strip()]
OFFTOPIC_RESPONSE = os.getenv(
    "OFFTOPIC_RESPONSE",
    "I can help with BigfootMediaTech services like voice, messaging, eSIM, plans, or support. How can I assist?",
)
OFFTOPIC_RESPONSE_SHORT = os.getenv(
    "OFFTOPIC_RESPONSE_SHORT",
    "Please say one of: voice, messaging, eSIM, plans, billing, or support.",
)
OFFTOPIC_REPEAT_LIMIT = max(1, int(os.getenv("OFFTOPIC_REPEAT_LIMIT", "2")))

# Audio input expected from FreeSWITCH: PCM16 little-endian, mono.
SAMPLE_RATE = int(os.getenv("FS_SAMPLE_RATE", "16000"))
SAMPLE_WIDTH = 2  # bytes per sample (PCM16)
FRAME_MS = 20     # VAD supports only 10/20/30ms
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH

MIN_UTTERANCE_MS = int(os.getenv("MIN_UTTERANCE_MS", "400"))
MIN_UTTERANCE_FRAMES = max(1, int(MIN_UTTERANCE_MS / FRAME_MS))
MIN_UTTERANCE_BYTES = MIN_UTTERANCE_FRAMES * FRAME_BYTES
MIN_UTTERANCE_RMS = float(os.getenv("MIN_UTTERANCE_RMS", "0.001"))

VAD_MODE = int(os.getenv("VAD_MODE", "2"))  # 0-3 (3 = most aggressive)
MAX_UTTERANCE_SECONDS = float(os.getenv("MAX_UTTERANCE_SECONDS", "12.0"))
END_SILENCE_MS = int(os.getenv("END_SILENCE_MS", "2000"))
END_SILENCE_FRAMES = max(1, int(END_SILENCE_MS / FRAME_MS))
START_SPEECH_FRAMES = max(1, int(os.getenv("START_SPEECH_FRAMES", "3")))
VAD_RMS_THRESHOLD = float(os.getenv("VAD_RMS_THRESHOLD", "0.0025"))
NOISE_FLOOR_ALPHA = float(os.getenv("NOISE_FLOOR_ALPHA", "0.95"))
NOISE_FLOOR_MULT = float(os.getenv("NOISE_FLOOR_MULT", "2.5"))
ALLOW_BARGE_IN = os.getenv("ALLOW_BARGE_IN", "false").lower() in ("1", "true", "yes")
LISTEN_DURING_TTS = os.getenv("LISTEN_DURING_TTS", "false").lower() in ("1", "true", "yes")
MIN_WORDS = max(1, int(os.getenv("MIN_WORDS", "1")))
MIN_AVG_LOGPROB = float(os.getenv("MIN_AVG_LOGPROB", "-1.0"))
MAX_NO_SPEECH_PROB = float(os.getenv("MAX_NO_SPEECH_PROB", "0.6"))
MIN_WORDS_NON_EN = max(1, int(os.getenv("MIN_WORDS_NON_EN", "1")))
MIN_AVG_LOGPROB_NON_EN = float(os.getenv("MIN_AVG_LOGPROB_NON_EN", "-1.6"))
MAX_NO_SPEECH_PROB_NON_EN = float(os.getenv("MAX_NO_SPEECH_PROB_NON_EN", "0.9"))
TTS_CHARS_PER_SECOND = float(os.getenv("TTS_CHARS_PER_SECOND", "14.0"))
POST_TTS_GRACE_MS = int(os.getenv("POST_TTS_GRACE_MS", "600"))
TTS_MIN_SUPPRESS_MS = int(os.getenv("TTS_MIN_SUPPRESS_MS", "0"))
MAX_TTS_SUPPRESS_SEC = float(os.getenv("MAX_TTS_SUPPRESS_SEC", "30.0"))

# Language / multilingual
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")  # "en" or "auto"
LANG_LOCK_MIN_PROB = float(os.getenv("LANG_LOCK_MIN_PROB", "0.70"))
LANG_FALLBACK = os.getenv("LANG_FALLBACK", "en")
ALLOWED_LANGS = [l.strip().lower() for l in os.getenv("ALLOWED_LANGS", "").split(",") if l.strip()]

# IVR language selection (DTMF-first, optional)
IVR_LANG_SELECT_ENABLED = os.getenv("IVR_LANG_SELECT_ENABLED", "false").lower() in ("1", "true", "yes")
IVR_LANG_TIMEOUT_MS = int(os.getenv("IVR_LANG_TIMEOUT_MS", "10000"))
IVR_LANG_PROMPT = os.getenv(
    "IVR_LANG_PROMPT",
    "For English, press 1. សម្រាប់ភាសាខ្មែរ សូមចុចលេខ 2.",
)
IVR_PROMPT_DELAY_MS = int(os.getenv("IVR_PROMPT_DELAY_MS", "-1"))
IVR_LANG_WELCOME_EN = os.getenv("IVR_LANG_WELCOME_EN", "Welcome. How can I help you today?")
IVR_LANG_WELCOME_KM = os.getenv("IVR_LANG_WELCOME_KM", "សូមស្វាគមន៍។ តើខ្ញុំអាចជួយអ្វីបានដែរ?")
IVR_LANG_WELCOME_HI = os.getenv("IVR_LANG_WELCOME_HI", "नमस्ते। मैं आपकी कैसे मदद कर सकता हूँ?")
IVR_SPEECH_ENABLED = os.getenv("IVR_SPEECH_ENABLED", "false").lower() in ("1", "true", "yes")
IVR_SPEECH_MAX_WORDS = max(1, int(os.getenv("IVR_SPEECH_MAX_WORDS", "3")))
IVR_SPEECH_MAP_RAW = os.getenv("IVR_SPEECH_MAP", "en=english,eng|km=khmer,kh,ខ្មែរ")
DTMF_DEBUG_RAW = os.getenv("DTMF_DEBUG_RAW", "false").lower() in ("1", "true", "yes")

# Optional hangup intent
HANGUP_ENABLED = os.getenv("HANGUP_ENABLED", "true").lower() in ("1", "true", "yes")
HANGUP_PHRASES_RAW = os.getenv(
    "HANGUP_PHRASES",
    "goodbye,bye,bye bye,see you,see ya,that's all,that is all,hang up,end call,disconnect,stop calling",
)
HANGUP_STRONG_PHRASES_RAW = os.getenv(
    "HANGUP_STRONG_PHRASES",
    "hang up,end call,disconnect,terminate call",
)
HANGUP_PHRASES = [p.strip().lower() for p in HANGUP_PHRASES_RAW.split(",") if p.strip()]
HANGUP_STRONG_PHRASES = [p.strip().lower() for p in HANGUP_STRONG_PHRASES_RAW.split(",") if p.strip()]
HANGUP_MAX_WORDS = max(1, int(os.getenv("HANGUP_MAX_WORDS", "6")))
HANGUP_TEXT = os.getenv("HANGUP_TEXT", "Thanks for calling. Goodbye.").strip()
HANGUP_DELAY_MS = int(os.getenv("HANGUP_DELAY_MS", "200"))

def _normalize_company_terms(text: str) -> str:
    if not text or not COMPANY_ALIASES or not _company_name:
        return text
    out = text
    for alias in COMPANY_ALIASES:
        alias = alias.strip()
        if not alias:
            continue
        try:
            out = re.sub(rf"\b{re.escape(alias)}\b", _company_name, out, flags=re.IGNORECASE)
        except re.error:
            out = out.replace(alias, _company_name)
    return out


def _normalize_text(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")

def _parse_lang_voice_map(raw: str) -> dict:
    mapping: dict[str, str] = {}
    for part in (raw or "").split(","):
        if not part.strip():
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k and v:
            mapping[k] = v
    return mapping


LANG_VOICE_MAP = _parse_lang_voice_map(os.getenv("LANG_VOICE_MAP", ""))

LANG_NAME_MAP = {
    "en": "English",
    "km": "Khmer",
    "hi": "Hindi",
    "th": "Thai",
    "zh": "Chinese",
}


def _lang_name(code: str) -> str:
    return LANG_NAME_MAP.get(code.lower(), code)


def _voice_for_lang(code: str) -> str:
    if not code:
        return DEFAULT_TTS_VOICE
    key = code.lower()
    if key in LANG_VOICE_MAP:
        return LANG_VOICE_MAP[key]
    base = key.split("-", 1)[0]
    if base in LANG_VOICE_MAP:
        return LANG_VOICE_MAP[base]
    return DEFAULT_TTS_VOICE


def _parse_lang_map(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not raw:
        return mapping
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip().lower()
        if k and v:
            mapping[k] = v
    return mapping


def _parse_ivr_speech_map(raw: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if not raw:
        return mapping
    for part in raw.split("|"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        lang, phrases = part.split("=", 1)
        lang = lang.strip().lower()
        if not lang:
            continue
        vals = [p.strip().lower() for p in phrases.split(",") if p.strip()]
        if vals:
            mapping[lang] = vals
    return mapping


IVR_LANG_MAP = _parse_lang_map(os.getenv("IVR_LANG_MAP", "1=en,2=km"))
IVR_SPEECH_MAP = _parse_ivr_speech_map(IVR_SPEECH_MAP_RAW)


def _match_ivr_speech_choice(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    if not t:
        return ""
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) > IVR_SPEECH_MAX_WORDS:
        return ""
    for lang, phrases in IVR_SPEECH_MAP.items():
        for phrase in phrases:
            if phrase and phrase in t:
                return lang
    return ""


def _apply_language_choice(state: "SessionState", lang: str) -> None:
    if not lang:
        return
    state.language = lang
    state.language_prob = 1.0
    state.language_locked = True
    state.voice = _voice_for_lang(lang)
    state.lang_select_active = False
    state.lang_selected = True
    LOG.info("[call %s] IVR language selected: %s -> voice=%s", state.session_id, lang, state.voice)


async def _handle_language_switch(state: "SessionState", lang: str) -> None:
    if not lang or not state.ws:
        return
    # Stop any in-flight TTS/FS playback so the welcome is clear.
    # Bump generation early to cancel any stale TTS still in flight.
    state.tts_generation += 1
    await cancel_task(state.ivr_prompt_task, "ivr_prompt")
    state.ivr_prompt_task = None
    await cancel_task(state.speaking_task, "tts")
    state.speaking_task = None
    state.suppress_until = 0.0
    state.block_listen = False
    state.lang_select_active = False

    _apply_language_choice(state, lang)
    # Fire-and-forget break to stop any already-playing FS audio.
    async def _break_playback():
        try:
            await asyncio.wait_for(
                _esl_api_cmd(f"uuid_break {state.session_id}"),
                timeout=1.0,
            )
        except Exception:
            pass

    asyncio.create_task(_break_playback())
    welcome = IVR_LANG_WELCOME_EN
    lang_lc = lang.lower()
    if lang_lc.startswith("km"):
        welcome = IVR_LANG_WELCOME_KM
    elif lang_lc.startswith("hi"):
        welcome = IVR_LANG_WELCOME_HI
    try:
        await play_tts_text(state.ws, state, welcome)
    except Exception as e:
        LOG.warning("[call %s] IVR welcome failed: %s", state.session_id, e)

def _piper_paths_for_lang(code: str) -> tuple[str, str]:
    if not code:
        code = ""
    key = code.lower()
    base = key.split("-", 1)[0] if key else ""
    model = ""
    config = ""
    if key in PIPER_MODEL_MAP:
        model = PIPER_MODEL_MAP[key]
    elif base and base in PIPER_MODEL_MAP:
        model = PIPER_MODEL_MAP[base]
    else:
        model = PIPER_MODEL

    if key in PIPER_CONFIG_MAP:
        config = PIPER_CONFIG_MAP[key]
    elif base and base in PIPER_CONFIG_MAP:
        config = PIPER_CONFIG_MAP[base]
    else:
        config = PIPER_CONFIG

    if not config and model:
        guess = model + ".json"
        if os.path.exists(guess):
            config = guess
    return model, config

# TTS
SERVER_TTS_ENABLED = os.getenv("SERVER_TTS_ENABLED", "true").lower() in ("1", "true", "yes")
DEFAULT_TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "480"))
TTS_AUDIO_TYPE = os.getenv("TTS_AUDIO_TYPE", "wav").lower()
EDGE_OUTPUT_FORMAT = os.getenv("EDGE_OUTPUT_FORMAT", "riff-16khz-16bit-mono-pcm")
TTS_WAV_SAMPLE_RATE = int(os.getenv("TTS_WAV_SAMPLE_RATE", "8000"))
TTS_RAW_SAMPLE_RATE = int(os.getenv("TTS_RAW_SAMPLE_RATE", "8000"))
USE_FS_TTS = os.getenv("USE_FS_TTS", "false").lower() in ("1", "true", "yes")
FS_TTS_ENGINE = os.getenv("FS_TTS_ENGINE", "tts_commandline")
FS_TTS_VOICE = os.getenv("FS_TTS_VOICE", DEFAULT_TTS_VOICE)
FFMPEG_PATH = shutil.which("ffmpeg")
TTS_BACKEND = os.getenv("TTS_BACKEND", "edge").lower()  # edge | piper
TTS_STREAMING_ENABLED = os.getenv("TTS_STREAMING_ENABLED", "false").lower() in ("1", "true", "yes")
TTS_STREAM_MIN_BYTES = max(2048, int(os.getenv("TTS_STREAM_MIN_BYTES", "12000")))
TTS_STREAM_MAX_DELAY_MS = max(50, int(os.getenv("TTS_STREAM_MAX_DELAY_MS", "200")))
PIPER_BIN = os.getenv("PIPER_BIN", "piper")
PIPER_MODEL = os.getenv("PIPER_MODEL", "")
PIPER_CONFIG = os.getenv("PIPER_CONFIG", "")
PIPER_MODEL_MAP = _parse_lang_voice_map(os.getenv("PIPER_MODEL_MAP", ""))
PIPER_CONFIG_MAP = _parse_lang_voice_map(os.getenv("PIPER_CONFIG_MAP", ""))
PIPER_SPEAKER = os.getenv("PIPER_SPEAKER", "")
PIPER_ESPEAK_DATA = os.getenv("PIPER_ESPEAK_DATA", "/usr/local/share/piper/espeak-ng-data")
PIPER_FALLBACK_EDGE = os.getenv("PIPER_FALLBACK_EDGE", "true").lower() in ("1", "true", "yes")

# Whisper
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "small")  # default model (path or name)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu").strip().lower()
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "").strip()
if not WHISPER_COMPUTE_TYPE:
    WHISPER_COMPUTE_TYPE = "int8" if WHISPER_DEVICE == "cpu" else "float16"
WHISPER_CPU_THREADS = _env_int("WHISPER_CPU_THREADS")
WHISPER_NUM_WORKERS = _env_int("WHISPER_NUM_WORKERS")
KHMER_WHISPER_PATH = os.getenv("KHMER_WHISPER_PATH", "")
USE_DUAL_WHISPER = bool(KHMER_WHISPER_PATH)

# ESL (auto-play streamAudio responses)
ESL_ENABLED = os.getenv("FS_ESL_ENABLED", "true").lower() in ("1", "true", "yes")
ESL_HOST = os.getenv("FS_ESL_HOST", "127.0.0.1")
ESL_PORT = int(os.getenv("FS_ESL_PORT", "8021"))
ESL_PASSWORD = os.getenv("FS_ESL_PASSWORD", "ClueCon")
ESL_API_CONCURRENCY = max(1, int(os.getenv("FS_ESL_API_CONCURRENCY", "2")))
ESL_API_SEMAPHORE = asyncio.Semaphore(ESL_API_CONCURRENCY)
ESL_TASK: Optional[asyncio.Task] = None

LOG = logging.getLogger("fs_engine")
if not logging.getLogger().handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
LOG.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

LOG_COLOR = os.getenv("LOG_COLOR", "true").lower() in ("1", "true", "yes")

def _colorize(text: str, color_code: str) -> str:
    if not LOG_COLOR:
        return text
    return f"\033[{color_code}m{text}\033[0m"

print("[boot] Loading Whisper model...")
_whisper_kwargs = {
    "device": WHISPER_DEVICE,
    "compute_type": WHISPER_COMPUTE_TYPE,
}
if WHISPER_CPU_THREADS:
    _whisper_kwargs["cpu_threads"] = WHISPER_CPU_THREADS
if WHISPER_NUM_WORKERS:
    _whisper_kwargs["num_workers"] = WHISPER_NUM_WORKERS
print(
    f"[boot] Whisper config: model={WHISPER_SIZE} device={WHISPER_DEVICE} "
    f"compute={WHISPER_COMPUTE_TYPE} "
    f"threads={WHISPER_CPU_THREADS or 'auto'} workers={WHISPER_NUM_WORKERS or 'auto'}"
)
whisper_default = WhisperModel(WHISPER_SIZE, **_whisper_kwargs)
whisper_kh = None
if USE_DUAL_WHISPER:
    try:
        whisper_kh = WhisperModel(KHMER_WHISPER_PATH, **_whisper_kwargs)
        print("[boot] Khmer Whisper loaded.")
    except Exception as e:
        whisper_kh = None
        print(f"[boot] Khmer Whisper failed to load: {e}")
print("[boot] Whisper loaded.")

app = FastAPI()
ACTIVE_SESSIONS: dict[str, "SessionState"] = {}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "esl": {
            "enabled": ESL_ENABLED,
            "host": ESL_HOST,
            "port": ESL_PORT,
            "task_running": bool(ESL_TASK and not ESL_TASK.done()),
        },
        "whisper": {
            "model": WHISPER_SIZE,
            "device": WHISPER_DEVICE,
            "compute_type": WHISPER_COMPUTE_TYPE,
            "cpu_threads": WHISPER_CPU_THREADS,
            "num_workers": WHISPER_NUM_WORKERS,
            "dual_model": USE_DUAL_WHISPER,
        },
        "ollama": {
            "model": MODEL,
            "base_url": OLLAMA_BASE_URL,
            "mode": OLLAMA_MODE,
            "stream": LLM_STREAM,
            "options": _ollama_options(),
        },
        "tts": {
            "enabled": SERVER_TTS_ENABLED,
            "backend": TTS_BACKEND,
            "audio_type": TTS_AUDIO_TYPE,
            "streaming": TTS_STREAMING_ENABLED,
            "use_fs_tts": USE_FS_TTS,
            "voice": DEFAULT_TTS_VOICE,
        },
    }


@dataclass
class SessionState:
    session_id: str
    speaking_task: Optional[asyncio.Task] = None
    ivr_prompt_task: Optional[asyncio.Task] = None
    llm_task: Optional[asyncio.Task] = None
    worker_task: Optional[asyncio.Task] = None
    interrupted: bool = False
    voice: str = DEFAULT_TTS_VOICE
    history: list[dict] = field(default_factory=list)
    turn_id: int = 0
    suppress_until: float = 0.0
    block_listen: bool = False
    mic_muted: bool = False
    mic_muted_reason: str = ""
    mic_unmute_task: Optional[asyncio.Task] = None
    language: str = ""
    language_prob: float = 0.0
    language_locked: bool = False
    offtopic_count: int = 0
    lang_select_active: bool = False
    lang_select_deadline: float = 0.0
    lang_selected: bool = False
    lang_switch_task: Optional[asyncio.Task] = None
    tts_generation: int = 0
    utter_q: Optional[asyncio.Queue] = None
    ws: Optional[WebSocket] = None
    caller_id: str = ""
    call_started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    transcript: list[dict] = field(default_factory=list)


@dataclass
class UtteranceJob:
    pcm16: Optional[bytes] = None


def pcm16_bytes_to_float32(pcm: bytes) -> np.ndarray:
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    a /= 32768.0
    return a


def pcm16_rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    a /= 32768.0
    return float(np.sqrt(np.mean(a * a)))



def _clip(text: str, limit: int = 200) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _set_mic_state(state: SessionState, muted: bool, reason: str = "") -> None:
    if muted:
        if not state.mic_muted or state.mic_muted_reason != reason:
            state.mic_muted = True
            state.mic_muted_reason = reason
            msg = f"[call {state.session_id}] ===== MIC OFF ({reason}) ====="
            LOG.warning(_colorize(msg, "1;31"))  # bold red
    else:
        if state.mic_muted:
            state.mic_muted = False
            state.mic_muted_reason = ""
            msg = f"[call {state.session_id}] ===== MIC ON ====="
            LOG.warning(_colorize(msg, "1;32"))  # bold green


def _schedule_mic_unmute(state: SessionState) -> None:
    if LISTEN_DURING_TTS or ALLOW_BARGE_IN:
        return
    if state.mic_unmute_task and not state.mic_unmute_task.done():
        state.mic_unmute_task.cancel()
    delay = max(0.0, state.suppress_until - time.time())
    if delay:
        msg = f"[call {state.session_id}] ===== MIC will unmute in {delay:.2f}s ====="
        LOG.warning(_colorize(msg, "1;33"))  # bold yellow

    async def _unmute_later():
        try:
            if delay:
                await asyncio.sleep(delay)
            if not state.block_listen and time.time() >= state.suppress_until:
                state.suppress_until = 0.0
                _set_mic_state(state, False, "")
        except asyncio.CancelledError:
            return
    state.mic_unmute_task = asyncio.create_task(_unmute_later())


def _estimate_tts_duration_seconds(text: str, audio_type: str, audio_bytes: bytes, sample_rate: int) -> float:
    # Conservative estimate to block mic pickup while TTS plays.
    try:
        if audio_type == "raw":
            dur = len(audio_bytes) / float(sample_rate * SAMPLE_WIDTH) if sample_rate else 0.0
            return max(0.3, min(MAX_TTS_SUPPRESS_SEC, dur))
        if audio_type == "wav":
            import wave
            import io
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate() or sample_rate
                channels = wf.getnchannels() or 1
                dur = frames / float(rate) if rate else 0.0
                # Fallback to byte-length estimate when headers are invalid/streaming
                if audio_bytes:
                    pcm_bytes = max(0, len(audio_bytes) - 44)
                    bps = rate * channels * SAMPLE_WIDTH if rate else 0.0
                    dur_from_bytes = (pcm_bytes / bps) if bps else dur
                    if dur <= 0.0 or dur > MAX_TTS_SUPPRESS_SEC * 4 or (dur_from_bytes > 0 and dur > dur_from_bytes * 4):
                        dur = dur_from_bytes
                return max(0.3, min(MAX_TTS_SUPPRESS_SEC, dur))
    except Exception:
        pass
    # Fallback: estimate by text length.
    if not text:
        return 0.5
    return max(0.5, min(MAX_TTS_SUPPRESS_SEC, len(text) / max(8.0, TTS_CHARS_PER_SECOND)))


def stt_whisper_from_pcm16(
    pcm: bytes, lang_hint: str = "", model: Optional[WhisperModel] = None
) -> tuple[str, float, float, str, float]:
    audio = pcm16_bytes_to_float32(pcm)
    stt_model = model or whisper_default
    lang_arg = None
    if lang_hint:
        lang_arg = lang_hint.lower()
    elif STT_LANGUAGE and STT_LANGUAGE.lower() not in ("auto", "none", ""):
        lang_arg = STT_LANGUAGE.lower()
    segments, info = stt_model.transcribe(audio, language=lang_arg)
    texts = []
    avg_logprobs = []
    no_speech_probs = []
    detected_lang = ""
    lang_prob = 0.0
    try:
        if info is not None:
            detected_lang = getattr(info, "language", "") or ""
            lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
    except Exception:
        detected_lang = ""
        lang_prob = 0.0
    if lang_arg:
        detected_lang = lang_arg
        if lang_prob <= 0.0:
            lang_prob = 1.0
    for seg in segments:
        texts.append(seg.text)
        if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
            avg_logprobs.append(float(seg.avg_logprob))
        if hasattr(seg, "no_speech_prob") and seg.no_speech_prob is not None:
            no_speech_probs.append(float(seg.no_speech_prob))
    text = "".join(texts).strip()
    if not text:
        return "", -99.0, 1.0, detected_lang, lang_prob
    avg_logprob = sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else -99.0
    no_speech_prob = max(no_speech_probs) if no_speech_probs else 0.0
    return text, avg_logprob, no_speech_prob, detected_lang, lang_prob


HISTORY_MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "8"))

def _trim_history(history: list[dict], max_messages: int = 16) -> None:
    if len(history) > max_messages:
        del history[:-max_messages]


def _maybe_offtopic_reply(state: SessionState, prompt: str) -> Optional[str]:
    if ALLOWED_TOPICS and not _matches_allowed_topics(prompt, ALLOWED_TOPICS):
        state.offtopic_count += 1
        if state.offtopic_count >= OFFTOPIC_REPEAT_LIMIT:
            return OFFTOPIC_RESPONSE_SHORT
        return OFFTOPIC_RESPONSE
    state.offtopic_count = 0
    return None


def _build_llm_messages(state: SessionState, prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if COMPANY_PROFILE_MODE != "off":
        facts = ""
        summary = COMPANY_FACTS.get("summary", "") if isinstance(COMPANY_FACTS, dict) else ""
        topics = COMPANY_FACTS.get("topics", {}) if isinstance(COMPANY_FACTS, dict) else {}
        if COMPANY_PROFILE_MODE == "summary":
            facts = _compact(summary, COMPANY_FACTS_MAX_CHARS)
        else:
            wanted = _detect_topics(prompt)
            chunks: list[str] = []
            if summary:
                chunks.append(_compact(summary, COMPANY_FACTS_MAX_CHARS))
            for topic in wanted:
                if topic in topics:
                    chunks.append(_compact(topics[topic], COMPANY_FACTS_TOPIC_MAX_CHARS))
            facts = "\n".join([c for c in chunks if c]).strip()
        if facts:
            messages.append({"role": "system", "content": "Company facts:\n" + facts})
    reply_lang = state.language if (state.language_locked and state.language) else (DEFAULT_LANGUAGE or "")
    if reply_lang:
        messages.append(
            {
                "role": "system",
                "content": f"Language: Reply in {_lang_name(reply_lang)} ({reply_lang}).",
            }
        )
    if state.history:
        messages.extend(state.history)
    messages.append({"role": "user", "content": prompt})
    return messages


async def ollama_reply(state: SessionState, prompt: str) -> str:
    off = _maybe_offtopic_reply(state, prompt)
    if off:
        return off
    messages = _build_llm_messages(state, prompt)

    async def _native_chat() -> str:
        if not OLLAMA_NATIVE_BASE:
            return ""
        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": _ollama_options(),
        }
        url = f"{OLLAMA_NATIVE_BASE}/api/chat"
        http_timeout = None if LLM_DISABLE_TIMEOUT else LLM_TIMEOUT_SEC
        async with httpx.AsyncClient(timeout=http_timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        msg = (data.get("message") or {}).get("content") or ""
        return msg.strip()

    async def _native_generate(
        prompt_override: Optional[str] = None,
        system_override: Optional[str] = None,
    ) -> str:
        if not OLLAMA_NATIVE_BASE:
            return ""
        payload = {
            "model": MODEL,
            "prompt": prompt_override if prompt_override is not None else _render_prompt(messages),
            "stream": False,
            "options": _ollama_options(),
        }
        stops = _parse_stop_list(OLLAMA_STOP_RAW)
        if stops and not LLM_STREAM:
            payload["options"]["stop"] = stops
        if system_override:
            payload["system"] = system_override
        url = f"{OLLAMA_NATIVE_BASE}/api/generate"
        http_timeout = None if LLM_DISABLE_TIMEOUT else LLM_TIMEOUT_SEC
        async with httpx.AsyncClient(timeout=http_timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        msg, used_thinking = _extract_generate_text(data)
        if not msg:
            LOG.warning(
                "[call %s] LLM generate empty (keys=%s)",
                state.session_id,
                ",".join(sorted(data.keys())),
            )
        elif used_thinking:
            LOG.info("[call %s] LLM generate used thinking fallback", state.session_id)
        return msg.strip()

    cleaned = ""
    system_text = _join_system_messages(messages)
    async with LLM_SEMAPHORE:
        try:
            if OLLAMA_MODE == "generate":
                dialog_prompt = _render_prompt(messages, include_system=False)
                cleaned = await _native_generate(
                    prompt_override=dialog_prompt,
                    system_override=system_text,
                )
                if not cleaned and LLM_RETRY_ON_EMPTY:
                    cleaned = await _native_generate(
                        prompt_override=prompt,
                        system_override=system_text,
                    )
                if not cleaned and LLM_RETRY_ON_EMPTY:
                    # last resort: no system prompt
                    cleaned = await _native_generate(prompt_override=prompt, system_override=None)
            else:
                if LLM_DISABLE_TIMEOUT:
                    resp = await llm_client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        max_tokens=LLM_MAX_TOKENS,
                        temperature=LLM_TEMPERATURE,
                        extra_body={
                            "options": _ollama_options(),
                        },
                    )
                else:
                    resp = await asyncio.wait_for(
                        llm_client.chat.completions.create(
                            model=MODEL,
                            messages=messages,
                            max_tokens=LLM_MAX_TOKENS,
                            temperature=LLM_TEMPERATURE,
                            extra_body={
                                "options": _ollama_options(),
                            },
                        ),
                        timeout=LLM_TIMEOUT_SEC,
                    )
                msg = resp.choices[0].message.content or ""
                cleaned = msg.strip()
        except asyncio.TimeoutError:
            LOG.warning(
                "[call %s] LLM timeout after %.1fs",
                state.session_id,
                LLM_TIMEOUT_SEC,
            )
        except Exception:
            cleaned = ""

        if not cleaned and OLLAMA_USE_NATIVE_FALLBACK and OLLAMA_MODE != "generate":
            try:
                cleaned = await _native_chat()
                if not cleaned:
                    cleaned = await _native_generate()
                if cleaned:
                    LOG.info("[call %s] LLM fallback via /api/chat succeeded", state.session_id)
            except Exception:
                cleaned = ""

        if not cleaned and not OLLAMA_USE_NATIVE_FALLBACK:
            LOG.warning("[call %s] LLM empty reply (no fallback)", state.session_id)

        if not cleaned:
            return LLM_EMPTY_REPLY
    if prompt:
        state.history.append({"role": "user", "content": prompt})
    if cleaned:
        state.history.append({"role": "assistant", "content": cleaned})
    else:
        LOG.warning("[call %s] LLM empty reply", state.session_id)
        cleaned = LLM_EMPTY_REPLY
    _trim_history(state.history, HISTORY_MAX_MESSAGES)
    return cleaned


async def ollama_stream_and_tts(
    ws: WebSocket,
    state: SessionState,
    prompt: str,
    turn_id: int,
) -> str:
    off = _maybe_offtopic_reply(state, prompt)
    if off:
        await play_tts_text(ws, state, off)
        return off

    messages = _build_llm_messages(state, prompt)
    system_text = _join_system_messages(messages)
    dialog_prompt = _render_prompt(messages, include_system=False)
    payload = {
        "model": MODEL,
        "prompt": dialog_prompt,
        "stream": True,
        "options": _ollama_options(),
    }
    if system_text:
        payload["system"] = system_text
    url = f"{OLLAMA_NATIVE_BASE}/api/generate"
    http_timeout = None if LLM_DISABLE_TIMEOUT else LLM_TIMEOUT_SEC

    full_reply = ""
    buffer = ""
    async with httpx.AsyncClient(timeout=http_timeout) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if data.get("error"):
                    raise RuntimeError(str(data.get("error")))
                chunk = data.get("response") or ""
                if not chunk:
                    if data.get("done"):
                        break
                    continue
                buffer += chunk
                chunks, buffer = _split_tts_chunks(buffer, TTS_STREAM_MIN_CHARS)
                for sentence in chunks:
                    if state.turn_id != turn_id:
                        return full_reply.strip()
                    full_reply = (full_reply + " " + sentence).strip()
                    await play_tts_text(ws, state, sentence)
                if data.get("done"):
                    break

    if buffer.strip() and state.turn_id == turn_id:
        tail = buffer.strip()
        full_reply = (full_reply + " " + tail).strip()
        await play_tts_text(ws, state, tail)
    return full_reply.strip()


async def synthesize_tts_mp3(text: str, voice: str) -> bytes:
    tts = edge_tts.Communicate(text=text, voice=voice)
    chunks = []
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


async def stream_tts_mp3(ws: WebSocket, state: SessionState, text: str, voice: str, tts_gen: int) -> int:
    tts = edge_tts.Communicate(text=text, voice=voice)
    buffer = bytearray()
    total = 0
    last_send = time.perf_counter()
    try:
        async for chunk in tts.stream():
            if tts_gen != state.tts_generation:
                return total
            if chunk["type"] != "audio":
                continue
            data = chunk["data"]
            if not data:
                continue
            total += len(data)
            buffer.extend(data)
            now = time.perf_counter()
            if len(buffer) >= TTS_STREAM_MIN_BYTES or (
                buffer and ((now - last_send) * 1000.0) >= TTS_STREAM_MAX_DELAY_MS
            ):
                await send_stream_audio(ws, bytes(buffer), audio_type="mp3")
                buffer.clear()
                last_send = now
        if buffer and tts_gen == state.tts_generation:
            await send_stream_audio(ws, bytes(buffer), audio_type="mp3")
            buffer.clear()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        LOG.warning("[call %s] TTS stream failed: %s", state.session_id, e)
    return total


async def synthesize_tts_piper(text: str, lang: str) -> bytes:
    model, config = _piper_paths_for_lang(lang)
    if not model:
        raise RuntimeError("Piper model not configured")
    tmp_wav = None
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(prefix="piper_", suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        cmd = [PIPER_BIN, "--model", model, "--output_file", tmp_wav]
        if config:
            cmd += ["--config", config]
        if PIPER_SPEAKER:
            cmd += ["--speaker", str(PIPER_SPEAKER)]
        if PIPER_ESPEAK_DATA:
            cmd += ["--espeak_data", PIPER_ESPEAK_DATA]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate(input=(text + "\n").encode("utf-8"))
        if proc.returncode != 0:
            raise RuntimeError(f"piper failed: {err.decode('utf-8', errors='ignore')}")
        with open(tmp_wav, "rb") as f:
            return f.read()
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.unlink(tmp_wav)
            except Exception:
                pass


def _wav_rate_from_format(output_format: str) -> int:
    fmt = (output_format or "").lower()
    if "8khz" in fmt:
        return 8000
    if "24khz" in fmt:
        return 24000
    if "48khz" in fmt:
        return 48000
    return 16000


async def _mp3_to_wav(mp3_bytes: bytes, sample_rate: int) -> bytes:
    if not FFMPEG_PATH:
        raise RuntimeError("ffmpeg not found on PATH")
    proc = await asyncio.create_subprocess_exec(
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(input=mp3_bytes), timeout=20)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError("ffmpeg convert timed out")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {stderr.decode('utf-8', 'ignore').strip()}")
    return stdout


async def _mp3_to_raw(mp3_bytes: bytes, sample_rate: int) -> bytes:
    if not FFMPEG_PATH:
        raise RuntimeError("ffmpeg not found on PATH")
    proc = await asyncio.create_subprocess_exec(
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(input=mp3_bytes), timeout=20)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError("ffmpeg convert timed out")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {stderr.decode('utf-8', 'ignore').strip()}")
    return stdout


async def synthesize_tts_wav(text: str, voice: str, output_format: str) -> bytes:
    mp3_audio = await synthesize_tts_mp3(text, voice)
    if not mp3_audio:
        return b""
    sample_rate = TTS_WAV_SAMPLE_RATE if TTS_WAV_SAMPLE_RATE > 0 else _wav_rate_from_format(output_format)
    return await _mp3_to_wav(mp3_audio, sample_rate)


async def synthesize_tts_raw(text: str, voice: str, sample_rate: int) -> bytes:
    mp3_audio = await synthesize_tts_mp3(text, voice)
    if not mp3_audio:
        return b""
    rate = sample_rate if sample_rate > 0 else 8000
    return await _mp3_to_raw(mp3_audio, rate)


async def send_stream_audio(ws: WebSocket, audio_bytes: bytes, audio_type: str = "mp3", sample_rate: int = SAMPLE_RATE):
    payload = {
        "type": "streamAudio",
        "data": {
            "audioDataType": audio_type,
            "audioData": base64.b64encode(audio_bytes).decode("ascii"),
        },
    }
    if audio_type == "raw":
        payload["data"]["sampleRate"] = int(sample_rate)
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


async def cancel_task(task: Optional[asyncio.Task], label: str):
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            LOG.warning("[warn] %s cancel exception: %s", label, e)


async def play_tts_text(ws: WebSocket, state: SessionState, text: str):
    if not text:
        return
    tts_gen = state.tts_generation
    reply = text
    if MAX_TTS_CHARS > 0:
        reply = reply[:MAX_TTS_CHARS]
    if SERVER_TTS_ENABLED and USE_FS_TTS:
        safe_text = reply.replace("\n", " ").replace("\r", " ").replace('"', "").replace("|", " ").strip()
        if not safe_text:
            return
        if tts_gen != state.tts_generation:
            LOG.info("[call %s] TTS skipped (stale generation)", state.session_id)
            return
        if not LISTEN_DURING_TTS and not ALLOW_BARGE_IN:
            est = _estimate_tts_duration_seconds(safe_text, "mp3", b"", SAMPLE_RATE)
            if TTS_MIN_SUPPRESS_MS > 0:
                est = max(est, TTS_MIN_SUPPRESS_MS / 1000.0)
            state.suppress_until = time.time() + est + (POST_TTS_GRACE_MS / 1000.0)
            _schedule_mic_unmute(state)
        speak_arg = f"speak::{FS_TTS_ENGINE}|{FS_TTS_VOICE}|{safe_text}"
        reply_text = await _esl_api_cmd(f"uuid_broadcast {state.session_id} {speak_arg} aleg")
        if reply_text and not reply_text.startswith("+OK"):
            LOG.warning("[call %s] uuid_broadcast speak failed: %s", state.session_id, reply_text)
        return

    if not SERVER_TTS_ENABLED:
        await ws.send_text(json.dumps({"type": "text", "text": reply}, ensure_ascii=False))
        return

    LOG.info("[call %s] TTS start (voice=%s, chars=%d)", state.session_id, state.voice, len(reply))
    if TTS_STREAMING_ENABLED and TTS_BACKEND == "edge" and TTS_AUDIO_TYPE == "mp3":
        if tts_gen != state.tts_generation:
            LOG.info("[call %s] TTS skipped (stale generation)", state.session_id)
            return
        if not LISTEN_DURING_TTS and not ALLOW_BARGE_IN:
            est = _estimate_tts_duration_seconds(reply, "mp3", b"", SAMPLE_RATE)
            if TTS_MIN_SUPPRESS_MS > 0:
                est = max(est, TTS_MIN_SUPPRESS_MS / 1000.0)
            state.suppress_until = time.time() + est + (POST_TTS_GRACE_MS / 1000.0)
            _schedule_mic_unmute(state)
        async def _stream():
            return await stream_tts_mp3(ws, state, reply, state.voice, tts_gen)
        state.speaking_task = asyncio.create_task(_stream())
        total = await state.speaking_task
        state.speaking_task = None
        if tts_gen != state.tts_generation:
            LOG.info("[call %s] TTS stream skipped (stale generation)", state.session_id)
            return
        LOG.info("[call %s] TTS stream done (bytes=%d) -> streamAudio(mp3)", state.session_id, total)
        return
    if TTS_BACKEND == "piper":
        try:
            audio = await synthesize_tts_piper(reply, state.language or LANG_FALLBACK)
            tts_type = "wav"
        except Exception as e:
            LOG.warning("[call %s] Piper TTS failed: %s", state.session_id, e)
            if not PIPER_FALLBACK_EDGE:
                raise
            if TTS_AUDIO_TYPE == "mp3":
                audio = await synthesize_tts_mp3(reply, state.voice)
                tts_type = "mp3"
            elif TTS_AUDIO_TYPE == "raw":
                audio = await synthesize_tts_raw(reply, state.voice, TTS_RAW_SAMPLE_RATE)
                tts_type = "raw"
            else:
                audio = await synthesize_tts_wav(reply, state.voice, EDGE_OUTPUT_FORMAT)
                tts_type = "wav"
    else:
        if TTS_AUDIO_TYPE == "mp3":
            audio = await synthesize_tts_mp3(reply, state.voice)
            tts_type = "mp3"
        elif TTS_AUDIO_TYPE == "raw":
            audio = await synthesize_tts_raw(reply, state.voice, TTS_RAW_SAMPLE_RATE)
            tts_type = "raw"
        else:
            audio = await synthesize_tts_wav(reply, state.voice, EDGE_OUTPUT_FORMAT)
            tts_type = "wav"

    if not audio:
        LOG.warning("[call %s] TTS produced no audio", state.session_id)
        return
    if tts_gen != state.tts_generation:
        LOG.info("[call %s] TTS skipped after synth (stale generation)", state.session_id)
        return
    LOG.info("[call %s] TTS done (bytes=%d) -> streamAudio(%s)", state.session_id, len(audio), tts_type)
    sample_rate = TTS_RAW_SAMPLE_RATE if tts_type == "raw" else SAMPLE_RATE
    if not LISTEN_DURING_TTS and not ALLOW_BARGE_IN:
        est = _estimate_tts_duration_seconds(reply, tts_type, audio, sample_rate)
        if TTS_MIN_SUPPRESS_MS > 0:
            est = max(est, TTS_MIN_SUPPRESS_MS / 1000.0)
        state.suppress_until = time.time() + est + (POST_TTS_GRACE_MS / 1000.0)
        _schedule_mic_unmute(state)
    state.speaking_task = asyncio.create_task(
        send_stream_audio(ws, audio, audio_type=tts_type, sample_rate=sample_rate)
    )
    await state.speaking_task
    state.speaking_task = None

async def handle_barge_in(state: SessionState):
    # Avoid uuid_break here; it can break the dialplan sleep/park and hang up the call.
    # Only stop in-progress TTS playback; do not cancel LLM.
    state.interrupted = True
    await cancel_task(state.speaking_task, "tts")
    state.speaking_task = None


async def run_llm_and_tts(
    ws: WebSocket,
    state: SessionState,
    user_text: str,
    turn_id: int,
    turn_start: Optional[float] = None,
):
    try:
        LOG.info("[call %s] LLM request: %s", state.session_id, _clip(user_text, 160))
        t2 = time.perf_counter()
        if LLM_STREAM and OLLAMA_MODE == "generate":
            if LISTEN_DURING_TTS:
                state.block_listen = False
            reply = await ollama_stream_and_tts(ws, state, user_text, turn_id)
        else:
            reply = await ollama_reply(state, user_text)
        t3 = time.perf_counter()
        LOG.info("[call %s] LLM %.0f ms", state.session_id, (t3 - t2) * 1000)
        if not reply:
            LOG.info("[call %s] LLM empty reply", state.session_id)
            return
        # If a newer utterance started, drop this reply.
        if state.turn_id != turn_id:
            LOG.info("[call %s] LLM reply dropped (superseded)", state.session_id)
            return
        LOG.info("[call %s] LLM reply: %s", state.session_id, _clip(reply, 200))
        if not (LLM_STREAM and OLLAMA_MODE == "generate"):
            if LISTEN_DURING_TTS:
                state.block_listen = False
            t4 = time.perf_counter()
            await play_tts_text(ws, state, reply)
            t5 = time.perf_counter()
        else:
            # streaming already played TTS
            t4 = t5 = time.perf_counter()
        if turn_start is not None:
            LOG.info(
                "[call %s] TTS %.0f ms | TURN %.0f ms",
                state.session_id,
                (t5 - t4) * 1000,
                (t5 - turn_start) * 1000,
            )
        else:
            LOG.info("[call %s] TTS %.0f ms", state.session_id, (t5 - t4) * 1000)
        # Update conversation history for streaming path
        if LLM_STREAM and OLLAMA_MODE == "generate":
            if user_text:
                state.history.append({"role": "user", "content": user_text})
            if reply:
                state.history.append({"role": "assistant", "content": reply})
            _trim_history(state.history, HISTORY_MAX_MESSAGES)
            if reply:
                state.transcript.append({"role": "assistant", "text": reply})
    except asyncio.CancelledError:
        raise
    except Exception as e:
        LOG.error("LLM/TTS error: %s", e)
    finally:
        # Re-enable listening after LLM/TTS completes.
        state.block_listen = False


async def process_utterance(ws: WebSocket, state: SessionState, utterance_pcm16: bytes):
    # Do not uuid_break here; it can break the dialplan sleep/park and hang up the call.
    state.interrupted = False
    state.turn_id += 1
    turn_id = state.turn_id
    t_turn0 = time.perf_counter()
    # Stop listening while we process this utterance to avoid splitting + queue buildup.
    state.block_listen = True
    try:
        if len(utterance_pcm16) < MIN_UTTERANCE_BYTES:
            LOG.info("[call %s] Utterance too short (%d bytes), skipped", state.session_id, len(utterance_pcm16))
            return

        if MIN_UTTERANCE_RMS > 0:
            rms = pcm16_rms(utterance_pcm16)
            if rms < MIN_UTTERANCE_RMS:
                LOG.info("[call %s] Utterance too quiet (rms=%.4f), skipped", state.session_id, rms)
                return

        try:
            loop = asyncio.get_running_loop()
            lang_hint = state.language if state.language_locked and state.language else ""
            model = whisper_kh if (state.language_locked and state.language == "km" and whisper_kh) else whisper_default
            utt_seconds = len(utterance_pcm16) / (SAMPLE_RATE * SAMPLE_WIDTH)
            LOG.info("[call %s] Utterance %.2fs", state.session_id, utt_seconds)
            t0 = time.perf_counter()
            async with STT_SEMAPHORE:
                text, avg_logprob, no_speech_prob, detected_lang, lang_prob = await loop.run_in_executor(
                    None, stt_whisper_from_pcm16, utterance_pcm16, lang_hint, model
                )
            t1 = time.perf_counter()
            LOG.info("[call %s] STT %.0f ms", state.session_id, (t1 - t0) * 1000)
        except Exception as e:
            LOG.error("STT error: %s", e)
            return

        if detected_lang:
            if ALLOWED_LANGS and detected_lang.lower() not in ALLOWED_LANGS:
                LOG.info(
                    "[call %s] Language %s not allowed; forcing %s",
                    state.session_id,
                    detected_lang,
                    LANG_FALLBACK,
                )
                detected_lang = LANG_FALLBACK
                lang_prob = 1.0
            if not state.language_locked:
                if STT_LANGUAGE and STT_LANGUAGE.lower() not in ("auto", "none", ""):
                    state.language = detected_lang
                    state.language_prob = lang_prob
                    state.language_locked = True
                elif lang_prob >= LANG_LOCK_MIN_PROB:
                    state.language = detected_lang
                    state.language_prob = lang_prob
                    state.language_locked = True
                    if state.language == "km" and whisper_kh:
                        LOG.info(
                            "[call %s] Switching to Khmer model for STT",
                            state.session_id,
                        )
                else:
                    LOG.info(
                        "[call %s] Language detect low confidence: %s (p=%.2f) -> ignore",
                        state.session_id,
                        detected_lang,
                        lang_prob,
                    )
            if state.language_locked and state.language:
                new_voice = _voice_for_lang(state.language)
                if new_voice != state.voice:
                    state.voice = new_voice
                LOG.info(
                    "[call %s] Language locked: %s (p=%.2f) -> voice=%s",
                    state.session_id,
                    state.language,
                    state.language_prob,
                    state.voice,
                )

        if not text.strip():
            LOG.info("[call %s] STT empty transcript", state.session_id)
            return

        # IVR speech language switching (always allowed; does not block normal STT)
        if IVR_LANG_SELECT_ENABLED and IVR_SPEECH_ENABLED:
            lang_choice = _match_ivr_speech_choice(text)
            if lang_choice:
                LOG.info("[call %s] Speech IVR detected language: %s", state.session_id, lang_choice)
                await _handle_language_switch(state, lang_choice)
                return

        lang_for_thresholds = (state.language or detected_lang or "").lower()
        use_non_en = bool(lang_for_thresholds) and not lang_for_thresholds.startswith("en")
        min_words = MIN_WORDS_NON_EN if use_non_en else MIN_WORDS
        min_avg_logprob = MIN_AVG_LOGPROB_NON_EN if use_non_en else MIN_AVG_LOGPROB
        max_no_speech_prob = MAX_NO_SPEECH_PROB_NON_EN if use_non_en else MAX_NO_SPEECH_PROB

        if no_speech_prob > max_no_speech_prob:
            LOG.info(
                "[call %s] STT skipped (no_speech_prob=%.2f > %.2f)",
                state.session_id,
                no_speech_prob,
                max_no_speech_prob,
            )
            return

        if avg_logprob < min_avg_logprob:
            LOG.info(
                "[call %s] STT skipped (avg_logprob=%.2f < %.2f)",
                state.session_id,
                avg_logprob,
                min_avg_logprob,
            )
            return

        word_count = len([w for w in text.strip().split() if w])
        if word_count < min_words:
            LOG.info(
                "[call %s] STT skipped (words=%d < %d)",
                state.session_id,
                word_count,
                min_words,
            )
            return

        text = _normalize_company_terms(text)
        state.transcript.append({"role": "user", "text": text})

        if _should_hangup(text):
            LOG.info("[call %s] Hangup intent detected: %s", state.session_id, _clip(text, 80))
            await _hangup_call(ws, state)
            return

        LOG.info(
            "[call %s] STT: %s (avg_logprob=%.2f no_speech_prob=%.2f)",
            state.session_id,
            _clip(text, 200),
            avg_logprob,
            no_speech_prob,
        )
        state.llm_task = asyncio.create_task(run_llm_and_tts(ws, state, text, turn_id, t_turn0))
        try:
            await state.llm_task
        except asyncio.CancelledError:
            pass
        finally:
            state.llm_task = None
    finally:
        # Always re-enable listening so short/empty transcripts don't lock the session.
        state.block_listen = False


async def utterance_worker(ws: WebSocket, state: SessionState, q: asyncio.Queue):
    try:
        while True:
            job = await q.get()
            if job is None:
                break
            if job.pcm16 is not None:
                await process_utterance(ws, state, job.pcm16)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        LOG.error("worker error: %s", e)
    finally:
        await cancel_task(state.speaking_task, "tts")
        await cancel_task(state.llm_task, "llm")


async def enqueue_utterance(utter_q: asyncio.Queue, job: UtteranceJob):
    if utter_q.full():
        LOG.warning("utter_q full, dropping oldest")
        try:
            _ = utter_q.get_nowait()
        except Exception:
            pass
    await utter_q.put(job)


def _parse_metadata(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _coerce_dtmf_digit(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(int(value))
    if isinstance(value, str):
        return value.strip()
    return ""


def _extract_session_id_from_meta(meta: dict) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    return (
        meta.get("session_id")
        or meta.get("uuid")
        or meta.get("call_uuid")
        or meta.get("channel_uuid")
        or meta.get("channel_uuid")
    )


def _extract_dtmf_from_meta(meta: dict) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(meta, dict):
        return None, None
    session_id = _extract_session_id_from_meta(meta)
    digit = _coerce_dtmf_digit(
        meta.get("digit")
        or meta.get("dtmf_digit")
        or meta.get("dtmfDigit")
        or meta.get("dtmf-digit")
        or meta.get("dtmf_digit_number")
        or meta.get("dtmfDigitNumber")
    )
    if not digit:
        dtmf_payload = meta.get("dtmf")
        if isinstance(dtmf_payload, dict):
            digit = _coerce_dtmf_digit(
                dtmf_payload.get("digit")
                or dtmf_payload.get("dtmf_digit")
                or dtmf_payload.get("dtmfDigit")
            )
        elif isinstance(dtmf_payload, str):
            digit = _coerce_dtmf_digit(dtmf_payload)
    return (session_id, digit) if digit else (session_id, None)


def _looks_like_dtmf_meta(meta: dict) -> bool:
    if not isinstance(meta, dict):
        return False
    event = str(meta.get("event") or meta.get("type") or "").lower()
    if "dtmf" in event:
        return True
    for key in meta.keys():
        key_lc = str(key).lower()
        if key_lc == "digit" or key_lc.startswith("dtmf"):
            return True
    return False


@app.websocket("/ws_fs")
async def ws_fs_endpoint(ws: WebSocket):
    await ws.accept()
    vad = webrtcvad.Vad(VAD_MODE)

    session_id = ws.query_params.get("session_id") or str(uuid.uuid4())
    lang_param = (ws.query_params.get("lang") or "").strip().lower()
    state = SessionState(session_id=session_id)
    state.ws = ws
    ACTIVE_SESSIONS[state.session_id] = state
    if lang_param:
        state.language = lang_param
        state.language_locked = True
        state.voice = _voice_for_lang(lang_param)
        state.lang_selected = True
        state.lang_select_active = False
        LOG.info(
            "[call %s] Language preset (dialplan): %s -> voice=%s",
            state.session_id,
            state.language,
            state.voice,
        )
    elif STT_LANGUAGE and STT_LANGUAGE.lower() not in ("auto", "none", ""):
        state.language = STT_LANGUAGE.lower()
        state.language_locked = True
        state.voice = _voice_for_lang(state.language)
        LOG.info(
            "[call %s] Language preset: %s -> voice=%s",
            state.session_id,
            state.language,
            state.voice,
        )
    elif DEFAULT_LANGUAGE:
        state.voice = _voice_for_lang(DEFAULT_LANGUAGE)
    LOG.info("[call %s] WS connected", state.session_id)

    utter_q: asyncio.Queue[Optional[UtteranceJob]] = asyncio.Queue(maxsize=2)
    state.utter_q = utter_q
    state.worker_task = asyncio.create_task(utterance_worker(ws, state, utter_q))

    if IVR_LANG_SELECT_ENABLED:
        state.lang_select_active = True
        # Timeout <= 0 means "no timeout"
        if IVR_LANG_TIMEOUT_MS > 0:
            state.lang_select_deadline = time.time() + (IVR_LANG_TIMEOUT_MS / 1000.0)
        else:
            state.lang_select_deadline = 0.0

    if IVR_LANG_SELECT_ENABLED and IVR_LANG_PROMPT and not state.lang_selected:
        # IVR-first flow: prompt for language, then greet after selection.
        try:
            state.block_listen = True
            _set_mic_state(state, True, "greeting")
            state.ivr_prompt_task = asyncio.create_task(play_tts_text(ws, state, IVR_LANG_PROMPT))
            try:
                await state.ivr_prompt_task
            except asyncio.CancelledError:
                # DTMF language switch can cancel the prompt; keep the call alive.
                pass
        finally:
            state.ivr_prompt_task = None
            state.block_listen = False
    elif GREETING_ENABLED and GREETING_TEXT:
        try:
            state.block_listen = True
            _set_mic_state(state, True, "greeting")
            state.ivr_prompt_task = asyncio.create_task(play_tts_text(ws, state, GREETING_TEXT))
            try:
                await state.ivr_prompt_task
            except asyncio.CancelledError:
                # Allow barge-in / DTMF to cancel greeting without killing the call.
                pass
        finally:
            state.ivr_prompt_task = None
            state.block_listen = False

    pcm_buffer = bytearray()
    speech_buffer = bytearray()
    pre_speech_buffer = bytearray()
    in_speech = False
    silence_frames = 0
    speech_start_frames = 0
    noise_floor = float(os.getenv("NOISE_FLOOR_INIT", "0.0005"))
    max_utt_bytes = int(MAX_UTTERANCE_SECONDS * SAMPLE_RATE * SAMPLE_WIDTH)

    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError as e:
                # Starlette raises on receive after disconnect
                if "disconnect" in str(e).lower():
                    break
                raise

            if msg.get("text") is not None:
                meta = _parse_metadata(msg["text"])
                meta_session_id = _extract_session_id_from_meta(meta)
                if meta_session_id:
                    old_id = state.session_id
                    state.session_id = str(meta_session_id)
                    if old_id != state.session_id:
                        ACTIVE_SESSIONS.pop(old_id, None)
                        ACTIVE_SESSIONS[state.session_id] = state
                    LOG.info("[call %s] Metadata session_id set", state.session_id)
                if meta.get("caller_id"):
                    state.caller_id = str(meta["caller_id"])
                    LOG.info("[call %s] Metadata caller_id set: %s", state.session_id, state.caller_id)
                if _looks_like_dtmf_meta(meta):
                    dtmf_session_id, digit = _extract_dtmf_from_meta(meta)
                    if digit:
                        await _handle_dtmf_digit(dtmf_session_id or state.session_id, digit, source="ws")
                        continue
                continue

            b = msg.get("bytes")
            if not b:
                continue

            pcm_buffer.extend(b)

            while len(pcm_buffer) >= FRAME_BYTES:
                if IVR_LANG_SELECT_ENABLED and state.lang_select_active and state.lang_select_deadline > 0:
                    if time.time() > state.lang_select_deadline:
                        LOG.info("[call %s] IVR language selection timed out", state.session_id)
                        _apply_language_choice(state, DEFAULT_LANGUAGE or LANG_FALLBACK)
                        state.lang_select_active = False
                frame = bytes(pcm_buffer[:FRAME_BYTES])
                del pcm_buffer[:FRAME_BYTES]

                muted_reason = ""
                if state.block_listen:
                    muted_reason = "processing"
                elif not LISTEN_DURING_TTS and state.suppress_until and time.time() < state.suppress_until:
                    muted_reason = "tts"

                if muted_reason:
                    _set_mic_state(state, True, muted_reason)
                    # LLM/TTS in progress; ignore mic input to avoid echo/false triggers.
                    in_speech = False
                    silence_frames = 0
                    speech_start_frames = 0
                    speech_buffer.clear()
                    pre_speech_buffer.clear()
                    continue
                else:
                    _set_mic_state(state, False, "")

                rms = pcm16_rms(frame)
                if not in_speech:
                    noise_floor = (NOISE_FLOOR_ALPHA * noise_floor) + ((1.0 - NOISE_FLOOR_ALPHA) * rms)
                dynamic_thresh = max(VAD_RMS_THRESHOLD, noise_floor * NOISE_FLOOR_MULT)
                is_speech = vad.is_speech(frame, SAMPLE_RATE) and rms >= dynamic_thresh

                if ALLOW_BARGE_IN and is_speech and (
                    state.speaking_task and not state.speaking_task.done()
                ):
                    await handle_barge_in(state)

                if is_speech:
                    if not in_speech:
                        speech_start_frames += 1
                        pre_speech_buffer.extend(frame)
                        if speech_start_frames < START_SPEECH_FRAMES:
                            continue
                        in_speech = True
                        silence_frames = 0
                        speech_buffer.clear()
                        speech_buffer.extend(pre_speech_buffer)
                        pre_speech_buffer.clear()
                        LOG.info("[call %s] Speech start", state.session_id)

                    speech_buffer.extend(frame)
                    silence_frames = 0

                    if len(speech_buffer) >= max_utt_bytes:
                        in_speech = False
                        utterance = bytes(speech_buffer)
                        speech_buffer.clear()
                        LOG.info("[call %s] Speech end (max_len), bytes=%d", state.session_id, len(utterance))
                        await enqueue_utterance(utter_q, UtteranceJob(pcm16=utterance))

                else:
                    speech_start_frames = 0
                    pre_speech_buffer.clear()
                    if in_speech:
                        silence_frames += 1
                        speech_buffer.extend(frame)

                        if silence_frames >= END_SILENCE_FRAMES:
                            in_speech = False
                            silence_frames = 0
                            utterance = bytes(speech_buffer)
                            speech_buffer.clear()
                            LOG.info("[call %s] Speech end (silence), bytes=%d", state.session_id, len(utterance))
                            await enqueue_utterance(utter_q, UtteranceJob(pcm16=utterance))

    except WebSocketDisconnect:
        pass
    finally:
        LOG.info("[call %s] WS disconnected", state.session_id)
        ACTIVE_SESSIONS.pop(state.session_id, None)
        try:
            await utter_q.put(None)
        except Exception:
            pass
        await cancel_task(state.worker_task, "worker")
        await cancel_task(state.speaking_task, "tts")
        await cancel_task(state.llm_task, "llm")
        await cancel_task(state.mic_unmute_task, "mic_unmute")


async def _esl_read_message(reader: asyncio.StreamReader) -> dict:
    headers = {}
    while True:
        line = await reader.readline()
        if not line:
            break
        if line in (b"\n", b"\r\n"):
            break
        decoded = line.decode("utf-8", errors="ignore").strip()
        if not decoded:
            break
        key, _, value = decoded.partition(":")
        if key:
            headers[key.strip()] = value.strip()

    body = b""
    content_len = headers.get("Content-Length")
    if content_len:
        try:
            length = int(content_len)
        except ValueError:
            length = 0
        if length > 0:
            body = await reader.readexactly(length)

    return {"headers": headers, "body": body.decode("utf-8", errors="ignore")}


def _parse_event_headers(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, value = line.partition(":")
        if key:
            k = unquote(key.strip())
            v = unquote(value.strip())
            out[k] = v
    return out


def _extract_dtmf(event_headers: dict, event_body: str) -> tuple[Optional[str], Optional[str]]:
    event_headers_lc = {k.lower(): v for k, v in event_headers.items()}
    session_id = (
        event_headers.get("Unique-ID")
        or event_headers.get("Channel-Call-UUID")
        or event_headers.get("Channel-UUID")
        or event_headers.get("variable_uuid")
        or event_headers_lc.get("unique-id")
        or event_headers_lc.get("channel-call-uuid")
        or event_headers_lc.get("channel-uuid")
        or event_headers_lc.get("variable_uuid")
    )
    digit = event_headers_lc.get("dtmf-digit") or event_headers_lc.get("dtmf-digit-number") or ""
    if not digit and event_body:
        for line in event_body.splitlines():
            if line.lower().startswith("dtmf-digit:"):
                digit = line.split(":", 1)[1].strip()
                break
    return (session_id, digit) if digit else (session_id, None)


async def _handle_dtmf_digit(session_id: Optional[str], digit: str, source: str = "esl") -> None:
    if not digit:
        return
    if session_id:
        LOG.info("[call %s] DTMF %s received via %s", session_id, digit, source)
    else:
        LOG.info("[call unknown] DTMF %s received via %s (no session)", digit, source)

    if IVR_LANG_SELECT_ENABLED:
        state = ACTIVE_SESSIONS.get(session_id) if session_id else None
        if not state and len(ACTIVE_SESSIONS) == 1:
            state = next(iter(ACTIVE_SESSIONS.values()))
            LOG.info("[call %s] DTMF %s mapped to only active session", state.session_id, digit)
        if state:
            lang = IVR_LANG_MAP.get(digit)
            if lang:
                if state.lang_switch_task and not state.lang_switch_task.done():
                    state.lang_switch_task.cancel()
                # Cancel any in-flight prompt/audio and switch immediately.
                state.tts_generation += 1
                state.lang_switch_task = asyncio.create_task(_handle_language_switch(state, lang))
            else:
                LOG.info("[call %s] DTMF %s ignored (no mapping)", state.session_id, digit)
        else:
            LOG.info(
                "[call unknown] DTMF %s received (no matching session, id=%s)",
                digit,
                session_id,
            )


async def _handle_dtmf_event(event_headers: dict, event_body: str) -> bool:
    event_name = event_headers.get("Event-Name", "")
    event_subclass = event_headers.get("Event-Subclass", "")
    event_name_lc = event_name.lower()
    event_subclass_lc = event_subclass.lower()
    body_lc = event_body.lower() if event_body else ""
    has_dtmf_header = any(
        k.lower() in ("dtmf-digit", "dtmf-digit-number") for k in event_headers.keys()
    )
    if (
        not has_dtmf_header
        and "dtmf" not in event_name_lc
        and "dtmf" not in event_subclass_lc
        and "dtmf-digit" not in body_lc
    ):
        return False
    if DTMF_DEBUG_RAW:
        LOG.info(
            "[dtmf raw] name=%s subclass=%s headers=%s body=%s",
            event_name,
            event_subclass,
            {k: event_headers.get(k) for k in sorted(event_headers.keys()) if "DTMF" in k or k in ("Event-Name", "Unique-ID", "Channel-Call-UUID")},
            event_body[:200] if event_body else "",
        )

    session_id, digit = _extract_dtmf(event_headers, event_body)
    if not digit:
        LOG.info(
            "[call %s] DTMF event missing digit (Event-Name=%s)",
            session_id or "unknown",
            event_name or event_subclass or "?",
        )
        return True
    await _handle_dtmf_digit(session_id, digit, source="esl")
    return True


async def _esl_api_cmd(cmd: str) -> str:
    async with ESL_API_SEMAPHORE:
        reader, writer = await asyncio.open_connection(ESL_HOST, ESL_PORT)
        _ = await _esl_read_message(reader)  # auth/request
        writer.write(f"auth {ESL_PASSWORD}\n\n".encode("utf-8"))
        await writer.drain()
        _ = await _esl_read_message(reader)  # auth reply
        writer.write(f"api {cmd}\n\n".encode("utf-8"))
        await writer.drain()
        reply = await _esl_read_message(reader)  # api reply
        writer.close()
        await writer.wait_closed()
        return reply.get("headers", {}).get("Reply-Text", "")


async def _esl_playback_loop():
    backoff = 1.0
    while True:
        try:
            LOG.info("ESL listener connecting to %s:%s", ESL_HOST, ESL_PORT)
            reader, writer = await asyncio.open_connection(ESL_HOST, ESL_PORT)
            _ = await _esl_read_message(reader)  # auth/request
            writer.write(f"auth {ESL_PASSWORD}\n\n".encode("utf-8"))
            await writer.drain()
            auth_reply = await _esl_read_message(reader)
            reply_text = auth_reply.get("headers", {}).get("Reply-Text", "")
            if not reply_text.startswith("+OK"):
                raise RuntimeError(f"ESL auth failed: {reply_text or 'unknown'}")

            # Subscribe to ALL events to ensure DTMF is never missed.
            # We'll filter aggressively in code to keep overhead manageable.
            writer.write(b"event plain ALL\n\n")
            await writer.drain()
            sub_reply = await _esl_read_message(reader)  # subscribe reply
            LOG.info(
                "ESL subscribed to ALL: %s",
                sub_reply.get("headers", {}).get("Reply-Text", "no-reply"),
            )

            backoff = 1.0
            while True:
                msg = await _esl_read_message(reader)
                headers = msg.get("headers", {})
                content_type = (headers.get("Content-Type") or "").lower()

                body = msg.get("body", "")
                event_headers = None
                event_body = ""

                if body:
                    if "text/event-plain" not in content_type:
                        body_lc = body.lower()
                        if "event-name:" not in body_lc and "dtmf-digit" not in body_lc:
                            continue
                    parts = body.split("\n\n", 1)
                    event_headers = _parse_event_headers(parts[0])
                    event_body = parts[1].strip() if len(parts) > 1 else ""
                else:
                    # Some ESL DTMF events may arrive without a body; fall back to top-level headers.
                    if headers.get("Event-Name"):
                        event_headers = headers
                        event_body = ""

                if not event_headers:
                    continue

                if await _handle_dtmf_event(event_headers, event_body):
                    continue

                if event_headers.get("Event-Subclass") != "mod_audio_stream::play":
                    continue

                try:
                    payload = json.loads(event_body) if event_body else {}
                except json.JSONDecodeError:
                    payload = {}

                file_path = payload.get("file")
                audio_type = payload.get("audioDataType")
                session_id = (
                    event_headers.get("Unique-ID")
                    or event_headers.get("Channel-Call-UUID")
                    or event_headers.get("Channel-UUID")
                    or event_headers.get("variable_uuid")
                )

                if audio_type == "raw":
                    LOG.info("[call %s] ESL play skipped for raw audio", session_id or "unknown")
                    continue

                if file_path and session_id:
                    LOG.info("[call %s] ESL play -> %s", session_id, file_path)
                    async def _broadcast():
                        reply = await _esl_api_cmd(f"uuid_broadcast {session_id} {file_path} aleg")
                        if reply and not reply.startswith("+OK"):
                            LOG.warning("[call %s] uuid_broadcast failed: %s", session_id, reply)
                            await asyncio.sleep(0.2)
                            reply2 = await _esl_api_cmd(f"uuid_broadcast {session_id} {file_path} aleg")
                            if reply2 and not reply2.startswith("+OK"):
                                LOG.warning("[call %s] uuid_broadcast retry failed: %s", session_id, reply2)

                    asyncio.create_task(_broadcast())

        except asyncio.CancelledError:
            break
        except Exception as e:
            LOG.warning("ESL playback loop error: %s", e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)


@app.on_event("startup")
async def on_startup():
    if ESL_ENABLED:
        global ESL_TASK
        ESL_TASK = asyncio.create_task(_esl_playback_loop())


@app.on_event("shutdown")
async def on_shutdown():
    global ESL_TASK
    if ESL_TASK and not ESL_TASK.done():
        ESL_TASK.cancel()
        try:
            await ESL_TASK
        except asyncio.CancelledError:
            pass
