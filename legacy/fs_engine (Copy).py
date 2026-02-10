# fs_engine.py
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
import shutil
import subprocess
import time
import uuid
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


def _matches_allowed_topics(text: str, topics: list[str]) -> bool:
    if not topics:
        return True
    t = text.lower()
    return any(topic in t for topic in topics)


# Ollama (OpenAI-compatible)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
LLM_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_MAX_CONCURRENCY", "4")))
LLM_SEMAPHORE = asyncio.Semaphore(LLM_MAX_CONCURRENCY)
llm_client = AsyncOpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)

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
_company_name = COMPANY_PROFILE.get("company_name", "").strip() if isinstance(COMPANY_PROFILE, dict) else ""
COMPANY_RULES = (
    "Company-only policy:\n"
    "- Only answer questions related to the company, its services, plans, pricing, support, or account issues.\n"
    "- If a request is unrelated, politely refuse and redirect to company topics.\n"
)
if PROFILE_TEXT:
    SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n\n" + COMPANY_RULES + "\nCompany profile:\n" + PROFILE_TEXT + "\n"
else:
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
whisper_default = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
whisper_kh = None
if USE_DUAL_WHISPER:
    try:
        whisper_kh = WhisperModel(KHMER_WHISPER_PATH, device="cpu", compute_type="int8")
        print("[boot] Khmer Whisper loaded.")
    except Exception as e:
        whisper_kh = None
        print(f"[boot] Khmer Whisper failed to load: {e}")
print("[boot] Whisper loaded.")

app = FastAPI()


@dataclass
class SessionState:
    session_id: str
    speaking_task: Optional[asyncio.Task] = None
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


def _trim_history(history: list[dict], max_messages: int = 16) -> None:
    if len(history) > max_messages:
        del history[:-max_messages]


async def ollama_reply(state: SessionState, prompt: str) -> str:
    if ALLOWED_TOPICS and not _matches_allowed_topics(prompt, ALLOWED_TOPICS):
        state.offtopic_count += 1
        if state.offtopic_count >= OFFTOPIC_REPEAT_LIMIT:
            return OFFTOPIC_RESPONSE_SHORT
        return OFFTOPIC_RESPONSE
    state.offtopic_count = 0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
    async with LLM_SEMAPHORE:
        resp = await llm_client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
    msg = resp.choices[0].message.content or ""
    cleaned = msg.strip()
    if prompt:
        state.history.append({"role": "user", "content": prompt})
    if cleaned:
        state.history.append({"role": "assistant", "content": cleaned})
    _trim_history(state.history)
    return cleaned


async def synthesize_tts_mp3(text: str, voice: str) -> bytes:
    tts = edge_tts.Communicate(text=text, voice=voice)
    chunks = []
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


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
    reply = text
    if MAX_TTS_CHARS > 0:
        reply = reply[:MAX_TTS_CHARS]
    if SERVER_TTS_ENABLED and USE_FS_TTS:
        safe_text = reply.replace("\n", " ").replace("\r", " ").replace('"', "").replace("|", " ").strip()
        if not safe_text:
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


async def run_llm_and_tts(ws: WebSocket, state: SessionState, user_text: str, turn_id: int):
    try:
        LOG.info("[call %s] LLM request: %s", state.session_id, _clip(user_text, 160))
        reply = await ollama_reply(state, user_text)
        if not reply:
            LOG.info("[call %s] LLM empty reply", state.session_id)
            return
        # If a newer utterance started, drop this reply.
        if state.turn_id != turn_id:
            LOG.info("[call %s] LLM reply dropped (superseded)", state.session_id)
            return
        LOG.info("[call %s] LLM reply: %s", state.session_id, _clip(reply, 200))
        await play_tts_text(ws, state, reply)
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
            text, avg_logprob, no_speech_prob, detected_lang, lang_prob = await loop.run_in_executor(
                None, stt_whisper_from_pcm16, utterance_pcm16, lang_hint, model
            )
        except Exception as e:
            LOG.error("STT error: %s", e)
            return

        if detected_lang:
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

        LOG.info(
            "[call %s] STT: %s (avg_logprob=%.2f no_speech_prob=%.2f)",
            state.session_id,
            _clip(text, 200),
            avg_logprob,
            no_speech_prob,
        )
        state.llm_task = asyncio.create_task(run_llm_and_tts(ws, state, text, turn_id))
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


@app.websocket("/ws_fs")
async def ws_fs_endpoint(ws: WebSocket):
    await ws.accept()
    vad = webrtcvad.Vad(VAD_MODE)

    session_id = ws.query_params.get("session_id") or str(uuid.uuid4())
    state = SessionState(session_id=session_id)
    if STT_LANGUAGE and STT_LANGUAGE.lower() not in ("auto", "none", ""):
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
    state.worker_task = asyncio.create_task(utterance_worker(ws, state, utter_q))

    if GREETING_ENABLED and GREETING_TEXT:
        try:
            state.block_listen = True
            _set_mic_state(state, True, "greeting")
            await play_tts_text(ws, state, GREETING_TEXT)
        finally:
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
                if meta.get("session_id"):
                    state.session_id = str(meta["session_id"])
                    LOG.info("[call %s] Metadata session_id set", state.session_id)
                continue

            b = msg.get("bytes")
            if not b:
                continue

            pcm_buffer.extend(b)

            while len(pcm_buffer) >= FRAME_BYTES:
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

            writer.write(b"event plain CUSTOM mod_audio_stream::play\n\n")
            await writer.drain()
            _ = await _esl_read_message(reader)  # subscribe reply
            LOG.info("ESL subscribed to mod_audio_stream::play")

            backoff = 1.0
            while True:
                msg = await _esl_read_message(reader)
                headers = msg.get("headers", {})
                if headers.get("Content-Type") != "text/event-plain":
                    continue

                body = msg.get("body", "")
                if not body:
                    continue

                parts = body.split("\n\n", 1)
                event_headers = _parse_event_headers(parts[0])
                event_body = parts[1].strip() if len(parts) > 1 else ""

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
                    reply = await _esl_api_cmd(f"uuid_broadcast {session_id} {file_path} aleg")
                    if reply and not reply.startswith("+OK"):
                        LOG.warning("[call %s] uuid_broadcast failed: %s", session_id, reply)
                        await asyncio.sleep(0.2)
                        reply = await _esl_api_cmd(f"uuid_broadcast {session_id} {file_path} aleg")
                        if reply and not reply.startswith("+OK"):
                            LOG.warning("[call %s] uuid_broadcast retry failed: %s", session_id, reply)

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
