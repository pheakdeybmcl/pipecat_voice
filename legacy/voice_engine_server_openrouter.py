# voice_engine_server.py
# Realtime WS voice engine with:
# - VAD (webrtcvad) utterance segmentation
# - STT (faster-whisper)
# - LLM (OpenRouter /v1)
# - (Optional) Server-side TTS streaming via edge_tts
# - FIX: heavy work moved off the websocket receive loop to prevent ping timeouts

import asyncio
import base64
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
import webrtcvad
import httpx
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from starlette.websockets import WebSocketDisconnect
from faster_whisper import WhisperModel
import edge_tts

# ---------------- Config ----------------
# OpenRouter (OpenAI-compatible)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "tngtech/deepseek-r1t2-chimera:free"
# Optional hardcoded key (not recommended). Prefer env: OPENROUTER_API_KEY.
OPENROUTER_API_KEY_INLINE = "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"
OPENROUTER_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60"))
OPENROUTER_MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))
OPENROUTER_RETRY_BASE_SECONDS = float(os.getenv("OPENROUTER_RETRY_BASE_SECONDS", "0.6"))
OPENROUTER_RETRY_MAX_SECONDS = float(os.getenv("OPENROUTER_RETRY_MAX_SECONDS", "6.0"))
LLM_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_MAX_CONCURRENCY", "4")))
LLM_BREAKER_THRESHOLD = max(1, int(os.getenv("LLM_BREAKER_THRESHOLD", "5")))
LLM_BREAKER_COOLDOWN_SECONDS = int(os.getenv("LLM_BREAKER_COOLDOWN_SECONDS", "20"))
LLM_EMPTY_RETRY_ENABLED = os.getenv("LLM_EMPTY_RETRY_ENABLED", "true").lower() in ("1", "true", "yes")
LLM_EMPTY_RETRY_PROMPT = os.getenv(
    "LLM_EMPTY_RETRY_PROMPT",
    "Please answer the previous user message in one short sentence.",
)

STT_MAX_WORKERS = max(1, int(os.getenv("STT_MAX_WORKERS", "2")))

HISTORY_PERSIST_ENABLED = os.getenv("HISTORY_PERSIST_ENABLED", "true").lower() in ("1", "true", "yes")
HISTORY_STORE_PATH = os.getenv("HISTORY_STORE_PATH", "voice_engine_history.sqlite3")
HISTORY_TTL_SECONDS = int(os.getenv("HISTORY_TTL_SECONDS", "3600"))
HISTORY_REDACT_PII = os.getenv("HISTORY_REDACT_PII", "true").lower() in ("1", "true", "yes")


def _get_openrouter_config() -> tuple[str, str, str]:
    key = (OPENROUTER_API_KEY_INLINE or os.getenv("OPENROUTER_API_KEY", "") or "").strip()
    referer = (os.getenv("OPENROUTER_HTTP_REFERER", "") or "http://localhost:3000").strip()
    title = (os.getenv("OPENROUTER_TITLE", "") or "TestApp").strip()
    return key, referer, title


def _mask_key(key: str) -> str:
    if not key:
        return "missing"
    if len(key) <= 10:
        return "set"
    return f"{key[:6]}...{key[-4:]}"


LLM_SEMAPHORE = asyncio.Semaphore(LLM_MAX_CONCURRENCY)
STT_EXECUTOR = ThreadPoolExecutor(max_workers=STT_MAX_WORKERS)

LLM_BREAKER_LOCK = threading.Lock()
LLM_BREAKER_STATE = {"failures": 0, "open_until": 0.0}


METRICS = Counter()
METRICS_LOCK = threading.Lock()


def _inc_metric(name: str, value: int = 1):
    with METRICS_LOCK:
        METRICS[name] += value


def _metric_snapshot() -> dict:
    with METRICS_LOCK:
        return dict(METRICS)


LOG = logging.getLogger("voice_engine")
if not logging.getLogger().handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
LOG.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

HTTP_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(OPENROUTER_TIMEOUT_SECONDS))


def log_event(event: str, session_id: Optional[str] = None, **fields):
    payload = {"event": event, "ts": time.time()}
    if session_id:
        payload["session_id"] = session_id
    payload.update(fields)
    LOG.info(json.dumps(payload, ensure_ascii=False))


def _breaker_check_open() -> float:
    with LLM_BREAKER_LOCK:
        open_until = LLM_BREAKER_STATE.get("open_until", 0.0)
        return open_until if time.time() < open_until else 0.0


def _breaker_record_failure():
    with LLM_BREAKER_LOCK:
        LLM_BREAKER_STATE["failures"] = LLM_BREAKER_STATE.get("failures", 0) + 1
        if LLM_BREAKER_STATE["failures"] >= LLM_BREAKER_THRESHOLD:
            LLM_BREAKER_STATE["open_until"] = time.time() + LLM_BREAKER_COOLDOWN_SECONDS
            LLM_BREAKER_STATE["failures"] = 0
            log_event("llm_circuit_open", cooldown_seconds=LLM_BREAKER_COOLDOWN_SECONDS)


def _breaker_record_success():
    with LLM_BREAKER_LOCK:
        LLM_BREAKER_STATE["failures"] = 0
        LLM_BREAKER_STATE["open_until"] = 0.0
SYSTEM_PROMPT = (
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

TOOLS = []

# Audio input expected from browser: PCM16 little-endian, 16kHz, mono
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes per sample (PCM16)
FRAME_MS = 20     # VAD supports only 10/20/30ms
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH  # 640 bytes
MIN_UTTERANCE_MS = int(os.getenv("MIN_UTTERANCE_MS", "400"))
MIN_UTTERANCE_FRAMES = max(1, int(MIN_UTTERANCE_MS / FRAME_MS))
MIN_UTTERANCE_BYTES = MIN_UTTERANCE_FRAMES * FRAME_BYTES
MIN_UTTERANCE_RMS = float(os.getenv("MIN_UTTERANCE_RMS", "0.012"))

VAD_MODE = 2  # 0-3 (3 = most aggressive)
MAX_UTTERANCE_SECONDS = 12.0
END_SILENCE_MS = 2000  # wait longer after user stops before finalizing utterance
END_SILENCE_FRAMES = int(END_SILENCE_MS / FRAME_MS)

# Toggle server-side audio streaming.
# If False: server sends only ai_text; browser should speak using SpeechSynthesis.
SERVER_TTS_ENABLED = True
DEFAULT_TTS_VOICE = "en-US-JennyNeural"
VOICE_CACHE_TTL_SECONDS = 3600
VOICE_CACHE = {"ts": 0.0, "voices": []}
ALLOW_BARGE_IN = False
MAX_HISTORY_MESSAGES = 16  # short-term memory window (user+assistant messages)

# Whisper
WHISPER_SIZE = "small"  # "tiny" faster

print("[boot] Loading Whisper model...")
whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
print("[boot] Whisper loaded.")
_key, _ref, _title = _get_openrouter_config()
print(f"[boot] OpenRouter key: {_mask_key(_key)}")

app = FastAPI()
APP_DIR = __import__("pathlib").Path(__file__).parent


@app.on_event("shutdown")
async def on_shutdown():
    try:
        await HTTP_CLIENT.aclose()
    except Exception:
        pass
    try:
        STT_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


@dataclass
class SessionState:
    speaking_task: Optional[asyncio.Task] = None
    llm_task: Optional[asyncio.Task] = None
    worker_task: Optional[asyncio.Task] = None
    interrupted: bool = False
    voice: str = DEFAULT_TTS_VOICE
    history: Optional[list[dict]] = None
    session_id: str = ""


@dataclass
class UtteranceJob:
    pcm16: Optional[bytes] = None
    text: Optional[str] = None


def pcm16_bytes_to_float32(pcm: bytes) -> np.ndarray:
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    a /= 32768.0
    return a


def pcm16_rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if a.size == 0:
        return 0.0
    a /= 32768.0
    return float(np.sqrt(np.mean(a * a)))


def stt_whisper_from_pcm16(pcm: bytes) -> str:
    audio = pcm16_bytes_to_float32(pcm)
    segments, _info = whisper.transcribe(audio, language="en")
    text = "".join(seg.text for seg in segments).strip()
    return text


PII_PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"), "[email]"),
    (re.compile(r"\\b\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b"), "[phone]"),
]


def _redact_pii(text: str) -> str:
    if not text:
        return text
    redacted = text
    for pattern, repl in PII_PATTERNS:
        redacted = pattern.sub(repl, redacted)
    return redacted


class SqliteHistoryStore:
    def __init__(self, path: str, ttl_seconds: int):
        self.path = path
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._last_purge = 0.0
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def _purge_if_needed(self):
        if self.ttl_seconds <= 0:
            return
        now = time.time()
        if now - self._last_purge < 30:
            return
        self._last_purge = now
        cutoff = now - self.ttl_seconds
        with self._connect() as conn:
            conn.execute("DELETE FROM history WHERE updated_at < ?", (cutoff,))
            conn.commit()

    def load(self, session_id: str) -> list[dict]:
        if not session_id:
            return []
        with self._lock:
            self._purge_if_needed()
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT data FROM history WHERE session_id = ?",
                    (session_id,),
                )
                row = cur.fetchone()
                if not row:
                    return []
                try:
                    return json.loads(row[0]) or []
                except Exception:
                    return []

    def save(self, session_id: str, history: list[dict]):
        if not session_id:
            return
        with self._lock:
            self._purge_if_needed()
            payload = json.dumps(history, ensure_ascii=False)
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO history(session_id, data, updated_at) VALUES (?, ?, ?)\n"
                    "ON CONFLICT(session_id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
                    (session_id, payload, time.time()),
                )
                conn.commit()

    def clear(self, session_id: str):
        if not session_id:
            return
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
                conn.commit()


HISTORY_STORE = SqliteHistoryStore(HISTORY_STORE_PATH, HISTORY_TTL_SECONDS) if HISTORY_PERSIST_ENABLED else None


def _ensure_history(state: SessionState) -> list[dict]:
    if state.history is None:
        state.history = []
    return state.history


def _trim_history(history: list[dict]):
    if MAX_HISTORY_MESSAGES <= 0:
        history.clear()
        return
    if len(history) <= MAX_HISTORY_MESSAGES:
        return
    del history[: max(0, len(history) - MAX_HISTORY_MESSAGES)]


def _sanitize_history_for_storage(history: list[dict]) -> list[dict]:
    if not history:
        return []
    sanitized = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if content is None:
            content = ""
        if HISTORY_REDACT_PII:
            content = _redact_pii(str(content))
        sanitized.append({"role": role, "content": content})
    return sanitized


def _normalize_reply(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned:
        return cleaned
    _inc_metric("llm_empty")
    return "Sorry, I didn't catch that."


async def _persist_history_async(state: SessionState):
    if not HISTORY_STORE or not state.session_id:
        return
    history = _ensure_history(state)
    sanitized = _sanitize_history_for_storage(history)
    await asyncio.to_thread(HISTORY_STORE.save, state.session_id, sanitized)


async def _load_history_async(session_id: str) -> list[dict]:
    if not HISTORY_STORE or not session_id:
        return []
    return await asyncio.to_thread(HISTORY_STORE.load, session_id)


async def _clear_history_async(session_id: str):
    if not HISTORY_STORE or not session_id:
        return
    await asyncio.to_thread(HISTORY_STORE.clear, session_id)


async def openrouter_chat_completion(messages: list, tools: Optional[list] = None, tool_choice: Optional[str] = None) -> dict:
    open_until = _breaker_check_open()
    if open_until:
        raise RuntimeError(f"LLM circuit open until {open_until:.0f}")
    _inc_metric("llm_requests")
    key, referer, title = _get_openrouter_config()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (export it before starting the server)")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer,
        "X-Title": title,
    }
    payload = {
        "model": MODEL,
        "messages": messages,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    retryable = {408, 409, 425, 429, 500, 502, 503, 504, 522, 524}
    last_error: Optional[str] = None
    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        try:
            async with LLM_SEMAPHORE:
                resp = await HTTP_CLIENT.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )
        except httpx.RequestError as e:
            last_error = f"request error: {e}"
            _inc_metric("llm_request_errors")
            if attempt < OPENROUTER_MAX_RETRIES:
                _inc_metric("llm_retries")
                delay = min(
                    OPENROUTER_RETRY_MAX_SECONDS,
                    OPENROUTER_RETRY_BASE_SECONDS * (2 ** attempt),
                ) + random.uniform(0, 0.2)
                await asyncio.sleep(delay)
                continue
            _breaker_record_failure()
            raise RuntimeError(f"OpenRouter request failed: {last_error}") from e

        if resp.status_code in retryable and attempt < OPENROUTER_MAX_RETRIES:
            _inc_metric("llm_retries")
            delay = min(
                OPENROUTER_RETRY_MAX_SECONDS,
                OPENROUTER_RETRY_BASE_SECONDS * (2 ** attempt),
            ) + random.uniform(0, 0.2)
            await asyncio.sleep(delay)
            continue

        if resp.status_code >= 400:
            detail = (resp.text or "").strip()
            if len(detail) > 800:
                detail = detail[:800] + "..."
            _inc_metric("llm_errors")
            _breaker_record_failure()
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {detail}")

        _breaker_record_success()
        _inc_metric("llm_success")
        return resp.json()

    _breaker_record_failure()
    raise RuntimeError(f"OpenRouter request failed after retries: {last_error or 'unknown'}")


async def openrouter_reply_with_tools(ws: WebSocket, state: SessionState, prompt: str) -> str:
    history = _ensure_history(state)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    try:
        resp = await openrouter_chat_completion(messages)
    except RuntimeError:
        raise
    msg = resp.get("choices", [{}])[0].get("message", {}) or {}
    final_raw = msg.get("content") or ""
    if not final_raw.strip() and LLM_EMPTY_RETRY_ENABLED:
        _inc_metric("llm_empty_retry")
        log_event("llm_empty_retry", session_id=state.session_id)
        retry_messages = list(messages)
        retry_messages.append({"role": "user", "content": LLM_EMPTY_RETRY_PROMPT})
        resp_retry = await openrouter_chat_completion(retry_messages)
        msg_retry = resp_retry.get("choices", [{}])[0].get("message", {}) or {}
        final_raw = msg_retry.get("content") or ""

    final = _normalize_reply(final_raw)
    if prompt:
        history.append({"role": "user", "content": prompt})
    if final:
        history.append({"role": "assistant", "content": final})
    _trim_history(history)
    await _persist_history_async(state)
    return final


async def stream_edge_tts_mp3(text: str, voice: str):
    tts = edge_tts.Communicate(text=text, voice=voice)
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


async def send_json(ws: WebSocket, payload: dict) -> bool:
    """
    Safe send. Returns False if client disconnected / socket closed.
    """
    try:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except Exception:
        return False


async def send_turn_state(ws: WebSocket, state: str):
    await send_json(ws, {"type": "turn_state", "state": state})


async def get_edge_voices() -> list[dict]:
    now = time.time()
    cached = VOICE_CACHE.get("voices") or []
    if cached and (now - VOICE_CACHE.get("ts", 0.0)) < VOICE_CACHE_TTL_SECONDS:
        return cached

    try:
        voices = await edge_tts.list_voices()
        simplified = []
        for v in voices:
            name = v.get("ShortName")
            if not name:
                continue
            simplified.append({
                "name": name,
                "locale": v.get("Locale"),
                "gender": v.get("Gender"),
            })
        if simplified:
            VOICE_CACHE["ts"] = now
            VOICE_CACHE["voices"] = simplified
            return simplified
    except Exception as e:
        print("[warn] voice list failed:", e)

    return [{"name": DEFAULT_TTS_VOICE, "locale": "en-US", "gender": "Female"}]


async def send_voice_list(ws: WebSocket):
    voices = await get_edge_voices()
    await send_json(ws, {"type": "voices", "voices": voices})


async def cancel_task(task: Optional[asyncio.Task], label: str):
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[warn] {label} cancel exception:", e)


async def handle_barge_in(ws: WebSocket, state: SessionState):
    """
    Cancel any ongoing speech/LLM tasks. Tell client to stop playback if applicable.
    """
    state.interrupted = True
    await cancel_task(state.speaking_task, "tts")
    await cancel_task(state.llm_task, "llm")
    state.speaking_task = None
    state.llm_task = None
    await send_json(ws, {"type": "stop_audio"})
    await send_turn_state(ws, "listening")


async def speak_to_client(ws: WebSocket, state: SessionState, text: str):
    """
    Server-side TTS streaming (MP3 chunks). Cancellable by barge-in.
    """
    await send_turn_state(ws, "speaking")
    # Tell client the text either way (useful for debug/UI)
    ok = await send_json(ws, {"type": "ai_text", "text": text})
    if not ok:
        return

    if not SERVER_TTS_ENABLED:
        await send_turn_state(ws, "listening")
        return

    try:
        sent_audio_start = False
        async for mp3_chunk in stream_edge_tts_mp3(text, state.voice):
            if state.interrupted:
                break
            if not sent_audio_start:
                ok = await send_json(ws, {"type": "audio_start", "format": "mp3"})
                if not ok:
                    return
                sent_audio_start = True
            b64 = base64.b64encode(mp3_chunk).decode("ascii")
            ok = await send_json(ws, {"type": "audio_chunk", "format": "mp3", "data": b64})
            if not ok:
                return

        if sent_audio_start:
            await send_json(ws, {"type": "audio_end"})
        await send_turn_state(ws, "listening")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _inc_metric("tts_errors")
        log_event("tts_error", session_id=state.session_id, error=str(e))
        await send_json(ws, {"type": "error", "message": f"TTS error: {e}"})


async def run_llm_and_maybe_tts(ws: WebSocket, state: SessionState, user_text: str):
    """
    LLM reply then (optional) server-side TTS.
    """
    try:
        reply = await openrouter_reply_with_tools(ws, state, user_text)
        state.speaking_task = asyncio.create_task(speak_to_client(ws, state, reply))
        await state.speaking_task
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _inc_metric("llm_errors")
        log_event("llm_error", session_id=state.session_id, error=str(e))
        await send_json(ws, {"type": "error", "message": f"LLM error: {e}"})
    finally:
        await send_turn_state(ws, "listening")


async def process_utterance(ws: WebSocket, state: SessionState, utterance_pcm16: bytes):
    """
    One utterance pipeline: barge-in -> STT -> LLM -> (optional TTS)
    """
    # New utterance interrupts prior response
    await handle_barge_in(ws, state)
    state.interrupted = False

    if len(utterance_pcm16) < MIN_UTTERANCE_BYTES:
        _inc_metric("stt_skipped_short")
        log_event(
            "stt_skipped_short",
            session_id=state.session_id,
            bytes=len(utterance_pcm16),
            min_bytes=MIN_UTTERANCE_BYTES,
        )
        await send_json(ws, {"type": "transcript", "text": "", "empty": True})
        await send_turn_state(ws, "listening")
        return

    if MIN_UTTERANCE_RMS > 0:
        rms = pcm16_rms(utterance_pcm16)
        if rms < MIN_UTTERANCE_RMS:
            _inc_metric("stt_skipped_quiet")
            log_event(
                "stt_skipped_quiet",
                session_id=state.session_id,
                rms=round(rms, 6),
                min_rms=MIN_UTTERANCE_RMS,
            )
            await send_json(ws, {"type": "transcript", "text": "", "empty": True})
            await send_turn_state(ws, "listening")
            return

    # STT off-thread (CPU)
    _inc_metric("stt_requests")
    try:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(STT_EXECUTOR, stt_whisper_from_pcm16, utterance_pcm16)
    except Exception as e:
        _inc_metric("stt_errors")
        log_event("stt_error", session_id=state.session_id, error=str(e))
        await send_json(ws, {"type": "error", "message": f"STT error: {e}"})
        return

    if not text.strip():
        await send_json(ws, {"type": "transcript", "text": "", "empty": True})
        await send_turn_state(ws, "listening")
        return

    ok = await send_json(ws, {"type": "transcript", "text": text})
    if not ok:
        return

    # LLM (+ optional TTS) as cancellable task
    state.llm_task = asyncio.create_task(run_llm_and_maybe_tts(ws, state, text))
    try:
        await state.llm_task
    except asyncio.CancelledError:
        pass
    finally:
        state.llm_task = None


async def process_text_utterance(ws: WebSocket, state: SessionState, text: str):
    """
    Text-only utterance pipeline: barge-in -> LLM -> (optional TTS).
    """
    await handle_barge_in(ws, state)
    state.interrupted = False
    _inc_metric("text_requests")

    clean = (text or "").strip()
    if not clean:
        await send_json(ws, {"type": "transcript", "text": "", "empty": True})
        await send_turn_state(ws, "listening")
        return

    ok = await send_json(ws, {"type": "transcript", "text": clean})
    if not ok:
        return

    state.llm_task = asyncio.create_task(run_llm_and_maybe_tts(ws, state, clean))
    try:
        await state.llm_task
    except asyncio.CancelledError:
        pass
    finally:
        state.llm_task = None


async def utterance_worker(ws: WebSocket, state: SessionState, q: asyncio.Queue):
    """
    FIX: Heavy STT/LLM/TTS work is handled here, not inside the websocket receive loop.
    This prevents keepalive ping timeouts.
    """
    try:
        while True:
            job = await q.get()
            if job is None:
                break
            if job.text is not None:
                await process_text_utterance(ws, state, job.text)
            elif job.pcm16 is not None:
                await process_utterance(ws, state, job.pcm16)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        # Don't crash server; best-effort error send
        await send_json(ws, {"type": "error", "message": f"worker error: {e}"})
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


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    vad = webrtcvad.Vad(VAD_MODE)

    session_id = (
        ws.query_params.get("session_id")
        or ws.headers.get("x-session-id")
        or str(uuid.uuid4())
    )
    state = SessionState(session_id=session_id)
    if HISTORY_STORE:
        state.history = await _load_history_async(session_id)

    _inc_metric("ws_connected")
    log_event("ws_connected", session_id=session_id)

    # FIX: utterance queue + worker
    utter_q: asyncio.Queue[Optional[UtteranceJob]] = asyncio.Queue(maxsize=2)
    state.worker_task = asyncio.create_task(utterance_worker(ws, state, utter_q))

    pcm_buffer = bytearray()
    speech_buffer = bytearray()
    in_speech = False
    silence_frames = 0
    max_utt_bytes = int(MAX_UTTERANCE_SECONDS * SAMPLE_RATE * SAMPLE_WIDTH)
    ptt_active = False

    await send_json(ws, {
        "type": "ready",
        "sample_rate": SAMPLE_RATE,
        "frame_ms": FRAME_MS,
        "frame_bytes": FRAME_BYTES,
        "format_in": "pcm16le",
        "server_tts_enabled": SERVER_TTS_ENABLED,
        "voice": state.voice,
        "session_id": state.session_id,
    })
    asyncio.create_task(send_voice_list(ws))
    await send_turn_state(ws, "idle")

    try:
        while True:
            msg = await ws.receive()

            # Control JSON (text)
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                mtype = data.get("type")
                if mtype == "barge_in":
                    await handle_barge_in(ws, state)
                    continue

                if mtype == "reset":
                    await handle_barge_in(ws, state)
                    pcm_buffer.clear()
                    speech_buffer.clear()
                    in_speech = False
                    silence_frames = 0
                    _ensure_history(state).clear()
                    await _clear_history_async(state.session_id)
                    await send_json(ws, {"type": "reset_ok"})
                    continue

                if mtype == "text_utterance":
                    await send_turn_state(ws, "processing")
                    await enqueue_utterance(utter_q, UtteranceJob(text=data.get("text", "")))
                    continue

                if mtype == "ptt_start":
                    ptt_active = True
                    if not in_speech:
                        in_speech = True
                        silence_frames = 0
                        speech_buffer.clear()
                        await send_json(ws, {"type": "speech_start", "ts": time.time(), "reason": "ptt"})
                        await send_turn_state(ws, "listening")
                    continue

                if mtype == "ptt_end":
                    ptt_active = False
                    if in_speech and speech_buffer:
                        in_speech = False
                        silence_frames = 0
                        await send_json(ws, {"type": "speech_end", "ts": time.time(), "reason": "ptt"})
                        await send_turn_state(ws, "processing")
                        utterance = bytes(speech_buffer)
                        speech_buffer.clear()
                        await enqueue_utterance(utter_q, UtteranceJob(pcm16=utterance))
                    continue

                if mtype == "get_voices":
                    asyncio.create_task(send_voice_list(ws))
                    continue

                if mtype == "set_voice":
                    voice = (data.get("voice") or "").strip()
                    if voice:
                        state.voice = voice
                        await send_json(ws, {"type": "voice_set", "voice": state.voice})
                    continue

                continue

            # Audio binary
            b = msg.get("bytes")
            if not b:
                continue

            pcm_buffer.extend(b)

            # Process 20ms frames
            while len(pcm_buffer) >= FRAME_BYTES:
                frame = bytes(pcm_buffer[:FRAME_BYTES])
                del pcm_buffer[:FRAME_BYTES]

                if ptt_active:
                    if not in_speech:
                        in_speech = True
                        silence_frames = 0
                        speech_buffer.clear()
                        await send_json(ws, {"type": "speech_start", "ts": time.time(), "reason": "ptt"})
                        await send_turn_state(ws, "listening")

                    speech_buffer.extend(frame)
                    silence_frames = 0

                    if len(speech_buffer) >= max_utt_bytes:
                        in_speech = False
                        await send_json(ws, {"type": "speech_end", "ts": time.time(), "reason": "max_len"})
                        await send_turn_state(ws, "processing")
                        utterance = bytes(speech_buffer)
                        speech_buffer.clear()
                        await enqueue_utterance(utter_q, UtteranceJob(pcm16=utterance))
                    continue

                is_speech = vad.is_speech(frame, SAMPLE_RATE)

                # If user talks while AI speaking, treat as barge-in
                if ALLOW_BARGE_IN and is_speech and state.speaking_task and not state.speaking_task.done():
                    await handle_barge_in(ws, state)

                if is_speech:
                    if not in_speech:
                        in_speech = True
                        silence_frames = 0
                        speech_buffer.clear()
                        await send_json(ws, {"type": "speech_start", "ts": time.time()})
                        await send_turn_state(ws, "listening")

                    speech_buffer.extend(frame)
                    silence_frames = 0

                    if len(speech_buffer) >= max_utt_bytes:
                        # Force end
                        in_speech = False
                        await send_json(ws, {"type": "speech_end", "ts": time.time(), "reason": "max_len"})
                        await send_turn_state(ws, "processing")
                        utterance = bytes(speech_buffer)
                        speech_buffer.clear()

                        await enqueue_utterance(utter_q, UtteranceJob(pcm16=utterance))

                else:
                    if in_speech:
                        silence_frames += 1
                        speech_buffer.extend(frame)  # keep trailing silence a bit

                        if silence_frames >= END_SILENCE_FRAMES:
                            in_speech = False
                            silence_frames = 0
                            await send_json(ws, {"type": "speech_end", "ts": time.time(), "reason": "silence"})
                            await send_turn_state(ws, "processing")
                            utterance = bytes(speech_buffer)
                            speech_buffer.clear()

                            await enqueue_utterance(utter_q, UtteranceJob(pcm16=utterance))

    except WebSocketDisconnect:
        pass
    finally:
        # stop worker
        try:
            await utter_q.put(None)
        except Exception:
            pass

        await cancel_task(state.worker_task, "worker")
        await cancel_task(state.speaking_task, "tts")
        await cancel_task(state.llm_task, "llm")
        await _persist_history_async(state)
        _inc_metric("ws_disconnected")
        log_event("ws_disconnected", session_id=state.session_id)


@app.get("/")
async def root():
    html_path = APP_DIR / "test_client.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return HTMLResponse("<h1>test_client.html not found</h1>", status_code=404)


@app.get("/test_client.html")
async def test_client():
    html_path = APP_DIR / "test_client.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return HTMLResponse("<h1>test_client.html not found</h1>", status_code=404)


@app.get("/metrics")
async def metrics():
    return {"metrics": _metric_snapshot()}
