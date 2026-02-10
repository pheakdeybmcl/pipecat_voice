# voice_engine_server.py
# Realtime WS voice engine with:
# - VAD (webrtcvad) utterance segmentation
# - STT (faster-whisper)
# - LLM (Ollama OpenAI-compatible /v1)
# - (Optional) Server-side TTS streaming via edge_tts
# - FIX: heavy work moved off the websocket receive loop to prevent ping timeouts

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from starlette.websockets import WebSocketDisconnect
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
import edge_tts

# ---------------- Config ----------------
# Ollama (OpenAI-compatible)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.1:8b"
llm_client = AsyncOpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
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
    "- Only call tools when needed; otherwise reply normally.\n"
    "\n"
    "Tool usage:\n"
    "- If the user asks to turn the app toggle on/off, call set_toggle with enabled true/false.\n"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_toggle",
            "description": "Turn the app toggle on or off.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"}
                },
                "required": ["enabled"],
            },
        },
    }
]

# Audio input expected from browser: PCM16 little-endian, 16kHz, mono
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes per sample (PCM16)
FRAME_MS = 20     # VAD supports only 10/20/30ms
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH  # 640 bytes

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

# Whisper
WHISPER_SIZE = "small"  # "tiny" faster

print("[boot] Loading Whisper model...")
whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
print("[boot] Whisper loaded.")

app = FastAPI()
APP_DIR = __import__("pathlib").Path(__file__).parent


@dataclass
class SessionState:
    speaking_task: Optional[asyncio.Task] = None
    llm_task: Optional[asyncio.Task] = None
    worker_task: Optional[asyncio.Task] = None
    interrupted: bool = False
    voice: str = DEFAULT_TTS_VOICE
    ui_toggle_enabled: bool = False


@dataclass
class UtteranceJob:
    pcm16: Optional[bytes] = None
    text: Optional[str] = None


def pcm16_bytes_to_float32(pcm: bytes) -> np.ndarray:
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    a /= 32768.0
    return a


def stt_whisper_from_pcm16(pcm: bytes) -> str:
    audio = pcm16_bytes_to_float32(pcm)
    segments, _info = whisper.transcribe(audio, language="en")
    text = "".join(seg.text for seg in segments).strip()
    return text


async def ollama_reply_with_tools(ws: WebSocket, state: SessionState, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    resp = await llm_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        assistant_msg = {"role": "assistant", "content": msg.content, "tool_calls": []}
        for call in tool_calls:
            assistant_msg["tool_calls"].append({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            })
        messages.append(assistant_msg)

        for call in tool_calls:
            if call.function.name != "set_toggle":
                continue
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            enabled = bool(args.get("enabled", False))
            await apply_toggle(ws, state, enabled, source="ai")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps({"enabled": state.ui_toggle_enabled}),
            })

        resp2 = await llm_client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        return resp2.choices[0].message.content or ""

    return msg.content or ""


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


async def apply_toggle(ws: WebSocket, state: SessionState, enabled: bool, source: str = "ai"):
    state.ui_toggle_enabled = bool(enabled)
    await send_json(ws, {"type": "toggle_state", "enabled": state.ui_toggle_enabled, "source": source})


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
        ok = await send_json(ws, {"type": "audio_start", "format": "mp3"})
        if not ok:
            return

        async for mp3_chunk in stream_edge_tts_mp3(text, state.voice):
            if state.interrupted:
                break
            b64 = base64.b64encode(mp3_chunk).decode("ascii")
            ok = await send_json(ws, {"type": "audio_chunk", "format": "mp3", "data": b64})
            if not ok:
                return

        await send_json(ws, {"type": "audio_end"})
        await send_turn_state(ws, "listening")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        await send_json(ws, {"type": "error", "message": f"TTS error: {e}"})


async def run_llm_and_maybe_tts(ws: WebSocket, state: SessionState, user_text: str):
    """
    LLM reply then (optional) server-side TTS.
    """
    try:
        reply = await ollama_reply_with_tools(ws, state, user_text)
        state.speaking_task = asyncio.create_task(speak_to_client(ws, state, reply))
        await state.speaking_task
    except asyncio.CancelledError:
        raise
    except Exception as e:
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

    # STT off-thread (CPU)
    try:
        text = await asyncio.to_thread(stt_whisper_from_pcm16, utterance_pcm16)
    except Exception as e:
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

    state = SessionState()

    # FIX: utterance queue + worker
    utter_q: asyncio.Queue[Optional[UtteranceJob]] = asyncio.Queue(maxsize=2)
    state.worker_task = asyncio.create_task(utterance_worker(ws, state, utter_q))

    pcm_buffer = bytearray()
    speech_buffer = bytearray()
    in_speech = False
    silence_frames = 0
    max_utt_bytes = int(MAX_UTTERANCE_SECONDS * SAMPLE_RATE * SAMPLE_WIDTH)

    await send_json(ws, {
        "type": "ready",
        "sample_rate": SAMPLE_RATE,
        "frame_ms": FRAME_MS,
        "frame_bytes": FRAME_BYTES,
        "format_in": "pcm16le",
        "server_tts_enabled": SERVER_TTS_ENABLED,
        "voice": state.voice,
        "toggle_enabled": state.ui_toggle_enabled,
    })
    asyncio.create_task(send_voice_list(ws))
    await apply_toggle(ws, state, state.ui_toggle_enabled, source="server")
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
                    await send_json(ws, {"type": "reset_ok"})
                    continue

                if mtype == "text_utterance":
                    await send_turn_state(ws, "processing")
                    await enqueue_utterance(utter_q, UtteranceJob(text=data.get("text", "")))
                    continue

                if mtype == "set_toggle":
                    await apply_toggle(ws, state, bool(data.get("enabled", False)), source="client")
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
