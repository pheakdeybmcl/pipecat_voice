from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import uuid
import wave
import re
from typing import Optional

from loguru import logger
from fastapi import WebSocket

from pipecat.frames.frames import (
    Frame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .codex_llm import run_codex, detect_end_call_intent
from .edge_tts import synthesize_to_wav
from .frames import TTSBytesFrame
from .barge_in import BargeInState, calc_rms
from .config import settings


class CodexLLMProcessor(FrameProcessor):
    def __init__(self, *, call_uuid: str, hangup_cb):
        super().__init__(name="CodexLLM")
        self._session_id: Optional[str] = None
        self._lock = asyncio.Lock()
        self._call_uuid = call_uuid
        self._hangup_cb = hangup_cb

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if not text:
                return
            async with self._lock:
                end_call, conf = await detect_end_call_intent(text)
                logger.info(
                    "End-call intent: end_call={} conf={} text={}",
                    end_call,
                    f"{conf:.2f}",
                    text,
                )
                if end_call and conf >= settings.end_call_threshold:
                    close_text = settings.end_call_close_text.strip()
                    if close_text:
                        await self.push_frame(LLMTextFrame(text=close_text), FrameDirection.DOWNSTREAM)
                    try:
                        await self._hangup_cb(delay_s=settings.end_call_hangup_delay_sec)
                    except Exception:
                        pass
                    return
                if end_call and settings.end_call_confirm and conf >= settings.end_call_confirm_threshold:
                    await self.push_frame(
                        LLMTextFrame(text="Do you want me to end the call?"),
                        FrameDirection.DOWNSTREAM,
                    )
                    return
                try:
                    reply, self._session_id = await run_codex(text, self._session_id)
                except asyncio.TimeoutError:
                    reply = "Sorry, I had trouble responding. Please try again."
                if not reply:
                    return
                await self.push_frame(LLMTextFrame(text=reply), FrameDirection.DOWNSTREAM)
        else:
            await self.push_frame(frame, direction)


class EdgeTTSProcessor(FrameProcessor):
    def __init__(self, *, audio_type: str = "raw"):
        super().__init__(name="EdgeTTS")
        self._audio_type = audio_type
        self._lock = asyncio.Lock()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            text = (frame.text or "").strip()
            if not text:
                return
            if settings.tts_use_ssml:
                text = _to_ssml(text, break_ms=settings.tts_break_ms)
            async with self._lock:
                tmp_path = self._alloc_temp_path()
                await synthesize_to_wav(text, tmp_path)
                try:
                    with wave.open(tmp_path, "rb") as wf:
                        sample_rate = wf.getframerate()
                    with open(tmp_path, "rb") as f:
                        wav_bytes = f.read()
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                if not wav_bytes:
                    return
                await self.push_frame(
                    TTSBytesFrame(
                        audio=wav_bytes,
                        audio_type=self._audio_type,
                        sample_rate=sample_rate,
                    ),
                    FrameDirection.DOWNSTREAM,
                )
        else:
            await self.push_frame(frame, direction)

    def _alloc_temp_path(self) -> str:
        tmp_dir = tempfile.gettempdir()
        name = f"tts_{uuid.uuid4().hex}.wav"
        path = os.path.join(tmp_dir, name)
        return path


def _to_ssml(text: str, break_ms: int = 200) -> str:
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Insert short pauses after punctuation for natural cadence
    pause = f"<break time='{max(50, break_ms)}ms'/>"
    safe = re.sub(r"([.!?])\s*", r"\\1 " + pause + " ", safe)
    return f"<speak>{safe}</speak>"


class FSSinkProcessor(FrameProcessor):
    def __init__(self, ws: WebSocket, barge_state: BargeInState):
        super().__init__(name="FSSink")
        self._ws = ws
        self._barge_state = barge_state
        self._clear_task: Optional[asyncio.Task] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSBytesFrame):
            if self._barge_state.should_mute():
                logger.info("Barge-in mute active, dropping TTS frame")
                return
            duration_s = 0.0
            if frame.sample_rate:
                duration_s = max(0.0, len(frame.audio) / 2 / frame.sample_rate)
            self._barge_state.set_tts_playing(duration_s)
            if self._clear_task:
                self._clear_task.cancel()
            self._clear_task = asyncio.create_task(self._clear_after(duration_s))
            await _send_stream_audio(
                self._ws,
                frame.audio,
                audio_type=frame.audio_type,
                sample_rate=frame.sample_rate,
                channels=1,
            )
        else:
            await self.push_frame(frame, direction)

    async def _clear_after(self, duration_s: float) -> None:
        try:
            await asyncio.sleep(max(0.0, duration_s))
        except Exception:
            return
        self._barge_state.tts_playing = False


async def _send_stream_audio(
    ws: WebSocket,
    pcm: bytes,
    *,
    audio_type: str,
    sample_rate: int,
    channels: int,
) -> None:
    if not pcm:
        return
    logger.info("FS streamAudio send audio_type={} bytes={} sr={} ch={}", audio_type, len(pcm), sample_rate, channels)
    # For WAV, send as a single payload (keeps RIFF header intact)
    if audio_type == "wav":
        b64 = base64.b64encode(pcm).decode("ascii")
        payload = {
            "type": "streamAudio",
            "data": {
                "audioDataType": audio_type,
                "audioData": b64,
            },
        }
        try:
            await ws.send_text(json.dumps(payload))
        except Exception as exc:
            logger.error("FS send failed: %s", exc)
        return

    if len(pcm) % 2:
        pcm = pcm[:-1]

    frame_ms = 20
    bytes_per_sample = 2
    chunk_size = int(sample_rate * frame_ms / 1000) * bytes_per_sample * channels
    if chunk_size <= 0:
        chunk_size = 640

    for i in range(0, len(pcm), chunk_size):
        chunk = pcm[i : i + chunk_size]
        if not chunk:
            continue
        b64 = base64.b64encode(chunk).decode("ascii")
        payload = {
            "type": "streamAudio",
            "data": {
                "audioDataType": audio_type,
                "sampleRate": sample_rate,
                "audioData": b64,
            },
        }
        try:
            await ws.send_text(json.dumps(payload))
        except Exception as exc:
            logger.error("FS send failed: %s", exc)
            break
        await asyncio.sleep(frame_ms / 1000.0)
