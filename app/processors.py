from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import tempfile
import uuid
import wave
from typing import Optional

from fastapi import WebSocket
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .barge_in import BargeInState, calc_rms
from .codex_llm import detect_end_call_intent, run_codex
from .config import settings
from .edge_tts import synthesize_to_wav
from .frames import TTSBytesFrame


_YES_RE = re.compile(
    r"^(yes|yeah|yep|correct|please do|do it|hang up|end (the )?call|goodbye|"
    r"haan|haan ji|theek hai|band karo|khatm karo|"
    r"co|dong y|ket thuc|tam biet|"
    r"បាទ|ចាស|យល់ព្រម|បិទការហៅ)",
    re.IGNORECASE,
)

_NO_RE = re.compile(
    r"^(no|nope|not yet|wait|continue|keep going|don't|"
    r"nahi|abhi nahi|mat karo|"
    r"khong|chua|tiep tuc|"
    r"ទេ|មិនទាន់|បន្ត)",
    re.IGNORECASE,
)

# Clear end-call phrases: hang up immediately with farewell.
_CLEAR_END_RE = re.compile(
    r"^(bye|goodbye|hang up|end (the )?call|end call|"
    r"i am done|i'm done|im done|that'?s all|nothing else|no more questions|"
    r"tam biet|k?t thuc (cuoc goi)?|ket thuc (cuoc goi)?|toi xong roi|toi xong|"
    r"namaste|alvida|call band karo|main done hu|main done hoon|mujhe jana hai|"
    r"លាហើយ|បញ្ចប់ការហៅ|ខ្ញុំរួចរាល់ហើយ|ខ្ញុំចប់ហើយ|អស់ហើយ|"
    r"au revoir)$",
    re.IGNORECASE,
)

# Ambiguous endings: confirm before hangup.
_AMBIGUOUS_END_RE = re.compile(
    r"^(ok|okay|done|thanks|thank you|thank you so much|"
    r"cam on|ok cam on|"
    r"thik hai|dhanyavaad|shukriya|"
    r"អរគុណ|បានហើយ|អូខេ)$",
    re.IGNORECASE,
)

# If caller clearly asks to continue or asks another question, do not treat as end-call.
_CONTINUE_HINT_RE = re.compile(
    r"(one more question|another question|i still have|tell me more|explain more|"
    r"not done|not finished|continue|keep talking|"
    r"toi con cau hoi|cho toi hoi|giai thich them|tiep tuc|"
    r"mere paas (ek )?aur sawaal|abhi khatam nahi|thoda aur batao|"
    r"ខ្ញុំនៅមានសំណួរ|សូមពន្យល់បន្ថែម|មិនទាន់ចប់|បន្ត)",
    re.IGNORECASE,
)

_DEFAULT_END_CALL_CLOSE_TEXT = "Thanks for calling. Goodbye."


def _lang_code(lang: str) -> str:
    l = (lang or "").strip().lower()
    if l.startswith("km"):
        return "km"
    if l.startswith("vi"):
        return "vi"
    if l.startswith("hi"):
        return "hi"
    return "en"


def _localized_text(lang: str, key: str) -> str:
    lc = _lang_code(lang)
    table = {
        "en": {
            "continue": "Understood. We can continue.",
            "confirm_end": "Do you want me to end the call?",
            "timeout": "Sorry, I had trouble responding. Please try again.",
            "close_default": "Thanks for calling. Goodbye.",
        },
        "km": {
            "continue": "បានយល់។ យើងអាចបន្តការហៅបាន។",
            "confirm_end": "តើអ្នកចង់បញ្ចប់ការហៅឥឡូវនេះទេ?",
            "timeout": "សូមអភ័យទោស ខ្ញុំឆ្លើយតបមានបញ្ហាបន្តិច។ សូមព្យាយាមម្តងទៀត។",
            "close_default": "សូមអរគុណសម្រាប់ការហៅមកកាន់យើង។ លាហើយ។",
        },
        "vi": {
            "continue": "Da hieu. Chung ta co the tiep tuc cuoc goi.",
            "confirm_end": "Ban co muon ket thuc cuoc goi bay gio khong?",
            "timeout": "Xin loi, toi gap su co khi phan hoi. Ban vui long thu lai.",
            "close_default": "Cam on ban da goi. Tam biet.",
        },
        "hi": {
            "continue": "Samajh gaya. Hum baat jari rakh sakte hain.",
            "confirm_end": "Kya aap chahte hain ki main ab call band kar doon?",
            "timeout": "Maaf kijiye, jawab dene mein dikkat aayi. Kripya dobara koshish karein.",
            "close_default": "Call karne ke liye dhanyavaad. Namaste.",
        },
    }
    return table.get(lc, table["en"]).get(key, table["en"][key])


def _close_text_for_lang(lang: str) -> str:
    configured = settings.end_call_close_text.strip()
    if not configured or configured == _DEFAULT_END_CALL_CLOSE_TEXT:
        return _localized_text(lang, "close_default")
    return configured


def _normalize_for_match(text: str) -> str:
    # Normalize punctuation/spacing for short intent phrases.
    t = (text or "").strip().lower()
    t = re.sub(r"[.!?។၊]+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _local_end_intent(text: str) -> str:
    """Return one of: end_now, confirm, continue."""
    t = _normalize_for_match(text)
    if not t:
        return "continue"
    if _CONTINUE_HINT_RE.search(t):
        return "continue"
    if _CLEAR_END_RE.match(t):
        return "end_now"
    if _AMBIGUOUS_END_RE.match(t):
        return "confirm"
    return "continue"


class CodexLLMProcessor(FrameProcessor):
    def __init__(self, *, call_uuid: str, call_lang: str, hangup_cb):
        super().__init__(name="CodexLLM")
        self._session_id: Optional[str] = None
        self._lock = asyncio.Lock()
        self._call_uuid = call_uuid
        self._call_lang = (call_lang or "en").strip().lower()
        self._hangup_cb = hangup_cb
        self._awaiting_end_call_confirm = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if not text:
                return
            async with self._lock:
                if self._awaiting_end_call_confirm:
                    if _YES_RE.search(text):
                        self._awaiting_end_call_confirm = False
                        close_text = _close_text_for_lang(self._call_lang)
                        if close_text:
                            await self.push_frame(LLMTextFrame(text=close_text), FrameDirection.DOWNSTREAM)
                        try:
                            await self._hangup_cb(
                                delay_s=settings.end_call_hangup_delay_sec,
                                reason="end_call_confirmed",
                            )
                        except Exception:
                            pass
                        return
                    if _NO_RE.search(text):
                        self._awaiting_end_call_confirm = False
                        await self.push_frame(
                            LLMTextFrame(text=_localized_text(self._call_lang, "continue")),
                            FrameDirection.DOWNSTREAM,
                        )
                        return
                    # Caller moved on; clear confirmation mode.
                    self._awaiting_end_call_confirm = False

                local_intent = _local_end_intent(text)
                logger.info("Local end-call intent: {} text={}", local_intent, text)

                if local_intent == "end_now":
                    close_text = _close_text_for_lang(self._call_lang)
                    if close_text:
                        await self.push_frame(LLMTextFrame(text=close_text), FrameDirection.DOWNSTREAM)
                    try:
                        await self._hangup_cb(
                            delay_s=settings.end_call_hangup_delay_sec,
                            reason="end_call_local",
                        )
                    except Exception:
                        pass
                    return

                if local_intent == "confirm" and settings.end_call_confirm:
                    self._awaiting_end_call_confirm = True
                    await self.push_frame(
                        LLMTextFrame(text=_localized_text(self._call_lang, "confirm_end")),
                        FrameDirection.DOWNSTREAM,
                    )
                    return

                # Optional slower fallback classifier, disabled by default.
                if settings.end_call_enabled:
                    end_call, conf = await detect_end_call_intent(text)
                    logger.info(
                        "LLM end-call intent: end_call={} conf={} text={}",
                        end_call,
                        f"{conf:.2f}",
                        text,
                    )
                    if end_call and conf >= settings.end_call_threshold:
                        close_text = _close_text_for_lang(self._call_lang)
                        if close_text:
                            await self.push_frame(LLMTextFrame(text=close_text), FrameDirection.DOWNSTREAM)
                        try:
                            await self._hangup_cb(
                                delay_s=settings.end_call_hangup_delay_sec,
                                reason="end_call_intent",
                            )
                        except Exception:
                            pass
                        return
                    if end_call and settings.end_call_confirm and conf >= settings.end_call_confirm_threshold:
                        self._awaiting_end_call_confirm = True
                        await self.push_frame(
                            LLMTextFrame(text=_localized_text(self._call_lang, "confirm_end")),
                            FrameDirection.DOWNSTREAM,
                        )
                        return

                try:
                    reply, self._session_id = await run_codex(
                        text,
                        self._session_id,
                        call_lang=self._call_lang,
                    )
                except asyncio.TimeoutError:
                    reply = _localized_text(self._call_lang, "timeout")
                if not reply:
                    return
                await self.push_frame(LLMTextFrame(text=reply), FrameDirection.DOWNSTREAM)
        else:
            await self.push_frame(frame, direction)


class EdgeTTSProcessor(FrameProcessor):
    def __init__(self, *, audio_type: str = "raw", voice: Optional[str] = None):
        super().__init__(name="EdgeTTS")
        self._audio_type = audio_type
        self._voice = voice
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
                await synthesize_to_wav(text, tmp_path, voice=self._voice)
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
