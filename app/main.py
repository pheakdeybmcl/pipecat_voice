from __future__ import annotations

import asyncio
import contextlib
import json
import time
import sys
import types
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket
from loguru import logger

from pipecat.frames.frames import InputAudioRawFrame, StartFrame, EndFrame, LLMTextFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
# Avoid importing Deepgram TTS (pulls websockets.asyncio) since we only use STT.
if "pipecat.services.deepgram.tts" not in sys.modules:
    sys.modules["pipecat.services.deepgram.tts"] = types.ModuleType("pipecat.services.deepgram.tts")

from pipecat.services.deepgram.stt import DeepgramSTTService as _DeepgramSTTService

try:
    from deepgram import LiveOptions
except Exception:
    LiveOptions = None  # type: ignore

from .config import settings
from .processors import CodexLLMProcessor, EdgeTTSProcessor, FSSinkProcessor
from .esl_listener import run_esl_autoplay, esl_api_command
from .barge_in import BargeInState, WebRTCBargeInVAD
from .google_stt import GoogleSegmentSTT

app = FastAPI()


# Patch Deepgram STT connect for websockets>=12 (protocol has no .response attribute).
_orig_dg_connect = _DeepgramSTTService._connect


async def _dg_connect_no_response(self, *args, **kwargs):
    try:
        return await _orig_dg_connect(self, *args, **kwargs)
    except AttributeError as exc:
        if "response" in str(exc):
            logger.warning("Deepgram websocket response headers unavailable; continuing")
            return None
        raise


_DeepgramSTTService._connect = _dg_connect_no_response
DeepgramSTTService = _DeepgramSTTService


def _client_ip(ws: WebSocket) -> str:
    if ws.client and ws.client.host:
        return ws.client.host
    return "unknown"


def _ws_reject_reason(ws: WebSocket) -> str | None:
    session_id = (ws.query_params.get("session_id", "") or "").strip()
    if settings.ws_enforce_session_id and not session_id:
        return "missing_session_id"

    if settings.ws_require_token:
        token = (ws.query_params.get("token", "") or "").strip()
        if not token or token != settings.ws_auth_token:
            return "invalid_ws_token"

    allowed = settings.allowed_ws_ip_set()
    client_ip = _client_ip(ws)
    if allowed and client_ip not in allowed:
        return f"ip_not_allowed:{client_ip}"

    return None


@app.on_event("startup")
async def _startup():
    if settings.fs_esl_enabled:
        app.state.esl_task = asyncio.create_task(
            run_esl_autoplay(settings.fs_esl_host, settings.fs_esl_port, settings.fs_esl_password)
        )


@app.on_event("shutdown")
async def _shutdown():
    task = getattr(app.state, "esl_task", None)
    if task:
        task.cancel()


def _build_stt(language: str | None = None) -> DeepgramSTTService:
    if LiveOptions is None:
        raise RuntimeError("Deepgram SDK not installed. Install deepgram-sdk.")
    if not settings.deepgram_api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set.")

    stt_language = language or settings.deepgram_language

    opts = LiveOptions(
        encoding="linear16",
        channels=1,
        sample_rate=settings.sample_rate,
        interim_results=False,
        punctuate=True,
        smart_format=True,
        vad_events=settings.deepgram_vad_events,
        model=settings.deepgram_model,
        language=stt_language,
    )

    return DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        sample_rate=settings.sample_rate,
        live_options=opts,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stt": "deepgram",
        "llm": "codex",
        "tts": "edge",
        "sample_rate": settings.sample_rate,
    }


@app.websocket("/ws_fs")
async def ws_fs(ws: WebSocket):
    reject_reason = _ws_reject_reason(ws)
    if reject_reason:
        logger.warning("WS rejected: {} from {}", reject_reason, _client_ip(ws))
        await ws.close(code=1008)
        return

    await ws.accept()
    call_start_ts = time.time()
    hangup_reason = "remote_disconnect"
    hangup_sent = False
    raw_lang = (ws.query_params.get("lang", "") or "").strip().lower()
    if raw_lang.startswith("km"):
        call_lang = "km"
    elif raw_lang.startswith("vi"):
        call_lang = "vi"
    else:
        call_lang = "en"
    logger.info("WS connected: {} lang={}", ws.client, call_lang)
    call_uuid = ws.query_params.get("session_id", "") or "unknown"
    use_google_km_stt = call_lang == "km" and settings.stt_provider_km == "google"

    try:
        deepgram_lang = "vi" if call_lang == "vi" else settings.deepgram_language
        stt = None if use_google_km_stt else _build_stt(deepgram_lang)
        if not use_google_km_stt:
            logger.info("Using Deepgram STT language: {}", deepgram_lang)
    except Exception:
        logger.exception("STT init failed")
        await ws.close()
        return

    google_km_stt = None
    if use_google_km_stt:
        try:
            google_km_stt = GoogleSegmentSTT(
                sample_rate=settings.sample_rate,
                language_code=settings.google_stt_language,
                model=settings.google_stt_model,
                min_utterance_ms=settings.google_stt_min_utterance_ms,
            )
            logger.info("Using Google STT for Khmer calls: {}", settings.google_stt_language)
        except Exception:
            logger.exception("Google STT init failed")
            await ws.close()
            return

    barge_state = BargeInState(call_uuid=call_uuid)

    async def _hangup_call(*, delay_s: float = 0.0, reason: str = "unspecified") -> None:
        nonlocal hangup_reason, hangup_sent
        if hangup_sent:
            return
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        hangup_reason = reason
        hangup_sent = True
        try:
            await esl_api_command(
                settings.fs_esl_host,
                settings.fs_esl_port,
                settings.fs_esl_password,
                f"uuid_kill {call_uuid}",
            )
        except Exception as exc:
            logger.warning("uuid_kill failed: %s", exc)

    selected_voice = settings.voice_for_lang(call_lang)
    logger.info("Selected TTS voice for {}: {}", call_lang, selected_voice)
    llm = CodexLLMProcessor(call_uuid=call_uuid, hangup_cb=_hangup_call)
    tts = EdgeTTSProcessor(audio_type="wav", voice=selected_voice)
    sink = FSSinkProcessor(ws, barge_state)

    processors = [llm, tts, sink] if stt is None else [stt, llm, tts, sink]
    pipeline = Pipeline(processors)
    params = PipelineParams(
        audio_in_sample_rate=settings.sample_rate,
        audio_out_sample_rate=settings.sample_rate,
    )
    # Disable built-in idle timeout cancellation. We use explicit silence hangup logic
    # that is aware of TTS playback and call state.
    task = PipelineTask(
        pipeline,
        params=params,
        idle_timeout_secs=None,
        cancel_on_idle_timeout=False,
    )
    runner = PipelineRunner()
    runner_task = asyncio.create_task(runner.run(task))

    await task.queue_frame(StartFrame(audio_in_sample_rate=settings.sample_rate))
    if settings.welcome_enabled and settings.welcome_text.strip():
        await task.queue_frame(LLMTextFrame(text=settings.welcome_text.strip()))

    last_audio_ts = time.time()

    async def _silence_keepalive() -> None:
        if stt is None:
            return
        silence_bytes = b"\x00" * int(settings.sample_rate * 0.02) * 2
        while True:
            await asyncio.sleep(1.0)
            if time.time() - last_audio_ts > 4.0:
                frame = InputAudioRawFrame(
                    audio=silence_bytes,
                    sample_rate=settings.sample_rate,
                    num_channels=1,
                )
                await task.queue_frame(frame)

    keepalive_task = asyncio.create_task(_silence_keepalive())

    vad = None
    if settings.barge_in_enabled:
        try:
            vad = WebRTCBargeInVAD(
                settings.sample_rate,
                settings.vad_mode,
                settings.vad_speech_frames,
                settings.vad_silence_frames,
                settings.vad_rms_threshold,
            )
            vad.touch()
        except Exception as exc:
            logger.warning("Barge-in VAD disabled: {}", exc)
            vad = None

    async def _silence_hangup_watch() -> None:
        if settings.silence_hangup_sec <= 0 or vad is None:
            return
        while True:
            await asyncio.sleep(1.0)
            if barge_state.tts_playing:
                continue
            last_activity_ts = max(vad.last_speech_ts, barge_state.tts_until)
            if time.time() - last_activity_ts > settings.silence_hangup_sec:
                logger.info("Silence timeout reached, hanging up {}", call_uuid)
                await _hangup_call(reason="silence_timeout")
                break

    silence_task = asyncio.create_task(_silence_hangup_watch())

    try:
        while True:
            msg = await ws.receive()
            msg_type = msg.get("type")
            if msg_type == "websocket.disconnect":
                break
            if msg_type != "websocket.receive":
                continue

            data_bytes = msg.get("bytes")
            data_text = msg.get("text")

            if data_bytes:
                last_audio_ts = time.time()
                evt = None
                if vad is not None:
                    evt = vad.push(data_bytes)
                    if evt == "speech_start" and barge_state.tts_playing:
                        logger.info("Barge-in detected, stopping TTS for {}", call_uuid)
                        barge_state.stop_tts()
                        try:
                            reply = await esl_api_command(
                                settings.fs_esl_host,
                                settings.fs_esl_port,
                                settings.fs_esl_password,
                                f"uuid_break {call_uuid} all",
                            )
                            logger.info("uuid_break reply: {}", reply)
                        except Exception as exc:
                            logger.warning("uuid_break failed: %s", exc)
                if google_km_stt is not None:
                    if vad is None:
                        logger.warning("Google Khmer STT requires VAD; skipping audio chunk")
                        continue
                    transcript = await google_km_stt.push(data_bytes, evt)
                    if transcript:
                        ts = datetime.now(timezone.utc).isoformat()
                        await task.queue_frame(
                            TranscriptionFrame(
                                text=transcript,
                                user_id=call_uuid,
                                timestamp=ts,
                            )
                        )
                else:
                    frame = InputAudioRawFrame(
                        audio=data_bytes,
                        sample_rate=settings.sample_rate,
                        num_channels=1,
                    )
                    await task.queue_frame(frame)
            elif data_text:
                # Ignore ping/metadata from FreeSWITCH
                try:
                    payload = json.loads(data_text)
                    if payload.get("type") == "ping":
                        continue
                except json.JSONDecodeError:
                    pass
    except Exception as exc:
        logger.error("ws error: %s", exc)
        hangup_reason = "ws_error"
    finally:
        try:
            await task.queue_frame(EndFrame())
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        await asyncio.sleep(0)
        keepalive_task.cancel()
        silence_task.cancel()
        runner_task.cancel()
        with contextlib.suppress(Exception):
            await runner_task
        call_duration_s = round(time.time() - call_start_ts, 2)
        provider = "google" if use_google_km_stt else "deepgram"
        logger.info(
            "Call ended uuid={} lang={} stt_provider={} reason={} duration_s={}",
            call_uuid,
            call_lang,
            provider,
            hangup_reason,
            call_duration_s,
        )
