from __future__ import annotations

import asyncio
import contextlib
import json
import time
import sys
import types

from fastapi import FastAPI, WebSocket
from loguru import logger

from pipecat.frames.frames import InputAudioRawFrame, StartFrame, EndFrame, LLMTextFrame
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


def _build_stt() -> DeepgramSTTService:
    if LiveOptions is None:
        raise RuntimeError("Deepgram SDK not installed. Install deepgram-sdk.")
    if not settings.deepgram_api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set.")

    opts = LiveOptions(
        encoding="linear16",
        channels=1,
        sample_rate=settings.sample_rate,
        interim_results=False,
        punctuate=True,
        smart_format=True,
        vad_events=settings.deepgram_vad_events,
        model=settings.deepgram_model,
        language=settings.deepgram_language,
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
    await ws.accept()
    logger.info("WS connected: {}", ws.client)
    call_uuid = ws.query_params.get("session_id", "") or "unknown"

    try:
        stt = _build_stt()
    except Exception:
        logger.exception("STT init failed")
        await ws.close()
        return

    barge_state = BargeInState(call_uuid=call_uuid)
    llm = CodexLLMProcessor()
    tts = EdgeTTSProcessor(audio_type="wav")
    sink = FSSinkProcessor(ws, barge_state)

    pipeline = Pipeline([stt, llm, tts, sink])
    params = PipelineParams(
        audio_in_sample_rate=settings.sample_rate,
        audio_out_sample_rate=settings.sample_rate,
    )
    task = PipelineTask(pipeline, params=params)
    runner = PipelineRunner()
    runner_task = asyncio.create_task(runner.run(task))

    await task.queue_frame(StartFrame(audio_in_sample_rate=settings.sample_rate))
    if settings.welcome_enabled and settings.welcome_text.strip():
        await task.queue_frame(LLMTextFrame(text=settings.welcome_text.strip()))

    last_audio_ts = time.time()

    async def _silence_keepalive() -> None:
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
        except Exception as exc:
            logger.warning("Barge-in VAD disabled: {}", exc)
            vad = None

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
        runner_task.cancel()
        with contextlib.suppress(Exception):
            await runner_task
