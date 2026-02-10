from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional

import edge_tts
from loguru import logger

from .config import settings


async def synthesize_to_wav(text: str, path: str, voice: Optional[str] = None) -> None:
    if not text.strip():
        raise ValueError("empty text")
    voice = voice or settings.tts_voice
    sample_rate = settings.sample_rate
    with tempfile.TemporaryDirectory() as tmpdir:
        mp3_path = os.path.join(tmpdir, "tts.mp3")
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(mp3_path)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                mp3_path,
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                path,
            ],
            check=True,
        )
    logger.info("edge-tts saved wav: {}", path)
