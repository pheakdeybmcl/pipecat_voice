from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class GoogleSegmentSTT:
    sample_rate: int
    language_code: str
    model: str = "latest_long"
    min_utterance_ms: int = 300

    def __post_init__(self) -> None:
        try:
            from google.cloud import speech  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "google-cloud-speech is not installed. Install requirements and set GOOGLE_APPLICATION_CREDENTIALS."
            ) from exc
        self._speech = speech
        self._client = speech.SpeechClient()
        self._capturing = False
        self._chunks: list[bytes] = []
        self._min_bytes = max(1, int(self.sample_rate * (self.min_utterance_ms / 1000.0)) * 2)

    async def push(self, pcm: bytes, vad_event: Optional[str]) -> Optional[str]:
        if not pcm:
            return None

        if vad_event == "speech_start":
            self._capturing = True
            self._chunks = [pcm]
            return None

        if self._capturing:
            self._chunks.append(pcm)

        if vad_event == "speech_end" and self._capturing:
            audio = b"".join(self._chunks)
            self._capturing = False
            self._chunks = []
            if len(audio) < self._min_bytes:
                return None
            return await asyncio.to_thread(self._recognize_blocking, audio)

        return None

    def _recognize_blocking(self, audio_bytes: bytes) -> Optional[str]:
        config = self._speech.RecognitionConfig(
            encoding=self._speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
            model=self.model,
            enable_automatic_punctuation=True,
        )
        audio = self._speech.RecognitionAudio(content=audio_bytes)
        response = self._client.recognize(config=config, audio=audio)
        texts: list[str] = []
        for result in response.results:
            if result.alternatives:
                t = (result.alternatives[0].transcript or "").strip()
                if t:
                    texts.append(t)
        text = " ".join(texts).strip()
        if text:
            logger.info("Google STT({}) transcript: {}", self.language_code, text)
            return text
        return None

