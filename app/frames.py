from __future__ import annotations

from dataclasses import dataclass

from pipecat.frames.frames import DataFrame


@dataclass
class TTSBytesFrame(DataFrame):
    audio: bytes
    audio_type: str = "raw"
    sample_rate: int = 16000

    def __str__(self) -> str:
        return f"{self.name}(bytes={len(self.audio)}, type={self.audio_type}, sr={self.sample_rate})"
