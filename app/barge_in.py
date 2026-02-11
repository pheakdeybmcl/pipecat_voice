from __future__ import annotations

import time
from dataclasses import dataclass

try:
    import webrtcvad
except Exception:
    webrtcvad = None  # type: ignore


def calc_rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    if len(pcm) % 2:
        pcm = pcm[:-1]
    import struct

    count = len(pcm) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack("<%dh" % count, pcm)
    acc = 0.0
    for s in samples:
        acc += float(s) * float(s)
    return (acc / count) ** 0.5 / 32768.0


@dataclass
class BargeInState:
    call_uuid: str
    tts_playing: bool = False
    mute_until: float = 0.0

    def set_tts_playing(self, duration_s: float) -> None:
        self.tts_playing = True
        self.mute_until = time.time() + max(0.0, duration_s)

    def stop_tts(self, cooloff_s: float = 0.8) -> None:
        self.tts_playing = False
        self.mute_until = time.time() + max(0.0, cooloff_s)

    def should_mute(self) -> bool:
        return time.time() < self.mute_until


class WebRTCBargeInVAD:
    def __init__(self, sample_rate: int, mode: int, speech_frames: int, silence_frames: int, rms_threshold: float):
        if webrtcvad is None:
            raise RuntimeError("webrtcvad not installed")
        self._vad = webrtcvad.Vad(mode)
        self._sr = sample_rate
        self._frame_bytes = int(sample_rate * 0.02) * 2  # 20ms
        self._speech_frames_req = max(1, speech_frames)
        self._silence_frames_req = max(1, silence_frames)
        self._rms_threshold = max(0.0, rms_threshold)
        self._buf = bytearray()
        self._speech_count = 0
        self._silence_count = 0
        self._in_speech = False
        self._last_speech_ts = time.time()

    def touch(self) -> None:
        self._last_speech_ts = time.time()

    @property
    def last_speech_ts(self) -> float:
        return self._last_speech_ts

    def push(self, pcm: bytes) -> str | None:
        if not pcm:
            return None
        self._buf.extend(pcm)
        event: str | None = None
        while len(self._buf) >= self._frame_bytes:
            frame = bytes(self._buf[: self._frame_bytes])
            del self._buf[: self._frame_bytes]
            if len(frame) < self._frame_bytes:
                continue
            speech = self._vad.is_speech(frame, self._sr)
            rms = calc_rms(frame)
            if speech and rms >= self._rms_threshold:
                self._speech_count += 1
                self._silence_count = 0
                self._last_speech_ts = time.time()
            else:
                self._silence_count += 1
                self._speech_count = 0

            if not self._in_speech and self._speech_count >= self._speech_frames_req:
                self._in_speech = True
                event = "speech_start"
            elif self._in_speech and self._silence_count >= self._silence_frames_req:
                self._in_speech = False
                event = "speech_end"
        return event
