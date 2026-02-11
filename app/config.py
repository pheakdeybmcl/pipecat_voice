from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional
    load_dotenv = None


def _load_env() -> None:
    if load_dotenv is None:
        return
    env_path = os.getenv("PIPECAT_ENV", ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=False)


_load_env()


@dataclass
class Settings:
    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Audio
    sample_rate: int = int(os.getenv("FS_SAMPLE_RATE", "16000"))
    vad_mode: int = int(os.getenv("VAD_MODE", "2"))
    vad_speech_frames: int = int(os.getenv("VAD_SPEECH_FRAMES", "3"))
    vad_silence_frames: int = int(os.getenv("VAD_SILENCE_FRAMES", "8"))
    vad_rms_threshold: float = float(os.getenv("VAD_RMS_THRESHOLD", "0.004"))
    barge_in_enabled: bool = os.getenv("BARGE_IN_ENABLED", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    silence_hangup_sec: int = int(os.getenv("SILENCE_HANGUP_SEC", "45"))

    # Deepgram STT
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    deepgram_model: str = os.getenv("DEEPGRAM_MODEL", "nova-3-general")
    deepgram_language: str = os.getenv("DEEPGRAM_LANGUAGE", "en")
    deepgram_vad_events: bool = os.getenv("DEEPGRAM_VAD_EVENTS", "true").lower() in (
        "1",
        "true",
        "yes",
    )

    # End-of-call intent (LLM-based)
    end_call_enabled: bool = os.getenv("END_CALL_ENABLED", "true").lower() in ("1", "true", "yes")
    end_call_threshold: float = float(os.getenv("END_CALL_THRESHOLD", "0.85"))
    end_call_confirm_threshold: float = float(os.getenv("END_CALL_CONFIRM_THRESHOLD", "0.65"))
    end_call_confirm: bool = os.getenv("END_CALL_CONFIRM", "true").lower() in ("1", "true", "yes")
    end_call_close_text: str = os.getenv("END_CALL_CLOSE_TEXT", "Thanks for calling. Goodbye.")
    end_call_timeout_sec: int = int(os.getenv("END_CALL_TIMEOUT_SEC", "12"))
    end_call_hangup_delay_sec: float = float(os.getenv("END_CALL_HANGUP_DELAY_SEC", "2.0"))

    # Codex LLM
    codex_cmd: str = os.getenv("CODEX_CMD", "codex")
    codex_cd: str = os.getenv("CODEX_CD", os.getcwd())
    codex_model: str = os.getenv("CODEX_MODEL", "")
    codex_profile: str = os.getenv("CODEX_PROFILE", "")
    codex_timeout_sec: int = int(os.getenv("CODEX_TIMEOUT_SEC", "60"))
    company_profile_path: str = os.getenv("COMPANY_PROFILE_PATH", "company_profile.json")
    support_only: bool = os.getenv("SUPPORT_ONLY", "true").lower() in ("1", "true", "yes")
    support_refusal: str = os.getenv(
        "SUPPORT_REFUSAL",
        "I can only help with company services, plans, pricing, or support questions.",
    )

    # Edge TTS
    tts_voice: str = os.getenv("TTS_VOICE", "en-US-JennyNeural")
    tts_format: str = os.getenv(
        "TTS_FORMAT", "riff-16khz-16bit-mono-pcm"
    )
    tts_use_ssml: bool = os.getenv("TTS_USE_SSML", "false").lower() in ("1", "true", "yes")
    tts_break_ms: int = int(os.getenv("TTS_BREAK_MS", "200"))
    welcome_enabled: bool = os.getenv("WELCOME_ENABLED", "true").lower() in ("1", "true", "yes")
    welcome_text: str = os.getenv("WELCOME_TEXT", "Welcome. How can I help you today?")

    # FreeSWITCH ESL (optional, for auto-playback of mod_audio_stream::play)
    fs_esl_enabled: bool = os.getenv("FS_ESL_ENABLED", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    fs_esl_host: str = os.getenv("FS_ESL_HOST", "127.0.0.1")
    fs_esl_port: int = int(os.getenv("FS_ESL_PORT", "8021"))
    fs_esl_password: str = os.getenv("FS_ESL_PASSWORD", "ClueCon")


settings = Settings()
