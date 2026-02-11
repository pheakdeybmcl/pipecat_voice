# Pipecat Voice (FreeSWITCH)

Clean Pipecat-based voice server for FreeSWITCH `mod_audio_stream`.

## Stack
- STT: Deepgram
- LLM: Codex CLI
- TTS: Edge TTS

## Run (local)
```bash
cd /home/smith/Documents/FreeSwitch/pipecat_voice
python3 -m venv .venv_pipecat
source .venv_pipecat/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
```

## FreeSWITCH dialplan
Point `mod_audio_stream` to:
```
ws://<app_host>:8000/ws_fs?session_id=${uuid}&lang=${lang}&token=${ws_token}
```
If `WS_REQUIRE_TOKEN=false`, `token` can be empty.
`lang=en` uses `TTS_VOICE_EN`; `lang=vi` uses `TTS_VOICE_VI`; `lang=km` uses `TTS_VOICE_KM`.
STT routing:
- `lang=en|vi` -> Deepgram
- `lang=km` -> Google Cloud Speech (`GOOGLE_APPLICATION_CREDENTIALS` must be set)

## Health
```
GET /health
```

## Production notes
- Restrict websocket access:
  - `WS_REQUIRE_TOKEN=true`
  - `WS_AUTH_TOKEN=<strong-random-token>`
  - `WS_ALLOWED_IPS=<comma-separated trusted IPs>`
- If `WS_REQUIRE_TOKEN=true`, set FreeSWITCH global `global_ws_token` to the same value used by `WS_AUTH_TOKEN`.
- Khmer STT requires Google credentials:
  - `GOOGLE_APPLICATION_CREDENTIALS=/opt/pipecat_voice/keys/google-stt.json`
- Watch call lifecycle logs:
  - `Call ended uuid=... lang=... stt_provider=... reason=... duration_s=...`

# pipecat_voice
