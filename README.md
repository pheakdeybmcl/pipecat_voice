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
ws://<app_host>:8000/ws_fs?session_id=${uuid}&lang=${lang}
```

## Health
```
GET /health
```
# pipecat_voice
