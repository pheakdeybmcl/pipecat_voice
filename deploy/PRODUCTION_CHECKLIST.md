# Production Checklist

## 1) Pull and install
```bash
cd /opt/pipecat_voice
git pull
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure env
Edit `/opt/pipecat_voice/.env` and set:
```env
WS_ENFORCE_SESSION_ID=true
WS_REQUIRE_TOKEN=true
WS_AUTH_TOKEN=<strong-random-token>
WS_ALLOWED_IPS=127.0.0.1,<your-freeswitch-ip>
DEEPGRAM_LANGUAGE_HI=hi

STT_PROVIDER_KM=google
GOOGLE_STT_LANGUAGE=km-KH
GOOGLE_STT_MODEL=latest_long
GOOGLE_STT_MIN_UTTERANCE_MS=200
GOOGLE_APPLICATION_CREDENTIALS=/opt/pipecat_voice/keys/google-stt.json

TTS_VOICE_EN=en-US-JennyNeural
TTS_VOICE_VI=vi-VN-HoaiMyNeural
TTS_VOICE_KM=km-KH-SreymomNeural
TTS_VOICE_HI=hi-IN-SwaraNeural

SILENCE_HANGUP_SEC=45
VAD_SPEECH_FRAMES=4
VAD_SILENCE_FRAMES=5
END_CALL_ENABLED=false
```

## 3) Restart service
```bash
systemctl restart pipecat_voice
systemctl status pipecat_voice --no-pager -l
```

## 4) Verify app health
```bash
curl -s http://127.0.0.1:8000/health
journalctl -u pipecat_voice -n 80 --no-pager -l
```

## 5) Verify language routing in logs
Make one call per language and confirm logs:
```bash
journalctl -u pipecat_voice -n 200 --no-pager -l | egrep "WS connected|Using Deepgram STT language|Using Google STT|Selected TTS voice|Call ended"
```

Expected:
- `lang=en` -> Deepgram STT + `TTS_VOICE_EN`
- `lang=vi` -> Deepgram STT language `vi` + `TTS_VOICE_VI`
- `lang=hi` -> Deepgram STT language `hi` + `TTS_VOICE_HI`
- `lang=km` -> Google STT + `TTS_VOICE_KM`

## 6) Verify FreeSWITCH dialplan deployed
```bash
docker cp /opt/pipecat_voice/deploy/freeswitch/dialplan/default/0000000000.xml fs:/usr/local/freeswitch/etc/freeswitch/dialplan/default/0000000000.xml
docker exec fs fs_cli -p ClueCon -x "reloadxml"
```

If websocket token is enabled, set matching FreeSWITCH global:
```bash
# inside vars.xml
# <X-PRE-PROCESS cmd="set" data="global_ws_token=<same-as-WS_AUTH_TOKEN>"/>
docker exec fs fs_cli -p ClueCon -x "reloadxml"
```

## 7) Firewall (recommended)
Only expose SIP/RTP/required ports. Keep app port `8000` private if possible.
