import asyncio
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
import edge_tts

# ---------------- Ollama ----------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.1:8b"
client = AsyncOpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)

# ---------------- Audio -----------------
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5  # simple + reliable

# ---------------- Whisper STT -----------
WHISPER_SIZE = "small"  # try "tiny" if you want faster

print("[1/3] Loading Whisper model (first run may download files)...")
whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
print("[2/3] Whisper loaded.")

def record_fixed(seconds: int = RECORD_SECONDS) -> np.ndarray:
    print(f"Recording for {seconds} seconds... Speak now.")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    return audio.reshape(-1)

def stt_whisper(audio: np.ndarray) -> str:
    segments, info = whisper.transcribe(audio, language="en")
    text = "".join(seg.text for seg in segments).strip()
    return text

async def ollama_reply(prompt: str) -> str:
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""

async def text_to_mp3(text: str, out_path: Path) -> None:
    tts = edge_tts.Communicate(text=text, voice="en-US-JennyNeural")
    await tts.save(str(out_path))

async def main() -> None:
    print("[3/3] Ready. Press Enter to record. Type 'quit' to exit.")
    while True:
        cmd = input("\nPress Enter to record (or type quit): ").strip()
        if cmd.lower() in {"quit", "exit"}:
            break

        audio = record_fixed(RECORD_SECONDS)
        prompt = stt_whisper(audio)

        if not prompt:
            print("STT: (didn't catch that) Try again.")
            continue

        print(f"STT: {prompt}")

        reply = await ollama_reply(prompt)
        print(f"AI: {reply}")

        out_file = Path(f"reply_{int(time.time())}.mp3")
        await text_to_mp3(reply, out_file)
        print(f"Saved MP3 -> {out_file.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())

