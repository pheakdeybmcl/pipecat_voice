#!/usr/bin/env python3
import argparse
import asyncio
import os
import subprocess
import tempfile

import edge_tts


def _run_ffmpeg(mp3_path: str, wav_path: str, sample_rate: int) -> None:
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
            wav_path,
        ],
        check=True,
    )


async def _synthesize(text: str, voice: str, wav_path: str, sample_rate: int) -> None:
    if not text.strip():
        # Generate 200ms silence if text is empty.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r={sample_rate}:cl=mono",
                "-t",
                "0.2",
                "-acodec",
                "pcm_s16le",
                wav_path,
            ],
            check=True,
        )
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        mp3_path = os.path.join(tmpdir, "tts.mp3")
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(mp3_path)
        _run_ffmpeg(mp3_path, wav_path, sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default=os.getenv("TTS_VOICE", "en-US-JennyNeural"))
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--rate", default="0")
    parser.add_argument("--volume", default="0")
    parser.add_argument("--text", nargs="+", required=True)
    args = parser.parse_args()

    text = " ".join(args.text).strip()
    sample_rate = int(os.getenv("TTS_WAV_SAMPLE_RATE", "8000"))
    asyncio.run(_synthesize(text, args.voice, args.outfile, sample_rate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
