#!/usr/bin/env python3
import asyncio
import contextlib
import shutil
import sys
from dataclasses import dataclass
from typing import Optional

import edge_tts
from openai import AsyncOpenAI

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

# ---------------- Ollama ----------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.1:8b"
client = AsyncOpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)

# ---------------- TTS -------------------
VOICE = "en-US-JennyNeural"

# ---------------- UI --------------------
STYLE = Style.from_dict(
    {
        "title": "ansicyan bold",
        "hint": "ansigray",
        "user": "ansiblue bold",
        "ai": "ansigreen bold",
        "sep": "ansigray",
        "warn": "ansiyellow",
        "err": "ansired bold",
    }
)

HELP = """\
Keys:
  Enter  send
  Esc    stop talking immediately

Commands:
  /help  show this help
  /quit  exit
"""


def ui_line(role: str, text: str) -> None:
    if role == "user":
        print_formatted_text(
            FormattedText([("class:user", "You"), ("class:sep", ": "), ("", text)]),
            style=STYLE,
        )
    else:
        print_formatted_text(
            FormattedText([("class:ai", "AI"), ("class:sep", ": "), ("", text)]),
            style=STYLE,
        )


async def ollama_stream(prompt: str):
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for event in stream:
        delta = event.choices[0].delta.content or ""
        if delta:
            yield delta


async def ollama_full(prompt: str) -> str:
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


@dataclass
class SpeakHandle:
    task: Optional[asyncio.Task] = None
    stop_event: Optional[asyncio.Event] = None
    proc: Optional[asyncio.subprocess.Process] = None

    def speaking(self) -> bool:
        return self.task is not None and not self.task.done()


class TTSSpeaker:
    """
    Streams edge-tts audio bytes into mpg123 via stdin.
    Old edge-tts compatible (no output_format needed).
    """

    def __init__(self, voice: str = VOICE):
        self.voice = voice
        self.current = SpeakHandle()

    async def interrupt(self) -> None:
        if self.current.stop_event:
            self.current.stop_event.set()

        if self.current.proc and self.current.proc.returncode is None:
            with contextlib.suppress(Exception):
                self.current.proc.terminate()

        if self.current.task and not self.current.task.done():
            self.current.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.current.task

        self.current = SpeakHandle()

    async def speak_from_queue(self, q: asyncio.Queue[str | None]) -> None:
        await self.interrupt()
        stop_event = asyncio.Event()
        task = asyncio.create_task(self._run_queue(q, stop_event))
        self.current = SpeakHandle(task=task, stop_event=stop_event, proc=None)

    async def _run_queue(self, q: asyncio.Queue[str | None], stop_event: asyncio.Event) -> None:
        """
        Consume streamed text, chunk it, speak each chunk.
        IMPORTANT: swallow exceptions so background task never explodes the app.
        """
        buf = ""
        try:
            while True:
                piece = await q.get()
                if piece is None:
                    break
                if stop_event.is_set():
                    continue

                buf += piece

                if len(buf) >= 200 or buf.endswith((".", "!", "?", "\n")):
                    chunk = buf.strip()
                    buf = ""
                    if chunk:
                        await self._speak_once(chunk, stop_event)
                        if stop_event.is_set():
                            return

            if not stop_event.is_set() and buf.strip():
                await self._speak_once(buf.strip(), stop_event)

        except asyncio.CancelledError:
            return
        except Exception:
            # Do not crash the whole program from a background task.
            return

    async def _speak_once(self, text: str, stop_event: asyncio.Event) -> None:
        if not text or stop_event.is_set():
            return

        # mpg123 reads MP3 from stdin with "-"
        # Much more stable than mpv for stdin MP3 streams.
        proc = await asyncio.create_subprocess_exec(
            "mpg123",
            "-q",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self.current.proc = proc

        # Force MP3 output for mpg123; fall back if edge-tts is older.
        try:
            tts = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                output_format="audio-24khz-48kbitrate-mono-mp3",
            )
        except TypeError:
            tts = edge_tts.Communicate(text=text, voice=self.voice)

        try:
            async for ch in tts.stream():
                if stop_event.is_set():
                    break
                if ch.get("type") != "audio":
                    continue

                if proc.returncode is not None:
                    # Player already exited
                    break

                try:
                    proc.stdin.write(ch["data"])
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    break

        finally:
            with contextlib.suppress(Exception):
                if proc.stdin:
                    proc.stdin.close()
            with contextlib.suppress(Exception):
                if stop_event.is_set():
                    proc.terminate()
            # Let mpg123 finish playback after EOF.
            with contextlib.suppress(Exception):
                await proc.wait()


async def main():
    # Requirements checks
    if shutil.which("mpg123") is None:
        print_formatted_text(
            FormattedText([("class:err", "ERROR: mpg123 not found. Install: sudo apt install -y mpg123")]),
            style=STYLE,
        )
        return

    speaker = TTSSpeaker()

    kb = KeyBindings()

    @kb.add("escape")
    def _(event):
        asyncio.get_event_loop().create_task(speaker.interrupt())
        print_formatted_text(FormattedText([("class:warn", "[stopped]")]), style=STYLE)

    session = PromptSession(
        message=FormattedText([("class:sep", "You"), ("class:sep", " › ")]),
        style=STYLE,
        key_bindings=kb,
    )

    print_formatted_text(FormattedText([("class:title", "Modern Chat + Interruptible TTS")]), style=STYLE)
    print_formatted_text(FormattedText([("class:hint", "Press ESC to stop talking. /help for help.")]), style=STYLE)
    print()

    with patch_stdout():
        while True:
            try:
                user = (await session.prompt_async()).strip()
            except (EOFError, KeyboardInterrupt):
                await speaker.interrupt()
                print()
                break

            if not user:
                continue

            if user.lower() in {"/quit", "quit", "exit"}:
                await speaker.interrupt()
                break

            if user.lower() in {"/help", "help"}:
                print_formatted_text(FormattedText([("class:hint", HELP)]), style=STYLE)
                continue

            # Stop voice when new question arrives
            await speaker.interrupt()

            ui_line("user", user)

            # Stream reply -> print + queue for TTS
            tts_q: asyncio.Queue[str | None] = asyncio.Queue()
            tts_task = asyncio.create_task(speaker.speak_from_queue(tts_q))

            # Print AI prefix (styled), then stream raw text after it
            print_formatted_text(
                FormattedText([("class:ai", "AI"), ("class:sep", ": ")]),
                style=STYLE,
                end="",
            )
            sys.stdout.flush()

            try:
                async for delta in ollama_stream(user):
                    print(delta, end="")
                    sys.stdout.flush()
                    await tts_q.put(delta)
                print()
            except Exception:
                full = await ollama_full(user)
                print(full)
                await tts_q.put(full)

            await tts_q.put(None)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await tts_task


if __name__ == "__main__":
    asyncio.run(main())
