from __future__ import annotations

import asyncio
import json
import re
import urllib.parse
from typing import Dict, Tuple, Optional

from loguru import logger


class ESLClient:
    def __init__(self, host: str, port: int, password: str) -> None:
        self._host = host
        self._port = port
        self._password = password
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
        # read auth/request
        headers, _ = await self._read_message()
        if headers.get("Content-Type") != "auth/request":
            raise RuntimeError(f"Unexpected ESL banner: {headers}")
        await self._send_cmd(f"auth {self._password}")
        headers, body = await self._read_message()
        if headers.get("Content-Type") != "command/reply":
            raise RuntimeError(f"ESL auth failed: {headers} body={body}")
        reply_text = headers.get("Reply-Text", "") + " " + (body or "")
        if "OK" not in reply_text and "+OK" not in reply_text:
            raise RuntimeError(f"ESL auth failed: {headers} body={body}")

    async def close(self) -> None:
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None

    async def _send_cmd(self, cmd: str) -> None:
        if not self._writer:
            raise RuntimeError("ESL not connected")
        self._writer.write((cmd + "\n\n").encode("utf-8"))
        await self._writer.drain()

    async def _read_message(self) -> Tuple[Dict[str, str], str]:
        if not self._reader:
            raise RuntimeError("ESL not connected")
        headers: Dict[str, str] = {}
        # read headers
        while True:
            line = await self._reader.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="ignore").rstrip("\r\n")
            if line == "":
                break
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip()] = v.strip()
        # read body
        body = ""
        content_len = int(headers.get("Content-Length", "0") or "0")
        if content_len > 0:
            data = await self._reader.readexactly(content_len)
            body = data.decode("utf-8", errors="ignore")
        return headers, body

    async def subscribe_play_events(self) -> None:
        await self._send_cmd("event plain ALL")
        headers, body = await self._read_message()
        logger.info("ESL subscribe reply: {} {}", headers.get("Reply-Text"), (body or "").strip())
        await self._send_cmd("filter Event-Subclass mod_audio_stream::play")
        headers, body = await self._read_message()
        logger.info("ESL filter reply: {} {}", headers.get("Reply-Text"), (body or "").strip())

    async def api(self, cmd: str) -> Tuple[Dict[str, str], str]:
        await self._send_cmd(f"api {cmd}")
        return await self._read_message()

    async def run_play_loop(self) -> None:
        if not self._reader:
            raise RuntimeError("ESL not connected")
        while True:
            headers, body = await self._read_message()
            if not headers:
                continue
            logger.info(
                "ESL frame: Content-Type={} Content-Length={}",
                headers.get("Content-Type"),
                headers.get("Content-Length"),
            )
            event_headers = headers
            event_body = body
            if headers.get("Content-Type") == "text/event-plain":
                event_headers, event_body = _parse_event_plain(body)
                if "Event-Subclass: mod_audio_stream::play" in (body or ""):
                    logger.info("ESL event-plain contains mod_audio_stream::play")
            if "Event-Name" in event_headers:
                subclass = event_headers.get("Event-Subclass", "")
                subclass = urllib.parse.unquote(subclass)
                logger.info("ESL event: {} {}", event_headers.get("Event-Name"), subclass)
            subclass = urllib.parse.unquote(event_headers.get("Event-Subclass", ""))
            if subclass != "mod_audio_stream::play":
                continue
            uuid = event_headers.get("Unique-ID") or event_headers.get("Caller-Unique-ID")
            if not uuid:
                continue
            file_path = _extract_file_from_body(event_body)
            if not file_path:
                for key in ("File-Path", "File", "file", "Audio-File"):
                    if key in event_headers:
                        file_path = event_headers.get(key)
                        break
            if not file_path:
                logger.warning(
                    "ESL play event missing file. Headers keys={} Body={}",
                    ",".join(sorted(event_headers.keys())),
                    event_body,
                )
                continue
            logger.info("ESL play event file: {}", file_path)
            cmd = f"uuid_broadcast {uuid} {file_path} aleg"
            _, reply = await self.api(cmd)
            logger.info("ESL uuid_broadcast: {}", reply.strip())


async def esl_api_command(host: str, port: int, password: str, cmd: str) -> str:
    client = ESLClient(host, port, password)
    await client.connect()
    try:
        _, body = await client.api(cmd)
    finally:
        await client.close()
    return (body or "").strip()


def _extract_file_from_body(body: str) -> Optional[str]:
    body = (body or "").strip()
    if not body:
        return None
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            if "file" in data:
                return data.get("file")
            stream_audio = data.get("streamAudio")
            if isinstance(stream_audio, dict):
                return stream_audio.get("file")
    except Exception:
        # Fallback: try regex
        m = re.search(r'"file"\\s*:\\s*"([^"]+)"', body)
        if m:
            return m.group(1)
        return None
    return None


def _parse_event_plain(body: str) -> Tuple[Dict[str, str], str]:
    headers: Dict[str, str] = {}
    text = body or ""
    header_text = text
    body_text = ""
    if "\n\n" in text:
        header_text, body_text = text.split("\n\n", 1)
    for line in header_text.splitlines():
        if not line.strip():
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip()] = v.strip()
    body_text = body_text.strip()
    if body_text.startswith("{"):
        return headers, body_text
    # Fallback: find first JSON object in the body
    m = re.search(r'(\{.*\})', body_text, flags=re.DOTALL)
    if m:
        return headers, m.group(1).strip()
    return headers, body_text or text


async def run_esl_autoplay(host: str, port: int, password: str) -> None:
    while True:
        try:
            client = ESLClient(host, port, password)
            await client.connect()
            await client.subscribe_play_events()
            logger.info("ESL subscribed to mod_audio_stream::play at {}:{}", host, port)
            await client.run_play_loop()
        except Exception:
            logger.exception("ESL listener error")
        await asyncio.sleep(2)
