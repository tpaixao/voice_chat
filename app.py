#!/usr/bin/env python3
"""Voice chat web app: talk to nanobot through voice.

Architecture:
  Browser records audio (MediaRecorder) -> uploads to /api/chat
  Backend: Groq Whisper STT -> Nanobot SDK (full context) -> edge_tts -> returns MP3
  Browser plays MP3 -> auto-starts listening again
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import struct
import time

import httpx
import edge_tts
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Voice Chat")

# Globals initialized at startup
bot = None
groq_api_key: str | None = None

GROQ_MODEL = "whisper-large-v3"
TTS_VOICE = "en-US-JennyNeural"
SESSION_KEY = "voice_chat"
LLM_TIMEOUT = 120  # seconds


@app.on_event("startup")
async def startup() -> None:
    global bot, groq_api_key

    # Load nanobot SDK with full config (SOUL.md, USER.md, MEMORY.md, tools)
    # but tailored for voice: no stealth browser MCP (cuts ~5.7s cold start and
    # 97 tool schemas from the prompt) and a fast preset (minimax3).
    from nanobot import Nanobot
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.hooks import create_file_edit_activity_hook
    from nanobot.config.loader import load_config, resolve_config_env_vars
    from nanobot.providers.image_generation import image_gen_provider_configs

    model_preset = os.environ.get("VOICE_MODEL_PRESET", "minimax3")
    exclude_mcp = os.environ.get(
        "VOICE_EXCLUDE_MCP", "stealth-browser-mcp"
    ).split(",")
    exclude_mcp = [n.strip().lower() for n in exclude_mcp if n.strip()]

    print(f"[voice-chat] Loading nanobot (preset={model_preset}, excluded MCP={exclude_mcp})...")
    config = resolve_config_env_vars(load_config(None))
    removed = [
        name
        for name in config.tools.mcp_servers
        if any(name.lower() == pat or pat in name.lower() for pat in exclude_mcp)
    ]
    for name in removed:
        config.tools.mcp_servers.pop(name)
    if removed:
        print(f"[voice-chat] Removed MCP servers: {removed}")
    config.agents.defaults.model_preset = model_preset

    loop = AgentLoop.from_config(
        config,
        image_generation_provider_configs=image_gen_provider_configs(config),
        hook_factories=[create_file_edit_activity_hook],
    )
    bot = Nanobot(loop, config=config)
    print(f"[voice-chat] Nanobot loaded. Model: {bot._loop.model}")

    # Read Groq API key from nanobot config
    config_path = os.path.expanduser("~/.nanobot/config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
        groq_api_key = config.get("providers", {}).get("groq", {}).get("apiKey")
    except Exception as e:
        print(f"[voice-chat] WARNING: Could not read Groq key: {e}")

    if groq_api_key:
        print("[voice-chat] Groq API key loaded.")
    else:
        print("[voice-chat] WARNING: No Groq API key found!")


async def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using Groq Whisper API."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "User-Agent": "voice-chat/1.0",
            },
            files={"file": ("audio.webm", audio_bytes, "audio/webm")},
            data={"model": GROQ_MODEL},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()


async def tts_stream(text: str, timeout: float = 30.0):
    """Yield MP3 audio chunks from edge_tts for a given text.

    Raises asyncio.TimeoutError if TTS takes too long.
    """
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


async def tts_stream_safe(text: str):
    """Like tts_stream but never raises: logs errors and yields nothing on failure."""
    try:
        async for chunk in tts_stream(text):
            yield chunk
    except Exception as e:
        print(f"[voice-chat] TTS failed for {len(text)} chars: {type(e).__name__}: {e}")


# Minimal valid MP3 frame of silence (MPEG1 Layer3, 32kbps, 44100Hz, mono, 104 bytes)
# Used as keepalive data to prevent browser connection timeout during long LLM responses
_MP3_SILENCE = bytes([0xFF, 0xFB, 0x50, 0xC4]) + bytes(100)

_SENTENCE_END = re.compile(r"[.!?;:\n]")
_COMMA = re.compile(r"[,]")
# Flush a chunk after this many chars even without punctuation (reduces TTFB)
_SOFT_LIMIT = 60
# Hard limit: never let a chunk exceed this (TTS quality degrades on very long input)
_HARD_LIMIT = 120

def split_sentences(text: str):
    """Split text into TTS-sized chunks, returning (chunks, remainder).

    Tries sentence-ending punctuation first, then commas, then hard-splits
    on length so the first audio reaches the browser ASAP.
    """
    chunks = []
    buf = text
    while True:
        if not buf:
            break
        # Try sentence-ending punctuation first
        m = _SENTENCE_END.search(buf)
        if m and m.end() <= _HARD_LIMIT:
            chunks.append(buf[: m.end()])
            buf = buf[m.end():]
            continue
        # If buffer is over soft limit, try splitting on comma
        if len(buf) > _SOFT_LIMIT:
            m2 = _COMMA.search(buf, _SOFT_LIMIT)
            if m2:
                chunks.append(buf[: m2.end()])
                buf = buf[m2.end():]
                continue
            # Hard split at hard limit if no punctuation at all
            if len(buf) > _HARD_LIMIT:
                # Try to split at a space to avoid cutting words
                cut = buf.rfind(" ", _SOFT_LIMIT, _HARD_LIMIT)
                if cut == -1:
                    cut = _HARD_LIMIT
                chunks.append(buf[:cut])
                buf = buf[cut:].lstrip()
                continue
        # Not enough text yet; wait for more
        break
    return chunks, buf


async def tts_to_bytes(text: str) -> bytes:
    """Generate complete MP3 bytes for a sentence. Catches per-sentence TTS errors."""
    try:
        data = b""
        async for chunk in tts_stream(text):
            data += chunk
        return data
    except Exception as e:
        print(f"[voice-chat] TTS error: {type(e).__name__}: {e}")
        return b""


async def stream_response(transcript: str):
    """Stream: LLM text deltas -> sentence-split -> TTS per sentence -> length-prefixed MP3.

    Each sentence's complete MP3 is wrapped as [4-byte big-endian length][MP3 data].
    The browser parses these frames and plays each MP3 as a separate audio element,
    so audio starts playing as soon as the first sentence is ready.

    Sends keepalive silence frames while waiting for the first LLM token to prevent
    the browser from dropping the connection during long thinking time.
    """
    sentence_buf = ""
    first_token_received = False

    def silence_frame() -> bytes:
        """A length-prefixed MP3 silence chunk (keepalive)."""
        return struct.pack(">I", len(_MP3_SILENCE)) + _MP3_SILENCE

    voice_prefix = (
        "[Voice mode] You are responding via text-to-speech. "
        "Do not use markdown, emojis, emoticons, or any special formatting. "
        "Respond in natural spoken language only.\n\n"
    )

    try:
        run_stream = await asyncio.wait_for(
            bot.run_streamed(
                voice_prefix + transcript,
                session_key=SESSION_KEY,
                channel="web",
                chat_id="voice",
                sender_id="tiago",
            ),
            timeout=30,
        )
    except asyncio.TimeoutError:
        print("[voice-chat] LLM startup timed out (30s)")
        mp3_data = await tts_to_bytes("Sorry, the AI took too long to start. Please try again.")
        if mp3_data:
            yield struct.pack(">I", len(mp3_data)) + mp3_data
        return
    except Exception as e:
        print(f"[voice-chat] run_streamed failed: {type(e).__name__}: {e}")
        mp3_data = await tts_to_bytes("Sorry, I could not connect to the AI.")
        if mp3_data:
            yield struct.pack(">I", len(mp3_data)) + mp3_data
        return

    tool_count = 0
    try:
        # Send an initial keepalive so the browser knows the connection is alive
        yield silence_frame()

        async for event in run_stream.stream_events():
            if event.type == "text.delta":
                if not first_token_received:
                    first_token_received = True
                sentence_buf += event.delta
                sentences, sentence_buf = split_sentences(sentence_buf)
                for s in sentences:
                    mp3_data = await tts_to_bytes(s)
                    if mp3_data:
                        yield struct.pack(">I", len(mp3_data)) + mp3_data
            elif event.type == "tool.started":
                tool_count += 1
                tool_name = event.name or "tool"
                print(f"[voice-chat] tool started: {tool_name}")
                # Send keepalive so the browser sees data during tool execution
                yield silence_frame()
            elif event.type == "tool.completed":
                # Send keepalive when tool finishes too
                yield silence_frame()
            elif event.type == "tool.failed":
                # Send keepalive on failure as well
                yield silence_frame()
            elif event.type == "run.completed":
                # Flush any remaining text
                if sentence_buf.strip():
                    mp3_data = await tts_to_bytes(sentence_buf)
                    if mp3_data:
                        yield struct.pack(">I", len(mp3_data)) + mp3_data
            elif event.type == "run.failed":
                # TTS the error so user hears something
                err_text = event.error or "Sorry, something went wrong."
                mp3_data = await tts_to_bytes(err_text)
                if mp3_data:
                    yield struct.pack(">I", len(mp3_data)) + mp3_data
    except Exception as e:
        print(f"[voice-chat] stream error: {type(e).__name__}: {e}")
        mp3_data = await tts_to_bytes("Sorry, I had an error.")
        if mp3_data:
            yield struct.pack(">I", len(mp3_data)) + mp3_data
    finally:
        try:
            await run_stream.aclose()
        except Exception:
            pass


@app.post("/api/chat")
async def chat(audio: UploadFile = File(...)) -> JSONResponse:
    timing: dict = {}
    t0 = time.time()

    # 1. Read uploaded audio
    audio_data = await audio.read()
    timing["size"] = len(audio_data)

    # 2. Transcribe with Groq Whisper
    t1 = time.time()
    try:
        transcript = await transcribe_audio(audio_data)
    except Exception as e:
        return JSONResponse(
            {"error": f"Transcription failed: {e}"}, status_code=500
        )
    timing["stt"] = round(time.time() - t1, 2)

    if not transcript:
        return JSONResponse({"error": "No speech detected", "timing": timing})

    # 3. Stream LLM + TTS as MP3 chunks
    t2 = time.time()
    print(f"[voice-chat] stt={timing['stt']}s | Q: {transcript[:80]}")

    from urllib.parse import quote
    headers = {
        "X-Transcript": quote(transcript),
        "X-TTFB": str(round(time.time() - t0, 2)),
        "Access-Control-Expose-Headers": "X-Transcript, X-TTFB",
    }
    return StreamingResponse(
        stream_response(transcript),
        media_type="application/octet-stream",
        headers=headers,
    )


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "bot_ready": bot is not None,
        "groq_ready": groq_api_key is not None,
    }


# Serve frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
