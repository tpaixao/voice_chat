# Voice Chat

Talk to your AI assistant through voice. Browser records audio, backend does STT -> LLM -> TTS, and streams the response back as MP3 audio that plays while the LLM is still generating.

## Architecture

```
Browser (MediaRecorder)  --audio-->  /api/chat
                                      |
                          Groq Whisper STT (whisper-large-v3)
                                      |
                          Nanobot SDK (run_streamed)
                            - full context: SOUL.md, USER.md, MEMORY.md
                            - tools, MCP, session memory
                                      |
                          edge_tts (sentence-by-sentence)
                                      |
                          [4-byte len][MP3 data] frames
                                      |
Browser (sequential MP3 playback)  <--
```

The backend sends a length-prefixed binary stream: each sentence is a complete, independently playable MP3 wrapped as `[4-byte big-endian length][MP3 data]`. The browser plays each chunk as it arrives, so you hear the first sentence while later ones are still being generated and TTS'd.

Keepalive silence frames (104-byte minimal MP3) are sent during tool execution to prevent the browser from dropping the connection when no text is flowing.

## Prerequisites

- **Python 3.11+**
- **[Nanobot](https://github.com/nanobot-ai/nanobot)** installed and configured (`~/.nanobot/config.json` with your model, API keys, SOUL.md, USER.md, MEMORY.md, etc.)
- **Groq API key** for Whisper STT, stored in `~/.nanobot/config.json` under `providers.groq.apiKey`
- **Ollama** (or any LLM provider) configured in nanobot for the LLM backend
- **OpenSSL** for generating a self-signed SSL certificate (browsers block `getUserMedia` on non-localhost HTTP)
- A microphone and browser with `getUserMedia` support (Chrome, Firefox, Safari)

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/tpaixao/voice_chat.git
   cd voice_chat
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install fastapi uvicorn httpx edge-tts nanobot
   ```

3. Generate a self-signed SSL certificate:
   ```bash
   openssl req -x509 -newkey rsa:2048 -keyout cert.key -out cert.pem -days 365 -nodes -subj '/CN=localhost'
   ```

4. Make sure your nanobot config is set up:
   ```bash
   # ~/.nanobot/config.json should contain:
   # - Your model configuration
   # - Groq API key under providers.groq.apiKey
   # - Any tools/MCP servers you want available
   ```

5. (Optional) Adjust the TTS voice and model in `app.py`:
   ```python
   GROQ_MODEL = "whisper-large-v3"
   TTS_VOICE = "en-US-JennyNeural"  # see edge_tts docs for available voices
   ```

## Running

```bash
./start_server.sh
```

Or manually:
```bash
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8092 \
  --ssl-keyfile cert.key --ssl-certfile cert.pem
```

Then open `https://<your-ip>:8092` in a browser. Accept the self-signed certificate warning.

## Usage

- **Tap** the button to start listening
- **Tap** again to stop recording and send (or just stop talking; silence detection auto-sends after ~1.6s)
- The AI response plays automatically
- **Tap** while speaking to interrupt and start a new query
- **Hold** the button for ~600ms to end the conversation

### Features

- Adaptive silence detection (calibrates to ambient noise in first 400ms, auto-sends after ~1.6s of silence)
- Screen wake lock (keeps the mic alive on mobile by preventing screen-off)
- Tap-to-interrupt and hold-to-end interaction model
- Sentence-level streaming (first audio plays while LLM still generating)
- Tool call keepalives (connection stays alive during long tool execution)
- Full nanobot context (SOUL.md, USER.md, MEMORY.md, tools, MCP, session memory)
- Voice mode prompt (tells the LLM to avoid markdown/emojis for cleaner TTS)

## Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Port | `start_server.sh` | 8092 |
| STT model | `app.py` | `whisper-large-v3` |
| TTS voice | `app.py` | `en-US-JennyNeural` |
| LLM timeout | `app.py` | 30s (startup), 120s (total) |
| Session key | `app.py` | `voice_chat` |
| Max recording | `index.html` | 30s |

## API Endpoints

- `POST /api/chat` - Upload audio, get streamed MP3 response
- `GET /api/health` - Health check (bot + Groq status)
- `GET /` - Frontend UI

## License

MIT
