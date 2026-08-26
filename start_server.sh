#!/bin/bash
# Start voice chat server as a detached daemon that survives exec session termination
cd /mnt/ssd1tb/nanobot/workspace/projects/voice_chat

# Kill any existing instance
lsof -ti:8092 2>/dev/null | xargs kill -9 2>/dev/null
sleep 1

# Start uvicorn fully detached
.venv/bin/uvicorn app:app \
  --host 0.0.0.0 \
  --port 8092 \
  --ssl-keyfile cert.key \
  --ssl-certfile cert.pem \
  >> server.log 2>&1 &

disown
echo "Server started, PID: $!"