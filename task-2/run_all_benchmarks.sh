#!/bin/bash
set -e

echo "[INFO] Starting FastAPI server from serving_rag.py..."
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

log "Starting server..."
# instead of echo "[INFO] Starting server..."

python serving_rag_expanded.py &


SERVER_PID=$!
trap "echo '[INFO] Cleaning up...'; kill $SERVER_PID 2>/dev/null; exit" INT TERM
echo "[INFO] Server started with PID $SERVER_PID"

SERVER_URL="http://localhost:8000/docs"
MAX_WAIT_TIME=900  # 15 minutes in seconds
INTERVAL=5         # Check every 5 seconds
ELAPSED=0

echo "[INFO] Waiting for server to become ready..."

until curl -s --head --fail "$SERVER_URL" > /dev/null; do
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    
    if [ $ELAPSED -ge $MAX_WAIT_TIME ]; then
        echo "[ERROR] Server did not become ready within $MAX_WAIT_TIME seconds."
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done

echo "[INFO] Server is ready."


if ! curl -s http://localhost:8000/docs > /dev/null; then
    echo "[ERROR] Server did not become ready in time."
    kill $SERVER_PID
    exit 1
fi

echo "[INFO] Running benchmark script..."
python testing_script.py --mode ideal --target original --rps 50 --num_requests 1000 # <-- Replace with your actual test script

echo "[INFO] Killing the server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

echo "[INFO] Done."
