#!/bin/bash
# Poll http://localhost:$PORT/v1/models until the ATOM server is ready or
# the max wait elapses. Exit 0 = ready, 1 = timeout / startup error.
#
# Usage: bash scripts/wait_server_ready.sh [PORT] [MAX_MIN] [POLL_SEC] [LOG_FILE]
#   PORT      default 8000
#   MAX_MIN   default 6  (server warmup typically 1-3 min; under debug agent 3-5 min)
#   POLL_SEC  default 30
#   LOG_FILE  default /app/logs_claude/atom_server.log (used to detect startup errors)
#
# Side effect: every poll prints "[t=Ns] ready=...". Last line shows pass/fail.

set -uo pipefail

PORT="${1:-8000}"
MAX_MIN="${2:-6}"
POLL="${3:-30}"
LOG_FILE="${4:-/app/logs_claude/atom_server.log}"
ITERS=$(( MAX_MIN * 60 / POLL ))

for ((i=1; i<=ITERS; i++)); do
    sleep "$POLL"
    READY=$(curl -s -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | head -c 60)
    ERR=$(grep -c "cluster_dims\|InductorError\|SHUTDOWN signal\|proc died" \
        "$LOG_FILE" 2>/dev/null | head -1)
    ERR="${ERR:-0}"
    echo "[t=$((i*POLL))s] ready=${READY:-(empty)} err=$ERR"
    if [ -n "$READY" ]; then
        echo "Server READY"
        exit 0
    fi
    if [ "$ERR" -gt 0 ]; then
        echo "Server FAILED to start (errors detected)"
        tail -30 "$LOG_FILE"
        exit 1
    fi
done

echo "Server NOT ready after ${MAX_MIN} min"
tail -30 "$LOG_FILE"
exit 1
