#!/bin/bash
# Watch an in-flight ATOM inference workload. Exit 0 when the engine drains
# (no progress for STUCK_POLLS polls AND no eval client running), exit 1 if
# a hang is detected while a client is still hammering the server.
#
# A "hang" is defined as STUCK_POLLS consecutive polls with:
#   - zero new "Engine Core: output send" lines in $LOG_FILE
#   - server process still alive (worker thread stuck on a kernel)
#   - an eval client (lm_eval / curl loop / etc.) still running
#
# Liveness: pgrep `atom.entrypoints` (NOT curl /v1/models — under heavy load
# the HTTP endpoint can fail to respond within reasonable timeout even when
# the server is alive, producing false-positive "dead" reports).
#
# Usage: bash scripts/wait_infer_drain.sh [PORT] [MAX_MIN] [POLL_SEC] [LOG_FILE] [STUCK_POLLS]
#   PORT         default 8000  (kept for API symmetry with wait_server_ready.sh)
#   MAX_MIN      default 30    (full GSM8K 1319 typically 5-15 min on V4-Pro)
#   POLL_SEC     default 10    (fast poll for quick hang detection)
#   LOG_FILE     default /app/logs_claude/atom_server.log
#   STUCK_POLLS  default 6     (6 × 10s = 1 min of no progress → declare hang)
#
# Exit codes:
#   0 — engine drained cleanly (no client + no pending output)
#   1 — hang detected (no progress while client running)
#   2 — fault detected (MEMORY_VIOLATION / ASSERT_TRAP / proc died)
#   3 — server process gone
#   4 — max wait elapsed without resolution

set -uo pipefail

PORT="${1:-8000}"
MAX_MIN="${2:-30}"
POLL="${3:-10}"
LOG_FILE="${4:-/app/logs_claude/atom_server.log}"
STUCK_POLLS="${5:-6}"
ITERS=$(( MAX_MIN * 60 / POLL ))

# Match anything that indicates the eval driver is still hammering the
# server. Extend if you use a different client.
CLIENT_PATTERN='lm_eval|curl.*v1/(completions|chat)|atom\.examples\.benchmark|atom\.benchmarks\.benchmark'
SERVER_PATTERN='atom\.entrypoints'

prev_outputs=0
stuck=0

for ((i=1; i<=ITERS; i++)); do
    sleep "$POLL"

    # Server alive? Only by process presence — no curl (false-positives under
    # heavy load).
    if ! pgrep -f "$SERVER_PATTERN" >/dev/null 2>&1; then
        echo "[t=$((i*POLL))s] server process GONE — exiting 3"
        tail -20 "$LOG_FILE" 2>/dev/null
        exit 3
    fi

    # GPU fault?
    FAULT=$(grep -cE "stopped, reason|MEMORY_VIOLATION|ASSERT_TRAP|proc died unexpectedly" \
        "$LOG_FILE" 2>/dev/null | head -1)
    FAULT="${FAULT:-0}"
    if [ "$FAULT" -gt 0 ]; then
        echo "[t=$((i*POLL))s] GPU fault detected ($FAULT signals) — exiting 2"
        grep -E "stopped, reason|MEMORY_VIOLATION|ASSERT_TRAP|proc died" \
            "$LOG_FILE" 2>/dev/null | head -3
        exit 2
    fi

    # Engine progress?
    cur_outputs=$(grep -c "Engine Core: output send" "$LOG_FILE" 2>/dev/null | head -1)
    cur_outputs="${cur_outputs:-0}"
    delta=$(( cur_outputs - prev_outputs ))

    # Client still running?
    client_alive=$(pgrep -af "$CLIENT_PATTERN" 2>/dev/null | grep -v grep | wc -l)

    echo "[t=$((i*POLL))s] outputs=${cur_outputs} (+${delta}) clients=${client_alive} stuck=${stuck}/${STUCK_POLLS}"

    if [ "$delta" -eq 0 ]; then
        stuck=$(( stuck + 1 ))
    else
        stuck=0
        prev_outputs=$cur_outputs
    fi

    # Drained cleanly: no progress AND no clients → done
    if [ "$stuck" -ge "$STUCK_POLLS" ] && [ "$client_alive" -eq 0 ]; then
        echo "Engine DRAINED (no client + no pending output for ${stuck} polls)"
        exit 0
    fi

    # Hung: no progress AND clients still alive → declare hang
    if [ "$stuck" -ge "$STUCK_POLLS" ] && [ "$client_alive" -gt 0 ]; then
        echo "HANG detected (no progress for ${stuck} polls while ${client_alive} client(s) still running)"
        echo "--- last 20 lines of $LOG_FILE ---"
        tail -20 "$LOG_FILE" 2>/dev/null
        exit 1
    fi
done

echo "MAX_WAIT ${MAX_MIN} min elapsed without resolution"
tail -20 "$LOG_FILE" 2>/dev/null
exit 4
