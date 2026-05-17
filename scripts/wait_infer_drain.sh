#!/bin/bash
# Watch an in-flight ATOM inference workload (server OR offline simple_inference).
# Exit 0 when the workload drains cleanly, exit 1 on hang, exit 2 on GPU fault.
#
# Workload modes (auto-detected by process pattern):
#   - Server mode  (atom.entrypoints): drains when eval client (lm_eval / curl
#                  / benchmark) is gone and a single poll shows no new output.
#                  Hang = STUCK_POLLS consecutive polls with no new "Engine
#                  Core: output send" while a client is still hammering.
#   - Offline mode (atom.examples.simple_inference / atom.examples.benchmark):
#                  drains when the inference process exits naturally (no fault
#                  in log). Hang detection is N/A (no client concept; no
#                  Engine Core ZMQ progress).
#
# Use this script for ANY blocking wait on an ATOM workload — don't fall back
# to `sleep + tail`. Offline path is supported in-script so you can wrap
# simple_inference the same way as server runs.
#
# Liveness: pgrep `$SERVER_PATTERN` (NOT curl /v1/models — under heavy load
# the HTTP endpoint can fail to respond within reasonable timeout even when
# the server is alive, producing false-positive "dead" reports).
#
# Usage: bash scripts/wait_infer_drain.sh [PORT] [MAX_MIN] [POLL_SEC] [LOG_FILE] [STUCK_POLLS]
#   PORT         default 8000  (kept for API symmetry with wait_server_ready.sh; unused in offline mode)
#   MAX_MIN      default 30    (full GSM8K 1319 typically 5-15 min on V4-Pro)
#   POLL_SEC     default 10    (fast poll for quick hang detection)
#   LOG_FILE     default /app/logs_claude/atom_server.log
#   STUCK_POLLS  default 6     (6 × 10s = 1 min of no progress → declare hang)
#
# Exit codes:
#   0 — workload drained cleanly (server: client gone + no pending output;
#                                 offline: process exited without fault)
#   1 — hang detected (server only: no progress while client running)
#   2 — fault detected (MEMORY_VIOLATION / ASSERT_TRAP / Memory access fault / proc died)
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
# Server- or offline-mode workload process. simple_inference and
# atom.examples.benchmark also count — process-exit + no-fault = drain.
SERVER_PATTERN='atom\.entrypoints|atom\.examples\.simple_inference'

prev_outputs=0
stuck=0

for ((i=1; i<=ITERS; i++)); do
    sleep "$POLL"

    # GPU fault? Check BEFORE process-exit so that fault-then-exit is
    # attributed to fault (exit 2), not normal drain.
    FAULT=$(grep -cE "stopped, reason|MEMORY_VIOLATION|ASSERT_TRAP|proc died unexpectedly|Memory access fault by GPU" \
        "$LOG_FILE" 2>/dev/null | head -1)
    FAULT="${FAULT:-0}"
    if [ "$FAULT" -gt 0 ]; then
        echo "[t=$((i*POLL))s] GPU fault detected ($FAULT signals) — exiting 2"
        grep -E "stopped, reason|MEMORY_VIOLATION|ASSERT_TRAP|proc died|Memory access fault by GPU" \
            "$LOG_FILE" 2>/dev/null | head -3
        exit 2
    fi

    # Workload process alive? Only by process presence — no curl (HTTP can
    # false-negative under heavy load).
    if ! pgrep -f "$SERVER_PATTERN" >/dev/null 2>&1; then
        # No fault grep matched above + process gone = clean exit (drain).
        # Covers both stop_atom_server (server mode) and simple_inference
        # finishing its prompts (offline mode).
        echo "[t=$((i*POLL))s] workload process exited cleanly — DRAINED"
        exit 0
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

    # Drained cleanly: client gone AND no new output this poll → done.
    # No need to wait STUCK_POLLS — once the client is gone no new requests
    # can arrive, so a single quiet poll is definitive.
    if [ "$client_alive" -eq 0 ] && [ "$stuck" -ge 1 ]; then
        echo "Engine DRAINED (client gone + no pending output)"
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
