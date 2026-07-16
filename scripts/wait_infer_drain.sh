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
# Server log auto-discovery: the canonical signal for "engine made progress"
# is `Engine Core: output send` in the server's stdout. Rather than asking
# the caller to know where the server log lives, the script `readlink`s
# `/proc/<pid>/fd/1` of the atom.entrypoints process and uses THAT as the
# authoritative server log — path-independent across repo layouts. Falls
# back to the caller-supplied LOG_FILE only if /proc lookup fails.
#
# The caller-supplied LOG_FILE is still useful as a SECONDARY signal:
#   - Fault grep runs on BOTH server log AND caller LOG_FILE (max coverage)
#   - File-mtime growth on caller LOG_FILE counts as progress when the
#     engine marker is absent (e.g. simple_inference offline, benchmark tqdm)
#
# Use this script for ANY blocking wait on an ATOM workload — don't fall back
# to `sleep + tail`. Offline path is supported in-script so you can wrap
# simple_inference the same way as server runs.
#
# Liveness: pgrep `$SERVER_PATTERN` (NOT curl /v1/models — under heavy load
# the HTTP endpoint can fail to respond within reasonable timeout even when
# the server is alive, producing false-positive "dead" reports).
#
# Usage: bash scripts/wait_infer_drain.sh [PORT] [MAX_MIN] [POLL_SEC] [LOG_FILE] [STUCK_POLLS] [STARTUP_MAX_MIN]
#   PORT             default 8000  (kept for API symmetry with wait_server_ready.sh; unused in offline mode)
#   MAX_MIN          default 30    (full GSM8K 1319 typically 5-15 min on V4-Pro)
#   POLL_SEC         default 10    (fast poll for quick hang detection)
#   LOG_FILE         default empty (server log auto-discovered via /proc). Pass a
#                    client/workload log (e.g. benchmark.log, simple_inference.log)
#                    to add it as a secondary signal source for fault grep and
#                    mtime-based progress detection.
#   STUCK_POLLS      default 6     (6 × 10s = 1 min of no progress → declare hang)
#   STARTUP_MAX_MIN  default 15    (how long to wait for the workload python
#                                   process to first appear before giving up;
#                                   V4-Pro launchers spend minutes in GPU
#                                   cleanup + cache prep + model load before
#                                   exec'ing python. During this window the
#                                   pgrep miss must NOT be interpreted as a
#                                   clean exit.)
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
LOG_FILE="${4:-}"
STUCK_POLLS="${5:-6}"
STARTUP_MAX_MIN="${6:-15}"
ITERS=$(( MAX_MIN * 60 / POLL ))
STARTUP_ITERS=$(( STARTUP_MAX_MIN * 60 / POLL ))

# Match anything that indicates the eval driver is still hammering the
# server. Extend if you use a different client.
CLIENT_PATTERN='lm_eval|curl.*v1/(completions|chat)|atom\.examples\.benchmark|atom\.benchmarks\.benchmark'
# Server- or offline-mode workload process. simple_inference and
# atom.examples.benchmark also count — process-exit + no-fault = drain.
SERVER_PATTERN='atom\.entrypoints|atom\.examples\.simple_inference'

# Resolve the authoritative server/workload log by reading the workload
# process's stdout fd. Re-run every poll because the workload PID can change
# (start_atom_server has just spawned it; CG capture forks; etc.) and we
# don't want a stale path. Returns empty string on failure.
discover_server_log() {
    local pid log
    pid=$(pgrep -f "$SERVER_PATTERN" 2>/dev/null | head -1)
    [ -z "$pid" ] && return
    log=$(readlink "/proc/$pid/fd/1" 2>/dev/null)
    # Reject /dev/pts/, pipes, sockets, /dev/null — only real files count.
    case "$log" in
        /*) [ -f "$log" ] && echo "$log" ;;
    esac
}

# Grep helper that tolerates missing/empty file and never returns a count
# polluted with grep's stderr.
count_in() {
    local pat=$1 file=$2 c
    [ -z "$file" ] || [ ! -r "$file" ] && { echo 0; return; }
    c=$(grep -cE "$pat" "$file" 2>/dev/null | head -1)
    echo "${c:-0}"
}

mtime_of() {
    local file=$1
    [ -z "$file" ] || [ ! -r "$file" ] && { echo 0; return; }
    stat -c %Y "$file" 2>/dev/null || echo 0
}

FAULT_PATTERN='stopped, reason|MEMORY_VIOLATION|ASSERT_TRAP|proc died unexpectedly|Memory access fault by GPU'

prev_outputs=0
prev_mtime=0
prev_server_mtime=0
stuck=0
server_log=""
# Whether we've ever observed the workload python process. Until this flips
# to 1, a pgrep miss means "launcher script still in pre-exec setup" (GPU
# cleanup, cache prep, model load), NOT "workload finished". Without this
# guard the script returns DRAINED in ~10s for any V4-Pro / large-model run.
ever_seen=0

for ((i=1; i<=ITERS; i++)); do
    sleep "$POLL"

    # Refresh discovered server log (PID may have just appeared).
    new_log=$(discover_server_log)
    if [ -n "$new_log" ] && [ "$new_log" != "$server_log" ]; then
        server_log="$new_log"
        echo "[t=$((i*POLL))s] server log auto-discovered: $server_log"
    fi

    # GPU fault? Scan BOTH server log AND caller LOG_FILE (max coverage).
    # Check BEFORE process-exit so that fault-then-exit is attributed to
    # fault (exit 2), not normal drain.
    fault_server=$(count_in "$FAULT_PATTERN" "$server_log")
    fault_caller=$(count_in "$FAULT_PATTERN" "$LOG_FILE")
    fault_total=$(( fault_server + fault_caller ))
    if [ "$fault_total" -gt 0 ]; then
        echo "[t=$((i*POLL))s] GPU fault detected ($fault_total signals) — exiting 2"
        for f in "$server_log" "$LOG_FILE"; do
            [ -n "$f" ] && [ -r "$f" ] && grep -E "$FAULT_PATTERN" "$f" 2>/dev/null | head -3
        done
        exit 2
    fi

    # Workload process alive? Only by process presence — no curl (HTTP can
    # false-negative under heavy load).
    if pgrep -f "$SERVER_PATTERN" >/dev/null 2>&1; then
        ever_seen=1
    else
        if [ "$ever_seen" -eq 0 ]; then
            # Launcher script still in pre-exec setup (GPU cleanup, cache prep,
            # initial model load). Don't mistake "not started yet" for "drained".
            if [ "$i" -ge "$STARTUP_ITERS" ]; then
                echo "[t=$((i*POLL))s] workload python never appeared in ${STARTUP_MAX_MIN} min — giving up (exit 4)"
                exit 4
            fi
            echo "[t=$((i*POLL))s] waiting for workload python to exec (startup ${i}/${STARTUP_ITERS})"
            continue
        fi
        # Process was seen, now gone, no fault → clean drain. Covers both
        # stop_atom_server (server mode) and simple_inference finishing its
        # prompts (offline mode).
        echo "[t=$((i*POLL))s] workload process exited cleanly — DRAINED"
        exit 0
    fi

    # Engine progress? Three signals — any one resets the stuck counter:
    #   1. "Engine Core: output send" count rising in server log
    #      (auto-discovered; authoritative engine progress — but absent
    #      during warmup phases when no outputs have shipped yet).
    #   2. Caller LOG_FILE mtime advancing (covers client logs / offline
    #      stdout where engine marker is absent: benchmark tqdm,
    #      simple_inference, etc. — but tqdm may not flush during warmup).
    #   3. Server log mtime advancing (universal "server is busy" signal:
    #      uvicorn HTTP access logs, scheduler trace, request arrival —
    #      grows during warmup even before any output ships, so it
    #      prevents false HANG during long warmup phases).
    cur_outputs=$(count_in "Engine Core: output send" "$server_log")
    delta_out=$(( cur_outputs - prev_outputs ))
    cur_mtime=$(mtime_of "$LOG_FILE")
    delta_mtime=$(( cur_mtime - prev_mtime ))
    cur_server_mtime=$(mtime_of "$server_log")
    delta_server_mtime=$(( cur_server_mtime - prev_server_mtime ))

    # Client still running?
    client_alive=$(pgrep -af "$CLIENT_PATTERN" 2>/dev/null | grep -v grep | wc -l)

    echo "[t=$((i*POLL))s] outputs=${cur_outputs} (+${delta_out}) mtime+${delta_mtime}s srv_mtime+${delta_server_mtime}s clients=${client_alive} stuck=${stuck}/${STUCK_POLLS}"

    if [ "$delta_out" -eq 0 ] && [ "$delta_mtime" -eq 0 ] && [ "$delta_server_mtime" -eq 0 ]; then
        stuck=$(( stuck + 1 ))
    else
        stuck=0
        prev_outputs=$cur_outputs
        prev_mtime=$cur_mtime
        prev_server_mtime=$cur_server_mtime
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
        for f in "$server_log" "$LOG_FILE"; do
            if [ -n "$f" ] && [ -r "$f" ]; then
                echo "--- last 20 lines of $f ---"
                tail -20 "$f" 2>/dev/null
            fi
        done
        exit 1
    fi
done

echo "MAX_WAIT ${MAX_MIN} min elapsed without resolution"
for f in "$server_log" "$LOG_FILE"; do
    if [ -n "$f" ] && [ -r "$f" ]; then
        echo "--- last 20 lines of $f ---"
        tail -20 "$f" 2>/dev/null
    fi
done
exit 4
