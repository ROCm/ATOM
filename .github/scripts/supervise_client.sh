#!/bin/bash
# Run a benchmark / accuracy client against a vLLM or SGLang plugin server and
# stop it early when the server hangs or faults, instead of letting the client
# burn the step's whole timeout. Same pattern as atom_test.sh does for native
# ATOM: the client runs in the background in its own process group, and
# scripts/wait_infer_drain.sh watches the server in the foreground.
#
# Usage: supervise_client.sh ENGINE CLIENT_LOG MAX_MIN -- CLIENT_COMMAND...
#   ENGINE      vllm | sglang: picks the server's PID file and log patterns.
#   CLIENT_LOG  the client's output is appended here (and still printed);
#               its growth counts as progress.
#   MAX_MIN     give up (exit 4) after this many minutes.
#
# Run it inside the server's container, from the repository root or not.
#
# Exit status: the client's own status when the server stayed healthy;
# otherwise wait_infer_drain.sh's: 1 hang, 2 fault (including the server
# exiting while the client still runs), 4 MAX_MIN elapsed.
#
# Environment:
#   WAIT_STUCK_POLLS  polls without progress that make a hang (default 60:
#                     10 min at the default poll). Longer than native ATOM's
#                     3 min because a plugin server can spend minutes on a
#                     lazy aiter JIT build for a new shape mid-run.
#   WAIT_POLL_SEC     seconds between polls (default 10).
#   WAIT_SERVER_PID   server PID; read from the engine's PID file if unset.
#   WAIT_FAULT_PATTERN_EXTRA / WAIT_IDLE_NOISE_PATTERN_EXTRA override the
#                     engine defaults below.

set -uo pipefail

if [ "$#" -lt 5 ] || [ "$4" != "--" ]; then
    echo "usage: $0 vllm|sglang CLIENT_LOG MAX_MIN -- CLIENT_COMMAND..." >&2
    exit 64
fi
ENGINE=$1
CLIENT_LOG=$2
MAX_MIN=$3
shift 4

case "$ENGINE" in
    vllm)
        pid_file=${VLLM_PID_FILE:-/tmp/vllm_oot.pid}
        # The fatal markers atom_oot_test.sh's FATAL_LOG_PATTERNS watches for
        # while the server starts; keep the two in step.
        fault_extra=${FATAL_LOG_PATTERNS:-'unhandled cuda error|uncorrectable ECC|EngineCore[_ ][A-Za-z0-9]* died|Engine core proc.* died|EngineCore failed to start|Failed to initialize EngineCore'}
        # vLLM's API server logs throughput every 10 s even while the engine is
        # wedged; an all-zero line is not progress.
        noise_extra='Avg prompt throughput: 0\.0 tokens/s, Avg generation throughput: 0\.0 tokens/s'
        ;;
    sglang)
        pid_file=${SGLANG_PID_FILE:-/tmp/atom_sglang.pid}
        fault_extra='Scheduler hit an exception|Received sigquit from a child process'
        noise_extra=''
        ;;
    *)
        echo "unknown ENGINE: $ENGINE (expected vllm or sglang)" >&2
        exit 64
        ;;
esac

export WAIT_SERVER_PID=${WAIT_SERVER_PID:-$(cat "$pid_file" 2>/dev/null)}
if [ -z "$WAIT_SERVER_PID" ]; then
    echo "supervise_client.sh: no server PID (set WAIT_SERVER_PID or write $pid_file)" >&2
    exit 64
fi
export WAIT_FAULT_PATTERN_EXTRA=${WAIT_FAULT_PATTERN_EXTRA:-$fault_extra}
export WAIT_IDLE_NOISE_PATTERN_EXTRA=${WAIT_IDLE_NOISE_PATTERN_EXTRA:-$noise_extra}
export WAIT_SERVER_EXIT_IS_FAULT=1

mkdir -p "$(dirname "$CLIENT_LOG")"
touch "$CLIENT_LOG"

# Own process group (set -m makes the subshell's pgid its PID), so the whole
# client pipeline can be signalled at once.
set -m
( "$@" 2>&1 | tee -a "$CLIENT_LOG"; exit "${PIPESTATUS[0]}" ) &
client_pid=$!
set +m
export WAIT_CLIENT_PID=$client_pid

drain_script="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/wait_infer_drain.sh"
echo "========== Supervising ${ENGINE} client (pid ${client_pid}, server pid ${WAIT_SERVER_PID}) =========="
bash "$drain_script" 0 "$MAX_MIN" "${WAIT_POLL_SEC:-10}" "$CLIENT_LOG" "${WAIT_STUCK_POLLS:-60}"
drain_rc=$?

if [ "$drain_rc" -ne 0 ]; then
    echo "wait_infer_drain.sh exit=${drain_rc}: stopping client process group ${client_pid}"
    kill -TERM -- "-${client_pid}" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        kill -0 "$client_pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-${client_pid}" 2>/dev/null || true
    wait "$client_pid" 2>/dev/null || true
    exit "$drain_rc"
fi

wait "$client_pid"
