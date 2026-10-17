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
# Optional explicit-PID server mode (e.g. vllm serve):
#   SERVER_PID and CLIENT_PID must be supplied together, in this script's PID
#   namespace. Missing/unreadable/zombie server = failure, never a clean drain.
#   The client must first be observed alive. Its exit means only that it stopped;
#   the harness must still wait for/check its exit code and accuracy result.
#   SERVER_LOG overrides discovery; LOG_FILE remains the client log. Progress is
#   ONLY completed POST /v1/{chat/completions,completions} HTTP 200 access lines
#   (non-streaming requests) or completed question counts from the three-arm tool.
#   Periodic stats, warnings and mere log growth do not reset the hang timer.
#   LOG_START_BYTES optionally sets the scan offset (default 0, as before).
#   PROC_ROOT defaults to /proc; override only for tests.
# With no explicit PIDs, native auto-discovery/progress behavior is unchanged.
#
# Exit codes:
#   0 — workload drained (explicit mode: NOT an accuracy/exit-code verdict;
#                        native server: client gone + no pending output;
#                        offline: process exited without fault)
#   1 — hang detected (server only: no progress while client running)
#   2 — fault detected (also server death/startup failure in explicit-PID mode)
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
SERVER_PID="${SERVER_PID:-}"
CLIENT_PID="${CLIENT_PID:-}"
SERVER_LOG="${SERVER_LOG:-}"
PROC_ROOT="${PROC_ROOT:-/proc}"
LOG_START_BYTES="${LOG_START_BYTES:-0}"
EXPLICIT_PIDS=0
if [ -n "$SERVER_PID$CLIENT_PID" ]; then
    if ! [[ "$SERVER_PID" =~ ^[1-9][0-9]*$ && "$CLIENT_PID" =~ ^[1-9][0-9]*$ ]]; then
        echo "Explicit server mode requires both SERVER_PID and CLIENT_PID"
        exit 2
    fi
    EXPLICIT_PIDS=1
fi
if ! [[ "$LOG_START_BYTES" =~ ^[0-9]+$ ]]; then
    echo "Invalid LOG_START_BYTES"
    exit 2
fi

# Pin field 22 (starttime) on first observation. Reuse means the original server
# died or the original client exited, even if the replacement PID is S/R.
# comm may contain spaces or ')'; field 3 starts after its last closing paren.
declare -A PID_STARTTIMES=()
pid_alive() {
    local pid=$1 info
    local -a fields
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
    IFS= read -r info 2>/dev/null < "$PROC_ROOT/$pid/stat" || return 1
    read -r -a fields <<< "${info##*) }"
    [[ "${fields[0]:-}" =~ ^[RSDTtWIP]$ && "${fields[19]:-}" =~ ^[0-9]+$ ]] || return 1
    if [ -n "${PID_STARTTIMES[$pid]+x}" ]; then
        [[ "${PID_STARTTIMES[$pid]}" == "${fields[19]}" ]]
    else
        PID_STARTTIMES[$pid]=${fields[19]}
    fi
}

check_server() {
    if [ "$EXPLICIT_PIDS" -eq 1 ] && ! pid_alive "$SERVER_PID"; then
        echo "Server FAILED (PID $SERVER_PID absent, unreadable, exited or reused) — exiting 2"
        exit 2
    fi
}
check_server

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
    if [ -n "$SERVER_LOG" ]; then
        echo "$SERVER_LOG"
        return
    fi
    pid=$SERVER_PID
    [ -n "$pid" ] || pid=$(pgrep -f "$SERVER_PATTERN" 2>/dev/null | head -1)
    [ -z "$pid" ] && return
    log=$(readlink "$PROC_ROOT/$pid/fd/1" 2>/dev/null)
    # Reject /dev/pts/, pipes, sockets, /dev/null — only real files count.
    case "$log" in
        /*) [ -f "$log" ] && echo "$log" ;;
    esac
}

# An optional fixed offset applies to faults and client completion records too.
# Use 0 for independent logs (the default); nonzero offsets assume no rotation.
log_text() {
    local file=$1
    [ -n "$file" ] && [ -r "$file" ] || return
    tail -c "+$((LOG_START_BYTES + 1))" "$file" 2>/dev/null
}

# Grep helper that tolerates missing/empty file and never returns a count
# polluted with grep's stderr.
count_in() {
    local pat=$1 file=$2 c
    [ -z "$file" ] || [ ! -r "$file" ] && { echo 0; return; }
    c=$(log_text "$file" | grep -cE "$pat" 2>/dev/null)
    echo "${c:-0}"
}

count_excluding() {
    local pat=$1 file=$2 c
    { [ -z "$file" ] || [ ! -r "$file" ]; } && { echo 0; return; }
    c=$(grep -cvE "$pat" "$file" 2>/dev/null | head -1)
    echo "${c:-0}"
}

mtime_of() {
    local file=$1
    [ -z "$file" ] || [ ! -r "$file" ] && { echo 0; return; }
    stat -c %Y "$file" 2>/dev/null || echo 0
}

FAULT_PATTERN='stopped, reason|MEMORY_VIOLATION|ASSERT_TRAP|proc died unexpectedly|Memory access fault by GPU'
if [ "$EXPLICIT_PIDS" -eq 1 ]; then
    FAULT_PATTERN+='|(openai_server\.py|api_server\.py|vllm): error:|Engine(Core| core) (initialization failed|failed to start|encountered (an error|an issue|a fatal error)|is dead|died|process (died|failed to start))([[:space:][:punct:]]|$)|EngineDeadError|[Ee]ngine process (died|failed to start)([[:space:][:punct:]]|$)|Fatal error in model execution|HSA_STATUS_ERROR'
fi
HTTP_DONE_PATTERN='"POST /v1/(chat/completions|completions) HTTP/[0-9.]+" 200([[:space:]]|$)'

# Reusable so every successful exit can rescan after progress/client probing.
# The API frontend can stay alive after its engine logs a fatal error.
check_faults() {
    local fault_server fault_caller fault_total f
    fault_server=$(count_in "$FAULT_PATTERN" "$server_log")
    fault_caller=$(count_in "$FAULT_PATTERN" "$LOG_FILE")
    fault_total=$(( fault_server + fault_caller ))
    if [ "$fault_total" -gt 0 ]; then
        echo "[t=$((i*POLL))s] workload fault detected ($fault_total signals) — exiting 2"
        for f in "$server_log" "$LOG_FILE"; do
            [ -n "$f" ] && [ -r "$f" ] && grep -E "$FAULT_PATTERN" "$f" 2>/dev/null | head -3
        done
        exit 2
    fi
}

# The arm summary is emitted only after every question in that arm completed.
# Count the denominator (completed), NOT the numerator (correct); never treat a
# repeated summary, FAILED request, or an in-flight request as new progress.
completed_questions() {
    log_text "$LOG_FILE" | awk '
        /^ARM (blind|cold|warm)[[:space:]]+correct [0-9]+\/[0-9]+ prompt_tokens=/ {
            split($4, n, "/")
            if (n[2] > done[$2]) done[$2] = n[2]
        }
        END { for (arm in done) total += done[arm]; print total + 0 }
    '
}

# Server-log lines that must not count as progress. aiter prints this one from
# a rank that is ALREADY blocked on the broadcast ring, so reading it as
# liveness inverts its meaning -- and it repeats every 60s, exactly the default
# STUCK_POLLS*POLL quiet window, so it resets the counter one poll before it can
# fire and the hang only ever surfaces at MAX_MIN.
IDLE_NOISE_PATTERN='No available shared memory broadcast block'

prev_outputs=0
prev_mtime=0
prev_server_lines=0
stuck=0
server_log=""
# Whether we've ever observed the workload python process. Until this flips
# to 1, a pgrep miss means "launcher script still in pre-exec setup" (GPU
# cleanup, cache prep, model load), NOT "workload finished". Without this
# guard the script returns DRAINED in ~10s for any V4-Pro / large-model run.
ever_seen=0
client_seen=0
prev_completed=0
# Observe an already-running short client before the first sleep too.
if [ "$EXPLICIT_PIDS" -eq 1 ] && pid_alive "$CLIENT_PID"; then
    client_seen=1
fi

for ((i=1; i<=ITERS; i++)); do
    check_server
    sleep "$POLL"
    check_server

    # Refresh discovered server log (PID may have just appeared).
    new_log=$(discover_server_log)
    if [ -n "$new_log" ] && [ "$new_log" != "$server_log" ]; then
        server_log="$new_log"
        echo "[t=$((i*POLL))s] server log auto-discovered: $server_log"
    fi

    # Scan both logs before interpreting process exit as normal drain.
    check_faults

    # Workload process alive? Only by process presence — no curl (HTTP can
    # false-negative under heavy load).
    if [ "$EXPLICIT_PIDS" -eq 1 ] || pgrep -f "$SERVER_PATTERN" >/dev/null 2>&1; then
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
        check_faults
        echo "[t=$((i*POLL))s] workload process exited cleanly — DRAINED"
        exit 0
    fi

    # Legacy native progress uses three signals; explicit-PID mode below instead
    # counts only completed non-streaming requests/questions.
    # Native signals — any one resets the stuck counter:
    #   1. "Engine Core: output send" count rising in server log
    #      (auto-discovered; authoritative engine progress — but absent
    #      during warmup phases when no outputs have shipped yet).
    #   2. Caller LOG_FILE mtime advancing (covers client logs / offline
    #      stdout where engine marker is absent: benchmark tqdm,
    #      simple_inference, etc. — but tqdm may not flush during warmup).
    #   3. Server log GROWING IN LINES, ignoring IDLE_NOISE_PATTERN (universal
    #      "server is busy" signal: uvicorn HTTP access logs, scheduler trace,
    #      request arrival — grows during warmup even before any output ships,
    #      so it prevents false HANG during long warmup phases). Counted by
    #      line rather than by mtime because mtime cannot tell what was
    #      written, and a wedge that keeps one periodic logger alive would
    #      otherwise read as progress.
    if [ "$EXPLICIT_PIDS" -eq 1 ]; then
        cur_outputs=$(count_in "$HTTP_DONE_PATTERN" "$server_log")
        cur_completed=$(completed_questions)
        delta_out=$(( cur_outputs - prev_outputs ))
        delta_completed=$(( cur_completed - prev_completed ))
        prev_outputs=$cur_outputs
        prev_completed=$cur_completed
        client_alive=0
        if pid_alive "$CLIENT_PID"; then
            client_alive=1
            client_seen=1
        fi
        if [ "$client_seen" -eq 0 ]; then
            if [ "$i" -ge "$STARTUP_ITERS" ]; then
                echo "Client PID $CLIENT_PID never observed alive — exiting 4"
                exit 4
            fi
            echo "[t=$((i*POLL))s] waiting to observe client PID $CLIENT_PID"
            continue
        fi
        if [ "$delta_out" -gt 0 ] || [ "$delta_completed" -gt 0 ]; then
            stuck=0
        else
            stuck=$(( stuck + 1 ))
        fi
        echo "[t=$((i*POLL))s] http200=${cur_outputs} (+${delta_out}) completed=${cur_completed} (+${delta_completed}) client=${client_alive} stuck=${stuck}/${STUCK_POLLS}"
    else
        cur_outputs=$(count_in "Engine Core: output send" "$server_log")
        delta_out=$(( cur_outputs - prev_outputs ))
        cur_mtime=$(mtime_of "$LOG_FILE")
        delta_mtime=$(( cur_mtime - prev_mtime ))
        cur_server_lines=$(count_excluding "$IDLE_NOISE_PATTERN" "$server_log")
        delta_server_lines=$(( cur_server_lines - prev_server_lines ))

        # Client still running?
        client_alive=$(pgrep -af "$CLIENT_PATTERN" 2>/dev/null | grep -v grep | wc -l)

        echo "[t=$((i*POLL))s] outputs=${cur_outputs} (+${delta_out}) mtime+${delta_mtime}s srv+${delta_server_lines}ln clients=${client_alive} stuck=${stuck}/${STUCK_POLLS}"

        if [ "$delta_out" -eq 0 ] && [ "$delta_mtime" -eq 0 ] && [ "$delta_server_lines" -eq 0 ]; then
            stuck=$(( stuck + 1 ))
        else
            stuck=0
            prev_outputs=$cur_outputs
            prev_mtime=$cur_mtime
            prev_server_lines=$cur_server_lines
        fi
    fi

    # Drained cleanly: client gone AND no new output this poll → done.
    # No need to wait STUCK_POLLS — once the client is gone no new requests
    # can arrive, so a single quiet poll is definitive.
    if [ "$client_alive" -eq 0 ] && [ "$stuck" -ge 1 ]; then
        check_faults
        check_server
        echo "Engine DRAINED (client gone + no pending output)"
        if [ "$EXPLICIT_PIDS" -eq 1 ]; then
            echo "Client exited; harness must check its exit code and result (not an accuracy verdict)"
        fi
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
