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
# Optional environment:
#   SERVER_PID              monitor this PID (in this script's /proc namespace);
#                           require the listener to belong to it/its descendants
#   SERVER_LOG              override LOG_FILE
#   EXPECTED_MODEL_ID       require this exact /v1/models data[].id
#   EXPECTED_MAX_MODEL_LEN  require max_model_len on the same model entry
#   LOG_START_BYTES         override initial log offset; use 0 for a fresh log
#   PROC_ROOT               default /proc (override only for tests)
# HTTP readiness always requires status 200 and a complete, nonempty models JSON.
# Side effect: every poll prints "[t=Ns] ready=...". Last line shows pass/fail.

set -uo pipefail

PORT="${1:-8000}"
MAX_MIN="${2:-6}"
POLL="${3:-30}"
LOG_FILE="${SERVER_LOG:-${4:-/app/logs_claude/atom_server.log}}"
SERVER_PID="${SERVER_PID:-}"
PROC_ROOT="${PROC_ROOT:-/proc}"
EXPECTED_MODEL_ID="${EXPECTED_MODEL_ID:-}"
EXPECTED_MAX_MODEL_LEN="${EXPECTED_MAX_MODEL_LEN:-}"
ITERS=$(( MAX_MIN * 60 / POLL ))

# Pin field 22 (starttime) on first observation: a reused PID is not our server.
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
    if [ -n "$SERVER_PID" ] && ! pid_alive "$SERVER_PID"; then
        echo "Server FAILED (PID $SERVER_PID absent, unreadable, exited or reused)"
        exit 1
    fi
}
check_server

# Verify the actual curl peer in this script's network namespace. A living PID
# plus valid JSON is not enough: a different/old server can own the same port.
# Require every matching LISTEN inode (including SO_REUSEPORT listeners) to be
# held by the specified live process tree. Missing/unreadable evidence fails
# closed. No psutil, process-name matching, or access to another PID namespace.
listener_owned() {
    python3 - "$PROC_ROOT" "$SERVER_PID" "$PORT" "$1" "$2" <<'PY'
import ipaddress
import os
from pathlib import Path
import re
import sys

root, server_pid, port, peer_ip, peer_port = sys.argv[1:]
root = Path(root)
server_pid = int(server_pid)
if peer_port != port:
    sys.exit(1)
peer = ipaddress.ip_address(peer_ip)
peer = getattr(peer, "ipv4_mapped", None) or peer
inodes = set()
try:
    for table in ("tcp", "tcp6"):
        try:
            lines = (root / "self/net" / table).read_text().splitlines()[1:]
        except FileNotFoundError:
            if table == "tcp6" and peer.version == 4:  # IPv6 may be disabled.
                continue
            raise
        for line in lines:
            fields = line.split()
            address, hex_port = fields[1].split(":")
            if fields[3] != "0A" or int(hex_port, 16) != int(port):
                continue
            # /proc stores addresses in native-endian 32-bit words, IPv6 too.
            packed = b"".join(int(address[i:i+8], 16).to_bytes(4, sys.byteorder)
                              for i in range(0, len(address), 8))
            address = ipaddress.ip_address(packed)
            address = getattr(address, "ipv4_mapped", None) or address
            if address == peer or (address.is_unspecified and
                                   (address.version == peer.version or table == "tcp6")):
                inodes.add(fields[9])
    if not inodes or "0" in inodes:
        sys.exit(1)

    parents = {}
    for process in root.iterdir():
        if not process.name.isdecimal():
            continue
        try:
            fields = (process / "stat").read_text().rsplit(") ", 1)[1].split()
            if fields[0] in {"R", "S", "D", "T", "t", "W", "I", "P"}:
                parents[int(process.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    if server_pid not in parents:
        sys.exit(1)
    descendants = {server_pid}
    while True:
        children = {pid for pid, ppid in parents.items() if ppid in descendants}
        if children <= descendants:
            break
        descendants.update(children)
    owned = set()
    for pid in descendants:
        try:
            for fd in (root / str(pid) / "fd").iterdir():
                try:
                    match = re.fullmatch(r"socket:\[([0-9]+)\]", os.readlink(fd))
                    if match:
                        owned.add(match[1])
                except OSError:
                    continue
        except OSError:
            continue
    sys.exit(0 if inodes <= owned else 1)
except (OSError, ValueError, IndexError):
    sys.exit(1)
PY
}

# By default ignore old errors. A dedicated new log can explicitly start at 0.
if [ -z "${LOG_START_BYTES+x}" ]; then
    LOG_START_BYTES=0
    if [ -f "$LOG_FILE" ]; then
        LOG_START_BYTES=$(stat -c %s "$LOG_FILE" 2>/dev/null || echo 0)
    fi
fi
if ! [[ "$LOG_START_BYTES" =~ ^[0-9]+$ ]] ||
   { [ -n "$EXPECTED_MAX_MODEL_LEN" ] && ! [[ "$EXPECTED_MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; }; then
    echo "Invalid LOG_START_BYTES or EXPECTED_MAX_MODEL_LEN"
    exit 1
fi

for ((i=1; i<=ITERS; i++)); do
    check_server
    sleep "$POLL"
    check_server
    READY=""
    # Keep the entire body and curl's exit status: partial JSON, connection
    # refusal and HTTP error bodies must never be mistaken for readiness.
    if RESPONSE=$(curl -s -m 3 --noproxy '*' -w '\n%{http_code}\n%{remote_ip}\n%{remote_port}' "http://localhost:${PORT}/v1/models" 2>/dev/null); then
        PEER_PORT=${RESPONSE##*$'\n'}
        HTTP_RESPONSE=${RESPONSE%$'\n'*}
        PEER_IP=${HTTP_RESPONSE##*$'\n'}
        HTTP_RESPONSE=${HTTP_RESPONSE%$'\n'*}
        READY=$(printf '%s' "$HTTP_RESPONSE" | python3 -c '
import json, sys
body, _, status = sys.stdin.read().rpartition("\n")
if status != "200":
    sys.exit(1)
payload = json.loads(body)
models = payload.get("data") if isinstance(payload, dict) else None
if not isinstance(models, list) or not models or not all(
    isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"]
    for m in models
):
    sys.exit(1)
model_id, max_len = sys.argv[1:]
for model in models:
    if model_id and model["id"] != model_id:
        continue
    if max_len and (type(model.get("max_model_len")) is not int or
                    model["max_model_len"] != int(max_len)):
        continue
    print(json.dumps(model))
    sys.exit(0)
sys.exit(1)
' "$EXPECTED_MODEL_ID" "$EXPECTED_MAX_MODEL_LEN" 2>/dev/null)
    fi
    check_server
    if [ -f "$LOG_FILE" ]; then
        # Detect truncation/recreation: if current size < snapshot, the file
        # was rewritten (e.g. start_atom_server.sh `tee` truncates). Reset
        # offset to 0 so we scan the new content.
        CUR_BYTES=$(stat -c %s "$LOG_FILE" 2>/dev/null || echo 0)
        if [ "$CUR_BYTES" -lt "$LOG_START_BYTES" ]; then
            echo "[t=$((i*POLL))s] log truncated (cur=$CUR_BYTES < snapshot=$LOG_START_BYTES), scanning from offset 0"
            LOG_START_BYTES=0
        fi
        # `openai_server.py: error:` is argparse refusing the command line --
        # a bad flag, or a quoted arg that word-split on its way through
        # EXTRA_ARGS. The process is gone in under a second, so without this
        # the poll waits out its whole budget for a server that never started.
        #
        # A bare `Traceback` does NOT belong here: torch logs whole tracebacks
        # at WARNING (`triton_kernel_wrap.py` does it hundreds of times during
        # a normal Inductor warmup), so matching it fails a healthy start.
        ERR=$(tail -c "+$((LOG_START_BYTES + 1))" "$LOG_FILE" 2>/dev/null \
            | grep -ciE 'cluster_dims|InductorError|SHUTDOWN signal|proc died|Memory access fault|HSA_STATUS_ERROR|(openai_server\.py|api_server\.py|vllm): error:|Engine(Core| core) (initialization failed|failed to start|encountered (an error|an issue|a fatal error)|is dead|died|process (died|failed to start))([[:space:][:punct:]]|$)|EngineDeadError|engine process (died|failed to start)([[:space:][:punct:]]|$)|Fatal error in model execution')
    else
        ERR=0
    fi
    ERR="${ERR:-0}"
    echo "[t=$((i*POLL))s] ready=${READY:-(empty)} err=$ERR"
    if [ "$ERR" -gt 0 ]; then
        echo "Server FAILED to start (errors detected)"
        tail -c "+$((LOG_START_BYTES + 1))" "$LOG_FILE" | tail -30
        exit 1
    fi
    if [ -n "$READY" ]; then
        check_server
        if [ -n "$SERVER_PID" ] && ! listener_owned "$PEER_IP" "$PEER_PORT"; then
            echo "[t=$((i*POLL))s] listener ownership not verified for SERVER_PID=$SERVER_PID; not ready"
            continue
        fi
        check_server
        echo "Server READY"
        exit 0
    fi
done

echo "Server NOT ready after ${MAX_MIN} min"
tail -c "+$((LOG_START_BYTES + 1))" "$LOG_FILE" | tail -30
exit 1
