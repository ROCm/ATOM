#!/bin/bash
# run_point.sh [<concurrency> ...]: benchmark MiniMax-M3 with MLPerf Endpoints at each concurrency (default 10), in the
# order given. Run it on the host, detached, e.g.
#   setsid nohup ./run_point.sh 10 >> "$CLIENT/results/run.log" 2>&1 < /dev/null &
# Each point: wait until the node has no GPU processes; start a fresh server (server.sh in $CONTAINER); wait for
# /health; run gate.py; run a warmup with other seeds; run the measured pass; print its summary line (also appended to
# $CLIENT/results/m3-sweep-<tag>.txt); stop the server. Stops at the first failure and leaves the GPUs free.
# Environment:
#   CAP        the measured run's issue cap in a unit the client parses (ms, s or m: 30m, 1800s; not h), or none for a
#              whole pass; default 30m
#   IDLE_WAIT  seconds to wait for an idle node before each point; default 43200
#   TG         1 = MiniMax-M3 prefill token graphs (ATOM_MINIMAX_M3_PREFILL_TOKEN_GRAPH); default 0. TG_SIZES: buckets
#   MAX_SEQS GPU_MEM KV   passed to server.sh (defaults 2 x C, 0.95, fp8)
set -uo pipefail
cd "$(dirname "$0")" && . ./config.env
[ $# -gt 0 ] || set -- 10
for C in "$@"; do [[ "$C" =~ ^[1-9][0-9]*$ ]] || { echo "not a concurrency: $C" >&2; exit 2; }; done
CAP=${CAP:-30m}
IDLE_WAIT=${IDLE_WAIT:-43200}
TG=${TG:-0}
[[ "$CAP" == none || "$CAP" =~ ^[0-9]+(ms|s|m)$ ]] || { echo "CAP must be like 30m or 1800s, or none" >&2; exit 2; }
export PATH=$HOME/.local/bin:$PATH NUM_GPUS
EP=$CLIENT/endpoints
RES=$CLIENT/results
TAG=$(date +%m%d-%H%M)$([ "$TG" = 1 ] && echo -tg)
SUMMARY=$RES/m3-sweep-$TAG.txt
API=http://127.0.0.1:$API_PORT
mkdir -p "$RES" "$WORK/logs"

say() { echo "[$(date '+%F %T')] $*"; }
idle() { [ -d /sys/class/kfd/kfd/proc ] && [ -z "$(ls /sys/class/kfd/kfd/proc)" ]; }

# Idle twice in a row, a minute apart, within <seconds>.
wait_idle() {
  local deadline=$(($(date +%s) + $1))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if idle; then
      sleep 60
      idle && return 0
    fi
    sleep 120
  done
  return 1
}

errors() {
  local n
  n=$(grep -cE 'Traceback|GPU Hang|HW Exception|Memory access fault|proc died unexpectedly|Watchdog caught collective|Failed to initialize PyNcclCommunicator' \
    "$WORK/logs/$1" 2> /dev/null)
  echo "${n:-0}"
}

alive() { [ "$(docker top "$CONTAINER" -eo pid,args 2> /dev/null | grep -cE 'openai_server|ATOM::')" -gt 0 ]; }

fail() {
  say "FAIL: $*"
  ./down.sh
  say "STOPPED"
  exit 1
}

# Up to 45 minutes for /health (a fresh aiter tree JIT-compiles first); gives up early on an error line or a dead server.
wait_healthy() {
  local i
  for i in $(seq 1 90); do
    curl -sf -m 5 "$API/health" > /dev/null && return 0
    [ "$(errors "$1")" -gt 0 ] && return 1
    if [ "$i" -gt 2 ] && ! alive; then return 1; fi
    sleep 30
  done
  return 1
}

cap_seconds() {
  case $1 in
    *ms) echo $((${1%ms} / 1000)) ;;
    *s) echo "${1%s}" ;;
    *m) echo $((${1%m} * 60)) ;;
  esac
}

progress() { tr '\r' '\n' < "$1" 2> /dev/null | grep -oE '[0-9]+/[0-9]+ \[' | tail -1 | cut -d/ -f1; }

stop_client() {
  local i
  kill -INT "$1" 2> /dev/null
  for i in $(seq 1 12); do
    kill -0 "$1" 2> /dev/null || return 0
    sleep 5
  done
  kill -TERM "$1" 2> /dev/null
  sleep 10
}

# run_client <yaml> <client log> <server log> <time budget, s>: the client's exit code, or 1 if it had to be stopped
# (a server error or exit, no progress for 20 minutes, or over budget).
run_client() {
  local pid t0 last=-1 lastt p
  (ulimit -n 524288 && cd "$EP" && exec env UV_FROZEN=1 uv run inference-endpoint benchmark from-config \
    --config "$1" --mode perf) > "$2" 2>&1 &
  pid=$!
  t0=$(date +%s)
  lastt=$t0
  while kill -0 "$pid" 2> /dev/null; do
    sleep 60
    if [ "$(errors "$3")" -gt 0 ] || ! alive; then
      say "server error or exit during $(basename "$1")"
      stop_client "$pid"
      return 1
    fi
    p=$(progress "$2")
    if [ "${p:-0}" != "$last" ]; then
      last=${p:-0}
      lastt=$(date +%s)
    fi
    if [ $(($(date +%s) - lastt)) -gt 1200 ]; then
      say "no client progress for 20 min"
      stop_client "$pid"
      return 1
    fi
    if [ $(($(date +%s) - t0)) -gt "$4" ]; then
      say "client over its $4 s budget"
      stop_client "$pid"
      return 1
    fi
  done
  wait "$pid"
}

say "m3 sweep $TAG: concurrencies $*, issue cap $CAP, TG=$TG; summary in $SUMMARY"
for C in "$@"; do
  if [ "$CAP" = none ]; then MN=m3-c$C-full-$TAG YCAP=- BUDGET=86400; else MN=m3-c$C-cap$CAP-$TAG YCAP=$CAP BUDGET=$(($(cap_seconds "$CAP") + 2700)); fi
  WN=m3-c$C-warmup-$TAG
  SLOG=m3_c$C-$TAG.log
  # The warmup ends when its first user has finished its trajectories: about two each below c16, one from c16.
  NW=$((C < 16 ? 2 * C : C))
  ./mkpoint_yaml.sh "$WN" "$C" "$NW" - warmup > "$RES/$WN.yaml"
  ./mkpoint_yaml.sh "$MN" "$C" 613 "$YCAP" submission > "$RES/$MN.yaml"
  for y in "$WN" "$MN"; do
    (cd "$EP" && UV_FROZEN=1 uv run inference-endpoint validate-yaml -c "$RES/$y.yaml" > /dev/null 2>&1) \
      || { say "invalid $RES/$y.yaml"; say "STOPPED"; exit 1; }
  done

  say "c$C: waiting for an idle node"
  wait_idle "$IDLE_WAIT" || { say "node not idle within $IDLE_WAIT s"; say "STOPPED"; exit 1; }
  say "c$C: starting the server, log $WORK/logs/$SLOG"
  docker exec -e CONC="$C" -e API_PORT="$API_PORT" -e MODEL="$MODEL" -e DRAFT="$DRAFT" -e TG="$TG" \
    -e TG_SIZES="${TG_SIZES:-}" -e MAX_SEQS="${MAX_SEQS:-$((2 * C))}" -e GPU_MEM="${GPU_MEM:-0.95}" -e KV="${KV:-fp8}" \
    "$CONTAINER" bash /ep/server.sh "/logs/$SLOG" || fail "server.sh c$C"
  wait_healthy "$SLOG" || fail "c$C server did not come up ($(errors "$SLOG") error lines)"
  say "c$C: healthy"
  gate=$(timeout 900 python3 gate.py "$API" "/models/$MODEL" 2>&1 | tail -1)
  [[ "$gate" == "GATE PASS" ]] || fail "c$C gate: $gate"
  say "c$C: GATE PASS; warmup with $NW trajectories (other seeds)"
  run_client "$RES/$WN.yaml" "$RES/$WN.log" "$SLOG" 3600 || fail "c$C warmup"
  say "c$C: measured run, issue cap $CAP"
  run_client "$RES/$MN.yaml" "$RES/$MN.log" "$SLOG" "$BUDGET" || fail "c$C measured run"
  python3 summarize.py "$RES/$MN" | tee -a "$SUMMARY"
  if ! python3 summarize.py --check "$RES/$MN" > /dev/null || [ "$(errors "$SLOG")" -gt 0 ]; then
    fail "c$C did not end cleanly"
  fi
  say "c$C: done"
  ./down.sh
done
say "DONE"
