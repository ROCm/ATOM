#!/bin/bash
# One measured MLPerf Endpoints agentic point, as in README.mlperf-agentic.md: run_ep.sh at concurrency
# <c>, stopped STOP_S seconds after the benchmark starts (default 615: a 10-minute point), then minutes
# [2, 8) of it: server-side generation, running and waiting per engine (window_stats.py) and the MLPerf
# interactivity, 1000 / p90 TPOT, from the client's events (tpot_p90.py). Runs inside the container.
#   usage: bench_point.sh <concurrency> <run dir>
#   env:   STOP_S (615; 0 = let the run end on its own), MODEL (tokenizer for tpot_p90.py), plus anything
#          run_ep.sh, serve_ep.sh and mkcfg.py read (agentic_20261101.env has the measured settings)
set -u
ulimit -c 0
C=$1
export RUNS=$2
STOP_S=${STOP_S:-615}
S=$(cd "$(dirname "$0")" && pwd)
ST=$RUNS/status.txt
mkdir -p "$RUNS"
# The client's event log lives in a tmpfs dir that outlives a stopped client: note what exists now.
SHM_BEFORE=$(ls -d /dev/shm/benchmark_cli_benchmark_* 2>/dev/null | sort)

bash "$S/run_ep.sh" "$C" > "$RUNS.console" 2>&1 &
RUN_PID=$!
until grep -qE "^BENCH|FAIL|DIED|MISMATCH|KERNARG" "$ST" 2>/dev/null; do
  sleep 10
  kill -0 $RUN_PID 2>/dev/null || break
done
if ! grep -q "^BENCH" "$ST" 2>/dev/null; then
  echo "the point did not reach the benchmark:"; cat "$ST" 2>/dev/null; tail -3 "$RUNS.console"
  pkill -f "[o]penai_server"; exit 1
fi
if [ "$STOP_S" -gt 0 ]; then
  b=$(date -u -d "$(grep -m1 ^BENCH "$ST" | grep -oE '[0-9T:-]+Z$')" +%s)
  while [ "$(date -u +%s)" -lt $((b + STOP_S)) ] && ! grep -qE "^DONE|DIED" "$ST"; do sleep 10; done
  pkill -f "[r]un_ep.sh $C"
  pkill -f "[i]nference-endpoint benchmark"
  pkill -f "[o]penai_server"
  sleep 8
  pkill -9 -f "[o]penai_server|ATOM[:][:]|[M]odelRunner"
  sleep 3
fi
wait $RUN_PID 2>/dev/null

NEW=$(comm -13 <(echo "$SHM_BEFORE") <(ls -d /dev/shm/benchmark_cli_benchmark_* 2>/dev/null | sort) | head -1)
EV=$NEW/events/events.jsonl
[ -n "$NEW" ] && [ -f "$EV" ] || EV=$RUNS/c$C/report/events.jsonl
echo "== $RUNS, concurrency $C, minutes 2-8"
python3 "$S/window_stats.py" "$RUNS" 2 8
if [ -f "$EV" ]; then
  python3 "$S/tpot_p90.py" "$EV" "${MODEL:-/data/Qwen3.5-35B-A3B-MXFP4}" 2 8
else
  echo "no client events found: tps_per_user not computed"
fi
# Only the event dir this point created, and only once no client runs anywhere.
if [ -n "$NEW" ] && ! pgrep -f "[i]nference-endpoint" > /dev/null; then rm -rf -- "$NEW"; fi
