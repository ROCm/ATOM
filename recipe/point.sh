#!/bin/bash
# One AgentX point of one arm (run INSIDE the p1 container under a gpurun lease; use launch.sh).
# Port of /shared_nfs/ygan/max/int1/point.sh: fresh server -> bench.sh replay -> stop -> ladder_stats/summaries.
# milestone3 change: an ENGINE-DEATH / STALL WATCH. While the server starts and while aiperf runs, the server log is
# scanned every 10 s; the point aborts (kills aiperf and the server, frees the GPUs, writes logs/abort_<TAG>) when
#  - a fatal line appears: HSA_STATUS_ERROR / "proc died unexpectedly" / EngineDeadError / "Engine core .* failed"
#    (milestone2 m2_B_034_1: a TP rank died but the API process stayed up and aiperf ran on for 13 min), or
#  - the server process is gone, or
#  - the watchdog reported >= STALL_MAX (default 8) long forwards ("a forward has run for"; milestone2 m2_B_035_2:
#    89 multi-second decode forwards made the run invalid; good C runs had 0).
# milestone4 change: after the point, score_diag.py prints the official record-span score (RS), the prorated PRO900
#    score, and the cutoff diagnostics (in flight at the sending cutoff, last request end, aiperf grace_period_timeout,
#    "leaked state"/"Abandoned pending join" warnings) so a client grace timeout is never mistaken for a server stall
#    (aiperf-anomaly NOTES); ladder_stats.py now reads the real summarize_ix.py and its fallback is record-span exact.
# c14 change: STRICT ABORT WATCH (task rule for C14). The point also aborts when
#  - any device VRAM of the lease exceeds VRAM_MAX_GIB (default 282 GiB; rocm-smi, checked on every 10 s watch tick), or
#  - any allocator-retry / out-of-memory / Traceback line appears in the server log, or
#  - any long-forward line appears ("a forward has run for" or "reported running long"; STALL_MAX default 1).
# usage: point.sh <ARM> <TAG> [CONC=32]      env: DURATION (900), PORT (default by GPU half: 18960 / 18980), STALL_MAX (1), VRAM_MAX_GIB (282)
set -u
ARM=$1; TAG=$2; N=${3:-32}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source $RCP/env_arm$ARM.sh || exit 2
check_repo || exit 2
export FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR:-$NLOC/flydsl_arm$ARM}; seed_flydsl
PORT=${PORT:-$((18960 + 20 * (${HIP_VISIBLE_DEVICES%%,*} / 4)))}; R=$MS/results/$TAG; SL=$MS/logs/server_$TAG.log
STALL_MAX=${STALL_MAX:-1}; ABORT=$MS/logs/abort_$TAG; VRAM_MAX_GIB=${VRAM_MAX_GIB:-282}
[ -d "$R" ] && mv -f "$R" "$R.prev.$(date +%s)"; mkdir -p $R
echo "### $(date -Is) point $TAG arm=$ARM conc=$N on $(hostname -s) GPUs $HIP_VISIBLE_DEVICES port=$PORT repo=$REPO extra='$EXTRA'"
CONC=$N TAG=$TAG PORT=$PORT bash $RCP/serve.sh & SPID=$!
SAMP=; BPID=
FATAL='HSA_STATUS_ERROR|proc died unexpectedly|EngineDeadError|Engine core .*(failed|died)|RuntimeError: .*(HIP error|hipError)'
# engine_dead: prints a reason and returns 0 when the engine is dead or stalling, 1 when healthy
engine_dead() {
  local f n
  kill -0 $SPID 2>/dev/null || { echo "server process exited"; return 0; }
  f=$(grep -m1 -E "$FATAL" $SL 2>/dev/null) && { echo "fatal: ${f:0:200}"; return 0; }
  f=$(grep -m1 -iE 'alloc.*retr|out of memory|OutOfMemory|Traceback' $SL 2>/dev/null) && { echo "alloc/oom/traceback: ${f:0:200}"; return 0; }
  n=$(grep -cE "a forward has run for|reported running long" $SL 2>/dev/null); n=${n:-0}
  [ "$n" -ge "$STALL_MAX" ] && { echo "stall: $n long forwards (>= $STALL_MAX)"; return 0; }
  for g in ${HIP_VISIBLE_DEVICES//,/ }; do
    f=$(rocm-smi -d $g --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR==2{print $3}')
    [ -n "$f" ] && [ "$f" -gt $(( VRAM_MAX_GIB * 1073741824 )) ] && { echo "vram: GPU $g used $f bytes > $VRAM_MAX_GIB GiB"; return 0; }
  done
  return 1
}
stop() {
  [ -n "$SAMP" ] && kill $SAMP 2>/dev/null; SAMP=; [ -n "${VSAMP:-}" ] && kill $VSAMP 2>/dev/null; VSAMP=
  if [ -n "$BPID" ]; then kill $BPID 2>/dev/null; pkill -f "aiperf profile .*localhost:$PORT " 2>/dev/null; sleep 3
    pkill -9 -f "aiperf profile .*localhost:$PORT " 2>/dev/null; BPID=; fi
  pkill -f "server-port $PORT( |$)" 2>/dev/null
  for _ in $(seq 120); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  pkill -9 -f "server-port $PORT( |$)" 2>/dev/null; wait $SPID 2>/dev/null; sleep 5
  gpus_free
}
trap 'exit 143' TERM INT
trap stop EXIT
t0=$(date +%s); ok=0; why=
while :; do
  grep -q "Server started successfully" $SL 2>/dev/null && { ok=1; break; }
  why=$(engine_dead) && break; why=
  [ $(( $(date +%s) - t0 )) -gt 2400 ] && { why="not ready after 2400 s"; break; }
  sleep 10
done
echo "### server ready=$ok after $(( $(date +%s) - t0 )) s ${why:+(ABORT: $why)}"
if [ $ok = 1 ]; then
  ( while :; do echo "# ts $(date +%s.%N)"; curl -s -m 10 "127.0.0.1:$PORT/metrics" | grep -E '^atom:'; sleep 30; done ) > $R/srvmetrics.prom 2>/dev/null & SAMP=$!
  # official-3600: allocator watch = per-GPU VRAM used (bytes) every 60 s + running counts of long forwards / alloc / OOM lines
  ( while :; do v=; for g in ${HIP_VISIBLE_DEVICES//,/ }; do v="$v $(rocm-smi -d $g --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR==2{print $3}')"; done
      echo "$(date +%s) vram_used$v longfwd $(grep -c 'a forward has run for' $SL) alloc $(grep -ciE 'alloc.*retr|out of memory|OutOfMemory' $SL)"; sleep 60; done ) > $R/vram_watch.txt 2>/dev/null & VSAMP=$!
  CONC=$N TAG=$TAG PORT=$PORT DURATION=${DURATION:-3600} bash $RCP/bench.sh & BPID=$!
  while kill -0 $BPID 2>/dev/null; do
    if why=$(engine_dead); then
      echo "### $(date -Is) ABORT during bench: $why"; kill $BPID 2>/dev/null; pkill -f "aiperf profile .*localhost:$PORT " 2>/dev/null
      sleep 5; pkill -9 -f "aiperf profile .*localhost:$PORT " 2>/dev/null; break
    fi
    why=; sleep 10
  done
  wait $BPID 2>/dev/null; echo "### bench rc=$?"; BPID=
  kill $SAMP $VSAMP 2>/dev/null; SAMP=; { echo "# ts $(date +%s.%N)"; curl -s -m 10 "127.0.0.1:$PORT/metrics" | grep -E '^atom:'; } >> $R/srvmetrics.prom; gzip -f $R/srvmetrics.prom
else tail -30 $SL; fi
[ -n "$why" ] && { echo "$(date -Is) $TAG ABORTED: $why" | tee $ABORT; tail -5 $SL > $R/ABORTED.txt; echo "$why" >> $R/ABORTED.txt; }
trap - EXIT; stop
python3 $RCP/ladder_stats.py --save $TAG 2>&1 | tail -4   # D=p1/milestone4: reads results/$TAG and logs/server_$TAG.log
python3 $MAXB/slim_srvmetrics.py $R 2>/dev/null
python3 $MAXB/summarize_ix.py $R 2>&1 | tail -3
# milestone4: official record-span score (RS) + prorated PRO900 + cutoff/grace diagnostics (in flight at the cutoff,
# last end, grace_period_timeout, aiperf leaked-join warnings) -> results/$TAG/score_diag.json
python3 $RCP/score_diag.py $R 2>&1
echo "ready=$ok ${why:+aborted=$why }$(date -Is)" > $MS/logs/done_pt_$TAG
echo "### $(date -Is) point done"
