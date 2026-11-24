#!/bin/bash
# Accuracy gate for MiniMax-M3.1-Flash TP4 (owner: accuracy agent; doc: /shared_nfs/ygan/max/notes/accuracy.md)
#
#   gate.sh <REPO> [PYPATH_PRE|-] [VAR=value ...]
#
# Leases 4 GPUs (gpurun.sh), starts serve.sh with AL=real (real DSpark-7 acceptance) from <REPO> (+ PYPATH_PRE
# overlay, + VAR=value env for the server, e.g. ATOM_FOO=1 or EXTRA="--some-arg x"), waits for ready, runs GSM8K
# the InferenceX way (gsm8k_ix.sh: lm-eval, full 1319, 5-shot chat, max_tokens 16384, greedy, conc 16; GSM_REPEATS
# runs, default 2, compared by their mean), then the
# long-context needle check (LONG=0 to skip), stops the server, prints PASS/FAIL vs the baseline in
# /shared_nfs/ygan/max/acc/baseline.json plus the measured acceptance length.
#
# Run it from the login node (it dispatches itself into a container, detached) or inside a container.
# env: NODE (yg-di-35 | yg-di-04; default yg-di-35)  CTR (container; default acc35 on 035, acc04 on 004)
#      GTAG (run name; default <repo basename>_<time>)  LONG (1)  GATE_CONC (16; server max-num-seqs = GATE_CONC)
#      PORT (18430)  WAIT (1: poll and print the verdict when run from the login node; 0: just print paths)
#      GSM_REPEATS (2)
# Output: /shared_nfs/ygan/max/acc/gates/<GTAG>/{gate.log,verdict.txt,gsm8k_r*/,long.json}
#         one line per gate appended to /shared_nfs/ygan/max/acc/gates/index.tsv
# One gate at a time (flock on /shared_nfs/ygan/max/acc/gate.lock; a second gate waits).
set -u
D=/shared_nfs/ygan/max
A=$D/acc
B=$D/bin
REPO=${1:?usage: gate.sh <REPO> [PYPATH_PRE|-] [VAR=value ...]}
PRE=${2:--}; [ "$PRE" = "-" ] && PRE=
shift; [ $# -gt 0 ] && shift
NODE=${NODE:-yg-di-35}
GTAG=${GTAG:-$(basename "$REPO")_$(date +%m%d-%H%M%S)}
OUT=/shared_nfs/ygan/p1/c17/gates/$GTAG

if [ ! -f /.dockerenv ] && [ "${GATE_INNER:-}" != 1 ]; then
  case "$NODE" in yg-di-04) CTR=${CTR:-acc04} ;; *) CTR=${CTR:-acc35} ;; esac
  envs=(-e "GTAG=$GTAG" -e "LONG=${LONG:-1}" -e "GATE_CONC=${GATE_CONC:-16}" -e "PORT=${PORT:-18430}"
        -e "GSM_REPEATS=${GSM_REPEATS:-2}" -e "GATE_NOLOCK=${GATE_NOLOCK:-}")
  cmd="docker exec -d"
  for e in "${envs[@]}"; do cmd+=" $(printf %q "$e")"; done
  cmd+=" $CTR bash $B/gate.sh $(printf %q "$REPO") $(printf %q "${PRE:--}")"
  for kv in "$@"; do cmd+=" $(printf %q "$kv")"; done
  NAME=$NODE bash ~/bin/on-node.sh "$cmd" < /dev/null || { echo "gate: dispatch failed"; exit 2; }
  echo "gate $GTAG dispatched to $NODE/$CTR; log $OUT/gate.log ; verdict $OUT/verdict.txt"
  [ "${WAIT:-1}" = 1 ] || exit 0
  for _ in $(seq 480); do [ -s "$OUT/verdict.txt" ] && break; sleep 30; done
  [ -s "$OUT/verdict.txt" ] || { echo "gate: no verdict after 4 h; tail of $OUT/gate.log:"; tail -5 "$OUT/gate.log"; exit 3; }
  cat "$OUT/verdict.txt"; grep -q ": PASS" "$OUT/verdict.txt"
fi

mkdir -p "$OUT"
# The gpurun.sh lease protocol (same lock files, owner files and reserved list), polled every 3 s instead of 15 s.
LK=$D/locks/$(hostname -s); OWNED=()
lease4() {
  mkdir -p "$LK"
  while :; do
    local reserved=" $(cat "$LK/reserved" 2>/dev/null) " set g fd skip ok fds
    for set in 4,5,6,7 0,1,2,3; do
      skip=0; for g in ${set//,/ }; do case "$reserved" in *" $g "*) skip=1 ;; esac; done; [ $skip = 1 ] && continue
      fds=(); ok=1
      for g in ${set//,/ }; do
        exec {fd}>"$LK/gpu$g.lock"
        if flock -n "$fd"; then fds+=("$fd"); else exec {fd}>&-; ok=0; break; fi
      done
      if [ $ok = 1 ]; then
        local v busy=0
        for g in ${set//,/ }; do
          v=$(rocm-smi -d "$g" --showmemuse --csv 2>/dev/null | awk -F, 'NR==2{print $2}')
          [ "${v:-0}" -gt 5 ] && busy=1
        done
        if [ $busy = 0 ]; then
          OWNED=()
          for g in ${set//,/ }; do echo "accuracy/gate-$GTAG | $(date -Is) | pid $$ | gate.sh $REPO" > "$LK/gpu$g.owner"; OWNED+=("$LK/gpu$g.owner"); done
          export HIP_VISIBLE_DEVICES=$set
          return
        fi
        [ "${_warned:-}" = "$set" ] || echo "### $(date -Is) GPUs $set are leasable but hold VRAM (used outside gpurun); skipping them for now"
        _warned=$set
      fi
      for fd in "${fds[@]}"; do exec {fd}>&-; done
    done
    sleep 3
  done
}
if [ "${GATE_LEASED:-}" != 1 ]; then
  exec >> "$OUT/gate.log" 2>&1
  echo "### $(date -Is) gate $GTAG repo=$REPO pre=${PRE:-none} env: $*  (waiting for the gate lock)"
  if [ "${GATE_NOLOCK:-}" != 1 ]; then exec 9> "$A/gate.lock"; flock 9; else echo "### GATE_NOLOCK=1 (measurement run, no gate lock)"; fi
  echo "### $(date -Is) lock acquired; leasing 4 GPUs"
  lease4
  trap 'rm -f "${OWNED[@]}"' EXIT
fi

# ---------------------------------------------------------------- inside the lease
for kv in "$@"; do case "$kv" in *=*) export "$kv" ;; esac; done
PORT=${PORT:-18430}; LONG=${LONG:-1}; GC=${GATE_CONC:-16}
TAG=gate_$GTAG
LOG=/shared_nfs/ygan/p1/c17/logs/server_$TAG.log
echo "### $(date -Is) leased GPUs ${HIP_VISIBLE_DEVICES:-?} on $(hostname -s); server log $LOG"
verdict() { echo "$1" | tee "$OUT/verdict.txt"; }
T0=$(date +%s)
AL=real CONC=$((GC / 2)) MAX_SEQS=$GC PORT=$PORT REPO=$REPO PYPATH_PRE=$PRE TAG=$TAG bash /shared_nfs/ygan/p1/c17/recipe/serve.sh &
SPID=$!
stop_server() {
  pkill -f "server-port $PORT" 2>/dev/null
  for _ in $(seq 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
  pkill -9 -f "server-port $PORT" 2>/dev/null
  wait $SPID 2>/dev/null
  sleep 5
}
trap 'stop_server; rm -f "${OWNED[@]}"' EXIT
ready=0
for _ in $(seq 720); do
  grep -q "Server started successfully" "$LOG" 2>/dev/null && { ready=1; break; }
  kill -0 $SPID 2>/dev/null || break
  sleep 5
done
LOADT=$(( $(date +%s) - T0 ))
if [ $ready != 1 ]; then
  verdict "GATE $GTAG: ERROR server not ready after ${LOADT}s (see $LOG)"
  printf "%s\t%s\t%s\tERROR\tnot-ready\n" "$(date -Is)" "$GTAG" "$REPO" >> /shared_nfs/ygan/p1/c17/gates/index.tsv
  exit 1
fi
echo "### server ready in ${LOADT}s"
curl -s -m 600 127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m3.1","messages":[{"role":"user","content":"Say hi."}],"max_tokens":64,"temperature":0}' > /dev/null
sleep 12
L0=$(wc -l < "$LOG")
for r in $(seq "${GSM_REPEATS:-2}"); do
  timeout 1500 bash $B/gsm8k_ix.sh "$PORT" "$OUT/gsm8k_r$r" "$GC" | tee -a "$OUT/gsm8k.txt"
done
sleep 12
L1=$(wc -l < "$LOG")
python3 $B/al_stats.py "$LOG" $((L0 + 1)) "$L1" | tee "$OUT/al_gsm8k.txt"
if [ "$LONG" = 1 ]; then
  NEEDLE_TIMEOUT_S=${NEEDLE_TIMEOUT_S:-180} timeout 600 python3 /shared_nfs/ygan/p1/c17/recipe/long_check.py "$PORT" "$OUT/long.json" 20000 80000 130000 | tee "$OUT/long.txt"
  sleep 12
  L2=$(wc -l < "$LOG")
  python3 $B/al_stats.py "$LOG" $((L1 + 1)) "$L2" | tee "$OUT/al_long.txt"
fi
trap - EXIT
stop_server
rm -f "${OWNED[@]}"
echo "### server stopped; total lease $(( $(date +%s) - T0 ))s"
python3 /shared_nfs/ygan/p1/c17/recipe/gate_verdict.py "$OUT" "$GTAG" "$REPO" "${PRE:-none}" "$*" > "$OUT/verdict.tmp"
mv "$OUT/verdict.tmp" "$OUT/verdict.txt"; cat "$OUT/verdict.txt"
