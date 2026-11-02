#!/bin/bash
# Host driver. Waits until no other process holds the GPUs it uses (GPUS, default all), then:
#   1. accuracy: agentic server config (CONC=$ACC_CONC point) with real EAGLE3 acceptance, GSM8K 5-shot full;
#      the perf ladder only runs if flexible-extract >= $ACC_MIN.
#   2. perf: one server launch + one 3600 s aiperf replay per concurrency point, fixed acceptance 0.5933.
# Usage: TAG=flydsl_fp8 bash run_ladder.sh [conc ...]   (default: 2 10 24 40)
# Env: KV (fp8) FLY (1) QR (INT4) OFFLOAD (1) TG TG_SIZES GPUS MONO to server.sh; DURATION (3600) to aiperf.sh;
#      API_PORT (8890) ENGINE_PORT (8896) to both; ACC_FIRST (1) ACC_CONC (10) ACC_MIN (0.92).
#      8-GPU MI355X: GPUS=0,1,4,5 (the recipe's NUMA split, required with OFFLOAD=1).
set -u
C=${C:-ruili_m3_flydsl}
TAG=${TAG:-flydsl_fp8}
CONCS=${*:-2 10 24 40}
ROOT=${ROOT:-/home/ruili/code/m3_fp8kv}   # host dir whose logs/ is mounted at /logs
ATOM_DIR=${ATOM_DIR:-$ROOT/ATOM_slim}     # mounted at /app/ATOM
AITER_DIR=${AITER_DIR:-$ROOT/aiter_6699}  # mounted at /app/aiter
HOST_DIR=$ROOT/logs/agentic/$TAG   # = /logs/agentic/$TAG in the container
OUT=/logs/agentic/$TAG
MODEL=/models/MiniMax-M3-MXFP4
mkdir -p $HOST_DIR
log() { echo "$(date '+%F %T') $*" | tee -a $HOST_DIR/ladder.log; }
health() { curl -s -m 5 localhost:${API_PORT:-8890}/health -o /dev/null -w '%{http_code}'; }
CID=$(docker inspect -f '{{.Id}}' $C)

stop_all() {
  docker exec $C bash -c "pkill -f '[a]iperf profile'; pkill -f '[l]m_eval'; pkill -f '[a]tom.entrypoints.openai_server'" 2>/dev/null
  for i in $(seq 1 60); do
    n=$(docker exec $C bash -c "ps -eo args | grep -cE '[o]penai_server|[a]iperf profile|[l]m_eval'")
    [ "${n:-0}" = 0 ] && break; sleep 5
  done
  sleep 20
}

# KFD gpu_ids of the GPUs this run uses: GPUS as ROCR_VISIBLE_DEVICES indices, every GPU when it is empty.
our_gpu_ids() {
  local i=0 n
  for n in $(ls /sys/class/kfd/kfd/topology/nodes | sort -n); do
    grep -q '^simd_count [1-9]' /sys/class/kfd/kfd/topology/nodes/$n/properties 2>/dev/null || continue
    if [ -z "${GPUS:-}" ] || [[ ",$GPUS," == *",$i,"* ]]; then cat /sys/class/kfd/kfd/topology/nodes/$n/gpu_id; fi
    i=$((i + 1))
  done
}

busy_gpus() {  # processes, not ours, holding over 1 GiB on a GPU this run uses; the node's other GPUs do not count
  local ids p id
  ids=$(our_gpu_ids)
  for p in $(ls /sys/class/kfd/kfd/proc 2>/dev/null); do
    grep -q "$CID" /proc/$p/cgroup 2>/dev/null && continue
    for id in $ids; do
      [ "$(cat /sys/class/kfd/kfd/proc/$p/vram_$id 2>/dev/null || echo 0)" -gt 1073741824 ] && { echo $p; break; }
    done
  done
}

wait_gpus() {
  local first=1
  while [ -n "$(busy_gpus)" ]; do
    [ $first = 1 ] && log "$1 waiting for GPUs, used by pids $(busy_gpus | tr '\n' ' ')"
    first=0; sleep 120
  done
  [ $first = 0 ] && log "$1 GPUs free"
}

start_server() {  # $1 conc, $2 acc(0/1), $3 server log (container path); returns 0 when healthy
  docker exec -e CONC=$1 -e ACC=$2 -e GPU_MEM=${GPU_MEM:-0.95} -e KV=${KV:-fp8} -e FLY=${FLY:-1} -e QR=${QR:-INT4} -e OFFLOAD=${OFFLOAD:-1} \
    -e TG=${TG:-0} -e TG_SIZES=${TG_SIZES:-} -e GPUS=${GPUS:-} -e MONO=${MONO:-} \
    -e API_PORT=${API_PORT:-8890} -e ENGINE_PORT=${ENGINE_PORT:-8896} \
    $C bash /scripts/agentic/server.sh $3
  log "c$1 acc=$2 server starting, log ${3/#\/logs/$ROOT/logs}"
  for i in $(seq 1 360); do
    [ "$(health)" = 200 ] && break
    if ! docker exec $C pgrep -f '[a]tom.entrypoints.openai_server' >/dev/null; then
      log "c$1 ABORT server died: $(docker exec $C grep -m1 -oE 'exitcode=-?[0-9]+|[A-Za-z]+Error: .{0,160}' $3)"
      return 1
    fi
    sleep 10
  done
  [ "$(health)" = 200 ] || { log "c$1 ABORT server not healthy after 1 h"; return 1; }
  local ans
  ans=$(curl -s -m 600 localhost:${API_PORT:-8890}/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23? Answer briefly.\"}],\"max_tokens\":1024,\"temperature\":0}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][-80:])" 2>&1 | tr '\n' ' ')
  log "c$1 smoke: $ans"
}

server_facts() {  # $1 server log
  docker exec $C bash -c "echo routes: \$(grep -hoE 'pa_decode -> [a-z]+' $1 | sort | uniq -c | tr -s ' ' | tr '\n' ';');
    grep -iE 'acceptance|dcp_config.indexer_dcp_only|lmcache_offload|Traceback' $1 | tail -n 4 | cut -c1-220" >> $HOST_DIR/ladder.log
}

log "start $TAG concs=[$CONCS] KV=${KV:-fp8} FLY=${FLY:-1} QR=${QR:-INT4} OFFLOAD=${OFFLOAD:-1}" \
  "ATOM=$(git -C $ATOM_DIR log -1 --format='%h(%D)')" \
  "aiter=$(git -C $AITER_DIR log -1 --format='%h(%D)' 2>/dev/null)" \
  "aiperf=$(docker exec $C git -C /opt/aiperf rev-parse --short HEAD)"

if [ "${ACC_FIRST:-1}" = 1 ]; then
  AC=${ACC_CONC:-10}
  stop_all; wait_gpus acc
  SLOG=$OUT/server_acc_c$AC.log
  start_server $AC 1 $SLOG || { stop_all; log "STOP: accuracy server failed"; exit 1; }
  docker exec -w /logs $C lm_eval --model local-chat-completions --apply_chat_template --fewshot_as_multiturn \
    --model_args model=$MODEL,base_url=http://localhost:${API_PORT:-8890}/v1/chat/completions,num_concurrent=16,max_retries=3,max_gen_toks=16384,tokenized_requests=False,trust_remote_code=True \
    --tasks gsm8k --num_fewshot 5 --log_samples --output_path $OUT/gsm8k_acc > $HOST_DIR/gsm8k_acc.log 2>&1
  flex=$(grep -E 'flexible' $HOST_DIR/gsm8k_acc.log | tail -1 | awk -F'|' '{print $8}' | tr -d ' ')
  strict=$(grep -E 'strict' $HOST_DIR/gsm8k_acc.log | tail -1 | awk -F'|' '{print $8}' | tr -d ' ')
  log "accuracy GSM8K 5-shot: flexible=$flex strict=$strict (real acceptance)"
  server_facts $SLOG
  stop_all
  if ! python3 -c "import sys; sys.exit(0 if float('${flex:-0}' or 0) >= ${ACC_MIN:-0.92} else 1)"; then
    log "STOP: flexible $flex below ${ACC_MIN:-0.92}, perf ladder not run"; exit 1
  fi
fi

for CONC in $CONCS; do
  stop_all; wait_gpus c$CONC
  ART=$OUT/c${CONC}_$(date +%m%d_%H%M)
  SLOG=$OUT/server_c${CONC}.log
  start_server $CONC 0 $SLOG || continue
  log "c$CONC aiperf start, artifacts ${ART/#\/logs/$ROOT/logs}"
  docker exec -e CONC=$CONC -e DURATION=${DURATION:-3600} -e API_PORT=${API_PORT:-8890} $C bash /scripts/agentic/aiperf.sh $ART \
    > $HOST_DIR/aiperf_c${CONC}.log 2>&1
  log "c$CONC aiperf rc=$?"
  docker exec $C python3 /scripts/agentic/summarize.py $ART 2>&1 | tail -n 1 | tee -a $HOST_DIR/ladder.log
  server_facts $SLOG
done
stop_all
log "summary:"
docker exec $C bash -c "python3 /scripts/agentic/summarize.py $OUT/c*_*/" 2>&1 | tee -a $HOST_DIR/ladder.log
log DONE
