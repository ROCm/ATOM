#!/bin/bash
# Host driver, accuracy only: the agentic server config (server.sh, CONC=$ACC_CONC) with real EAGLE3
# acceptance, then GSM8K 5-shot full. Waits for the aiter JIT pre-build and for GPUs held by others.
# Usage: C=<container> TAG=<name> bash acc_only.sh
# Env: KV (fp8) FLY (1) QR (INT4) GPU_MEM (0.95) to server.sh; ACC_CONC (10); BUILD_LOG (container path).
set -u
C=${C:-ruili_m3_stack}
TAG=${TAG:-m3stack_acc}
ROOT=${ROOT:-/home/ruili/code/m3_fp8kv}   # host dir whose logs/ is mounted at /logs
HOST_DIR=$ROOT/logs/agentic/$TAG
OUT=/logs/agentic/$TAG
MODEL=/models/MiniMax-M3-MXFP4
AC=${ACC_CONC:-10}
mkdir -p $HOST_DIR
log() { echo "$(date '+%F %T') $*" | tee -a $HOST_DIR/acc.log; }
health() { curl -s -m 5 localhost:8890/health -o /dev/null -w '%{http_code}'; }
CID=$(docker inspect -f '{{.Id}}' $C) || exit 1

stop_all() {
  docker exec $C bash -c "pkill -f '[l]m_eval'; pkill -f '[a]tom.entrypoints.openai_server'" 2>/dev/null
  for i in $(seq 1 60); do
    n=$(docker exec $C bash -c "ps -eo args | grep -cE '[o]penai_server|[l]m_eval'")
    [ "${n:-0}" = 0 ] && break; sleep 5
  done
  sleep 20
}

busy_gpus() {  # KFD processes that are not ours
  rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | while read p; do
    grep -q "$CID" /proc/$p/cgroup 2>/dev/null || echo $p
  done
}

if [ -n "${BUILD_LOG:-}" ]; then
  until docker exec $C grep -q '^EXIT=' $BUILD_LOG 2>/dev/null; do sleep 30; done
  log "aiter pre-build: $(docker exec $C tail -n 1 $BUILD_LOG)"
fi

first=1
while [ -n "$(busy_gpus)" ]; do
  [ $first = 1 ] && log "waiting for GPUs, used by pids $(busy_gpus | tr '\n' ' ')"
  first=0; sleep 120
done
[ $first = 0 ] && log "GPUs free"

log "start $TAG c$AC KV=${KV:-fp8} FLY=${FLY:-1} QR=${QR:-INT4}" \
  "ATOM=$(docker exec $C git -C /app/ATOM log -1 --format='%h(%D)')" \
  "aiter=$(docker exec $C git -C /app/aiter log -1 --format='%h(%D)')"
stop_all
SLOG=$OUT/server_acc_c$AC.log
docker exec -e CONC=$AC -e ACC=1 -e GPU_MEM=${GPU_MEM:-0.95} -e KV=${KV:-fp8} -e FLY=${FLY:-1} -e QR=${QR:-INT4} \
  -e TG=${TG:-0} -e TG_SIZES=${TG_SIZES:-} \
  $C bash /scripts/agentic/server.sh $SLOG
log "server starting, log ${SLOG/#\/logs/$ROOT/logs}"
t0=$(date +%s)
for i in $(seq 1 540); do
  [ "$(health)" = 200 ] && break
  if ! docker exec $C pgrep -f '[a]tom.entrypoints.openai_server' >/dev/null; then
    log "ABORT server died: $(docker exec $C grep -m1 -oE 'exitcode=-?[0-9]+|[A-Za-z]+Error: .{0,160}' $SLOG)"
    stop_all; exit 1
  fi
  sleep 10
done
[ "$(health)" = 200 ] || { log "ABORT server not healthy after 90 min"; stop_all; exit 1; }
log "server healthy after $(( $(date +%s) - t0 )) s"

ans=$(curl -s -m 600 localhost:8890/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23? Answer briefly.\"}],\"max_tokens\":1024,\"temperature\":0}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][-80:])" 2>&1 | tr '\n' ' ')
log "smoke: $ans"

t0=$(date +%s)
docker exec -w /logs $C lm_eval --model local-chat-completions --apply_chat_template --fewshot_as_multiturn \
  --model_args model=$MODEL,base_url=http://localhost:8890/v1/chat/completions,num_concurrent=16,max_retries=3,max_gen_toks=16384,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k --num_fewshot 5 --log_samples --output_path $OUT/gsm8k_acc > $HOST_DIR/gsm8k_acc.log 2>&1
flex=$(grep -E 'flexible' $HOST_DIR/gsm8k_acc.log | tail -1 | awk -F'|' '{print $8}' | tr -d ' ')
strict=$(grep -E 'strict' $HOST_DIR/gsm8k_acc.log | tail -1 | awk -F'|' '{print $8}' | tr -d ' ')
log "accuracy GSM8K 5-shot: flexible=$flex strict=$strict (real acceptance, $(( $(date +%s) - t0 )) s)"
docker exec $C bash -c "echo routes: \$(grep -hoE 'pa_decode -> [a-z]+' $SLOG | sort | uniq -c | tr -s ' ' | tr '\n' ';');
  echo errors: \$(grep -cE 'Traceback|RuntimeError' $SLOG);
  grep -iE 'acceptance' $SLOG | tail -n 2 | cut -c1-220" | tee -a $HOST_DIR/acc.log
stop_all
log DONE
