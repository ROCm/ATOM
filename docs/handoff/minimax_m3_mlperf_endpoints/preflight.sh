#!/bin/bash
# preflight.sh: read-only checks before a run. Prints PREFLIGHT PASS when every check is OK.
set -uo pipefail
cd "$(dirname "$0")" && . ./config.env && . ./lib.sh
fails=0
check() {  # check <what> <command...>
  local what=$1
  shift
  if "$@" > /dev/null 2>&1; then echo "OK   $what"; else echo "FAIL $what"; fails=$((fails + 1)); fi
}
arch_ok() { [[ "$(gpu_arch)" =~ ^gfx(950|1250)$ ]]; }
# GPUS names NUM_GPUS distinct GPUs, all on the node; or, empty, the node has exactly NUM_GPUS.
gpus_ok() {
  local want=$NUM_GPUS
  [ -n "${GPUS:-}" ] && want=$(echo "$GPUS" | tr ',' '\n' | sort -u | wc -l)
  [ "$want" = "$NUM_GPUS" ] && [ "$(gpu_ids | wc -l)" = "$NUM_GPUS" ]
}
numa_off() { [ "$(cat /proc/sys/kernel/numa_balancing)" = 0 ]; }

echo "node: $(gpu_arch 2> /dev/null || echo 'no GPUs'), $(node_gpu_count) GPUs; this run: GPUS=${GPUS:-all} (KFD gpu_id $(gpu_ids | tr '\n' ' '))"
check "arch gfx950 (MI355X) or gfx1250 (MI455X)" arch_ok
check "GPUS selects $NUM_GPUS GPUs" gpus_ok
check "the selected GPUs are idle (no process over 1 GiB)" gpus_idle
check "NUMA balancing off (kernel.numa_balancing=0)" numa_off
check "image $IMAGE present" docker image inspect "$IMAGE"
check "model $MODEL_PATH" test -f "$MODEL_PATH/config.json"
check "draft $DRAFT_PATH" test -f "$DRAFT_PATH/config.json"
check "ATOM tree $ATOM_DIR" test -f "$ATOM_DIR/atom/models/minimax_m3/model.py"
check "aiter tree $AITER_DIR" test -d "$AITER_DIR/aiter"
check "client ready (setup_client.sh)" test -x "$CLIENT/endpoints/.venv/bin/inference-endpoint"
check "dataset present" test -f "$CLIENT/data/mlperf_agentic_inference_dataset.jsonl"
for p in "$API_PORT" "$ENGINE_PORT"; do
  check "port $p free" bash -c "! (exec 3<> /dev/tcp/127.0.0.1/$p) 2> /dev/null"
done
[ "$fails" = 0 ] && echo "PREFLIGHT PASS" || { echo "PREFLIGHT FAIL ($fails)"; exit 1; }
