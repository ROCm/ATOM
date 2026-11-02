#!/bin/bash
# preflight.sh: read-only checks before a run. Prints PREFLIGHT PASS when every check is OK.
set -uo pipefail
cd "$(dirname "$0")" && . ./config.env
fails=0
check() {  # check <what> <command...>
  local what=$1
  shift
  if "$@" > /dev/null 2>&1; then echo "OK   $what"; else echo "FAIL $what"; fails=$((fails + 1)); fi
}
gpus() { [ "$(grep -l '^simd_count [1-9]' /sys/class/kfd/kfd/topology/nodes/*/properties 2> /dev/null | wc -l)" = "$NUM_GPUS" ]; }
idle() { [ -d /sys/class/kfd/kfd/proc ] && [ -z "$(ls /sys/class/kfd/kfd/proc)" ]; }
numa_off() { [ "$(cat /proc/sys/kernel/numa_balancing)" = 0 ]; }

check "amdgpu loaded, $NUM_GPUS GPUs" gpus
check "no GPU processes (KFD) right now" idle
check "NUMA balancing off (kernel.numa_balancing=0)" numa_off
check "image $IMAGE present" docker image inspect "$IMAGE"
check "model $MODEL_HOST_DIR/$MODEL" test -f "$MODEL_HOST_DIR/$MODEL/config.json"
check "draft $MODEL_HOST_DIR/$DRAFT" test -f "$MODEL_HOST_DIR/$DRAFT/config.json"
check "ATOM tree $ATOM_DIR" test -f "$ATOM_DIR/atom/models/minimax_m3/model.py"
check "aiter tree $AITER_DIR" test -d "$AITER_DIR/aiter"
check "client ready (setup_client.sh)" test -x "$CLIENT/endpoints/.venv/bin/inference-endpoint"
check "dataset present" test -f "$CLIENT/data/mlperf_agentic_inference_dataset.jsonl"
check "API port $API_PORT free" bash -c "! (exec 3<> /dev/tcp/127.0.0.1/$API_PORT) 2> /dev/null"
[ "$fails" = 0 ] && echo "PREFLIGHT PASS" || { echo "PREFLIGHT FAIL ($fails)"; exit 1; }
