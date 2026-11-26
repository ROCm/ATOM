#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

set -euo pipefail

: "${LUMEN_IMAGE:?LUMEN_IMAGE is required}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${LUMEN_LLAMA2_MODEL:?LUMEN_LLAMA2_MODEL is required}"
: "${LUMEN_QWEN3_MODEL:?LUMEN_QWEN3_MODEL is required}"
: "${LUMEN_NPY_DATA:?LUMEN_NPY_DATA is required}"
: "${LUMEN_ALPACA_DATA:?LUMEN_ALPACA_DATA is required}"

LUMEN_LLAMA2_MODEL_READY="${LUMEN_LLAMA2_MODEL_READY:-0}"
LUMEN_QWEN3_MODEL_READY="${LUMEN_QWEN3_MODEL_READY:-0}"
failures=()
log_root="$GITHUB_WORKSPACE/lumen-ci-logs"
result_root="${RUNNER_TEMP}/lumen-ci-results"
mkdir -p "$log_root" "$result_root"

device_flags=(--device=/dev/dri)
if [[ -f /etc/podinfo/gha-render-devices ]]; then
  read -r -a device_flags < /etc/podinfo/gha-render-devices
fi
group_flags=(--group-add video)
if [[ "${CONTAINER_ENGINE:-docker}" == podman ]]; then
  group_flags=(--group-add keep-groups)
fi

cleanup_case() {
  local container="$1"
  docker rm -f "$container" 2>/dev/null || true
}

record() {
  local name="$1"
  shift
  if "$@"; then
    printf 'PASS %s\n' "$name" | tee -a "$GITHUB_STEP_SUMMARY"
  else
    failures+=("$name")
    printf 'FAIL %s\n' "$name" | tee -a "$GITHUB_STEP_SUMMARY"
  fi
}

model_unavailable() {
  echo "Required model is unavailable: $1" >&2
  return 1
}

record_model_case() {
  local name="$1" ready="$2" model="$3"
  shift 3
  if [[ "$ready" == 1 ]]; then
    record "$name" "$@"
  else
    record "$name" model_unavailable "$model"
  fi
}

run_unit() {
  local container="lumen_readme_${GITHUB_RUN_ID}_unit"
  local log_dir="$log_root/case-unit"
  mkdir -p "$log_dir"
  cleanup_case "$container"
  set +e
  docker run --rm --name "$container" \
    --device=/dev/kfd "${device_flags[@]}" "${group_flags[@]}" \
    --ipc=host --privileged \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -w /workspace/ATOM \
    "$LUMEN_IMAGE" bash -lc "python3 -m pytest atomrl/tests/lumen -q" \
    2>&1 | tee "$log_dir/run.log"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

run_host_case() {
  local id="$1" command="$2" model="${3:-none}" data="${4:-none}"
  local container="lumen_readme_${GITHUB_RUN_ID}_${id}"
  local log_dir="$log_root/case-${id}"
  local result_dir="$result_root/case-${id}"
  mkdir -p "$log_dir" "$result_dir"
  cleanup_case "$container"

  export IMAGE="$LUMEN_IMAGE"
  export CONTAINER_NAME="$container"
  export RESULTS_DIR="$result_dir"
  export HOST_RESULTS="$result_dir"
  export LOG_DIR="$result_dir"
  unset HOST_MODEL HOST_DATA TRAIN_FILE VAL_FILE
  [[ "$model" == llama2 ]] && export HOST_MODEL="$LUMEN_LLAMA2_MODEL"
  [[ "$model" == qwen3 ]] && export HOST_MODEL="$LUMEN_QWEN3_MODEL"
  [[ "$data" == npy ]] && export HOST_DATA="$LUMEN_NPY_DATA"
  if [[ "$data" == alpaca ]]; then
    export HOST_DATA="$LUMEN_ALPACA_DATA"
    export TRAIN_FILE=train.jsonl
    export VAL_FILE=valid.jsonl
  fi

  set +e
  (
    cd "$GITHUB_WORKSPACE/atomrl/lumen"
    bash -c "$command"
  ) 2>&1 | tee "$log_dir/run.log"
  local status=${PIPESTATUS[0]}
  set -e
  cleanup_case "$container"
  return "$status"
}

run_container_case_13() {
  local id=13 container="lumen_readme_${GITHUB_RUN_ID}_13"
  local log_dir="$log_root/case-13"
  local result_dir="$result_root/case-13"
  mkdir -p "$log_dir" "$result_dir"
  cleanup_case "$container"

  set +e
  docker run --rm --name "$container" \
    --device=/dev/kfd "${device_flags[@]}" "${group_flags[@]}" \
    --ipc=host --privileged \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "$GITHUB_WORKSPACE:/workspace/ATOM" \
    -v "$LUMEN_QWEN3_MODEL:/models/Qwen3-8B:ro" \
    -v "$LUMEN_ALPACA_DATA:/ci-data:ro" \
    -v "$result_dir:/results" \
    -w /workspace/ATOM/atomrl/lumen \
    "$LUMEN_IMAGE" bash -lc \
    "LUMEN_BUILD_EXTENSIONS=1 python -m pip install --no-build-isolation --no-deps -e /workspace/ATOM/atomrl &&
     MODEL_PATH=/models/Qwen3-8B TRAIN_DATA_PATH=/ci-data/train.jsonl \
     VALID_DATA_PATH=/ci-data/valid.jsonl RESULTS_DIR=/results \
     TRAIN_STEPS=20 WARMUP_STEPS=2 bash examples/qwen3/run_qwen3_fsdp_mxfp4_sft.sh" \
    2>&1 | tee "$log_dir/run.log"
  local status=${PIPESTATUS[0]}
  set -e
  cleanup_case "$container"
  return "$status"
}

printf '## Lumen GPU suite\n' >> "$GITHUB_STEP_SUMMARY"
record "unit" run_unit
record "case-1" run_host_case 1 \
  "PRECISION=bf16 TRAIN_STEPS=20 bash examples/llama2/run_pretrain_llama2_7b.sh"
record "case-2" run_host_case 2 \
  "TRAIN_STEPS=20 bash examples/llama2/run_pretrain_llama2_7b.sh"
record "case-3" run_host_case 3 \
  "MODEL=llama2 TRAIN_STEPS=20 bash examples/run_fp8_blockwise_pretrain.sh"
record_model_case "case-4" "$LUMEN_LLAMA2_MODEL_READY" "$LUMEN_LLAMA2_MODEL" \
  run_host_case 4 \
  "MODE=fp8_blockwise SEQ_LEN=2048 TRAIN_STEPS=20 bash examples/llama2/run_fsdp_lora_7b.sh" \
  llama2 npy
record "case-5" run_host_case 5 \
  "PRECISION=bf16 TRAIN_STEPS=20 bash examples/llama31/run_pretrain_llama31_8b.sh"
record "case-6" run_host_case 6 \
  "TRAIN_STEPS=20 bash examples/llama31/run_pretrain_llama31_8b.sh"
record "case-7" run_host_case 7 \
  "MODEL=llama31 TRAIN_STEPS=20 bash examples/run_fp8_blockwise_pretrain.sh"
record "case-8" run_host_case 8 \
  "PRECISION=bf16 TRAIN_STEPS=20 bash examples/qwen3/run_pretrain_qwen3_8b.sh"
record "case-9" run_host_case 9 \
  "TRAIN_STEPS=20 bash examples/qwen3/run_pretrain_qwen3_8b.sh"
record "case-10" run_host_case 10 \
  "MODEL=qwen3 TRAIN_STEPS=20 bash examples/run_fp8_blockwise_pretrain.sh"
record_model_case "case-11" "$LUMEN_QWEN3_MODEL_READY" "$LUMEN_QWEN3_MODEL" \
  run_host_case 11 \
  "MAX_STEPS=20 EVAL_INTERVAL=20 bash examples/qwen3/run_qwen3_fsdp.sh" \
  qwen3 alpaca
record "case-12" run_host_case 12 \
  "TRAIN_STEPS=20 bash examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh"
record_model_case "case-13" "$LUMEN_QWEN3_MODEL_READY" "$LUMEN_QWEN3_MODEL" \
  run_container_case_13

if ((${#failures[@]})); then
  printf 'Failed Lumen checks: %s\n' "${failures[*]}" >&2
  exit 1
fi
