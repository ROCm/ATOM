#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

set -euo pipefail

: "${IMAGE_TAG:?IMAGE_TAG is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"

failures=()
results_dir="$GITHUB_WORKSPACE/atomrl/output/regression"

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

run_unit() {
  local container="atomrl_gpu_unit_${GITHUB_RUN_ID}_${GITHUB_RUN_ATTEMPT}"
  mkdir -p "$results_dir"
  docker rm -f "$container" 2>/dev/null || true
  set +e
  docker run --rm --name "$container" \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --network=host --privileged \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$DATA_ROOT/models/Qwen3-8B-Base:/dev/shm/atomrl_weight_sync:ro" \
    -v "$DATA_ROOT/hf:/data/rl_data/hf:ro" \
    -w /opt/atomrl/ATOM \
    "$IMAGE_TAG" bash -lc \
    'python3 -m pytest atomrl/tests/unit atomrl/tests/atomrl \
      -m "gpu or multigpu" -q' \
    2>&1 | tee "$results_dir/gpu-unit.log"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

run_example() {
  local id="$1"
  export CONTAINER_NAME="atomrl-ex-${GITHUB_RUN_ID}-${id}"
  export ATOMRL_CONTAINER="$CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME-release" 2>/dev/null || true
  bash "$GITHUB_WORKSPACE/atomrl/examples/run.sh" "$id" \
    --data-root "$DATA_ROOT" \
    --image "release=${IMAGE_TAG}" \
    --results "$results_dir" \
    --strict
}

printf '## AtomRL DAPO GPU suite\n' >> "$GITHUB_STEP_SUMMARY"
record "gpu-unit" run_unit

for id in 1 2 3 4 5 6 7 9; do
  record "example-${id}" run_example "$id"
  docker rm -f "atomrl-ex-${GITHUB_RUN_ID}-${id}-release" 2>/dev/null || true
done

if ((${#failures[@]})); then
  printf 'Failed DAPO checks: %s\n' "${failures[*]}" >&2
  exit 1
fi
