#!/usr/bin/env bash
set -euo pipefail
container="k3-ipc-probe-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
cleanup() {
  local rc=$? cleanup_rc=0
  timeout --kill-after=2s 10s docker rm -f "${container}" > "${PROBE_RUN_DIR}/cleanup.log" 2>&1 || cleanup_rc=$?
  printf '%s\n' "${cleanup_rc}" > "${PROBE_RUN_DIR}/cleanup.rc"
  if [[ "${rc}" -eq 0 && "${cleanup_rc}" -ne 0 ]]; then
    rc="${cleanup_rc}"
  fi
  printf '%s\n' "${rc}" > "${PROBE_RUN_DIR}/probe.rc"
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
exec > "${PROBE_RUN_DIR}/probe.log" 2>&1
hostname
date -u
image=vllm/vllm-openai-rocm:nightly@sha256:659b28319fef4ea0e3d8f33e25b4c35d6f663f5d818a5b2fa06dceaf859234e4
timeout --kill-after=5s 90s docker pull "${image}"
groups=()
for group in video render; do
  gid="$(getent group "${group}" | cut -d: -f3 || true)"
  [[ -z "${gid}" ]] || groups+=(--group-add "${gid}")
done
timeout --kill-after=5s 240s docker run --name "${container}" \
  --user "$(id -u):$(id -g)" "${groups[@]}" \
  --device /dev/kfd --device /dev/dri --ipc host \
  --security-opt seccomp=unconfined \
  -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${GITHUB_WORKSPACE}/.github/scripts/k3-ipc-probe:/probe:ro" \
  -v "${PROBE_RUN_DIR}:/results" --entrypoint bash "${image}" -lc '
    set -euo pipefail
    timeout 100s /opt/rocm/bin/hipcc -O2 --offload-arch=gfx950 /probe/event_destroy.cpp -o /results/event_destroy
    timeout 30s /results/event_destroy > /results/event-destroy.csv
    timeout 100s python3 -u /probe/lifetime.py
  '
