#!/usr/bin/env bash
set -euo pipefail
export GPOOL_LEASE="${1:?lease id}"
case_name="${2:?case name}"
[[ "${GPOOL_LEASE}" =~ ^L[0-9a-f]{6}$ ]] || exit 2
case "${case_name}" in
  tp4) contract_file=single-node-functional.json; budget=90m ;;
  pp4) contract_file=single-node-pp4-functional.json; budget=90m ;;
  tp4-perf) contract_file=single-node-tp4-performance.json; budget=120m ;;
  tp4-baseline) contract_file=single-node-tp4-baseline.json; budget=120m ;;
  pp4-dep4) contract_file=single-node-pp4-dep4-functional.json; budget=90m ;;
  *) exit 2 ;;
esac
export REPO_ROOT="${3:?checkout path}"
export RUN_DIR="${4:?output path}"
export MODEL_PATH="${5:?model path}"
export JOB_ID="gpool-${GPOOL_LEASE}-${case_name}"
export SRTCTL_JOB_ID="${JOB_ID}-sa-functional"
unset SLURM_JOB_ID SPUR_JOB_ID SPUR_TASK_OFFSET SPUR_PEER_NODES
mkdir -p "${RUN_DIR}/bin" "${RUN_DIR}/logs"
gpool=/share_nfs/yajizhan/gpu-pool/bin/gpool
finish_pool_test() {
  local rc=$? cleanup_rc=0 release_rc=0 id attempt
  trap - EXIT INT TERM HUP
  docker ps -aq --filter "label=srtctl.job=${SRTCTL_JOB_ID}" > "${RUN_DIR}/pool-cleanup-ids.txt" || cleanup_rc=$?
  if [[ "${cleanup_rc}" == 0 ]]; then
    while IFS= read -r id; do
      [[ -z "${id}" ]] || timeout 20 docker rm -f "${id}" >> "${RUN_DIR}/pool-cleanup.log" 2>&1 || cleanup_rc=1
    done < "${RUN_DIR}/pool-cleanup-ids.txt"
  fi
  : > "${RUN_DIR}/gpool-release.log"
  # ROCm can retain VRAM briefly after the last container exits.
  for attempt in {1..13}; do
    release_rc=0
    "${gpool}" release "${GPOOL_LEASE}" --note "M3.1 ${case_name} ATOMesh69ed5e7 workload_rc=${rc} cleanup_rc=${cleanup_rc}" \
      >> "${RUN_DIR}/gpool-release.log" 2>&1 || release_rc=$?
    [[ "${release_rc}" == 3 && "${attempt}" -lt 13 ]] || break
    sleep 10
  done
  printf '%s\n' "${release_rc}" > "${RUN_DIR}/gpool-release.rc"
  printf '%s\n' "${rc}" > "${RUN_DIR}/pool-workload.rc"
  [[ "${cleanup_rc}" == 0 && "${release_rc}" == 0 ]] || rc=1
  printf '%s\n' "${rc}" > "${RUN_DIR}/pool-exit.rc"
  exit "${rc}"
}
trap finish_pool_test EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
python3 "${REPO_ROOT}/.github/scripts/atomesh/pd_gpool_guard.py" prepare
[[ -d "${MODEL_PATH}" ]] || { echo 'Pinned model path is missing'; exit 1; }
python3 - <<'PY'
import socket
for port in (6100, 6132, 7200, 8000):
    with socket.socket() as listener:
        listener.bind(("0.0.0.0", port))
PY
export ATOMESH_ENV_ATOMESH_PR5090_CONTRACT=".github/benchmark/pr5090-router-ab/${contract_file}"
export GITHUB_WORKSPACE="${REPO_ROOT}"
export SRT_FUNCTIONAL_CONTRACT="${ATOMESH_ENV_ATOMESH_PR5090_CONTRACT}"
cd "${REPO_ROOT}"
bash .github/scripts/atomesh/prepare_srt_functional.sh > "${RUN_DIR}/source-prepare.log" 2>&1
mapfile -t network < <(python3 - <<'PY'
import json, subprocess
route = json.loads(subprocess.check_output(["ip", "-j", "route", "get", "10.19.0.25"]))[0]
print(route["prefsrc"])
print(route["dev"])
PY
)
[[ "${#network[@]}" == 2 ]] || exit 1
export ATOMESH_NODE_IP="${network[0]}" MORI_SOCKET_IFNAME="${network[1]}"
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export NCCL_IB_GID_INDEX=1 NCCL_IB_TC=104
export VIDEO_GID="$(getent group video | cut -d: -f3 || true)"
export RENDER_GID="$(getent group render | cut -d: -f3 || true)"
export HOST_IONIC="$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null || true)"
cat > "${RUN_DIR}/bin/nvcc" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == --version ]]; then exec hipcc --version; fi
exit 127
EOF
chmod +x "${RUN_DIR}/bin/nvcc"
export ATOMESH_ROUTER_BUILD_TIMEOUT_SECONDS=1500
timeout --kill-after=120s "${budget}" bash .github/scripts/atomesh/pd_srt_functional.sh
