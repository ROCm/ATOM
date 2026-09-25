#!/usr/bin/env bash
set -euo pipefail
RESULT_DIR="${GITHUB_WORKSPACE}/ipc-probe-results/${GITHUB_RUN_ATTEMPT}"
mkdir -p "${RESULT_DIR}"
PROBE_RUN_DIR="/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG/k3-ipc-probe-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
mkdir -p "${PROBE_RUN_DIR}"
chmod 0777 "${PROBE_RUN_DIR}"
export PROBE_RUN_DIR
CURRENT_USER="$(id -un)"
ATOMESH_CELL_ID=k3-ipc-probe
SLURM_JOB_NAME="k3-ipc-probe-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
SLURM_OUTPUT="${PROBE_RUN_DIR}/slurm-%j.out"
SLURM_ERROR="${PROBE_RUN_DIR}/slurm-%j.err"
SLURM_CANCEL_HELPER="${RESULT_DIR}/slurm-cancel.sh"
SLURM_ACCOUNT=amd-atom
SLURM_LOG_POLL_INTERVAL=10
source "${GITHUB_WORKSPACE}/.github/scripts/slurm_submit_helpers.sh"
detect_slurm_backend
install_slurm_cancel_traps
write_slurm_cancel_helper ""
controller_args=()
if [[ "${USES_SPUR_CONTROLLER}" == 1 && -n "${SPUR_CONTROLLER_ADDR}" ]]; then
  controller_args=(--controller "${SPUR_CONTROLLER_ADDR}")
fi
slurm_node_selection_args "pit2-p03-g03,pit2-p03-g13,pit2-p03-g19,pit2-p03-g42" 1
submission="$(sbatch "${controller_args[@]}" --parsable --exclusive --export=ALL \
  --job-name "${SLURM_JOB_NAME}" --account amd-atom --qos amd-atom-qos \
  --partition amd-spur --nodes 1 --ntasks 1 --cpus-per-task 4 --gres gpu:1 \
  --time 00:06:00 "${SLURM_NODE_SELECTION_ARGS[@]}" \
  --output "${SLURM_OUTPUT}" --error "${SLURM_ERROR}" \
  "${GITHUB_WORKSPACE}/.github/scripts/k3-ipc-probe/rank.sh")"
printf '%s\n' "${submission}"
JOB_ID="$(parse_sbatch_job_id "${submission}")"
SLURM_JOB_ACTIVE=1
printf '%s\n' "${JOB_ID}" > "${RESULT_DIR}/job-id.txt"
write_slurm_cancel_helper "${JOB_ID}"
set_slurm_job_log_paths "${JOB_ID}"
monitor_slurm_job "${JOB_ID}"
read_slurm_exit_code "${JOB_ID}"
SLURM_JOB_ACTIVE=0
printf '%s|%s\n' "${SLURM_STATE}" "${SLURM_EXIT_CODE}" > "${PROBE_RUN_DIR}/scheduler-result.txt"
bash "${GITHUB_WORKSPACE}/.github/scripts/atomesh/pd_collect_logs.sh" "${PROBE_RUN_DIR}" "${RESULT_DIR}"
[[ -s "${PROBE_RUN_DIR}/probe.rc" ]]
[[ "$(cat "${PROBE_RUN_DIR}/probe.rc")" == 0 ]]
[[ "$(cat "${PROBE_RUN_DIR}/cleanup.rc")" == 0 ]]
exit "${SLURM_JOB_RC}"
