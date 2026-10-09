#!/usr/bin/env bash
#SBATCH --job-name=plugin-ci-vllm
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job

set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
JOB_ID="${SLURM_JOB_ID:-${SPUR_JOB_ID:-local}}"
RUN_DIR="${LOG_ROOT}/slurm_job-${JOB_ID}"
STATUS_FILE="${RUN_DIR}/slurm-job.rc"
mkdir -p "${RUN_DIR}"
rm -f "${STATUS_FILE}" "${STATUS_FILE}.tmp"

export GITHUB_WORKSPACE="${REPO_ROOT}"
export RESULT_DIR="${RUN_DIR}/results"
mkdir -p "${RESULT_DIR}"

echo "=== plugin CI vLLM Slurm job start: id=${PLUGIN_CI_CELL_ID} job=${JOB_ID} host=$(hostname) ==="
cd "${REPO_ROOT}"
set +e
bash "${REPO_ROOT}/.github/scripts/plugin_ci/run_vllm_ci.sh"
rc=$?
set -e
echo "=== plugin CI vLLM Slurm job finished: rc=${rc} ==="
printf '%s\n' "${rc}" > "${STATUS_FILE}.tmp"
mv "${STATUS_FILE}.tmp" "${STATUS_FILE}"
exit "${rc}"
