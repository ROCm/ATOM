#!/usr/bin/env bash
#SBATCH --job-name=plugin-ci-sglang
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job

set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
JOB_ID="${SLURM_JOB_ID:-local}"
RUN_DIR="${LOG_ROOT}/slurm_job-${JOB_ID}"
mkdir -p "${RUN_DIR}"

export GITHUB_WORKSPACE="${REPO_ROOT}"
export RESULT_DIR="${RUN_DIR}/results"
mkdir -p "${RESULT_DIR}"

echo "=== plugin CI SGLang Slurm job start: id=${PLUGIN_CI_CELL_ID} job=${JOB_ID} host=$(hostname) ==="
cd "${REPO_ROOT}"
bash "${REPO_ROOT}/.github/scripts/plugin_ci/run_sglang_ci.sh"
rc=$?
echo "=== plugin CI SGLang Slurm job finished: rc=${rc} ==="
exit "${rc}"
