#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/slurm-24.05.5.1/bin:$PATH"
: "${PR:?}" "${HEAD_SHA:?}" "${BASE_SHA:?}" "${GITHUB_WORKSPACE:?}"
[[ "$PR" =~ ^331[12]$ && "$HEAD_SHA" =~ ^[0-9a-f]{40}$ && "$BASE_SHA" =~ ^[0-9a-f]{40}$ ]]
export SLURM_ACCOUNT=amd-atom SLURM_PARTITION=amd-spur SLURM_QOS=amd-atom-qos
export SLURM_SUBMIT_RUNNER=atomesh-cicd
shared_base=/it-share/ATOMESH_LOG
export MODEL_ROOT=/mnt/models
submit_host=$(hostname)
if [[ "${submit_host%%.*}" == pit2-vm-amd-xl-02 ]]; then
  shared_base=/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG
  export MODEL_ROOT=/share_nfs/models
fi
export SHARED_ROOT="${shared_base}/pr-review-${GITHUB_RUN_ID:-dryrun}-${GITHUB_RUN_ATTEMPT:-1}-${PR}"
export SHARED_SOURCE="$SHARED_ROOT/source" LOG_ROOT="$SHARED_ROOT/logs"
export SLURM_JOB_NAME="pr-review-${PR}-${GITHUB_RUN_ID:-dryrun}-${GITHUB_RUN_ATTEMPT:-1}"
export SLURM_OUTPUT="$LOG_ROOT/slurm-%j.out" SLURM_ERROR="$LOG_ROOT/slurm-%j.err"
SBATCH_CMD=(sbatch --parsable --exclusive --export=ALL --account "$SLURM_ACCOUNT"
  --partition "$SLURM_PARTITION" --qos "$SLURM_QOS" --nodes 1 --ntasks 1
  --ntasks-per-node 1 --cpus-per-task 64 --gres gpu:8 --time 04:00:00
  --job-name "$SLURM_JOB_NAME" --chdir "$SHARED_SOURCE"
  --output "$SLURM_OUTPUT" --error "$SLURM_ERROR"
  "$SHARED_SOURCE/_review_harness/.github/scripts/pr_prefill_slurm_job.sh")
if [[ "${1:-}" == --dry-run ]]; then
  printf '%q ' "${SBATCH_CMD[@]}"
  printf '\nsource=%s\nmodel_root=%s\n' "$SHARED_SOURCE" "$MODEL_ROOT"
  exit 0
fi
mkdir -p "$GITHUB_WORKSPACE/review-evidence"
printf '%s\n' "$SHARED_ROOT" > "$GITHUB_WORKSPACE/review-evidence/shared-root.txt"
test ! -e "$SHARED_ROOT"
workspace_real=$(realpath "$GITHUB_WORKSPACE")
shared_real=$(realpath -m "$SHARED_ROOT")
[[ "$shared_real" != "$workspace_real" && "$shared_real" != "$workspace_real/"* ]]
mkdir -p "$SHARED_SOURCE" "$LOG_ROOT"
cp -a "$GITHUB_WORKSPACE/." "$SHARED_SOURCE/"
export CURRENT_USER="$(id -un)"
export SLURM_CANCEL_HELPER="$GITHUB_WORKSPACE/review-evidence/slurm-cancel.sh"
export SLURM_LOG_POLL_INTERVAL=30
source "$GITHUB_WORKSPACE/.github/scripts/slurm_submit_helpers.sh"
detect_slurm_backend
install_slurm_cancel_traps
write_slurm_cancel_helper ""
printf '%q ' "${SBATCH_CMD[@]}"
printf '\n'
SLURM_JOB_ACTIVE=1
submitted=$("${SBATCH_CMD[@]}")
JOB_ID=$(parse_sbatch_job_id "$submitted")
printf '%s\n' "$JOB_ID" > "$GITHUB_WORKSPACE/review-evidence/slurm-job-id.txt"
write_slurm_cancel_helper "$JOB_ID"
set_slurm_job_log_paths "$JOB_ID"
SLURM_STATUS_DIR="$LOG_ROOT"
SLURM_STATUS_RANKS=1
monitor_slurm_job "$JOB_ID"
read_slurm_exit_code "$JOB_ID"
printf 'state=%s\nexit=%s\n' "$SLURM_STATE" "$SLURM_EXIT_CODE" \
  > "$GITHUB_WORKSPACE/review-evidence/slurm-accounting.txt"
read_slurm_status_files "$LOG_ROOT" 1
SLURM_JOB_ACTIVE=0
printf 'state=%s\nexit=%s\nrc=%s\n' "$SLURM_STATE" "$SLURM_EXIT_CODE" "$SLURM_JOB_RC" \
  > "$GITHUB_WORKSPACE/review-evidence/slurm-result.txt"
exit "$SLURM_JOB_RC"
