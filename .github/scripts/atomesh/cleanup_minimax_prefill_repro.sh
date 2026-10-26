#!/usr/bin/env bash
# Source after slurm_submit_helpers.sh; runs only on the CI submit runner.
cleanup_previous_minimax_repro() {
  local previous_run="$1" rows job_id job_name owner state remaining
  local -a query=(squeue)
  [[ -n "${previous_run}" ]] || { echo 'No previous reproduction run specified.'; return 0; }
  if [[ ! "${previous_run}" =~ ^[1-9][0-9]+$ || "${previous_run}" == "${GITHUB_RUN_ID}" ]]; then
    echo 'ERROR: cleanup requires a different, numeric GitHub run ID.' >&2
    return 1
  fi
  if [[ "${USES_SPUR_CONTROLLER}" == 1 && -n "${SPUR_CONTROLLER_ADDR}" ]]; then
    query+=(--controller "${SPUR_CONTROLLER_ADDR}")
  fi
  query+=(--noheader '--format=%A|%j|%u|%T')
  rows="$(run_slurm_query "${query[@]}")" || return 1
  local found=0
  while IFS='|' read -r job_id job_name owner state; do
    # squeue may pad fields; strip surrounding whitespace only.
    job_id="${job_id//[[:space:]]/}"
    job_name="${job_name//[[:space:]]/}"
    owner="${owner//[[:space:]]/}"
    [[ "${job_name}" =~ ^minimax-prefill-repro-${previous_run}-[0-9]+$ ]] || continue
    if [[ ! "${job_id}" =~ ^[0-9]+$ || "${owner}" != "${CURRENT_USER}" ]]; then
      echo "ERROR: refusing to cancel unmatched owner/id: ${job_id}|${job_name}|${owner}" >&2
      return 1
    fi
    found=1
    echo "Previous reproduction: job=${job_id} name=${job_name} owner=${owner} state=${state}"
    run_scancel "${job_id}"
    wait_for_slurm_cancel "${job_id}" TERM || return 1
    remaining="$(query_slurm_job "${job_id}")" || return 1
    if [[ -n "${remaining}" ]]; then
      echo "ERROR: previous reproduction is still queued: ${remaining}" >&2
      return 1
    fi
    echo "Confirmed previous reproduction ${job_id} is no longer in the queue."
  done <<< "${rows}"
  if [[ "${found}" == 0 ]]; then
    echo "No queued/running reproduction job for GitHub run ${previous_run}."
  fi
}
