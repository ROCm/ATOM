#!/usr/bin/env bash

# Shared helpers for submitting, monitoring, and cancelling Slurm jobs.
# Callers configure the SLURM_* variables and SLURM_CANCEL_HELPER before
# sourcing this file.

JOB_ID="${JOB_ID:-}"
SLURM_JOB_ACTIVE="${SLURM_JOB_ACTIVE:-0}"
SCANCEL_SENT="${SCANCEL_SENT:-0}"
SLURM_LOG_POLL_INTERVAL="${SLURM_LOG_POLL_INTERVAL:-30}"
USES_SPUR_CONTROLLER="${USES_SPUR_CONTROLLER:-0}"
SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR:-}"
SPUR_ACCOUNTING_ADDR="${SPUR_ACCOUNTING_ADDR:-}"

run_scancel() {
  local -a scancel_cmd=(scancel)
  if [[ "${USES_SPUR_CONTROLLER}" == "1" ]]; then
    scancel_cmd+=(--controller "${SPUR_CONTROLLER_ADDR}")
  fi
  scancel_cmd+=("$@")

  if command -v timeout >/dev/null 2>&1; then
    timeout "${SLURM_SCANCEL_TIMEOUT_SECONDS:-8}" "${scancel_cmd[@]}" || true
  else
    "${scancel_cmd[@]}" || true
  fi
}

scancel_slurm_job_by_name() {
  if [[ -z "${SLURM_JOB_NAME:-}" ]]; then
    return 0
  fi

  echo "=== cancelling Slurm job by name ${SLURM_JOB_NAME} user=${CURRENT_USER} ===" >&2
  run_scancel --user "${CURRENT_USER}" --name "${SLURM_JOB_NAME}"
}

query_slurm_job() {
  local job_id="$1"
  local -a squeue_cmd=(squeue)
  local output

  if [[ "${USES_SPUR_CONTROLLER}" == "1" ]]; then
    squeue_cmd+=(--controller "${SPUR_CONTROLLER_ADDR}")
  fi

  if ! output="$("${squeue_cmd[@]}" --noheader --format="%A|%T|%M|%D|%R" 2>&1)"; then
    echo "ERROR: unable to query Slurm job ${job_id}: ${output}" >&2
    return 2
  fi

  awk -F'|' -v job_id="${job_id}" '
    {
      current_job_id = $1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", current_job_id)
      if (current_job_id == job_id) {
        print
        exit
      }
    }
  ' <<< "${output}"
}

slurm_job_in_queue() {
  local job_id="$1"
  local job_line rc

  if job_line="$(query_slurm_job "${job_id}")"; then
    [[ -n "${job_line}" ]]
    return
  else
    rc=$?
    return "${rc}"
  fi
}

wait_for_slurm_cancel() {
  local job_id="$1"
  local initial_signal="$2"
  local deadline=$(( $(date +%s) + ${SLURM_CANCEL_WAIT_SECONDS:-60} ))
  local kill_deadline query_rc

  while true; do
    if slurm_job_in_queue "${job_id}"; then
      :
    else
      query_rc=$?
      if [[ "${query_rc}" -eq 1 ]]; then
        break
      fi
      echo "WARNING: retrying failed Slurm queue query for job ${job_id}" >&2
    fi
    if [[ "$(date +%s)" -ge "${deadline}" ]]; then
      echo "=== Slurm job ${job_id} still queued after ${initial_signal}; sending KILL ===" >&2
      run_scancel --signal=KILL "${job_id}"
      kill_deadline=$(( $(date +%s) + ${SLURM_CANCEL_KILL_WAIT_SECONDS:-30} ))
      while [[ "$(date +%s)" -lt "${kill_deadline}" ]]; do
        if slurm_job_in_queue "${job_id}"; then
          :
        else
          query_rc=$?
          [[ "${query_rc}" -eq 1 ]] && break
        fi
        sleep 5
      done
      break
    fi
    sleep 5
  done
}

scancel_slurm_job() {
  local reason="$1"
  if [[ "${SCANCEL_SENT}" == "1" ]]; then
    return 0
  fi
  if [[ "${SLURM_JOB_ACTIVE}" != "1" && -z "${JOB_ID}" && -z "${SLURM_JOB_NAME:-}" ]]; then
    return 0
  fi

  SCANCEL_SENT=1
  if command -v scancel >/dev/null 2>&1; then
    if [[ -n "${JOB_ID}" ]]; then
      echo "=== cancelling Slurm job ${JOB_ID}: ${reason} ===" >&2
      run_scancel "${JOB_ID}"
      wait_for_slurm_cancel "${JOB_ID}" "TERM" || true
    else
      echo "=== cancelling Slurm job before id was recorded: ${reason} ===" >&2
      scancel_slurm_job_by_name
    fi
  else
    echo "WARNING: scancel not found; unable to cancel Slurm job ${JOB_ID:-${SLURM_JOB_NAME:-unknown}}" >&2
  fi
}

parse_sbatch_job_id() {
  local output="$1"
  output="${output//$'\r'/}"

  if [[ "${output}" =~ ^[[:space:]]*([0-9]+)(\;.*)?[[:space:]]*$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi

  if [[ "${output}" =~ Submitted[[:space:]]+batch[[:space:]]+job[[:space:]]+([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi

  echo "ERROR: unable to parse Slurm job id from sbatch output: ${output}" >&2
  return 1
}

on_slurm_cancel() {
  local signal="$1"
  local rc="$2"
  scancel_slurm_job "received ${signal}"
  exit "${rc}"
}

on_slurm_exit() {
  local rc=$?
  if [[ "${rc}" -ne 0 && "${SLURM_JOB_ACTIVE}" == "1" ]]; then
    scancel_slurm_job "exiting rc=${rc}"
  fi
}

install_slurm_cancel_traps() {
  trap on_slurm_exit EXIT
  trap 'on_slurm_cancel HUP 129' HUP
  trap 'on_slurm_cancel INT 130' INT
  trap 'on_slurm_cancel TERM 143' TERM
}

set_slurm_job_log_paths() {
  local job_id="$1"
  SLURM_JOB_OUTPUT="${SLURM_OUTPUT//%j/${job_id}}"
  SLURM_JOB_ERROR="${SLURM_ERROR//%j/${job_id}}"
  echo "slurm_job_id=${job_id}"
  echo "slurm_output=${SLURM_JOB_OUTPUT}"
  echo "slurm_error=${SLURM_JOB_ERROR}"
}

write_slurm_cancel_helper() {
  local job_id="${1:-}"
  local helper="${SLURM_CANCEL_HELPER:?SLURM_CANCEL_HELPER must be set}"

  mkdir -p "$(dirname "${helper}")"
  {
    cat <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EOF
    printf 'job_id=%q\n' "${job_id}"
    printf 'job_name=%q\n' "${SLURM_JOB_NAME}"
    printf 'current_user=%q\n' "${CURRENT_USER}"
    printf 'controller=%q\n' "${SPUR_CONTROLLER_ADDR}"
    printf 'uses_spur=%q\n' "${USES_SPUR_CONTROLLER}"
    cat <<'EOF'

run_scancel() {
  command -v scancel >/dev/null 2>&1 || return 0
  local -a cmd=(scancel)
  if [[ "${uses_spur}" == "1" ]]; then
    cmd+=(--controller "${controller}")
  fi
  cmd+=("$@")
  if command -v timeout >/dev/null 2>&1; then
    timeout "${SLURM_SCANCEL_TIMEOUT_SECONDS:-8}" "${cmd[@]}" || true
  else
    "${cmd[@]}" || true
  fi
}

job_id_in_queue() {
  [[ -n "${job_id}" ]] || return 1
  command -v squeue >/dev/null 2>&1 || return 1
  local -a cmd=(squeue)
  local output
  if [[ "${uses_spur}" == "1" ]]; then
    cmd+=(--controller "${controller}")
  fi
  if ! output="$("${cmd[@]}" --noheader --format="%A" 2>&1)"; then
    echo "WARNING: unable to query Slurm job ${job_id}: ${output}" >&2
    return 0
  fi
  awk -v job_id="${job_id}" '
    {
      current_job_id = $1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", current_job_id)
      if (current_job_id == job_id) {
        found = 1
        exit
      }
    }
    END { exit(found ? 0 : 1) }
  ' <<< "${output}"
}

if [[ -n "${job_id}" ]]; then
  run_scancel "${job_id}"
  deadline=$(( $(date +%s) + ${SLURM_CANCEL_WAIT_SECONDS:-60} ))
  while job_id_in_queue; do
    if [[ "$(date +%s)" -ge "${deadline}" ]]; then
      run_scancel --signal=KILL "${job_id}"
      break
    fi
    sleep 5
  done
elif [[ -n "${job_name}" ]]; then
  run_scancel --user "${current_user}" --name "${job_name}"
  sleep "${SLURM_CANCEL_NAME_KILL_DELAY_SECONDS:-5}"
  run_scancel --signal=KILL --user "${current_user}" --name "${job_name}"
fi
EOF
  } > "${helper}"
  chmod +x "${helper}"
}

stream_file_lines() {
  local file="$1"
  local prefix="$2"
  local current_line="$3"
  local total_lines

  if [[ ! -f "${file}" ]]; then
    printf '%s\n' "${current_line}"
    return 0
  fi

  total_lines="$(wc -l < "${file}" | tr -d ' ')"
  if [[ "${total_lines}" -gt "${current_line}" ]]; then
    awk -v start="${current_line}" -v prefix="${prefix}" 'NR > start { print prefix $0 }' "${file}" >&2
  fi
  printf '%s\n' "${total_lines}"
}

stream_slurm_logs_once() {
  OUT_LINE="$(stream_file_lines "${SLURM_JOB_OUTPUT}" "[slurm.out] " "${OUT_LINE}")"
  ERR_LINE="$(stream_file_lines "${SLURM_JOB_ERROR}" "[slurm.err] " "${ERR_LINE}")"
}

monitor_slurm_job() {
  local job_id="$1"
  local job_line query_rc query_failures=0
  local job_seen=0 initial_empty_queries=0
  local current_job_id state elapsed nodes reason
  OUT_LINE=0
  ERR_LINE=0

  echo "=== monitoring Slurm job ${job_id} ==="
  while true; do
    if job_line="$(query_slurm_job "${job_id}")"; then
      query_failures=0
      if [[ -z "${job_line}" ]]; then
        if [[ "${job_seen}" -eq 1 ]]; then
          break
        fi
        initial_empty_queries=$((initial_empty_queries + 1))
        if [[ "${initial_empty_queries}" -ge "${SLURM_SQUEUE_INITIAL_ATTEMPTS:-6}" ]]; then
          echo "WARNING: Slurm job ${job_id} did not appear in squeue" >&2
          break
        fi
        sleep "${SLURM_SQUEUE_RETRY_INTERVAL:-5}"
        continue
      fi
      job_seen=1
    else
      query_rc=$?
      query_failures=$((query_failures + 1))
      if [[ "${query_failures}" -ge "${SLURM_SQUEUE_MAX_FAILURES:-3}" ]]; then
        echo "ERROR: failed to query Slurm job ${job_id} ${query_failures} consecutive times" >&2
        return "${query_rc}"
      fi
      sleep "${SLURM_SQUEUE_RETRY_INTERVAL:-5}"
      continue
    fi

    IFS='|' read -r current_job_id state elapsed nodes reason <<< "${job_line}"
    echo "[slurm] job=${current_job_id} state=${state} elapsed=${elapsed} nodes=${nodes} reason=${reason}"
    stream_slurm_logs_once
    if [[ -n "${SLURM_EXTRA_LOG_STREAMER:-}" ]]; then
      "${SLURM_EXTRA_LOG_STREAMER}" "${job_id}"
    fi
    sleep "${SLURM_LOG_POLL_INTERVAL}"
  done

  stream_slurm_logs_once
  if [[ -n "${SLURM_EXTRA_LOG_STREAMER:-}" ]]; then
    "${SLURM_EXTRA_LOG_STREAMER}" "${job_id}"
  fi
}

read_slurm_exit_code() {
  local job_id="$1"
  local sacct_line exit_status exit_signal deadline

  SLURM_STATE="unknown"
  SLURM_EXIT_CODE="unknown"
  SLURM_JOB_RC=2

  SLURM_ACCOUNTING_TIMEOUT="${SLURM_ACCOUNTING_TIMEOUT:-30}"
  SLURM_ACCOUNTING_POLL_INTERVAL="${SLURM_ACCOUNTING_POLL_INTERVAL:-2}"

  if ! command -v sacct >/dev/null 2>&1; then
    echo "WARNING: sacct not found; unable to read Slurm job exit code" >&2
    return 0
  fi

  deadline=$(( $(date +%s) + SLURM_ACCOUNTING_TIMEOUT ))
  while true; do
    if [[ "${USES_SPUR_CONTROLLER}" == "1" ]]; then
      if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
        sacct_line="$(sacct --account "${SLURM_ACCOUNT}" --brief --noheader 2>/dev/null | awk -v job_id="${job_id}" '$1 == job_id { print $2 "|" $3; exit }' || true)"
      elif [[ -n "${SPUR_ACCOUNTING_ADDR:-}" ]]; then
        sacct_line="$(sacct --accounting "${SPUR_ACCOUNTING_ADDR}" --brief --noheader 2>/dev/null | awk -v job_id="${job_id}" '$1 == job_id { print $2 "|" $3; exit }' || true)"
      else
        sacct_line="$(sacct -j "${job_id}" -X -n -P -o State,ExitCode 2>/dev/null | awk -F'|' 'NF { print; exit }' || true)"
      fi
    else
      sacct_line="$(sacct -j "${job_id}" -X -n -P -o State,ExitCode 2>/dev/null | awk -F'|' 'NF { print; exit }' || true)"
    fi
    [[ -n "${sacct_line}" ]] && break

    if [[ "$(date +%s)" -ge "${deadline}" ]]; then
      echo "ERROR: unable to read final Slurm state for job ${job_id} after ${SLURM_ACCOUNTING_TIMEOUT}s" >&2
      return 0
    fi
    sleep "${SLURM_ACCOUNTING_POLL_INTERVAL}"
  done

  SLURM_STATE="${sacct_line%%|*}"
  SLURM_STATE="${SLURM_STATE%%+*}"
  SLURM_EXIT_CODE="${sacct_line##*|}"
  exit_status="${SLURM_EXIT_CODE%%:*}"
  exit_signal="${SLURM_EXIT_CODE##*:}"

  if ! [[ "${exit_status}" =~ ^[0-9]+$ ]]; then
    SLURM_JOB_RC=1
  elif [[ "${exit_signal}" =~ ^[0-9]+$ && "${exit_status}" -eq 0 && "${exit_signal}" -ne 0 ]]; then
    SLURM_JOB_RC=$((128 + exit_signal))
  else
    SLURM_JOB_RC="${exit_status}"
  fi

  if [[ "${SLURM_STATE}" != COMPLETE && "${SLURM_STATE}" != COMPLETED && "${SLURM_JOB_RC}" -eq 0 ]]; then
    SLURM_JOB_RC=1
  fi
}
