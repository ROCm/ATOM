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
# EX_TEMPFAIL describes an unknown scheduler outcome, not a failed batch job.
SLURM_STATUS_UNAVAILABLE_RC=75

detect_slurm_backend() {
  local help
  if [[ "${USES_SPUR_CONTROLLER}" == "1" ]]; then
    return 0
  fi
  # Runner labels and inherited Spur addresses also exist on native Slurm
  # runners. Inspect the installed client before adding Spur-only arguments.
  help="$(run_slurm_query scontrol --help 2>&1 || true)"
  if [[ "${help}" == *Spur* ]]; then
    USES_SPUR_CONTROLLER=1
  fi
}

run_slurm_query() {
  # Leave stderr visible, including unsupported arguments and transport errors.
  if command -v timeout >/dev/null 2>&1; then
    timeout "${SLURM_QUERY_TIMEOUT_SECONDS:-20}" "$@"
  else
    "$@"
  fi
}

slurm_node_selection_args() {
  local candidates="$1" count="$2" all_nodes excluded
  local -a candidate_nodes
  SLURM_NODE_SELECTION_ARGS=()
  [[ -n "${candidates}" ]] || return 0
  IFS=',' read -r -a candidate_nodes <<< "${candidates}"
  if [[ "${USES_SPUR_CONTROLLER}" == "1" || "${#candidate_nodes[@]}" -le "${count}" ]]; then
    SLURM_NODE_SELECTION_ARGS=(-w "${candidates}")
    return 0
  fi
  # Native Slurm requires every host in -w. Exclude the complement
  # instead, so the scheduler can choose only the requested number of nodes.
  if ! all_nodes="$(run_slurm_query sinfo -N -h -o '%N')" || [[ -z "${all_nodes}" ]]; then
    echo "ERROR: Cannot query cluster nodes to restrict the candidate pool." >&2
    return 1
  fi
  excluded="$(python3 - "${candidates}" "${count}" "${all_nodes}" <<'PY'
import sys

candidates = set(sys.argv[1].split(","))
cluster = set(sys.argv[3].split())
missing = candidates - cluster
if missing:
    raise SystemExit("Unknown candidate nodes: " + ",".join(sorted(missing)))
if len(candidates) < int(sys.argv[2]):
    raise SystemExit("Not enough distinct candidate nodes")
print(",".join(sorted(cluster - candidates)))
PY
  )" || return 1
  if [[ -n "${excluded}" ]]; then
    SLURM_NODE_SELECTION_ARGS=(--exclude "${excluded}")
  fi
}

slurm_state_is_terminal() {
  case "${1%%+*}" in
    COMPLETE|COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
      return 0 ;;
    *) return 1 ;;
  esac
}

slurm_state_is_active() {
  case "${1%%+*}" in
    PENDING|RUNNING|COMPLETING|SUSPENDED|CONFIGURING|RESIZING|REQUEUED|REQUEUE_FED|REQUEUE_HOLD|RESV_DEL_HOLD|SIGNALING|STAGE_OUT|STOPPED)
      return 0 ;;
    *) return 1 ;;
  esac
}

query_slurm_controller_job() {
  local job_id="$1" output
  local -a cmd=(scontrol)
  if [[ "${USES_SPUR_CONTROLLER}" == "1" && -n "${SPUR_CONTROLLER_ADDR}" ]]; then
    cmd+=(--controller "${SPUR_CONTROLLER_ADDR}")
  fi
  if ! output="$(run_slurm_query "${cmd[@]}" show job "${job_id}")"; then
    echo "ERROR: scontrol query failed for job ${job_id}" >&2
    return 2
  fi
  awk -v id="${job_id}" '
    /JobId=/ { selected = 0 }
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "JobId=" id) selected = 1
        if (selected && $i ~ /^JobState=/) { split($i, a, "="); state = a[2] }
        if (selected && $i ~ /^ExitCode=/) { split($i, a, "="); code = a[2] }
      }
    }
    END { if (state != "") print state "|" code }
  ' <<< "${output}"
}

query_slurm_accounting_job() {
  local job_id="$1" output
  local -a cmd=(sacct)
  if [[ "${USES_SPUR_CONTROLLER}" == "1" ]]; then
    if [[ -n "${SPUR_CONTROLLER_ADDR}" ]]; then
      cmd+=(--controller "${SPUR_CONTROLLER_ADDR}")
    fi
    # Spur serves accounting on the controller port. Avoid native-only -X/-P
    # and fixed-width --brief output, which truncates CANCELLED to CANCELLE.
    cmd+=(-j "${job_id}" --noheader --format 'JobID%30,State%30,ExitCode%20')
    if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
      cmd+=(--account "${SLURM_ACCOUNT}")
    fi
  else
    cmd+=(-j "${job_id}" -X -n -P -o JobIDRaw,State,ExitCode)
  fi
  if ! output="$(run_slurm_query "${cmd[@]}")"; then
    echo "ERROR: sacct query failed for job ${job_id}" >&2
    return 2
  fi
  if [[ "${USES_SPUR_CONTROLLER}" == "1" ]]; then
    output="$(awk -v id="${job_id}" '$1 == id { print $1 "|" $2 "|" $3; exit }' <<< "${output}")"
  fi
  # Some Spur versions ignore -j. Always select the exact job ourselves.
  awk -F'|' -v id="${job_id}" '
    { for (i = 1; i <= 3; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i) }
    $1 == id { split($2, state, " "); print state[1] "|" $3; exit }
  ' <<< "${output}"
}

query_slurm_result() {
  local job_id="$1" result
  if result="$(query_slurm_controller_job "${job_id}")" && [[ -n "${result}" ]]; then
    # Live PENDING/RUNNING/COMPLETING takes precedence over accounting.
    printf '%s\n' "${result}"
    return 0
  fi
  query_slurm_accounting_job "${job_id}"
}

run_scancel() {
  local -a scancel_cmd=(scancel)
  if [[ "${USES_SPUR_CONTROLLER}" == "1" && -n "${SPUR_CONTROLLER_ADDR}" ]]; then
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

  if [[ "${USES_SPUR_CONTROLLER}" == "1" && -n "${SPUR_CONTROLLER_ADDR}" ]]; then
    squeue_cmd+=(--controller "${SPUR_CONTROLLER_ADDR}")
  fi

  if ! output="$(run_slurm_query "${squeue_cmd[@]}" --noheader --format="%A|%T|%M|%D|%R")"; then
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
  if [[ "${rc}" -eq "${SLURM_STATUS_UNAVAILABLE_RC}" && "${SLURM_STATE:-unknown}" == "unknown" ]]; then
    echo "WARNING: scheduler status is unavailable; preserving job ${JOB_ID:-unknown} for later inspection" >&2
    if [[ -n "${SLURM_CANCEL_HELPER:-}" ]]; then
      printf '%s\n' "${JOB_ID:-unknown}" > "${SLURM_CANCEL_HELPER}.status-unknown"
    fi
  elif [[ "${rc}" -ne 0 && "${SLURM_JOB_ACTIVE}" == "1" ]]; then
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
  # The wrapper uses this marker to distinguish monitoring failure from a real
  # batch exit code of 75. Reset it when submitting a new job with this helper.
  : > "${helper}.status-unknown"
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
  if [[ "${uses_spur}" == "1" && -n "${controller}" ]]; then
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
  if [[ "${uses_spur}" == "1" && -n "${controller}" ]]; then
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
  local job_line result state current_job_id elapsed nodes reason
  local missing_queries=0 unknown_since=0 now
  SLURM_CONFIRMED_RESULT=""
  OUT_LINE=0
  ERR_LINE=0

  echo "=== monitoring Slurm job ${job_id} ==="
  while true; do
    state=""
    result=""
    missing_queries=$((missing_queries + 1))
    if job_line="$(query_slurm_job "${job_id}")" && [[ -n "${job_line}" ]]; then
      IFS='|' read -r current_job_id state elapsed nodes reason <<< "${job_line}"
      echo "[slurm] job=${current_job_id} state=${state} elapsed=${elapsed} nodes=${nodes} reason=${reason}"
    else
      echo "WARNING: job ${job_id} absent from queue or queue query failed (${missing_queries}); verifying controller/accounting" >&2
      if result="$(query_slurm_result "${job_id}")" && [[ -n "${result}" ]]; then
        state="${result%%|*}"
        echo "[slurm] job=${job_id} verified_state=${state}"
      fi
    fi

    stream_slurm_logs_once
    if [[ -n "${SLURM_EXTRA_LOG_STREAMER:-}" ]]; then
      "${SLURM_EXTRA_LOG_STREAMER}" "${job_id}"
    fi

    if slurm_state_is_terminal "${state}"; then
      # A missing queue entry alone is never evidence of completion.
      SLURM_CONFIRMED_RESULT="${result}"
      return 0
    elif slurm_state_is_active "${state}"; then
      # Includes PENDING, RUNNING, COMPLETING, SUSPENDED and requeue states.
      unknown_since=0
      missing_queries=0
    else
      now="$(date +%s)"
      if [[ "${unknown_since}" -eq 0 ]]; then
        unknown_since="${now}"
      fi
      if [[ "${missing_queries}" -ge "${SLURM_SQUEUE_INITIAL_ATTEMPTS:-6}" &&
            $((now - unknown_since)) -ge "${SLURM_STATUS_UNKNOWN_TIMEOUT:-300}" ]]; then
        # Published batch/rank results can outlive the controller's job record.
        SLURM_STATE=unknown
        SLURM_EXIT_CODE=unknown
        SLURM_JOB_RC="${SLURM_STATUS_UNAVAILABLE_RC}"
        if [[ -n "${SLURM_STATUS_DIR:-}" ]]; then
          read_slurm_status_files "${SLURM_STATUS_DIR}" "${SLURM_STATUS_RANKS:-1}"
          if slurm_state_is_terminal "${SLURM_STATE}"; then
            SLURM_CONFIRMED_RESULT="${SLURM_STATE}|${SLURM_EXIT_CODE}"
            return 0
          fi
        fi
        echo "ERROR: unable to determine state of Slurm job ${job_id}; monitoring failed, job execution outcome is unknown. Job has NOT been cancelled. Logs: ${SLURM_JOB_OUTPUT}" >&2
        return "${SLURM_STATUS_UNAVAILABLE_RC}"
      fi
    fi

    if [[ "${missing_queries}" -gt 0 ]]; then
      sleep "${SLURM_SQUEUE_RETRY_INTERVAL:-5}"
    else
      sleep "${SLURM_LOG_POLL_INTERVAL}"
    fi
  done
}

read_slurm_exit_code() {
  local job_id="$1"
  local result exit_status exit_signal deadline state=""
  SLURM_STATE=unknown
  SLURM_EXIT_CODE=unknown
  SLURM_JOB_RC="${SLURM_STATUS_UNAVAILABLE_RC}"
  deadline=$(( $(date +%s) + ${SLURM_ACCOUNTING_TIMEOUT:-30} ))

  while true; do
    result="${SLURM_CONFIRMED_RESULT:-}"
    if [[ -z "${result}" ]]; then
      result="$(query_slurm_result "${job_id}")" || result=""
    fi
    state="${result%%|*}"
    state="${state%%+*}"
    if slurm_state_is_terminal "${state}" && [[ "${result##*|}" =~ ^-?[0-9]+:[0-9]+$ ]]; then
      break
    fi
    if [[ "$(date +%s)" -ge "${deadline}" ]]; then
      echo "ERROR: unable to read final Slurm state/exit code for job ${job_id}; last state=${state:-unknown}. This is a status query failure, not a batch failure." >&2
      return 0
    fi
    # A terminal state without an exit code can be completed by accounting.
    SLURM_CONFIRMED_RESULT=""
    if slurm_state_is_terminal "${state}"; then
      SLURM_CONFIRMED_RESULT="$(query_slurm_accounting_job "${job_id}")" || SLURM_CONFIRMED_RESULT=""
    fi
    sleep "${SLURM_ACCOUNTING_POLL_INTERVAL:-2}"
  done

  SLURM_STATE="${state}"
  SLURM_EXIT_CODE="${result##*|}"
  exit_status="${SLURM_EXIT_CODE%%:*}"
  exit_signal="${SLURM_EXIT_CODE##*:}"
  if ! [[ "${exit_status}" =~ ^[0-9]+$ ]]; then
    SLURM_JOB_RC=1
  elif [[ "${exit_status}" -eq 0 && "${exit_signal}" -ne 0 ]]; then
    SLURM_JOB_RC=$((128 + exit_signal))
  else
    SLURM_JOB_RC="${exit_status}"
  fi
  if [[ "${SLURM_STATE}" != COMPLETE && "${SLURM_STATE}" != COMPLETED && "${SLURM_JOB_RC}" -eq 0 ]]; then
    SLURM_JOB_RC=1
  fi
}

# Spur runs its batch script once per node. When accounting is unavailable,
# accept a batch result or a complete set of atomically published rank results.
read_slurm_status_files() {
  local status_dir="$1"
  local num_ranks="$2"
  local rc_file rc rank
  local ranks_reported=0 worst_rank_rc=0

  [[ "${SLURM_STATE}" == "unknown" ]] || return 0

  rc_file="${status_dir}/slurm-job.rc"
  if [[ -s "${rc_file}" ]]; then
    rc="$(tr -d '[:space:]' < "${rc_file}")"
    if [[ "${rc}" =~ ^(0|[1-9][0-9]{0,2})$ && "${rc}" -le 255 ]]; then
      worst_rank_rc="${rc}"
      echo "Using batch script exit status because Slurm accounting is unavailable."
    else
      echo "WARNING: invalid batch script exit status: ${rc}" >&2
      return 0
    fi
  else
    # Read each expected rank exactly once; ignore .tmp and unrelated files.
    for ((rank = 0; rank < num_ranks; rank++)); do
      rc_file="${status_dir}/rank-rc-${rank}"
      [[ -s "${rc_file}" ]] || continue
      rc="$(tr -d '[:space:]' < "${rc_file}")"
      [[ "${rc}" =~ ^(0|[1-9][0-9]{0,2})$ && "${rc}" -le 255 ]] || continue
      ranks_reported=$((ranks_reported + 1))
      echo "Slurm rank status: rank-${rank}=${rc}"
      if [[ "${rc}" -gt "${worst_rank_rc}" ]]; then
        worst_rank_rc="${rc}"
      fi
    done
    if [[ "${ranks_reported}" -ne "${num_ranks}" || "${num_ranks}" -le 0 ]]; then
      echo "WARNING: only ${ranks_reported}/${num_ranks} ranks reported an exit status" >&2
      return 0
    fi
    echo "Using per-rank exit status because Slurm accounting is unavailable."
  fi

  SLURM_JOB_RC="${worst_rank_rc}"
  SLURM_EXIT_CODE="${worst_rank_rc}:0"
  if [[ "${worst_rank_rc}" -eq 0 ]]; then
    SLURM_STATE="COMPLETED"
  else
    SLURM_STATE="FAILED"
  fi
}
