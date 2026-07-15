#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/slurm-24.05.5.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

usage() {
  cat <<'USAGE'
Usage:
  pd_submit.sh --cell-json <json> [--result-dir <dir>] [--dry-run]

Submits one expanded ATOMesh real P/D benchmark cell to Slurm. The cell JSON is
produced by .github/scripts/atomesh/pd_matrix.py.
USAGE
}

CELL_JSON=""
RESULT_DIR="${RESULT_DIR:-atomesh-results}"
DRY_RUN=0
JOB_ID=""
SLURM_JOB_ACTIVE=0
SCANCEL_SENT=0
declare -A SPUR_SHARED_LOG_LINES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cell-json)
      CELL_JSON="$2"
      shift 2
      ;;
    --result-dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CELL_JSON}" ]]; then
  echo "ERROR: --cell-json is required" >&2
  usage >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
JOB_SCRIPT="${REPO_ROOT}/.github/scripts/atomesh/pd_slurm_job.sh"
mkdir -p "${RESULT_DIR}"

export CELL_JSON
eval "$(
python3 - <<'PY'
import json
import os
import shlex

cell = json.loads(os.environ["CELL_JSON"])
runner = cell.get("runner", {})
service = cell.get("service", {})
prefill = service.get("prefill", {})
decode = service.get("decode", {})
router = service.get("router", {})
server_args = cell.get("server_args", {})
accuracy = cell.get("accuracy", {})

def shell_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value

def csv_value(value):
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)

def q(value):
    return shlex.quote(str(shell_value(value)))

exports = {
    "ATOMESH_CELL_ID": cell["id"],
    "MODEL_NAME": cell["model"],
    "BACKEND": cell["backend"],
    "DOCKER_IMAGE": cell["image"],
    "MODEL_PATH": cell["model_path"],
    "PRECISION": cell.get("precision", ""),
    "TOPOLOGY": cell["topology"],
    "DISPLAY_TOPOLOGY": cell.get("display_topology", cell["topology"]),
    "ATOMESH_PD_WORKER_LAYOUT": cell.get("pd_worker_layout", "multi_node"),
    "NODE_LIST": ",".join(cell["nodes"]),
    "NUM_NODES": cell["num_nodes"],
    "ISL_LIST": ",".join(str(v) for v in cell["isl"]),
    "OSL": cell["osl"],
    "CONC_LIST": ",".join(str(v) for v in cell["concurrency"]),
    "BENCH_MAX_CONCURRENCY": cell["concurrency_x"],
    "RANDOM_RANGE_RATIO": cell["random_range_ratio"],
    "REQUEST_RATE": cell["request_rate"],
    "BENCH_NUM_PROMPTS_MULTIPLIER": cell["num_prompts_multiplier"],
    "WAIT_SERVER_TIMEOUT": cell["wait_server_timeout"],
    "WAIT_ROUTER_TIMEOUT": cell["wait_router_timeout"],
    "PREFILL_WORKERS": prefill.get("workers", 1),
    "DECODE_WORKERS": decode.get("workers", 1),
    "PREFILL_TP": prefill.get("tp", 8),
    "DECODE_TP": decode.get("tp", 8),
    "PREFILL_ENABLE_DP": str(prefill.get("enable_dp_attention", False)).lower(),
    "DECODE_ENABLE_DP": str(decode.get("enable_dp_attention", False)).lower(),
    "PREFILL_CUDAGRAPH": prefill.get("cudagraph", ""),
    "DECODE_CUDAGRAPH": decode.get("cudagraph", ""),
    "PREFILL_PORT": prefill.get("port", 8010),
    "DECODE_PORT": decode.get("port", 8020),
    "ROUTER_PORT": router.get("port", 8000),
    "ROUTER_POLICY": router.get("policy", "random"),
    "PROMETHEUS_PORT": router.get("prometheus_port", 29100),
    "KV_CACHE_DTYPE": server_args.get("kv_cache_dtype", "fp8"),
    "BLOCK_SIZE": server_args.get("block_size", 16),
    "MEM_FRACTION": server_args.get("gpu_memory_utilization", 0.85),
    "MAX_MODEL_LEN": server_args.get("max_model_len", ""),
    "MAX_NUM_SEQS": server_args.get("max_num_seqs", 256),
    "DECODE_MAX_NUM_SEQS": server_args.get("decode_max_num_seqs", ""),
    "MAX_NUM_BATCHED_TOKENS": server_args.get("max_num_batched_tokens", ""),
    "ONLINE_QUANT_CONFIG": server_args.get("online_quant_config", ""),
    "HF_OVERRIDES": server_args.get("hf_overrides", ""),
    "SPEC_METHOD": server_args.get("method", ""),
    "DRAFT_MODEL_PATH": server_args.get("draft_model", ""),
    "NUM_SPEC_TOKENS": server_args.get("num_speculative_tokens", ""),
    "EXTRA_SERVER_ARGS": server_args.get("extra_args", ""),
    "PREFILL_EXTRA_SERVER_ARGS": prefill.get("extra_args", ""),
    "DECODE_EXTRA_SERVER_ARGS": decode.get("extra_args", ""),
    "RUN_EVAL": str(cell.get("run_eval", False)).lower(),
    "EVAL_TASK": accuracy.get("task", "gsm8k"),
    "EVAL_FEWSHOT": accuracy.get("fewshot", 3),
    "EVAL_LIMIT": "" if accuracy.get("limit") is None else accuracy.get("limit"),
    "EVAL_MODEL_TYPE": accuracy.get("model_type", "local-completions"),
    "EVAL_ENDPOINT": accuracy.get("endpoint", "completions"),
    "EVAL_BATCH_SIZE": "" if accuracy.get("batch_size") is None else accuracy.get("batch_size"),
    "EVAL_MAX_GEN_TOKS": "" if accuracy.get("max_gen_toks") is None else accuracy.get("max_gen_toks"),
    "EVAL_APPLY_CHAT_TEMPLATE": str(accuracy.get("apply_chat_template", False)).lower(),
    "EVAL_FEWSHOT_AS_MULTITURN": str(accuracy.get("fewshot_as_multiturn", False)).lower(),
    "EVAL_CONCURRENCY": csv_value(
        accuracy.get("concurrency") or cell.get("concurrency", [])
    ),
    "SLURM_SUBMIT_RUNNER": runner.get("slurm_submit_runner", "atomesh-cicd"),
    "SLURM_ACCOUNT": runner.get("slurm_account", "amd-frameworks"),
    "SLURM_PARTITION": runner.get("slurm_partition", "amd-frameworks"),
    "SLURM_CPUS_PER_TASK": runner.get("cpus_per_task", 114),
    "SLURM_GPUS_PER_NODE": runner.get("gpus_per_node", 8),
    "SLURM_TIME_LIMIT": runner.get("time_limit", "06:00:00"),
    "SLURM_LOG_ROOT": runner.get("log_root", "/it-share/ATOMESH_LOG/"),
    "SPUR_CONTROLLER_ADDR": runner.get(
        "spur_controller_addr",
        os.environ.get("SPUR_CONTROLLER_ADDR", "http://134.199.196.72:6817"),
    ),
    "SPUR_ACCOUNTING_ADDR": runner.get(
        "spur_accounting_addr",
        os.environ.get("SPUR_ACCOUNTING_ADDR", "http://134.199.196.72:6819"),
    ),
}

for key, value in exports.items():
    print(f"export {key}={q(value)}")

for key, value in cell.get("env", {}).get("common", {}).items():
    print(f"export ATOMESH_ENV_{key}={q(value)}")
for key, value in cell.get("env", {}).get("prefill", {}).items():
    print(f"export ATOMESH_PREFILL_ENV_{key}={q(value)}")
for key, value in cell.get("env", {}).get("decode", {}).items():
    print(f"export ATOMESH_DECODE_ENV_{key}={q(value)}")
PY
)"

export RESULT_DIR
CURRENT_USER="$(id -un)"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT//\$\{USER\}/${CURRENT_USER}}"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT//\$USER/${CURRENT_USER}}"
export LOG_ROOT="${SLURM_LOG_ROOT%/}/${ATOMESH_CELL_ID}-${GITHUB_RUN_ID:-local}-$(date +%Y%m%d%H%M%S)"
if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
  export SLURM_OUTPUT="/tmp/atomesh-%j.out"
  export SLURM_ERROR="/tmp/atomesh-%j.err"
else
  export SLURM_OUTPUT="${LOG_ROOT}/slurm-%j.out"
  export SLURM_ERROR="${LOG_ROOT}/slurm-%j.err"
fi
SLURM_LOG_POLL_INTERVAL="${SLURM_LOG_POLL_INTERVAL:-30}"

echo "=== ATOMesh benchmark cell ==="
echo "cell=${ATOMESH_CELL_ID}"
echo "model=${MODEL_NAME}"
echo "topology=${DISPLAY_TOPOLOGY}"
echo "nodes=${NODE_LIST}"
echo "isl=${ISL_LIST} osl=${OSL} concurrency=${CONC_LIST}"
echo "log_root=${LOG_ROOT}"
if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
  echo "spur_controller=${SPUR_CONTROLLER_ADDR}"
  echo "spur_accounting=${SPUR_ACCOUNTING_ADDR}"
fi

mkdir -p "${RESULT_DIR}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "=== dry-run only; sbatch is not invoked ==="
  python3 - <<'PY'
import json
import os
from pathlib import Path
cell = json.loads(os.environ["CELL_JSON"])
Path(os.environ["RESULT_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["RESULT_DIR"], f"{cell['id']}-dry-run.json").write_text(
    json.dumps({"cell": cell, "log_root": os.environ["LOG_ROOT"]}, indent=2),
    encoding="utf-8",
)
PY
  exit 0
fi

mkdir -p "${LOG_ROOT}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found; use --dry-run on non-Slurm runners" >&2
  echo "PATH=${PATH}" >&2
  echo "host=$(hostname) user=$(id -un)" >&2
  for candidate in /usr/local/slurm-24.05.5.1/bin/sbatch /usr/bin/sbatch /etc/alternatives/sbatch; do
    if [[ -e "${candidate}" || -L "${candidate}" ]]; then
      printf '%s -> %s\n' "${candidate}" "$(readlink -f "${candidate}" 2>/dev/null || true)" >&2
      ls -l "${candidate}" >&2 2>/dev/null || true
    fi
  done
  exit 127
fi

scancel_slurm_job() {
  local reason="$1"
  if [[ "${SLURM_JOB_ACTIVE}" != "1" || -z "${JOB_ID}" || "${SCANCEL_SENT}" == "1" ]]; then
    return 0
  fi

  SCANCEL_SENT=1
  echo "=== cancelling Slurm job ${JOB_ID}: ${reason} ===" >&2
  if command -v scancel >/dev/null 2>&1; then
    if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
      scancel --controller "${SPUR_CONTROLLER_ADDR}" "${JOB_ID}" || true
    else
      scancel "${JOB_ID}" || true
    fi
    wait_for_slurm_cancel "${JOB_ID}" "TERM" || true
  else
    echo "WARNING: scancel not found; unable to cancel Slurm job ${JOB_ID}" >&2
  fi
}

slurm_job_in_queue() {
  local job_id="$1"
  local squeue_cmd=(squeue)

  if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
    squeue_cmd+=(--controller "${SPUR_CONTROLLER_ADDR}")
  fi

  [[ -n "$("${squeue_cmd[@]}" -h -j "${job_id}" 2>/dev/null)" ]]
}

wait_for_slurm_cancel() {
  local job_id="$1"
  local initial_signal="$2"
  local deadline=$(( $(date +%s) + ${SLURM_CANCEL_WAIT_SECONDS:-60} ))
  local kill_deadline

  while slurm_job_in_queue "${job_id}"; do
    if [[ "$(date +%s)" -ge "${deadline}" ]]; then
      echo "=== Slurm job ${job_id} still queued after ${initial_signal}; sending KILL ===" >&2
      if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
        scancel --controller "${SPUR_CONTROLLER_ADDR}" --signal=KILL "${job_id}" || true
      else
        scancel --signal=KILL "${job_id}" || true
      fi
      kill_deadline=$(( $(date +%s) + ${SLURM_CANCEL_KILL_WAIT_SECONDS:-30} ))
      while slurm_job_in_queue "${job_id}" && [[ "$(date +%s)" -lt "${kill_deadline}" ]]; do
        sleep 5
      done
      break
    fi
    sleep 5
  done
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

on_cancel() {
  local signal="$1"
  local rc="$2"
  scancel_slurm_job "received ${signal}"
  exit "${rc}"
}

on_exit() {
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    scancel_slurm_job "exiting rc=${rc}"
  fi
}

trap on_exit EXIT
trap 'on_cancel HUP 129' HUP
trap 'on_cancel INT 130' INT
trap 'on_cancel TERM 143' TERM

set_slurm_job_log_paths() {
  local job_id="$1"
  SLURM_JOB_OUTPUT="${SLURM_OUTPUT//%j/${job_id}}"
  SLURM_JOB_ERROR="${SLURM_ERROR//%j/${job_id}}"
  echo "slurm_job_id=${job_id}"
  echo "slurm_output=${SLURM_JOB_OUTPUT}"
  echo "slurm_error=${SLURM_JOB_ERROR}"
}

write_slurm_cancel_helper() {
  local job_id="$1"
  local helper="${RESULT_DIR}/${ATOMESH_CELL_ID}.slurm-cancel.sh"

  if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
    cat > "${helper}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
job_id="${job_id}"
job_in_queue() {
  command -v squeue >/dev/null 2>&1 || return 1
  [[ -n "\$(squeue --controller "${SPUR_CONTROLLER_ADDR}" -h -j "\${job_id}" 2>/dev/null)" ]]
}
if command -v scancel >/dev/null 2>&1; then
  scancel --controller "${SPUR_CONTROLLER_ADDR}" "\${job_id}" || true
  deadline=\$(( \$(date +%s) + \${SLURM_CANCEL_WAIT_SECONDS:-60} ))
  while job_in_queue; do
    if [[ "\$(date +%s)" -ge "\${deadline}" ]]; then
      scancel --controller "${SPUR_CONTROLLER_ADDR}" --signal=KILL "\${job_id}" || true
      break
    fi
    sleep 5
  done
fi
EOF
  else
    cat > "${helper}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
job_id="${job_id}"
job_in_queue() {
  command -v squeue >/dev/null 2>&1 || return 1
  [[ -n "\$(squeue -h -j "\${job_id}" 2>/dev/null)" ]]
}
if command -v scancel >/dev/null 2>&1; then
  scancel "\${job_id}" || true
  deadline=\$(( \$(date +%s) + \${SLURM_CANCEL_WAIT_SECONDS:-60} ))
  while job_in_queue; do
    if [[ "\$(date +%s)" -ge "\${deadline}" ]]; then
      scancel --signal=KILL "\${job_id}" || true
      break
    fi
    sleep 5
  done
fi
EOF
  fi
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

stream_spur_shared_logs_once() {
  local job_id="$1"
  local run_dir="${LOG_ROOT}/slurm_job-${job_id}"
  local log_file rel_path current_line

  [[ -d "${run_dir}" ]] || return 0

  shopt -s nullglob
  for log_file in "${run_dir}"/rank-*/container.log "${run_dir}"/logs/*.log; do
    rel_path="${log_file#"${run_dir}/"}"
    current_line="${SPUR_SHARED_LOG_LINES[${log_file}]:-0}"
    SPUR_SHARED_LOG_LINES["${log_file}"]="$(stream_file_lines "${log_file}" "[spur:${rel_path}] " "${current_line}")"
  done
  shopt -u nullglob
}

monitor_slurm_job() {
  local job_id="$1"
  local squeue_cmd=(squeue)
  OUT_LINE=0
  ERR_LINE=0
  SPUR_SHARED_LOG_LINES=()

  if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
    squeue_cmd+=(--controller "${SPUR_CONTROLLER_ADDR}")
  fi

  echo "=== monitoring Slurm job ${job_id} ==="
  while "${squeue_cmd[@]}" -h -j "${job_id}" >/dev/null 2>&1 && [[ -n "$("${squeue_cmd[@]}" -h -j "${job_id}" 2>/dev/null)" ]]; do
    "${squeue_cmd[@]}" -h -j "${job_id}" -o "[slurm] job=%i state=%T elapsed=%M nodes=%D reason=%R" || true
    stream_slurm_logs_once
    if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
      stream_spur_shared_logs_once "${job_id}"
    fi
    sleep "${SLURM_LOG_POLL_INTERVAL}"
  done

  stream_slurm_logs_once
  if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
    stream_spur_shared_logs_once "${job_id}"
  fi
}

read_slurm_exit_code() {
  local job_id="$1"
  local sacct_line exit_status exit_signal

  SLURM_STATE="unknown"
  SLURM_EXIT_CODE="unknown"
  SLURM_JOB_RC=1

  if ! command -v sacct >/dev/null 2>&1; then
    echo "WARNING: sacct not found; unable to read Slurm job exit code" >&2
    return 0
  fi

  if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
    sacct_line="$(sacct --accounting "${SPUR_ACCOUNTING_ADDR}" --brief --noheader 2>/dev/null | awk -v job_id="${job_id}" '$1 == job_id { print $2 "|" $3; exit }' || true)"
  else
    sacct_line="$(sacct -j "${job_id}" -X -n -P -o State,ExitCode 2>/dev/null | awk -F'|' 'NF { print; exit }' || true)"
  fi
  if [[ -z "${sacct_line}" ]]; then
    return 0
  fi

  SLURM_STATE="${sacct_line%%|*}"
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

IFS=',' read -r -a NODE_ARRAY <<< "${NODE_LIST}"
if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
  SUBMIT_SCRIPT="${LOG_ROOT}/submit-${ATOMESH_CELL_ID}.sbatch.sh"
  cat > "${SUBMIT_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${ATOMESH_CELL_ID}
#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=${SLURM_TIME_LIMIT}
#SBATCH --chdir=/tmp
EOF
  if [[ -n "${NODE_LIST}" ]]; then
    printf '#SBATCH --nodelist=%s\n' "${NODE_LIST}" >> "${SUBMIT_SCRIPT}"
  fi
  cat >> "${SUBMIT_SCRIPT}" <<EOF
#SBATCH --output=${SLURM_OUTPUT}
#SBATCH --error=${SLURM_ERROR}
EOF
  {
    printf 'export %q=%q\n' GITHUB_WORKSPACE "${REPO_ROOT}"
    printf 'exec %q\n' "${JOB_SCRIPT}"
  } >> "${SUBMIT_SCRIPT}"
  chmod +x "${SUBMIT_SCRIPT}"
  SBATCH_CMD=(sbatch --controller "${SPUR_CONTROLLER_ADDR}" "${SUBMIT_SCRIPT}")
else
  SBATCH_CMD=(
    sbatch
    --parsable
    --exclusive
    --account "${SLURM_ACCOUNT}"
    --partition "${SLURM_PARTITION}"
    --nodes "${NUM_NODES}"
    --ntasks "${NUM_NODES}"
    --ntasks-per-node 1
    --cpus-per-task "${SLURM_CPUS_PER_TASK}"
    --gres "gpu:${SLURM_GPUS_PER_NODE}"
    --time "${SLURM_TIME_LIMIT}"
    --nodelist "${NODE_LIST}"
    --output "${SLURM_OUTPUT}"
    --error "${SLURM_ERROR}"
    "${JOB_SCRIPT}"
  )
fi

echo "=== submitting Slurm job ==="
printf ' %q' "${SBATCH_CMD[@]}"
echo

set +e
SBATCH_OUTPUT="$("${SBATCH_CMD[@]}")"
SBATCH_RC=$?
set -e
echo "${SBATCH_OUTPUT}"

if [[ "${SBATCH_RC}" -ne 0 ]]; then
  echo "sbatch submit exit code: ${SBATCH_RC}"
  exit "${SBATCH_RC}"
fi

JOB_ID="$(parse_sbatch_job_id "${SBATCH_OUTPUT}")"
echo "${JOB_ID}" | tee "${RESULT_DIR}/${ATOMESH_CELL_ID}.slurm-job-id"
write_slurm_cancel_helper "${JOB_ID}"

SLURM_JOB_ACTIVE=1
set_slurm_job_log_paths "${JOB_ID}"
monitor_slurm_job "${JOB_ID}"
read_slurm_exit_code "${JOB_ID}"
SLURM_JOB_ACTIVE=0
SBATCH_RC="${SLURM_JOB_RC}"
echo "slurm_state=${SLURM_STATE}"
echo "slurm_exit_code=${SLURM_EXIT_CODE}"
echo "slurm job exit code: ${SBATCH_RC}"

if [[ -d "${LOG_ROOT}" ]]; then
  mkdir -p "${RESULT_DIR}/${ATOMESH_CELL_ID}"
  if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
    tar \
      --exclude='.cache' \
      --exclude='./.cache' \
      --exclude='.aiter' \
      --exclude='./.aiter' \
      -C "${LOG_ROOT}" \
      -cf - . | tar \
      --no-same-owner \
      --no-same-permissions \
      -C "${RESULT_DIR}/${ATOMESH_CELL_ID}" \
      -xf - || true
  else
    cp -a "${LOG_ROOT}/." "${RESULT_DIR}/${ATOMESH_CELL_ID}/" || true
  fi
fi

exit "${SBATCH_RC}"
