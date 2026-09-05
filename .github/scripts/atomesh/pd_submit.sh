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
benchmark = cell.get("benchmark", {})
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

slurm_submit_runner = runner.get("slurm_submit_runner", "atomesh-cicd")
spur_controller_addr = runner.get("spur_controller_addr")
crusoe_runner_labels = {
    "atomesh-cicd-crusoe-mi355",
    "atomesh-cicd-mi355-crusoe",
}
if not spur_controller_addr:
    if slurm_submit_runner in crusoe_runner_labels:
        spur_controller_addr = "http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817"
    else:
        spur_controller_addr = os.environ.get(
            "SPUR_CONTROLLER_ADDR", "http://134.199.196.72:6817"
        )

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
    "BENCHMARK_KIND": benchmark.get("kind", "random"),
    "AIPERF_DIR": benchmark.get("aiperf_dir", ""),
    "AIPERF_VENV": benchmark.get("aiperf_venv", ""),
    "AIPERF_COMMIT": benchmark.get("aiperf_commit", ""),
    "AIPERF_SCENARIO": benchmark.get("scenario", ""),
    "AIPERF_PUBLIC_DATASET": benchmark.get("public_dataset", ""),
    "AIPERF_MAX_CONTEXT_LENGTH": benchmark.get("max_context_length", ""),
    "AIPERF_NUM_DATASET_ENTRIES": benchmark.get("num_dataset_entries", ""),
    "AIPERF_BENCHMARK_DURATION": benchmark.get("benchmark_duration", ""),
    "AIPERF_WARMUP_REQUESTS_PER_LANE": benchmark.get(
        "warmup_requests_per_lane", ""
    ),
    "AIPERF_TRACE_IDLE_GAP_CAP_SECONDS": benchmark.get(
        "trace_idle_gap_cap_seconds", ""
    ),
    "AIPERF_WARMUP_GRACE_PERIOD": benchmark.get("warmup_grace_period", ""),
    "AIPERF_TRAJECTORY_START_MIN_RATIO": benchmark.get(
        "trajectory_start_min_ratio", ""
    ),
    "AIPERF_TRAJECTORY_START_MAX_RATIO": benchmark.get(
        "trajectory_start_max_ratio", ""
    ),
    "AIPERF_FAILED_REQUEST_THRESHOLD": benchmark.get("failed_request_threshold", ""),
    "AIPERF_SLICE_DURATION": benchmark.get("slice_duration", ""),
    "AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT": benchmark.get(
        "cancel_drain_timeout", ""
    ),
    "AIPERF_HTTP_TCP_USER_TIMEOUT": benchmark.get("http_tcp_user_timeout", ""),
    "AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES": benchmark.get(
        "dataset_weka_live_assistant_responses", ""
    ),
    "AIPERF_DATASET_CONFIGURATION_TIMEOUT": benchmark.get(
        "dataset_configuration_timeout", ""
    ),
    "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": benchmark.get(
        "service_profile_configure_timeout", ""
    ),
    "AIPERF_UNSAFE_OVERRIDE": benchmark.get("unsafe_override", ""),
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
    "ENABLE_PREFIX_CACHING": str(
        server_args.get("enable_prefix_caching", False)
    ).lower(),
    "MAX_MODEL_LEN": server_args.get("max_model_len", ""),
    "MAX_NUM_SEQS": server_args.get("max_num_seqs", 256),
    "DECODE_MAX_NUM_SEQS": server_args.get("decode_max_num_seqs", ""),
    "MAX_NUM_BATCHED_TOKENS": server_args.get("max_num_batched_tokens", ""),
    "DECODE_MAX_NUM_BATCHED_TOKENS": server_args.get(
        "decode_max_num_batched_tokens", ""
    ),
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
    "EVAL_THRESHOLD": "" if accuracy.get("threshold") is None else accuracy.get("threshold"),
    "SWEBENCH_AGENT_WORKERS": "" if accuracy.get("agent_workers") is None else accuracy.get("agent_workers"),
    "SWEBENCH_AGENT_STEP_LIMIT": "" if accuracy.get("agent_step_limit") is None else accuracy.get("agent_step_limit"),
    "SWEBENCH_CASE_TIMEOUT": "" if accuracy.get("case_timeout") is None else accuracy.get("case_timeout"),
    "SWEBENCH_AGENT_TIMEOUT": "" if accuracy.get("agent_timeout") is None else accuracy.get("agent_timeout"),
    "SWEBENCH_SCORE_TIMEOUT": "" if accuracy.get("score_timeout") is None else accuracy.get("score_timeout"),
    "SWEBENCH_MAX_WORKERS": "" if accuracy.get("max_workers") is None else accuracy.get("max_workers"),
    "SWEBENCH_EVAL_TIMEOUT": "" if accuracy.get("instance_timeout") is None else accuracy.get("instance_timeout"),
    "SLURM_SUBMIT_RUNNER": slurm_submit_runner,
    "SLURM_ACCOUNT": runner.get("slurm_account", "amd-frameworks"),
    "SLURM_PARTITION": runner.get("slurm_partition", "amd-frameworks"),
    "SLURM_CPUS_PER_TASK": runner.get("cpus_per_task", 114),
    "SLURM_GPUS_PER_NODE": runner.get("gpus_per_node", 8),
    "SLURM_TIME_LIMIT": runner.get("time_limit", "06:00:00"),
    "SLURM_LOG_ROOT": runner.get("log_root", "/it-share/ATOMESH_LOG/"),
    "SPUR_CONTROLLER_ADDR": spur_controller_addr,
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
CURRENT_USER="$(id -un 2>/dev/null || id -u)"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT//\$\{USER\}/${CURRENT_USER}}"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT//\$USER/${CURRENT_USER}}"
export LOG_ROOT="${SLURM_LOG_ROOT%/}/${ATOMESH_CELL_ID}-${GITHUB_RUN_ID:-local}-$(date +%Y%m%d%H%M%S)"
export SLURM_JOB_NAME="${ATOMESH_CELL_ID}-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
export SLURM_CANCEL_HELPER="${RESULT_DIR}/${ATOMESH_CELL_ID}.slurm-cancel.sh"
if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
  export SLURM_OUTPUT="/tmp/atomesh-%j.out"
  export SLURM_ERROR="/tmp/atomesh-%j.err"
else
  export SLURM_OUTPUT="${LOG_ROOT}/slurm-%j.out"
  export SLURM_ERROR="${LOG_ROOT}/slurm-%j.err"
fi
SLURM_LOG_POLL_INTERVAL="${SLURM_LOG_POLL_INTERVAL:-30}"
SLURM_ACCOUNTING_TIMEOUT="${SLURM_ACCOUNTING_TIMEOUT:-30}"
SLURM_ACCOUNTING_POLL_INTERVAL="${SLURM_ACCOUNTING_POLL_INTERVAL:-2}"
USES_SPUR_CONTROLLER=0
if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" || "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-crusoe-mi355" || "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi355-crusoe" ]]; then
  USES_SPUR_CONTROLLER=1
fi

echo "=== ATOMesh benchmark cell ==="
echo "cell=${ATOMESH_CELL_ID}"
echo "model=${MODEL_NAME}"
echo "topology=${DISPLAY_TOPOLOGY}"
echo "nodes=${NODE_LIST}"
echo "isl=${ISL_LIST} osl=${OSL} concurrency=${CONC_LIST}"
echo "slurm_job_name=${SLURM_JOB_NAME}"
echo "log_root=${LOG_ROOT}"
if [[ "${USES_SPUR_CONTROLLER}" == "1" ]]; then
  echo "spur_controller=${SPUR_CONTROLLER_ADDR}"
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
  echo "host=$(hostname) user=${CURRENT_USER}" >&2
  for candidate in /usr/local/slurm-24.05.5.1/bin/sbatch /usr/bin/sbatch /etc/alternatives/sbatch; do
    if [[ -e "${candidate}" || -L "${candidate}" ]]; then
      printf '%s -> %s\n' "${candidate}" "$(readlink -f "${candidate}" 2>/dev/null || true)" >&2
      ls -l "${candidate}" >&2 2>/dev/null || true
    fi
  done
  exit 127
fi

stream_spur_shared_logs_once() {
  local job_id="$1"
  local run_dir="${LOG_ROOT}/slurm_job-${job_id}"
  local log_file rel_path current_line

  [[ -d "${run_dir}" ]] || return 0

  shopt -s nullglob
  for log_file in \
    "${run_dir}"/rank-*/container*.log \
    "${run_dir}"/logs/*.log \
    "${run_dir}"/logs/*/*.log; do
    rel_path="${log_file#"${run_dir}/"}"
    current_line="${SPUR_SHARED_LOG_LINES[${log_file}]:-0}"
    SPUR_SHARED_LOG_LINES["${log_file}"]="$(stream_file_lines "${log_file}" "[spur:${rel_path}] " "${current_line}")"
  done
  shopt -u nullglob
}

if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
  SLURM_EXTRA_LOG_STREAMER=stream_spur_shared_logs_once
fi
source "${REPO_ROOT}/.github/scripts/slurm_submit_helpers.sh"
install_slurm_cancel_traps

IFS=',' read -r -a NODE_ARRAY <<< "${NODE_LIST}"
if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi350" ]]; then
  SUBMIT_SCRIPT="${LOG_ROOT}/submit-${ATOMESH_CELL_ID}.sbatch.sh"
  cat > "${SUBMIT_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${SLURM_JOB_NAME}
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
  SBATCH_CMD=(sbatch --controller "${SPUR_CONTROLLER_ADDR}" --export=ALL "${SUBMIT_SCRIPT}")
else
  SBATCH_CMD=(
    sbatch
    --parsable
    --exclusive
    --export=ALL 
    --job-name "${SLURM_JOB_NAME}"
  )
  if [[ "${USES_SPUR_CONTROLLER}" == "1" ]]; then
    SBATCH_CMD+=(--controller "${SPUR_CONTROLLER_ADDR}")
  fi
  if [[ -n "${SLURM_ACCOUNT}" ]]; then
    SBATCH_CMD+=(--account "${SLURM_ACCOUNT}")
  fi
  if [[ -n "${SLURM_PARTITION}" ]]; then
    SBATCH_CMD+=(--partition "${SLURM_PARTITION}")
  fi
  if [[ "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-crusoe-mi355" || "${SLURM_SUBMIT_RUNNER}" == "atomesh-cicd-mi355-crusoe" ]]; then
    SBATCH_CMD+=(--qos amd-aifw-dev-qos)
  fi
  SBATCH_CMD+=(
    --nodes "${NUM_NODES}"
    --ntasks "${NUM_NODES}"
    --ntasks-per-node 1
    --cpus-per-task "${SLURM_CPUS_PER_TASK}"
    --gres "gpu:${SLURM_GPUS_PER_NODE}"
    --time "${SLURM_TIME_LIMIT}"
  )
  if [[ -n "${NODE_LIST}" ]]; then
    SBATCH_CMD+=(--nodelist "${NODE_LIST}")
  fi
  SBATCH_CMD+=(
    --output "${SLURM_OUTPUT}"
    --error "${SLURM_ERROR}"
    "${JOB_SCRIPT}"
  )
fi

echo "=== submitting Slurm job ==="
printf ' %q' "${SBATCH_CMD[@]}"
echo
write_slurm_cancel_helper ""

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
SLURM_JOB_ACTIVE=1
echo "${JOB_ID}" | tee "${RESULT_DIR}/${ATOMESH_CELL_ID}.slurm-job-id"
write_slurm_cancel_helper "${JOB_ID}"

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
