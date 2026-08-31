#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/slurm-24.05.5.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

usage() {
  cat <<'USAGE'
Usage:
  submit.sh --plugin <vllm|sglang> --cell-json <json> [--result-dir <dir>] [--dry-run]

Submits one plugin CI accuracy cell (vLLM or SGLang) to Slurm. The cell JSON is
built by atom-vllm-test.yaml / atom-sglang-test.yaml matrix entries.
USAGE
}

PLUGIN=""
CELL_JSON=""
RESULT_DIR="${RESULT_DIR:-plugin-ci-results}"
DRY_RUN=0
JOB_ID=""
SLURM_JOB_ACTIVE=0
SCANCEL_SENT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plugin)
      PLUGIN="$2"
      shift 2
      ;;
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

if [[ -z "${PLUGIN}" || -z "${CELL_JSON}" ]]; then
  echo "ERROR: --plugin and --cell-json are required" >&2
  usage >&2
  exit 2
fi

case "${PLUGIN}" in
  vllm|sglang) ;;
  *)
    echo "ERROR: unsupported plugin ${PLUGIN}; expected vllm or sglang" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
JOB_SCRIPT="${REPO_ROOT}/.github/scripts/plugin_ci/${PLUGIN}_slurm_job.sh"
mkdir -p "${RESULT_DIR}"

export CELL_JSON PLUGIN
eval "$(
python3 - <<'PY'
import json
import os
import shlex

cell = json.loads(os.environ["CELL_JSON"])
runner = cell.get("runner", {})

def shell_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value

def q(value):
    return shlex.quote(str(shell_value(value)))

slurm_submit_runner = runner.get(
    "slurm_submit_runner", "atom-mi355-8gpu-vllm-sgl-ci"
)

default_spur_accounting_addr = "http://crs-m2m-cpu-spur-005.crusoe.amd.com:6819"
spur_controller_addr = "http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817"

nodes = cell.get("nodes") or []
if isinstance(nodes, str):
    node_list = nodes
else:
    node_list = ",".join(str(node) for node in nodes)

exports = {
    "PLUGIN_CI_CELL_ID": cell["id"],
    "PLUGIN_CI_PLUGIN": os.environ["PLUGIN"],
    "SLURM_SUBMIT_RUNNER": slurm_submit_runner,
    "SLURM_ACCOUNT": runner.get("slurm_account", "amd-aifw-dev"),
    "SLURM_PARTITION": runner.get("slurm_partition", "amd-spur"),
    "SLURM_CPUS_PER_TASK": runner.get("cpus_per_task", 114),
    "SLURM_GPUS_PER_NODE": runner.get("gpus_per_node", 8),
    "SLURM_TIME_LIMIT": runner.get("time_limit", "03:00:00"),
    "SLURM_LOG_ROOT": runner.get(
        "log_root", "/home/junyyang/ATOM_PLUGIN_RUNNER/LOG/"
    ),
    "NODE_LIST": node_list,
    "NUM_NODES": cell.get("num_nodes", 1),
    "SPUR_CONTROLLER_ADDR": spur_controller_addr,
    "SPUR_ACCOUNTING_ADDR": runner.get(
        "spur_accounting_addr",
        os.environ.get("SPUR_ACCOUNTING_ADDR", default_spur_accounting_addr),
    ),
    "GITHUB_REPO_URL": cell.get("github_repo_url", ""),
    "GITHUB_COMMIT_SHA": cell.get("github_commit_sha", ""),
    "PR_BASE_SHA": cell.get("pr_base_sha", ""),
    "PR_HEAD_SHA": cell.get("pr_head_sha", ""),
    "AITER_ARTIFACT_ID": cell.get("aiter_artifact_id", ""),
    "SKIP_DOCKER_LOGIN": cell.get("skip_docker_login", "0"),
    "ATOM_BASE_NIGHTLY_IMAGE": cell.get("atom_base_nightly_image", "rocm/atom-dev:latest"),
    "NIGHTLY_PLUGIN_IMAGE_TAG": cell.get("nightly_plugin_image_tag", ""),
    "CONTAINER_NAME": cell.get("container_name", f"plugin_ci_{cell['id']}"),
}

for key, value in cell.get("matrix", {}).items():
    exports[f"MATRIX_{key.upper()}"] = value

for key, value in exports.items():
    print(f"export {key}={q(value)}")
PY
)"

SLURM_QOS="${SLURM_QOS:-amd-burst-qos}"
USES_SPUR_CONTROLLER=1
case "${RUNNER_NAME:-}" in
  plugin_runner_crusoe_v2_01|plugin_runner_crusoe_v2_02)
    export SPUR_CONTROLLER_ADDR="http://crs-m2m-cpu-spur-v2-001.crusoe.amd.com:6817"
    export SPUR_ACCOUNTING_ADDR="${SPUR_V2_ACCOUNTING_ADDR:-http://crs-m2m-cpu-spur-v2-001.crusoe.amd.com:6819}"
    export SLURM_ACCOUNT="amd-aifw-dev"
    export SLURM_PARTITION=""
    SLURM_QOS="amd-aifw-dev-qos"
    ;;
  gbt350-odcdh1-b10-1)
    USES_SPUR_CONTROLLER=0
    NODE_LIST=""
    export SLURM_ACCOUNT=""
    export SLURM_PARTITION=""
    SLURM_QOS=""
    ;;
esac

export RESULT_DIR
CURRENT_USER="$(id -un 2>/dev/null || id -u)"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT//\$\{USER\}/${CURRENT_USER}}"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT//\$USER/${CURRENT_USER}}"
export LOG_ROOT="${SLURM_LOG_ROOT%/}/${PLUGIN_CI_CELL_ID}-${GITHUB_RUN_ID:-local}-$(date +%Y%m%d%H%M%S)"
export SLURM_JOB_NAME="${PLUGIN_CI_CELL_ID}-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
export SLURM_OUTPUT="${LOG_ROOT}/slurm-%j.out"
export SLURM_ERROR="${LOG_ROOT}/slurm-%j.err"
export SLURM_CANCEL_HELPER="${RESULT_DIR}/${PLUGIN_CI_CELL_ID}.slurm-cancel.sh"
SUBMIT_LOCK_FILE="${SLURM_SUBMIT_LOCK_FILE:-/home/junyyang/ATOM_PLUGIN_RUNNER/LOG/.sbatch-submit.lock}"
SLURM_LOG_POLL_INTERVAL="${SLURM_LOG_POLL_INTERVAL:-30}"

echo "=== plugin CI cell ==="
echo "cell=${PLUGIN_CI_CELL_ID}"
echo "plugin=${PLUGIN_CI_PLUGIN}"
echo "runner_name=${RUNNER_NAME:-unknown}"
echo "nodes=${NODE_LIST:-auto}"
echo "slurm_account=${SLURM_ACCOUNT:-default}"
echo "slurm_partition=${SLURM_PARTITION:-default}"
echo "slurm_qos=${SLURM_QOS:-default}"
if [[ "${USES_SPUR_CONTROLLER}" == "1" ]]; then
  echo "spur_controller=${SPUR_CONTROLLER_ADDR}"
else
  echo "spur_controller=default"
fi
echo "slurm_job_name=${SLURM_JOB_NAME}"
echo "log_root=${LOG_ROOT}"
echo "submit_lock_file=${SUBMIT_LOCK_FILE}"

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
  exit 127
fi

source "${REPO_ROOT}/.github/scripts/slurm_submit_helpers.sh"
install_slurm_cancel_traps

SBATCH_CMD=(
  sbatch
  --parsable
  --exclusive
  --export=ALL
)
if [[ -n "${SLURM_ACCOUNT}" ]]; then
  SBATCH_CMD+=(--account "${SLURM_ACCOUNT}")
fi
if [[ -n "${SLURM_QOS}" ]]; then
  SBATCH_CMD+=(--qos "${SLURM_QOS}")
fi
if [[ -n "${SLURM_PARTITION}" ]]; then
  SBATCH_CMD+=(--partition "${SLURM_PARTITION}")
fi
if [[ -n "${NODE_LIST}" ]]; then
  SBATCH_CMD+=(--nodelist "${NODE_LIST}")
fi
SBATCH_CMD+=(
  --job-name "${SLURM_JOB_NAME}"
)
if [[ "${USES_SPUR_CONTROLLER}" == "1" ]]; then
  SBATCH_CMD+=(--controller "${SPUR_CONTROLLER_ADDR}")
fi
SBATCH_CMD+=(
  --nodes "${NUM_NODES}"
  --ntasks "${NUM_NODES}"
  --ntasks-per-node 1
  --cpus-per-task "${SLURM_CPUS_PER_TASK}"
  --gres "gpu:${SLURM_GPUS_PER_NODE}"
  --time "${SLURM_TIME_LIMIT}"
  --output "${SLURM_OUTPUT}"
  --error "${SLURM_ERROR}"
  "${JOB_SCRIPT}"
)

echo "=== submitting Slurm job ==="
printf ' %q' "${SBATCH_CMD[@]}"
echo
write_slurm_cancel_helper ""

if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock not found; cannot serialize Slurm submissions" >&2
  exit 127
fi
mkdir -p "$(dirname "${SUBMIT_LOCK_FILE}")"
exec 9>"${SUBMIT_LOCK_FILE}"
echo "=== waiting for global Slurm submission lock ==="
flock -x 9
echo "=== acquired global Slurm submission lock ==="

set +e
SBATCH_OUTPUT="$("${SBATCH_CMD[@]}")"
SBATCH_RC=$?
set -e
flock -u 9
exec 9>&-
echo "${SBATCH_OUTPUT}"

if [[ "${SBATCH_RC}" -ne 0 ]]; then
  echo "sbatch submit exit code: ${SBATCH_RC}"
  exit "${SBATCH_RC}"
fi

JOB_ID="$(parse_sbatch_job_id "${SBATCH_OUTPUT}")"
SLURM_JOB_ACTIVE=1
echo "${JOB_ID}" | tee "${RESULT_DIR}/${PLUGIN_CI_CELL_ID}.slurm-job-id"
write_slurm_cancel_helper "${JOB_ID}"

set_slurm_job_log_paths "${JOB_ID}"
monitor_slurm_job "${JOB_ID}"
read_slurm_exit_code "${JOB_ID}"
SLURM_STATUS_FILE="${LOG_ROOT}/slurm_job-${JOB_ID}/slurm-job.rc"
if [[ "${SLURM_STATE}" == "unknown" && -s "${SLURM_STATUS_FILE}" ]]; then
  batch_rc="$(tr -d '[:space:]' < "${SLURM_STATUS_FILE}")"
  if [[ "${batch_rc}" =~ ^[0-9]+$ ]]; then
    SLURM_JOB_RC="${batch_rc}"
    SLURM_EXIT_CODE="${batch_rc}:0"
    if [[ "${batch_rc}" -eq 0 ]]; then
      SLURM_STATE="COMPLETED"
    else
      SLURM_STATE="FAILED"
    fi
    echo "Using batch script exit status because Slurm accounting is unavailable."
  else
    echo "WARNING: invalid batch script exit status: ${batch_rc}" >&2
  fi
fi
SLURM_JOB_ACTIVE=0
SBATCH_RC="${SLURM_JOB_RC}"
echo "slurm_state=${SLURM_STATE}"
echo "slurm_exit_code=${SLURM_EXIT_CODE}"
echo "slurm job exit code: ${SBATCH_RC}"

if [[ -d "${LOG_ROOT}" ]]; then
  mkdir -p "${RESULT_DIR}/${PLUGIN_CI_CELL_ID}"
  cp -a "${LOG_ROOT}/." "${RESULT_DIR}/${PLUGIN_CI_CELL_ID}/" || true
fi

exit "${SBATCH_RC}"
