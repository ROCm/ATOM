#!/usr/bin/env bash
set -euo pipefail

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
    "DECODE_CUDAGRAPH": decode.get("cudagraph", ""),
    "PREFILL_PORT": prefill.get("port", 8010),
    "DECODE_PORT": decode.get("port", 8020),
    "ROUTER_PORT": router.get("port", 8000),
    "ROUTER_POLICY": router.get("policy", "random"),
    "PROMETHEUS_PORT": router.get("prometheus_port", 29100),
    "KV_CACHE_DTYPE": server_args.get("kv_cache_dtype", "fp8"),
    "BLOCK_SIZE": server_args.get("block_size", 16),
    "MEM_FRACTION": server_args.get("gpu_memory_utilization", 0.85),
    "MAX_NUM_SEQS": server_args.get("max_num_seqs", 256),
    "EXTRA_SERVER_ARGS": server_args.get("extra_args", ""),
    "RUN_EVAL": str(cell.get("run_eval", False)).lower(),
    "EVAL_TASK": accuracy.get("task", "gsm8k"),
    "EVAL_FEWSHOT": accuracy.get("fewshot", 3),
    "EVAL_LIMIT": "" if accuracy.get("limit") is None else accuracy.get("limit"),
    "SLURM_ACCOUNT": runner.get("slurm_account", "amd-frameworks"),
    "SLURM_PARTITION": runner.get("slurm_partition", "amd-frameworks"),
    "SLURM_CPUS_PER_TASK": runner.get("cpus_per_task", 114),
    "SLURM_GPUS_PER_NODE": runner.get("gpus_per_node", 8),
    "SLURM_TIME_LIMIT": runner.get("time_limit", "06:00:00"),
    "SLURM_LOG_ROOT": runner.get("log_root", "/it-share/ATOMESH_LOG/"),
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
export LOG_ROOT="${SLURM_LOG_ROOT%/}/${ATOMESH_CELL_ID}-${GITHUB_RUN_ID:-local}-$(date +%Y%m%d%H%M%S)"
export SLURM_OUTPUT="${LOG_ROOT}/slurm-%j.out"
export SLURM_ERROR="${LOG_ROOT}/slurm-%j.err"

echo "=== ATOMesh benchmark cell ==="
echo "cell=${ATOMESH_CELL_ID}"
echo "model=${MODEL_NAME}"
echo "topology=${DISPLAY_TOPOLOGY}"
echo "nodes=${NODE_LIST}"
echo "isl=${ISL_LIST} osl=${OSL} concurrency=${CONC_LIST}"
echo "log_root=${LOG_ROOT}"

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

IFS=',' read -r -a NODE_ARRAY <<< "${NODE_LIST}"
SBATCH_CMD=(
  sbatch
  --parsable
  --wait
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

echo "=== submitting Slurm job ==="
printf ' %q' "${SBATCH_CMD[@]}"
echo

set +e
JOB_ID="$("${SBATCH_CMD[@]}")"
SBATCH_RC=$?
set -e
echo "${JOB_ID}" | tee "${RESULT_DIR}/${ATOMESH_CELL_ID}.slurm-job-id"
echo "sbatch exit code: ${SBATCH_RC}"

if [[ -d "${LOG_ROOT}" ]]; then
  mkdir -p "${RESULT_DIR}/${ATOMESH_CELL_ID}"
  cp -a "${LOG_ROOT}/." "${RESULT_DIR}/${ATOMESH_CELL_ID}/" || true
fi

exit "${SBATCH_RC}"
