#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_kimi_local.sh [options]

Run one Kimi ATOMesh benchmark locally through Slurm, without GitHub Actions.

Options:
  --case NAME           Benchmark case (default: kimi-k25-1p1d-tp4)
  --image IMAGE         Docker image (default: rocm/atom-dev:mooncake_main)
  --nodes LIST          Optional comma-separated Slurm node list
  --concurrency LIST    Benchmark concurrency (default: 1)
  --eval-concurrency N  Accuracy concurrency if eval is enabled (default: 64)
  --with-eval           Run the catalog's accuracy evaluation
  --result-dir DIR      Local result directory
  --log-root DIR        Persistent Slurm log directory
  --dry-run             Generate and print the cell without submitting
  -h, --help            Show this help
EOF
}

CASE_NAME="kimi-k25-1p1d-tp4"
IMAGE="rocm/atom-dev:mooncake_main"
NODES=""
CONCURRENCY="1"
EVAL_CONCURRENCY="64"
WITH_EVAL=0
DRY_RUN=0
TIMESTAMP="$(date +%Y%m%d%H%M%S)"
RESULT_DIR="${HOME}/ATOMesh_LOCAL_RESULTS/kimi-${TIMESTAMP}"
LOG_ROOT="${HOME}/ATOMesh_LOCAL_LOG"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case) CASE_NAME="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --eval-concurrency) EVAL_CONCURRENCY="$2"; shift 2 ;;
    --with-eval) WITH_EVAL=1; shift ;;
    --result-dir) RESULT_DIR="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MATRIX_SCRIPT="${REPO_ROOT}/.github/scripts/atomesh/pd_matrix.py"
SUBMIT_SCRIPT="${REPO_ROOT}/.github/scripts/atomesh/pd_submit.sh"
MATRIX_FILE="$(mktemp "${TMPDIR:-/tmp}/kimi-local-matrix.XXXXXX.json")"
trap 'rm -f "${MATRIX_FILE}"' EXIT

# Crusoe defaults formerly supplied by atomesh-benchmark.yaml.
export ATOMESH_SLURM_ACCOUNT="${ATOMESH_SLURM_ACCOUNT:-amd-aifw-dev}"
export ATOMESH_SLURM_PARTITION="${ATOMESH_SLURM_PARTITION:-}"
export ATOMESH_SLURM_SUBMIT_RUNNER="${ATOMESH_SLURM_SUBMIT_RUNNER:-atomesh-cicd-mi355-crusoe}"
export ATOMESH_LOG_ROOT="${ATOMESH_LOG_ROOT:-${LOG_ROOT}}"
export ATOMESH_MODEL_ROOT="${ATOMESH_MODEL_ROOT:-${HOME}/models/amd}"
export ATOMESH_PD_RANK_MAPPING_POLICY="${ATOMESH_PD_RANK_MAPPING_POLICY:-none}"
export ATOMESH_1P1D_NODES="${ATOMESH_1P1D_NODES:-${NODES}}"
export ATOMESH_2P1D_NODES="${ATOMESH_2P1D_NODES:-${NODES}}"
export SLURM_NODELIST="${SLURM_NODELIST:-${NODES}}"

python3 "${MATRIX_SCRIPT}" \
  --suite nightly \
  --case "${CASE_NAME}" \
  --image "${IMAGE}" \
  --benchmark-concurrency "${CONCURRENCY}" \
  --eval-concurrency "${EVAL_CONCURRENCY}" \
  --output "${MATRIX_FILE}"

CELL_JSON="$(
  python3 - "${MATRIX_FILE}" "${WITH_EVAL}" <<'PY'
import json
import sys
from pathlib import Path

matrix = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cells = matrix.get("include", [])
if len(cells) != 1:
    raise SystemExit(f"expected exactly one benchmark cell, got {len(cells)}")
cell = cells[0]
if sys.argv[2] != "1":
    cell["run_eval"] = False
cell.setdefault("env", {}).setdefault("common", {})[
    "HSA_NO_SCRATCH_RECLAIM"
] = "1"
print(json.dumps(cell, separators=(",", ":")))
PY
)"

mkdir -p "${RESULT_DIR}" "${ATOMESH_LOG_ROOT}"
echo "=== Local Kimi ATOMesh run ==="
echo "case=${CASE_NAME}"
echo "image=${IMAGE}"
echo "nodes=${NODES:-<auto>}"
echo "result_dir=${RESULT_DIR}"
echo "log_root=${ATOMESH_LOG_ROOT}"
echo "run_eval=$([[ "${WITH_EVAL}" -eq 1 ]] && echo true || echo false)"
echo "HSA_NO_SCRATCH_RECLAIM=1"

submit_args=(
  --cell-json "${CELL_JSON}"
  --result-dir "${RESULT_DIR}"
)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  submit_args+=(--dry-run)
fi

exec bash "${SUBMIT_SCRIPT}" "${submit_args[@]}"
