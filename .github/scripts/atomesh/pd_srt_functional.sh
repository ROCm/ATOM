#!/usr/bin/env bash
set -euo pipefail
runtime="${REPO_ROOT:?}/.sa-srt-runtime"
export SRTSLURM_CONFIG="${RUN_DIR:?}/srtslurm.yaml"
export SRTCTL_JOB_ID="${JOB_ID:?}-sa-functional"
export SRTCTL_OUTPUT_DIR="${RUN_DIR}"
mkdir -p "${RUN_DIR}/logs"
sweep_pid=""
cleanup_srt() {
  local rc=$? query_rc=0 id
  trap - EXIT INT TERM
  if [[ -n "${sweep_pid}" ]]; then
    kill -TERM -- "-${sweep_pid}" 2>/dev/null || true
  fi
  # The exact job label belongs only to this allocation and case.
  docker ps -aq --filter "label=srtctl.job=${SRTCTL_JOB_ID}" > "${RUN_DIR}/srt-cleanup-ids.txt" || query_rc=$?
  if [[ "${query_rc}" == 0 ]]; then
    while IFS= read -r id; do
      [[ -z "${id}" ]] || timeout 20 docker rm -f "${id}" >> "${RUN_DIR}/srt-cleanup.log" 2>&1 || rc=1
    done < "${RUN_DIR}/srt-cleanup-ids.txt"
  else
    rc=1
  fi
  docker ps -aq --filter "label=srtctl.job=${SRTCTL_JOB_ID}" > "${RUN_DIR}/srt-cleanup-remaining.txt" || rc=1
  [[ ! -s "${RUN_DIR}/srt-cleanup-remaining.txt" ]] || rc=1
  printf '%s\n' "${rc}" > "${RUN_DIR}/srt-exit-code.txt"
  if [[ ! -f "${RUN_DIR}/logs/perf-analysis.md" ]]; then
    printf '# Functional run\n\nWorkload exit: %s. No performance window was requested. TTFT p50/p95/p99, ITL p50/p99, throughput and AgentX Pareto/SLO are unavailable by design. Functionality and KV evidence require log review. See srtctl.log, sa-flow-contract.json and srt cleanup records. Run dispatched using run-pd-ci.\n' "${rc}" > "${RUN_DIR}/logs/perf-analysis.md"
  fi
  exit "${rc}"
}
trap cleanup_srt EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
[[ -f "${runtime}/source-sha.txt" ]] || { echo 'Pinned srtctl source is missing' >&2; exit 1; }
stage="$(mktemp -d "${TMPDIR:-/tmp}/atomesh-srt-${JOB_ID}.XXXXXX")"
export SRTCTL_SOURCE_DIR="${stage}/source"
python3 - "${runtime}/source" "${SRTCTL_SOURCE_DIR}" <<'PY'
import shutil
import sys
shutil.copytree(sys.argv[1], sys.argv[2], ignore=shutil.ignore_patterns(".git", ".venv"))
PY
python3 -m venv "${stage}/venv" || uv venv --seed --python python3 "${stage}/venv"
python="${stage}/venv/bin/python"
"${python}" -m pip install "${SRTCTL_SOURCE_DIR}" > "${RUN_DIR}/srt-install.log" 2>&1
"${python}" -m pip freeze > "${RUN_DIR}/requirements-installed.txt"
cp "${runtime}/source-sha.txt" "${RUN_DIR}/"
"${python}" "${REPO_ROOT}/.github/scripts/atomesh/pd_srt_functional.py" \
  --contract "${REPO_ROOT}/${ATOMESH_ENV_ATOMESH_PR5090_CONTRACT:?}" \
  --model-path "${MODEL_PATH:?}" --run-dir "${RUN_DIR}" --uid "$(id -u):$(id -g)"
"${python}" -m srtctl.cli.submit dry-run -f "${RUN_DIR}/recipe.yaml" \
  -o "${RUN_DIR}/srt-plan" > "${RUN_DIR}/srt-dry-run.log" 2>&1
echo "[functional] starting native srtctl orchestration job=${SRTCTL_JOB_ID}"
setsid "${python}" -m srtctl.cli.do_sweep "${RUN_DIR}/recipe.yaml" > "${RUN_DIR}/srtctl.log" 2>&1 &
sweep_pid=$!
wait "${sweep_pid}"
sweep_pid=""
