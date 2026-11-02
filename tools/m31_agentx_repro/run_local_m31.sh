#!/usr/bin/env bash
# Run one ATOMesh CI cell of MiniMax-M3.1-Flash-MXFP4-Stack locally on THIS node
# (which must be held by my own Slurm job), reusing the CI scripts:
# pd_matrix.py -> pd_submit.sh export block -> pd_slurm_job.sh --spur-worker.
#
# Usage (on the node): run_local_m31.sh <tag> <case-name> [overrides.sh]
#   BASE      output root (logs/, src/, cells/); required
#   SRC_REPO  ATOM checkout (default: this repo) whose tree (incl. uncommitted edits) is snapshotted
#   IMAGE     docker image (default rocm/atom-dev:m31-mxfp4-stack-20261031)
#   SUITE     suite (default mooncake_store)
#   overrides.sh is sourced after the CI exports (ATOMESH_PREFILL_ENV_* etc.)
set -euo pipefail
TAG="${1:?tag}"; CASE="${2:?case name}"; OVERRIDES="${3:-}"
SRC_REPO="${SRC_REPO:-$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)}"
IMAGE="${IMAGE:-rocm/atom-dev:m31-mxfp4-stack-20261031}"
SUITE="${SUITE:-mooncake_store}"
MODEL="${MODEL:-MiniMax-M3.1-Flash-MXFP4-Stack}"
BASE="${BASE:?set BASE to a writable shared output dir}"
STAMP="$(date +%Y%m%d%H%M%S)"
SNAP="${BASE}/src/${TAG}-${STAMP}"
mkdir -p "${BASE}/src" "${BASE}/cells" "${BASE}/logs"

# Snapshot the working tree (tracked files + uncommitted edits) into a detached worktree.
snap_sha="${SNAP_SHA:-$(git -C "${SRC_REPO}" stash create)}"
snap_sha="${snap_sha:-$(git -C "${SRC_REPO}" rev-parse HEAD)}"
git -C "${SRC_REPO}" worktree add --detach "${SNAP}" "${snap_sha}" >/dev/null
echo "snapshot ${snap_sha} -> ${SNAP}"
cd "${SNAP}"

CELL_FILE="${BASE}/cells/cell-${TAG}-${STAMP}.json"
ATOMESH_SLURM_ACCOUNT= ATOMESH_MODEL_ROOT=/share_nfs/models ATOMESH_LOG_ROOT="${BASE}/logs" \
ATOMESH_SLURM_PARTITION=amd-spur ATOMESH_SLURM_SUBMIT_RUNNER=atomesh-cicd \
ATOMESH_PD_RANK_MAPPING_POLICY=none ATOMESH_1P1D_NODES="$(hostname)" \
  python3 .github/scripts/atomesh/pd_matrix.py --suite "${SUITE}" --model "${MODEL}" --case "${CASE}" \
  --image "${IMAGE}" --output "${CELL_FILE}.all" >/dev/null
python3 - "${CELL_FILE}.all" "${CELL_FILE}" <<'PY'
import json, sys
cells = json.load(open(sys.argv[1]))["include"]
assert len(cells) == 1, [c["id"] for c in cells]
json.dump(cells[0], open(sys.argv[2], "w"), indent=2)
PY
export CELL_JSON="$(cat "${CELL_FILE}")"
# Reuse pd_submit.sh's own cell->env mapping (its first python heredoc).
eval "$(awk '/^python3 - <<.PY.$/{f=1;next} f&&/^PY$/{exit} f' .github/scripts/atomesh/pd_submit.sh | python3 -)"

# The image ships the atomesh router built from the same source.
export ATOMESH_BUILD_MESH=false ATOMESH_MESH_BUILD_PROFILE=release ATOMESH_DASHBOARD_HARDWARE=MI355X
export ATOMESH_HOST_CACHE_DROP_TIMEOUT="${ATOMESH_HOST_CACHE_DROP_TIMEOUT:-900}"
export ATOMESH_RUN_TOKEN="$(python3 -c 'import uuid;print(uuid.uuid4().hex)')"
export LOG_ROOT="${BASE}/logs/${TAG}-${STAMP}"
export GITHUB_WORKSPACE="${SNAP}"
export SPUR_JOB_ID="${LOCAL_JOB_ID:-31${STAMP: -6}}"
export SPUR_TASK_OFFSET=0
export SPUR_PEER_NODES="$(getent hosts "$(hostname)" | awk '{print $1}')"
export SPUR_NODELIST="$(hostname)"
mkdir -p "${LOG_ROOT}"
if [[ -n "${OVERRIDES}" ]]; then
  echo "applying overrides ${OVERRIDES}"; cp "${OVERRIDES}" "${LOG_ROOT}/overrides.sh"
  # shellcheck disable=SC1090
  source "${OVERRIDES}"
fi
echo "${snap_sha}" > "${LOG_ROOT}/source-sha"
git -C "${SRC_REPO}" diff "$(git -C "${SRC_REPO}" rev-parse HEAD)" "${snap_sha}" > "${LOG_ROOT}/source.diff" || true
cp "${CELL_FILE}" "${LOG_ROOT}/cell.json"
echo "LOG_ROOT=${LOG_ROOT}"
set +e
bash .github/scripts/atomesh/pd_slurm_job.sh --spur-worker
rc=$?
set -e
echo "rc=${rc}" | tee "${LOG_ROOT}/local-rc"
# aiperf's server-metrics export is ~10 GB per run and only feeds dashboards
# we do not publish locally; keep the profile exports and server logs.
find "${LOG_ROOT}" -name 'server_metrics_export.json' -size +1G -print -delete 2>/dev/null || true
cd /; git -C "${SRC_REPO}" worktree remove --force "${SNAP}" || true
exit "${rc}"
