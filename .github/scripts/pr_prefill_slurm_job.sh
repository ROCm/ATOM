#!/usr/bin/env bash
set -euo pipefail
: "${SHARED_SOURCE:?}" "${LOG_ROOT:?}" "${IMAGE:?}" "${MODEL_ROOT:?}"
job_id="${SLURM_JOB_ID:-${SPUR_JOB_ID:-}}"
[[ "$job_id" =~ ^[0-9]+$ ]]
export CONTAINER_NAME="atom-pr-review-${GITHUB_RUN_ID}-${PR}-${job_id}"
test -d "$MODEL_ROOT"
mkdir -p "$SHARED_SOURCE/review-evidence"
evidence="$SHARED_SOURCE/review-evidence"
printf 'node=%s\njob=%s\naccount=%s\npartition=%s\nqos=%s\n' \
  "$(hostname)" "$job_id" "$SLURM_ACCOUNT" "$SLURM_PARTITION" "$SLURM_QOS" \
  > "$evidence/slurm-allocation.txt"
cleanup() {
  rc=$?
  trap - EXIT HUP INT TERM
  timeout 30 docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  if ! timeout 10 docker ps -a --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}} {{.Status}}' \
      > "$evidence/containers-after.txt"; then
    printf 'container query failed\n' > "$evidence/containers-after.txt"
    rc=1
  fi
  [[ ! -s "$evidence/containers-after.txt" ]] || rc=1
  printf '%s\n' "$rc" > "$evidence/workload-exit-code.txt"
  printf '%s\n' "$rc" > "$LOG_ROOT/slurm-job.rc.tmp"
  mv "$LOG_ROOT/slurm-job.rc.tmp" "$LOG_ROOT/slurm-job.rc"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
cat > "$LOG_ROOT/model-env.list" <<'ENV'
AITER_LOG_LEVEL=WARNING
PYTHONHASHSEED=0
PYTHONNOUSERSITE=1
ATOM_MONO_ENABLE=0
ATOM_DCP_PREFILL_QREP=1
ATOM_MLA_ABSORB_BF16=1
AITER_CUSTOM_AR_DISABLE_LAMPORT=1
AITER_REUSE_IDENTICAL_COMM_GROUPS=0
ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0
ATOM_GDN_SSM_DTYPE=fp32
ATOM_STATE_CHECKPOINT_DEMAND=0
AITER_SITUV2_A4W4=1
AITER_FLYDSL_USE_HERD=1
AITER_FLYDSL_HERD_MIN_M=64
NCCL_SOCKET_IFNAME=lo
MORI_SOCKET_IFNAME=lo
ENV
group_flags=()
for group in video render; do
  gid=$(getent group "$group" | cut -d: -f3 || true)
  [[ -z "$gid" ]] || group_flags+=(--group-add "$gid")
done
docker run -dt --pull always --user "$(id -u):$(id -g)" --device=/dev/kfd --device=/dev/dri \
  -v "$SHARED_SOURCE":/workspace -v "$MODEL_ROOT":/models -w /workspace \
  --ipc=host "${group_flags[@]}" --shm-size=16G --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864 \
  --env-file "$LOG_ROOT/model-env.list" -e HF_TOKEN -e PR -e HEAD_SHA -e BASE_SHA -e AITER_GIT_REF \
  -e HOME=/tmp/atom-pr-review-runtime/home \
  -e XDG_CACHE_HOME=/tmp/atom-pr-review-runtime/cache \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/atom-pr-review-runtime/cache/torchinductor \
  -e TRITON_CACHE_DIR=/tmp/atom-pr-review-runtime/cache/triton \
  -e AITER_CACHE_DIR=/tmp/atom-pr-review-runtime/cache/aiter \
  -e AITER_JIT_DIR=/tmp/atom-pr-review-runtime/cache/aiter/jit \
  -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/atom-pr-review-runtime/cache/flydsl \
  -e AITER_SOURCE_PATH=/tmp/atom-pr-review-runtime/aiter \
  --name "$CONTAINER_NAME" "$IMAGE"
docker inspect --format '{{.Image}}' "$CONTAINER_NAME" > "$evidence/image-id.txt"
cd "$SHARED_SOURCE"
docker exec "$CONTAINER_NAME" bash /workspace/_review_harness/.github/scripts/pr_prefill_slurm_runtime.sh
docker exec "$CONTAINER_NAME" bash -c '
  export PATH="/tmp/atom-pr-review-runtime/venv/bin:/tmp/atom-pr-review-runtime/bin:$PATH"
  export PYTHONPATH="/workspace:$AITER_SOURCE_PATH${PYTHONPATH:+:$PYTHONPATH}"
  exec python /workspace/_review_harness/.github/scripts/pr_prefill_validation.py
'
