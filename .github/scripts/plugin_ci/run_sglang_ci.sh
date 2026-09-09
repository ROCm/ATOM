#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
cd "${REPO_ROOT}"

CONTAINER_NAME="${CONTAINER_NAME:-plugin_ci_sglang}"
SGLANG_IMAGE_TAG="${SGLANG_IMAGE_TAG:-atom_sglang:ci}"
NIGHTLY_SGLANG_IMAGE_TAG="${NIGHTLY_PLUGIN_IMAGE_TAG:-rocm/atom-dev:sglang-latest}"
ATOM_BASE_NIGHTLY_IMAGE="${ATOM_BASE_NIGHTLY_IMAGE:-rocm/atom-dev:latest}"
RESULT_DIR="${RESULT_DIR:-plugin-ci-results}"
GITHUB_REPO_URL="${GITHUB_REPO_URL:?GITHUB_REPO_URL is required}"
GITHUB_COMMIT_SHA="${GITHUB_COMMIT_SHA:?GITHUB_COMMIT_SHA is required}"
AITER_ARTIFACT_ID="${AITER_ARTIFACT_ID:-}"
ARTIFACTS_COLLECTED=0

collect_artifacts() {
  if [[ "${ARTIFACTS_COLLECTED}" -eq 1 ]]; then
    return
  fi
  mkdir -p "${RESULT_DIR}"
  docker cp "${CONTAINER_NAME}:/tmp/atom_sglang_accuracy_results" "${RESULT_DIR}/atom_sglang_accuracy_results" || true
  docker cp "${CONTAINER_NAME}:/tmp/atom_sglang.log" "${RESULT_DIR}/atom_sglang.log" || true
  docker cp "${CONTAINER_NAME}:/tmp/atom_sglang_accuracy_output.txt" "${RESULT_DIR}/atom_sglang_accuracy_output.txt" || true
  ARTIFACTS_COLLECTED=1
}

cleanup() {
  local rc=$?
  trap - EXIT
  collect_artifacts
  docker exec "${CONTAINER_NAME}" bash -lc "if [ -f /tmp/atom_sglang.pid ]; then kill \$(cat /tmp/atom_sglang.pid) || true; fi" || true
  docker stop "${CONTAINER_NAME}" || true
  docker rm -f "${CONTAINER_NAME}" || true
  rm -f Dockerfile.mod || true
  docker rmi "atom_sglang_base:ci" 2>/dev/null || true
  docker rmi "atom_sglang:ci" 2>/dev/null || true
  exit "${rc}"
}
trap cleanup EXIT

MODEL_NAME="${MATRIX_MODEL_NAME:-}"
MODEL_PATH="${MATRIX_MODEL_PATH:-}"
DRAFT_MODEL_PATH="${MATRIX_DRAFT_MODEL_PATH:-}"
EXTRA_ARGS="${MATRIX_EXTRA_ARGS:-}"
ENV_VARS="${MATRIX_ENV_VARS:-}"
LM_EVAL_NUM_FEWSHOT="${MATRIX_LM_EVAL_NUM_FEWSHOT:-3}"
LM_EVAL_NUM_CONCURRENT="${MATRIX_LM_EVAL_NUM_CONCURRENT:-65}"
LM_EVAL_USE_CHAT_COMPLETIONS="${MATRIX_LM_EVAL_USE_CHAT_COMPLETIONS:-0}"
LM_EVAL_EXTRA_MODEL_ARGS="${MATRIX_LM_EVAL_EXTRA_MODEL_ARGS:-}"
ACCURACY_TEST_THRESHOLD="${MATRIX_ACCURACY_TEST_THRESHOLD:-0.0}"

# shellcheck disable=SC1091
source .github/scripts/plugin_ci/cache_model.sh

if [[ -z "${MODEL_NAME}" || -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MATRIX_MODEL_NAME and MATRIX_MODEL_PATH are required" >&2
  exit 2
fi

SGLANG_REF="$(python3 - <<'PY'
import re
from pathlib import Path

text = Path(".github/workflows/docker-release.yaml").read_text(encoding="utf-8")
match = re.search(r'^\s*SGLANG_REF:\s*"([^"]+)"', text, re.MULTILINE)
if not match:
    raise SystemExit("Failed to read SGLANG_REF from .github/workflows/docker-release.yaml")
print(match.group(1))
PY
)"
export SGLANG_REF

if [[ "${SKIP_DOCKER_LOGIN}" != "1" && -n "${DOCKER_PASSWORD:-}" && -n "${DOCKER_USERNAME:-}" ]]; then
  echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
fi

if [[ ! -d aiter-whl ]] || ! compgen -G "aiter-whl/amd_aiter*.whl" >/dev/null; then
  echo "ERROR: aiter wheel not found under ${REPO_ROOT}/aiter-whl" >&2
  exit 1
fi

cat <<'EOF' > Dockerfile.mod
ARG SGLANG_BASE_IMAGE
FROM ${SGLANG_BASE_IMAGE}
ARG GITHUB_REPO_URL
ARG GITHUB_COMMIT_SHA
ARG SGLANG_REF
ARG AITER_ARTIFACT_ID
ARG INSTALL_LM_EVAL=1
LABEL com.rocm.atom.sglang_ref="${SGLANG_REF}"
LABEL com.rocm.atom.aiter_artifact_id="${AITER_ARTIFACT_ID}"
COPY aiter-whl/ /tmp/aiter-whl/
RUN if [ "${INSTALL_LM_EVAL}" = "1" ]; then pip install -U "lm-eval[api]"; else echo "Skip lm-eval install"; fi
RUN pip install hf_transfer
RUN pip install --upgrade "pybind11>=3.0.1"
RUN echo "=== Aiter version BEFORE uninstall ===" && pip show amd-aiter || true && \
    pip uninstall -y amd-aiter || true && \
    rm -rf /app/aiter-test && \
    AITER_WHL="$(ls -t /tmp/aiter-whl/amd_aiter*.whl 2>/dev/null | head -1)" && \
    if [ -z "${AITER_WHL}" ]; then echo "ERROR: No amd_aiter wheel found under /tmp/aiter-whl"; ls -la /tmp/aiter-whl || true; exit 1; fi && \
    pip install "${AITER_WHL}" && \
    rm -rf /tmp/aiter-whl
RUN echo "=== ATOM version BEFORE uninstall ===" && pip show atom || true && \
    pip uninstall -y atom || true && \
    rm -rf /app/ATOM && \
    git clone "${GITHUB_REPO_URL}" /app/ATOM && \
    cd /app/ATOM && \
    git checkout "${GITHUB_COMMIT_SHA}" && \
    (ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true) && \
    pip install -e . --no-deps
EOF

docker rmi "atom_sglang_base:ci" 2>/dev/null || true
docker rmi "atom_sglang:ci" 2>/dev/null || true

BUILD_MODE="full"
if docker pull "${NIGHTLY_SGLANG_IMAGE_TAG}"; then
  LATEST_SGLANG_REF="$(docker inspect --format '{{ index .Config.Labels "com.rocm.atom.sglang_ref" }}' "${NIGHTLY_SGLANG_IMAGE_TAG}" 2>/dev/null || true)"
  if [[ -n "${LATEST_SGLANG_REF}" && "${LATEST_SGLANG_REF}" = "${SGLANG_REF}" ]]; then
    BUILD_MODE="fast"
  fi
fi

if [[ "${BUILD_MODE}" = "fast" ]]; then
  DOCKER_BUILDKIT=1 docker build --network=host \
    -t atom_sglang:ci \
    --build-arg SGLANG_BASE_IMAGE="${NIGHTLY_SGLANG_IMAGE_TAG}" \
    --build-arg GITHUB_REPO_URL="${GITHUB_REPO_URL}" \
    --build-arg GITHUB_COMMIT_SHA="${GITHUB_COMMIT_SHA}" \
    --build-arg SGLANG_REF="${SGLANG_REF}" \
    --build-arg AITER_ARTIFACT_ID="${AITER_ARTIFACT_ID}" \
    --build-arg INSTALL_LM_EVAL=1 \
    -f Dockerfile.mod .
else
  DOCKER_BUILDKIT=1 docker build --pull --network=host \
    --no-cache \
    -t atom_sglang_base:ci \
    --build-arg SGLANG_BASE_IMAGE="${ATOM_BASE_NIGHTLY_IMAGE}" \
    --build-arg GITHUB_REPO_URL="${GITHUB_REPO_URL}" \
    --build-arg GITHUB_COMMIT_SHA="${GITHUB_COMMIT_SHA}" \
    --build-arg SGLANG_REF="${SGLANG_REF}" \
    --build-arg AITER_ARTIFACT_ID="${AITER_ARTIFACT_ID}" \
    --build-arg INSTALL_LM_EVAL=1 \
    -f Dockerfile.mod .

  DOCKER_BUILDKIT=1 docker build --network=host \
    --no-cache \
    -t atom_sglang:ci \
    --build-arg SGLANG_BASE_IMAGE="atom_sglang_base:ci" \
    --build-arg MAX_JOBS=64 \
    --build-arg SGLANG_REF="${SGLANG_REF}" \
    --build-arg INSTALL_LM_EVAL=1 \
    --build-arg INSTALL_FASTSAFETENSORS=1 \
    -f docker/sglang_release.dockerfile .
fi

MODEL_CACHE_MOUNT=""
if [[ -d "/models" ]]; then
  MODEL_CACHE_MOUNT="-v /models:/models"
elif [[ -d "/it-share/models" ]]; then
  MODEL_CACHE_MOUNT="-v /it-share/models:/models"
elif [[ -d "/mnt/dcgpuval/models" ]]; then
  MODEL_CACHE_MOUNT="-v /mnt/dcgpuval/models:/models"
elif [[ -d "/shared_nfs/huggingface_models/amd" ]]; then
  MODEL_CACHE_MOUNT="-v /shared_nfs/huggingface_models/amd:/models"
elif [[ -d "/shareddata/models" ]]; then
  MODEL_CACHE_MOUNT="-v /shareddata/models:/models"
elif [[ -d "/data/models" ]]; then
  MODEL_CACHE_MOUNT="-v /data/models:/models"
fi

if [[ -f "/etc/podinfo/gha-render-devices" ]]; then
  DEVICE_FLAG="$(cat /etc/podinfo/gha-render-devices)"
else
  DEVICE_FLAG="--device /dev/dri"
fi

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
docker run -dt --device=/dev/kfd ${DEVICE_FLAG} \
  -v "${REPO_ROOT}":/workspace \
  ${MODEL_CACHE_MOUNT} \
  -w /workspace \
  --ipc=host --network=host --group-add video \
  --shm-size=16G \
  --privileged \
  --cap-add=SYS_PTRACE \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name "${CONTAINER_NAME}" \
  "${SGLANG_IMAGE_TAG}"

GPU_PREFLIGHT_KILL_DOCKER=1 bash .github/scripts/gpu_preflight_check.sh "${CONTAINER_NAME}" docker

SGLANG_RESOLVED_MODEL_PATH="$(plugin_ci_download_model "${MODEL_PATH}")"
DRAFT_ID="$(plugin_ci_infer_draft_model_id "${DRAFT_MODEL_PATH}" "${EXTRA_ARGS}")"
if [[ -n "${DRAFT_ID}" ]]; then
  DRAFT_HF_ID="$(plugin_ci_hf_id "${DRAFT_ID}")"
  if [[ "${DRAFT_HF_ID}" != "$(plugin_ci_hf_id "${MODEL_PATH}")" ]]; then
    DRAFT_RESOLVED_PATH="$(plugin_ci_download_model "${DRAFT_HF_ID}")"
    EXTRA_ARGS="$(plugin_ci_rewrite_extra_args "${EXTRA_ARGS}" "${DRAFT_HF_ID}" "${DRAFT_RESOLVED_PATH}")"
  fi
fi

docker exec \
  -e SGLANG_MODEL_NAME="${MODEL_NAME}" \
  -e SGLANG_MODEL_PATH="${SGLANG_RESOLVED_MODEL_PATH}" \
  -e SGLANG_EXTRA_ARGS="${EXTRA_ARGS}" \
  -e SGLANG_ENV_VARS="${ENV_VARS}" \
  -e MAX_WAIT_RETRIES="120" \
  -e STREAM_SGLANG_LOGS="1" \
  -e LM_EVAL_TASK="gsm8k" \
  -e LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT}" \
  -e LM_EVAL_NUM_CONCURRENT="${LM_EVAL_NUM_CONCURRENT}" \
  -e LM_EVAL_USE_CHAT_COMPLETIONS="${LM_EVAL_USE_CHAT_COMPLETIONS}" \
  -e LM_EVAL_EXTRA_MODEL_ARGS="${LM_EVAL_EXTRA_MODEL_ARGS}" \
  "${CONTAINER_NAME}" bash -lc "
    set -euo pipefail
    bash .github/scripts/atom_sglang_test.sh accuracy
  "

collect_artifacts

result_file="$(ls -1t "${RESULT_DIR}/atom_sglang_accuracy_results"/*.json 2>/dev/null | head -n 1 || true)"
if [[ -z "${result_file}" || ! -f "${result_file}" ]]; then
  echo "ERROR: No results JSON file found in ${RESULT_DIR}/atom_sglang_accuracy_results/" >&2
  exit 2
fi

flexible_extract_value="$(python3 - "${result_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(data["results"]["gsm8k"]["exact_match,flexible-extract"])
PY
)"

echo "Flexible extract value: ${flexible_extract_value}"
echo "Accuracy test threshold: ${ACCURACY_TEST_THRESHOLD}"
result="$(awk -v val="${flexible_extract_value}" -v threshold="${ACCURACY_TEST_THRESHOLD}" 'BEGIN {print (val < threshold) ? 1 : 0}')"
if [[ "${result}" -eq 1 ]]; then
  echo "Accuracy test failed: ${flexible_extract_value} < ${ACCURACY_TEST_THRESHOLD}" >&2
  exit 1
fi

echo "Accuracy test passed for ${MODEL_NAME}"
