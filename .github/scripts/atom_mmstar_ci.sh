#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen/Qwen3.5-35B-A3B-FP8}"
ATOM_PORT="${ATOM_PORT:-8000}"
RESULT_DIR="${RESULT_DIR:-/tmp/atom_mmstar_results}"
LIMIT="${LIMIT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
NUM_CONCURRENT="${NUM_CONCURRENT:-64}"
TIMEOUT="${TIMEOUT:-900}"
MMSTAR_THRESHOLD="${MMSTAR_THRESHOLD:-0.72}"
SERVER_READY_RETRIES="${SERVER_READY_RETRIES:-90}"
SERVER_READY_INTERVAL="${SERVER_READY_INTERVAL:-30}"
SERVER_LOG="${SERVER_LOG:-/tmp/atom_mmstar_server.log}"
CLIENT_LOG="${CLIENT_LOG:-/tmp/atom_mmstar_client.log}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "=== MMStar CI configuration ==="
echo "MODEL_PATH=${MODEL_PATH}"
echo "ATOM_PORT=${ATOM_PORT}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "LIMIT=${LIMIT}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "NUM_CONCURRENT=${NUM_CONCURRENT}"
echo "MMSTAR_THRESHOLD=${MMSTAR_THRESHOLD}"

mkdir -p "${RESULT_DIR}"
rm -f "${SERVER_LOG}" "${CLIENT_LOG}"

echo "=== Installed package versions ==="
python3 - <<'PY'
import importlib.metadata as md

for name in ("torch", "torchvision", "torchaudio", "transformers", "accelerate", "vllm", "lmms-eval", "atom"):
    try:
        version = md.version(name)
    except md.PackageNotFoundError:
        version = "<not-installed>"
    print(f"{name}=={version}")
PY

echo "=== Launching ATOM OpenAI server ==="
read -r -a extra_server_args <<< "${EXTRA_SERVER_ARGS}"
PYTHONUNBUFFERED=1 python3 -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --server-port "${ATOM_PORT}" \
  --no-enable_prefix_caching \
  "${extra_server_args[@]}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for ((i=1; i<=SERVER_READY_RETRIES; i++)); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ATOM server exited before becoming ready." >&2
    tail -200 "${SERVER_LOG}" || true
    exit 1
  fi

  if curl -fsS "http://127.0.0.1:${ATOM_PORT}/health" >/dev/null 2>&1; then
    echo "ATOM server health endpoint is ready."
    break
  fi

  if (( i == SERVER_READY_RETRIES )); then
    echo "ATOM server did not become ready in time." >&2
    tail -200 "${SERVER_LOG}" || true
    exit 1
  fi

  echo "Waiting for ATOM server... (${i}/${SERVER_READY_RETRIES})"
  sleep "${SERVER_READY_INTERVAL}"
done

echo "=== Running lmms-eval MMStar ==="
lmms_eval_args=(
  --model openai \
  --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:${ATOM_PORT}/v1,api_key=EMPTY,timeout=${TIMEOUT},max_retries=3,num_concurrent=${NUM_CONCURRENT},max_size_in_mb=50" \
  --tasks mmstar \
  --batch_size 1 \
  --process_with_media \
  --gen_kwargs "temperature=0,max_new_tokens=${MAX_NEW_TOKENS}" \
  --log_samples \
  --log_samples_suffix atom_mmstar_ci \
  --output_path "${RESULT_DIR}"
)
if [[ -n "${LIMIT}" ]]; then
  lmms_eval_args+=(--limit "${LIMIT}")
fi

python3 -m lmms_eval "${lmms_eval_args[@]}" 2>&1 | tee "${CLIENT_LOG}"

echo "=== Checking MMStar score ==="
RESULT_DIR="${RESULT_DIR}" MMSTAR_THRESHOLD="${MMSTAR_THRESHOLD}" python3 - <<'PY'
import json
import os
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR"])
threshold = float(os.environ["MMSTAR_THRESHOLD"])
json_files = sorted(result_dir.rglob("*.json"), key=lambda path: path.stat().st_mtime_ns)
if not json_files:
    raise SystemExit(f"No lmms-eval result JSON found under {result_dir}")

result_file = json_files[-1]
with result_file.open(encoding="utf-8") as handle:
    data = json.load(handle)

metrics = data["results"]["mmstar"]
if "average" in metrics:
    score = float(metrics["average"])
elif "average,none" in metrics:
    score = float(metrics["average,none"])
else:
    raise SystemExit(f"Could not find MMStar average in {result_file}; keys={sorted(metrics)}")

print(f"Result file: {result_file}")
print(f"MMStar average: {score:.4f}")
print(f"Threshold: {threshold:.4f}")
if score < threshold:
    raise SystemExit(f"MMStar score {score:.4f} is below threshold {threshold:.4f}")
PY

echo "MMStar CI check passed."
