#!/usr/bin/env bash
# Shared Hugging Face cache helpers for plugin CI (vLLM / SGLang).
# Callers must set CONTAINER_NAME. MODEL_CACHE_MOUNT and HF_TOKEN are optional.

plugin_ci_hf_id() {
  local model_id="${1:-}"
  model_id="${model_id#/models/}"
  printf '%s' "${model_id}"
}

plugin_ci_download_model() {
  local model_id="${1:-}"
  local hf_id
  hf_id="$(plugin_ci_hf_id "${model_id}")"
  if [[ -z "${hf_id}" ]]; then
    echo "ERROR: plugin_ci_download_model requires a model id" >&2
    return 2
  fi
  if [[ -z "${MODEL_CACHE_MOUNT:-}" ]]; then
    printf '%s' "${hf_id}"
    return 0
  fi

  local model_dir="/models/${hf_id}"
  docker exec \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e MODEL_ID="${hf_id}" \
    -e TARGET_DIR="${model_dir}" \
    -e MODEL_USE_LOCK="true" \
    -e MODEL_DOWNLOAD_TIMEOUT="3h" \
    -e MODEL_LOCK_WAIT_SECONDS="7200" \
    -e MODEL_LOCK_POLL_INTERVAL="30" \
    -e MODEL_PROGRESS_INTERVAL="60" \
    "${CONTAINER_NAME}" \
    bash -lc 'bash /workspace/.github/scripts/download_model_with_lock.sh "$MODEL_ID" "$TARGET_DIR"' >&2
  printf '%s' "${model_dir}"
}

plugin_ci_infer_draft_model_id() {
  DRAFT_MODEL_PATH="${1:-}" EXTRA_ARGS="${2:-}" python3 - <<'PY'
import os
import re
import shlex

explicit = os.environ.get("DRAFT_MODEL_PATH", "").strip()
if explicit:
    print(explicit)
    raise SystemExit(0)

raw = os.environ.get("EXTRA_ARGS", "")
tokens = []
try:
    tokens = shlex.split(raw)
except ValueError:
    tokens = raw.split()

for flag in ("--speculative-draft-model-path", "--draft-model"):
    if flag in tokens:
        index = tokens.index(flag)
        if index + 1 < len(tokens):
            print(tokens[index + 1])
            raise SystemExit(0)

match = re.search(r'"model"\s*:\s*"([^"]+)"', raw)
if match:
    print(match.group(1))
PY
}

plugin_ci_rewrite_extra_args() {
  EXTRA_ARGS="${1:-}" DRAFT_ID="${2:-}" RESOLVED_PATH="${3:-}" python3 - <<'PY'
import os

args = os.environ.get("EXTRA_ARGS", "")
draft_id = os.environ.get("DRAFT_ID", "").strip()
resolved = os.environ.get("RESOLVED_PATH", "").strip()
if not args or not draft_id or not resolved:
    print(args, end="")
    raise SystemExit(0)

hf_id = draft_id[len("/models/") :] if draft_id.startswith("/models/") else draft_id
variants = []
for value in (draft_id, hf_id, f"/models/{hf_id}"):
    if value and value not in variants:
        variants.append(value)

for value in sorted(variants, key=len, reverse=True):
    args = args.replace(value, "@@PLUGIN_CI_DRAFT@@")
args = args.replace("@@PLUGIN_CI_DRAFT@@", resolved)
print(args, end="")
PY
}
