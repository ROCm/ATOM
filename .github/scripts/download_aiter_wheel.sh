#!/usr/bin/env bash
# Resolve and download the aiter wheel: latest-main S3 manifest first, then
# fall back to the newest matching aiter-whl-* artifact from ROCm/aiter.
# De-inlined from atom-test.yaml / atomesh-accuracy-validation.yaml (identical
# blocks). Inputs via env: ATOM_PYTHON_TAG (required), GITHUB_TOKEN (required);
# S3_MAIN_MANIFEST_URL / API_URL / AITER_TEST_WORKFLOW_ID are overridable.
# Output: aiter-whl/amd_aiter*.whl in the current directory.
set -euo pipefail
: "${ATOM_PYTHON_TAG:?ATOM_PYTHON_TAG must be set}"
: "${GITHUB_TOKEN:?GITHUB_TOKEN must be set}"
echo "=== Trying latest main aiter wheel manifest from S3 first ==="

S3_MAIN_MANIFEST_URL="${S3_MAIN_MANIFEST_URL:-https://rocm.frameworks-nightlies.amd.com/whl-staging/gfx942-gfx950/main/latest.json}"
API_URL="${API_URL:-https://api.github.com}"
AUTH_HEADER="Authorization: token ${GITHUB_TOKEN}"
AITER_TEST_WORKFLOW_ID="${AITER_TEST_WORKFLOW_ID:-179476100}"
AITER_WHEEL_DOWNLOAD_MAX_ATTEMPTS="${AITER_WHEEL_DOWNLOAD_MAX_ATTEMPTS:-3}"
AITER_WHEEL_RETRY_DELAY_SECONDS="${AITER_WHEEL_RETRY_DELAY_SECONDS:-30}"
AITER_WHEEL_CURL_CONNECT_TIMEOUT_SECONDS="${AITER_WHEEL_CURL_CONNECT_TIMEOUT_SECONDS:-30}"
AITER_WHEEL_CURL_MAX_TIME_SECONDS="${AITER_WHEEL_CURL_MAX_TIME_SECONDS:-540}"

ARTIFACT_ID=""
ARTIFACT_NAME=""
ARTIFACT_RUN_ID=""
ARTIFACT_RUN_SHA=""
ARTIFACT_RUN_CREATED_AT=""

retry_cmd() {
  local max_attempts="$1"
  shift
  local attempt=1
  local rc=0

  while true; do
    if "$@"; then
      return 0
    fi
    rc=$?
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Command failed after ${attempt} attempts" >&2
      return "$rc"
    fi
    local sleep_seconds=$((attempt * AITER_WHEEL_RETRY_DELAY_SECONDS))
    echo "Attempt ${attempt}/${max_attempts} failed; retrying in ${sleep_seconds}s" >&2
    sleep "$sleep_seconds"
    attempt=$((attempt + 1))
  done
}

curl_with_retry() {
  retry_cmd "$AITER_WHEEL_DOWNLOAD_MAX_ATTEMPTS" \
    curl --fail --silent --show-error --location \
      --connect-timeout "$AITER_WHEEL_CURL_CONNECT_TIMEOUT_SECONDS" \
      --max-time "$AITER_WHEEL_CURL_MAX_TIME_SECONDS" \
      "$@"
}

resolve_download_url() {
  # The python body must be column-0: indenting continuation lines to match the
  # bash block puts leading whitespace inside the single-quoted source and makes
  # python raise "IndentationError: unexpected indent". The leading newline
  # keeps the first line blank (valid) so every statement starts at column 0.
  python3 -c '
import sys
from urllib.parse import quote, unquote, urlsplit, urlunsplit
parts = urlsplit(sys.argv[1])
encoded_path = "/".join(quote(unquote(segment), safe="") for segment in parts.path.split("/"))
print(urlunsplit((parts.scheme, parts.netloc, encoded_path, parts.query, parts.fragment)))
' "$1"
}

find_latest_artifact() {
  local runs_json artifact_json run_id python_artifact_suffix

  if [ -n "$ARTIFACT_ID" ] && [ "$ARTIFACT_ID" != "null" ]; then
    return 0
  fi

  python_artifact_suffix="py${ATOM_PYTHON_TAG#cp}"
  python_artifact_suffix="${python_artifact_suffix:0:3}.${python_artifact_suffix:3}"

  echo "=== Finding latest aiter-whl-* artifact for ${python_artifact_suffix} from ROCm/aiter ==="
  runs_json=$(curl_with_retry -H "$AUTH_HEADER" \
    "$API_URL/repos/ROCm/aiter/actions/workflows/$AITER_TEST_WORKFLOW_ID/runs?per_page=100&branch=main&event=push")

  for run_id in $(echo "$runs_json" | jq -r '.workflow_runs[].id'); do
    artifact_json=$(curl_with_retry -H "$AUTH_HEADER" \
      "$API_URL/repos/ROCm/aiter/actions/runs/$run_id/artifacts" \
      | jq --arg artifact_suffix "-${python_artifact_suffix}" '[.artifacts[] | select(.name | startswith("aiter-whl-") and endswith($artifact_suffix)) | select(.expired == false)] | sort_by(.created_at) | last')

    if [ "$artifact_json" != "null" ] && [ -n "$artifact_json" ]; then
      ARTIFACT_ID=$(echo "$artifact_json" | jq -r '.id')
      ARTIFACT_NAME=$(echo "$artifact_json" | jq -r '.name')
      ARTIFACT_RUN_ID="$run_id"
      ARTIFACT_RUN_SHA=$(echo "$runs_json" | jq -r --arg run_id "$run_id" '.workflow_runs[] | select((.id | tostring) == $run_id) | .head_sha')
      ARTIFACT_RUN_CREATED_AT=$(echo "$runs_json" | jq -r --arg run_id "$run_id" '.workflow_runs[] | select((.id | tostring) == $run_id) | .created_at')
      echo "Found artifact in run $ARTIFACT_RUN_ID: $ARTIFACT_NAME (ID: $ARTIFACT_ID, SHA: $ARTIFACT_RUN_SHA)"
      return 0
    fi
  done

  return 1
}

download_from_s3_manifest() {
  local manifest_file manifest_fetch_url manifest_branch manifest_timestamp manifest_commit wheel_name wheel_url resolved_wheel_url

  mkdir -p aiter-whl
  rm -f aiter-whl/amd_aiter*.whl

  manifest_file=$(mktemp)
  trap 'rm -f "$manifest_file"' RETURN
  manifest_fetch_url="${S3_MAIN_MANIFEST_URL}?ts=$(date +%s)"
  curl_with_retry -H "Cache-Control: no-cache" "$manifest_fetch_url" -o "$manifest_file" || return 1

  manifest_branch=$(jq -r '.branch // empty' "$manifest_file")
  manifest_timestamp=$(jq -r '.timestamp // empty' "$manifest_file")
  manifest_commit=$(jq -r '.commit // empty' "$manifest_file")

  wheel_name=$(jq -r ".wheels.${ATOM_PYTHON_TAG}.wheel_name // empty" "$manifest_file")
  wheel_url=$(jq -r ".wheels.${ATOM_PYTHON_TAG}.wheel_url // empty" "$manifest_file")
  if [ -n "$wheel_name" ] && [ -n "$wheel_url" ]; then
    echo "Selected ${ATOM_PYTHON_TAG} wheel from versioned manifest"
  else
    wheel_name=$(jq -r '.wheel_name // empty' "$manifest_file")
    wheel_url=$(jq -r '.wheel_url // empty' "$manifest_file")
    echo "Versioned manifest not available, using top-level wheel fields"
  fi

  if [ "$manifest_branch" != "main" ] || [ -z "$manifest_timestamp" ] || [ -z "$manifest_commit" ] || [ -z "$wheel_name" ] || [ -z "$wheel_url" ]; then
    echo "Invalid latest main wheel manifest"
    return 1
  fi

  if [[ "$wheel_name" == *cp* ]] && [[ "$wheel_name" != *${ATOM_PYTHON_TAG}* ]]; then
    echo "WARNING: wheel $wheel_name does not match target Python ${ATOM_PYTHON_TAG}"
    return 1
  fi

  if find_latest_artifact; then
    if [ -n "$ARTIFACT_RUN_SHA" ] && [ "$manifest_commit" != "$ARTIFACT_RUN_SHA" ]; then
      if [ -n "$ARTIFACT_RUN_CREATED_AT" ] && [[ "$manifest_timestamp" < "$ARTIFACT_RUN_CREATED_AT" ]]; then
        echo "Manifest commit $manifest_commit is older than latest artifact run $ARTIFACT_RUN_ID ($ARTIFACT_RUN_SHA); treating manifest as stale"
        return 1
      fi
      echo "Manifest commit $manifest_commit differs from latest artifact run $ARTIFACT_RUN_ID ($ARTIFACT_RUN_SHA), but manifest timestamp is not older"
    fi
  else
    echo "No GitHub fallback artifact found while checking manifest freshness"
  fi

  resolved_wheel_url=$(resolve_download_url "$wheel_url")

  echo "Selected latest main wheel manifest: $S3_MAIN_MANIFEST_URL"
  echo "Manifest timestamp: $manifest_timestamp"
  echo "Manifest commit: $manifest_commit"
  echo "Manifest wheel: $wheel_name"
  echo "Downloading manifest-selected wheel: $resolved_wheel_url"
  curl_with_retry "$resolved_wheel_url" -o "aiter-whl/$wheel_name" || return 1
  echo "Downloaded wheel from manifest: aiter-whl/$wheel_name"

  rm -f "$manifest_file"
  trap - RETURN
}

download_from_artifact() {
  local fallback_wheel fallback_wheel_name

  echo "=== Falling back to latest ${ATOM_PYTHON_TAG} aiter-whl-* artifact from ROCm/aiter ==="
  find_latest_artifact || {
    echo "ERROR: No ${ATOM_PYTHON_TAG} aiter-whl-* artifact found in recent Aiter Test runs"
    return 1
  }

  mkdir -p aiter-whl
  rm -f aiter-whl/amd_aiter*.whl
  curl_with_retry -H "$AUTH_HEADER" \
    "$API_URL/repos/ROCm/aiter/actions/artifacts/$ARTIFACT_ID/zip" \
    -o aiter-whl.zip
  unzip -o aiter-whl.zip -d aiter-whl
  rm -f aiter-whl.zip

  fallback_wheel=$(ls -t aiter-whl/amd_aiter*.whl 2>/dev/null | head -1)
  fallback_wheel_name=$(basename "${fallback_wheel:-}")
  if [ -z "$fallback_wheel" ] || [[ "$fallback_wheel_name" != *${ATOM_PYTHON_TAG}* ]]; then
    echo "ERROR: artifact fallback did not produce a ${ATOM_PYTHON_TAG} wheel"
    ls -la aiter-whl/ || true
    return 1
  fi
  echo "Downloaded artifact-selected wheel: $fallback_wheel"
}

if download_from_s3_manifest; then
  echo "Using wheel from S3 main manifest"
else
  echo "Main wheel manifest download failed, falling back to GitHub artifact"
  download_from_artifact
fi

AITER_WHL=$(ls -t aiter-whl/amd_aiter*.whl 2>/dev/null | head -1)
if [ -z "$AITER_WHL" ]; then
  echo "ERROR: No amd_aiter wheel available after S3/artifact attempts"
  ls -la aiter-whl/ || true
  exit 1
fi
if [[ "$(basename "$AITER_WHL")" != *${ATOM_PYTHON_TAG}* ]]; then
  echo "ERROR: selected wheel $AITER_WHL does not match target Python ${ATOM_PYTHON_TAG}"
  exit 1
fi

echo "Selected wheel: $AITER_WHL"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "aiter_artifact_id=${ARTIFACT_ID}" >> "$GITHUB_OUTPUT"
  echo "aiter_wheel_name=$(basename "$AITER_WHL")" >> "$GITHUB_OUTPUT"
fi
