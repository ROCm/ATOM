#!/usr/bin/env bash

# Decide whether an expensive PR CI workflow should run.
# Non-PR events keep their existing behavior. For PR events, run only when the
# current event explicitly adds one of CI_GATE_LABELS, or the PR has any
# historical APPROVED review.

set -euo pipefail

OUTPUT_FILE="${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"
SUMMARY_FILE="${GITHUB_STEP_SUMMARY:-}"
EVENT_NAME="${GITHUB_EVENT_NAME:-}"
EVENT_PATH="${GITHUB_EVENT_PATH:-}"
REPO="${GITHUB_REPOSITORY:-}"
CI_GATE_LABELS="${CI_GATE_LABELS:-ci:full}"

trim() {
  sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

label_is_allowed() {
  local candidate="$1"
  local raw_label allowed_label
  local -a raw_labels

  IFS=',' read -ra raw_labels <<< "${CI_GATE_LABELS}"
  for raw_label in "${raw_labels[@]}"; do
    allowed_label="$(printf '%s' "${raw_label}" | trim)"
    if [ -n "${allowed_label}" ] && [ "${candidate}" = "${allowed_label}" ]; then
      return 0
    fi
  done

  return 1
}

check_relevant_paths() {
  if [ -z "${CI_GATE_PATHS_IGNORE:-}" ]; then
    return 0
  fi

  local changed_files filter_result
  if ! changed_files="$(gh api --paginate "repos/${REPO}/pulls/${PR_NUMBER}/files" --jq '.[].filename')"; then
    echo "Failed to query PR changed files; skipping heavy CI."
    emit_decision "false" "files-query-failed"
    exit 0
  fi

  if ! filter_result="$(
    CHANGED_FILES="${changed_files}" python3 - <<'PY'
import fnmatch
import os
import sys

files = [line.strip() for line in os.environ["CHANGED_FILES"].splitlines() if line.strip()]
patterns = [
    line.strip()
    for line in os.environ.get("CI_GATE_PATHS_IGNORE", "").splitlines()
    if line.strip()
]

def matches(pattern, path):
    if pattern == "**/*.md":
        return path.endswith(".md")
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return path == prefix or path.startswith(prefix + "/")
    return fnmatch.fnmatchcase(path, pattern)

relevant = [
    path
    for path in files
    if not any(matches(pattern, path) for pattern in patterns)
]

print(f"Changed files: {len(files)}; relevant files for this workflow: {len(relevant)}")
for path in relevant[:20]:
    print(f"  relevant: {path}")
if len(relevant) > 20:
    print(f"  ... {len(relevant) - 20} more relevant file(s)")

sys.exit(0 if relevant else 1)
PY
  )"; then
    echo "${filter_result}"
    emit_decision "false" "ignored-paths-only"
    exit 0
  fi

  echo "${filter_result}"
}

emit_decision() {
  local should_run="$1"
  local reason="$2"
  local matched_label="${3:-}"
  local review_decision="${4:-}"
  local approval_count="${5:-0}"
  local changes_requested_count="${6:-0}"

  {
    echo "should_run=${should_run}"
    echo "reason=${reason}"
    echo "matched_label=${matched_label}"
    echo "review_decision=${review_decision}"
    echo "approval_count=${approval_count}"
    echo "changes_requested_count=${changes_requested_count}"
  } >> "${OUTPUT_FILE}"

  echo "Heavy CI gate: should_run=${should_run} reason=${reason}"
  if [ -n "${matched_label}" ]; then
    echo "Matched label: ${matched_label}"
  fi
  if [ -n "${review_decision}" ]; then
    echo "Review decision: ${review_decision}"
    echo "Latest approvals: ${approval_count}; latest changes requested: ${changes_requested_count}"
  fi

  if [ -n "${SUMMARY_FILE}" ]; then
    {
      echo "### Heavy CI gate"
      echo "- Decision: \`${should_run}\`"
      echo "- Reason: \`${reason}\`"
      if [ -n "${matched_label}" ]; then
        echo "- Matched label: \`${matched_label}\`"
      fi
      if [ -n "${review_decision}" ]; then
        echo "- Review decision: \`${review_decision}\`"
        echo "- Latest approvals: \`${approval_count}\`"
        echo "- Latest changes requested: \`${changes_requested_count}\`"
      fi
    } >> "${SUMMARY_FILE}"
  fi
}

case "${EVENT_NAME}" in
  pull_request|pull_request_target|pull_request_review) ;;
  *)
    emit_decision "true" "non-pull-request-event"
    exit 0
    ;;
esac

if [ -z "${EVENT_PATH}" ] || [ ! -f "${EVENT_PATH}" ]; then
  emit_decision "false" "missing-event-payload"
  exit 0
fi

ACTION="$(jq -r '.action // ""' "${EVENT_PATH}")"
PR_NUMBER="$(jq -r '.pull_request.number // empty' "${EVENT_PATH}")"
IS_DRAFT="$(jq -r '.pull_request.draft // false' "${EVENT_PATH}")"
EVENT_LABEL="$(jq -r '.label.name // empty' "${EVENT_PATH}")"
BASE_REF="$(jq -r '.pull_request.base.ref // ""' "${EVENT_PATH}")"

if [ "${BASE_REF}" != "main" ]; then
  emit_decision "false" "non-main-base-branch"
  exit 0
fi

if [ "${EVENT_NAME}" = "pull_request_review" ]; then
  REVIEW_STATE="$(jq -r '.review.state // ""' "${EVENT_PATH}")"
  if [ "${ACTION}" != "submitted" ] || [ "${REVIEW_STATE}" != "approved" ]; then
    emit_decision "false" "review-not-approved-event"
    exit 0
  fi
fi

if [ "${ACTION}" = "closed" ]; then
  emit_decision "false" "pull-request-closed"
  exit 0
fi

if [ -z "${PR_NUMBER}" ] || [ -z "${REPO}" ]; then
  emit_decision "false" "missing-pr-context"
  exit 0
fi

OWNER="${REPO%%/*}"
NAME="${REPO#*/}"

if [ "${ACTION}" = "labeled" ] && label_is_allowed "${EVENT_LABEL}"; then
  emit_decision "true" "manual-label" "${EVENT_LABEL}"
  exit 0
fi

if [ "${IS_DRAFT}" = "true" ]; then
  emit_decision "false" "draft-pull-request"
  exit 0
fi

check_relevant_paths

if ! REVIEW_STATES="$(gh api --paginate "repos/${REPO}/pulls/${PR_NUMBER}/reviews" --jq '.[].state')"; then
  echo "Failed to query PR review state; skipping heavy CI."
  emit_decision "false" "review-query-failed"
  exit 0
fi

APPROVAL_COUNT="$(printf '%s\n' "${REVIEW_STATES}" | awk '$0 == "APPROVED" || $0 == "DISMISSED" { count++ } END { print count + 0 }')"
CHANGES_REQUESTED_COUNT="$(printf '%s\n' "${REVIEW_STATES}" | awk '$0 == "CHANGES_REQUESTED" { count++ } END { print count + 0 }')"

if [ "${APPROVAL_COUNT}" -gt 0 ]; then
  emit_decision "true" "approved-history" "" "APPROVED_HISTORY" "${APPROVAL_COUNT}" "${CHANGES_REQUESTED_COUNT}"
else
  emit_decision "false" "not-approved" "" "" "${APPROVAL_COUNT}" "${CHANGES_REQUESTED_COUNT}"
fi
