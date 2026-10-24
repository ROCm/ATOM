#!/usr/bin/env bash

# Decide whether an expensive PR CI workflow should run.
# Non-PR events keep their existing behavior. For PR events, run when the PR
# currently has one of CI_GATE_LABELS, or GitHub reports that the PR's current
# aggregate review decision is APPROVED. Dismissed or superseded reviews must
# not authorize unrelated heavy CI workflows.

set -euo pipefail

OUTPUT_FILE="${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"
SUMMARY_FILE="${GITHUB_STEP_SUMMARY:-}"
EVENT_NAME="${GITHUB_EVENT_NAME:-}"
EVENT_PATH="${GITHUB_EVENT_PATH:-}"
REPO="${GITHUB_REPOSITORY:-}"
CI_GATE_LABELS="${CI_GATE_LABELS:-ci:full}"
CURRENT_PR=""

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

find_allowed_pr_label() {
  local labels label
  if [ -n "${CURRENT_PR:-}" ]; then
    labels="$(jq -r '.labels[]?.name // empty' <<< "${CURRENT_PR}")"
  else
    labels="$(jq -r '.pull_request.labels[]?.name // empty' "${EVENT_PATH}")"
  fi

  if [ -z "${labels}" ] && [ -n "${REPO:-}" ] && [ -n "${PR_NUMBER:-}" ]; then
    labels="$(gh api --paginate "repos/${REPO}/issues/${PR_NUMBER}/labels" --jq '.[].name' 2>/dev/null || true)"
  fi

  while IFS= read -r label; do
    if [ -n "${label}" ] && label_is_allowed "${label}"; then
      printf '%s\n' "${label}"
      return 0
    fi
  done <<< "${labels}"

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

allow_heavy_ci() {
  # Reviews release a gate; they are not requests to retry existing work.
  # Workflow concurrency serializes these events with the original PR run.
  # Explicit Actions reruns, labels and revision events retain their behavior.
  if [ "${EVENT_NAME}" != "pull_request_review" ] \
    || [ "${GITHUB_RUN_ATTEMPT:-1}" -gt 1 ]; then
    emit_decision "true" "$@"
    return
  fi

  local head_sha head_ref head_repo runs run_ids run_id jobs started
  head_sha="$(jq -r '.pull_request.head.sha // empty' "${EVENT_PATH}")"
  head_ref="$(jq -r '.pull_request.head.ref // empty' "${EVENT_PATH}")"
  head_repo="$(jq -r '.pull_request.head.repo.full_name // empty' "${EVENT_PATH}")"
  if [ -z "${CI_GATE_WORKFLOW:-}" ] || [ -z "${GITHUB_RUN_ID:-}" ] || [ -z "${head_sha}" ]; then
    emit_decision "false" "missing-run-context"
    return
  fi

  if ! runs="$(gh api --paginate --slurp \
    "repos/${REPO}/actions/workflows/${CI_GATE_WORKFLOW}/runs?head_sha=${head_sha}&per_page=100")"; then
    emit_decision "false" "runs-query-failed"
    return
  fi
  run_ids="$(jq -r --arg sha "${head_sha}" --arg branch "${head_ref}" --arg repo "${head_repo}" \
    --argjson pr "${PR_NUMBER}" --argjson current "${GITHUB_RUN_ID}" '
    .[].workflow_runs[]
    | select(.id < $current and .head_sha == $sha and .head_branch == $branch)
    | select(.event == "pull_request" or .event == "pull_request_review")
    | select(any(.pull_requests[]?; .number == $pr) or
        ((.pull_requests | length) == 0 and $repo != "" and .head_repository.full_name == $repo))
    | .id' <<< "${runs}")"

  while IFS= read -r run_id; do
    [ -n "${run_id}" ] || continue
    if ! jobs="$(gh api --paginate --slurp \
      "repos/${REPO}/actions/runs/${run_id}/jobs?filter=all&per_page=100")"; then
      emit_decision "false" "jobs-query-failed"
      return
    fi
    # A successful workflow can contain only skipped GPU jobs. Check entry to
    # the gated chain instead, across all attempts, including failures. Leaving
    # the chain untouched also preserves successful cells of a partial failure.
    if ! started="$(jq -r 'any(.[].jobs[]; .name == "Check Pre Checkin Signal" and .conclusion != "skipped")' <<< "${jobs}")"; then
      emit_decision "false" "jobs-query-failed"
      return
    fi
    if [ "${started}" = "true" ]; then
      emit_decision "false" "heavy-ci-already-started"
      echo "existing_run_id=${run_id}" >> "${OUTPUT_FILE}"
      echo "Heavy CI already belongs to run ${run_id}; use an explicit rerun to retry it."
      if [ -n "${SUMMARY_FILE}" ]; then
        echo "- Existing heavy CI: https://github.com/${REPO}/actions/runs/${run_id}" >> "${SUMMARY_FILE}"
      fi
      return
    fi
  done <<< "${run_ids}"

  emit_decision "true" "$@"
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
BASE_REF="$(jq -r '.pull_request.base.ref // ""' "${EVENT_PATH}")"

if [ "${EVENT_NAME}" = "pull_request_review" ]; then
  REVIEW_STATE="$(jq -r '.review.state // ""' "${EVENT_PATH}")"
  if [ "${ACTION}" != "submitted" ] || [ "${REVIEW_STATE}" != "approved" ]; then
    emit_decision "false" "review-not-approved-event"
    exit 0
  fi
fi

# An approval can wait in the concurrency queue while the PR changes.
# Never authorize obsolete code or labels from the original review payload.
if [ "${EVENT_NAME}" = "pull_request_review" ] \
  && [ -n "${PR_NUMBER}" ] && [ -n "${REPO}" ]; then
  if ! CURRENT_PR="$(gh api "repos/${REPO}/pulls/${PR_NUMBER}")"; then
    emit_decision "false" "pr-query-failed"
    exit 0
  fi
  if [ "$(jq -r '.state' <<< "${CURRENT_PR}")" != "open" ]; then
    emit_decision "false" "pull-request-closed"
    exit 0
  fi
  EXPECTED_HEAD_SHA="$(jq -r '.pull_request.head.sha // empty' "${EVENT_PATH}")"
  if [ -z "${EXPECTED_HEAD_SHA}" ] || [ "$(jq -r '.head.sha // empty' <<< "${CURRENT_PR}")" != "${EXPECTED_HEAD_SHA}" ]; then
    emit_decision "false" "stale-head-sha"
    exit 0
  fi
  BASE_REF="$(jq -r '.base.ref' <<< "${CURRENT_PR}")"
  IS_DRAFT="$(jq -r '.draft // false' <<< "${CURRENT_PR}")"
fi

if [ "${BASE_REF}" != "main" ]; then
  emit_decision "false" "non-main-base-branch"
  exit 0
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

MATCHED_LABEL="$(find_allowed_pr_label || true)"
if [ -n "${MATCHED_LABEL}" ]; then
  allow_heavy_ci "label-present" "${MATCHED_LABEL}"
  exit 0
fi

if [ "${IS_DRAFT}" = "true" ]; then
  emit_decision "false" "draft-pull-request"
  exit 0
fi

check_relevant_paths

if ! REVIEW_DECISION="$(
  gh api graphql \
    -f owner="${OWNER}" \
    -f name="${NAME}" \
    -F number="${PR_NUMBER}" \
    -f query='query($owner: String!, $name: String!, $number: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $number) {
          reviewDecision
        }
      }
    }' \
    --jq '.data.repository.pullRequest.reviewDecision // ""'
)"; then
  echo "Failed to query the current PR review decision; skipping heavy CI."
  emit_decision "false" "review-query-failed"
  exit 0
fi

case "${REVIEW_DECISION}" in
  APPROVED)
    allow_heavy_ci "current-approval" "" "${REVIEW_DECISION}" "1" "0"
    ;;
  CHANGES_REQUESTED)
    emit_decision "false" "changes-requested" "" "${REVIEW_DECISION}" "0" "1"
    ;;
  *)
    emit_decision "false" "not-approved" "" "${REVIEW_DECISION}" "0" "0"
    ;;
esac
