#!/usr/bin/env bash

# Decide whether an expensive PR CI workflow should run.
# Non-PR events keep their existing behavior. For PR events, run when the PR
# currently has one of CI_GATE_LABELS, or GitHub reports that the PR's current
# aggregate review decision is APPROVED. Dismissed or superseded reviews must
# not authorize unrelated heavy CI workflows.
# A label event is not a code change; it can only open the gate. It is judged
# like a code change only when, in its own payload, the label it adds is the
# PR's gate-opening label (find_allowed_pr_label); any other label event is a
# no-op that must not replace the PR's checks (check_label_event).

set -euo pipefail

OUTPUT_FILE="${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"
SUMMARY_FILE="${GITHUB_STEP_SUMMARY:-}"
EVENT_NAME="${GITHUB_EVENT_NAME:-}"
EVENT_PATH="${GITHUB_EVENT_PATH:-}"
ACTION=""
REPO="${GITHUB_REPOSITORY:-}"
CI_GATE_LABELS="${CI_GATE_LABELS:-ci:full}"
CURRENT_PR=""
PR_LABELS=""
PREVIOUS_RUNS=""
PREVIOUS_RUNS_ERROR=""
PR_CHECK_TAKEOVER="${PR_CHECK_TAKEOVER:-false}"

trim() {
  sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# CI_GATE_LABELS in gate-opening priority: their listed order, but with the
# catch-all ci:full last. Each workflow's concurrency group and gate check
# name spell out the same order.
GATE_LABELS=()
load_gate_labels() {
  local raw_label label has_full="false"
  local -a raw_labels

  IFS=',' read -ra raw_labels <<< "${CI_GATE_LABELS}"
  for raw_label in "${raw_labels[@]}"; do
    label="$(printf '%s' "${raw_label}" | trim)"
    if [ "${label}" = "ci:full" ]; then
      has_full="true"
    elif [ -n "${label}" ]; then
      GATE_LABELS+=("${label}")
    fi
  done
  if [ "${has_full}" = "true" ]; then
    GATE_LABELS+=("ci:full")
  fi
}
load_gate_labels

label_is_allowed() {
  local allowed_label
  for allowed_label in "${GATE_LABELS[@]}"; do
    if [ "$1" = "${allowed_label}" ]; then
      return 0
    fi
  done
  return 1
}

# The payload's PR labels. The label a labeled event added counts as on the PR
# even when the payload's list lacks it, as the workflows' concurrency and gate
# name expressions assume.
payload_labels() {
  jq -r '.pull_request.labels[]?.name // empty, (select(.action == "labeled") | .label.name // empty)' "${EVENT_PATH}"
}

load_pr_labels() {
  if [ -n "${CURRENT_PR:-}" ]; then
    PR_LABELS="$(jq -r '.labels[]?.name // empty' <<< "${CURRENT_PR}")"
    return 0
  fi

  # A rerun replays the original event payload, whose labels may be stale.
  if [ "${GITHUB_RUN_ATTEMPT:-1}" -le 1 ]; then
    PR_LABELS="$(payload_labels)"
    [ -z "${PR_LABELS}" ] || return 0
  fi

  if ! PR_LABELS="$(gh api --paginate "repos/${REPO}/issues/${PR_NUMBER}/labels" --jq '.[].name')"; then
    emit_undecidable "labels-query-failed"
    exit 0
  fi
}

# The gate-opening label among the newline-separated labels in $1: the first
# of GATE_LABELS that they include.
find_allowed_pr_label() {
  local candidate
  for candidate in "${GATE_LABELS[@]}"; do
    if grep -Fxq -- "${candidate}" <<< "$1"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

check_relevant_paths() {
  if [ -z "${CI_GATE_PATHS_IGNORE:-}" ] && [ -z "${CI_GATE_PATHS_INCLUDE:-}" ]; then
    return 0
  fi

  local changed_files filter_result
  if ! changed_files="$(gh api --paginate "repos/${REPO}/pulls/${PR_NUMBER}/files" --jq '.[].filename')"; then
    echo "Failed to query PR changed files; skipping heavy CI."
    emit_undecidable "files-query-failed"
    exit 0
  fi

  if ! filter_result="$(
    CHANGED_FILES="${changed_files}" python3 - <<'PY'
import fnmatch
import os
import sys

files = [line.strip() for line in os.environ["CHANGED_FILES"].splitlines() if line.strip()]
ignore_patterns = [
    line.strip()
    for line in os.environ.get("CI_GATE_PATHS_IGNORE", "").splitlines()
    if line.strip()
]
include_patterns = [
    line.strip()
    for line in os.environ.get("CI_GATE_PATHS_INCLUDE", "").splitlines()
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
    if (not include_patterns or any(matches(pattern, path) for pattern in include_patterns))
    and not any(matches(pattern, path) for pattern in ignore_patterns)
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
    local reason="ignored-paths-only"
    if [ -n "${CI_GATE_PATHS_INCLUDE:-}" ] && [ -z "${CI_GATE_PATHS_IGNORE:-}" ]; then
      reason="paths-not-included"
    fi
    emit_decision "false" "${reason}"
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
  local decidable="${7:-true}"
  local smoke_requested="false"

  if [ "${decidable}" != "true" ]; then
    smoke_requested="false"
  elif [ "${should_run}" = "true" ] || [ "${PR_CHECK_TAKEOVER}" = "true" ]; then
    smoke_requested="true"
  elif [ "${EVENT_NAME}" != "pull_request_review" ] \
    && [ "${ACTION}" != "labeled" ] \
    && [ "${reason}" != "ignored-paths-only" ] \
    && [ "${reason}" != "paths-not-included" ] \
    && [ "${reason}" != "draft-pull-request" ] \
    && [ "${reason}" != "pull-request-closed" ]; then
    smoke_requested="true"
  fi

  {
    echo "should_run=${should_run}"
    echo "reason=${reason}"
    echo "decidable=${decidable}"
    echo "smoke_requested=${smoke_requested}"
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
      echo "- Decidable: \`${decidable}\`"
      echo "- Smoke requested: \`${smoke_requested}\`"
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

emit_undecidable() {
  emit_decision "false" "$1" "" "" "0" "0" "false"
}

load_previous_runs() {
  if [ -n "${PREVIOUS_RUNS}" ]; then
    return 0
  fi

  local head_sha head_ref head_repo runs
  head_sha="$(jq -r '.pull_request.head.sha // empty' "${EVENT_PATH}")"
  head_ref="$(jq -r '.pull_request.head.ref // empty' "${EVENT_PATH}")"
  head_repo="$(jq -r '.pull_request.head.repo.full_name // empty' "${EVENT_PATH}")"
  if [ -z "${CI_GATE_WORKFLOW:-}" ] || [ -z "${GITHUB_RUN_ID:-}" ] || [ -z "${head_sha}" ]; then
    PREVIOUS_RUNS_ERROR="missing-run-context"
    return 1
  fi

  if ! runs="$(gh api --paginate --slurp \
    "repos/${REPO}/actions/workflows/${CI_GATE_WORKFLOW}/runs?head_sha=${head_sha}&per_page=100")"; then
    PREVIOUS_RUNS_ERROR="runs-query-failed"
    return 1
  fi
  if ! PREVIOUS_RUNS="$(jq -c --arg sha "${head_sha}" --arg branch "${head_ref}" --arg repo "${head_repo}" \
    --argjson pr "${PR_NUMBER}" --argjson current "${GITHUB_RUN_ID}" '
    [.[].workflow_runs[]
    | select(.id < $current and .head_sha == $sha and .head_branch == $branch)
    | select(.event == "pull_request" or .event == "pull_request_review")
    | select(any(.pull_requests[]?; .number == $pr) or
        ((.pull_requests | length) == 0 and $repo != "" and .head_repository.full_name == $repo))]
    | sort_by(.id)' <<< "${runs}")"; then
    PREVIOUS_RUNS=""
    PREVIOUS_RUNS_ERROR="runs-query-failed"
    return 1
  fi
}

query_run_jobs() {
  gh api --paginate --slurp "repos/${REPO}/actions/runs/$1/jobs?filter=all&per_page=100"
}

run_published_pr_check() {
  local jobs published
  if ! jobs="$(query_run_jobs "$1")" \
    || ! published="$(jq -r --arg name "${CI_GATE_PR_CHECK_JOB}" \
      'any(.[].jobs[]; .name == $name and .conclusion != "skipped")' <<< "${jobs}")"; then
    return 2
  fi
  [ "${published}" = "true" ]
}

detect_replaced_pr_run() {
  [ -n "${CI_GATE_PR_CHECK_JOB:-}" ] || return 0

  local pr_run_ids candidate noop run_id="" conclusion jobs job_count later_ids later_id status
  if ! load_previous_runs; then
    echo "Could not read earlier runs (${PREVIOUS_RUNS_ERROR}); assuming no PR run was replaced."
    return 0
  fi

  # The latest pull_request run of the PR's concurrency group. No-op label
  # runs are pull_request runs in groups of their own: they never replace or
  # get replaced, so look past them. Their first job carries CI_GATE_NOOP_JOB
  # as its name from the start, unlike the gate check, which is named last.
  pr_run_ids="$(jq -r '[.[] | select(.event == "pull_request") | .id] | reverse | .[]' <<< "${PREVIOUS_RUNS}")"
  while IFS= read -r candidate; do
    [ -n "${candidate}" ] || continue
    if ! jobs="$(query_run_jobs "${candidate}")" \
      || ! noop="$(jq -r --arg name "${CI_GATE_NOOP_JOB:-}" \
        '$name != "" and any(.[].jobs[]; .name == $name)' <<< "${jobs}")"; then
      echo "Could not read jobs of run ${candidate}; assuming no PR run was replaced."
      return 0
    fi
    [ "${noop}" = "true" ] && continue
    # A run with no jobs that GitHub has not cancelled is still queued. The PR
    # run an approval replaced is cancelled by then, so a queued run can only
    # be a no-op label run whose first job does not exist yet: look past it.
    conclusion="$(jq -r --argjson id "${candidate}" '.[] | select(.id == $id) | .conclusion // empty' <<< "${PREVIOUS_RUNS}")"
    if [ "$(jq -r '[.[].jobs[]] | length' <<< "${jobs}")" = "0" ] && [ "${conclusion}" != "cancelled" ]; then
      continue
    fi
    run_id="${candidate}"
    break
  done <<< "${pr_run_ids}"
  [ -n "${run_id}" ] || return 0
  conclusion="$(jq -r --argjson id "${run_id}" '.[] | select(.id == $id) | .conclusion // empty' <<< "${PREVIOUS_RUNS}")"
  [ "${conclusion}" = "cancelled" ] || return 0

  job_count="$(jq -r '[.[].jobs[]] | length' <<< "${jobs}")"
  # GitHub cancels a queued run without creating its jobs when a newer run
  # takes its place in the concurrency group, so a cancelled run with no jobs
  # never published its PR checks.
  [ "${job_count}" = "0" ] || return 0

  later_ids="$(jq -r --argjson id "${run_id}" \
    '.[] | select(.id > $id and .event == "pull_request_review") | .id' <<< "${PREVIOUS_RUNS}")"
  while IFS= read -r later_id; do
    [ -n "${later_id}" ] || continue
    status=0
    run_published_pr_check "${later_id}" || status=$?
    if [ "${status}" = "0" ]; then
      echo "Run ${later_id} already took over the checks of replaced PR run ${run_id}."
      return 0
    fi
    if [ "${status}" != "1" ]; then
      echo "Could not read jobs of run ${later_id}; assuming it took over replaced PR run ${run_id}."
      return 0
    fi
  done <<< "${later_ids}"

  PR_CHECK_TAKEOVER="true"
  echo "pr_check_takeover=true" >> "${OUTPUT_FILE}"
  echo "replaced_run_id=${run_id}" >> "${OUTPUT_FILE}"
  echo "Approval replaced pending PR run ${run_id} before it started; this run takes over its PR checks."
  if [ -n "${SUMMARY_FILE}" ]; then
    echo "- Replaced pending PR run: https://github.com/${REPO}/actions/runs/${run_id}" >> "${SUMMARY_FILE}"
  fi
}

allow_heavy_ci() {
  # Reviews release a gate; they are not requests to retry existing work.
  # Explicit Actions reruns and revision events retain their behavior; label
  # events get here only when they open the gate (see check_label_event).
  if [ "${EVENT_NAME}" != "pull_request_review" ] \
    || [ "${GITHUB_RUN_ATTEMPT:-1}" -gt 1 ]; then
    emit_decision "true" "$@"
    return
  fi

  local run_ids run_id jobs started
  if ! load_previous_runs; then
    emit_undecidable "${PREVIOUS_RUNS_ERROR}"
    return
  fi
  run_ids="$(jq -r '.[].id' <<< "${PREVIOUS_RUNS}")"

  while IFS= read -r run_id; do
    [ -n "${run_id}" ] || continue
    if ! jobs="$(query_run_jobs "${run_id}")"; then
      emit_undecidable "jobs-query-failed"
      return
    fi
    # A successful workflow can contain only skipped GPU jobs. Check whether
    # any attempt got past the Pre Checkin signal: GPU jobs start only after it
    # succeeds, so a failed or cancelled signal leaves the approval free to run
    # them. Leaving a started chain alone preserves successful cells of a
    # partial failure.
    if ! started="$(jq -r 'any(.[].jobs[]; .name == "Check Pre Checkin Signal" and (.conclusion == null or .conclusion == "success"))' <<< "${jobs}")"; then
      emit_undecidable "jobs-query-failed"
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

check_label_event() {
  # The event opens the gate only if, in its own payload, the label it adds is
  # the PR's gate-opening label. That holds for exactly one of two gate labels
  # added at once, whose payloads both carry both. The workflow's concurrency
  # group and gate check name test the same payload, so only that event joins
  # the PR's group, can cancel its running work and publishes the real gate
  # check; every other label event is a no-op in a group of its own. Adding
  # ci:full and later the workflow's own label opens the gate twice; that one
  # duplicate is accepted.
  #
  # A rerun replays the payload, and with it the run's group and gate check
  # name, so it is decided the same way: a no-op stays a no-op whatever the
  # live labels say (heavy CI started there would run beside the PR group's),
  # and a gate-opening run is re-judged like a code-change rerun on the PR's
  # current state. To retry heavy CI, rerun a run of the PR's group (the
  # code-change or gate-opening run).
  #
  # A current approval is deliberately not checked here. The concurrency group
  # cannot see it, so a gate-opening run may have cancelled the approval's
  # heavy CI and must take that work over rather than leave the sha without it.
  local added_label opening_label
  added_label="$(jq -r '.label.name // ""' "${EVENT_PATH}")"
  if ! label_is_allowed "${added_label}"; then
    emit_decision "false" "label-not-gating"
    exit 0
  fi
  opening_label="$(find_allowed_pr_label "$(payload_labels)" || true)"
  if [ "${added_label}" != "${opening_label}" ]; then
    emit_decision "false" "gate-already-open" "${opening_label}"
    exit 0
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
  emit_undecidable "missing-event-payload"
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
    emit_undecidable "pr-query-failed"
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
  if [ "${BASE_REF}" = "main" ]; then
    detect_replaced_pr_run
  fi
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
  emit_undecidable "missing-pr-context"
  exit 0
fi

OWNER="${REPO%%/*}"
NAME="${REPO#*/}"

if [ "${ACTION}" = "labeled" ]; then
  check_label_event
fi

load_pr_labels
MATCHED_LABEL="$(find_allowed_pr_label "${PR_LABELS}" || true)"
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
  emit_undecidable "review-query-failed"
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
