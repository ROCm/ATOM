import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = REPO_ROOT / ".github" / "scripts" / "check_heavy_ci_gate.sh"


def _run_gate(
    tmp_path,
    *,
    labels=(),
    review_decision="",
    review_query_fails=False,
    allowed="ci:full,ci:vllm",
):
    event_path = tmp_path / "event.json"
    output_path = tmp_path / "output.txt"
    summary_path = tmp_path / "summary.md"
    event_path.write_text(
        json.dumps(
            {
                "action": "synchronize",
                "pull_request": {
                    "number": 2166,
                    "draft": False,
                    "base": {"ref": "main"},
                    "labels": [{"name": label} for label in labels],
                },
            }
        ),
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ " $* " == *"/pulls/"*"/files"* ]]; then
  printf '%s\\n' 'atom/model_engine/scheduler.py'
elif [[ "${2:-}" == "graphql" ]]; then
  if [[ "${FAKE_REVIEW_QUERY_FAIL:-0}" == "1" ]]; then
    exit 1
  fi
  printf '%s\\n' "${FAKE_REVIEW_DECISION:-}"
elif [[ " $* " == *" /labels "* ]]; then
  exit 0
else
  echo "unexpected gh invocation: $*" >&2
  exit 2
fi
""",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "GITHUB_OUTPUT": str(output_path),
            "GITHUB_STEP_SUMMARY": str(summary_path),
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_EVENT_PATH": str(event_path),
            "GITHUB_REPOSITORY": "ROCm/ATOM",
            "CI_GATE_LABELS": allowed,
            "CI_GATE_PATHS_IGNORE": "docs/**",
            "FAKE_REVIEW_DECISION": review_decision,
            "FAKE_REVIEW_QUERY_FAIL": "1" if review_query_fails else "0",
        }
    )
    subprocess.run(["bash", str(GATE_SCRIPT)], check=True, env=env)
    return dict(
        line.split("=", 1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    )


def test_matching_label_runs_without_review(tmp_path):
    result = _run_gate(tmp_path, labels=("ci:vllm",))

    assert result["should_run"] == "true"
    assert result["reason"] == "label-present"
    assert result["matched_label"] == "ci:vllm"


def test_unrelated_label_does_not_authorize_workflow(tmp_path):
    result = _run_gate(tmp_path, labels=("ci:atom",))

    assert result["should_run"] == "false"
    assert result["reason"] == "not-approved"


def test_current_approval_runs_without_label(tmp_path):
    result = _run_gate(tmp_path, review_decision="APPROVED")

    assert result["should_run"] == "true"
    assert result["reason"] == "current-approval"
    assert result["review_decision"] == "APPROVED"


def test_dismissed_or_superseded_review_does_not_run(tmp_path):
    result = _run_gate(tmp_path, review_decision="REVIEW_REQUIRED")

    assert result["should_run"] == "false"
    assert result["reason"] == "not-approved"
    assert result["approval_count"] == "0"


def test_changes_requested_does_not_run(tmp_path):
    result = _run_gate(tmp_path, review_decision="CHANGES_REQUESTED")

    assert result["should_run"] == "false"
    assert result["reason"] == "changes-requested"
    assert result["changes_requested_count"] == "1"


def test_review_query_failure_fails_closed(tmp_path):
    result = _run_gate(tmp_path, review_query_fails=True)

    assert result["should_run"] == "false"
    assert result["reason"] == "review-query-failed"
