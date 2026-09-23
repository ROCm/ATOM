"""Additional authorization boundaries for the disposable live-test PR."""

import pytest
from test_heavy_ci_gate import _current_pr, _run_gate


@pytest.mark.parametrize("workflow_label", ["ci:atom", "ci:vllm", "ci:sglang"])
@pytest.mark.parametrize(
    "label", ["ci:full", "ci:atom", "ci:vllm", "ci:sglang", "bug", "CI:FULL", "ci:"]
)
def test_label_scope_is_exact(tmp_path, workflow_label, label):
    result = _run_gate(tmp_path, allowed=f"ci:full,{workflow_label}", labels=(label,))
    assert result["should_run"] == str(label in {"ci:full", workflow_label}).lower()


@pytest.mark.parametrize("workflow_label", ["ci:atom", "ci:vllm", "ci:sglang"])
@pytest.mark.parametrize("state", ["commented", "changes_requested", "dismissed"])
def test_labels_do_not_turn_other_reviews_into_approval(
    tmp_path, workflow_label, state
):
    result = _run_gate(
        tmp_path,
        allowed=f"ci:full,{workflow_label}",
        labels=(workflow_label,),
        event_name="pull_request_review",
        action="submitted",
        review_state=state,
        review_decision="APPROVED",
    )
    assert result["reason"] == "review-not-approved-event"
    assert result["should_run"] == "false"


@pytest.mark.parametrize("workflow_label", ["ci:atom", "ci:vllm", "ci:sglang"])
@pytest.mark.parametrize("labeled", [False, True])
def test_draft_keeps_existing_explicit_label_override(
    tmp_path, workflow_label, labeled
):
    labels = [{"name": workflow_label}] if labeled else []
    result = _run_gate(
        tmp_path,
        allowed=f"ci:full,{workflow_label}",
        current_pr=_current_pr(draft=True, labels=labels),
        event_name="pull_request_review",
        action="submitted",
        review_decision="APPROVED",
    )
    assert result["reason"] == ("label-present" if labeled else "draft-pull-request")
    assert result["should_run"] == str(labeled).lower()


@pytest.mark.parametrize("workflow_label", ["ci:atom", "ci:vllm", "ci:sglang"])
@pytest.mark.parametrize("labeled", [False, True])
def test_ignored_paths_keep_existing_explicit_label_override(
    tmp_path, workflow_label, labeled
):
    result = _run_gate(
        tmp_path,
        allowed=f"ci:full,{workflow_label}",
        labels=(workflow_label,) if labeled else (),
        changed_files="docs/guide.rst",
        event_name="pull_request_review",
        action="submitted",
        review_decision="APPROVED",
    )
    assert result["reason"] == ("label-present" if labeled else "ignored-paths-only")
    assert result["should_run"] == str(labeled).lower()
