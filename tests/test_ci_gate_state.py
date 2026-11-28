# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""How the gate reads a PR and its runs (ci_gate/state.py), on a fake GitHub."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

from ci_gate.state import load_pr, load_runs
from ci_gate.suites import SUITES

HEAD = "a" * 40
SUITE = SUITES[0]


class FakeGitHub:
    repo = "ROCm/ATOM"
    owner = "ROCm"
    name = "ATOM"

    def __init__(self, *, pr=None, files=(), deps=None, runs=(), jobs=None):
        self.pr = pr
        self.files = list(files)
        self.deps = deps or {}
        self.runs = list(runs)
        self.jobs = jobs or {}
        self.dependency_queries = 0

    def graphql(self, query, variables):
        if "pullRequest(number" in query:
            return {"repository": {"pullRequest": self.pr}}
        self.dependency_queries += 1
        asked = [int(n) for n in re.findall(r"\bd(\d+):", query)]
        return {"repository": {f"d{n}": self.deps.get(n) for n in asked}}

    def rest_pages(self, path, key, params, limit):
        if path.endswith("/files"):
            return self.files[:limit]
        if "/actions/workflows/" in path:
            return self.runs[:limit]
        run_id = int(path.split("/actions/runs/")[1].split("/")[0])
        return self.jobs.get(run_id, [])

    def rest(self, method, path, params=None, body=None):
        if "/attempts/" in path:
            return {"triggering_actor": {"login": "someone"}}
        raise AssertionError(path)


def pr_node(**changes):
    node = {
        "number": 7,
        "state": "OPEN",
        "isDraft": False,
        "baseRefName": "main",
        "headRefOid": HEAD,
        "title": "x",
        "body": "",
        "reviewDecision": None,
        "changedFiles": 1,
        "headRepository": {"nameWithOwner": "ROCm/ATOM"},
        "labels": {"nodes": []},
        "timelineItems": {"nodes": []},
    }
    node.update(changes)
    return node


def test_a_missing_pr_is_none():
    assert load_pr(FakeGitHub(pr=None), 7) is None


def test_a_base_that_never_changed_bounds_no_run():
    assert load_pr(FakeGitHub(pr=pr_node()), 7).pr.last_base_change_at is None


def test_a_rename_counts_both_paths():
    files = [{"filename": "docs/moved.py", "previous_filename": "atom/moved.py"}]
    state = load_pr(FakeGitHub(pr=pr_node(), files=files), 7)
    assert state.pr.changed.paths == ("atom/moved.py", "docs/moved.py")
    assert state.pr.changed.complete


def test_a_file_list_shorter_than_the_pr_is_incomplete():
    files = [{"filename": "docs/a.md"}]
    state = load_pr(FakeGitHub(pr=pr_node(changedFiles=3001), files=files), 7)
    assert not state.pr.changed.complete


def test_a_dependency_that_is_not_a_pr_here_does_not_hold():
    # A GraphQL NOT_FOUND leaves the field null (github_api.py).
    gh = FakeGitHub(
        pr=pr_node(body="Depends on #8636 and #12"),
        deps={
            8636: None,
            12: {"__typename": "PullRequest", "state": "OPEN", "title": "", "body": ""},
        },
    )
    state = load_pr(gh, 7)
    assert state.pr.open_dependencies == (12,)
    assert state.dependency_notes == ("#8636 is not a PR here",)


def test_every_declared_dependency_is_checked_however_many():
    merged = {"__typename": "PullRequest", "state": "MERGED", "title": "", "body": ""}
    opened = {**merged, "state": "OPEN"}
    numbers = range(100, 221)  # more than one query holds
    deps = {n: merged for n in numbers} | {220: opened}
    body = "Depends on " + ", ".join(f"#{n}" for n in numbers)
    gh = FakeGitHub(pr=pr_node(body=body), deps=deps)
    state = load_pr(gh, 7)
    # The newest is often the one not merged yet.
    assert state.pr.open_dependencies == (220,)
    assert gh.dependency_queries > 1


def test_a_dependency_cycle_does_not_hold():
    back = {
        "__typename": "PullRequest",
        "state": "OPEN",
        "title": "",
        "body": "Depends on #7",
    }
    state = load_pr(FakeGitHub(pr=pr_node(body="Depends on #12"), deps={12: back}), 7)
    assert state.pr.open_dependencies == ()


def run_item(run_id, event, created="2026-11-02T00:00:00Z", repo="ROCm/ATOM"):
    return {
        "id": run_id,
        "event": event,
        "head_sha": HEAD,
        "head_repository": {"full_name": repo},
        "created_at": created,
        "run_attempt": 1,
        "status": "completed",
        "conclusion": "success",
        "triggering_actor": {"login": "a"},
    }


def test_only_runs_speaking_for_the_head_are_read():
    runs = [
        run_item(1, "pull_request"),
        run_item(2, "pull_request_review"),
        run_item(3, "workflow_dispatch"),
        run_item(
            4, "pull_request", created="2026-10-01T00:00:00Z"
        ),  # before the base change
        # A fork PR's review run reports the base repository: still its head.
        run_item(5, "pull_request_review", repo="ROCm/ATOM"),
    ]
    state = load_pr(
        FakeGitHub(
            pr=pr_node(timelineItems={"nodes": [{"createdAt": "2026-10-15T00:00:00Z"}]})
        ),
        7,
    )
    read = load_runs(FakeGitHub(runs=runs), state.pr, SUITE)
    assert sorted(r.run_id for r in read) == [1, 2, 5]
    assert {r.event for r in read} == {"pull_request", "pull_request_review"}
