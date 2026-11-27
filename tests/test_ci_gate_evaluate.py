# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The heavy-CI gate's evaluator (ci_gate/evaluate.py, shadow phase) against a
fake GitHub: which PRs an event names, which the scheduled pass picks, what it
writes for one PR, and that it takes no action."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

from ci_gate import decide as d
from ci_gate import evaluate as ev
from ci_gate.suites import SUITES

ATOM_TEST = SUITES[0]
HEAD = "a" * 40


class FakeGitHub:
    repo = "ROCm/ATOM"
    owner = "ROCm"
    name = "ATOM"

    def __init__(
        self,
        *,
        pr=None,
        files=(),
        runs=None,
        jobs=None,
        statuses=(),
        pages=(),
        pulls=(),
        pr_rest=None,
    ):
        self.pr = pr
        self.files = list(files)
        self.runs = runs or {}  # workflow file -> run items
        self.jobs = jobs or {}  # run id -> job items
        self.statuses = list(statuses)
        self.pages = list(pages)
        self.pulls = list(pulls)
        self.pr_rest = pr_rest
        self.posts: list[tuple[str, dict]] = []
        self.searches: list[str] = []

    def rest(self, method, path, params=None, body=None):
        if method == "POST":
            self.posts.append((path, body))
            return {}
        if path == "search/issues":
            self.searches.append(params["q"])
            return {"items": self.pulls}
        if "/actions/workflows/" in path:
            return {
                "workflow_runs": self.runs.get(
                    path.split("/actions/workflows/")[1].split("/")[0], []
                )
            }
        if path.endswith("/status"):
            return {"statuses": self.statuses}
        if path.endswith("/check-runs"):
            return {"check_runs": [{"conclusion": "success", "started_at": "x"}]}
        raise AssertionError(f"unexpected GET {path}")

    def rest_optional(self, path, params=None):
        return self.pr_rest

    def rest_pages(self, path, key, params, limit):
        if path.endswith("/files"):
            return self.files
        if "/actions/workflows/" in path:
            return self.runs.get(path.split("/actions/workflows/")[1].split("/")[0], [])
        return self.jobs.get(int(path.split("/actions/runs/")[1].split("/")[0]), [])

    def graphql(self, query, variables):
        if "pullRequests(" in query:
            return {"repository": {"pullRequests": self.pages.pop(0)}}
        return {"repository": {"pullRequest": self.pr}}


def pr_node(**changes):
    node = {
        "number": 7,
        "state": "OPEN",
        "isDraft": False,
        "baseRefName": "main",
        "headRefOid": HEAD,
        "createdAt": "2026-11-01T00:00:00Z",
        "title": "feat: x",
        "body": "",
        "reviewDecision": "APPROVED",
        "changedFiles": 2,
        "headRepository": {"nameWithOwner": "ROCm/ATOM"},
        "labels": {"nodes": []},
        "timelineItems": {"nodes": []},
    }
    node.update(changes)
    return node


# Core and plugin code: every suite but ATOMesh tests it.
FILES = [{"filename": "atom/x.py"}, {"filename": "atom/plugin/common.py"}]


def run_item(run_id, event="pull_request"):
    return {
        "id": run_id,
        "event": event,
        "head_sha": HEAD,
        "head_repository": {"full_name": "ROCm/ATOM"},
        "created_at": "2026-11-02T00:00:00Z",
        "run_attempt": 1,
        "status": "completed",
        "conclusion": "success",
        "triggering_actor": {"login": "author"},
    }


def job(name, conclusion="success"):
    return {"name": name, "conclusion": conclusion, "status": "completed"}


# ---- input -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, number", [("3862", 3862), ("#3862", 3862), (" 12 ", 12)]
)
def test_pr_numbers_as_people_type_them(text, number):
    assert ev.parse_pr_number(text) == number


@pytest.mark.parametrize("text", ["", "abc", "#", "0", "-3", "38 62", "²", "٣٨", "012"])
def test_anything_else_is_refused(text):
    with pytest.raises(ev.InputError):
        ev.parse_pr_number(text)


# ---- which PRs an event names ----------------------------------------------------


def test_pull_request_target_names_its_pr():
    assert ev.event_targets(
        FakeGitHub(), "pull_request_target", {"pull_request": {"number": 12}}, ""
    ) == [12]


def test_dispatch_names_its_input_or_runs_the_scheduled_pass():
    assert ev.event_targets(FakeGitHub(), "workflow_dispatch", {}, "#34") == [34]
    assert ev.event_targets(FakeGitHub(), "workflow_dispatch", {}, " ") is None
    assert ev.event_targets(FakeGitHub(), "schedule", {}, "") is None
    with pytest.raises(ev.InputError):
        ev.event_targets(FakeGitHub(), "workflow_dispatch", {}, "abc")


def test_relay_run_names_the_pr_in_its_title():
    run = {
        "path": ".github/workflows/ci-review-relay.yaml",
        "display_title": "review-relay #56",
    }
    assert ev.event_targets(
        FakeGitHub(), "workflow_run", {"workflow_run": run}, ""
    ) == [56]
    run["display_title"] = "review-relay #56; rm -rf /"
    assert (
        ev.event_targets(FakeGitHub(), "workflow_run", {"workflow_run": run}, "") == []
    )


@pytest.mark.parametrize(
    "event, traced",
    [
        ("pull_request", True),
        ("pull_request_review", True),
        ("workflow_dispatch", False),
    ],
)
def test_a_suite_run_is_traced_to_prs_only_when_it_speaks_for_one(event, traced):
    # A fork PR's review run reports the base repository and the fork's branch
    # name: only the commit identifies the PR.
    run = {
        "path": ".github/workflows/atom-test.yaml",
        "event": event,
        "head_sha": HEAD,
        "head_branch": "feat",
        "head_repository": {"owner": {"login": "ROCm"}},
    }
    gh = FakeGitHub(pulls=[{"number": 78}])
    assert ev.event_targets(gh, "workflow_run", {"workflow_run": run}, "") == (
        [78] if traced else []
    )
    if traced:
        assert gh.searches == [f"repo:ROCm/ATOM is:pr is:open {HEAD}"]


def test_no_search_without_a_full_commit():
    run = {
        "path": ".github/workflows/atom-test.yaml",
        "event": "pull_request",
        "head_sha": "abc",
    }
    assert (
        ev.event_targets(FakeGitHub(), "workflow_run", {"workflow_run": run}, "") == []
    )


def test_same_named_workflow_that_is_not_a_suite_is_ignored():
    run = {"path": ".github/workflows/atom-mmstar-ci.yaml", "event": "schedule"}
    assert (
        ev.event_targets(FakeGitHub(), "workflow_run", {"workflow_run": run}, "") == []
    )


# ---- the scheduled pass -----------------------------------------------------------


def page_entry(number, suites, draft=False):
    contexts = [{"context": s.status_context} for s in suites]
    return {
        "number": number,
        "isDraft": draft,
        "commits": {"nodes": [{"commit": {"status": {"contexts": contexts}}}]},
    }


def test_scheduled_pass_takes_prs_missing_a_status_first_then_every_other():
    pages = [
        {
            "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
            "nodes": [
                # Whatever a Status says, a lost event may have outdated it.
                page_entry(5, SUITES),
                page_entry(3, SUITES),
                page_entry(9, SUITES[:1]),  # missing the others
            ],
        },
        {
            "pageInfo": {"hasNextPage": False, "endCursor": None},
            "nodes": [page_entry(8, ()), page_entry(4, (), draft=True)],
        },
    ]
    assert ev.reconcile_targets(FakeGitHub(pages=pages), limit=10, pass_index=0) == [
        9,
        8,
        3,
        5,
    ]


def test_prs_missing_a_status_come_first_newest_first():
    assert ev.select_for_pass([30, 10, 20], [5, 6], limit=2, pass_index=0) == [30, 20]


def test_every_pr_that_needs_a_look_gets_one_within_a_bounded_number_of_passes():
    others = list(range(1, 51))  # more than fit in one pass
    seen: set[int] = set()
    for pass_index in range(5):  # ceil(50 / 10) passes
        seen.update(ev.select_for_pass([], others, limit=10, pass_index=pass_index))
    assert seen == set(others)


def test_missing_ones_share_a_pass_with_the_rotation():
    picked = ev.select_for_pass([99], [1, 2, 3], limit=2, pass_index=1)
    assert picked[0] == 99 and len(picked) == 2


# ---- what it writes for one PR ---------------------------------------------------


def test_writer_marks_preview_truncates_and_skips_unchanged():
    gh = FakeGitHub()
    writer = ev.StatusWriter(gh)
    assert writer.write(HEAD, "C", {}, ("pending", "x" * 300), "u") == "written"
    path, body = gh.posts[0]
    assert path == f"repos/ROCm/ATOM/statuses/{HEAD}"
    assert (
        body["description"].startswith(ev.PREVIEW) and len(body["description"]) == 140
    )
    same = {"C": ("pending", body["description"])}
    assert writer.write(HEAD, "C", same, ("pending", "x" * 300), "u") == "unchanged"


def test_dry_run_posts_nothing():
    gh = FakeGitHub()
    assert (
        ev.StatusWriter(gh, dry_run=True).write(HEAD, "C", {}, ("pending", "x"), "u")
        == "dry run: would write"
    )
    assert not gh.posts


def test_evaluation_writes_every_suite_and_takes_no_action():
    gh = FakeGitHub(
        pr=pr_node(),
        files=FILES,
        runs={
            "atom-test.yaml": [run_item(101, "pull_request_review")],
            "atom-sglang-test.yaml": [run_item(202)],
        },
        jobs={
            101: [
                job(ATOM_TEST.old_signal_job),
                job("Accuracy (m1)"),
                job(ATOM_TEST.old_gate_job),
            ],
            202: [job(SUITES[1].not_run_job)],
        },
    )
    report = ev.Report()

    ev.evaluate_pr(gh, 7, ev.StatusWriter(gh), report, "https://run")

    written = {body["context"]: body for _, body in gh.posts}
    assert all(path.endswith(f"/statuses/{HEAD}") for path, _ in gh.posts)  # no re-run
    assert (
        written[ATOM_TEST.status_context]["state"] == "success"
    )  # proved by the approval run
    assert written[SUITES[1].status_context]["description"] == ev.PREVIEW + "Re-running"
    assert any(f"{d.ACT_RERUN} 202" in row for row in report.rows)
    assert (
        written[SUITES[2].status_context]["description"] == ev.PREVIEW + d.PUSH_NEEDED
    )


def test_a_pr_that_does_not_exist_writes_nothing():
    gh = FakeGitHub(pr=None)
    ev.evaluate_pr(gh, 7, ev.StatusWriter(gh), ev.Report(), "u")
    assert not gh.posts


def test_a_failed_evaluation_shows_as_error_but_never_over_a_success():
    statuses = [
        {"context": ATOM_TEST.status_context, "state": "success", "description": "ok"}
    ]
    gh = FakeGitHub(
        statuses=statuses,
        pr_rest={"state": "open", "draft": False, "head": {"sha": HEAD}},
    )

    ev.write_error(gh, 7, ev.StatusWriter(gh), "https://run")

    contexts = {body["context"] for _, body in gh.posts}
    assert contexts == {s.status_context for s in SUITES[1:]}
    assert all(body["state"] == "error" for _, body in gh.posts)


@pytest.mark.parametrize(
    "pr_rest", [None, {"state": "closed", "draft": False, "head": {"sha": HEAD}}]
)
def test_no_error_is_written_where_no_status_belongs(pr_rest):
    gh = FakeGitHub(pr_rest=pr_rest)
    ev.write_error(gh, 7, ev.StatusWriter(gh), "u")
    assert not gh.posts
