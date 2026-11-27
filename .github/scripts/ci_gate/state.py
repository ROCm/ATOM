# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Reads a PR's current state and its runs from the GitHub API into the inputs
of decide(). Nothing is taken from an event payload."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .decide import Attempt, ChangedPaths, Job, PullRequest, Run, speaks_for
from .deps import declared_dependencies
from .github_api import GitHub
from .suites import Suite

_MAX_FILES = 3000  # what the PR files endpoint returns at most
# Every declared dependency is checked; a body's length bounds how many.
_DEPENDENCIES_PER_QUERY = 50
_MAX_RUNS = 200  # runs of one commit listed; far more than a commit gets

_PR_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      number state isDraft baseRefName headRefOid title body reviewDecision changedFiles
      headRepository { nameWithOwner }
      labels(first: 100) { nodes { name } }
      timelineItems(last: 1, itemTypes: [BASE_REF_CHANGED_EVENT, AUTOMATIC_BASE_CHANGE_SUCCEEDED_EVENT]) {
        nodes {
          ... on BaseRefChangedEvent { createdAt }
          ... on AutomaticBaseChangeSucceededEvent { createdAt }
        }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class PrState:
    pr: PullRequest
    dependency_notes: tuple[str, ...]  # declared but not holding: why


def timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def load_pr(gh: GitHub, number: int) -> PrState | None:
    """The PR's state; None when there is no such PR."""
    data = gh.graphql(_PR_QUERY, {"owner": gh.owner, "name": gh.name, "number": number})
    node = data["repository"]["pullRequest"]
    if node is None:
        return None
    base_changes = node["timelineItems"]["nodes"]
    open_deps, notes = _dependencies(
        gh, number, f"{node['title']}\n{node['body'] or ''}"
    )
    pr = PullRequest(
        number=number,
        is_open=node["state"] == "OPEN",
        draft=node["isDraft"],
        base_ref=node["baseRefName"],
        head_sha=node["headRefOid"],
        head_repo=(node.get("headRepository") or {}).get("nameWithOwner", ""),
        base_repo=gh.repo,
        labels=frozenset(label["name"] for label in node["labels"]["nodes"]),
        review_decision=node["reviewDecision"] or "",
        changed=_changed_paths(gh, number, node["changedFiles"]),
        last_base_change_at=(
            timestamp(base_changes[-1]["createdAt"]) if base_changes else None
        ),
        open_dependencies=open_deps,
    )
    return PrState(pr, notes)


def _changed_paths(gh: GitHub, number: int, changed_files: int) -> ChangedPaths:
    files = gh.rest_pages(f"repos/{gh.repo}/pulls/{number}/files", None, {}, _MAX_FILES)
    paths = {f["filename"] for f in files} | {
        f["previous_filename"] for f in files if f.get("previous_filename")
    }
    return ChangedPaths(tuple(sorted(paths)), complete=len(files) >= changed_files)


def _dependencies(
    gh: GitHub, number: int, text: str
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    declared = declared_dependencies(text, number, gh.repo)
    found: dict[str, dict | None] = {}
    for start in range(0, len(declared), _DEPENDENCIES_PER_QUERY):
        batch = declared[start : start + _DEPENDENCIES_PER_QUERY]
        fields = " ".join(
            f"d{n}: issueOrPullRequest(number: {n}) {{ __typename ... on PullRequest {{ state title body }} }}"
            for n in batch
        )
        query = f"query($owner: String!, $name: String!) {{ repository(owner: $owner, name: $name) {{ {fields} }} }}"
        found |= gh.graphql(query, {"owner": gh.owner, "name": gh.name})["repository"]
    holding: list[int] = []
    notes: list[str] = []
    for n in declared:
        node = found.get(f"d{n}")
        if node is None or node["__typename"] != "PullRequest":
            notes.append(f"#{n} is not a PR here")
        elif node["state"] == "MERGED":
            continue
        elif node["state"] == "CLOSED":
            notes.append(f"#{n} was closed without merging")
        elif number in declared_dependencies(
            f"{node['title']}\n{node['body'] or ''}", n, gh.repo
        ):
            notes.append(f"#{n} declares this PR back (a cycle)")
        else:
            holding.append(n)
    return tuple(holding), tuple(notes)


def load_runs(gh: GitHub, pr: PullRequest, suite: Suite) -> tuple[Run, ...]:
    """Every run of the suite's workflow that speaks for the PR's head, newest
    first, each with its latest attempt's jobs: the evidence decide() weighs."""
    listed = gh.rest_pages(
        f"repos/{gh.repo}/actions/workflows/{suite.workflow_file}/runs",
        "workflow_runs",
        {"head_sha": pr.head_sha},
        _MAX_RUNS,
    )
    runs: list[Run] = []
    for item in sorted(listed, key=lambda r: r["created_at"], reverse=True):
        created = timestamp(item["created_at"])
        if speaks_for(pr, item["event"], item["head_sha"], created):
            runs.append(_run(gh, item, created, suite, newest=not runs))
    return tuple(runs)


def _run(gh: GitHub, item: dict, created: float, suite: Suite, newest: bool) -> Run:
    """One run, its latest attempt's jobs read. Who triggered the earlier
    attempts is read only for the newest run, the only one decide() may re-run."""
    run_id = item["id"]
    latest_number = item["run_attempt"]
    jobs = tuple(
        Job(job["name"], job["conclusion"] or job["status"])
        for job in gh.rest_pages(
            f"repos/{gh.repo}/actions/runs/{run_id}/attempts/{latest_number}/jobs",
            "jobs",
            {},
            500,
        )
    )
    earlier = tuple(
        Attempt(n, _attempt_actor(gh, run_id, n) if newest else "", "completed")
        for n in range(1, latest_number)
    )
    latest = Attempt(
        latest_number,
        (item.get("triggering_actor") or {}).get("login", ""),
        _attempt_status(item),
        jobs,
        item.get("conclusion") or "",
    )
    new_names = {suite.result_job, suite.not_run_job, suite.stale_merge_job}
    return Run(
        run_id=run_id,
        event=item["event"],
        head_sha=item["head_sha"],
        created_at=created,
        attempts=earlier + (latest,),
        old_definition=not any(job.name in new_names for job in jobs),
    )


def _attempt_actor(gh: GitHub, run_id: int, number: int) -> str:
    attempt = gh.rest("GET", f"repos/{gh.repo}/actions/runs/{run_id}/attempts/{number}")
    return (attempt.get("triggering_actor") or {}).get("login", "")


def _attempt_status(item: dict) -> str:
    if item.get("conclusion") == "action_required":
        return "action_required"
    if item["status"] == "completed":
        return "completed"
    return (
        "queued"
        if item["status"] in ("queued", "waiting", "requested", "pending")
        else "in_progress"
    )
