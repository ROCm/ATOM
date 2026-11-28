# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The evaluator (ci-evaluator.yaml), shadow phase. Two commands, one job each:

  resolve        the PRs an event names, or the scheduled pass selects; writes
                 `prs` (a JSON list) to GITHUB_OUTPUT
  evaluate PR    write every suite's Status for one PR from decide()

Each PR is evaluated in its own job, in a concurrency group of that PR, so
one PR is never evaluated twice at once and a failure stays with its PR.
Statuses carry PREVIEW and are not required yet; the actions a decision asks
for (re-run, cancel, dependency hold) are only reported.

Run from .github/scripts: python3 -m ci_gate.evaluate {resolve | evaluate PR}
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

from . import decide as d
from .github_api import GitHub
from .state import load_pr, load_runs
from .suites import SUITES, SUITES_BY_WORKFLOW_FILE, Suite

PREVIEW = "[preview] "
RELAY_WORKFLOW_FILE = "ci-review-relay.yaml"
RELAY_TITLE_PREFIX = "review-relay #"
_DESCRIPTION_LIMIT = 140
# Per scheduled pass: up to four Status writes a PR, under the secondary
# limit of 500 content-creating requests an hour at one pass per 10 minutes.
_DEFAULT_MAX_PRS = 20
_PASS_PERIOD_S = 600  # the scheduled pass's cron period (ci-evaluator.yaml)

_RECONCILE_QUERY = """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: 100, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number isDraft
        commits(last: 1) { nodes { commit { status { contexts { context } } } } }
      }
    }
  }
}
"""


class InputError(ValueError):
    pass


_PR_NUMBER = re.compile(r"#?([1-9][0-9]*)")


def parse_pr_number(text: str) -> int:
    """A PR number as a person types it: `3862` or `#3862` (ASCII digits)."""
    match = _PR_NUMBER.fullmatch(text.strip())
    if match is None:
        raise InputError(f"not a PR number: {text!r}")
    return int(match.group(1))


@dataclass
class StatusWriter:
    gh: GitHub
    dry_run: bool = False  # decide and report, write nothing
    written: int = 0

    def write(
        self,
        sha: str,
        context: str,
        current: dict[str, tuple[str, str]],
        status: tuple[str, str],
        target_url: str,
    ) -> str:
        state, description = status
        text = (PREVIEW + description)[:_DESCRIPTION_LIMIT]
        if current.get(context) == (state, text):
            return "unchanged"
        self.written += 1
        if self.dry_run:
            return "dry run: would write"
        self.gh.rest(
            "POST",
            f"repos/{self.gh.repo}/statuses/{sha}",
            body={
                "state": state,
                "context": context,
                "description": text,
                "target_url": target_url,
            },
        )
        return "written"


@dataclass
class Report:
    rows: list[str] = field(default_factory=list)

    def add(
        self,
        pr: int,
        suite: Suite,
        decision: str,
        old_gate: str,
        action: str,
        write: str,
    ) -> None:
        self.rows.append(
            f"| #{pr} | {suite.key} | {decision} | {old_gate} | {action} | {write} |"
        )

    def render(self) -> str:
        head = "| PR | suite | new Status | old gate | action (not taken) | write |\n|---|---|---|---|---|---|"
        return "\n".join([head, *self.rows]) if self.rows else "Nothing written."


# ---- resolve ---------------------------------------------------------------


def event_targets(
    gh: GitHub, event_name: str, payload: dict, dispatch_input: str
) -> list[int] | None:
    """PR numbers an event names; None for the scheduled pass."""
    if event_name == "pull_request_target":
        return [int(payload["pull_request"]["number"])]
    if event_name == "workflow_dispatch":
        return [parse_pr_number(dispatch_input)] if dispatch_input.strip() else None
    if event_name == "schedule":
        return None
    if event_name != "workflow_run":
        return []
    run = payload["workflow_run"]
    workflow_file = run.get("path", "").rsplit("/", 1)[-1].split("@", 1)[0]
    if workflow_file == RELAY_WORKFLOW_FILE:
        try:
            return [
                parse_pr_number(
                    run.get("display_title", "").removeprefix(RELAY_TITLE_PREFIX)
                )
            ]
        except InputError:
            return []
    if (
        workflow_file not in SUITES_BY_WORKFLOW_FILE
        or run.get("event") not in d.PR_EVENTS
    ):
        return []
    return prs_with_commit(gh, run.get("head_sha", ""))


def prs_with_commit(gh: GitHub, sha: str) -> list[int]:
    """Open PRs that contain the commit, found by the commit itself: a run's
    repository and branch name do not identify a fork PR (its review runs
    report the base repository). A PR that only contains the commit is
    evaluated too, harmlessly: evaluation reads the PR afresh."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        return []
    found = gh.rest(
        "GET",
        "search/issues",
        {"q": f"repo:{gh.repo} is:pr is:open {sha}", "per_page": 10},
    )
    return [item["number"] for item in found["items"]]


def select_for_pass(
    missing: list[int], other: list[int], limit: int, pass_index: int
) -> list[int]:
    """At most `limit` PRs for one scheduled pass: those missing a Status
    first (newest first), then the others in turn, each pass starting where
    the previous one stopped, so every PR gets a look within a bounded number
    of passes, without state kept between passes."""
    chosen = sorted(missing, reverse=True)[:limit]
    slots = limit - len(chosen)
    if slots <= 0 or not other:
        return chosen
    ring = sorted(other)
    start = (pass_index * slots) % len(ring)
    rotated = ring[start:] + ring[:start]
    return chosen + rotated[:slots]


def reconcile_targets(gh: GitHub, limit: int, pass_index: int) -> list[int]:
    """The open non-draft PRs, whatever their base (a PR moved off main has a
    Status to correct), as select_for_pass picks them for this
    pass. Every one is a candidate: a Status depends on the PR's text, its
    runs and its dependencies, which this page does not show, so after a lost
    event any Status may be out of date."""
    missing: list[int] = []
    other: list[int] = []
    after = None
    while True:
        data = gh.graphql(
            _RECONCILE_QUERY, {"owner": gh.owner, "name": gh.name, "after": after}
        )
        page = data["repository"]["pullRequests"]
        for node in page["nodes"]:
            if node["isDraft"]:
                continue
            target = missing if _misses_a_status(node) else other
            target.append(node["number"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
    return select_for_pass(missing, other, limit, pass_index)


def _misses_a_status(node: dict) -> bool:
    commits = node["commits"]["nodes"]
    status = (commits[0]["commit"]["status"] or {}) if commits else {}
    written = {c["context"] for c in status.get("contexts", [])}
    return any(suite.status_context not in written for suite in SUITES)


# ---- evaluate --------------------------------------------------------------


def _needs_runs(pr: d.PullRequest, suite: Suite) -> bool:
    """Whether decide() gets past the rules that do not look at runs."""
    return (
        pr.is_open
        and not pr.draft
        and pr.base_ref == "main"
        and not pr.open_dependencies
        and d.needs_suite(pr.changed, suite)
    )


def _old_gate(gh: GitHub, sha: str, suite: Suite) -> str:
    runs = gh.rest(
        "GET",
        f"repos/{gh.repo}/commits/{sha}/check-runs",
        {"check_name": suite.old_gate_job, "filter": "latest", "per_page": 10},
    )["check_runs"]
    if not runs:
        return "none"
    latest = max(runs, key=lambda r: r.get("started_at") or "")
    return latest.get("conclusion") or latest.get("status") or "?"


def _current_statuses(gh: GitHub, sha: str) -> dict[str, tuple[str, str]]:
    combined = gh.rest(
        "GET", f"repos/{gh.repo}/commits/{sha}/status", {"per_page": 100}
    )
    return {
        s["context"]: (s["state"], s["description"] or "") for s in combined["statuses"]
    }


def evaluate_pr(
    gh: GitHub, number: int, writer: StatusWriter, report: Report, run_url: str
) -> None:
    state = load_pr(gh, number)
    if state is None:
        print(f"PR #{number} does not exist; nothing to evaluate.")
        return
    pr = state.pr
    current = _current_statuses(gh, pr.head_sha)
    now = time.time()
    notes = "; ".join(state.dependency_notes)
    for suite in SUITES:
        runs = load_runs(gh, pr, suite) if _needs_runs(pr, suite) else ()
        decision = d.decide(pr, runs, suite, now)
        action = (
            decision.action
            if decision.action == d.ACT_NONE
            else f"{decision.action} {decision.run_id or ''}".strip()
        )
        if decision.status is None:
            report.add(number, suite, "(not written)", "-", action, "-")
            continue
        verdict, description = decision.status
        if notes and verdict != d.SUCCESS:
            description = f"{description} (not holding: {notes})"
        target = (
            f"https://github.com/{gh.repo}/pull/{pr.open_dependencies[0]}"
            if pr.open_dependencies
            else run_url
        )
        result = writer.write(
            pr.head_sha, suite.status_context, current, (verdict, description), target
        )
        report.add(
            number,
            suite,
            f"{verdict}: {description}",
            _old_gate(gh, pr.head_sha, suite),
            action,
            result,
        )


def write_error(gh: GitHub, number: int, writer: StatusWriter, run_url: str) -> None:
    """Make a failed evaluation visible on the PR: error, never over a success."""
    pr = gh.rest_optional(f"repos/{gh.repo}/pulls/{number}")
    if pr is None or pr["state"] != "open" or pr["draft"]:
        return
    sha = pr["head"]["sha"]
    current = _current_statuses(gh, sha)
    for suite in SUITES:
        if current.get(suite.status_context, ("", ""))[0] == d.SUCCESS:
            continue
        writer.write(
            sha,
            suite.status_context,
            current,
            ("error", "Evaluator error; the next pass retries"),
            run_url,
        )


# ---- commands --------------------------------------------------------------


def _append(path_variable: str, text: str) -> None:
    path = os.environ.get(path_variable)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)


def resolve(gh: GitHub) -> list[int]:
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as handle:
        payload = json.load(handle)
    targets = event_targets(
        gh, os.environ["GITHUB_EVENT_NAME"], payload, os.environ.get("CI_GATE_PR", "")
    )
    if targets is None:
        targets = reconcile_targets(
            gh,
            int(os.environ.get("CI_GATE_MAX_PRS", _DEFAULT_MAX_PRS)),
            pass_index=int(time.time() // _PASS_PERIOD_S),
        )
    return sorted(set(targets))


def _run_url(gh: GitHub) -> str:
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    return f"{server}/{gh.repo}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"


def _writer(gh: GitHub) -> StatusWriter:
    return StatusWriter(gh, dry_run=os.environ.get("CI_GATE_DRY_RUN") == "1")


def evaluate(gh: GitHub, number: int) -> None:
    writer = _writer(gh)
    report = Report()
    evaluate_pr(gh, number, writer, report, _run_url(gh))
    summary = f"## CI evaluator (shadow): PR #{number}\n\nStatuses written: {writer.written}.\n\n{report.render()}\n"
    print(summary)
    _append("GITHUB_STEP_SUMMARY", summary)


def main(argv: list[str]) -> int:
    """`resolve`; `evaluate PR`; `error PR`, which the workflow runs when the
    evaluate step failed, so the failure shows on the PR."""
    gh = GitHub(os.environ["GH_TOKEN"], os.environ["GITHUB_REPOSITORY"])
    command, args = (argv[0], argv[1:]) if argv else ("", [])
    try:
        if command == "resolve" and not args:
            prs = resolve(gh)
            print(f"PRs to evaluate: {prs}")
            _append("GITHUB_OUTPUT", f"prs={json.dumps(prs)}\n")
            return 0
        if command == "evaluate" and len(args) == 1:
            evaluate(gh, parse_pr_number(args[0]))
            return 0
        if command == "error" and len(args) == 1:
            write_error(gh, parse_pr_number(args[0]), _writer(gh), _run_url(gh))
            return 0
    except InputError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    print(
        "usage: python3 -m ci_gate.evaluate {resolve | evaluate PR | error PR}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
