#!/usr/bin/env python3
"""Say so when a review ended without saying anything.

Everything that reports a failed review runs inside the review job: run_one.sh writes the status
and _notify.py posts it. `if: always()` covers a step that failed, but not a job that stopped
existing -- if the self-hosted runner loses contact mid-review the job is marked failed and not
one of its steps runs, so nothing is written and nothing is posted. From the PR it looks exactly
like a review that is still going, forever.

This runs on hosted infra after the review job, whatever became of it, and speaks only when the
review job ended badly AND left no sign that it had already reported. It is the same idea as the
watchdog, moved to the other end: the watchdog covers a review that never started, this covers
one that started and then disappeared.

What lands here is "the job said nothing and the cause is in the log" -- a triage job, not
necessarily a box problem -- so aiter's workflow pages the FLOW owner here, not the runner
owner it pages from the watchdog. ATOM's workflow injection for OWNER_OVERRIDE was removed along
with the repo-variable channel (owners.txt is now the only map), so this script's fallback is
the only owner that is ever paged here; nothing currently populates OWNER_OVERRIDE for any of
the three scripts that read it.

Usage (from the report job): python3 .claude/skills/review-pr/lost_review.py
Reads REVIEW_RESULT, NOTIFIED, PR, RUN_URL, GH_TOKEN, ATOM_BOT_TOKEN, OWNER_OVERRIDE and the
GITHUB_* that Actions sets. OWNER_OVERRIDE is read but nothing in the workflow sets it, so this
always falls back to the hardcoded owner below.
"""

import os
import sys

from watchdog import GitHub      # sibling; sys.path[0] is this script's directory


def body(result, owner, run_url):
    # Says what is known and no more. This fires for a runner that lost contact mid-review, for
    # a review that ran fine and then failed to publish its card, and for a notice that could not
    # be posted -- naming one of them as the cause would send the reader to the wrong place.
    return (
        "\u26a0\ufe0f **rocm-atom-bot** \u2014 the review job ended as `%s` and nothing was posted "
        "for this PR: no review card, and no failure report either.\n\n"
        "Everything that reports a failure runs inside that job, so a job that stops existing, "
        "or one whose posting step is what failed, cannot report itself. The run log says which "
        "\u2014 a runner that lost contact mid-review and a card that could not be published "
        "both end up here. Nothing is wrong with this PR, and nothing on it was changed.\n\n"
        "@%s \u2014 the run below has the cause.%s\n\n"
        "<sub>Re-comment `atom-bot review` to try again.</sub>"
        % (result, owner, ("\n\nRun: " + run_url) if run_url else ""))


def run(api, env):
    result = (env.get("REVIEW_RESULT") or "").strip()
    # The review job reports its own failures; only speak when it could not.
    if (env.get("NOTIFIED") or "").strip().lower() == "true":
        print("::notice::the review job already reported; staying quiet")
        return 0
    if result in ("", "success", "skipped"):
        print("::notice::review result %r needs no report" % result)
        return 0

    owner = (env.get("OWNER_OVERRIDE") or "").strip() or "zufayu"
    status = api.comment(env["GITHUB_REPOSITORY"], env["PR"],
                         body(result, owner, (env.get("RUN_URL") or "").strip()))
    if not 200 <= status < 300:
        print("::warning::could not post the lost-review notice: %s%s" % (
            status, " -- %s" % api.last_error_detail if api.last_error_detail else ""))
    print("::error::the review job ended as %s with nothing reported" % result)
    return 1


def main():
    api = GitHub(os.environ["GH_TOKEN"], os.environ.get("ATOM_BOT_TOKEN", ""))
    return run(api, os.environ)


if __name__ == "__main__":
    sys.exit(main())
