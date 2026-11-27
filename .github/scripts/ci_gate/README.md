# Heavy-CI gate (`ci_gate`)

One decision per PR and heavy-CI suite, written as a commit **Status** on the
PR's head commit by the **CI Evaluator** workflow (`ci-evaluator.yaml`).

| Suite | Workflow | Status context |
|---|---|---|
| `atom-test` | `atom-test.yaml` | `ATOM Test Status` |
| `atom-sglang-test` | `atom-sglang-test.yaml` | `ATOM SGLang Test Status` |
| `atom-vllm-test` | `atom-vllm-test.yaml` | `ATOM vLLM Test Status` |
| `atomesh-accuracy-validation` | `atomesh-accuracy-validation.yaml` | `ATOMesh Accuracy Status` |

## The rule (`decide.py`)

A Status is **success** only when a run that speaks for the PR's head proved
heavy CI on it, or the change needs no suite run. Two steps:

**1. Does the gate apply?** A closed PR or a draft gets no Status (drafts never
run heavy CI, labels included).

**2. The verdict**, in order:

| Status | When |
|---|---|
| pending `Waiting: base is <ref>` | the base is not `main` (a stacked PR) |
| pending `⛔ BLOCKED by unmerged #N` | the title or body declares a dependency on an open PR here |
| success `Not required for this change` | every changed path is known and none is one the suite tests |
| success `Passed (run …)` | a run speaking for the head proved heavy CI |
| pending `Passed, but this PR changes its own CI definition: …` | the PR changes the suite's run definition (below), so the run that passed ran the PR's own; and the PR has no approved review |
| failure `Failed (run …)` / `Smoke test failed` | the latest such run failed |
| pending `Running` / `Re-running` / `… push to start one` / `… approve the fork run` | authorized (approved, or a suite label), not proved yet |
| pending `Waiting for approval or a ci label` | not authorized |

Defined once and used everywhere:

- **A run speaks for the PR's head** (`speaks_for`) when it is a PR event
  (`pull_request`, `pull_request_review`) of the head commit and, if the base
  ever changed, created after the last change (GitHub's times are to the
  second: a run in that second may be against the old base). A manual
  dispatch, a schedule or a push run never does: it may use other inputs.
- **What a run proves** (`evidence`): its latest attempt **passed** heavy CI,
  **failed** it, or proves **nothing** (still running, waiting for a fork
  approval, cancelled, or heavy CI not run). A run of the pre-cutover workflow
  passed when its signal job (`Check Pre Checkin Signal`) and its gate job both
  succeeded; one whose signal job did not run proves nothing. Any passed run
  speaking for the head is the proof; otherwise the latest run that failed
  decides.
- **Changed paths** (`ChangedPaths`) are every path the PR adds, removes or
  renames from / to. When the file list is cut short, nothing is ruled out.
- **A suite is relevant to a change** when the change can alter what the
  suite's PR run executes (`suites.py`). Of the product code (`PRODUCT_DIRS`,
  the packages `pyproject.toml` declares) each suite takes what it tests:
  ATOM Test all of it but the plugins and ATOMesh, the plugin and ATOMesh
  suites `atom/plugin/**` and `atom/mesh/**`, the rest left to ATOM Test and
  the runs on `main`. Everything else a run executes or reads is its **run
  definition**, and which files that is cannot be proven from the
  repository's text (a run may compute a name), so the lists err on the side
  of running: every area a PR run may execute (`RUN_AREAS`: the packaging
  files, `scripts/`, `docker/`, the CI's scripts, actions and data) counts for
  every suite, with the suite's own workflow and the workflows its run reads;
  a suite leaves a path out only by a reviewed ignore entry. Other workflows,
  CPU unit tests, docs, tools, recipes and `atomrl/` start no suite.
  `tests/test_ci_gate_suites.py` holds this together: every file of the
  repository is classified once (product, run area, workflow, or not run, with
  why), and what `tests/ci_gate_relevance.py` can derive of each run (the
  files its PR-run steps name, followed through the scripts, Dockerfiles and
  actions it executes) must be covered by the suite's list and never classified
  as not run.
- **A PR changes its suite's CI definition** (`changes_ci_definition`) when it
  changes a path relevant to the suite outside `PRODUCT_DIRS`: the run that
  tested it then ran the PR's own definition, so its pass counts once the PR
  has an approved review.
- **PR references** (`refs.py`): `#N`, `owner/repo#N`, and the PR's URL; names
  compare case-insensitively. **Dependencies** (`deps.py`) are the references
  to this repository after `depends on`, `stacked on` or `blocked by`, to the
  end of that paragraph (or in the next one, when the paragraph ends at the
  keyword: `Depends on:` then a list), however decorated (`**Depends on** PR
  #12`, a link); every one is checked. The text is not read for meaning: "was
  stacked on #N" holds too, as does any reference in that paragraph; reword it
  and the edit re-evaluates. A number that is not an open PR here, or that
  declares this PR back, does not hold; the Status says so.

## When it runs

- `pull_request_target`: opened, pushed, reopened, edited (title, body, base),
  ready / back to draft, labelled, closed.
- `workflow_run`: a suite workflow run that speaks for a PR, or **CI Review
  Relay** (`ci-review-relay.yaml`, which forwards reviews) finished.
- every 10 minutes: the open non-draft PRs, whatever their base (one moved off
  `main` has a Status to correct), at most 20 a pass: PRs
  missing a Status first, newest first; the remaining slots rotate through
  the others by the pass's 10-minute slot, so every PR is reached however many
  there are, with no state kept between passes. Events can be lost, and a
  Status depends on what the PR list does not show (the PR's text, its runs,
  its dependencies), so any Status may be out of date: this pass is what
  makes it catch up. A pass's API calls (about 15 a PR) are a small part of
  the `GITHUB_TOKEN` hourly limit.
- by hand: **Run workflow** with a PR number (`3862` or `#3862`; ASCII digits
  only, no leading zero).

The workflow has two jobs. `resolve` turns the event into PR numbers;
`evaluate` runs once per PR (a matrix), in a concurrency group of that PR, so a
PR is never evaluated twice at once, a queued evaluation is replaced by a newer
one that reads the PR afresh, and a failure stays with its PR. A failed
evaluation writes `error` on its PR (never over a success); the next pass
retries. Both jobs run the gate code from the default branch and never check
out PR code.

`github_api.py` classifies every API call: a fact (absence is one: a 404, a
GraphQL `NOT_FOUND`), transient (network, 5xx, rate limits, a REST 403 / 429 or
a GraphQL `RATE_LIMITED` in a 200: retried, waiting as GitHub says), or an
error.

## Phase: shadow

The Statuses are marked `[preview]` and are **not required**. The re-runs,
cancels and dependency holds a decision asks for are only listed in the job
summary, next to the old gate's result (`… Gate`) for comparison. The old
`check_heavy_ci_gate.sh` still decides what runs.

Cutover, one suite at a time: the suite workflow gets a plan job
(`plan_for_executor`) and an aggregate job named by `result_job_name`, the
evaluator starts acting, and the `main` ruleset requires the suite's Status
in place of its `… Gate`. Before that, a required Status must be one nobody
else can write:

- Anyone with write access can post a Status with their own token, and any
  workflow a PR adds can with the default `GITHUB_TOKEN`, as GitHub Actions,
  the same source as this evaluator. So the evaluator must write as a GitHub
  App of its own, and the ruleset require the Status from that App.
- The repository's default workflow permissions go to read, and "Allow GitHub
  Actions to create and approve pull requests" off: else a PR's own workflow
  can approve it.
- `.github/CODEOWNERS` names its owners: until then no file has one, and any
  write user's approval authorizes heavy CI and a self-changed CI definition.

## Boundaries

What the gate does not decide, on purpose:

- The plugin and ATOMesh suites test their own code (`atom/plugin/**`,
  `atom/mesh/**`); a change to the rest of `atom` is ATOM Test's and the runs
  on `main`'s. The product code does not import the plugins (where it does,
  that import is the defect).
- Runs are matched to a PR by commit: another PR with the same head commit,
  based on a branch whose CI definition was changed, produces runs that count
  for this one (GitHub does not say which PR a fork's run belongs to, nor list
  a fork's or a closed PR's commits under it). It takes such a branch: one a
  write user creates (who until the App cutover can post a Status directly
  anyway), or an existing branch whose CI definition reports a pass it did
  not test.
- ATOM vLLM Test's PR run installs ATOM at the PR's head commit
  (`GITHUB_COMMIT_SHA`), not at its merge into `main` as the other suites'
  runs do; its pass does not cover a conflict with what `main` gained since
  the PR's base.
- A manual run is authorized by the dispatching user's role in the workflow
  file of the ref it runs; a write user can dispatch a branch whose file does
  not ask.

## Tests

`tests/test_ci_gate_*.py`. `tests/ci_gate_sim.py` is a protocol simulator: a
model of the GitHub behaviour the design relies on, with each behaviour it does
not rely on modelled both ways, checking the rule after every step of random
event sequences. A long pass: `python tests/ci_gate_sim.py 2000`; each of
`--buggy-aggregator`, `--no-merge-base-check`, `--no-time-rule`,
`--trust-manual-runs`, `--trust-incomplete-file-lists`,
`--trust-own-ci-definition` removes one defence and must produce violations.

## What it assumes of GitHub

Each checked against GitHub's documentation (docs) or live data (measured on
this repository and on throwaway branches, 2026-11):

| Fact | Relied on by | Source |
|---|---|---|
| Times (`created_at`, a base change's `createdAt`) are to the second | `speaks_for` (no bound without a base change; strict after one) | measured |
| A re-run keeps its run's `created_at`, `event` and commit | `speaks_for`; re-runs stay evidence | measured |
| A fork run awaiting approval is `completed` / `action_required` | `evidence` (proves nothing) | measured |
| `reviewDecision` is `APPROVED` exactly when an approval counts for `main` (open PRs: 0 of 100 differ) | `is_authorized`, `changes_ci_definition` | measured |
| A fork PR's review runs report the base repository; the commit identifies the code | `speaks_for` | measured |
| `pull_request_target` runs the default branch's workflow; `workflow_run` fires only for it | the jobs never run PR code | measured |
| Rate limits: REST 403 / 429 (`retry-after`, else `x-ratelimit-reset` at 0 remaining, else 1 min); GraphQL a 200 with an error (primary: `x-ratelimit-remaining` 0) or a 403 | `github_api.py` retries them | docs |
| `GITHUB_TOKEN`: 15,000 requests / points an hour a repository on Enterprise Cloud (ROCm is) | the scheduled pass's size | docs, measured plan |
| GraphQL `first` / `last` at most 100; a missing number is a `NOT_FOUND` error with a null field | page sizes; dependencies | docs, measured |
| Reading statuses and issues by number needs only `contents: read`, `pull-requests: read` here (a public repository) | the jobs' permissions | measured |
| At most 1,000 statuses per commit and context | a Status is written only when it changes | docs |
| The PR files list stops at 3,000 files | a cut-short list rules nothing out | docs |
| Search finds an open PR by its head commit (it may lag; the scheduled pass catches up) | `workflow_run` to PRs | measured |
| Events can be lost | the scheduled pass | assumed |

Local read-only run against live data: `CI_GATE_DRY_RUN=1` (decide and
report, write nothing).
