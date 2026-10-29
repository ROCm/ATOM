# validate-atom-pr

Deterministic base-vs-head pytest comparison for one ATOM PR and one pytest target. This is
the evidence source behind review-pr's `(deterministic)` validation line — see
`../review-pr/fetch.sh` (auto-invocation around lines 400-485) and `../review-pr/triage.py`
(`VALIDATION_REPORT_NAME`, the `(deterministic)`/`(advisory)` vocabulary).

## What it does

1. Verifies identity: records `repo.base` (`git rev-parse HEAD` in `--repo` before the
   patch), echoes `repo.head` (the caller-supplied `--head-sha`, not independently
   re-derived), and `repo.patch_sha256` — exactly what fetch.sh's own gate cross-checks
   against the PR it is reviewing.
2. Claims a GPU honestly: picks an index via `torch.cuda.device_count()`, reads arch from
   `torch.cuda.get_device_properties().gcnArchName` (rocm-smi's product-name lookup is
   broken on this box — "Error when calling libdrm" — so this validator doesn't depend on
   it), and records `idleness_basis: "activity"` when `rocm-smi --showuse --json` works or
   `"unknown"` when it doesn't, rather than inventing a classification.
3. Runs `--target` once at base (before the patch), applies the patch (`merge_sim`), checks
   the target still collects at least one item at head (`test_policy` — checked *after* the
   patch, not before, so a target the PR itself adds is judged on its head state, not on a
   base where it cannot exist), runs `--target` again at head, then reverts the patch.
4. Compares base and head pass/fail and writes a `report_schema.json`-valid report with a
   verdict.

## What it deliberately does not do

- **No performance scraping.** review-pr/fetch.sh has its own perf fallback recipe; this
  validator's report has no `perf` stage.
- **No independent shape grid.** `correctness_s1_grid` is permanently `"skip"` with a
  stated reason — base/head single-target comparison only.
- **No runtime route/kernel-symbol receipt.** `execution_receipt` is permanently `"skip"` —
  this validator does not instrument which kernel route actually ran.
- **No index-width scan.** `index_width_scan` is permanently `"skip"` — fetch.sh already
  runs `scan_index_width.py` separately; this does not duplicate it.

Every stage above is still present in the report, `"skip"` with a reason, never omitted —
`report_schema.json` requires all 9 stage keys so a minimal report can never overclaim by
leaving one out.

## Verdict table (the whole point)

| base   | head   | verdict    | meaning                                            |
|--------|--------|------------|-----------------------------------------------------|
| pass   | pass   | PASS       | —                                                    |
| pass   | fail   | BLOCK      | this PR introduced the failure                       |
| fail   | fail   | NEEDS_WORK | pre-existing, not this PR's fault                    |
| fail   | pass   | PASS       | the PR fixes a pre-existing failure (noted in findings) |

A target that does not exist at base at all (the PR adds it) is `base_missing`: there is no
baseline to run, so `baseline_control` is `"skip"`, and a failing new test at head is judged
BLOCK, not NEEDS_WORK — nothing "pre-exists" for a file that didn't exist.

Any stage before both runs complete (no GPU claimable, `import atom` doesn't resolve inside
`--repo`, the patch doesn't apply, or `--target` collects nothing at head) means base and
head were never both actually run: that is `INCONCLUSIVE`, a fifth state, not a sixth cell
of the table.

Exit code: `0` PASS, `2` INCONCLUSIVE, `1` otherwise (BLOCK/NEEDS_WORK) — this always holds,
and a report is always written on every path, including BLOCK, so fetch.sh's gate always has
something to read.

## How it gets invoked, and why it does not run itself

`review-pr`'s `fetch.sh` calls this validator automatically only when
`REVIEW_AUTO_VALIDATE=1`. The workflow sets it to `'0'`, and so does aiter's: the
comment on both lines reads `kernel validation off by default (self-hosted + public
repo)`.

That default is a security decision, not an oversight. ROCm/ATOM is public, anyone can
open a PR, and validating one means executing its code on a self-hosted box. The trigger
allowlist bounds this -- a review only starts when one of the 88 logins in
`authorized.txt` asks for it -- so the exposure is not "any stranger runs code here" but
"a maintainer triggering a review on a stranger's PR runs that stranger's code". That is
the same hazard as any CI that builds PR branches on self-hosted runners, and it is
worth deciding deliberately rather than inheriting.

**The capability being off is not the same as the gap being silent.** When triage decides
a PR needs validation and none ran, `fetch.sh` tells the worker, in these words, to
`Report it as a gap in the evidence`, and hands it the reason. It distinguishes three
cases that collapse into one unhelpful sentence if you let them:

- not applicable -- no runtime surface changed, so `a static-only review is complete
  here, not deficient`;
- required but not run -- with the recorded reason, which for the default case is
  `auto-validation is disabled (REVIEW_AUTO_VALIDATE=0)`;
- required, not run, and the reason was not recorded -- `which is itself a defect in
  this step`.

So the reviewer reading the card is told when this PR is one that wanted validation. The
reason is archived in `auto_validation_outcome.txt` alongside the report, because
months later "validation was not applicable" and "validation was required and skipped"
are different facts about a PR and the card alone does not separate them.

**To run it for one review**, set the flag for that invocation rather than changing the
default:

```bash
REVIEW_AUTO_VALIDATE=1 bash .claude/skills/review-pr/run_one.sh <PR> ROCm/ATOM
```

Turning it on repo-wide means deciding the untrusted-code question first -- restricting
it to PRs from known authors, or running it somewhere disposable. Flip the workflow's
`REVIEW_AUTO_VALIDATE` only after that, and record which answer you chose.

## CLI

```
validate_pr.sh --repo <worktree> --patch <diff> --head-sha <sha> --target <pytest target> \
                --label <str> --out <report.json>
                [--expected-route ...] [--shape-vars ...] [--shape-env ...] \
                [--shape-arg ...] [--shape-argnames ...] [--grid ...]
```

The passthrough flags are accepted (never a hard CLI error) because fetch.sh's auto-run
conditionally supplies them, but this validator does no route/grid instrumentation with
them — if given, they are threaded into `test_selection` as caller-asserted hints, not
claims this validator verified itself.

## Files

- `validate_pr.sh` — the doing: git worktree ops, GPU claim, pytest invocations.
- `report_lib.py` — the reporting: pure, side-effect-free report assembly and the verdict
  table, unit-testable with no GPU/pytest/network.
- `report_schema.json` — the contract, ported from ROCm/aiter's `validate-kernel-pr` and
  trimmed to this validator's smaller scope (see its own `description` fields for exactly
  what was relaxed and why).
- `tests/` — hermetic tests (no network, no real GPU, no untracked-file dependence) for
  `report_lib.py`'s verdict table and schema validity.
