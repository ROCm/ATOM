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
