# DEP8 prefill admission and prefix-owner routing

Opt-in integration of the runtime changes used by the DEP8 interval96 experiment:
short requests first inside an agreed prefill step, bounded long-request aging,
reserved progress for long partials, integer cost-ceil, optional busy-phase limits,
and new-session affinity to a confirmed cached prefix with a load guard.

All new knobs default off. Existing session ownership and explicit-rank routing
remain authoritative. Prefix credit requires every ancestor interval to be
confirmed; a descendant cannot cross a missing/provisional ancestor. Cost-limit
temporary overrides are restored in finally, including exceptions.

Source: plugin_poc#112 at231a480, adapted to current ATOM main rather than replacing
the files. Current-main scheduler differential, blocked ticks, skip aging,
uneven DP rank decisions, prefix ownership and load guard are tested. The
experiment kept interval96 in all phases, not the source PR's64/70.

The three new test files run with the existing scheduler/routing tests.
Set GPU3_ADMISSION_BASE to a separate current-main source tree to run the optional
exact baseline trace comparison; without it those comparisons skip visibly.

Whole-stack context only: final pinned experiment229.11 tok/s/user and
213385.82 total tok/s/GPU, AgentX1.0.14, 3600s, nativevalid; two tail cancellations.
Raw point `/root/runs_night/v_sa_r4_all_i96_h1014/dep8-c256-20261120-112420`.
Not isolated runtime attribution, not performance evidence for this rebase.
Rebased full-server validation and same-version repeated A/B remain pending.

## Main compatibility update (2026-11-27)

Merged main `c6bfb0ad280ee23e5edb2bda78bf8c0323093eef` and carried back the scheduler/routing
conflict resolutions from #3761 (`de423cb1`, test follow-up `7f875a13`), without
the plugin-host or Engram changes:

- Prefix-owner load guards use `load_reqs`; only the shared-pool migration
  path supplies `rank_reqs`. Main's positional `rank_reqs`/`margin` API is retained.
- The load-adaptive decode interval arms after prefill-only steps, not mixed steps.
- Short-first partials receive the current mixed-step prefill budget, including
  the zero-budget guard. Late partial admission keeps the decode-page reservation.

CPU regression selection: 1,082 passed / 10 skipped across 26 test files.
The skips are nine optional pinned-baseline comparisons and one upstream
connector test whose import path needs updating. Black and Ruff pass for all
13 Python files in this PR. Tests additionally reject three deliberately
reintroduced conflict-resolution mistakes (then restore the sources).
The full suite is **not** a pass: candidate and the same main snapshot encounter
the same 91 collection-error files in the isolated ROCm environment, including
missing optional dependencies and GPU-stack imports. No new GPU, full-server
accuracy or performance validation was performed for this compatibility update.
