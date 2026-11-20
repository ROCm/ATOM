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
