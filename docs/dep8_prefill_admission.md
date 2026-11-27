# DEP8 prefill admission and prefix-owner routing

Opt-in scheduler, delayer and DP-routing features of the DeepSeek-V4.1 DEP8
(DP attention + EP) AgentX runs. Every switch defaults off: with none set, the
scheduler, the prefill delayer and the router do what they did before (the
existing scheduler / routing tests are unchanged and pass, and each feature has
a default-off equivalence test). Existing session ownership and explicit-rank
routing stay authoritative.

All variables are listed in [environment_variables.md](environment_variables.md).

| Feature | Switch (default off) | What it does |
|---|---|---|
| Short requests first | `ATOM_PREFILL_SHORT_FIRST_ROWS` (+ `_LONG_ROWS`, `_MAX_SKIPS`) | Inside a prefill step the DP group has agreed on, a rank admits requests with few new tokens left (after prefix reuse) before longer prompts. The long prompts already in progress keep `_LONG_ROWS` tokens of the step; a request passed over `_MAX_SKIPS` times (and a preemption victim) goes ahead of the short ones for good. Per rank: the delayer's decision, the cost target and the long-prefill cap are untouched. Not with a KV connector. |
| Busy-phase long-prefill cap | `ATOM_PREFILL_LONG_CAP_BUSY` (+ `_SEQS`) | A lower per-prompt chunk cap while the DP group decodes at least `_SEQS` sequences (the delayer's reduced count, the same on every rank). |
| Busy decode window | `ATOM_PREFILL_DECODE_INTERVAL_BUSY` (+ `_SEQS`) | After a prefill-only step with a busy group, the decode-protection window is this long instead of `ATOM_PREFILL_DECODE_INTERVAL`. |
| Integer cost ceiling | `ATOM_PREFILL_COST_CEIL` | A rank's prefill cost rides the delayer's reduce rounded up, so the cost target does not cut the costliest rank's chunk one token short. |
| Confirmed prefix-owner routing | `ATOM_DP_SESSION_PREFIX_OWNER_TOKENS` (+ `_MAX_EXTRA_REQS`) | With `ATOM_DP_SESSION_AFFINITY`, a new session goes to the rank holding the longest confirmed prefix of its prompt when that is at least the given tokens more than the lightest rank holds, and that rank has at most `_MAX_EXTRA_REQS` more requests in flight. Needs `--enable_prefix_caching` and `ATOM_FRONTEND_PREFIX_HASH=1`. Counters: `atom:dp_affinity_new_prefix_owner`, `atom:dp_affinity_prefix_owner_gained_tokens`, `atom:dp_affinity_prefix_owner_refused{reason}`. |

Prefix credit requires every ancestor interval to be confirmed (the prefill that
produced it finished); a descendant cannot cross a missing or provisional
ancestor. Temporary cost-limit overrides are restored in `finally`, including on
exceptions.

## Tests

`tests/test_prefill_short_first.py`, `tests/test_dp_prefix_owner.py` and
`tests/test_prefill_admission_takeover.py` run with the existing scheduler and
routing tests (no GPU). Set `GPU3_ADMISSION_BASE` to a separate current-main
source tree to run the optional exact baseline trace comparison; without it
those comparisons skip visibly.

## Provenance

Content of ROCm/ATOM#3644 as carried by the plugin-host PR (ROCm/ATOM#3761),
adapted to current main. The DEP8 AgentX recipes of coderfeli/plugin_poc
(`dep8_c224_eplb`, `dep8_base`) switch these features on through their
environment; ATOM itself keeps all of them off.

Whole-stack context only: the interval-96 DEP8 c256 experiment reached 229.11
tok/s/user and 213385.82 total tok/s/GPU (AgentX 1.0.14, 3600 s, native valid,
two tail cancellations). That is not an isolated attribution of these switches
and not performance evidence for this rebase.
