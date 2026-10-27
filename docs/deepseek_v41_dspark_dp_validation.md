# DSpark joint DP validation

The joint DP planner and online publication execute on both TP2×DP2 and
TP1×DP4. Real V4.1 TP2×DP2 execution also contracts verification graphs and
drains the requests. **Output equivalence is not established:** the dynamic
and full-width hashes differ, and repeating the unchanged full-width reference
also produces different hashes. This remains a draft validation item, not a
claim of DP correctness or performance parity.

## Scope

The production changes were exercised on base
`b968a353300cef0d39d0ee5f7747632a7d851948`, with four MI355X GPUs. AITER was not
modified. The real-model configuration is DeepSeek-V4.1-Flash, TP2×DP2 with DP
attention, no EP, `TR_HYBRID` level 0, memory utilization 0.7, five draft tokens,
and a maximum of four requests per rank.

SPS was measured with the real weights on this DP topology. STS temperatures
were reused from the earlier TP2 profile solely for this correctness probe;
they are not a DP confidence fit. The dynamic probe replaces confidence with
0.05 to exercise contraction, without changing draft tokens or target sampling.
Both comparison processes capture the same q=1/3/6 graph buckets. The reference
forces full-width verification. Each TP group has one scheduler, with production
prefill alignment, dummy forwards and separate scheduler/worker batch snapshots.

## Passed component checks

- 213 targeted CPU tests passed, two skipped. Coverage includes the exhaustive
  joint-prefix oracle, bonus floors, non-nested query divisors, topology-bound
  profiles, observe-only decisions and async-copy ownership.
- Four-process CPU components passed for TP2×DP2 and TP1×DP4.
- Four-GPU graph/event components passed for TP2×DP2 and TP1×DP4. Each exercises
  176 forwards: 144 decode, eight idle, eight mixed P/D and 16 request-boundary
  steps. Both count 168 group decode steps, accept 83 timing samples and publish
  21 versions; all four workers finish with identical cost models.
- Both GPU components execute graphs with 4/6/12/24 rows and perform 148
  contractions. Median host planning/collective cost was approximately 0.465 ms
  for TP2×DP2 and 0.647 ms for TP1×DP4 in this synthetic probe. These are overhead
  measurements, not model throughput gains.

## Real-model result and unresolved comparison

Full-width and dynamic runs each complete four greedy requests: 80 output tokens
on DP rank 0 and three requests of 128 tokens each on DP rank 1, 464 in total.
Prompts and lifetimes differ by rank; the run includes chunked prefill, request
reorder and idle-rank dummy execution. Production prefill alignment keeps these
model steps homogeneous; mixed P/D exclusion is covered by the component probe.
TP peers agree on their forward-output digest.

The dynamic run executes smaller common token graphs, including 18→9 and 12→6
rows. Its online cost state agrees across DP ranks. This verifies that the DP
path actually changes the executed graph, rather than only metadata.

However, final greedy output hashes differ for all four requests. First
differences between full-width and dynamic appear at token positions 13 on rank
0 and 22/3/3 on rank 1. A fresh repeat of the unchanged full-width reference
also differs for all four requests, first at positions 13 and 13/10/4. Thus the
current hash comparison cannot isolate a regression caused by dynamic planning.
The origin of the full-width variation has not been determined; it must not be
assumed to be harmless numerical noise. Establish a stable reference or explain
and validate the numerical variation before declaring output equivalence.

`tests.validate_dspark_dp_model --compare-with <reference.json>` makes a hash
mismatch an explicit failure. An execution-complete message alone is not an
equivalence assertion. `--diagnostic-prefixes` records prefix hashes for locating
the first difference without writing token values.

## Remaining validation limits

- Actual EP/MoRI model execution and multi-node execution were not tested.
- No DP SA throughput/latency run was performed. The earlier +3.26% TP2
  observation belongs to its earlier evaluated snapshot, not this DP patch.
- Whole-tree Black passes (1437 Python files), changed Python files pass Ruff,
  and `git diff --check` passes. Whole-tree Ruff still reports findings outside
  the changed Python files.
- The full native unit suite stops during collection with 37 errors in the
  existing validation image. Missing dependencies include `jax`, `jsonschema`
  and `aiter.ops.triton.fused_fp8_quant`; GPU-driver-dependent imports also fail
  when GPU visibility is disabled. The full suite is not green.

The new options remain disabled by default. See
[the DSpark guide](deepseek_v41_dspark.md) for configuration and test commands.
