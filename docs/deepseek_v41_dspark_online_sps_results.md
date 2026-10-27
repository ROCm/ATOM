# Online DSpark SPS: V4.1 SA comparison

This is a single sequential A/B measurement of the opt-in online SPS and
token-graph scheduling policy. It is evidence for this workload, not a guarantee
of a repeatable speedup or a saturated-throughput result.

## Configuration

- DeepSeek-V4.1-Flash on two MI355X GPUs, one TP2 instance; no PD, EP or TBO.
- `TR_HYBRID`, compilation level 0, graph q buckets `1,3,6`.
- GPU memory utilization 0.7, maximum sequences 128, prefill chunk 16384,
  checkpoint interval 8192; BF16 KV and FP4 index.
- SA workload: 393 traces, 16 lanes, seed 42, idle-gap cap 300 seconds.
- Fresh server for each arm with the same warmed JIT cache. Each arm has
  16-lane decode warmup followed by 49 SA warmup requests, then 900 seconds of
  submissions and up to 90 seconds of drain.
- Static baseline: calibrated SPS/STS, confidence scheduling and ragged verify.
- Treatment: the same configuration with `online_sps=true`,
  `bucket_global=true`, sampling every 8 eligible decode steps, publication every
  64 steps and a 5000-decode-step history.

Both arms were measured on base
`e3a00a9281ee0ce84a3cc0129fac28a974b17c9b` plus the evaluated changes. The PR is
rebased onto newer main; the following GPU measurements were not rerun on that
rebased tree. The shared calibration profile SHA256 was
`f098e4518253b292b75624ca7c5b1564f361560797f9b2117213d72bd1f03696`.

## Results

| Metric | Static | Online affine SPS + graph policy | Change |
| --- | ---: | ---: | ---: |
| Server output tokens/s/GPU, submission window | 230.097 | 237.594 | +3.26% |
| Server output tokens/s/GPU, including drain | 214.177 | 222.016 | +3.66% |
| Completed-client output tokens/s/GPU | 201.691 | 211.668 | +4.95% |
| Mean request TPOT, ms/token | 12.278 | 12.255 | -0.19% |
| Mean TTFT, ms | 225.998 | 228.245 | +0.99% |
| Completed / cancelled / errors | 412 / 3 / 0 | 427 / 1 / 0 | |

Server output counts include tokens produced for unfinished or eventually
cancelled requests, before final scheduler/client stop filtering. Count one TP
rank once and divide by two GPUs; summing rank outputs would double-count.
Counters are sampled at approximately five-second intervals. The submission
windows cover 905.056 and 905.077 seconds, with 416502 and 430082 output tokens;
rates use those actual durations. The drain windows cover 995.060 and 995.082
seconds, with 426237 and 441849 output tokens.

## Mechanism and limits

The affine estimator keeps fixed serving overhead separate from additional
marginal graph-row cost. In this run, fixed offsets finished at about
4.77-5.17 ms. There were too few comparable graph-size changes to support an
additional slope, so none was added to the measured startup curve. This is not
a measurement that marginal cost is exactly zero.

The treatment used a smaller executed graph in only 10 of 27350 decode steps
(0.0047% reduction in graph rows relative to its own full-width layouts).
The static baseline also rarely shrank: 44 of 28198 steps (0.0192% fewer rows).
Consequently the throughput result must not be described as a speedup from
more aggressive graph shrinking.

For 3416 timing observations, mean target time was 24.721 ms and mean post-target
time was 4.406 ms. Predicted total time averaged 29.011 ms, with 0.388 ms mean
absolute error. TP publication cost averaged 17.692 microseconds per decode
step. Timing starts after `prepare_model`; preparation, host scheduling and
output transfer costs are not included in this estimate.

Output tokens per sequence-return increased from 3.009 to 3.115, while
sequence-returns per second changed from 152.927 to 152.569. These counts explain
the total-output difference algebraically, but are not acceptance probabilities
or a controlled comparison of identical requests at identical KV/batch states.
Closed-loop replay changes arrival times, cache hits, request mix and completion
boundaries. Across 410 common completed requests, mean TPOT changed by only
-0.0021 ms/token. The experiment therefore supports a positive observed output
result, not a causal or repeatable 3.26% decode-speedup claim.

## Validation of the measured implementation

- 303 CPU tests passed across estimator, runtime, graph-budget policy,
  calibration/capture tools and V4.1 runtime contracts.
- Two-rank real-GPU timing/publication validation passed.
- Nine full-model dynamic lifecycle scenarios passed; all observed decode steps
  used token graphs and both TP ranks agreed on the published cost state.
- Final history retained 624 observations from the latest 5000 decode steps;
  no estimator rejects or pending-event capacity skips were recorded.

Both feature flags remain disabled by default. See
[the DSpark guide](deepseek_v41_dspark.md) for calibration and configuration.
