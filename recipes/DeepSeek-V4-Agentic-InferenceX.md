# DeepSeek-V4-Pro agentic on ATOM — the exact commands

For anyone reproducing ATOM's numbers on the InferenceX AgentX MVP scenario.
Two commands, both copied from a CI run rather than written by hand: the server
args come from `.github/benchmark/models.json`, and the client line is what
AIPerf itself echoed as `CLI Command:` in the run below.

- Hardware: MI355X ×8, single node, no PD disaggregation
- Model: `deepseek-ai/DeepSeek-V4-Pro`, FP4 weights, FP8 KV, FP4 index cache
- Scenario: `inferencex-agentx-mvp`, dataset `semianalysis_cc_traces_weka_062126`
- Reference run: ATOM Benchmark [run 33074134043], 3600 s per cell, 9/9 green

Two variants: TP-only for the low-concurrency band, and DP-attention for the
high one. They differ by `--enable-dp-attention` plus four routing env vars,
and nothing else.

## Server — TP (concurrency 1–48)

```bash
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1

python3 -m atom.entrypoints.openai_server \
  --model $MODEL_PATH --server-port $PORT \
  -tp 8 --kv_cache_dtype fp8 --index_cache_dtype fp4 \
  --method mtp --num-speculative-tokens 3 \
  --spec-decode-acceptance-rate 0.4966666667 \
  --enable_prefix_caching \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL \
  --max-num-seqs $(( CONC * 2 ))
```

## Server — DP attention (concurrency 48–256)

```bash
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_DP_SESSION_AFFINITY=1
export ATOM_DP_LB_REQ_EQUIV=512
export ATOM_ENABLE_PREFILL_DELAYER=1
export ATOM_PREFILL_DECODE_INTERVAL=10

python3 -m atom.entrypoints.openai_server \
  --model $MODEL_PATH --server-port $PORT \
  -tp 8 --kv_cache_dtype fp8 --index_cache_dtype fp4 \
  --method mtp --num-speculative-tokens 3 \
  --spec-decode-acceptance-rate 0.4966666667 \
  --enable_prefix_caching \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL \
  --enable-dp-attention \
  --max-num-seqs $(( CONC * 2 ))
```

Against the TP command this adds `--enable-dp-attention` and the four
`ATOM_DP_*` variables; everything else is identical.

`ATOM_DP_SESSION_AFFINITY` is not optional here. Without it a conversation's
turns land on different DP ranks, so the prefix KV written by one turn sits on
a rank the next turn never reaches — and an agentic trace is nothing but
multi-turn sessions, so the whole workload degrades to cold prefill.

## Client

```bash
aiperf profile --scenario inferencex-agentx-mvp \
  --url http://localhost:$PORT --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model $MODEL_PATH --tokenizer $MODEL_PATH --tokenizer-trust-remote-code \
  --concurrency $CONC --benchmark-duration 3600 --stats-interval 30 \
  --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --server-metrics http://localhost:$PORT/metrics \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir ./aiperf-artifacts-c$CONC
```

### One deliberate difference from `benchmark_lib.sh`

`build_replay_cmd` passes `--warmup-grace-period ${AGENTIC_WARMUP_GRACE_PERIOD:-1800}`.
We pass **`--agentic-warmup-grace-period`** instead, because AIPerf's own help
says the plain flag is inert here:

> The agentic warmup is synthesized from the profiling phase rather than a
> user-declared warmup phase, so it does NOT honor `--warmup-grace-period`
> (which requires `--warmup-duration`).

An agentic run never sets `--warmup-duration`, so the plain flag has no effect
and the barrier falls back to waiting indefinitely. The variable name in
`benchmark_lib.sh` is already `AGENTIC_WARMUP_GRACE_PERIOD`, which suggests the
agentic knob was the intent. Worth a look on your side — if the plain flag is
right after all, we will match it.

### AIPerf version

Both sides run the SemiAnalysis fork at 0.12.0. Our images track its default
branch; the CI client pins `754356e9`, the same commit the InferenceX
`utils/aiperf` submodule points at, so the two agree.

## Measured

8 chips, 3600 s per cell. `per_chip` and `P90 intvty` follow
`utils/agentic/aggregation/request_metrics.py`: `(ΣISL + ΣOSL) / duration /
num_gpus`, and `1 / p90(ITL)`.

| mode | conc | tok/s/chip | P90 intvty | ITL p90 | TTFT avg | cache hit |
|---|---|---|---|---|---|---|
| TP | 1 | 1,482 | 127.2 | 7.9 ms | 1.3 s | 95.9% |
| TP | 2 | 1,620 | 115.9 | 8.6 ms | 0.7 s | 94.4% |
| TP | 8 | 4,883 | 78.0 | 12.8 ms | 0.6 s | 95.9% |
| TP | 16 | 9,049 | 55.7 | 18.0 ms | 0.7 s | 95.5% |
| TP | 48 | 20,308 | 27.7 | 36.1 ms | 1.1 s | 95.7% |
| DPA | 64 | 21,888 | 28.6 | 34.9 ms | 10.5 s | 94.5% |
| DPA | 128 | 30,709 | 16.3 | 61.5 ms | 10.9 s | 93.4% |
| DPA | 256 | 44,722 | 10.2 | 97.6 ms | 13.4 s | 93.4% |

A c=32 cell also ran but its replay stopped at 834 s rather than 3600 s, so it
is left out — the rates were in line, the sample was not.

Two things to read carefully in that table. `tok/s/chip` counts input tokens,
and 93–96% of them are prefix-cache hits, so it is dominated by cache reads
rather than compute — at c=256, roughly 2,900 of the 44,722 tok/s/chip actually
went through prefill. And the TTFT step from ~1 s (TP) to ~10 s (DPA) is not a
concurrency effect: TP holds ~1 s all the way from c=1 to c=48. DP buys
throughput by queueing.

## Related

- [`DeepSeek-V4-Agentic-Benchmark.md`](DeepSeek-V4-Agentic-Benchmark.md) —
  earlier cross-engine head-to-head (vLLM / SGLang / B200). Its ATOM commands
  predate MTP, the FP4 index cache and `--cudagraph-mode FULL`; this file is the
  current one.
- `.github/benchmark/models.json` — the catalog these commands are generated
  from, so they cannot drift from what CI runs.
