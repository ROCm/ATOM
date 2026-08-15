# DSpark Usage Guide

**DSpark** is a semi-autoregressive *block* drafter for speculative decoding.
Unlike serial MTP (which drafts `k` tokens over `k` sequential passes), DSpark
drafts a whole block in a **single parallel backbone pass** (parallel backbone +
Markov sequential head + confidence head), then the target model **verifies** the
block. A per-request **confidence head** predicts how many drafted tokens are
worth verifying, so each request can verify a different length and the freed
batch capacity lifts throughput.

ATOM supports two DSpark flavors, which differ only in **where the draft lives**:

| Target | Draft | How it's loaded |
|---|---|---|
| **DeepSeek-V4-Pro** | in-checkpoint (`mtp.*` namespace) | auto-detected via the `dspark_block_size` config field — no `--draft-model` |
| **Kimi-K3** | separate checkpoint | passed explicitly with `--draft-model` |

Everything downstream of loading — the parallel block draft, confidence-scheduled
ragged verify, and `--dspark-config` knobs — is shared between the two.

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below will be executed inside the container.

## Launching server

### DeepSeek-V4-Pro — FP8 on 8xMI355X GPUs (TP8 + FP8 KV Cache + DSpark)

The V4 draft ships inside the [DeepSeek-V4-Pro-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
checkpoint (`mtp.*` namespace), so no `--draft-model` is needed:

```bash
python -m atom.entrypoints.openai_server \
  --model /data/DeepSeek-V4-Pro-DSpark \
  --tensor-parallel-size 8 \
  --kv_cache_dtype fp8 \
  --method dspark \
  --num-speculative-tokens 7 \
  --trust-remote-code \
  --server-port 7777 \
  --torch-profiler-dir ./log \
  --cudagraph-mode PIECEWISE \
  --enable-dp-attention \
  --dspark-config '{"confidence_schedule": true, "ragged": true, "ragged_graph_sizes": "8"}'
```

### Kimi-K3 — FP8 on 8xMI355X GPUs (TP8 + FP8 KV Cache + DSpark)

Kimi-K3 (`kimi_linear`: KDA linear-attention + MLA full-attention + MXFP4 MoE)
uses a **separate** DSpark draft checkpoint, so the draft is passed with
`--draft-model`. The target is [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
and the draft is [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark):

```bash
python -m atom.entrypoints.openai_server \
  --model moonshotai/Kimi-K3 \
  --draft-model Inferact/Kimi-K3-DSpark \
  --tensor-parallel-size 8 \
  --kv_cache_dtype fp8 \
  --method dspark \
  --num-speculative-tokens 7 \
  --trust-remote-code \
  --gpu-memory-utilization 0.93 \
  --block-size 128 \
  --no-enable_prefix_caching \
  --server-port 7777
```

Notes for Kimi-K3:
- The draft is a standalone MLA model. Its 576-wide latent is the same shape the
  target's own full-attention layers cache, so the draft's 5 layers bind into
  the target's KV pool as extra rows and inherit `--kv_cache_dtype`.
- `--draft-model` accepts a local path or an HF repo id (downloaded on startup).
- The confidence-scheduled ragged verify (`--dspark-config`) also applies here;
  omit it to run the plain batch-uniform verify.

### `--dspark-config` knobs

DSpark runtime knobs are passed as a single JSON dict via `--dspark-config`
(dynamic config, à la vLLM `--speculative-config`). It is resolved once in the
parent process and pickled into every engine-core worker (see `DSparkConfig` in
`atom/config.py`).

| Key | Type | Meaning |
|---|---|---|
| `confidence_schedule` | bool | Use the DSpark confidence head to pick a per-request verify length `ell_r` (paper Algorithm 1) + variable-length verification. **Prerequisite** for the ragged scheduler. |
| `ragged` | bool | Per-request ragged verify (paper §5.2 avoid-padding): each decode seq forwards its own `ell_r+1` tokens, no batch-level padding to a single `q`. |
| `ragged_graph_sizes` | str | Comma-separated per-seq CUDA-graph query-length buckets to capture for the ragged path, e.g. `"1,3,6"` or `"8"`. Smaller buckets are what actually free dense/MoE compute; a single full bucket (`mtp_k+1`) only saves attention. |
| `q_buckets` | str | CUDA-graph query-length buckets for the older batch-uniform q-bucket verify path (independent of the ragged path). |
| `disable_sps_calib` | bool | Skip SPS calibration (replays captured graphs at warmup) and use the synthetic SPS stub. |

Tips on server configuration:
- **`--num-speculative-tokens 7`** sets the draft block; the max verify length is
  `mtp_k+1 = 8` (`full_q`). Per-request scheduling verifies `1..8` per seq.
- **`ragged_graph_sizes`**: `"8"` == the full bucket, so graph capacity never
  shrinks (only attention saves via the `-1` marker bail). To actually free
  dense/MoE compute, capture smaller buckets, e.g. `"1,3,6,8"` or `"2,4,8"`.
- **No env vars**: DSpark is configured purely through `--dspark-config`,
  parsed once into a `DSparkConfig` object (`atom/config.py`) and carried on
  `Config.dspark` into every worker. The old `ATOM_DSPARK_*` env vars have been
  removed.
- Do **not** pass `--enforce-eager` with the ragged CUDA-graph path — ragged
  replays captured `(bs, q_eff)` graphs. Eager also works for correctness checks.
- Clear compile cache before restarting after code changes: `rm -rf /root/.cache/atom/*`

## Performance baseline (DeepSeek-V4-Pro)

The numbers in this section were measured on **DeepSeek-V4-Pro DSpark**. Kimi-K3
DSpark uses the same launch/benchmark flow (swap `--model`/`--draft-model`); its
latest acceptance and accuracy are tracked on the dashboard linked at the end.

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
  --model /data/DeepSeek-V4-Pro-DSpark --backend=vllm --base-url=http://localhost:7777 \
  --dataset-name=random \
  --random-input-len=${ISL} --random-output-len=${OSL} \
  --random-range-ratio=1.0 \
  --num-prompts=$(( $CONC * 10 )) \
  --max-concurrency=$CONC \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

Performance on 8xMI355X GPUs with the following environment:
- Date measured: 2026-07-21.
- Docker image: rocm/atom:latest.
- `--kv_cache_dtype fp8`, `--method dspark --num-speculative-tokens 7`,
  `--cudagraph-mode PIECEWISE`, `--enable-dp-attention`.
- DSpark config: `confidence_schedule=true, ragged=true, ragged_graph_sizes="8"`.

### FP8 (TP8, FP8 KV Cache) — DSpark (confidence-scheduled ragged verify)

Mixed-length serving run (random dataset, avg ISL ≈ 7387, avg OSL ≈ 922):

```
============ Serving Benchmark Result ============
Successful requests:                     1280
Benchmark duration (s):                  394.18
Total input tokens:                      9454961
Total generated tokens:                  1181330
Request throughput (req/s):              3.25
Output token throughput (tok/s):         2996.97
Total Token throughput (tok/s):          26983.66
---------------Time to First Token----------------
Mean TTFT (ms):                          4426.86
Median TTFT (ms):                        3455.09
P99 TTFT (ms):                           16575.08
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          35.52
Median TPOT (ms):                        31.09
P99 TPOT (ms):                           90.13
---------------Inter-token Latency----------------
Mean ITL (ms):                           189.56
Median ITL (ms):                         73.00
P99 ITL (ms):                            2129.34
==================================================
```

```
============ Serving Benchmark Result ============
Successful requests:                     2560
Benchmark duration (s):                  741.58
Total input tokens:                      18877524
Total generated tokens:                  2366401
Request throughput (req/s):              3.45
Output token throughput (tok/s):         3191.01
Total Token throughput (tok/s):          28646.71
---------------Time to First Token----------------
Mean TTFT (ms):                          4324.45
Median TTFT (ms):                        2106.55
P99 TTFT (ms):                           37325.13
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          68.29
Median TPOT (ms):                        63.71
P99 TPOT (ms):                           163.37
---------------Inter-token Latency----------------
Mean ITL (ms):                           293.17
Median ITL (ms):                         124.32
P99 ITL (ms):                            1830.16
==================================================
```

### Acceptance

DSpark drafts a block of `mtp_k = 7` tokens per step; the target then verifies
it. On the mixed-length run above:

- **Acceptance rate: 63.6%** — accepted draft tokens / total drafted tokens.
- **Mean accepted length: 5.45 tokens/forward** — 1 verified token + ~4.45
  accepted draft tokens, i.e. each target forward emits ~5.4x the tokens of
  non-speculative decode.

### Accuracy (GSM8K, 3-shot)

Lossless speculative decoding — verify always emits the target-greedy token, so
DSpark matches the plain target's accuracy:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|↑  |0.9469|±  |0.0062|
|     |       |strict-match    |     3|exact_match|↑  |0.9500|±  |0.0060|
```

The numbers above are a snapshot. For the latest data tracked across commits, see
[rocm.github.io/ATOM/benchmark-dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/).
