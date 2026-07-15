# DeepSeek-V4-Pro DSpark Usage Guide

[DeepSeek-V4-Pro-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) adds
**DSpark** — a semi-autoregressive *block* drafter — on top of the DeepSeek-V4-Pro
backbone. Unlike serial MTP (which drafts `k` tokens over `k` sequential passes),
DSpark drafts a whole block in a **single parallel backbone pass** (parallel
backbone + Markov sequential head + confidence head), then the target model
**verifies** the block. A per-request **confidence head** predicts how many
drafted tokens are worth verifying, so each request can verify a different
length and the freed batch capacity lifts throughput. DSpark ships inside the V4
checkpoint under the same `mtp.*` namespace and is detected by the
`dspark_block_size` config field.

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below will be executed inside the container.

## Launching server

### FP8 on 8xMI355X GPUs (TP8 + FP8 KV Cache + DSpark)

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

For a benchmark / production run, drop `--dspark-debug` (diagnostics force host
syncs and cost throughput). Add it back only when tuning:

```bash
  --dspark-config '{"confidence_schedule": true, "ragged": true, "ragged_graph_sizes": "8"}' \
  --dspark-debug
```

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

Separate flag:

| Flag | Meaning |
|---|---|
| `--dspark-debug` | Enable scheduler diagnostics (avg verify length / truncation rate / anchor OOB checks). **Forces host syncs; keep OFF in perf runs.** |

Tips on server configuration:
- **`--num-speculative-tokens 7`** sets the draft block; the max verify length is
  `mtp_k+1 = 8` (`full_q`). Per-request scheduling verifies `1..8` per seq.
- **`ragged_graph_sizes`**: `"8"` == the full bucket, so graph capacity never
  shrinks (only attention saves via the `-1` marker bail). To actually free
  dense/MoE compute, capture smaller buckets, e.g. `"1,3,6,8"` or `"2,4,8"`.
- **No env vars**: DSpark is configured purely through `--dspark-config` /
  `--dspark-debug`, parsed once into a `DSparkConfig` object (`atom/config.py`)
  and carried on `Config.dspark` into every worker. The old `ATOM_DSPARK_*` env
  vars have been removed.
- Do **not** pass `--enforce-eager` with the ragged CUDA-graph path — ragged
  replays captured `(bs, q_eff)` graphs. Eager also works for correctness checks.
- Clear compile cache before restarting after code changes: `rm -rf /root/.cache/atom/*`

## Performance baseline

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
- Date measured: 2026-07-15.
- Docker image: rocm/atom:latest.
- ATOM: `feat/dspark-spec-decode` branch (commit a1d51a73).
- `--kv_cache_dtype fp8`, `--method dspark --num-speculative-tokens 7`,
  `--cudagraph-mode PIECEWISE`, `--enable-dp-attention`.
- DSpark config: `confidence_schedule=true, ragged=true, ragged_graph_sizes="8"`.

### FP8 (TP8, FP8 KV Cache) — DSpark (confidence-scheduled ragged verify)

Mixed-length serving run (random dataset, avg ISL ≈ 7387, avg OSL ≈ 922):

Throughput:

| Requests | Duration (s) | Input tok | Output tok | Req/s | Output tok/s | Total tok/s |
| -------- | ------------ | --------- | ---------- | ----- | ------------ | ----------- |
| 1280     | 456.26       | 9,454,961 | 1,180,719  | 2.81  | 2587.83      | 23310.65    |

Latency (ms):

| Metric | Mean | Median | P99 |
| ------ | ---- | ------ | ---- |
| TTFT   | 3208.86 | 2058.98 | 18162.68 |
| TPOT   | 39.79   | 36.29   | 101.90   |
| ITL    | 176.43  | 97.30   | 1641.30  |

The numbers above are a snapshot. For the latest data tracked across commits, see
[rocm.github.io/ATOM/benchmark-dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/).
