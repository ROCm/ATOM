# ATOM / DeepSeek-V4.1-Flash Agentic Benchmark

**Update:** The [four-GPU supplement](#d-four-gpu-supplement--october-29-2026) adds TP4 1c, TP4 4c and DPA4 32c, including SA results, validation and recorded commands. Sections 1–7 and Appendices A–C below describe the original two-GPU batch.

Test batch: the evening of October 28 through the early morning of October 29, 2026. All displayed times are **Beijing time (UTC+8)**. The measurement sending windows span **10-28 23:43:06 to 10-29 03:53:44**. Report prepared on October 29, 2026.

The batch completed **10 points**: plain TP2 at concurrency 1, 2, 4, 8, 16, 32, 64 and 128, plus DPA at concurrency 64 and 128. Each point used **2 GPUs and a 3600-second measurement sending window**. In total, **54,730** requests were issued, **54,686** completed successfully, **0** were reported as request errors, and **44** were canceled during post-window draining. DPA 256c is outside this report's scope.

## 1. Complete results using SA / ATOMesh metric definitions

The table provides the data for **Total Throughput per Chip vs E2E Normalized Interactivity**. Both interactivity metrics are in **tok/s/user**. Input, output and total throughput are in **tok/s/GPU (tok/s/chip)**. Higher values are better. Throughput is normalized by the 2 GPUs assigned to each point; interactivity is not divided by the GPU count.

| Mode | Concurrency | E2E Normalized Interactivity | Interactivity | Total Throughput / GPU | Input Throughput / GPU | Output Throughput / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP2 | 1 | 318.01 | 498.54 | 14,919.23 | 14,828.35 | 90.88 |
| TP2 | 2 | 260.85 | 429.86 | 14,353.44 | 14,252.74 | 100.69 |
| TP2 | 4 | 221.54 | 333.00 | 18,391.42 | 18,264.40 | 127.02 |
| TP2 | 8 | 230.34 | 278.18 | 32,021.25 | 31,794.55 | 226.71 |
| TP2 | 16 | 155.30 | 186.35 | 57,745.71 | 57,377.87 | 367.84 |
| TP2 | 32 | 83.29 | 99.18 | 111,018.47 | 110,370.47 | 648.01 |
| TP2 | 64 | 50.03 | 57.06 | 173,290.89 | 172,037.64 | 1,253.25 |
| TP2 | 128 | 17.54 | 18.63 | 173,431.05 | 171,922.95 | 1,508.10 |
| DPA | 64 | 44.73 | 70.89 | 180,627.57 | 179,322.05 | 1,305.52 |
| DPA | 128 | 24.49 | 39.94 | 238,834.99 | 236,828.16 | 2,006.83 |

**Total throughput counts both input and output tokens, including input tokens served from the prefix cache.** It is therefore distinct from output-generation throughput and from uncached prefill compute throughput.

## 2. Latency details

Each cell lists **p50 / p90 / p95**. TTFT and decode ITL are in **milliseconds (ms)**; whole-request latency (`request_latency`) is in **seconds (s)**. ITL uses AIPerf's `inter_token_latency`, not `full_response_*`. This table preserves the exported latency percentiles. The normalized interactivity metric is calculated per request and cannot be reconstructed by combining these aggregate percentiles.

| Mode | Concurrency | TTFT p50 / p90 / p95 (ms) | ITL p50 / p90 / p95 (ms) | Request latency p50 / p90 / p95 (s) |
| --- | ---: | ---: | ---: | ---: |
| TP2 | 1 | 137.45 / 274.42 / 541.69 | 1.781 / 2.006 / 2.267 | 2.060 / 9.064 / 11.808 |
| TP2 | 2 | 130.89 / 322.43 / 764.79 | 1.798 / 2.326 / 2.843 | 1.268 / 9.750 / 12.615 |
| TP2 | 4 | 113.93 / 435.05 / 1,118.70 | 1.825 / 3.003 / 3.680 | 1.116 / 8.793 / 12.522 |
| TP2 | 8 | 113.82 / 238.08 / 426.25 | 2.297 / 3.595 / 4.034 | 1.305 / 7.633 / 12.092 |
| TP2 | 16 | 120.72 / 286.11 / 499.38 | 3.157 / 5.366 / 6.017 | 1.409 / 7.981 / 13.058 |
| TP2 | 32 | 155.93 / 444.79 / 948.56 | 5.601 / 10.082 / 12.267 | 2.800 / 16.675 / 27.839 |
| TP2 | 64 | 232.75 / 628.32 / 1,045.77 | 12.298 / 17.525 / 20.479 | 6.132 / 28.285 / 47.007 |
| TP2 | 128 | 360.54 / 1,301.29 / 2,648.76 | 31.338 / 53.666 / 59.617 | 15.577 / 78.077 / 125.488 |
| DPA | 64 | 640.02 / 2,140.22 / 3,824.42 | 10.101 / 14.106 / 15.928 | 5.923 / 27.343 / 43.506 |
| DPA | 128 | 1,131.35 / 5,221.10 / 8,223.34 | 17.833 / 25.039 / 27.163 | 10.743 / 45.721 / 73.311 |

## 3. Request accounting and validity

Drain cancellations are requests still unfinished after the 3600-second sending window and its grace period. They are counted separately from request-processing errors. Zero request errors does not mean that every issued request completed successfully. TTFT/ITL coverage refers to **time coverage of the measurement window**, not the fraction of requests with a metric.

| Mode | Concurrency | Issued | Successful | Request errors | Drain cancellations | Interactivity-eligible requests | TTFT / ITL time coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP2 | 1 | 328 | 328 | 0 | 0 | 326 | 99.6395% / 99.8062% |
| TP2 | 2 | 465 | 465 | 0 | 0 | 464 | 99.8278% / 99.8443% |
| TP2 | 4 | 699 | 699 | 0 | 0 | 697 | 99.8601% / 99.8643% |
| TP2 | 8 | 1,416 | 1,416 | 0 | 0 | 1,414 | 99.8322% / 100.0000% |
| TP2 | 16 | 2,673 | 2,673 | 0 | 0 | 2,670 | 99.9956% / 100.0000% |
| TP2 | 32 | 4,546 | 4,546 | 0 | 0 | 4,544 | 99.9820% / 100.0000% |
| TP2 | 64 | 9,266 | 9,260 | 0 | 6 | 9,258 | 100.0000% / 100.0000% |
| TP2 | 128 | 11,384 | 11,360 | 0 | 24 | 11,355 | 99.9886% / 100.0000% |
| DPA | 64 | 9,573 | 9,570 | 0 | 3 | 9,568 | 100.0000% / 100.0000% |
| DPA | 128 | 14,380 | 14,369 | 0 | 11 | 14,364 | 100.0000% / 100.0000% |
| Total | — | 54,730 | 54,686 | 0 | 44 | 54,660 | Every point ≥95% |

Both interactivity definitions use the same **54,660** eligible requests. Another **26** successful requests produced exactly one token and have no ITL. The original script excludes these from both interactivity metrics but retains their tokens in successful-request throughput totals. Warmup and failed/canceled requests are excluded from the main results.

All points were checked for the full measurement window, configured concurrency, AgentX scenario, `submission_valid`, request accounting, metric time coverage, fixed versions and actual launch configuration. Every client exited with code 0. TP2 64/128c and DPA 64/128c retain explicit drain-cancellation annotations.

### Measurement times and throughput denominators

Times below are Beijing time (UTC+8). The sending window excludes server startup, model loading, graph capture, warmup, draining and export.

| Point | GPU IDs | Sending started | Sending ended | Sending duration (s) | SA request span (s) | AIPerf export denominator (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tp2-c1 | 4,5 | 10-29 02:47:42 | 10-29 03:47:42 | 3,600.000 | 3,593.018448 | 3,600.000097 |
| tp2-c2 | 2,3 | 10-28 23:43:06 | 10-29 00:43:06 | 3,600.000 | 3,594.390514 | 3,599.999846 |
| tp2-c4 | 4,5 | 10-28 23:44:02 | 10-29 00:44:02 | 3,600.000 | 3,595.109421 | 3,600.000303 |
| tp2-c8 | 6,7 | 10-28 23:47:54 | 10-29 00:47:54 | 3,600.001 | 3,602.005440 | 3,602.012413 |
| tp2-c16 | 0,1 | 10-29 02:46:51 | 10-29 03:46:51 | 3,600.001 | 3,603.482736 | 3,603.493941 |
| tp2-c32 | 2,3 | 10-29 00:58:36 | 10-29 01:58:36 | 3,600.001 | 3,611.930926 | 3,611.942581 |
| tp2-c64 | 4,5 | 10-29 01:04:48 | 10-29 02:04:48 | 3,600.001 | 3,625.789018 | 3,930.001218 |
| tp2-c128 | 6,7 | 10-29 01:13:49 | 10-29 02:13:49 | 3,600.003 | 3,627.805261 | 3,930.000381 |
| dpa-c64 | 0,1 | 10-29 01:11:24 | 10-29 02:11:24 | 3,600.002 | 3,619.167432 | 3,930.001767 |
| dpa-c128 | 2,3 | 10-29 02:53:44 | 10-29 03:53:44 | 3,600.002 | 3,629.010177 | 3,930.000433 |

The four points with drain cancellations waited for a 30-second grace period and then a 300-second cancellation-drain timeout before that drain phase was forced to end. Their AIPerf export denominators are consequently about 3930 seconds. The main table uses the span from the first successful request's start to the last successful request's end (approximately 3593–3629 seconds). It does not force every denominator to 3600 seconds or count canceled requests as successful.

## 4. Metric definitions and calculation

The main table was computed by directly calling `agentic_interactivity()` and `agentic_throughput()` in the unchanged saved copy of `.github/scripts/atomesh/interactivity.py`. Both use each point's original **per-request `profile_export.jsonl`**. Interactivity was not reconstructed from aggregate percentiles in `profile_export_aiperf.json`.

```text
ITL_i = inter_token_latency_i / 1000                 # s/token
TTFT_i = time_to_first_token_i / 1000                # s
OSL_i = output_sequence_length_i                    # token
eff_i = ITL_i + TTFT_i / OSL_i                       # s/token

Interactivity = 1 / p90(ITL_i)                        # tok/s/user
E2E Normalized Interactivity = 1 / p90(eff_i)          # tok/s/user

T = (max(request_end_ns) - min(request_start_ns)) / 1e9
Total Throughput per Chip = (sum(ISL) + sum(OSL)) / T / 2
Input Throughput per Chip = sum(ISL) / T / 2
Output Throughput per Chip = sum(OSL) / T / 2
```

- Percentiles are computed by linearly interpolating per-request **latencies**, then taking the reciprocal, matching NumPy's default `percentile(..., method="linear")`. No p10 of per-request rates is substituted.
- Interactivity filter: profiling phase, no error, OSL > 0, and both ITL and TTFT present. Both definitions use the same eligible requests.
- Throughput filter: the original script's successful profiling records, without requiring ITL. The script also accepts records with no phase field, but this batch contains no such ambiguous records.
- Interactivity is a per-user rate, not divided again by concurrency or GPU count. Throughput is divided by the 2 GPUs used by each point.

Calculation script SHA256: `85c0767c2a4f02544424a8cbe73902f276cbba7d613d1b40be2dd14d207f8cb3`. All ten raw JSONL SHA256 values were verified against the collected evidence.

## 5. Fixed test environment

| Item | Recorded configuration |
| --- | --- |
| Model | deepseek-ai/DeepSeek-V4.1-Flash; 48 safetensors shards; runtime path `/scratch/DeepSeek-V4.1-Flash` |
| Model quantization | `quant_method=fp8`, `expert_dtype=fp4`, `weight_block_size=[32,32]`, `scale_fmt=ue8m0`; model dtype bfloat16 |
| GPUs | 8× AMD gfx950, approximately 288 GiB VRAM per GPU; 2 GPUs per test instance |
| Server | Container-visible system model MiTAC G8825Z5U2BC-350X-755 |
| ATOM | 0.1.7.dev213+gc8f7f0583 |
| AITER | 0.1.25.dev570+gbc7011fb0 |
| PyTorch | 2.10.0+rocm7.2.4.lw.git3d3aa833 |
| ROCm | 7.2.4, as identified by this run's PyTorch environment |
| FlyDSL | 0.4.0 |
| Transformers | 5.16.1 |
| Client | agentx-harness 1.0.12 / Python 3.12; standard AgentX agentx scenario |
| Dataset | semianalysis_cc_traces_weka_062126; 393 traces; seed 42 |
| Warmup | 10 additional requests per lane after snapshot primers; excluded from the measurement window |
| Scheduling | Up to four GPU pairs concurrently; at most one DPA instance at a time; a fresh server for each point, without cross-point prefix-cache reuse |

ATOM commit: [`c8f7f0583c07e5d3cf77ab01a44f2fce860e7835`](https://github.com/ROCm/ATOM/commit/c8f7f0583c07e5d3cf77ab01a44f2fce860e7835).

AITER commit: [`bc7011fb079b7101711a6dd5c8823a18999dbe48`](https://github.com/ROCm/aiter/commit/bc7011fb079b7101711a6dd5c8823a18999dbe48).

CK submodule: `96afae1d54fc79c26857de7e1683b0dde1ff3c6f`. These were the upstream main revisions selected when restoring this test environment and were held fixed throughout the measurements. Both ATOM and AITER working trees were clean at completion.

Dataset revision: `23f152f6f0f9399a85901b89a6458def0ef16729`. Dataset SHA256: `29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`. Model reference revision: `314113f8a7dadb388215396ba5fbee2c610a04d6`. The provider's model copy was checked for shard headers, index contents and byte lengths; full-file SHA256 validation of the approximately 510 GB weights was not performed.

## 6. Server and client configuration

The server used the classic Python OpenAI frontend, `python3 -m atom.entrypoints.openai_server`, with `USE_ATOMESH_ENTRYPOINTS=0`. The main shared server settings are listed below; full recorded commands are in Appendix C.

| Parameter | Value |
| --- | --- |
| Tensor parallelism | `-tp 2` |
| KV cache / index cache | `--kv_cache_dtype bf16 --index-cache-dtype fp4` |
| GPU memory fraction | `--gpu-memory-utilization 0.9` |
| Prefill / batched tokens | `--attn-prefill-chunk-size 16384 --max-num-batched-tokens 16384` |
| Maximum sequences | 512, the default used in this run |
| Prefix cache | `--enable_prefix_caching` |
| Block / state checkpoint | `--block-size 16 --state-checkpoint-interval-tokens 8192` |
| Graph | `--level 3 --cudagraph-mode FULL` |
| Graph capture sizes | `[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,48,64,128]` |
| Speculative decode | `--method dspark --num-speculative-tokens 5 --spec-decode-acceptance-length 3.51` |
| Tool parser | `--tool-call-parser dsml_v41` |
| HTTP keepalive | `--timeout-keep-alive 900` |
| TBO / KV offload / expert parallelism | All disabled |

**Synthetic acceptance length 3.51 was enabled** (`ATOM_DSV41_BENCHMARK_SYNTHETIC=1`). These are performance measurements, not model accuracy, actual acceptance-rate or output-quality evaluations.

| Mode-specific setting | Plain TP2 | DPA |
| --- | --- | --- |
| Additional server flags | None | `--enable-dp-attention --dp-load-balance least_tokens` |
| ATOM_DP_SESSION_AFFINITY | `0` | `1` |
| ATOM_DP_DISPATCH_DEPTH | `0` | `0` |
| ATOM_PREFILL_DECODE_INTERVAL | `0` | `10` |
| ATOM_DP_LB_REQ_EQUIV | Not explicitly set | `512` |
| ATOM_ENABLE_PREFILL_DELAYER | Not explicitly set | `1` |
| GPU_MAX_HW_QUEUES | Not explicitly set | `5` |
| Client session header | Disabled | `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1` |

Shared environment settings included `OMP_NUM_THREADS=4`, `ATOM_NUMA_BIND=0`, `ATOM_DISABLE_MMAP=true`, `AITER_LOG_LEVEL=WARNING` and `AITER_MAX_JOBS=8`. Each group used a separate GPU pair, HTTP/internal/DP ports and logs.

Each client used `agentx profile`, the point's `--concurrency`, `--benchmark-duration 3600`, and `--public-dataset semianalysis_cc_traces_weka_062126`, with tokenizer trust-remote-code and server metrics enabled. The tokenizer came from the provider's model copy (`provider-copy-20261028`). Client settings included `AIPERF_HTTP_TCP_USER_TIMEOUT=900000`, `AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300` and `AIPERF_REQUIRED_SERVER_METRIC_PREFIX=atom:`. Model, tokenizer and dataset files were already cached.

## 7. SA reference and measurement scope

Launch parameters were based on [SemiAnalysis InferenceX run 40362093275](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40362093275?pr=3965), reference commit `dc77ba6147548643ca198ce78ca7cc8e03462545`, recipe `inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/dsv41flash/atom/mi355x-fp4-mtp/agentic.yaml`. The main table follows the local ATOMesh script's SA metric definitions.

- This run used the classic Python frontend, while the SA reference used Atomesh. Client Python versions were 3.12 and 3.11 respectively. The software, frontend and hardware environments are not identical to the reference CI environment.
- DPA includes session affinity, least_tokens load balancing and prefill interval 10; plain TP2 uses interval 0. These settings describe the complete mode-specific recipes.
- Points could share host CPU, memory and I/O with other GPU-pair tests; no exclusive-node cross-over rerun was performed.
- Agentic request lengths vary with the traces. Different concurrency points complete different requests and numbers of traces within the fixed duration; total throughput includes full input-token accounting and cache reuse.
- Four high-concurrency points have documented drain cancellations. Canceled requests do not contribute to successful-request throughput or interactivity.
- Only the ten fixed-version points with complete measurement windows and reviewed evidence are included. Interrupted runs from the previous container, earlier DPA runs with request errors, and DPA 256c are excluded.

## Appendix A. Unmodified AIPerf throughput exports

These values are the original AIPerf exports for the **entire two-GPU instance**, before dividing by 2. Their time accounting differs from Section 1. The difference between raw-request SA aggregation and the original AIPerf export is a metric-definition difference, not a change in engine performance.

| Mode | Concurrency | Total throughput (tok/s, 2 GPUs) | Output throughput (tok/s, 2 GPUs) | Request throughput (requests/s) |
| --- | ---: | ---: | ---: | ---: |
| TP2 | 1 | 29,780.60 | 181.42 | 0.0911 |
| TP2 | 2 | 28,662.14 | 201.07 | 0.1292 |
| TP2 | 4 | 36,732.87 | 253.69 | 0.1942 |
| TP2 | 8 | 64,042.38 | 453.41 | 0.3931 |
| TP2 | 16 | 115,491.06 | 735.69 | 0.7418 |
| TP2 | 32 | 222,036.23 | 1,296.01 | 1.2586 |
| TP2 | 64 | 319,753.70 | 2,312.48 | 2.3562 |
| TP2 | 128 | 320,190.34 | 2,784.27 | 2.8906 |
| DPA | 64 | 332,682.50 | 2,404.52 | 2.4351 |
| DPA | 128 | 441,086.26 | 3,706.26 | 3.6562 |

## Appendix B. Retained data and evidence

- `sa-interactivity-metrics.json`: full-precision main-table values, both interactivity definitions, eligible-request counts, filter audits and raw JSONL SHA256 values.
- `profile_export.jsonl`: all per-request records; all ten files were collected and verified.
- `profile_export_aiperf.json`: unmodified aggregate exports; timeslices are also retained.
- `reviewed-results.json`: evidence for measurement windows, request accounting, exit codes, scenario, coverage and actual configuration.
- Per-point server/client launch arguments, environments, logs and complete compressed archives are retained.

Displayed values are rounded; full precision and per-request records are retained in the source result files.

## Appendix C. Recorded commands and AgentX / AIPerf methodology

This run used **agentx-harness 1.0.12** (`aiperf_version=1.0.12` in the exports). AgentX is an AIPerf-based entry point. For flag-based `agentx profile` commands without a scenario/config override, it automatically inserts `--scenario agentx`. When invoking `aiperf` from the same virtual environment directly, specify that scenario explicitly.

### C.1 Profiling methodology

| Item | Setting |
| --- | --- |
| Endpoint | OpenAI-compatible streaming chat; server token counts enabled |
| Replay | `scenario=agentx`, `timing_mode=agentic_replay`; multi-turn and branched trace replay |
| Concurrency | Target session-tree/lane concurrency, set by `--concurrency`; not a requests/s target |
| Dataset / seed | `semianalysis_cc_traces_weka_062126` / 393 traces / seed 42 |
| Trajectory start | `trajectory_start_min_ratio=0.25`, `trajectory_start_max_ratio=0.75` |
| Idle gap caps | `trace_idle_gap_cap_seconds=300`, `system_idle_gap_cap_seconds=10` |
| Warmup | Snapshot primers followed by 10 additional requests per lane; accelerated warmup uses max_tokens=1; warmup grace=1800s |
| Measurement | 3600s per point; excludes startup, loading, graph capture, warmup, drain and export |
| Drain | Profiling grace=30s; cancellation-drain timeout=300s |
| Scenario defaults | `cache_bust=first_turn_prefix`, `extra_inputs.ignore_eos=true` |
| Exports | Per-request `profile_export.jsonl`; aggregate metrics, 1s timeslices, server metrics |

### C.2 Server commands (recorded 64c runs)

The following Bash commands are reconstructed from each point's recorded `launch.json` argv and environment. `env -u` clears offload/TBO, GPU visibility and mode-specific variables before applying the recorded values. These are historical configuration examples; use free GPUs/ports and a fresh output directory for a new run. Run the server and its corresponding client in separate terminals, starting the client after the server is ready.

**Plain TP2, 64c: GPUs 4,5; HTTP port 8102.**

```bash
env \
  -u ATOM_KV_OFFLOAD \
  -u ATOM_KV_OFFLOAD_EXTRA_CONFIG \
  -u ROCR_VISIBLE_DEVICES \
  -u CUDA_VISIBLE_DEVICES \
  -u ATOM_BUNDLE_WORK \
  -u GPU_MAX_HW_QUEUES \
  -u ATOM_DP_LB_REQ_EQUIV \
  -u ATOM_ENABLE_PREFILL_DELAYER \
  OMP_NUM_THREADS=4 \
  ATOM_NUMA_BIND=0 \
  ATOM_DISABLE_MMAP=true \
  AITER_LOG_LEVEL=WARNING \
  BENCHMARK_PRECISION=fp4 \
  PRECISION=fp4 \
  BENCHMARK_MODEL_KEY=dsv41flash \
  ATOM_DSV41_BENCHMARK_SYNTHETIC=1 \
  ATOM_DP_DISPATCH_DEPTH=0 \
  ATOM_DP_SESSION_AFFINITY=0 \
  ATOM_PREFILL_DECODE_INTERVAL=0 \
  HIP_VISIBLE_DEVICES=4,5 \
  ATOM_DP_BASE_PORT=26000 \
  ATOM_DP_MASTER_PORT=26500 \
  USE_ATOMESH_ENTRYPOINTS=0 \
  AITER_MAX_JOBS=8 \
  /opt/venv/bin/python3 -m atom.entrypoints.openai_server \
  --model /scratch/DeepSeek-V4.1-Flash \
  --host 127.0.0.1 \
  --server-port 8102 \
  --port 9200 \
  --kv_cache_dtype bf16 \
  --trust-remote-code \
  --index-cache-dtype fp4 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 \
  --attn-prefill-chunk-size 16384 \
  --enable_prefix_caching \
  --block-size 16 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 \
  --cudagraph-mode FULL \
  --method dspark \
  --num-speculative-tokens 5 \
  --spec-decode-acceptance-length 3.51 \
  --tool-call-parser dsml_v41 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,48,64,128]' \
  -tp 2 \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --timeout-keep-alive 900
```

**DPA, 64c: GPUs 0,1; HTTP port 8100.**

```bash
env \
  -u ATOM_KV_OFFLOAD \
  -u ATOM_KV_OFFLOAD_EXTRA_CONFIG \
  -u ROCR_VISIBLE_DEVICES \
  -u CUDA_VISIBLE_DEVICES \
  -u ATOM_BUNDLE_WORK \
  -u GPU_MAX_HW_QUEUES \
  -u ATOM_DP_LB_REQ_EQUIV \
  -u ATOM_ENABLE_PREFILL_DELAYER \
  OMP_NUM_THREADS=4 \
  ATOM_NUMA_BIND=0 \
  ATOM_DISABLE_MMAP=true \
  AITER_LOG_LEVEL=WARNING \
  BENCHMARK_PRECISION=fp4 \
  PRECISION=fp4 \
  BENCHMARK_MODEL_KEY=dsv41flash \
  ATOM_DSV41_BENCHMARK_SYNTHETIC=1 \
  ATOM_DP_DISPATCH_DEPTH=0 \
  ATOM_DP_SESSION_AFFINITY=1 \
  ATOM_DP_LB_REQ_EQUIV=512 \
  ATOM_ENABLE_PREFILL_DELAYER=1 \
  ATOM_PREFILL_DECODE_INTERVAL=10 \
  GPU_MAX_HW_QUEUES=5 \
  HIP_VISIBLE_DEVICES=0,1 \
  ATOM_DP_BASE_PORT=24000 \
  ATOM_DP_MASTER_PORT=24500 \
  USE_ATOMESH_ENTRYPOINTS=0 \
  AITER_MAX_JOBS=8 \
  /opt/venv/bin/python3 -m atom.entrypoints.openai_server \
  --model /scratch/DeepSeek-V4.1-Flash \
  --host 127.0.0.1 \
  --server-port 8100 \
  --port 9000 \
  --kv_cache_dtype bf16 \
  --trust-remote-code \
  --index-cache-dtype fp4 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 \
  --attn-prefill-chunk-size 16384 \
  --enable_prefix_caching \
  --block-size 16 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 \
  --cudagraph-mode FULL \
  --method dspark \
  --num-speculative-tokens 5 \
  --spec-decode-acceptance-length 3.51 \
  --tool-call-parser dsml_v41 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,48,64,128]' \
  -tp 2 \
  --enable-dp-attention \
  --dp-load-balance least_tokens \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --timeout-keep-alive 900
```

### C.3 Recorded AgentX client commands

**Plain TP2, 64c:**

```bash
env \
  HF_HOME=/scratch/atom-agentic-20261028/hf-cache \
  HF_DATASETS_CACHE=/scratch/atom-agentic-20261028/hf-cache/datasets \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  AIPERF_HTTP_TCP_USER_TIMEOUT=900000 \
  AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300 \
  AIPERF_REQUIRED_SERVER_METRIC_PREFIX=atom: \
  AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=0 \
  TOKENIZERS_PARALLELISM=false \
  /scratch/atom-agentic-20261028/agentx-venv/bin/agentx profile \
  --url http://127.0.0.1:8102 \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer-revision provider-copy-20261028 \
  --concurrency 64 \
  --benchmark-duration 3600 \
  --tokenizer-trust-remote-code \
  --server-metrics http://127.0.0.1:8102/metrics \
  --output-artifact-dir /scratch/atom-agentic-20261028/recovered-20261028-2235/points/tp2-c64/aiperf_artifacts \
  --public-dataset semianalysis_cc_traces_weka_062126
```

**DPA, 64c:**

```bash
env \
  HF_HOME=/scratch/atom-agentic-20261028/hf-cache \
  HF_DATASETS_CACHE=/scratch/atom-agentic-20261028/hf-cache/datasets \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  AIPERF_HTTP_TCP_USER_TIMEOUT=900000 \
  AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300 \
  AIPERF_REQUIRED_SERVER_METRIC_PREFIX=atom: \
  AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1 \
  TOKENIZERS_PARALLELISM=false \
  /scratch/atom-agentic-20261028/agentx-venv/bin/agentx profile \
  --url http://127.0.0.1:8100 \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer-revision provider-copy-20261028 \
  --concurrency 64 \
  --benchmark-duration 3600 \
  --tokenizer-trust-remote-code \
  --server-metrics http://127.0.0.1:8100/metrics \
  --output-artifact-dir /scratch/atom-agentic-20261028/recovered-20261028-2235/points/dpa-c64/aiperf_artifacts \
  --public-dataset semianalysis_cc_traces_weka_062126
```

For another point, keep the same mode-specific settings and change `--concurrency`, server GPUs/ports, client URL/metrics URL, and `--output-artifact-dir`. The measured plain TP2 points are 1/2/4/8/16/32/64/128; DPA points are 64/128. Actual GPU assignments and times are listed in Section 3.

### C.4 Invoking AIPerf directly

The full AIPerf command below is equivalent to the DPA64 AgentX command: it uses the same environment and arguments, with explicit `--scenario agentx`. The recorded measurement used the AgentX entry point in the preceding section. Choose one entry point when reproducing the test.

```bash
env \
  HF_HOME=/scratch/atom-agentic-20261028/hf-cache \
  HF_DATASETS_CACHE=/scratch/atom-agentic-20261028/hf-cache/datasets \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  AIPERF_HTTP_TCP_USER_TIMEOUT=900000 \
  AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300 \
  AIPERF_REQUIRED_SERVER_METRIC_PREFIX=atom: \
  AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1 \
  TOKENIZERS_PARALLELISM=false \
  /scratch/atom-agentic-20261028/agentx-venv/bin/aiperf profile \
  --scenario agentx \
  --url http://127.0.0.1:8100 \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer-revision provider-copy-20261028 \
  --concurrency 64 \
  --benchmark-duration 3600 \
  --tokenizer-trust-remote-code \
  --server-metrics http://127.0.0.1:8100/metrics \
  --output-artifact-dir /scratch/atom-agentic-20261028/recovered-20261028-2235/points/dpa-c64/aiperf_artifacts \
  --public-dataset semianalysis_cc_traces_weka_062126
```

### C.5 Computing SA metrics from per-request JSONL

Run from the ATOM checkout root containing the version of `.github/scripts/atomesh/interactivity.py` identified in Section 4. The example uses the original DPA64 JSONL; substitute any measured point's file. The first command reports both interactivity definitions; the second also reports per-GPU throughput.

```bash
RECORDS=/scratch/atom-agentic-20261028/recovered-20261028-2235/points/dpa-c64/aiperf_artifacts/profile_export.jsonl
python3 .github/scripts/atomesh/interactivity.py "$RECORDS" --json

python3 - "$RECORDS" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path('.github/scripts/atomesh').resolve()))
from interactivity import agentic_interactivity, agentic_throughput

record_path = Path(sys.argv[1])
interactivity = agentic_interactivity(record_path)
throughput = agentic_throughput(record_path)
num_gpus = 2
print(json.dumps({
    'e2e_normalized_interactivity_tok_s_user': interactivity['value'],
    'interactivity_tok_s_user': interactivity['itl_value'],
    'interactivity_requests': interactivity['n_requests'],
    'total_tok_s_gpu': throughput['total_tput_tps'] / num_gpus,
    'input_tok_s_gpu': throughput['input_tput_tps'] / num_gpus,
    'output_tok_s_gpu': throughput['output_tput_tps'] / num_gpus,
    'request_span_seconds': throughput['duration_s'],
}, indent=2))
PY
```

Filtering and percentile rules are specified in Section 4: calculate per-request latency, take its linearly interpolated p90, then invert. These commands do not reconstruct the metrics from aggregate JSON and do not use `full_response_*`.

## D. Four-GPU supplement — October 29, 2026

This supplement adds **TP4 1c, TP4 4c and DPA4 32c**, each with **4 GPUs and a 3600-second formal sending window**. Across these three points, **5,810 requests** were issued, **5,807 succeeded**, **0 had request errors**, and **3 were canceled during post-window draining**. Every scenario and submission validation passed, all clients exited with code 0, and no run was marked canceled. The earlier interrupted TP4 4c attempt is excluded.

The same pinned ATOM/AITER revisions, model copy, AgentX/AIPerf 1.0.12, dataset revision and synthetic acceptance recipe from Sections 5–7 were retained. ATOM remained at `c8f7f0583c07e5d3cf77ab01a44f2fce860e7835` and AITER at `bc7011fb079b7101711a6dd5c8823a18999dbe48`. TBO, KV offload and expert parallel were disabled. All accepted four-GPU points ran sequentially on GPUs 0–3; a persistent TP2 service occupied GPUs 4–5 on the same host during these measurements. Host resources were shared.

### D.1 SA results

**X-axis: E2E Normalized Interactivity (tok/s/user). Y-axis: Total Throughput per Chip (tok/s/GPU).** The alternative interactivity column is decode-only interactivity. Throughput is divided by **4** for this supplement; interactivity is per user. Total throughput counts input and output tokens, including cached input tokens.

| Mode | Concurrency | E2E Normalized Interactivity | Interactivity | Total Throughput / GPU | Input Throughput / GPU | Output Throughput / GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4 | 1 | 256.19 | 401.50 | 7,123.59 | 7,079.69 | 43.90 |
| TP4 | 4 | 355.06 | 536.70 | 10,315.00 | 10,245.20 | 69.79 |
| DPA4 | 32 | 111.24 | 162.25 | 59,036.07 | 58,696.24 | 339.82 |

### D.2 Latency and request accounting

Each latency cell is **p50 / p90 / p95**. These exported latency percentiles describe successful profiling requests. E2E normalized interactivity above is calculated from per-request records, not from these aggregate percentiles.

| Mode | Concurrency | TTFT (ms) | Decode ITL (ms) | Request latency (s) |
| --- | ---: | ---: | ---: | ---: |
| TP4 | 1 | 136.188 / 471.680 / 1342.663 | 1.508 / 2.491 / 3.145 | 2.115 / 10.435 / 12.710 |
| TP4 | 4 | 77.752 / 197.637 / 326.734 | 1.485 / 1.863 / 2.158 | 0.899 / 6.048 / 8.673 |
| DPA4 | 32 | 199.559 / 641.550 / 1041.387 | 4.318 / 6.164 / 7.074 | 2.308 / 12.457 / 20.122 |

| Point | Issued | Successful | Errors | Drain canceled | Interactivity eligible | TTFT time coverage | ITL time coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp4-c1 | 319 | 319 | 0 | 0 | 317 | 99.983128% | 100.000000% |
| tp4-c4 | 763 | 763 | 0 | 0 | 761 | 99.937518% | 99.938753% |
| dpa4-c32 | 4,728 | 4,725 | 0 | 3 | 4,723 | 99.998832% | 100.000000% |

Each point contains two successful one-token outputs without a decode ITL. These contribute to throughput but are excluded from interactivity. Non-profiling records are excluded; drain cancellations do not contribute to successful-request metrics. All three points passed the 95% minimum time coverage requirement. DPA4 canceled three remaining requests after its 30-second request grace period; no forced cancellation-drain completion was recorded.

| Point | Sending start (UTC+8) | Sending end (UTC+8) | Formal window (s) | Successful-request span (s) |
| --- | --- | --- | ---: | ---: |
| tp4-c1 | 10-29 10:47:53.740 | 10-29 11:47:53.740 | 3600.000 | 3609.352627028 |
| tp4-c4 | 10-29 13:10:16.604 | 10-29 14:10:16.604 | 3600.000 | 3597.791440276 |
| dpa4-c32 | 10-29 11:56:01.074 | 10-29 12:56:01.076 | 3600.002 | 3604.124866694 |

Formal sending-window duration and the successful-request span are different quantities. SA throughput below uses the latter, including successful requests that finish during draining. A span slightly below 3600 seconds does not shorten the configured or observed sending window.

### D.3 Per-request calculation method

The unchanged `.github/scripts/atomesh/interactivity.py` module identified in Section 4 reads each point’s `profile_export.jsonl`. Only successful profiling requests contribute to throughput. Interactivity additionally requires positive OSL and available TTFT/ITL. Warmup and canceled requests are excluded.

```text
itl_i = inter_token_latency_i / 1000                         # milliseconds → seconds
eff_i = itl_i + (time_to_first_token_i / 1000) / OSL_i
Interactivity = 1 / p90(itl_i)
E2E Normalized Interactivity = 1 / p90(eff_i)
request_span_s = (max request_end_ns - min request_start_ns) / 1e9
Total Throughput / GPU = (sum ISL + sum OSL) / request_span_s / 4
Output Throughput / GPU = sum OSL / request_span_s / 4
```

Percentiles use linear interpolation of latency followed by its reciprocal. ITL is AIPerf `inter_token_latency`, not `full_response_*`. To reproduce the JSONL calculation in Appendix C.5, set `num_gpus = 4` and use the corresponding supplemental raw file under `/scratch/atom-agentic-20261029/tp4-supplement/points/`. Do not use a two-GPU divisor.

### D.4 Recorded server and client commands

The commands below are reconstructed directly from each accepted point’s recorded argument list and environment. Source installations, tokenizer snapshot, dataset cache and package versions are those described earlier in this report. Server launches clear the listed optional variables before setting the recorded recipe. Each point uses a fresh server. Run the server and client in separate terminals, and run the points sequentially.

The recorded client entry point was `agentx profile`, which applies the `agentx` scenario. Warmup used 10 requests per lane; the formal duration was 3600 seconds. The public dataset was `semianalysis_cc_traces_weka_062126` at revision `23f152f6f0f9399a85901b89a6458def0ef16729`, with 393 traces and seed 42. Offline cache and session-ID behavior are included below. An equivalent direct AIPerf invocation replaces `agentx profile` with `aiperf profile --scenario agentx` from the same virtual environment and keeps all other arguments and environment settings.

<details>
<summary>TP4, concurrency 1: recorded commands</summary>

Server:

```bash
env \
  -u ATOM_KV_OFFLOAD \
  -u ATOM_KV_OFFLOAD_EXTRA_CONFIG \
  -u ROCR_VISIBLE_DEVICES \
  -u CUDA_VISIBLE_DEVICES \
  -u ATOM_BUNDLE_WORK \
  -u GPU_MAX_HW_QUEUES \
  -u ATOM_DP_LB_REQ_EQUIV \
  -u ATOM_ENABLE_PREFILL_DELAYER \
  OMP_NUM_THREADS=4 \
  ATOM_NUMA_BIND=0 \
  ATOM_DISABLE_MMAP=true \
  AITER_LOG_LEVEL=WARNING \
  BENCHMARK_PRECISION=fp4 \
  PRECISION=fp4 \
  BENCHMARK_MODEL_KEY=dsv41flash \
  ATOM_DSV41_BENCHMARK_SYNTHETIC=1 \
  ATOM_DP_DISPATCH_DEPTH=0 \
  ATOM_DP_SESSION_AFFINITY=0 \
  ATOM_PREFILL_DECODE_INTERVAL=0 \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  ATOM_DP_BASE_PORT=30000 \
  ATOM_DP_MASTER_PORT=30500 \
  USE_ATOMESH_ENTRYPOINTS=0 \
  AITER_MAX_JOBS=8 \
  /opt/venv/bin/python3 -m atom.entrypoints.openai_server \
  --model /scratch/DeepSeek-V4.1-Flash \
  --host 127.0.0.1 \
  --server-port 8200 \
  --port 10000 \
  --kv_cache_dtype bf16 \
  --trust-remote-code \
  --index-cache-dtype fp4 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 \
  --attn-prefill-chunk-size 16384 \
  --enable_prefix_caching \
  --block-size 16 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 \
  --cudagraph-mode FULL \
  --method dspark \
  --num-speculative-tokens 5 \
  --spec-decode-acceptance-length 3.51 \
  --tool-call-parser dsml_v41 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,48,64,128]' \
  -tp 4 \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --timeout-keep-alive 900
```

Client:

```bash
env \
  HF_HOME=/scratch/atom-agentic-20261028/hf-cache \
  HF_DATASETS_CACHE=/scratch/atom-agentic-20261028/hf-cache/datasets \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  AIPERF_HTTP_TCP_USER_TIMEOUT=900000 \
  AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300 \
  AIPERF_REQUIRED_SERVER_METRIC_PREFIX=atom: \
  AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=0 \
  TOKENIZERS_PARALLELISM=false \
  /scratch/atom-agentic-20261028/agentx-venv/bin/agentx profile \
  --url http://127.0.0.1:8200 \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer-revision provider-copy-20261028 \
  --concurrency 1 \
  --benchmark-duration 3600 \
  --tokenizer-trust-remote-code \
  --server-metrics http://127.0.0.1:8200/metrics \
  --output-artifact-dir /scratch/atom-agentic-20261029/tp4-supplement/points/tp4-c1/aiperf_artifacts \
  --public-dataset semianalysis_cc_traces_weka_062126
```

</details>

<details>
<summary>TP4, concurrency 4: recorded commands</summary>

Server:

```bash
env \
  -u ATOM_KV_OFFLOAD \
  -u ATOM_KV_OFFLOAD_EXTRA_CONFIG \
  -u ROCR_VISIBLE_DEVICES \
  -u CUDA_VISIBLE_DEVICES \
  -u ATOM_BUNDLE_WORK \
  -u GPU_MAX_HW_QUEUES \
  -u ATOM_DP_LB_REQ_EQUIV \
  -u ATOM_ENABLE_PREFILL_DELAYER \
  OMP_NUM_THREADS=4 \
  ATOM_NUMA_BIND=0 \
  ATOM_DISABLE_MMAP=true \
  AITER_LOG_LEVEL=WARNING \
  BENCHMARK_PRECISION=fp4 \
  PRECISION=fp4 \
  BENCHMARK_MODEL_KEY=dsv41flash \
  ATOM_DSV41_BENCHMARK_SYNTHETIC=1 \
  ATOM_DP_DISPATCH_DEPTH=0 \
  ATOM_DP_SESSION_AFFINITY=0 \
  ATOM_PREFILL_DECODE_INTERVAL=0 \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  ATOM_DP_BASE_PORT=31000 \
  ATOM_DP_MASTER_PORT=31500 \
  USE_ATOMESH_ENTRYPOINTS=0 \
  AITER_MAX_JOBS=8 \
  /opt/venv/bin/python3 -m atom.entrypoints.openai_server \
  --model /scratch/DeepSeek-V4.1-Flash \
  --host 127.0.0.1 \
  --server-port 8201 \
  --port 10100 \
  --kv_cache_dtype bf16 \
  --trust-remote-code \
  --index-cache-dtype fp4 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 \
  --attn-prefill-chunk-size 16384 \
  --enable_prefix_caching \
  --block-size 16 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 \
  --cudagraph-mode FULL \
  --method dspark \
  --num-speculative-tokens 5 \
  --spec-decode-acceptance-length 3.51 \
  --tool-call-parser dsml_v41 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,48,64,128]' \
  -tp 4 \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --timeout-keep-alive 900
```

Client:

```bash
env \
  HF_HOME=/scratch/atom-agentic-20261028/hf-cache \
  HF_DATASETS_CACHE=/scratch/atom-agentic-20261028/hf-cache/datasets \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  AIPERF_HTTP_TCP_USER_TIMEOUT=900000 \
  AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300 \
  AIPERF_REQUIRED_SERVER_METRIC_PREFIX=atom: \
  AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=0 \
  TOKENIZERS_PARALLELISM=false \
  /scratch/atom-agentic-20261028/agentx-venv/bin/agentx profile \
  --url http://127.0.0.1:8201 \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer-revision provider-copy-20261028 \
  --concurrency 4 \
  --benchmark-duration 3600 \
  --tokenizer-trust-remote-code \
  --server-metrics http://127.0.0.1:8201/metrics \
  --output-artifact-dir /scratch/atom-agentic-20261029/tp4-supplement/points/tp4-c4/aiperf_artifacts \
  --public-dataset semianalysis_cc_traces_weka_062126
```

</details>

<details>
<summary>DPA4, concurrency 32: recorded commands</summary>

Server:

```bash
env \
  -u ATOM_KV_OFFLOAD \
  -u ATOM_KV_OFFLOAD_EXTRA_CONFIG \
  -u ROCR_VISIBLE_DEVICES \
  -u CUDA_VISIBLE_DEVICES \
  -u ATOM_BUNDLE_WORK \
  -u GPU_MAX_HW_QUEUES \
  -u ATOM_DP_LB_REQ_EQUIV \
  -u ATOM_ENABLE_PREFILL_DELAYER \
  OMP_NUM_THREADS=4 \
  ATOM_NUMA_BIND=0 \
  ATOM_DISABLE_MMAP=true \
  AITER_LOG_LEVEL=WARNING \
  BENCHMARK_PRECISION=fp4 \
  PRECISION=fp4 \
  BENCHMARK_MODEL_KEY=dsv41flash \
  ATOM_DSV41_BENCHMARK_SYNTHETIC=1 \
  ATOM_DP_DISPATCH_DEPTH=0 \
  ATOM_DP_SESSION_AFFINITY=1 \
  ATOM_DP_LB_REQ_EQUIV=512 \
  ATOM_ENABLE_PREFILL_DELAYER=1 \
  ATOM_PREFILL_DECODE_INTERVAL=10 \
  GPU_MAX_HW_QUEUES=5 \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  ATOM_DP_BASE_PORT=32000 \
  ATOM_DP_MASTER_PORT=32500 \
  USE_ATOMESH_ENTRYPOINTS=0 \
  AITER_MAX_JOBS=8 \
  /opt/venv/bin/python3 -m atom.entrypoints.openai_server \
  --model /scratch/DeepSeek-V4.1-Flash \
  --host 127.0.0.1 \
  --server-port 8202 \
  --port 10200 \
  --kv_cache_dtype bf16 \
  --trust-remote-code \
  --index-cache-dtype fp4 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 \
  --attn-prefill-chunk-size 16384 \
  --enable_prefix_caching \
  --block-size 16 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 \
  --cudagraph-mode FULL \
  --method dspark \
  --num-speculative-tokens 5 \
  --spec-decode-acceptance-length 3.51 \
  --tool-call-parser dsml_v41 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,48,64,128]' \
  -tp 4 \
  --enable-dp-attention \
  --dp-load-balance least_tokens \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --timeout-keep-alive 900
```

Client:

```bash
env \
  HF_HOME=/scratch/atom-agentic-20261028/hf-cache \
  HF_DATASETS_CACHE=/scratch/atom-agentic-20261028/hf-cache/datasets \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  AIPERF_HTTP_TCP_USER_TIMEOUT=900000 \
  AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300 \
  AIPERF_REQUIRED_SERVER_METRIC_PREFIX=atom: \
  AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1 \
  TOKENIZERS_PARALLELISM=false \
  /scratch/atom-agentic-20261028/agentx-venv/bin/agentx profile \
  --url http://127.0.0.1:8202 \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer-revision provider-copy-20261028 \
  --concurrency 32 \
  --benchmark-duration 3600 \
  --tokenizer-trust-remote-code \
  --server-metrics http://127.0.0.1:8202/metrics \
  --output-artifact-dir /scratch/atom-agentic-20261029/tp4-supplement/points/dpa4-c32/aiperf_artifacts \
  --public-dataset semianalysis_cc_traces_weka_062126
```

</details>

### D.5 Preserved export values and evidence

The following values are retained directly from the AIPerf aggregate export for traceability. They are whole-instance values and use AIPerf’s throughput accounting; the SA table in D.1 is recomputed independently from raw successful-request timestamps.

| Point | Exported total throughput (tok/s) | Exported output throughput (tok/s) | Exported benchmark duration (s) |
| --- | ---: | ---: | ---: |
| tp4-c1 | 28,494.292961 | 175.613076 | 3609.352627200 |
| tp4-c4 | 41,234.673318 | 279.008346 | 3597.791440384 |
| dpa4-c32 | 234,460.950029 | 1,349.603591 | 3604.124866560 |

Raw JSONL SHA256 values (the complete logs, exports and manifests are retained in the local evidence archive):

| Point | SHA256 |
| --- | --- |
| tp4-c1 | `9aeee6d63c20e34d9a3464df0901d2c4baeb685adfe77bf5aa425f548b77f8f0` |
| tp4-c4 | `47deeba75db74085cb8af472e5edc9ad414d9abbe189b172a0e506e18bf83164` |
| dpa4-c32 | `d6a4ab8255c59dd9dbc137009733d475aabf3992fefc375a38985fe382cdb86b` |
