# 455-CI detailed report · 2026-11-14 · MLPerf Endpoint · EXAMPLE

> ⚠️ **EXAMPLE**: this report only illustrates the format. All numbers are made up; they are not real test results.

[← Report](455-CI-report-20261114.md)

## Environment

| Item | Value |
|---|---|
| Test image | `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114` (digest `sha256:0000mock`) |
| ATOM | `main` @ [`000000000`](https://github.com/ROCm/ATOM/commit/0000000000000000000000000000000000000000), 2026-11-14 (mock) ATOM main head |
| aiter | `main` @ [`111111111`](https://github.com/ROCm/aiter/commit/1111111111111111111111111111111111111111), 2026-11-14 (mock) aiter main head |
| Test date | 2026-11-14 |

## Performance · MLPerf Endpoint

| Model | Parallelism | Concurrency | tok/s/GPU | vs 2026-11-07 | E2E avg interactivity (tok/s/user) | vs 2026-11-07 |
|---|---|---|---|---|---|---|
| DeepSeek R1 | tp4 | 4 | 150 | −1.7% | 149 | −0.6% |
| DeepSeek R1 | tp4 | 64 | 1,363 | −1.4% | 57.8 | −0.9% |
| DeepSeek-V4.1-Flash | dp4ep4 | 4 | 250 | 🟢 +6.6% | 209 | −0.5% |
| DeepSeek-V4.1-Flash | dp4ep4 | 64 | 2,838 | 🟢 +6.9% | 94.5 | +0.4% |
| Kimi-K3 (2.8T) | tp8dcp8 | 4 | 56.5 | +0.7% (vs 2026-10-31) | 105 | −0.7% (vs 2026-10-31) |
| Kimi-K3 (2.8T) | tp8dcp8 | 64 | 340 | +1.0% (vs 2026-10-31) | 45.2 | −0.5% (vs 2026-10-31) |
| Qwen-35B-A3B | tp1 | 4 | 1,005 | −0.1% | 262 | +0.5% |
| Qwen-35B-A3B | tp1 | 64 | N/A | N/A | N/A | — |

## Accuracy (GSM8K, 3-shot) · MLPerf Endpoint

| Model | flexible-extract | strict-match | 2026-11-07 | vs 2026-11-07 |
|---|---|---|---|---|
| DeepSeek R1 | 95.0% | 94.2% | 95.6% | −0.6 pts |
| DeepSeek-V4.1-Flash | 93.9% | 93.1% | 94.1% | −0.2 pts |
| Kimi-K3 (2.8T) | 96.1% | 95.3% | 96.2% (2026-10-31) | −0.1 pts (vs 2026-10-31) |
| Qwen-35B-A3B | 90.1% | 89.3% | 90.2% | −0.1 pts |

## Trends · MLPerf Endpoint

![MLPerf Endpoint throughput per GPU](../runs/20261114/charts/perf-endpoint.png)

![MLPerf Endpoint GSM8K](../runs/20261114/charts/accuracy-endpoint.png)

## Failures

### Qwen-35B-A3B (MLPerf Endpoint): c64

- Error: `RuntimeError: HIP error: out of memory (mock)`
- Full excerpt: [error.log](../runs/20261114/qwen-35b-a3b/error.log)

Core error log:

```text
(mock) INFO  [model_runner] capturing CUDA graphs for batch sizes 1..64
(mock) Traceback (most recent call last):
(mock)   File "/app/ATOM/atom/model_engine/model_runner.py", line 1234, in capture_cudagraph
(mock)     self.run_model(dummy_batch)
(mock)   File "/app/ATOM/atom/model_engine/model_runner.py", line 987, in run_model
(mock)     hidden = self.model(input_ids, positions)
(mock) RuntimeError: HIP error: out of memory (mock)
----- last 30 lines -----
(mock) ERROR [engine_core] EngineCore_DP0 died, shutting down the server
```
