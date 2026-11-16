# 2026-11-14 MI455 Workload Weekly Status Report · EXAMPLE

> ⚠️ **EXAMPLE**: this file only illustrates the format. All numbers are made up; they are not real test results.

## Lowlights

| Model | What happened | Cause | Fix | Owner |
|---|---|---|---|---|
| ❌ Qwen-35B-A3B | c64 failed | (mock) out of memory capturing CUDA graphs | No fix yet; [reproducer](../455-CI/tools/examples/runs/20261114/qwen-35b-a3b/repro.md) |  |
| ⚠️ DeepSeek R1 | throughput down 4.5% | (mock) a new MoE dispatch default | Passes with [ROCm/ATOM#99901](https://github.com/ROCm/ATOM/pull/99901) (verified) |  |
| ⚠️ DeepSeek-V4.1-Flash | GSM8K down 5.2 points to 88.9% | (mock) MoE scale rounding | Needs [ROCm/aiter#99902](https://github.com/ROCm/aiter/pull/99902) (open) |  |

## Highlights

| Model | What happened | Owner |
|---|---|---|
| 🟢 DeepSeek-V4.1-Flash | throughput up 6.9% |  |
| ✅ Kimi-K3 | recovered from the previous failure |  |

## Results

| Model | Scenario | 2026-11-07 (tok/s/GPU) | 2026-11-14 (tok/s/GPU) | Delta | GSM8K delta |
|---|---|---|---|---|---|
| DeepSeek R1 | MLPerf Endpoint | 1,482 | 1,416 | ⚠️ −4.5% | −0.6 pts (now 95.0%) |
| DeepSeek-V4.1-Flash | MLPerf Endpoint | 2,655 | 2,838 | 🟢 +6.9% | ⚠️ −5.2 pts (now 88.9%) |
| Kimi-K3 (2.8T) | MLPerf Endpoint | 336 (2026-10-31) | 340 | +1.0% | −0.1 pts (now 96.1%) (vs 2026-10-31) |
| Qwen-35B-A3B | MLPerf Endpoint | 6,345 | N/A | N/A | −0.1 pts (now 90.1%) |

## Environment

- Image: `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114`
- ATOM: `main` @ `0000000000000000000000000000000000000000` (2026-11-14)
- aiter: `main` @ `1111111111111111111111111111111111111111` (2026-11-14)

## Findings

- (mock) Kimi-K3: the server took 41 min to start on the new image, most of it
  JIT compiling; plan an hour for the first start.
- (mock) DeepSeek R1: the c64 drop comes from the new MoE dispatch default in
  ATOM main; turning it off restores the 11-07 number (fix in the follow-ups).
- (mock) DeepSeek-V4.1-Flash: faster at both concurrencies, but GSM8K fell at the
  same time; the aiter MoE change behind the speedup also rounds the scales.

## Details

### Environment details

| Item | Value |
|---|---|
| Test image | `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114` (digest `sha256:0000mock`) |
| ATOM | `main` @ [`0000000000000000000000000000000000000000`](https://github.com/ROCm/ATOM/commit/0000000000000000000000000000000000000000), 2026-11-14 (mock) ATOM main head |
| aiter | `main` @ [`1111111111111111111111111111111111111111`](https://github.com/ROCm/aiter/commit/1111111111111111111111111111111111111111), 2026-11-14 (mock) aiter main head |
| Test date | 2026-11-14 |

### Performance · MLPerf Endpoint

| Model | Parallelism | Concurrency | tok/s/GPU | vs 2026-11-07 | E2E avg interactivity (tok/s/user) | vs 2026-11-07 |
|---|---|---|---|---|---|---|
| DeepSeek R1 | tp4 | 4 | 164 | +0.2% | 149 | −0.6% |
| DeepSeek R1 | tp4 | 64 | 1,416 | ⚠️ −4.5% | 57.8 | −0.9% |
| DeepSeek-V4.1-Flash | dp4ep4 | 4 | 250 | 🟢 +6.6% | 209 | −0.5% |
| DeepSeek-V4.1-Flash | dp4ep4 | 64 | 2,838 | 🟢 +6.9% | 94.5 | +0.4% |
| Kimi-K3 (2.8T) | tp8dcp8 | 4 | 56.5 | +0.7% (vs 2026-10-31) | 105 | −0.7% (vs 2026-10-31) |
| Kimi-K3 (2.8T) | tp8dcp8 | 64 | 340 | +1.0% (vs 2026-10-31) | 45.2 | −0.5% (vs 2026-10-31) |
| Qwen-35B-A3B | tp1 | 4 | 1,005 | −0.1% | 262 | +0.5% |
| Qwen-35B-A3B | tp1 | 64 | N/A | N/A | N/A | — |

### Accuracy (GSM8K, 3-shot) · MLPerf Endpoint

| Model | flexible-extract | strict-match | 2026-11-07 | vs 2026-11-07 |
|---|---|---|---|---|
| DeepSeek R1 | 95.0% | 94.2% | 95.6% | −0.6 pts |
| DeepSeek-V4.1-Flash | 88.9% | 88.1% | 94.1% | ⚠️ −5.2 pts |
| Kimi-K3 (2.8T) | 96.1% | 95.3% | 96.2% (2026-10-31) | −0.1 pts (vs 2026-10-31) |
| Qwen-35B-A3B | 90.1% | 89.3% | 90.2% | −0.1 pts |

### Trends · MLPerf Endpoint

![MLPerf Endpoint throughput per GPU](../455-CI/tools/examples/runs/20261114/charts/perf-endpoint.png)

![MLPerf Endpoint GSM8K](../455-CI/tools/examples/runs/20261114/charts/accuracy-endpoint.png)

### Follow-ups

| Model | Problem | Next step | Note |
|---|---|---|---|
| DeepSeek R1 | (mock) a new MoE dispatch default | Passes with [ROCm/ATOM#99901](https://github.com/ROCm/ATOM/pull/99901) (verified) | (mock) verified on 455: c64 back to 1,560 tok/s/GPU |
| DeepSeek-V4.1-Flash | (mock) MoE scale rounding | Needs [ROCm/aiter#99902](https://github.com/ROCm/aiter/pull/99902) (open) | (mock) the open aiter PR fixes the MoE scale rounding |
| Qwen-35B-A3B | (mock) out of memory capturing CUDA graphs | No fix yet; [reproducer](../455-CI/tools/examples/runs/20261114/qwen-35b-a3b/repro.md) | (mock) OOM while capturing CUDA graphs, at c64 only |

### Failures

#### Qwen-35B-A3B (MLPerf Endpoint): c64

- Error: `RuntimeError: HIP error: out of memory (mock)`
- Full excerpt: [error.log](../455-CI/tools/examples/runs/20261114/qwen-35b-a3b/error.log)

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

Reproduce:

```bash
# the test container (455-CI-recipe.md); the server and GSM8K commands below run inside it
docker run -d --name ci455-qwen-35b-a3b --network host --ipc host --shm-size 128g \
  --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined -v <weights dir>:/models/<model dir>:ro -v <results dir>:/results \
  rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114
```

`server_cmd.sh`:

```bash
# (mock) 455-CI recipe, Qwen-35B-A3B server
HIP_VISIBLE_DEVICES=0 python3 -m atom.entrypoints.openai_server \
  --model /models/Qwen3.5-35B-A3B-MXFP4 --tensor-parallel-size 1 --server-port 8000
```

`bench_cmd.sh`:

```bash
SERVED_NAME='/models/Qwen3.5-35B-A3B-MXFP4'; SERVER='127.0.0.1:8000'  # (mock)
# ...then the shared client below
```

#### Shared MLPerf Endpoint client

Each bench command above sets its model's variables and then runs this, on the host (455-CI-recipe.md):

```bash
# point_yaml <name> <concurrency> <trajectories> <warmup|run> <report_dir>
point_yaml() { echo '(mock) the shared MLPerf Endpoint client config'; }
```
