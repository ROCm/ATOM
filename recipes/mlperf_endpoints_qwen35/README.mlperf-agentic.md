# Qwen3.5-35B-A3B MXFP4 on 4 x MI455X (gfx1250): MLPerf Endpoints agentic, the 2026-11-01 points

How to reproduce the numbers below: the exact code, the settings, the run, and how each number is computed.
They are 10-minute measurement points, not official results (see [What these numbers are](#what-these-numbers-are)).

## Results

DP4 attention + EP4 MoE, MTP with 3 draft tokens, fp8 KV cache. Minutes 2-8 of 10-minute points.

| concurrency | output tok/s | TPOT p50 / p90 (ms) | tps_per_user (1000 / p90 TPOT) | requests/s | requests in the server | running / waiting per engine |
|---|---:|---|---:|---:|---:|---|
| 384 | 31,973 | 5.71 / 7.12 | 140.4 | 81.5 | 203 | 47.1 / 3.0 |
| 512 | 37,852 | 6.78 / 8.65 | 115.6 | 99.5 | 289 | 65.2 / 5.5 |
| 768 | 47,713 | 8.32 / 10.61 | 94.3 | 125.2 | 466 | 101.6 / 11.9 |
| 1024 | 52,244 | 8.66 / 10.90 | 91.7 | 136.9 | 667 | 113.9 / 51.5 |

100 tok/s/user is crossed between c512 and c768, at about 45k tok/s by interpolation. Above c768 most of the
extra concurrency waits for prefill (TTFT), which TPOT does not count.

## Code

| component | where | what it is |
|---|---|---|
| image | `rocm/fw-bringup-mlperf:gfx1250-atom-20261031-qwen35` | ROCm, PyTorch, FlyDSL 0.4.0 (`/opt/flydsl-0.4.0`), the MLCommons client (`/app/mlcommons-endpoints`, d3c57f2) |
| ATOM | `github.com/vgokhale/ATOM`, branch `vgokhale/qwen35-mlperf-agentic-20261101` | this recipe and the server code that ran |
| aiter | `github.com/ROCm/aiter`, branch `vgokhale/qwen35-mlperf-agentic-20261101` (1a168ae00) | the kernels that ran |
| mori | `github.com/ROCm/mori` c1f2f646 + 460a03c2 (ROCm/mori#881) | EP dispatch / combine, with the EP4 hidden-2048 dispatch schedule |
| model | `/data/Qwen3.5-35B-A3B-MXFP4` | Qwen3.5-35B-A3B with OCP MXFP4 weights (Quark), MTP head included |

The two `vgokhale/qwen35-mlperf-agentic-20261101` branches pin what ran. The same changes are on their way into
`alizaidy/gfx1250-qwen-support`: aiter 2558555bf, 4d14b5df0, 06ab18f6c (pushed); ATOM PRs #2884 (state image
budget), #2888 (tokenize worker processes), #2889 (fp8 fused QK-norm + MRoPE + KV write), #2890 (this recipe's
client knobs). That branch also has newer kernels, so its numbers can differ.

## Setup

Host: 4 x MI455X with the amdgpu driver loaded. A container of the image, with the model under `/data` and a
work directory under `/work`:

```bash
docker run -d --name qwen35 --network host --ipc host --privileged --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v /data:/data -v <host work dir>:/work \
  rocm/fw-bringup-mlperf:gfx1250-atom-20261031-qwen35 sleep infinity
```

Inside it (`docker exec -it qwen35 bash`):

```bash
cd /work
git clone -b vgokhale/qwen35-mlperf-agentic-20261101 https://github.com/vgokhale/ATOM.git
git clone -b vgokhale/qwen35-mlperf-agentic-20261101 https://github.com/ROCm/aiter.git
(cd aiter && git submodule update --init 3rdparty/composable_kernel)   # aiter JIT-builds its C++ ops

git clone https://github.com/ROCm/mori.git
cd mori && git checkout c1f2f646 && git cherry-pick 460a03c2
git submodule update --init 3rdparty/spdlog 3rdparty/msgpack-c
ROCM_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel GPU_TARGETS=gfx1250 \
  python3 setup.py build_ext --inplace
cd /work

R=/work/qwen35_ep bash ATOM/recipes/mlperf_endpoints_qwen35/get_dataset.sh
```

The first server start JIT-compiles aiter's ops into the aiter checkout. Later starts reuse them.

## Run a point

```bash
cd /work/ATOM/recipes/mlperf_endpoints_qwen35
set -a; . ./agentic_20261101.env; set +a
export ATOM_DIR=/work/ATOM PYTHONPATH=/work/mori/python:/opt/flydsl-0.4.0:/work/aiter
export R=/work/qwen35_ep AITER_LOG_LEVEL=WARNING
STOP_S=615 ./bench_point.sh 1024 /work/qwen35_ep/c1024_repro
```

`bench_point.sh` starts `run_ep.sh` (server, smoke and prefix checks, then the MLCommons client), stops the
point 615 s after the benchmark starts, and prints minutes 2-8:

```
c1024_repro      min 2-8: gen   51,794 tok/s  prompt   176,217  running/eng 113.9  waiting/eng  51.5  ...
window min 2-8: 49036 responses, 52,244 output tok/s (client-counted)
  TPOT ms: p50 8.66  p90 10.90  p99 13.50  ->  tps_per_user (1000/p90) 91.7  (1000/p50 115.5)
```

A point takes about 13 minutes (server start ~1.5 min, smoke checks, 10 minutes of benchmark). Run one point at
a time: a server uses all four GPUs.

## What the settings do

`agentic_20261101.env`:

| setting | why |
|---|---|
| `ATOM_TOKENIZE_PROCESSES=16`, `ATOM_TOKENIZE_CACHE_MB=0` | The API server tokenizes every prompt (~100k tokens on average). In one process that caps the request rate near 100 req/s whatever the concurrency: with the segment tokenizer cache its threads queue on the cache lock (3-8 s before the engine sees a request), without it the encodes do not run in parallel. 16 worker processes: 120-164 req/s at c1024. |
| `AITER_GFX12_TOKEN_SPLIT=1`, tiles 16-128 | Decode attention split into equal-size KV segments, so the longest contexts do not leave a tail. |
| `ATOM_DP_LM_HEAD_MODE=default` | The LM head runs on each DP rank's own rows: no per-step hidden and logits all-gathers (~0.7 ms a decode step). |
| `MEGA_DISPATCH_WIRE=fp4`, `ATOM_MEGA_STAGE1_FUSED=0` | MoE activations go over the EP dispatch in MXFP4. The compact stage-1 path stays off. |
| `ATOM_STATE_CHECKPOINT_GPU_SIZE=32` | GDN state checkpoint images get a 32 GiB budget an engine and keep each conversation's chain root and tip; without a budget they fill ~70% of the KV pool by minute 7. |
| `ATOM_PREFILL_DECODE_INTERVAL=16`, `ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT=28000` | 16 decode steps after any rank prefills; DP prefill balance. |
| `CLIENT_MAX_IDLE=4` | The client keeps idle connections 4 s (its own default) instead of the example's 0.5 s. |
| `CAP_MIN=600 STOP_FIRST=0 TURN_TIMEOUT=14400` | The example's issue cap and turn timeout; `bench_point.sh` stops the point after `STOP_S`. |
| `ATOM_PRE_ENGINE_TIMING=1`, `ATOM_STATE_CHECKPOINT_DIAG=1` | Logging only: a pre-engine timing summary and an image-budget line every 10 s in `server.log`. |

## What these numbers are

- **Window**: minutes 2-8 after the benchmark starts. Minutes 0-1 hold the cold start of every conversation.
- **output tok/s**: output tokens of the responses issued in the window, counted with the model's tokenizer over
  each response's text, reasoning and tool-call text (`tpot_p90.py`), divided by 360 s. The server's own count
  (`window_stats.py`, `gen`) agrees within ~1%.
- **tps_per_user**: 1000 / p90 TPOT, as in the rules. TPOT of a response = (completion - first chunk) /
  (its tokens - 1), from the client's event log. The rules count the tokens after the first output-bearing chunk;
  with MTP that chunk often holds 2-3 tokens, so the official p90 TPOT comes out slightly higher (a few % lower
  tps_per_user).
- **Not official results**: a compliant point runs every trajectory (`NUM_TRAJ`, a multiple of 613), scores
  inline accuracy (`ACC_INLINE=1`) and reports from the client's final summary over the steady-state window. These
  points stop after 10 minutes and ran without inline accuracy. The model is Qwen3.5; the benchmark's model is
  Qwen3.6, whose chat template keeps earlier reasoning (`preserve_thinking`), which Qwen3.5's ignores.
- **Run to run**: two c1024 runs of this configuration gave 136.9 and 135.8 req/s and 51.8k and 51.3k
  server-side tok/s. Reproduced on 2026-11-02 from these branches with the commands above (fresh mori build):
  c1024 52,053 output tok/s, TPOT p50 / p90 8.62 / 10.86 ms, tps_per_user 92.1.

## Known traps

- **`ATOM_MEGA_STAGE1_FUSED=1` hung all four GPUs** at c768 (a cross-rank wait that never completed). On a
  host with amdgpu `gpu_recovery=0` only a reboot recovers. Keep it 0.
- **A stopped client leaves its event log** (`/dev/shm/benchmark_cli_benchmark_*`, 10-17 GB for a 10-minute
  point at c1024) in tmpfs, i.e. RAM. `bench_point.sh` removes the one its point made.
- **Concurrency** must be 1-2048 and a power of two or three times one (`run_ep.sh` refuses others).
- **Stale compile cache**: after a code change, `rm -rf /root/.cache/atom/*` before the next server start.
