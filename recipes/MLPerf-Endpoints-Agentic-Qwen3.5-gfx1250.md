# Qwen3.5-35B-A3B MLPerf Endpoints (agentic), 30-minute points, on gfx1250

This recipe runs the MLPerf Endpoints agentic benchmark (MLCommons `inference-endpoint` client,
`qwen_agentic_benchmark.yaml`) against `amd/Qwen3.5-35B-A3B-MXFP4` with ATOM on 4 gfx1250 GPUs.
Qwen3.5-35B-A3B stands in for Qwen3.6-35B-A3B, the benchmark's model; the architecture is the same.

A full compliant run takes hours (see [Run length](#run-length)), so the recipe measures
**30-minute points**, the convention of the DeepSeek-V4.1 endpoints recipe: a fresh server per
concurrency, the client in PERF mode over the 613 agentic trajectories with salt and tool delays on,
issuing capped at 30 minutes, then the in-flight turns drained. Use these points to iterate; run the
full benchmark once at the end.

The scripts are in [`mlperf_endpoints_qwen35/`](mlperf_endpoints_qwen35/). Related:
[`Qwen3.5.md`](Qwen3.5.md) (serving, MTP, GSM8K), [`Agentic-Qwen3.5.md`](Agentic-Qwen3.5.md)
(AgentX on MI355X, a different workload).

## Configuration

| Item | Value |
|---|---|
| Hardware | 4 gfx1250 GPUs (GPUs 0-3) |
| Image | `rocm/fw-bringup-mlperf:gfx1250-atom-20261026-dsv41-qwen-e15-08` or later |
| ATOM / aiter | `alizaidy/gfx1250-qwen-support` of both, checked out over the image's (validated: ATOM b4f41cdd, aiter 0f95c0f4); see [Setup](#setup) |
| FlyDSL | 0.4.0 (`/opt/flydsl-0.4.0`, put first on `PYTHONPATH` by `serve_ep.sh`) |
| Model | `/data/Qwen3.5-35B-A3B-MXFP4` |
| Client | MLCommons endpoints at `/app/mlcommons-endpoints` (with its venv) |
| Dataset | `mlperf_agentic_inference_dataset`, 613 trajectories, 20,297 assistant turns |

Concurrency: **1-2048, a power of two or three times one** (1, 2, 3, 4, 6, 8, 12, 16, 24, ..., 768,
1024, 1536, 2048). Every point runs MTP, with 4 draft tokens on TP points and 3 on DP (`SPEC=<n>`
overrides, `SPEC=0` turns it off); `run_ep.sh` picks the server from the concurrency:

| Concurrency | Server |
|---|---|
| 1-16 | TP4, custom all-reduce from 16 KB, fused verify sampling, all-gather token sync, fused sigmoid gates, Qwen tokenizer cache, aiter small-k top-p |
| 24-2048 | DP4 attention + EP4 MoE (mori v2 dispatch, session affinity, [prefill / decode interval](#prefill--decode-interval) 8) |

Both: BF16 KV, prefix caching on (every turn resends the history), `--max-model-len 262144`,
`--level 0`, 16k-token prefill chunks, decode graphs captured at every size up to `max-num-seqs`.
`max-num-seqs` per engine is the concurrency (DP: a quarter, plus 25% slack for the session
affinity's imbalance), **at most 512**: see [Known issues](#known-issues). DP keeps one copy of the
KV cache per request instead of TP4's duplicated heads, about twice the capacity (TP4 BF16 holds
~34M tokens).

## Setup

The image's own ATOM predates the MTP draft fixes on the branch (see Known issues), so check out
the branch in both repositories and let aiter rebuild its JIT modules (the image's prebuilt ones
do not match the branch's sources: the server then fails with a `topk_softmax` argument error):

```bash
cd /app/ATOM  && git fetch https://github.com/ROCm/ATOM.git  alizaidy/gfx1250-qwen-support && git checkout -f FETCH_HEAD
cd /app/aiter && git fetch https://github.com/ROCm/aiter.git alizaidy/gfx1250-qwen-support && git checkout -f FETCH_HEAD
mkdir -p /tmp/aiter_jit_old && mv /app/aiter/aiter/jit/*.so /tmp/aiter_jit_old/   # rebuilt on first use (~6 min)
```

## Run

Inside the serving container, with the weights at `/data` and a work directory `$R`:

```bash
export R=/work/qwen35_ep           # dataset/ and runs/ go here
cd /app/ATOM/recipes/mlperf_endpoints_qwen35
bash get_dataset.sh                # fetches and checksums the dataset into $R/dataset
bash run_ep.sh 16                  # one point: ~35 min (server, checks, 30 min, drain)
bash sweep.sh "1 4 16 64 256 768 1536 2048"   # one fresh server per point
```

Each point writes `$R/runs/c<N>/` (`report/report.txt` is the client's result; `server.log`,
`server_cmd.txt`, `server_env.txt`) and appends to `$R/runs/status.txt`: `READY`, `SMOKE_OK` (a
chat request with the client's sampling parameters and a tool call), `PREFIX_OK` (the same greedy
request twice, the second from the prefix cache, gives the same text), `DONE` with the client's
success count, and `OUTPUTS` (`check_outputs.py`: the share of outputs that are token salad; it
should be 0).

Knobs: `CAP_MIN` (default 30), `STOP_FIRST` (1: stop issuing at the first trajectory completion
once all have started), `TRAJ_X` (trajectories = max(613, `TRAJ_X` x c), default 4: the client
repeats trajectories with a fresh salt, so the 30-minute cap and not an exhausted pool ends every
point), `TURN_TIMEOUT` (default 14400 s, the benchmark's own: the drain after the cap lasts until the
in-flight turns finish, up to ~50 min at c2048; 900 bounds it but at high concurrency fails long turns and
their trajectories: 22% of turns at c2048), `MODE` / `SPEC`
(override the table above), `SEQ_CAP` (512). A full compliant run is `CAP_MIN=600 STOP_FIRST=0`.
`ATOM_PREFILL_DECODE_INTERVAL` in the environment overrides the DP points' default of 8 (0 is the
scheduling before it; see [Prefill / decode interval](#prefill--decode-interval)).

Traces: `TRACE_AT_MIN=5 TRACE_S=10 TRACE_STOP=1 bash run_ep.sh 16` takes a 10 s torch-profiler window of
every rank 5 minutes into the benchmark (`$RUNS/c16/trace/rank_<r>/`) and then ends the point, so a
traced point takes about 10 minutes.

## Results

30-minute points on image `gfx1250-atom-20261026-dsv41-qwen-e15-08`; "tip" = the validated branch
tips (ATOM b4f41cdd, aiter 0f95c0f4), "image" = the image's own ATOM / aiter:

| c | Stack | Server | Turns (failed) | TPS | Interactivity (tok/s) | TTFT median | TPOT median | Outputs |
|---|---|---|---|---|---|---|---|---|
| 16 | image | TP4 + MTP3 | 6,390 (0) | 1,457 | 140.2 | 0.45 s | 5.2 ms | clean |
| 2048 | tip | DP4 + EP4 + MTP3, 512 / engine | 50,234 (0) | 6,465 | 3.53 | 0.93 s | 313 ms | clean |
| 4096 | image | DP4 + EP4, 512 / engine, no MTP, 900 s turn timeout | 59,302 (1,873) | 6,779 | n/a | 63 s | 289 ms | clean |

MTP accepted 2.99 tokens a forward at c16 and 2.58 at c2048 on the real data. The c4096 point is
above the range: 2,048 conversations run and the rest queue (hence the TTFT and the turns that hit
the then 900 s turn timeout); it shows the cap, not the hardware. These points predate the
[prefill / decode interval](#prefill--decode-interval) (they ran with 0). Interactivity is the
report's "E2E average interactivity": output tokens over turn latency, TTFT included.

## Prefill / decode interval

The four DP engines step in lockstep: every layer's MoE dispatch waits for all of them, so a step
lasts as long as the busiest engine, and an engine runs either a prefill or a decode in a step. Once
the KV cache fills (c2048: past 90% by minute 7) the prefill delayer's KV-pressure rule releases
prefill on every step, so engines prefill and decode in the same step and every decode waits for the
prefill beside it. In steady state at c2048 (interval 0) 46% of the steps were mixed and none was
all-decode: an engine decoded on 15% of its steps, each stretched to ~145 ms against ~55 ms for an
aligned decode step of the same size (~500 sequences an engine, graphed), and run eagerly (a captured
decode graph replays only when every engine decodes).

`serve_ep.sh` sets `ATOM_PREFILL_DECODE_INTERVAL=8` on DP points: after any engine prefills, every
engine runs 8 decode-only passes before new prefill is admitted. Steps then alternate between one
prefill on all engines, with the turns that arrived meanwhile batched into bigger chunks, and decode
steps that replay graphs. DeepSeek-R1's DP4 + EP4 + MTP endpoints script and MLPerf configs set 32 for
the same reason. c2048 with issuing capped at 15 minutes (minutes 10-15 are inside the KV-saturated
regime; these are not 30-minute points), mori's dispatch geometry tuned for this shape (worth ~0.5%
of the prefill time, not upstream):

| Interval | Mixed steps | Out tok/s, min 10-15 | Turns (failed) | TPS | TPOT median | 1000 / TPOT | E2E interactivity | NATL median | TTFT median |
|---|---|---|---|---|---|---|---|---|---|
| 0 (22-minute point, untuned mori) | 46% | 5,622 | 43,009 (0) | 7,339 | 257 ms | 3.9 | 4.02 | 4.7 | 0.91 s |
| 4 | 0 | 9,273 | 45,975 (0) | 11,330 | 65 ms | 15.3 | 6.60 | 7.6 | 13.5 s |
| 8 | 0 | 10,217 | 46,237 (0) | 10,824 | 36 ms | 28.1 | 6.70 | 7.6 | 23.0 s |

Mixed steps are counted over a 10-15 s profiler window (minute 20 for interval 0, minute 12 for the
others); outputs were clean on every run. With the interval, prompt throughput nearly doubles
(75-91k against 33-47k tok/s at minutes 6-15: aligned, full chunks) and decode is fast enough that
prefill becomes the bottleneck: ~300 conversations an engine wait for prefill, which is the TTFT.
The interval trades TTFT against TPOT: against 8, interval 4 cuts the TTFT by ~40% and nearly doubles
the TPOT. Which
side to favour depends on the interactivity definition: the report's E2E interactivity and the
client's experimental per-trajectory NATL include TTFT, 1000 / TPOT does not (the client reports
1 / mean TPOT as the official per-user TPS on non-agentic benchmarks and computes no official agentic
steady state yet). TP points do not set the interval: one engine has nothing to skew against, and a
nonzero interval there also turns on the delayer's prefill coalescing, untested here.

## Run length

Per assistant turn the context is 71k tokens at the median (p90 165k), the new uncached tokens
116, the output ~360-420 tokens. A 25-minute c1 window gives a turn's service time as 1.2 s plus
12.9 s per million context tokens; the gaps between turns are the dataset's injected tool delays
(8.9 h in all). `ep_extrap.py <events.jsonl> <dataset> <c>` scales a window to the full dataset:
a full compliant run is ~22 h at c1 (past the client's 10-hour cap) and ~1.5-2 h at c32; above
that the longest trajectory (304 turns) sets the floor. A 30-minute point only reaches each
trajectory's early turns (contexts ~36-53k), so it is optimistic against a full run; compare
points with points. `ep_drift.py <events.jsonl>` profiles a run in 5-minute bins.

## Known issues

1. **More than 512 sequences an engine returns partly garbage** (TP4 and DP4 + EP4, images of
   2026-10-26). `paris_probe.py` against `MAXSEQ=2048 bash serve_ep.sh tp <dir>`: N concurrent
   greedy "capital of France" chats are all correct up to N=512-700, 2-20% token salad at
   1024-2048 ("The capitalGermany is 10 minutes."), every bad answer derailing at its third token.
   Agentic TP4 c2048 without the cap was ~50% salad, DP4 + EP4 c4096 at 1024 an engine almost all
   salad. Not the custom all-reduce or its RMSNorm fusion, the fused GDN decode, CUDA graphs, the
   bf16 GEMMs, the GDN prefill kernels, the decode attention / MoE / all-gather backends, prefix
   caching or GDN checkpoints; KV block tables and slot mappings check out
   (`PY_ENTRY=serve_check.py` logs shared blocks and wrong slots). At most 512 an engine every run
   was clean, hence `SEQ_CAP=512`: concurrency above 2048 (DP4) is not measurable until it is
   fixed.
2. DP4 + EP4 without the cap once hit a GPU memory fault on one rank a minute in and hung the
   others (`HSA_STATUS_ERROR_MEMORY_FAULT`); not seen with the cap.
3. MTP above c16 needs the branch. On the image's own ATOM (c0883e9d9, before the branch's draft
   block-table / KV-lookahead / GDN draft-metadata / ReplaySSM fixes), DP4 + EP4 + MTP3 returned
   ~0.2% token salad at 512 an engine and hung on a 2,048-request burst at 256 an engine; on the
   validated tips the same probe is all correct (1,024 and 2 x 2,048 at once).
4. An earlier branch head (ATOM 3fd8f4a8, aiter 2fcdd108) faulted in `moe_route_psum_fused` on TP4
   serving (MTP3, 8k prompts, c16); the validated tips do not (c16 8k/1k: TPOT 1.98 ms, no faults).
5. GSM8K with prefix caching on is not yet checked on this image (the image's own Qwen3.5 scripts
   serve without prefix caching for accuracy at TP>1). The prefix-cache check above is the gate here.
