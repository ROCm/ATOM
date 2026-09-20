# GLM agentic scaling CI

The `agentic_scaling` suite compares the four existing GLM CPP4/DCP4 LMCache
agentic baselines with twice the P/D instances and twice the concurrency.
The matrix derives each pair from the baseline in
`.github/benchmark/models_atomesh.yaml`; it does not copy the workload settings
into a second set of cases. This suite is manual and is not added to nightly runs.

| Existing baseline case suffix | 1P1D concurrency | 2P2D concurrency |
| --- | ---: | ---: |
| `cpp4-dcp4-agentic-lmcache-1m-c32` | 32 | 64 |
| `cpp4-dcp4-agentic-lmcache-1m-c40` | 40 | 80 |
| `cpp4-dcp4-agentic-lmcache-1m-c48` | 48 | 96 |
| `cpp4-dcp4-agentic-lmcache-1m-c56` | 56 | 112 |

The full prefix is `glm-52-mxfp4-1p1d-`. Each allocation is exclusive:

- 1P1D: one 8-GPU node.
- 2P2D: **two 8-GPU nodes**, each with P on GPUs 0–3 (PP4/TP1,
  layer partition `20,20,20,18`) and D on GPUs 4–7 (TP4/DCP4).
- One router and AIPerf run on node 0. The router registers both P endpoints and
  both D endpoints; the random policy can transfer between nodes. Results include
  the cost of these network transfers.

Both sides use the existing `random` router policy. Cache routing observation is
disabled for this comparison. There is no measured PP4→TP4/DCP4 calibration yet,
so this suite does not run or claim benefits from `kv_cache_aware`. Enabling that
comparison later requires measured costs and verified transfer paths as described
in [the cache routing recipe](kv_cache_routing.md); example curves are unsuitable.

The inherited settings include the AgentX `inferencex-agentx-mvp` scenario,
`semianalysis_cc_traces_weka_062126`, 393 entries, 1M context, seed 42, 3,600-second
profiling, 10 warmup requests per lane and 1,800-second warmup grace. Each P keeps
the baseline 256 GiB CPU budget, split by the existing launcher across its four
PP workers (64 GiB each). Two P instances therefore have twice the total CPU budget.
LMCache is the unmodified package from the image. The fixed MTP acceptance rate
`0.6633` is retained for comparability with these cases. These are trace replay
performance measurements, **not agent task correctness or natural MTP acceptance
measurements**.

## Run from the stack's top branch

Use the existing **Atomesh Benchmark** workflow with:

```bash
gh workflow run atomesh-benchmark.yaml \
  --ref Jasen/kv-routing-agentic-ci \
  -f suite=agentic_scaling \
  -f atomesh_image=rocm/atom-dev:latest \
  -f run_model_benchmark=true \
  -f publish_dashboard=false
```

Empty `case_names` selects all four pairs (eight jobs). To run one pair, set
`case_names` to its **existing 1P1D name**, for example
`glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-lmcache-1m-c32`. Multiple baseline names are
comma separated. `benchmark_concurrency` and `eval_concurrency` overrides are
rejected for this suite so they cannot silently break C/2C pairing. Existing
runner/account/node-pool inputs still apply; the candidate pool must contain at
least two available 8-GPU nodes. Jobs run sequentially (`max-parallel: 1`) and
may queue for resources; the full suite takes over eight hours including startup
and warmup. It does not cancel other Slurm jobs or stop unrelated containers.

The workflow resolves the latest nightly once and pins its digest for all eight
jobs. Each job **builds Mesh from the checked-out stack source** using the release
profile in a separate container, then passes that artifact to the router's
`ATOMESH_MESH_BINARY`. Supplying a prebuilt binary or disabling the build is an
error. Build artifacts include source commit, dirty state, Cargo version, lockfile
hash and binary SHA256. The build container uses root to access the image's Rust
toolchain under `/root`; model containers retain the Slurm UID.

No venv, pip installation or Rust installation is performed in this suite.
Matrix validation and regression tests also run in the ATOM image. Missing
preinstalled tools cause failure. AIPerf must already support the agentic CLI;
its actual version and source hash are recorded in `aiperf-version.json`.
The catalog's `aiperf_commit` is recorded for reference, but the preinstalled
version is used for **both freshly rerun sides**. Historical results produced by
a different AIPerf revision are not used as this experiment's baseline.

## Read the result

The `atomesh-agentic-scaling` artifact contains Markdown and JSON reports for
C32→64, C40→80, C48→96 and C56→112. It reports:

- Request/s and output token/s speedup, each divided by the 1P1D result.
- Throughput per active and allocated GPU; efficiency is speedup / 2
  (1.0 means linear scaling, 0.9 means 1.8× throughput with 2× GPUs).
- TTFT, ITL and end-to-end p95/p99, failure rate and API-reported cache-hit rate.
  Latencies describe successful profiling requests; cache-hit rate does not
  distinguish CPU from HBM reuse.

Missing/duplicate results, failed jobs, missing failure counts/rates, mismatched
workloads, image digests, Mesh source commits or AIPerf sources leave a pair
**INCOMPLETE** and fail the comparison step. Failure rates above the baseline's
10% threshold also prevent a valid comparison. The raw artifacts are retained.
An otherwise valid run reports measured ratios without assuming that doubling
concurrency improves GPU efficiency or tail latency.
