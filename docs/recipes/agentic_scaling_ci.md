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
  both D endpoints; P/D pairs can transfer between nodes. Results include
  the cost of these network transfers.

1P1D retains its existing `random` policy: there is only one P/D pair to choose.
2P2D explicitly sets **both** `--prefill-policy kv_cache_aware` and
`--decode-policy kv_cache_aware`, as well as `--policy kv_cache_aware`.
`kv_cache_aware` is the new HBM+CPU policy; the older `cache_aware` name refers to
a different policy. Ratios in this experiment include both the resource increase
and the policy change; they do not isolate the contribution of each.

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
  -f cache_routing_bundle=/shared/calibration/glm-pp4-dcp4.json \
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

Use `run_model_benchmark=false` to validate the matrix, build the reviewed Mesh
and run its HTTP routing test in the image without allocating GPUs. This does
not require calibration. A performance run fails its calibration preflight
before allocating GPUs if `cache_routing_bundle` is missing. A policy name alone
would otherwise silently exercise load fallback.

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

## Measured routing bundle

`cache_routing_bundle` is a JSON file readable on the submit runner. Its outer
fields are:

| Field | Required value |
| --- | --- |
| `schema_version` | `1` |
| `image` | The measured ATOM image including its immutable `@sha256:...` digest |
| `nodes` | Two distinct measured Slurm node names, in slot order (short hostnames) |
| `measurement_artifact_sha256` | SHA256 of the raw measurement artifact |
| `namespace_manifest` | All seven immutable semantic identity fields from the [routing recipe](kv_cache_routing.md) |
| `calibration.executions` | Measured entries for `p0`, `d0`, `p1`, `d1`, using that recipe's cost/path schema |

Slot 0 runs on `nodes[0]`, slot 1 on `nodes[1]`. Prefill slots use GPUs 0–3 and
decode slots GPUs 4–7. Each P needs context-dependent prefill and H2D curves and
both verified Mooncake transfer paths (to `d0` and `d1`); each D needs a measured
decode-step cost. Curves must cover the 1M trace range. Do not populate these
fields with example costs or mark an untested link verified. The preflight
validates the supplied metadata; recording an artifact hash does not itself
perform the measurements.

The matrix pins the allocation to these measured nodes. Overrides/runners that
cannot honor that placement are rejected. Every P/D process receives a unique
job/slot execution ID and a reachable Catalog URL (P port 18610, D port 18611,
plus the service port offset). Before launching the router, the script queries
all live catalogs and checks namespace, physical layout, exact-prefix support
and PP/TP/DCP geometry. Only then does it bind the slot calibration to the fresh
execution IDs. Changed layouts, images or placements require new measurements.

No production PP4→TP4/DCP4 calibration is checked into this PR. The synthetic
curves in protocol tests are never passed to GPU benchmarks.

## Read the result

The `atomesh-agentic-scaling` artifact contains Markdown and JSON reports for
C32→64, C40→80, C48→96 and C56→112. It reports:

- Request/s and output token/s speedup, each divided by the 1P1D result.
- Throughput per active and allocated GPU; efficiency is speedup / 2
  (1.0 means linear scaling, 0.9 means 1.8× throughput with 2× GPUs).
- TTFT, ITL and end-to-end p95/p99, failure rate and API-reported cache-hit rate.
  Latencies describe successful profiling requests; cache-hit rate does not
  distinguish CPU from HBM reuse.
- Actual calibrated selection fraction and fallback attempts for 2P2D, from
  `atomesh_kv_cache_routing_decisions_total{outcome="selected|fallback"}`.
  Counters span the AIPerf invocation, including warmup and retries. An entirely
  fallback run is invalid; mixed runs report their coverage explicitly.

Missing/duplicate results, failed jobs, missing failure counts/rates, mismatched
workloads, image digests, Mesh source commits or AIPerf sources leave a pair
**INCOMPLETE** and fail the comparison step. Failure rates above the baseline's
10% threshold, absent routing evidence or zero calibrated selections also prevent
a valid comparison. The raw artifacts are retained.
An otherwise valid run reports measured ratios without assuming that doubling
concurrency improves GPU efficiency or tail latency.
