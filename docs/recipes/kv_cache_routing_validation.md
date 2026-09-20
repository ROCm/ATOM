# Native cache routing validation

The native HBM+CPU routing milestone is delivered through three ATOM draft PRs:
[layout](https://github.com/ROCm/ATOM/pull/2306),
[catalog/render/planner](https://github.com/ROCm/ATOM/pull/2307), and
[pair policy](https://github.com/ROCm/ATOM/pull/2308).
The ATOM branches are managed with `gh stack` in the order
layout → catalog/render/planner → pair policy. CPU observation uses stock LMCache's
public `get_keys()` API through connector-supplied callbacks. The separate
[LMCache residency proposal](https://github.com/LMCache/LMCache/pull/5267) is an
independent draft and is not required to build, test, or deploy these ATOM changes.

## Environment and checks

All local tests ran in existing container images. No host test environment was
installed for this validation. Runtime tests used the latest native ATOM nightly
available at the start of the work:

- `rocm/atom-dev:nightly_202609200436-lirzhang-rocm-ordering-edge`
- Image ID `sha256:0b452871af778b1a28bcbd6b52e40de5daad90356f2179aed0c38110dcaf2278`
- Python 3.12, Torch `2.10.0+rocm7.2.4.git3d3aa833`, eight gfx950 GPUs,
  approximately 288 GiB per GPU.

The final CPU adapter was validated in a fresh container from this image with its
unmodified LMCache `0.5.5rc3+rocm7.2.4.torch2.10.git3d3aa833.cxx11abi1` package.
Inspection confirmed that `LocalCPUBackend.residency_snapshot` is absent. No
LMCache Python modules or extensions were overlaid for this compatibility run.

| Final stock-LMCache validation | Result |
|---|---|
| Catalog, CPU observer, dense/offload connector and early-release regression | 395 passed |
| Real GPU/native CPU/disk round trips, including HTTP catalog and eviction | 3 passed |
| Built Rust router against HTTP catalogs and controlled P/D endpoints | 1 passed |
| Changed Python files, Black and Ruff in prebuilt formatter images | Passed |

These checks cover sampled membership, late token bindings, bounded publication,
failed observations and stale removal, lost report acknowledgments and recovery,
CPU sample age, native codec byte sizes and byte-identical GPU cache restoration.
The reporter consumes callbacks without importing LMCache or its object types.
The stock-LMCache GLM 2P2D performance matrix has not been rerun.

## Earlier integration validation

The broader suites and GLM smoke below predate the switch to sampled CPU
observation. Those runs overlaid the two proposed LMCache Python modules while
retaining the image's native extension ABI. They validate the broader routing
implementation but are not stock-LMCache GLM acceptance. Cargo used an external
build-cache volume. Formatting used the prebuilt Ruff, Black and Rust development
images. The GLM run used an immutable checkout mounted read-only, so stack rebases
could not change code during initialization or inference.

| Validation | Result |
|---|---|
| Python scheduler, sequence, connector, frontend, PP and routing regression | 1,022 passed |
| Final receive-event, BlockManager and routing contracts | 173 passed |
| Rust library, including placement, routing and recovery | 1,215 passed |
| LMCache LocalCPUBackend, including concurrent replay | 34 passed |
| Rust HTTP relay and real GPU/native CPU/disk round trips | 4 passed |
| Black whole-repository check before this report/probe was added | 884 files unchanged |

The suites overlap; their counts must not be added. CPU stub tests use empty
`HIP_VISIBLE_DEVICES` inside the runtime image; GPU round trips retain device
visibility. The controlled relay test runs the real Rust binary against actual
HTTP catalogs and controlled P/D serving endpoints. Its timing curves are test
fixtures.

## GLM execution smoke test

The run used local GLM-5.2-MXFP4 weights (index metadata: 438,001,945,864 bytes),
FP8 KV/index, physical block size 16, MTP with one speculative token, eager
execution, maximum length 4,096, and native CPU offload on P. Forced MTP
acceptance was not enabled. Each P used TP1 × PP2 with partition `39,39`;
each D used TP2 × DCP2. This is one test placement, not a topology restriction.

| Execution | GPUs | Serving port | Catalog port | Mooncake base port |
|---|---|---|---|---|
| p1 | 0,1 | 18510 | 18610 | 18700 |
| p2 | 2,3 | 18511 | 18611 | 18800 |
| d1 | 4,5 | 18520 | 18620 | 18900 |
| d2 | 6,7 | 18521 | 18621 | 19000 |

Use nonoverlapping Mooncake **port ranges**: each PP/TP worker offsets the base.
For this single-host TCP experiment, `NCCL_SOCKET_IFNAME=lo` and
`NCCL_IB_DISABLE=1` avoided a communicator initialization stall in the default
network setup. Each execution also used a separate internal engine port.
P CPU capacity was configured at 2 GiB per worker (8 GiB across both P instances),
separate from GPU capacity and catalog metadata.

All four P/D combinations completed a 363-token prompt and eight output tokens.
After the prompt, each P catalog contained 22 HBM entries (352 tokens) and one
native CPU chunk (256 tokens). Each D contained 11 HBM entries (352 tokens at
its 32-token DCP hash span). Engine cache metrics agreed with reuse on repeated
requests. This experiment caught and fixed two missing observation paths:
publication after a PP head step and local GPU publication after D receive.

Temperature zero plus `top_k=1` did **not** produce identical text across all
repetitions. A direct D control without the P/D router also produced different
outputs. The cause was not isolated, so this run is not marked as token-exact
GLM correctness acceptance. The component tests do verify byte-identical native
cache round trips and consistent planner/hash behavior.

## Real Rust router selection

A separate 378-token prefix was warmed only on P2/D2, followed by a request
through the built Rust `atomesh --policy kv_cache_aware` relay. Both P2 and D2
reported the same newly accepted dispatch ID; P1 and D1 accepted none. Engine
metrics increased by **368 HBM-reused tokens on P2** and **352 on D2**. The HTTP
request completed successfully. This establishes that selection reaches the
chosen real executions and their native reuse path.

This smoke used coarse measured HTTP-stage costs (109.5 ms for the P request,
935.3 ms for a transfer-stage proxy and 60.9 ms for the decode-step proxy), shared
between equivalent layouts. They include serving overhead and are insufficient
for production component calibration or speedup claims. CPU cost credit was
disabled by omitting an H2D curve in this real-router smoke; the CPU selection
path is exercised by the controlled relay test with real HTTP catalogs.

## Reproduce the request probe

Start the configured engines first, using the runtime image and the same
namespace manifest on all four executions. Run the probe inside that image:

```bash
python /workspace/ATOM/tests/experiments/kv_cache_routing_probe.py \
  --model glm52 \
  --prefill http://127.0.0.1:18510 --prefill http://127.0.0.1:18511 \
  --decode http://127.0.0.1:18520 --decode http://127.0.0.1:18521 \
  --repeat 2 --output /tmp/kv-routing-probe.json
```

It records discovery, raw responses, stage HTTP durations, load samples and the
first catalog snapshot page. It reports text differences without normalizing
or suppressing them. Durations include HTTP and startup/kernel effects; they
are not isolated prefill/H2D/link measurements.

## Remaining acceptance

No throughput, p95/p99 latency, event p99, 1% overhead, or CPU-copy-versus-recompute
speedup target is claimed. The GLM smoke configuration has a small CPU tier
relative to HBM and does not establish the HBM-evicted/CPU-resident performance
case. CPU lifecycle/readability, eviction, full-worker quorum, actual GPU copies
and CPU-based pair selection are covered by the component and relay tests.

Production calibration needs measured context-dependent compute, copy and link
costs on the deployed layout. Shared storage, D CPU preload, cross-host
performance matrices and final performance acceptance remain later milestones.
See [configuration and recovery](kv_cache_routing.md) for supported paths,
budgets and fallback behavior.
