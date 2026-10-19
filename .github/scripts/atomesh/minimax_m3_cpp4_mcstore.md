# MiniMax M3: PP4 prefill, Mooncake Store, C72

Case: `minimax-m3-fp4-1p1d-cpp4-tp4-eagle3-agentic-mcstore-1m-c72`
under suite `mooncake_store` in `models_atomesh.yaml`.

The experiment tests whether pipeline prefill and a larger shared CPU tier
recover prefix reuse at C72. The previous TP4 + LMCache gather baseline
completed C72 but reported 36.01% actual cache hit, 515.30 output tok/s and
227.635 s p90 TTFT (Actions run 39034151209). A result from this new recipe
is required before drawing a performance conclusion.

| Setting | Value |
| --- | --- |
| Placement | Single node, 8 GPUs, 1P1D |
| Prefill | PP4 x TP1, 15 target layers/stage, EAGLE3 on last stage, eager |
| Decode | TP4, DCP1, original EAGLE3 settings |
| P / D batched-token budget | 8192 / 2048 |
| CPU tier | Shared Mooncake Store, 2 NUMA owners x 768 GiB = 1536 GiB |
| Gather workspace | Native `--gqa-prefill-kv-workspace-mb 8192` |
| Performance | AgentX 1M, C72, 3600 s profiling, 10 warmup requests/lane |
| Allocation | 8 hours, TW `amd-spur`, default Slurm account, automatic node |

Budgets and Store size follow
`https://github.com/SemiAnalysisAI/InferenceX/pull/4045`, inspected at
`9f774fe3f040329eb38b9fbde85e0e8d3ce818bc`.
That GLM recipe also uses DCP4; this M3 experiment retains DCP1. The Store
owners use the existing TW NIC mapping (`rdma4,rdma5` and `rdma6,rdma7`).
The selected image is the successful gather baseline's
`rocm/atom-dev:nightly_202610141644`; the launcher checks for Store binaries
and its Python client before model startup. AITER stays pinned to
`5eb866a95d66a38d64eecf151ac4ef3567a07ac5`.

## Compatibility changes

- Port the requested native gather commits and their initial implementation
  dependency. Remove the CI monkey-patch gather module and its injection.
- Save all M3 PAGE regions through the Store, including K/V, their scales,
  sparse index cache, and the last stage's draft pages; isolate the codec
  version in the persistent key namespace.
- Carry EAGLE3 auxiliary captures between PP stages. Permit producer-only
  q1 input preparation with PP and speculative decoding enabled.
- Match P/D PAGEs by global target layer or draft identity. A TP1 producer
  sends each TP4 consumer its head slice; sparse KV uses eight physical
  page-16 groups inside a scheduler page-128, dense/draft KV uses one.
  The index cache is replicated. DCP relayout is not enabled by this change.

## Ordered validation

The case opts into `eval_before_benchmark`:

1. A real GPU/Store round trip checks target, index, scale and draft-like PAGE
   payloads. A skip is a failure of this gate.
2. Full GSM8K, 20-shot, EAGLE3 real acceptance, concurrency 32, completions
   endpoint, 8192 max generated tokens. The flexible-extract score must be
   at least 0.90. This is an experiment gate, not an already measured score.
3. Fresh services and a fresh Store run C72 AgentX. The two phases use separate
   container lifetimes and ports; accuracy cache warmup is not carried into
   performance. An accuracy failure prevents the performance phase.

CPU tests cover full PP auxiliary propagation, production RDMA descriptor
construction with reordered layers, TP head slicing and replicated index
bytes, all-plane Store codec creation, and phase sequencing. Actual model,
GPU Store, GSM8K and performance validation run only on ATOMESH benchmark CI.
