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
| Gather workspace | Prefill 32768 MiB (32 GiB), decode 8192 MiB (8 GiB), native grouped gather |
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

## Cached-prefill memory budget

TP1 EAGLE3 has 64 KV heads of width 128. Gather emits BF16 K and V,
so a 1,048,576-token request needs exactly 32 GiB (2 x 64 x 128 x 2
bytes/token). The former 8 GiB setting fits only 262,144 tokens under TP1.
The prefill role overrides the common 8 GiB setting to 32 GiB; decode TP4
retains 8 GiB. Batching keeps whole requests and does not truncate context.

`ModelRunner.get_num_blocks` explicitly subtracts the local dense-gather
reserve from both the utilization budget and the physical-free-memory clamp,
in addition to the measured activation peak, CUDA graphs, 2% safety margin,
and connector buffers. The largest bound across local target and draft layers
is reserved, not the sum of all layers. TBO reserves two bounds. For this eager
PP4 case the stage reservations are 32 / 0 / 0 / 32 GiB: the middle stages
contain only sparse attention. Logs report `prefill_workspace` and
`connector_reserve` next to the final KV budget and global-minimum block count.

The 1536 GiB Store still holds at most 18,837,504 complete payload tokens
(256-token chunks), at 87,552 bytes/token over all four stages. The last stage
costs 34,656 bytes/token and is the likely GPU-capacity bottleneck. Using the
old run's 287.98 GiB HBM, 65.94 GiB Torch peak and 26.06 GiB non-Torch usage
only as a scenario, with PP eager graphs=0, 32 GiB gather, approximately
1.35 GiB Store buffers and 2% safety, the last-stage KV budget is about
142.5 GiB, or 4.41M tokens. PP4 weight and activation measurements can change
this estimate; the CI's per-stage budgets are required for a measured claim.

## October 19 startup failures and fixes

Gather run `39203564775` / job `117901909731` stopped before model startup
on `pit2-p03-g52`. After file-cache advice, NUMA nodes 0 and 1 still had only
280 / 303 GiB free and 1185 / 1180 GiB of file cache. Both 768 GiB THP
preparations timed out after 600 seconds, but the old launcher treated these
as warnings and started the owners. Owner 1 failed RDMA registration with
`Cannot allocate memory`, then aborted with exit 134. The logs do not contain
a successful owner THP measurement or any GSM8K / performance result.

The Store now uses the existing bounded memory-preparation helper with
explicit owner NUMA budgets. It reclaims file cache using base pages before
THP compaction, requires full THP coverage and sufficient free capacity, and
stops startup if preparation fails. It retains 128 GiB per-node reserve and
the previous total preparation deadline (600 seconds per owner NUMA node),
with at most three attempts and bounded child cleanup. Store preparation
skips VM sysctl writes. Snapshots, child logs and the outcome are saved under
`logs/<phase>/mooncake-memory-rank-<rank>/`. Store size remains 1536 GiB.

Paged run `39208890418` / job `117918188182` passed all 33 AITER numerical
cases, then failed the two cached/uncached TBO straddle fixture cases with
`TypeError: a bytes-like object is required, not 'list'`. Production block
tables export an int32 buffer; the fixture now constructs them with
`new_block_table` and also checks the resulting block IDs. This correction
is shared with gather, which inherited the same fixture. No attention kernel
change was needed for either observed failure. Full Store RDMA, GSM8K and
C72 validation still require the subsequent GPU CI run.

## October 19 evaluation argument fix

Run `39231577736` / job `117991267513` successfully mounted the full
1536 GiB Store, passed the GPU/Store functional round trip, and initialized
both model services and the router. The PP4 prefill pool was 37,824 blocks;
the last stage reported 156.27 GiB available for KV and 4,435,968 bytes/block.
These are startup measurements from the accuracy phase, not C72 results.

GSM8K then exited before sending an inference request:
`ValueError: When fewshot_as_multiturn is True, apply_chat_template must be set.`
The case selected raw completions and disabled the chat template but inherited
`fewshot_as_multiturn: true` from the M3 model's chat accuracy defaults.
The case now explicitly sets it to false. Matrix generation also rejects this
incompatible combination for enabled GSM8K evaluations before GPU allocation.
The shared-memory errors after SIGTERM were secondary shutdown errors.

The evaluation remains full GSM8K, 20-shot, concurrency 32, 8192 max generated
tokens and flexible-extract >= 0.90 before the fresh C72 performance phase.
The original flags reproduce the exception with local `lm-eval 0.4.13`;
the corrected production matrix passes its configuration validation.

## Checks before GPU allocation

The hosted config job now exercises the production `run_eval` command with
`lm-eval 0.4.13` and `transformers 5.16.1` against a loopback completions
server. It needs no model, dataset or GPU. It checks real CLI validation,
client initialization, concurrent requests, preservation of long raw prompts,
response ordering, stop strings and the 8192-token generation limit. Disabled
evaluations skip this check. A failure stops the workflow before its GPU job
is queued. This checks client plumbing; GSM8K accuracy is still measured by
the subsequent model run.

The PP test imports now clean up all temporary AITER modules so later real
attention tests remain collectible. Draft transfer tests use the current
builder constructor and verify all four FP8 PAGE planes at TP1 and TP4 head
counts. FlyDSL geometry tests isolate optional API availability and explicitly
verify fallback when the required API is absent; tests of the actual optional
FlyDSL module still require an image that provides it.

Paged run `39231588890` failed with the same inherited GSM8K chat flag conflict
as gather run `39231577736`; both experiment branches carry the case-level
flag correction and these checks.
