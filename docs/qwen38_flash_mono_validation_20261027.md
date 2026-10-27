# Qwen3.8 Flash Mono validation — 2026-10-27

The branch supports both FP8 weights with BF16 KV and Quark MXFP4 weights with
FP8 KV, for TP1 and TP8+EP8 ordinary decode at widths 1, 2 and 4. Prefill and
unsupported decode steps retain the native path. MTP is outside this contract.
See the [serving recipe](../recipes/Qwen3.8-Flash-Next.md).

## Version and dependency boundary

The short-input rows below report the last completed whole-model acceptance, delivered as
`56026053c8178a8294239179c52fa6e3e6617c14` on main `e893780f`. Its frozen runtime
source was `1ab5297ce82317f823158f21b79a5626406407cb`; the recorded source-equivalence
audit establishes the relationship to that delivery.

Both Native and Mono in that acceptance used AITER
`0.1.1.dev1+g508736740` **with the same local BF16-to-MXFP4 endpoint patch**.
The submitted branch now uses stock AITER: G1 writes BF16 and a separate native
operation quantizes the intermediate to MXFP4. The patch and its marker check
have been removed. The branch is rebased onto main `f324de48a444e9b6edf3603b126c079608ad3f7d`.
The source cleanup and subsequent host-contract fixes are documented separately below.

**The table is not a fresh stock-AITER whole-model acceptance of this cleanup.**
No new stock-AITER TPOT or full GSM8K result was available at this publication.
In particular, the MXFP4 speedups below cannot establish that the patch-free
version still meets every speedup target. The patch removal changes the G1
execution path; source cleanup equivalence does not prove its performance.

## Unified performance and accuracy summary

TPOT is milliseconds per output token. Short-input rows use nominal 1,024-token
input and 256-token output. Each configuration ran Native → Mono → full accuracy →
Mono repeat → Native repeat. The reference TPOT is the arithmetic mean of the
two per-round request medians for that implementation; speedup is Native/Mono.

| 配置／场景（权重＋KV） | 输入／输出 token | Native TPOT | 对照 Mono TPOT | 加速比 | 最近短输入 Mono TPOT | 精度／功能验证 |
|---|---:|---:|---:|---:|---:|---|
| MXFP4＋FP8 · TP1 C1 | 1,024／256 | 6.4971 | 4.7113 | **1.379×** | 4.7119 | C1 Mono 路由通过；全量 GSM8K 见同配置 C4 |
| MXFP4＋FP8 · TP1 C2 | 1,024／256 | 6.9712 | 5.0072 | **1.392×** | 5.0069 | C2 Mono 路由通过；全量 GSM8K 见同配置 C4 |
| MXFP4＋FP8 · TP1 C4 | 1,024／256 | 8.7415 | 5.6151 | **1.557×** | 5.6174 | GSM8K **1280/1319，97.04%，通过参考线**；C4 Mono 路由通过 |
| MXFP4＋FP8 · TP8+EP8 C1 | 1,024／256 | 6.2379 | 3.7895 | **1.646×** | 3.7892 | 全 rank C1 Mono 路由通过；全量 GSM8K 见同配置 C4 |
| MXFP4＋FP8 · TP8+EP8 C2 | 1,024／256 | 6.3613 | 3.9646 | **1.605×** | 3.9615 | 全 rank C2 Mono 路由通过；全量 GSM8K 见同配置 C4 |
| MXFP4＋FP8 · TP8+EP8 C4 | 1,024／256 | 7.6012 | 4.2562 | **1.786×** | 4.2521 | GSM8K **1287/1319，97.57%，通过参考线**；全 rank C4 Mono 路由通过 |
| FP8＋BF16 · TP1 C1 | 1,024／256 | 7.7516 | 6.7434 | **1.150×** | 6.7438 | C1 Mono 路由通过；全量 GSM8K 见同配置 C4 |
| FP8＋BF16 · TP1 C2 | 1,024／256 | 8.4397 | 5.9885 | **1.409×** | 5.9918 | C2 Mono 路由通过；全量 GSM8K 见同配置 C4 |
| FP8＋BF16 · TP1 C4 | 1,024／256 | 10.3090 | 6.9308 | **1.487×** | 6.9292 | GSM8K **1286/1319，97.50%，通过参考线**；C4 Mono 路由通过 |
| FP8＋BF16 · TP8+EP8 C1 | 1,024／256 | 6.9046 | 4.2305 | **1.632×** | 4.2302 | 全 rank C1 Mono 路由通过；全量 GSM8K 见同配置 C4 |
| FP8＋BF16 · TP8+EP8 C2 | 1,024／256 | 7.1862 | 4.7570 | **1.511×** | 4.7583 | 全 rank C2 Mono 路由通过；全量 GSM8K 见同配置 C4 |
| FP8＋BF16 · TP8+EP8 C4 | 1,024／256 | 8.1126 | 5.0461 | **1.608×** | 5.0451 | GSM8K **1289/1319，97.73%，通过参考线**；全 rank C4 Mono 路由通过 |
| FP8＋BF16 · TP1 C1（10/23 历史） | 16,384／128 | 7.9134 | 6.0484 | **1.308×** | — | 长上下文检索 A1/B/A2 **各 4/4 正确** |
| FP8＋BF16 · TP1 C1（10/23 历史） | 31,744／128 | 7.9713 | 6.0646 | **1.314×** | — | 长上下文检索 A1/B/A2 **各 4/4 正确** |
| FP8＋BF16 · TP8+EP8 C1（10/23 历史） | 16,384／128 | 7.0024 | 4.4280 | **1.581×** | — | 长上下文检索 A1/B/A2 **各 4/4 正确** |
| FP8＋BF16 · TP8+EP8 C1（10/23 历史） | 31,744／128 | 7.0669 | 4.4823 | **1.577×** | — | 长上下文检索 A1/B/A2 **各 4/4 正确** |

“最近短输入 Mono TPOT” is the second Mono measurement (`M1R`) of the same
short-input acceptance version, not a measurement of the current patch-free
branch. Speedups use the “对照 Mono TPOT” column; no cross-version speedup is
computed from the latest column. GSM8K entries show flexible-extract; full
evaluation was at C4 only, while C1/C2 entries state their actual route coverage.

Within this protocol, maximum absolute drift between the two Native medians
was 0.422%; maximum Mono repeat drift was 0.193%. This describes repeatability
of the reported median TPOT, not tail latency or every throughput metric.

Service configuration: block size 64, deployment capacity 8,192, max sequences
4, max batched tokens 2,048, FULL graphs with capture sizes [1,2,4], no prefix
cache, no MTP, BF16 index cache and layer activations. Measurements used
`benchmark_serving`, seed 42, ignore EOS, 2×C warmups and 8×C measured requests.
The request records retain actual token lengths and errors.

The hardware was MI355X; the image was `rocm/atom-dev:nightly_202610221638`, with
Torch 2.10.0 / ROCm 7.2.4, FlyDSL 0.4.0, transformers 5.16.1 and lm_eval 0.4.13.
Each pair used the same checkpoint, dependency environment and BF16 GEMM
tuning CSV. MXFP4 ran under job 244199; FP8/BF16 ran under job 244381 on node 049.

The MXFP4 rows met the earlier #2610 FP8/BF16 relative-speedup targets in the
patched environment. Those earlier targets used output length 128, capacity
32K and older dependencies; they are targets, not a controlled quantization
comparison. The newer FP8/BF16 TP1 C1 result is sensitive to deployment capacity:
a separate 8K→32K→8K Mono diagnostic measured 6.729→5.887→6.645 ms. It does not
replace the paired values above or establish a new Native/Mono speedup.

The four historical long-context rows belong to #2610 head `b9a8f2db` versus main `e914a093` on 2026-10-23,
at 32K deployment capacity with older AITER/FlyDSL. They are retained as historical
coverage and are not mixed into the 2026-10-27 short-input speedups.

The original result JSON was recovered from the successful tool outputs of
the acceptance session; this is not a new run. Source and recovery provenance
are retained in `/home/yifehuan/data/qwen_pr2610_gap_analysis_20261026/` as
`PR2610_ACCEPTANCE_RECOVERED.json` and `PR2610_RESULT_RECOVERED.md`.

## Last completed full GSM8K and route acceptance

Each of the four GSM8K entries is one full 1,319-question evaluation with client concurrency 4;
these are not separate full evaluations for C1 and C2. The protocol is 5-shot
multiturn chat, greedy decoding, 4,096 generated tokens and `<|im_end|>` stop.
Result audits checked unique question IDs, dataset correspondence, both filters,
successful exit and configuration-specific response caches.

Strict-match counts were 1280/1319 (MXFP4 TP1), 1288/1319 (MXFP4 TP8),
1281/1319 (FP8/BF16 TP1), and 1285/1319 (FP8/BF16 TP8). The corresponding
unrounded flexible-extract scores were 0.9704321455648218,
0.9757391963608795, 0.974981046247157, and 0.9772554965883244.

All four scores exceed the Flash-Next FP8 catalog's 0.96 flexible-extract
threshold in `.github/benchmark/sglang_models_accuracy.json`. That catalog entry
is SGLang TP2+EP2. These Native-ATOM TP1/TP8 deployments are separate validation
configurations; Quark has no independent catalog entry and is compared with
the FP8 reference threshold. These results are not official runs of that CI entry.

Separate profiles verified C1/C2/C4 Mono execution on every rank, by linking
CPU decode annotations to `hipGraphLaunch` and then to kernel correlation IDs.
TP8 Quark exercised the fused QSA selection path; FP8/BF16 exercised the expected
GDN and QSA seams. A high GSM8K score alone was not used to prove Mono dispatch.

## Checks completed before the host-contract review

The pure-cleanup comparison freezes `f7b13d3a` (already using stock AITER) as
`before` and the cleaned source as `after`. Both run in a fresh container on
node `crsuse2-m2m-v3-049`, exclusive job 244381, with the unmodified AITER wheel.

- Fresh compile comparison: **73/73 artifact records identical** in ABI argument
  count, normalized ISA, VGPR, SGPR, LDS, private memory and spills.
- Relevant pytest suite: **241 passed**, covering Qwen dispatch/cache,
  quantization, QSA planning, shared runtime/plans, DeepSeek contract/index plan
  and GDN cache binding.
- QSA GPU regression: **48/48 cases passed**, covering widths 1/2/4, mixed contexts
  up to 8K, shuffled pages, padding, ties, eager and graph replay. Selected indices
  and final FP8 attention outputs match the native reference exactly.
- Shared top-k extraction: AST identical except the module docstring; no
  arithmetic or synchronization change.
- Formatting and lint: Black passes, with zero Ruff issues in changed files.
  Full-tree Ruff has 428 existing diagnostics, identical to main; no new diagnostics.

The compile matrix covers FP8 TP1/TP8 at widths 1/2/4 with 8K capacity and Quark
GDN/HC/router/projection/selection builders. It does not compare every Triton
specialization, stock AITER G1/G2 ISA, or 32K builds. QSA regression covers its
selection/attention seam, not the complete MoE or serving model.

Cleanup removes the obsolete metadata adapter/installer route, unused fallback
options and state-view fields, duplicate HC construction, and an unused device
argument/constant. Persistent scratch layout, offsets and allocation order are
unchanged. Active FP8/BF16 C2, TP8 long-context finish, native metadata-free QSA
profiling and independent check-mode references remain.

`deepseek_v41/mono/kernels/attn_post.py` remains the existing DeepSeek K2a kernel:
sparse attention/index selection, inverse RoPE and quantization, output
projection, TP reduction, residual/norm and MoE input preparation. Qwen does not
call it. Only its `lds_words` import changes because the unchanged top-k helper
is shared via `atom/mono/device/topk.py`; `index_score.py` changes the matching
helper import for the same reason.

At that cleanup checkpoint, fresh stock-AITER whole-model accuracy, serving route
checks and paired performance were still unmeasured. Earlier 16K/near-32K
acceptance belongs to the prior FP8/BF16 branch and is not a current MXFP4
long-context result.

## First five review findings: fixes and validation

The review used `9c70e6caf718c1aaa396ee61ffe88795ceb0a2f7`. The fixes preserve both
quantization routes and are folded into the existing feature commits; they do
not add the later plan/ABI refactors or QSA/GDN performance experiments.

1. The installer accepts the public optional `metadata_builder` keyword used by
   `ModelRunner`, including disabled and unsupported deployments. This restores
   the entry contract without restoring the obsolete metadata adapter.
2. FP8 local workspace allocation and address validation agree across ranks
   before peer registration. Mailbox construction also agrees, and failure
   closes the owned peer mappings. The injected failures cover these Qwen-local
   phases; they do not inject GPU OOM or test failures inside the shared HIP IPC
   allocator itself.
3. QSA CHECK completes reference K1 and K2 before restoring all touched Mono
   cache rows in `finally`. Mono K2 then reads its own cache. GDN reference state
   is also restored on exceptions. CHECK coverage is explicitly FP8/BF16 S1;
   Quark and B2/B4 do not run this diagnostic.
4. Immutable Quark bindings validate HC/router/projection storage, norm and
   rotary parameters, MXFP4 1x32 E8M0 scale planes, and the full KV/index/position
   cache layouts before publishing pointers. Native `uint8` E8M0 storage and
   typed E8M0 storage are both admitted, with shape, stride and encoding checks.
   Native RoPE `[positions,1,1,32]` and its contiguous 2D view are both admitted.
5. `compile_kernels` materializes an iterable once. `WidthBuilds` owns compilation,
   shared scratch admission and the model resource hook before rank agreement;
   Qwen builders no longer repeat compilation.

Before test consolidation, the relevant pytest run passed **226 tests**, including an eight-process
Gloo failure/release test, cache-restore exception tests, real installer keyword
calls, 33 Quark binding cases, generator compilation and resource-hook rejection.
The distributed fault test uses CPU collectives with an injected allocation
exception, not eight-GPU OOM injection. Three warnings come from existing shared
runtime tests using `fork`; the new distributed test uses `spawn`.

A fresh compilation of **73 artifact records** matches the previous stock-AITER
cleanup in ABI argument count, normalized ISA and resource metadata. Subsequent
edits only corrected host storage admission and tests; device arithmetic,
kernel signatures and launch schedules did not change. This is evidence about
the compiled kernels, not a new measurement of whole-model speedup.

Stock-AITER serving was exercised on node 049, job 244381, with the same 8K
capacity, block size 64 and FULL capture sizes [1,2,4]. All four combinations
(FP8/BF16 and MXFP4/FP8, each at TP1 and TP8+EP8) passed startup and seven basic
arithmetic requests at C1/C2/C4: **28/28 responses correct**. Each rank built all
three widths without Mono refusal. Separate C4 traces associate CPU decode
annotations with `hipGraphLaunch` and GPU kernel correlation IDs: **16 decoded
steps per rank** show the expected Mono GDN and QSA/HC kernels. This establishes
actual execution for C4; the C1/C2 evidence is build/capture and serving smoke.

Those graph runs use frozen source `69447378804f052fb88b8077c8855bc042e04bed`
on main `6bb43090`. The final rebase to `f324de48` brings upstream gfx1250
DeepSeek support and an explicit gfx950 Mono admission check. All Qwen source
files are byte-identical across that rebase; the relevant 226 tests passed again.
The source and upstream-delta audit is retained with the test artifacts.

On the final main base, FP8/BF16 TP8 also completed an eager S1 request with
`ATOM_MONO_CHECK=1 --level 0 --enforce-eager`. All eight ranks built S1; the answer
was correct. Rank-zero diagnostics cover 46 decode steps across all 36 GDN and
12 QSA layers, including reference K1 and K2, with finite differences throughout.
This validates the diagnostic execution path; the logged differences are not a
separate task-accuracy acceptance threshold.

Black checked 78 relevant files and Ruff passed. There is **no new full GSM8K
or paired TPOT acceptance** in this repair batch; the single summary table above
retains its original historical version and dependency boundary.

## Test audit and consolidation

The follow-up audit freezes `6bfe8abdd91571d6114ba3de74496283039a9625` and reviews
all eight test files changed by this PR (1,038 added lines). The PR now adds only
`tests/test_qwen4_exp_mono.py`, a CPU unit-test entry for installer/step admission,
zero-copy GDN cache binding, QSA workspace/RoPE layout and generator/resource
admission. The existing `test_mono_runtime.py` and `test_aiter_flydsl_gdn.py` match
main again. This is a test-only consolidation; production source is unchanged. The retained
232-line UT passed 27 parameterized cases with GPU visibility disabled and imports
of AITER, FlyDSL, Triton and MORI blocked; CUDA remained uninitialized. Black and
Ruff also passed.

The standalone GPU QSA script, full binding matrix, reference-cache exception
tests and multiprocess failure-injection harness remain in the external
validation archive. Their earlier results above retain their original source
identity and coverage; they are not claims about the smaller CI unit test.

## Artifact locations

Raw artifacts are retained outside the source repository under
`/home/yifehuan/data/qwen_mxfp4_fp8kv_experiment_20261026/`:

- `rebase_20261027/{TP1,TP8,BF16_TP1,BF16_TP8}_PERFORMANCE_ACCEPTANCE.json`
- The same prefixes with `_ACCURACY_AUDIT.json` and `_ROUTE_AUDIT.json`
- `rebase_20261027/VALIDATION_SOURCE_EQUIVALENCE.json`
- `rebase_20261027/CHECKPOINT_AND_TUNING_MANIFEST.json`
- `cleanup_20261027/compile_comparison.json`, the before/after ELF/ISA manifests,
  `pytest.log`, `qsa_regression.json`, and their exit-code files
- `review_fixes_20261027/`: `compile_comparison.json`, `pytest_final.log`,
  `pytest_latest_main_ready.log`, `black_final.log`, `ruff_final.log`,
  `tested_source.json`, `latest_rebase_equivalence.json`, the four
  `*_native_layout_route_audit.json` files, and `runs/*_native_layout/` with
  server commands, output records, GPU memory evidence and per-rank traces
- `review_fixes_20261027/eager_check_audit.json` and
  `runs/fp8_tp8_rebased_eager_check/`, including the diagnostic log and response
- `test_audit_20261027/`: the per-file audit, source manifest, original test delta
  and verbatim archived tests, including `check_qwen_quark_qsa_select.py`

The source trees and raw results are preserved so later stock-AITER measurements
can be reported separately without relabeling the patched acceptance.
