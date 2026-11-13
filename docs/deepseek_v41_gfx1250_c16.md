# DeepSeek-V4.1-Flash: gfx1250 TP4 C16 recipe

This is the C16 agentic recipe used to compare PR #3176 with native ATOM.
The reference is the E13 recipe in `/home/yifehuan/ws/e13-02-c16-dsv41/`,
recorded as `1110T0844_c16_pr3109`. This document records deployment and
reproduction requirements. The official ordinary 1109 image plus the four API
packages below completed the full standard native C16 benchmark on CT17. The
bundled AITER was unchanged. The dynamic UG schedule stalled in the first
standard C16 batch and has been removed, including its atomic work-claim helper,
dispatch switch and dedicated counter slot. Every service width uses the PR's
static split. The current revision also accepts actual scheduled request widths
inside padded native graph buckets and supports the model's 1M index capacity.
The current wide-target query-resident revision completed real TP4 consumers
and a matched three-arm standard comparison, with all arms passing inline
quality. It was 0.63% faster than the same-snapshot flat-query control and 5.38%
slower than native in one window per arm. This does not establish a stable or
substantial performance gain. Results below identify the tested revisions;
earlier measurements on CT17 are separate comparisons.

## Subsequent index-selector experiment

The coverage fixes below remain enabled. Bounded threshold compaction was
evaluated separately and archived after its default-path trials did not reproduce
the initial throughput gain. The serving recipe keeps the validated
query-resident implementation; no new selector flag or AITER patch is required.

The experiment kept 256 threads, 8192-key segments and the original selection
tree and physical-row consumers. It used 12/11/9-bit refinement and compacted a
crossing bucket only when it fit 2048 keys, with a full masked scan on overflow.
Stable emission read original keys and indices. Local full index chains saved
8.96/15.83/21.04 us at 54/72/96 rows, with all eight alternating pairs faster.
Those local gains did not establish a full-network benefit.

Initial opt-in source `4dd00e326b` completed the original E13 600-second issue
window and all request finalization on CT11, using the same boot, four GPUs,
model configuration, ordinary official 1109 image and bundled AITER:

| Arm | token/s | TPOT p50 / p90 ms | Inline | Completed |
|---|---:|---:|---:|---:|
| Query-resident control, compaction off | 3107.034862 | 2.724456 / 3.704712 | 53.54% | 2507/2507 |
| Query-resident, compaction on | 3139.161639 | 2.710059 / 3.661552 | 53.72% | 2430/2430 |
| Same-snapshot native | 3368.317937 | 2.424987 / 3.305037 | 53.09% | 2639/2639 |

The opt-in candidate improved throughput by 1.0340% over control but remained
6.8033% below native. All 7576 requests completed, failed/dropped were zero,
and all inline results passed the matching 52.36% threshold. Each arm has one
window; this does not establish stable gains or compare GitHub latest main.

The proposed default revision `92561177ff` compiled all 128 keys. Only 82 ELFs
were identical to the opt-in build: all 64 target ELFs and 18 draft ELFs. The
other 46 draft ELFs changed, requiring separate default-path validation with
the original recipe and no external selector override:

| Default trial | token/s | TPOT p50 / p90 ms | Inline | Completed |
|---|---:|---:|---:|---:|
| First complete window | 3082.019019 | 2.671584 / 3.630549 | 51.79%, below threshold | 2587/2587 |
| One unchanged-recipe quality recheck | 3089.570927 | 2.669966 / 3.613164 | 53.94%, passed | 2576/2576 |

Both default trials matched all 128 actual ELFs to their default CPU builds,
had capacity at least 256, completed all requests with zero failed/dropped,
and passed report integrity, source, model, image, recipe and release checks.
No GPU fault, poll timeout or foreign GPU overlap was observed. The first
quality failure is retained. Mathematics and the quality threshold were not
changed; local float or token differences were diagnostic.

The default trials have better TPOT than the earlier control window but did
not reproduce its throughput improvement. They are separate validation
windows, not an interleaved default/control comparison, and do not isolate
the cause of that difference. The experiment is archived locally and is absent
from serving code.

A separate 45-second diagnostic plus full finalization counted 5634/5634
target steps using single-launch on all ranks, including 1814 beyond 64K and
maximum context 93036. It sampled widths 2–16; the earlier 5797-step window
below sampled all 1–16. Its 96 real TP4 frames at 54/72/96 rows passed integer
metadata, guards, finite output, bounded completion and complete consumers.
Its short inline score was 46.17%, below threshold, and is retained separately
from full-window quality. These are short coverage checks, not formal
600-second coverage or full-model 1M tests. Twelve synthetic FULL/BLK/REINDEX
consumer cases through 1M and 208 selector-unit cases also passed.

The next investigation binds actual per-layer visible bounds, REINDEX compact
lengths and rank ownership before evaluating rank-sharded scoring, local
selection, global merging and real attention consumers as one complete unit.
Sequence context lengths do not represent every layer's index candidate length.

## Decode coverage revision

`8c7e861880` selects the single-launch variant from the actual scheduled verify
requests, rather than the padded native graph bucket. For example, 9–11 actual
requests in a 12-request bucket use 54–66 target rows, and 13–15 in a 16-request
bucket use 78–90 rows. Complete six-row verify blocks for 1–16 requests are
supported; the server graph sizes and DSpark recipe do not need to change.

`e4f02f7fb0` raises score, block-max, selection-list and staged page-table capacity
from 64K to 1M index rows. The page table covers 1M tokens at the smallest
supported 16-token page size. Selection uses the original two passes when they
can merge all lists and a third pass only when necessary; every CTA computes the
same pass count from the row bounds. REINDEX keeps its two-pass tree over at most
2048 retained blocks. Capacity checks remain enabled. The additional index
workspace at the maximum 16-request width is approximately 534 MiB per rank.

In a completed 45-second diagnostic issue window, all 5812 target decode steps
used single-launch on every rank, including 3570 with context beyond 64K (maximum
91088). There were zero `row_shape` or `precheck:index` refusals. All 161 requests
completed, with no failures; diagnostic inline quality was 55.22% against the
52.36% threshold. This is diagnostic coverage, not a 600-second performance or
coverage result. Prefill and unsupported features such as structured output,
logprobs or incomplete verify blocks still use their existing supported path.

Validation includes 128 production CPU ELF builds and a single-GPU production
selection consumer at bounds 65537, 131073, 262145 and 1048576, in FULL, block and
REINDEX modes. Integer indices, guards and bounded waits passed. A separate TP4
model diagnostic verified 32 complete long-context frames at 54, 60, 66 and 72
target rows, including target/draft, sampling/verify metadata, finite outputs and
scratch guards. The natural window did not sample a 96-row long-context frame,
so a subsequent TP4 probe warmed a shared 70K-token prefix and issued 16 requests
together. All requests completed and eight full target96/draft80 frames passed
the same metadata, finite-output, scratch-guard and timeout checks. All 128
actual ELF keys matched their production or diagnostic CPU artifacts, including
the eight 96-row target/draft keys; capacity was at least 256. GPUs were released
and an independent health check passed. This synthetic completion probe is not
a performance or quality benchmark, nor a full-model 1M-context test.

The width-only revision `8c7e861880` completed a separate standard 600-second
comparison: single-launch 3464.501522 token/s versus native 3434.919896 (+0.8612%),
TPOT p50 2.366766/2.332498 ms and p90 3.205739/3.221519 ms. Both passed inline
quality (53.75%/53.97%) and completed all 2656/2698 requests without failures.
This is one window per arm and does not establish a substantial or stable gain.
It must not be reused as the performance result of the new index kernels.

## Three-row index supply revision

`c0c754cb37273088423abb076614766526eedd29` keeps the scheduled-width and
1M-capacity revisions and shares FP4 key payloads and scales across three query
rows within each six-row verify block for wide target FULL layers. REINDEX uses
each row's own candidate table; draft and narrow target tasks remain single-row.
The full launch LDS peak remains 315904 bytes. All 128 production ELF artifacts
were rebuilt and checked against the executed artifacts.

A separate 45-second diagnostic plus finalization recorded 5837/5837 target
decode steps using single-launch on every rank, including 3398 beyond 64K and
a maximum context of 96340. Request widths 2–16 appeared with zero
`row_shape` or `precheck:index` refusals. The earlier 5812-step diagnostic
covered all widths 1–16 with zero refusals. Twenty-four completed TP4 frames at
54, 72 and 96 target rows passed integer metadata, scratch guards, finite-output
and timeout checks, with two frames per width per rank. All 24 diagnostic ELF
keys matched CPU artifacts and capacity was at least 256. This remains short
diagnostic coverage, not a measured 600-second coverage percentage or a
full-model 1M-context test.

The ordinary official 1109 image and unchanged bundled AITER completed the
candidate's original E13 C16 600-second window plus full finalization:
3259.118385 token/s, TPOT p50/p90 2.637501/3.553179 ms, inline accuracy 53.60%
against 52.36%, and 2446/2446 completed requests with zero failures or drops.
All 2227 expected accuracy turns were scored. Complete duration was
613.032963 seconds. The actual 128 production ELF binaries matched CPU
artifacts, capacity was at least 256, recipe and released baselines were
verified, and the independent four-device health check passed.

Both arms completed on the same source snapshot, CT17 boot and four UUIDs,
with the original E13 C16 configuration. Native used `ATOM_MONO_ENABLE=0`;
mono used `1`. Both independent post-run four-device health checks passed.

| Arm | Full-client token/s | TPOT p50 / p90, ms | Inline accuracy | Completed requests |
|---|---:|---:|---:|---:|
| Native | 3343.081669 | 2.329955 / 3.216402 | 53.47% | 2708/2708 |
| Three-row index (`c0c754cb37`) | 3259.118385 | 2.637501 / 3.553179 | 53.60% | 2446/2446 |

TPS changed by -2.5116%, TPOT p50 by +13.1997%, and TPOT p90 by +10.4706%.
Native's complete duration was 612.601845 seconds; all 2451 expected accuracy
turns were scored. Both arms had zero failed or dropped requests and passed
52.36% inline quality. This is one full window per arm against snapshot native,
not a repeated stability test or a benchmark of the latest GitHub main.
The revision has not exceeded native. The older capacity-revision result below
and the synthetic 11–17% local index-chain gain are separate measurements and
do not establish a causal full-request improvement. Final quality uses the
matched full-network inline metric; local floating-point, quantization and
token differences remain diagnostic.

## Query-resident revision and same-host control

`1d4f9fd197d23484a67434fe8c2e29c31472f2a3` assigns each eligible wide-target
FULL scorer CTA a fixed three-query group. CTA counts follow each group's
effective key-task count, and query payloads, scales and head weights remain
resident across its key tasks. It preserves M16, three-query key reuse, the
four-task page lookahead and complete key coverage. REINDEX retains each row's
candidate table and the general scoring path; draft, narrow target and non-FP4
variants use their previous path. The effective variant is bound to the JIT
key. Dynamic work claiming remains removed. No new service environment
variable is required; the standard recipe below is unchanged.

The ordinary official 1109 image, unchanged bundled AITER and matching DSV4.1
configuration completed the following on the same CT11 boot, four GPU UUIDs
and ATOM snapshot. Each arm used the original E13 600-second issue window and
full finalization, without phase instrumentation. The flat control explicitly
disabled query residency in the external harness's builder; all other code
and recipe settings matched. It is an algorithmic GROUP3 control on this
snapshot, not a claim that its ELF is identical to the earlier CT17 revision.

| Arm | Full-client token/s | TPOT p50 / p90, ms | Inline accuracy | Completed requests |
|---|---:|---:|---:|---:|
| Native | 3335.103023 | 2.452837 / 3.306005 | 53.50% | 2651/2651 |
| Flat-query GROUP3 control | 3135.923022 | 2.737631 / 3.707120 | 54.70% | 2420/2420 |
| Query resident (`1d4f9fd197`) | 3155.823423 | 2.717198 / 3.691327 | 52.49% | 2468/2468 |

Resident TPS was +0.6346% versus flat, and TPOT p50 was -0.7464%. Against
native, resident TPS was -5.3755% and TPOT p50/p90 were +10.7777%/+11.6552%.
Flat TPS was -5.9722% versus native. All three passed the 52.36% full-network
inline threshold, with zero failed or dropped requests and every expected
accuracy turn scored (native 2400, flat 2198, resident 2241). Complete durations
were 616.590848, 617.055644 and 619.096742 seconds, respectively. Resident
remains the current eligible-wide default for further optimization. The small
full-request difference is not repeated stability evidence or a measurement
against latest GitHub main; the runs do not have identical RNG or trajectory
sets. Natural emitted tokens per request step were 3.302875, 3.665863 and
3.685635, respectively, so acceptance differences remain part of the result.
Do not infer the query revision's effect from the earlier CT17 numbers.

All 128 production ELF keys in each mono arm matched that arm's CPU artifacts,
with actual cooperative capacity at least 256. Recipe, image, model and released
baselines passed audit. Every arm's independent four-device health check passed;
no driver faults, poll timeouts or foreign GPU overlap were observed. The full
launch LDS peak remains 315904 bytes; existing full-kernel spills are retained.

A separate 45-second diagnostic plus finalization recorded 2766/2766 target
decode steps using single-launch on all ranks, including 508 beyond 64K and a
maximum context of 81530. Widths 2–16 appeared, with zero `row_shape` or
`precheck:index` refusals; width 1 appeared in the earlier 5812-step diagnostic.
Twenty-four real target54/72/96 frames passed integer metadata, scratch
guards, finite-output and bounded-wait checks. All 128 loaded production or
diagnostic ELF keys matched CPU artifacts. A subsequent shared-70K-prefix
completion probe finished 16/16 requests and eight target96/draft80 frames;
its 128 mixed production/diagnostic ELF keys also matched, capacity was at least
256, and independent health passed after release. These are structural and
short coverage checks, not formal 600-second coverage or a full-model 1M test.

A subsequent diagnostic copying real per-layer MoE routes recorded 5797/5797
target steps using single-launch on all four ranks. All widths 1–16 appeared,
including 2481 steps beyond 64K and a maximum context of 95876. Its 24 completed
target54/72/96 frames matched the diagnostic CPU ELF artifacts and passed the
integer, guard, finite-output and timeout checks. The additional copies affect
timing; this short run is a coverage and input-capture check. Its inline score
was 48.36%, below 52.36%, and does not replace the standard-window quality
results above.

The independent synthetic score + ordinary selection + irow chain improved
by 8.5592/9.8986/13.7114 microseconds at 54/72/96 rows, with 8/8 balanced pairs
faster per width. That fixture uses effective key counts and excludes candidate
block production and REINDEX. Its savings cannot be multiplied by 40: the real
target scorer executes at layers 2/8/14/20/24/28/32/36, with layer 20 REINDEX.
Whole-network matched quality remains the numerical gate; local floating-point,
quantization and token differences remain diagnostic.

## Matched coverage-revision official-image result

Both arms used `e4f02f7fb0e69984af32c226fc2c655787cda05e`, the same CT17 boot
and four GPU UUIDs, the ordinary official image and unchanged bundled AITER,
and the original E13 600-second client with full finalization. The only service
configuration difference was `ATOM_MONO_ENABLE=0` for native and `1` for mono.
Formal runs did not use the diagnostic phase instrumentation.

| Arm | Full-client token/s | TPOT p50 / p90, ms | Inline accuracy | Completed requests |
|---|---:|---:|---:|---:|
| Native | 3510.611401 | 2.353001 / 3.194160 | 53.02% | 2642/2642 |
| Width + 1M capacity (`e4f02f7fb0`) | 3207.071319 | 2.650720 / 3.542977 | 54.17% | 2553/2553 |

TPS changed by -8.6464%, TPOT p50 by +12.6527%, and TPOT p90 by +10.9204%.
Complete durations were 609.588119 and 616.010311 seconds, with zero failed or
dropped requests. Both passed 52.36% quality; all expected inline turns were
scored (2390 / 2319). Each arm has one full window, not a repeated stability
test or a benchmark against the latest GitHub main. There is no performance gain
and the 1.5x native target remains unmet.

All 128 actual production ELF binaries matched the capacity revision's CPU
artifacts, with cooperative capacity at least 256. Recipe, model, image, dataset
and released GPU baselines were verified. No poll timeouts, driver faults or
foreign GPU processes were observed, and both independent four-device health
checks passed.

The completed request metrics, after subtracting the pre-benchmark values,
showed natural draft acceptance of 53.3052% for mono and 46.2398% for native.
Higher acceptance did not compensate for the current mono chain's cost. The
100% coverage figure above remains a short diagnostic measurement, not a
measured coverage percentage over these 600-second windows.

## Earlier dynamic-removal official-image result

Both arms used the configuration below, the same CT17 boot and four GPU UUIDs,
the ordinary official image, bundled AITER, and the original 600-second E13
client with full finalization. Both arms used the same ATOM commit,
`c24f023a1d6ac7be0667fbba1e4830abccfcc9f2`, with `ATOM_MONO_ENABLE=0` for
native and `1` for the removal revision. The external harness's historical
`pr-dynamic` arm label does not enable dynamic scheduling.

| Arm | Full-client token/s | TPOT p50 / p90, ms | Inline accuracy | Completed requests |
|---|---:|---:|---:|---:|
| Native | 3432.912857 | 2.335890 / 3.167202 | 54.99% | 2701/2701 |
| Dynamic UG removed (`c24f023a1d`) | 3429.396378 | 2.390900 / 3.274220 | 52.87% | 2638/2638 |

The complete client durations were 607.582274 and 604.233740 seconds,
respectively, with zero failed/dropped requests and all expected inline turns
scored (2447 / 2398). Both passed the 52.36% quality threshold. Candidate TPS
changed by -0.1024%, TPOT p50 by +2.3550%, and TPOT p90 by +3.3790%. These are
one full window per arm, not a repeated-run stability result or a comparison
with the latest GitHub main. The 1.5x native target was not reached.

The earlier static revision `ff3bef1322` completed a separate window at
3399.508549 token/s against native 3439.382277 (-1.1593%). Removing the unused
dynamic implementation retained the static schedule, shared expert's three-unit
geometry and remaining counter offsets. All 64 target ELF binaries matched that
earlier static revision; 47 of 64 draft binaries changed, with unchanged
resource/LDS/descriptor layouts. The table above is the new full revalidation
of the removal revision, rather than a reuse of the earlier static result.

In the tested window, every rank's target/draft ELF at all 16 service widths
matched its CPU compilation artifact (128 total), with actual cooperative
capacity 256. There were no observed poll timeouts, driver faults or foreign
GPU processes. Both tasks released their GPUs; the candidate's independent
four-device compute health check also passed.

The official-image removal window's decode histogram, after subtracting the
pre-benchmark metrics, contained 65482 forwards: 29.95% with 1–8 real requests,
61.70% with 9–12, and 8.35% with 13–16. C16 therefore does not imply every
decode uses 96 target rows. These counts describe real requests, not the mono
dispatch coverage or a weighting by device time; the current index/page-table
capacity guard remains enabled.

## Configuration identity

| Item | Standard C16 setting |
|---|---|
| Checkpoint | `DeepSeek-V4.1-Flash-5254448103b6` |
| `config.json` SHA256 | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| Parallelism | TP4, without expert parallelism or DP attention |
| Client concurrency / server capacity | 16 / `max-num-seqs=64` |
| Speculation | DSpark, 5 speculative tokens; probabilistic draft and block verify |
| KV / index cache | FP8 / FP4 |
| Prefill budget / chunk | 16384 / 16384 tokens |
| Cache block size / compilation level | 16 / 0 |
| Decoder request graph sizes | `1,2,3,4,5,6,7,8,12,16` |
| Capture sizes | `1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64` |
| Sampling | temperature 1, top-p 0.95, top-k 0, maximum output 8192 |
| Chat template | `thinking=true`, `reasoning_effort=max` |
| Load | 600 seconds issuing, 1226 trajectories, salt and tool delays enabled |
| Dataset SHA256 | `1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae` |
| Inline accuracy threshold | 52.36% |

C16 is the client concurrency, not the tensor-parallel size. A full C16
DSpark verify has 96 target rows and 80 draft rows. Do not substitute a
256-output-token microbenchmark or a 1024-token prefill budget for this
standard request measurement.

## Image and dependencies

Use the official ordinary image directly:

```text
rocm/fw-bringup-mlperf:gfx1250-atom-20261109-dsv41
```

The tested local image ID is
`sha256:72354ad10d4562da7bfab2606bdae2533371878ec74d30382ea7fba1cf54194d`.
Keep its bundled AITER (`d1c0aeb6089d6784ad04806d5b003ce7294e86a4`),
MORI, FlyDSL, Torch and ROCm. Mount the current ATOM checkout at `/app/ATOM`;
ATOM source updates do not require rebuilding or transferring the image.
A custom `codex-*` image and an AITER patch are not prerequisites for this path.

The confirmed missing runtime dependency is JSON Schema support added by the
current ATOM serving tool parsers. Prepare only these Python packages in an
external directory. This preparation container needs package-network access and
maps no GPU devices:

```bash
export C16_IMAGE=rocm/fw-bringup-mlperf:gfx1250-atom-20261109-dsv41
mkdir -p "${C16_CACHE:?external writable cache}"
docker run --rm --entrypoint python3 \
  -v "$C16_CACHE:/image-cache" "$C16_IMAGE" \
  -m pip install --target /image-cache/python-api --no-deps \
  jsonschema==4.25.1 jsonschema-specifications==2025.9.1 \
  referencing==0.36.2 rpds-py==0.27.1
```

Use the same cache mount in the service container and prepend
`/image-cache/python-api` to `PYTHONPATH`. The image already contains
`attrs==26.1.0`, needed by these packages. The four additions passed a CPU
`Draft202012Validator.check_schema` import check without changing the bundled
Torch, Transformers or FlyDSL versions. Do not use an unrestricted
`pip install -e .` to prepare this pinned recipe: it can replace the image's
runtime dependencies.

Validation on 2026-11-12 used the official image without an AITER modification.
Native completed all four 45.8 GiB Engram UVA registrations, graph preparation,
tool-call smoke, idle prefix reset and the standard 600-second client with full
finalization: 2617/2617 requests, zero failures, 3439.382277 token/s and 54.46%
inline accuracy (threshold 52.36%). The task released all four devices. This
demonstrates that bundled AITER can run this model and recipe; the PR's dynamic
UG first-batch stall is a separate mono execution issue.

| Dependency | Official ordinary 1109 image / required addition |
|---|---|
| ATOM implementation | Current PR #3176 checkout, including the serving fixes below, mounted at `/app/ATOM`; kernels based on `d830a37db9717ecc193689bb3b5c7790f484ed26` |
| AITER | Bundled `d1c0aeb6089d6784ad04806d5b003ce7294e86a4`; no patch required by the startup recipe |
| MORI | `c1f2f646cc82a9b98cb83abf72c55418e7c33ce3`, retain the image's two modified files |
| MLCommons endpoints | `d3c57f2eba127c819e5a8fd83101d9e5f4e15fce`, retain the image's four client modifications |
| Torch | `2.11.0+rocm10.1.0a20260811.bkc.20260820` |
| Transformers / FlyDSL | `5.16.0.dev0` / `0.4.1` |
| Added API packages | `jsonschema==4.25.1`, `jsonschema-specifications==2025.9.1`, `referencing==0.36.2`, `rpds-py==0.27.1` |

MORI's modified files are `python/mori/ops/dispatch_combine.py` and
`src/ops/dispatch_combine/intranode_1250x.hpp`. The endpoints modifications are
`token_metrics.py`, `execute.py`, `watchdog.py` and
`agentic_inference_strategy.py`. Their tracked diffs in the ordinary official
image and the earlier comparison image have matching SHA256 values; no extra
MORI or endpoints patch is required to move to this ordinary image.

The earlier native/old-CTA comparison used a private reconstruction of E13's
clean-image dependencies, `codex-c16-e13-server-20261112`, image ID
`sha256:1a83bb3271f354154e5eb7633b6cd0411ee885954ad5ec73cb345040a5ac5d34`.
It restored AITER to the historical E13 clean tree
`b7b8d2a37738a8f1c7f796f63808c241ede9011d` and added the same API packages.
That AITER change restores historical MoE tuning/dispatch choices plus formatting;
it is not evidence that ordinary 1109 AITER cannot run ATOM.
The original E13 clean image is
`rocm/fw-bringup-mlperf:gfx1250-atom-20261109-dsv41-clean`, image ID
`sha256:672b2c0a53ee3c0e3acbf07a21df2768de4fbc94900246789fb0ba0be734ca64`.
Neither clean-tree restoration nor private image distribution is needed for the
ordinary-image startup path above. Performance measured with the reconstructed
image must not be reported as a result on the ordinary image.

Upstream AITER `main` was inspected at
`66be4f6d24b4aad5426b63a71839bc3a8f55914e` (2026-11-12). It contains the relevant
gfx1250 grouped-MoE, GEMM and FP4-index source modules, but replacing the bundled
AITER with that commit has not been GPU-validated for this recipe. Use bundled
AITER for the documented startup path; report a new dependency identity and
repeat the matched comparison when changing it.

Use a checkout containing both serving compatibility fixes:

- `824a0d162a`: the idle `POST /reset_prefix_cache` API required by the E13 runner.
- `6a417b7d9d`: draft postprocessing for a served single-launch step, which returns
  no target hidden states because it has already computed verification and drafts.
  The host still commits speculative KV state and processes the ready draft IDs.
  Native draft input views continue to include running, possibly padded rows.

The original kernel implementation commit alone is insufficient for the current
E13 runner. These host fixes preserve the single-launch kernel source and math.

The dynamic UG implementation, claim helper, dispatch switch and dedicated
counter slot have been removed. The current candidate uses the existing static
UG split for every supported width. It retains the PR's striped routed
units and dedicated shared geometry: 220 routed / 36 shared CTAs at target96,
226 routed / 30 shared at draft80. This avoids the dynamic claiming path that
failed to finish in the standard C16 batch; it does not import the old
CTA-ready ownership formula. The matched standard result above validates this
static path for one full C16 window, without claiming that the exact backend
instruction causing the dynamic-loop stall has been identified.

## Host and container requirements

- Four currently available gfx1250 GPUs with a compatible BKC driver/runtime;
  identify them by UUID and their actual render nodes before launch.
- Model and dataset mounted read-only at `/data/DeepSeek-V4.1-Flash` and
  `/dataset`, respectively. Keep the checkpoint's tokenizer and chat template.
- Private `/dev/shm` of at least 64 GiB and unlimited container memlock.
  Engram page-locking uses about 45.8 GiB of host memory per TP rank on this
  checkpoint; provide host-memory headroom for all four ranks and loading.
- CT17 completed all four actual Engram registrations with
  `/sys/module/amdgpu/parameters/no_system_mem_limit=N`. Its available host memory
  exceeded 384 GiB, each selected device's GTT budget exceeded 256 GiB, and
  container memlock was unlimited. The original E13 launcher's fixed `Y`
  assertion rejected an earlier attempt before loading; `Y` is not a universal
  startup prerequisite. Preserve the host setting, screen available memory/GTT,
  and require four successful, distinct Engram UVA registrations before timing.
  For this checkpoint the server logs must cover heads `[0,6)`, `[6,12)`,
  `[12,18)`, `[18,24)`, each approximately 45.8 GiB page-locked. A capacity screen
  alone does not prove registration succeeded. This validation changed no shared
  host parameter.
- External writable cache and result mounts; keep the implementation checkout
  read-only. Preserve the image's default HIP library for performance, including
  its `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=1` TP4 setting.

The current harness uses a private IPC namespace and `--network=none`, with
server and client in the same container communicating over loopback. A
container template, after selecting four idle UUIDs and their render nodes:

```bash
docker run --name c16-bench --network none --ipc private --shm-size 64g \
  --ulimit core=0 --ulimit memlock=-1 --device /dev/kfd \
  "${C16_RENDER_ARGS[@]}" \
  -e ROCR_VISIBLE_DEVICES="${C16_GPU_UUIDS:?four selected physical UUIDs}" \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -v "${C16_ATOM:?checkout}:/app/ATOM:ro" \
  -v "${C16_MODEL:?checkpoint}:/data/DeepSeek-V4.1-Flash:ro" \
  -v "${C16_DATASET_ROOT:?dataset root}:/dataset:ro" \
  -v "${C16_RESULTS:?external results}:/results" \
  -v "${C16_CACHE:?external cache}:/image-cache" \
  --entrypoint bash "${C16_IMAGE:?official image}" -lc 'sleep infinity'
```

`C16_RENDER_ARGS` is an array of `--device /dev/dri/renderD...` pairs for
the selected GPUs, determined on that host. Enter the container with
`docker exec`; this template only keeps the container available. It does not
start the service or client by itself.

## Server recipe

Inside the container, use the prepared API directory and retain the E13 environment:

```bash
export PYTHONPATH=/image-cache/python-api:/app/ATOM:/app/aiter
export AITER_CUSTOM_AR_MIN_SIZE=131072 AITER_CUSTOM_AR_MAX_SIZE=8388608
export AITER_RUNTIME_GPU_ARCH=gfx1250 AITER_USE_SYSTEM_TRITON=1
export FLYDSL_GPU_ARCH=gfx1250 MORI_GPU_ARCHS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0 HSA_XNACK=1
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=1
export HIP_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core
export HIP_DEVICE_LIB_PATH="$HIP_PATH/lib/llvm/amdgcn/bitcode"
export ATOM_DSPARK_PROB_DRAFT=1 ATOM_DSPARK_BLOCK_VERIFY=1
export ATOM_ENGRAM_MLOCK=1 ATOM_GC_THRESHOLD=20000,50,50
export ATOM_MEGA_MASK_PAD_ROWS=1 ATOM_MLA_QH128_ASM_PAGE_SIZE=64
export ATOM_DECODER_GRAPH_SIZES=1,2,3,4,5,6,7,8,12,16
export ATOM_MONO_ENABLE=1  # PR candidate; set 0 for the same-head native arm

python3 -m atom.entrypoints.openai_server \
  --model /data/DeepSeek-V4.1-Flash --trust-remote-code \
  --host 0.0.0.0 --server-port 8000 -tp 4 \
  --kv_cache_dtype fp8 --index-cache-dtype fp4 \
  --max-num-seqs 64 --max-num-batched-tokens 16384 \
  --attn-prefill-chunk-size 16384 --block-size 16 --level 0 \
  --method dspark --num-speculative-tokens 5 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64]' \
  --disable-uvicorn-access-log
```

When using the E13 `run-c16.sh` wrapper, use `MODE=native` or `MODE=mono`.
Its `serve_tp4.sh` sets C16's default mono switch to 0 before applying
`EXTRA_ENV`. Set the final override through that hook; merely exporting
`ATOM_MONO_ENABLE=1` before the shell script is insufficient:

```bash
export EXTRA_ENV='ATOM_MONO_ENABLE=1 HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=1'
# Equivalent E13 wrapper selection:
MODE=mono bash /path/to/e13-02-c16-dsv41/run-c16.sh /results/NEW_C16
```

The comparison's external `server_entry.py` installs a preparation-time
occupancy check before invoking the same server module. Before timed client
work it requires actual target/draft ELF capacities of at least 256 CTAs on
each rank, including the C16 build. It does not instrument steady-state decode.

## Client configuration and reporting

Use the E13 `client/c16.yaml`. Its rendered configuration must remain identical
except for `report_dir`:

```yaml
name: deepseek-v4.1-flash-agentic-benchmark
version: '1.0'
type: online
submission_ref:
  model: deepseek-ai/DeepSeek-V4.1-Flash
  ruleset: mlperf-endpoints-v1.0-2026-10-C1-A
model_params:
  name: /data/DeepSeek-V4.1-Flash
  tokenizer_name: /data/DeepSeek-V4.1-Flash
  temperature: 1.0
  top_k: 0
  top_p: 0.95
  max_new_tokens: 8192
  streaming: 'on'
  chat_template_kwargs:
    thinking: true
    reasoning_effort: max
datasets:
- name: agentic_combined
  type: performance
  path: /dataset/mlperf_agentic_inference_dataset/mlperf_agentic_inference_dataset.jsonl
  accuracy_config:
    eval_method: agentic_inference_inline
    num_repeats: 1
  agentic_inference:
    turn_timeout_s: 86400.0
    enable_salt: true
    inject_tool_delay: true
    num_trajectories_to_issue: 1226
    stop_issuing_on_first_user_complete: true
settings:
  load_pattern:
    type: agentic_inference
    target_concurrency: 16
  client:
    warmup_connections: 0
    max_idle_time: 0.5
  runtime:
    max_issue_duration_ms: 600000
  timeouts:
    run_timeout_s: 14400
endpoint_config:
  endpoints: [http://127.0.0.1:8000]
  api_type: openai
report_dir: /results/c16/report
```

The preserved E13 runner starts a fresh server, waits for readiness, sends its
smoke/warmup request, resets the prefix cache, and invokes
`/app/mlcommons-endpoints/venv/bin/inference-endpoint benchmark from-config
--config CLIENT_YAML --mode perf`. After 600 seconds it waits for the client
to finalize rather than signaling it. Report full-client TPS, TPS/4, E2E
interactivity, TPOT and TTFT p50/p90, completed/failed/dropped requests and
inline accuracy. Require `state=complete`, `complete=true`, client exit 0,
and zero failed/dropped requests. Keep first batches, warmup logs and tails.

The E13 runner requires `POST /reset_prefix_cache` after smoke/warmup. Use a
checkout that includes this endpoint: the original PR implementation commit
`d830a37d` does not, and returns HTTP 404 before client load begins. The reset
fences admission, checks that every scheduler is idle, then clears its prefix
index and waits for all acknowledgements. It retains physical KV allocations,
weights, graphs, JIT caches and cumulative statistics. It returns HTTP 409 while
scheduler cleanup is pending, 501 for disaggregated/pipeline or external-cache
setups, and 504/500 on timeout/failure. A partial reset is never reported as
success. Do not skip this step or replace it with physical KV zeroing.

Quality is accepted through the matched full-network inline metric. Local
floating-point, quantization, MID or generated-token differences are diagnostic;
addresses, integer metadata, epochs, guards, synchronization and finite outputs
remain strict. Do not require generated tokens to match between the arms.

## Exact comparison with E13

The actual server argument list matches E13, with an external preparation
hook as its entry point. Every environment item recorded by E13's
`server_env.txt` matches, except the deliberate candidate mono override.
The actual client YAML was also checked against E13, excluding `report_dir`.
This is recipe equivalence for the earlier three-arm comparison; the full image
and execution environment differ. The new ordinary-image startup check uses
the same model/server settings but its bundled AITER differs from historical E13:

For the recorded E13 server environment, `ATOM_MONO_ENABLE=0` is the only value
changed by the PR arm (`1`). The native arm matches every recorded value.
The actual native and PR argument lists and rendered client YAML were checked
against E13; the client differs only in its output directory. This statement
does not imply that the reconstructed image or unrecorded environment settings
are identical to the historical E13 run.

| Additional comparison setting | Purpose / scope |
|---|---|
| `ATOM_DISABLE_MMAP=true` | Checkpoint loading policy; common to the three arms |
| `ATOM_MONO_TIMELINE=''`, `ATOM_MONO_SINGLE_TIMING=0` | Disable mono diagnostic timing |
| `ATOM_MONO_C16_EY_ROLE=0`, `ATOM_MONO_C16_DE_ROLE=0` | Disable old experimental front helpers |
| `ATOM_MONO_C16_MOE_ROLE=0` for PR/native, `1` only for old CTA-ready | Select the old benchmark branch's helper; not a PR dynamic-writer transplant |
| `ATOM_MONO_SINGLE_SERVE=1` | Retained legacy harness setting; the current PR routes through `ATOM_MONO_ENABLE` |
| `ROCR_VISIBLE_DEVICES`, logical HIP/CUDA device lists | Restrict the task to the four selected physical GPUs |
| `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1` | Offline model/client operation and a read-only source checkout |
| AITER/FlyDSL/Triton/Torch/XDG cache directories | External, task-owned build and runtime caches |
| `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` | Common CPU threading settings; not captured in E13's filtered environment record |
| `MAX_JOBS=16` in the server, `AITER_COMPILE_THREADS=4`, `AITER_LOG_LEVEL=WARNING` | Common compilation and logging controls |
| Private IPC, 64-GiB shared memory, loopback-only networking | Task isolation with a colocated client |

The old CTA-ready branch uses a fixed producer-owner map. The current PR
candidate uses its existing static UG schedule after the dynamic queue stalled
in standard C16 serving. Its shared geometry and routed placement still differ
from the old branch, so the old dependency map cannot be copied into it.
The complete implementations also differ outside MoE; their standard results
cannot isolate the cost of dynamic scheduling alone. All three earlier standard arms
used the same reconstructed image, host and client configuration. An ordinary-image
performance comparison must use that ordinary image for every arm. Historical
E13 throughput and old local-MoE timings are not substituted for this comparison.
