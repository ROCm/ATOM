# Qwen3.8 FP8 validation for ATOM PR #2529

Updated 2026-10-12. The local 131-shard checkpoint and cached GSM8K test set
were used without downloading model or evaluation data. The paired run compared
pre-PR ATOM (`98a03df21`) with the current PR branch on separate MI355X GPUs.
Both servers used TP1, CUDA graphs, an 8,192-token context, eight maximum
sequences, `transformers==5.16.1`, and MTP disabled for the GSM8K run.

## Accuracy: full GSM8K test split

All 1,319 questions ran with 5-shot multiturn chat, greedy generation, and a
4,096-token output limit. Each side produced 1,319 samples under each scoring
filter. Document and prompt hashes matched on every paired sample; neither
run logged API errors.

| GSM8K exact match | Pre-PR ATOM | PR #2529 |
| --- | ---: | ---: |
| Flexible extraction | 1,289/1,319 (97.73%) | 1,288/1,319 (97.65%) |
| Strict extraction | 1,287/1,319 (97.57%) | 1,286/1,319 (97.50%) |

Flexible extraction had six questions correct only on pre-PR ATOM and five
correct only on the PR. Strict extraction had eight and seven, respectively.
The net change is one question (0.08 percentage points) under each filter.
Raw output was identical for 345 questions; flexible extracted answers matched
for 1,304. The request phase took 14:01 on pre-PR ATOM and 12:53 on the PR, a 1.09×
question-throughput gain.

A separate MTP2 check ran the first 100 GSM8K test questions with the same
5-shot template and 4,096-token cap. Prompt and document hashes matched,
and neither side had API errors. Flexible and strict extraction each scored
98/100 on pre-PR ATOM and 99/100 on the PR. The extracted answer matched on
99/100; the single change corrected the baseline's 44 to the target 104.
Raw text matched on 30/100, so the subset does not establish bitwise output
parity or a systematic accuracy gain. Comparison output is
`/tmp/qwen38-gsm8k-mtp100-comparison.jsonl`.

Aggregate results and samples:

- `/tmp/qwen38-gsm8k-original-all`
- `/tmp/qwen38-gsm8k-fma-pr-all`
- `/tmp/qwen38-review-fma-gsm8k-comparison.jsonl`

The exact invocation is in `/tmp/run_qwen38_gsm8k_fma.sh`; the paired checker is
`/tmp/compare_qwen38_gsm8k.py`. The evaluation used cached `openai/gsm8k`
`main/test` data and `HF_DATASETS_OFFLINE=1`.

## Performance

The earlier 96-token TP1 API probe had MTP disabled. It ended at the cap with
reasoning tokens and no visible answer. It cannot represent the Qwen service
recipe, which already uses `--method mtp --num-speculative-tokens 2` on the
pre-PR checkout. A bare server command has no speculation until these flags
are given. The PR's HC and short-QSA flags are also opt-in.

The corrected TP1 MTP2 probe used the same MI355X GPU sequentially for both
checkouts, CUDA graphs, an 8,192-token context, eight maximum sequences,
and the same local FP8 checkpoint. The PR enabled HC and short-QSA flags.
With `enable_thinking=false`, each request listed integers 1–200 and produced
891 completion tokens of correct visible content with `finish_reason=stop`.
Every response was correct on both versions. Five-run median client-wall-time
completion throughput, including prefill and HTTP response time:

| Concurrent requests | Pre-PR MTP2 | PR #2529 MTP2 | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 218.51 tokens/s | 237.61 tokens/s | 1.09× |
| 4 | 825.09 tokens/s | 854.22 tokens/s | 1.04× |
| 8 | 1,573.86 tokens/s | 1,657.36 tokens/s | 1.05× |

The list workload had 100% MTP draft acceptance on both versions. A separate
256-token default-thinking probe had about 67% acceptance and no completed
visible answer. Its same-GPU median rates at 1/4/8 requests were
164.54/549.39/1,010.97 tokens/s pre-PR and
177.87/579.33/1,038.10 tokens/s on the PR.
Repeating the completed-answer PR run after the old checkout yielded
238.82/856.67/1,662.62 tokens/s at 1/4/8 requests, close to its first run.
These same-GPU results support a modest gain in this workload, not 2×.

At TP4+EP4, 16 deterministic checkpoint-weight arithmetic prompts had the
same correct final content with all optimization flags off and on (16/16).
A separate 16-request, 96-token-capped, default-thinking API probe gave
medians of 849.96 and 891.30 `usage.completion_tokens`/s over five runs. Its flags-off baseline
used the PR's model loader because the older checkout cannot recognize
`qwen4_exp_text` at TP4 on this host. Request scheduling caused noticeable
run-to-run variation.

### TP4 MTP2 completed-answer follow-up

The same four MI355X GPUs (0–3) served the local FP8 checkpoint sequentially
with TP4+EP4, CUDA graphs, `--method mtp --num-speculative-tokens 2`, an
8,192-token maximum context, and 16 maximum sequences. The flags-off and
flags-on servers both used this PR's loader: the pre-PR checkout fails at
startup with `Unrecognized model identifier: qwen4_exp_text` at TP4. The
flags-on server set `ATOM_QWEN38_MONO_HC=1`, `ATOM_QWEN38_QSA_SHORT=1`, and
`ATOM_QWEN38_QSA_FUSED_PROJ=1`; QSA index CP was off on both sides. Each
request generated the correct visible integer list 1–200, used 891 completion
tokens, and ended with `stop`. The server reported 100% MTP2 draft acceptance
on this predictable workload. Client-wall-time throughput includes prefill
and HTTP response time:

| Concurrent requests | PR flags off | PR flags on | Ratio |
| ---: | ---: | ---: | ---: |
| 4 | 920.48 | 937.81 | 1.019× |
| 8 | 1,800.15 | 1,809.75 | 1.005× |
| 16 | 3,242.45 | 3,254.49 | 1.004× |

The four- and eight-request figures are five-run medians after one warmup.
At 16 requests, the first flags-off five-run set was unstable: its run rates
ranged from 364.83 to 3,084.84 tokens/s, with a 39-second stall in one run.
The table therefore uses a second, seven-run set for **both** variants; their
run ranges were 2,768.07–3,312.08 and 2,827.73–3,300.84 tokens/s. A
separate synchronized 16-request arithmetic check with MTP2 and thinking
disabled returned correct final answers on 16/16 requests with each server.
These small throughput differences are within the observed run variation and
do not establish a meaningful TP4 service gain. The prompt stays below the
2,048-token short-QSA threshold; this probe does not measure 128k/256k
checkpoint generation. Raw runs are in
`/tmp/qwen38-tp4-mtp2-{off,on}-{completed,b16-repeat}.jsonl`.

No TP8 checkpoint serving result is reported. During this follow-up, GPU 5
remained occupied by another workload (up to 100% utilization and 25% VRAM),
so an eight-GPU throughput comparison would measure contention. The current
QSA index CP path explicitly accepts only TP2/TP4; enabling its flag at TP8
raises a configuration error. The merged QKV/index projection has exact TP4
weight-shape gates, so its TP4 result also cannot be projected onto TP8.

Synthetic-weight complete decoder-layer graph reruns on TP4+EP4, 100 replays,
after the final HC helper cleanup:

| Layer | Context | Batch | Baseline → tuned | Gain |
| --- | ---: | ---: | ---: | ---: |
| QSA | 128k | 4 | 262.24 → 220.14 µs | 1.19× |
| QSA | 128k | 16 | 306.31 → 267.37 µs | 1.15× |
| QSA | 256k | 4 | 285.75 → 238.63 µs | 1.20× |
| QSA | 256k | 16 | 371.02 → 280.79 µs | 1.32× |
| GDN, FP32 state | — | 16 | 203.12 → 181.39 µs | 1.12× |

MTP2 verifies three tokens per request, so the former HC dispatch gate at
4/8/16 rows missed the common target shapes. After extending the measured
winning shapes, a synthetic-weight TP1 QSA layer at 2k context took
239.28 → 205.11 µs at three rows (1.17×), 253.01 → 236.53 µs at 12 rows
(1.07×), and 274.02 → 255.65 µs at 24 rows (1.07×). QSA selected IDs
matched exactly; the maximum compared stream/pending NRMSE was 0.0041.
The corresponding TP1 GDN layer with FP32 state took 196.35 → 165.29 µs at
three rows (1.19×), 275.65 → 257.56 µs at 12 rows (1.07×), and
333.20 → 315.57 µs at 24 rows (1.06×). Recurrent-state NRMSE was below
0.00002; one synthetic MoE row changed its top-10 route at 24 rows.
The synthetic QSA layer uses one request per row; the checkpoint API probes
exercise actual three-token MTP2 verification per request.

An initial 128k/batch-4 baseline measured 325 µs while tuned stayed near
220 µs; the table uses an idle-GPU rerun of that case.

The 128k/256k tests also checked exact QSA selected IDs against unsplit
selection and reported at most 0.0031 NRMSE for the compared complete QSA
stream output. These long-context layer numbers use dummy weights; they do
not establish checkpoint-weight long-context generation accuracy.

## Local test setup and limits

- Review guards: native QSA index CP is disabled in plugin mode and falls back
  to local scoring when pages cannot form equal TP shards or a shard cannot
  retain global top-k. Focused QSA/config tests passed 13/13, including graph
  replay. The expanded HC and selected MTP tests passed 94/94; six QSA RoPE
  cache cases were outside this focused rerun.
- The new packed BF16 HC helper now uses public FlyDSL vector casts and `fma`
  arithmetic. The inspected 1/4/16-row FlyDSL HC stages have no dot2/MFMA
  instruction or scratch spills; served 16-row HC still uses the CK MFMA up
  GEMM. The 4/16-row HC and complete-layer reruns above retained their
  measured gains. No AITER or FlyDSL source changed.
- Model: `/root/models/Qwen3.8-Flash-Next-FP8`, pinned revision
  `236dfdf285828023ca3bcd3f37366c58a3469b13`.
- Original source: `/tmp/atom-qwen38-base` at `98a03df21`.
- PR source: `/root/ATOM-qwen38-inferencemax-mono`, branch
  `codex/qwen38-inferencemax-mono-pr`.
- Local launch script: `/tmp/launch_qwen38_accuracy_pr2529.sh`, using GPUs 4
  and 5 and ports 8011 and 8012. Set `QWEN38_USE_GRAPHS=1`.
- MTP2 same-GPU launcher: `/tmp/launch_qwen38_mtp_samegpu.sh` with
  `original_mtp` and `pr_mtp` in sequence on GPU 4. Completed-answer probe:
  `/tmp/qwen38_mtp_completed_probe.py`; raw JSONL files are
  `/tmp/qwen38-mtp2-original-completed.jsonl`,
  `/tmp/qwen38-mtp2-pr-mtp-shapes-completed.jsonl`, and
  `/tmp/qwen38-mtp2-pr-mtp-shapes-repeat-completed.jsonl`.
- The installed AITER/FlyDSL combination could not compile its GDN prefill
  and some BF16 GEMM kernels. Both variants used the existing Triton GDN
  prefill and `/tmp/qwen38-bf16-no-flydsl.csv`, which removes incompatible
  FlyDSL A16W16 tuned rows from a temporary AITER table. No dependency source
  was changed. Bypass the host HTTP proxy for local API calls with
  `NO_PROXY=127.0.0.1,localhost`.
- TP1 GSM8K and API probes cover the HC, short-context QSA, and GDN decode
  changes. TP4 merged projection and long-context index CP were checked by
  synthetic layer parity/performance, and TP4 checkpoint arithmetic serving
  was checked at 16 concurrent requests. Checkpoint-weight 128k/256k
  generation and InferenceMAX throughput remain unmeasured.

The complete layer is still below the requested 2× gain. Its QSA selection,
attention, HC, communication, and MoE remain separate launches; the current
PR specializes components rather than merging the full decoder layer.

## 2026-10-12 TP1 throughput audit

The published 98.62/318.82/610.42 and 100.68/333.19/640.87 medians
match the archived seven-run API JSONL files exactly. The probe used the
default thinking mode, a 96-token cap, and no MTP. A response spot-check on
the merged PR head had `finish_reason=length`, 96 reasoning tokens, and empty
visible `content`. This is a decode screening workload rather than completed
answer throughput.

A same-device A→B rerun on the merged PR feature code used the launcher's
`HIP_VISIBLE_DEVICES=4` setting sequentially for baseline and PR. It found
seven-run API medians of 98.34 → 100.13, 299.88 → 333.89, and
584.23 → 640.37 tokens/s at batches 1/4/8. A separate three-run,
256-token streaming probe, timing from first to last generated token,
measured 106.21 → 108.32, 334.01 → 373.37, and
642.11 → 702.84 tokens/s. These are single-pass observations.
A subsequent baseline repeat on that device dropped to 25.77/73.93/158.45
API tokens/s while machine load changed sharply. That repeat invalidates a
stable same-device speedup estimate; it was not substituted into the PR table.
The PR description therefore keeps the original raw medians as screening
results and removes the gain claim.

Raw follow-up files are `/tmp/qwen38-tp1-samegpu-{original,pr}-{api,stream}.jsonl`
and `/tmp/qwen38-tp1-samegpu-original2-api.jsonl`.
