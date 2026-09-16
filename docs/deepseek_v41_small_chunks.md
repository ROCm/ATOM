# DeepSeek V4.1 small-chunk projections

This report describes the historical padding workaround. The current output
projection uses V4/AITER BF16 batched GEMM for 2..32 rows and native einsum
otherwise, without padding to 128 rows. Production mHC uses AITER delayed-pre
stages; only its test/reference coefficient helper retains padding. See the
[current performance report](deepseek_v41_performance.md).

On the tested gfx950/TP4 configuration, native wo_a BF16 and mHC FP32 GEMMs
change reduction order with row count. Downstream quantization can amplify
those differences. Original chunk63/checkpoint128 has mean NLL 0.637223 versus
0.604261 for full prefill on the fixed 37-case corpus. About 86% of that increase
occurs in the first chunk. Fixed-input attention and inverse-RoPE checks on two
text samples/four ranks were bitwise equal; those checks do not establish all
possible attention layouts, but give no reason to replace the V4 kernels.

## Operator policy

`atom/model_ops/deepseek_v41/projections.py` owns both projection entry points.
On GPU, it internally pads total row counts 2..64 to 128 with zero rows, executes
the original BF16 einsum or FP32 linear, and returns only the valid outputs.
M=1, M>64 and CPU execution use the original dispatch. Normalization/Sinkhorn
remain in mHC; grouped projection composition remains in the attention model.
No request-mode flag, scheduler coupling, global proxy, fixed hipBLASLt
algorithm number or attention chunk-size restriction is introduced.

This policy includes batched decode with 2..64 rows. It is a bounded numerical
choice, not a guarantee of bitwise invariance for arbitrary batch sizes or
positions. Other GPUs and TP layouts require their own qualification.

## Measured quality and accepted difference

The 37-case corpus has 7,258 labels and 7,295 positions. All 148 frozen ATOM baseline
records reproduce exactly. Independent reference mean NLL is 0.5959472319675303;
the unchanged allowance is +0.01.

| Schedule | Mean NLL |
| --- | ---: |
| Original full |0.6042607703|
| Original chunk63 |0.6372226241|
| Small-row full |0.6059338419|
| Small-row chunk63 |0.6044000274|

Both candidate schedules pass; full is only 0.0000133901 below the threshold.
Do not infer robust improvement or exact arithmetic from this result.

| Task metric | Baseline | Full | Chunk63 |
| --- | ---: | ---: | ---: |
| Code multiple choice |35/60|35/60|38/60|
| HellaSwag raw / normalized |33/64,49/64|34/64,50/64|34/64,50/64|
| ARC raw / normalized |51/64,54/64|53/64,56/64|53/64,56/64|
| Chinese raw / normalized |475/651,425/651|unchanged branch|471/651,423/651|
| GSM8K strict / flexible |12/16,12/16|not rerun|14/16,14/16|

Full MC is an actual execution of this policy: 754 responses agree across all
four ranks and document/prompt/target identities match. Chunk MC is reused from
the earlier 2..127-row experiment: every call has at most 63 rows, so the executed
branches are identical. Chinese chunk and GSM8K are reused from the scoped
experiment: all prompts exceed 63 rows and every internal call is at most 63,
with the same M=1 decode behavior. Chinese covers 651 documents / 2,604 responses;
GSM8K covers only 16 fixed five-shot questions, greedy with a 256-token cap.
Chinese full option inputs have 73..396 rows, so none enters the changed branch;
this is branch accounting rather than an additional full-corpus execution.

The user explicitly accepted the measured Chinese difference on 2026-09-14.
Raw has 19 gains / 23 losses; normalized has 19 gains / 21 losses. Paired 95% intervals
include zero; they do not prove equivalence. This acceptance does not waive
quality checks for future P10–P13 changes. Default experts remain eager.

## Runtime and performance

Real TP4 packed-cache/PIECEWISE-graph validation passes with two and three
concurrent image requests, including reorder, preemption, prefix forks, final
abort and release of the last vision lease. Model/operator regression has 44
passing tests, including real TP4 projection shapes and graph input updates.

Isolated operator timing includes padding and copying. At M=63, wo_a rises from
14.41us to 21.75us; full mHC mix prediction rises from 398.40us to 406.54us. At M=2,
wo_a rises from 11.07us to 21.22us. Padding is a quality workaround with measurable
cost, not a kernel speedup.

Sequential TP4 ModelRunner/Scheduler measurements use packed KV, PIECEWISE
graphs, eager experts, chunk63,16 output tokens and five warm repeats. They
include scheduling and Engram work, excluding loading, graph capture and HTTP.
All four cases generate identical IDs in both versions and across all ranks.

| Requests / prompt tokens | TTFT native → P08 (ms) | TPOT native → P08 (ms) | Output throughput change |
| --- | ---: | ---: | ---: |
|1 /128|1921.6 →1988.0|205.94 →207.88|−1.77%|
|2 /128|2269.9 →2291.7|227.50 →235.51|−2.41%|
|3 /128|2418.7 →2478.8|244.23 →251.07|−2.46%|
|1 /1024|12739.0 →13475.6|204.27 →212.37|−5.17%|

Peak allocated HBM is unchanged in these cases. These bounded measurements
show a cost, not a speedup or a guarantee for other loads. Further projection
and mHC optimization belongs to P11 and must preserve the accepted quality.

Commands, hashes, paired reports, rejected candidates and runtime traces are
archived under `/app/logs_claude/atom_dsv41_flash_impl_20260912/p08_multimodal/` in
`ljin_dev`, with a host copy under `/home/ljin1/dk/logs_claude/`. The final
operator-only run directory is `small_row_padding_20260914_8dfjq_ae`.
