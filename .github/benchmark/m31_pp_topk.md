# MiniMax-M3 top-k reuse across pipeline stages

With `use_index_cache=true` and `index_topk_freq=4`, a 60-layer M3.1 model
computes top-k at layers 0, 4, ..., 56. PP4's default `15,15,15,15` split
starts stages 1–3 on reuse layers 15/30/45. Those stages now import the
selection from layers 12/28/44 before their first attention call.

The existing ModelRunner shared-indexer hooks handle both GLM and M3. M3
uses its existing sparse-layer schedule (including pattern and offset) to choose
boundaries, then exports/imports its cached `(indices, block_table, context_lens)`
value. The head stage assigns block IDs and fans out the same scheduled batch;
with the same TP geometry and KV layout across stages, physical page IDs and
lengths are reusable too. The receiver binds the value to its local memo key.
A dense-only stage relays the value until the next sparse layer needs it.

The PP tensor-dict helpers accept `rank_local_keys` for tensors that are not
replicated across TP. They send those tensors without all-gather, while keeping
the existing activation send/all-gather optimization. Both sends use the same
AITER tensor-dict protocol and the existing async buffer ownership mechanism.
There is no model-specific transport header or new page-table kernel.

PP remains eager. Each forward gets its own M3 memo, so previous steps and
interleaved microbatches cannot supply stale selections. FP8/BF16 and MXFP4
carry all three cached tensors; `None` page tables/lengths are also supported
for NVFP4's IDs-only result. Full-model validation below exercises PP prefill;
unit checks cover both attention phases, and the four-GPU transport check
covers TP ranks within each stage.

## Regression experiment

The `MiniMax-M3.1-Flash-MXFP4-Stack-PP-TopK` model family contains two accuracy
cases on the same MXFP4 feature stack:

| Case | Prefill layer partition |
| --- | --- |
| `m31-stack-mxfp4-pp4-index-freq4-equal-gsm8k20` | `15,15,15,15` |
| `m31-stack-mxfp4-pp4-index-freq4-aligned-gsm8k20` | `16,16,16,12` |

Both use PP4/TP1 prefill, TP4 decode, MXFP4 KV/index, DSpark7, and full GSM8K
20-shot at concurrency 32. Maximum generation is 16384 tokens. The prefill
batch-token limit is 8192. Image: `rocm/atom-dev:m31-mxfp4-stack-20261031`.
This image includes the pinned AITER build, so the cases skip reinstalling it.

Dispatch `atomesh-benchmark.yaml` on the change's branch with `suite=accuracy`,
`atom_source=mounted`, `case_names` set to both names above,
`atomesh_image=rocm/atom-dev:m31-mxfp4-stack-20261031`,
`run_model_benchmark=true`, `publish_dashboard=false`,
`atomesh_slurm_submit_runner=atomesh-cicd`, and `atomesh_single_node=auto`.
The expanded cells have no fixed nodes; account/partition/qos are
`amd-atom` / `amd-spur` / `amd-atom-qos`.

Check completion of the first prefill step, full evaluation completion, both
GSM8K extraction metrics, and per-question differences. The accuracy threshold
alone is not evidence of parity. The aligned split provides a control that
needs no cross-stage top-k payload.

## Local checks

```bash
python -m pytest tests/test_minimax_m3_pp_topk.py tests/test_pp.py \
  tests/test_minimax_m3_index_cache_rows.py -q
# Four GPUs; real AITER tensor-dict transport, including async buffer lifetime:
torchrun --standalone --nproc-per-node=4 tests/minimax_m3_pp_topk_comm_smoke.py
```
