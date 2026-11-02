# MiniMax-M3 top-k reuse across pipeline stages

With `use_index_cache=true` and `index_topk_freq=4`, a 60-layer M3.1 model
computes top-k at layers 0, 4, ..., 56. PP4's default `15,15,15,15` split
starts stages 1–3 on reuse layers 15/30/45. Those stages now import the
selection from layers 12/28/44 before their first attention call.

The model wrapper uses the existing sparse-layer schedule (including pattern
and offset overrides) to decide whether each boundary needs a payload. It sends
logical block IDs, and the receiving stage reconstructs physical pages from its
own block table. Each TP rank sends its own index heads without the activation
send/all-gather optimization. A stage containing only dense layers relays the
selection if a later sparse layer needs it. Per-forward metadata owns the memo,
so consecutive steps and interleaved microbatches do not share selections.

PP continues to use its existing eager execution mode. The receiver supports
FP8/BF16 page-16, MXFP4 page-128, and NVFP4 (which consumes logical IDs directly).
The transport does not change how a compute layer selects top-k or the TP-only
path. End-to-end validation below exercises PP prefill; the unit and transport
checks also cover decode geometry and TP ranks within a stage.

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
python -m pytest tests/test_minimax_m3_index_topk.py -k received_topk -q
# Four GPUs; real AITER tensor-dict transport, including async buffer lifetime:
torchrun --standalone --nproc-per-node=4 tests/minimax_m3_pp_topk_comm_smoke.py
```
