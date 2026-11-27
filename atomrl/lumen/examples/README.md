# Lumen Examples Runbook

Small, single-node training recipes used to check a new feature. Each row is one
command, and every command runs 20 steps. Rows are ordered by model, then by
precision: BF16, FP8 delayed, FP8 blockwise, MXFP4.

> 中文：[README_cn.md](README_cn.md)

The step times in the last table come from one 20-step run on one image, done
for this page. The per-example READMEs have their own numbers, measured under
different settings and not comparable to this table.

MXFP4 runs on gfx950 (MI350 / MI355). `run_qwen3_fsdp.sh` refuses `MODE=mxfp4`.

Runs above 10B, and the two-node Qwen3-30B-A3B run, are not listed here. The
Llama3.1 real-data pretrain is `examples/llama31/run_pretrain.sh`. Details are
in [llama2](llama2/README.md), [llama31](llama31/README.md),
[qwen3](qwen3/README.md), and [dsv4](dsv4/README.md).

## Container setup

The examples support two dependency strategies:

- **Latest source**: build `atomrl/docker/Dockerfile.lumen`. It fetches the
  current AITER revision from the upstream repository.
- **Fixed source**: build `atomrl/docker/Dockerfile.lumenrelease`. It uses the
  fixed AITER revision recorded in that Dockerfile.

Both images install the current `atom` and `atomrl` checkouts, plus the
`peft` and `accelerate` versions required by the examples. From the repository
root, run one of:

```bash
docker build --pull --no-cache \
  --build-arg AITER_REF=main \
  -f atomrl/docker/Dockerfile.lumen \
  -t atomrl:lumen-latest .

docker build --pull --no-cache \
  -f atomrl/docker/Dockerfile.lumenrelease \
  -t atomrl:lumen-release .
```

If you do not need to rebuild the image, pull the fixed Lumen examples image:

```bash
docker pull rocm/atom-dev:lumen-examples-runbook-gfx950-rocm7.14.0-261107
```

The CI README-case shard uses the latest-source `Dockerfile.lumen` build.

## Required rows

Run rows **1, 6, 10, 12, and 13**. Together they cover BF16, FP8 delayed, FP8
blockwise, MXFP4, and the FSDP path. The other rows run only when that path
changes.

| # | What it covers | Launch |
|---|---|---|
| 1 | Llama2-7B, BF16, Megatron | `PRECISION=bf16 TRAIN_STEPS=20 bash examples/llama2/run_pretrain_llama2_7b.sh` |
| 6 | Llama3.1-8B, FP8 delayed, Megatron | `TRAIN_STEPS=20 bash examples/llama31/run_pretrain_llama31_8b.sh` |
| 10 | Qwen3-8B, FP8 blockwise, Megatron | `MODEL=qwen3 TRAIN_STEPS=20 bash examples/run_fp8_blockwise_pretrain.sh` |
| 12 | Qwen3-8B, MXFP4, Megatron | `TRAIN_STEPS=20 bash examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh` |
| 13 | Qwen3-8B, MXFP4, FSDP2 LoRA | `TRAIN_STEPS=20 WARMUP_STEPS=2 bash examples/qwen3/run_qwen3_fsdp_mxfp4_sft.sh` |

---

## Llama2-7B

| # | Run | Precision | Backend | Case | GPU | Launch |
|---|---|---|---|---|---|---|
| 1 | required | BF16 | Megatron | Pretrain, mock data | 8× MI350X | `PRECISION=bf16 TRAIN_STEPS=20 bash examples/llama2/run_pretrain_llama2_7b.sh` |
| 2 | if changed | FP8 delayed | Megatron | Pretrain, mock data | 8× MI350X | `TRAIN_STEPS=20 bash examples/llama2/run_pretrain_llama2_7b.sh` |
| 3 | if changed | FP8 blockwise | Megatron | 20-step smoke, mock data | 8× MI350X | `MODEL=llama2 TRAIN_STEPS=20 bash examples/run_fp8_blockwise_pretrain.sh` |
| 4 | if changed | FP8 blockwise | FSDP | LoRA SFT | 8× MI350X | `MODE=fp8_blockwise SEQ_LEN=2048 TRAIN_STEPS=20 bash examples/llama2/run_fsdp_lora_7b.sh` |

Row 2 is hybrid FP8 with delayed scaling. Row 4's launcher also accepts
`fp8_delayed` and `fp8_blockwise2d`; the row is the blockwise run. It needs
`HOST_MODEL` (HuggingFace Llama-2-7b) and `HOST_DATA` (a directory with
`train.npy` and `validation.npy`). Logs go to `examples/llama2/results/`.

---

## Llama3.1-8B

| # | Run | Precision | Backend | Case | GPU | Launch |
|---|---|---|---|---|---|---|
| 5 | if changed | BF16 | Megatron | Pretrain, mock data | 8× MI350X | `PRECISION=bf16 TRAIN_STEPS=20 bash examples/llama31/run_pretrain_llama31_8b.sh` |
| 6 | required | FP8 delayed | Megatron | Pretrain, mock data | 8× MI350X | `TRAIN_STEPS=20 bash examples/llama31/run_pretrain_llama31_8b.sh` |
| 7 | if changed | FP8 blockwise | Megatron | 20-step smoke, mock data | 8× MI350X | `MODEL=llama31 TRAIN_STEPS=20 bash examples/run_fp8_blockwise_pretrain.sh` |

Row 6 is hybrid FP8 with delayed scaling.

---

## Qwen3-8B

| # | Run | Precision | Backend | Case | GPU | Launch |
|---|---|---|---|---|---|---|
| 8 | if changed | BF16 | Megatron | Pretrain, mock data | 8× MI350X | `PRECISION=bf16 TRAIN_STEPS=20 bash examples/qwen3/run_pretrain_qwen3_8b.sh` |
| 9 | if changed | FP8 delayed | Megatron | Pretrain, mock data | 8× MI350X | `TRAIN_STEPS=20 bash examples/qwen3/run_pretrain_qwen3_8b.sh` |
| 10 | required | FP8 blockwise | Megatron | 20-step smoke, mock data | 8× MI350X | `MODEL=qwen3 TRAIN_STEPS=20 bash examples/run_fp8_blockwise_pretrain.sh` |
| 11 | if changed | FP8 blockwise | FSDP / FSDP2 | LoRA SFT, e4m3 blockwise2d | 8× MI350X | `MAX_STEPS=20 EVAL_INTERVAL=20 bash examples/qwen3/run_qwen3_fsdp.sh` |
| 12 | required | MXFP4 | Megatron | Pretrain, mock data | 8× MI350X | `TRAIN_STEPS=20 bash examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh` |
| 13 | required | MXFP4 | FSDP2 | LoRA SFT | 8× MI350X | `TRAIN_STEPS=20 WARMUP_STEPS=2 bash examples/qwen3/run_qwen3_fsdp_mxfp4_sft.sh` |

Row 8 is the BF16 baseline for both row 9 and row 12. Row 9 is hybrid FP8 with
delayed scaling. The Megatron MXFP4 recipe keeps the last 5 layers in BF16.

Row 12 also takes `LAUNCH=native` (no Docker). Its image is `lumen:dev`;
`examples/qwen3/build_mxfp4_lumen_image.sh` builds `lumen/mxfp4-lumen:gfx950`
with the Lumen tree baked in.

Row 13 uses `run_qwen3_fsdp_mxfp4_sft.sh`. It needs `MODEL_PATH` and an
alpaca-style `TRAIN_DATA_PATH`. Row 11's `run_qwen3_fsdp.sh` needs
`HOST_MODEL` and `HOST_DATA` and exits on `MODE=mxfp4`. `MXFP4_COMM=1` opts
into packed-FP4 all-gather. The flags and
the shapes that stay BF16 are in [qwen3/README.md](qwen3/README.md).

`run_qwen3_fsdp_mxfp4_pretrain.sh` is the full-parameter FSDP2 pretrain and is
not a numbered row. On 8× MI350X, GBS 16, seq 8192, its median over steps
21–250 is 1,310 ms with `--aiter-attn --lumen-norm --fuse-rope`, against
1,844 ms for MXFP4 alone and 1,951 ms for BF16.

`examples/scripts/train_pretrain.sh` is the body behind row 12 and accepts
`MODEL=llama2_7b|llama31_8b|qwen3_8b` with `PRECISION=bf16|fp8|mxfp4`. Only
Qwen3-8B has a Docker wrapper for the MXFP4 arm; the other two models run by
exporting `LUMEN_ROOT`, `MEGATRON_ROOT`, `RESULTS_ROOT`, and `TOKENIZER_PATH`
and calling that script directly. `DRY_RUN=1` prints the `torchrun` line.

---

## 20 steps, 8× MI350X

The benchmark table below was measured with the historical runbook image and
is kept for comparison. Micro-batch 1, global batch 8, seq 2048, seed 1234.
Step time is the median of steps 2–20.

| # | Step time | Loss at step 20 | Val loss |
|---|---|---|---|
| 1 Llama2-7B BF16 | 268 ms | 2.53 | 2.49 |
| 2 Llama2-7B FP8 delayed | 310 ms | 2.64 | 2.59 |
| 3 Llama2-7B FP8 blockwise | 354 ms | 2.36 | 2.34 |
| 4 Llama2-7B FP8 blockwise FSDP LoRA | 610 ms | 1.84 | 1.84 |
| 5 Llama3.1-8B BF16 | 305 ms | 6.42 | 6.39 |
| 6 Llama3.1-8B FP8 delayed | 331 ms | 6.73 | 6.68 |
| 7 Llama3.1-8B FP8 blockwise | 370 ms | 6.80 | 6.75 |
| 8 Qwen3-8B BF16 | 312 ms | 2.70 | 2.67 |
| 9 Qwen3-8B FP8 delayed | 355 ms | 2.52 | 2.52 |
| 10 Qwen3-8B FP8 blockwise | 378 ms | 2.51 | 2.49 |
| 11 Qwen3-8B e4m3 blockwise2d FSDP LoRA | 720 ms | 1.90 | 1.91 |
| 12 Qwen3-8B MXFP4 | 294 ms | 2.93 | 2.91 |
| 13 Qwen3-8B MXFP4 FSDP2 LoRA | 570 ms | 1.90 | 1.91 |
