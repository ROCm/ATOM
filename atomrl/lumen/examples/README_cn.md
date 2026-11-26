# Lumen Examples Runbook

用来检查新功能的小模型、单机训练。一行一个命令，每条都跑 20 step。先按模型排，再按精度排：BF16、FP8 delayed、FP8 blockwise、MXFP4。

> English: [README.md](README.md)

文末的 step time 是这页单独跑的一组 20 step 结果，同一镜像。各例子自己的 README 里是另外的数字，测量设置不同，和文末的表不可直接比较。

MXFP4 跑在 gfx950（MI350 / MI355）。`run_qwen3_fsdp.sh` 遇到 `MODE=mxfp4` 会直接退出。

超过 10B 的跑法，以及两机的 Qwen3-30B-A3B，不列在这里。Llama3.1 的真实数据预训练是 `examples/llama31/run_pretrain.sh`。细节见 [llama2](llama2/README.md)、[llama31](llama31/README.md)、[qwen3](qwen3/README.md)、[dsv4](dsv4/README.md)。

## Docker 环境

这些例子支持两种依赖策略：

- **最新源码**：使用 `atomrl/docker/Dockerfile.lumen`，从上游仓库拉取最新
  AITER。
- **固定版本**：使用 `atomrl/docker/Dockerfile.lumenrelease`，使用该文件中
  记录的固定 AITER 版本。

两个镜像都会安装当前 checkout 中的 `atom`、`atomrl`，以及例子需要的
`peft` 和 `accelerate` 版本。在仓库根目录执行下面任意一种构建：

```bash
docker build --pull --no-cache \
  --build-arg AITER_REF=main \
  -f atomrl/docker/Dockerfile.lumen \
  -t atomrl:lumen-latest .

docker build --pull --no-cache \
  -f atomrl/docker/Dockerfile.lumenrelease \
  -t atomrl:lumen-release .
```

如果不需要重新构建，也可以直接拉取固定的 Lumen examples 镜像：

```bash
docker pull rocm/atom-dev:lumen-examples-runbook-gfx950-rocm7.14.0-261107
```

CI 的 README case 使用最新源码版 `Dockerfile.lumen`。

## 必跑的行

跑第 **1、6、10、12、13** 行。这五行合在一起覆盖 BF16、FP8 delayed、FP8 blockwise、MXFP4 和 FSDP。其余行只有改到对应路径时再跑。

| # | 覆盖 | 启动 |
|---|---|---|
| 1 | Llama2-7B，BF16，Megatron | `PRECISION=bf16 TRAIN_STEPS=20 bash examples/llama2/run_pretrain_llama2_7b.sh` |
| 6 | Llama3.1-8B，FP8 delayed，Megatron | `TRAIN_STEPS=20 bash examples/llama31/run_pretrain_llama31_8b.sh` |
| 10 | Qwen3-8B，FP8 blockwise，Megatron | `MODEL=qwen3 TRAIN_STEPS=20 bash examples/run_fp8_blockwise_pretrain.sh` |
| 12 | Qwen3-8B，MXFP4，Megatron | `TRAIN_STEPS=20 bash examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh` |
| 13 | Qwen3-8B，MXFP4，FSDP2 LoRA | `TRAIN_STEPS=20 WARMUP_STEPS=2 bash examples/qwen3/run_qwen3_fsdp_mxfp4_sft.sh` |

---

## Llama2-7B

| # | 何时跑 | 精度 | 后端 | 例子 | GPU | 启动 |
|---|---|---|---|---|---|---|
| 1 | 每次必跑 | BF16 | Megatron | 预训练，mock 数据 | 8× MI350X | `PRECISION=bf16 TRAIN_STEPS=20 bash examples/llama2/run_pretrain_llama2_7b.sh` |
| 2 | 按需 | FP8 delayed | Megatron | 预训练，mock 数据 | 8× MI350X | `TRAIN_STEPS=20 bash examples/llama2/run_pretrain_llama2_7b.sh` |
| 3 | 按需 | FP8 blockwise | Megatron | 20 step smoke，mock 数据 | 8× MI350X | `MODEL=llama2 TRAIN_STEPS=20 bash examples/run_fp8_blockwise_pretrain.sh` |
| 4 | 按需 | FP8 blockwise | FSDP | LoRA SFT | 8× MI350X | `MODE=fp8_blockwise SEQ_LEN=2048 TRAIN_STEPS=20 bash examples/llama2/run_fsdp_lora_7b.sh` |

第 2 行是 hybrid FP8、delayed scaling。第 4 行的启动脚本也接受 `fp8_delayed` 和 `fp8_blockwise2d`，这一行是 blockwise。它需要 `HOST_MODEL`（HuggingFace Llama-2-7b）和 `HOST_DATA`（里面有 `train.npy` 和 `validation.npy`）。日志写到 `examples/llama2/results/`。

---

## Llama3.1-8B

| # | 何时跑 | 精度 | 后端 | 例子 | GPU | 启动 |
|---|---|---|---|---|---|---|
| 5 | 按需 | BF16 | Megatron | 预训练，mock 数据 | 8× MI350X | `PRECISION=bf16 TRAIN_STEPS=20 bash examples/llama31/run_pretrain_llama31_8b.sh` |
| 6 | 每次必跑 | FP8 delayed | Megatron | 预训练，mock 数据 | 8× MI350X | `TRAIN_STEPS=20 bash examples/llama31/run_pretrain_llama31_8b.sh` |
| 7 | 按需 | FP8 blockwise | Megatron | 20 step smoke，mock 数据 | 8× MI350X | `MODEL=llama31 TRAIN_STEPS=20 bash examples/run_fp8_blockwise_pretrain.sh` |

第 6 行是 hybrid FP8、delayed scaling。

---

## Qwen3-8B

| # | 何时跑 | 精度 | 后端 | 例子 | GPU | 启动 |
|---|---|---|---|---|---|---|
| 8 | 按需 | BF16 | Megatron | 预训练，mock 数据 | 8× MI350X | `PRECISION=bf16 TRAIN_STEPS=20 bash examples/qwen3/run_pretrain_qwen3_8b.sh` |
| 9 | 按需 | FP8 delayed | Megatron | 预训练，mock 数据 | 8× MI350X | `TRAIN_STEPS=20 bash examples/qwen3/run_pretrain_qwen3_8b.sh` |
| 10 | 每次必跑 | FP8 blockwise | Megatron | 20 step smoke，mock 数据 | 8× MI350X | `MODEL=qwen3 TRAIN_STEPS=20 bash examples/run_fp8_blockwise_pretrain.sh` |
| 11 | 按需 | FP8 blockwise | FSDP / FSDP2 | LoRA SFT，e4m3 blockwise2d | 8× MI350X | `MAX_STEPS=20 EVAL_INTERVAL=20 bash examples/qwen3/run_qwen3_fsdp.sh` |
| 12 | 每次必跑 | MXFP4 | Megatron | 预训练，mock 数据 | 8× MI350X | `TRAIN_STEPS=20 bash examples/qwen3/run_pretrain_qwen3_8b_mxfp4.sh` |
| 13 | 每次必跑 | MXFP4 | FSDP2 | LoRA SFT | 8× MI350X | `TRAIN_STEPS=20 WARMUP_STEPS=2 bash examples/qwen3/run_qwen3_fsdp_mxfp4_sft.sh` |

第 8 行同时是第 9 行和第 12 行的 BF16 基线。第 9 行是 hybrid FP8、delayed scaling。Megatron 的 MXFP4 配方最后 5 层保持 BF16。

第 12 行还可以 `LAUNCH=native`（不进 Docker）。镜像是 `lumen:dev`；`examples/qwen3/build_mxfp4_lumen_image.sh` 会打出带 Lumen 源码的 `lumen/mxfp4-lumen:gfx950`。

第 13 行走 `run_qwen3_fsdp_mxfp4_sft.sh`，需要 `MODEL_PATH` 和 alpaca 格式的 `TRAIN_DATA_PATH`。第 11 行的 `run_qwen3_fsdp.sh` 需要 `HOST_MODEL` 和 `HOST_DATA`，遇到 `MODE=mxfp4` 会退出。`MXFP4_COMM=1` 才打开 packed-FP4 all-gather。参数和哪些层留在 BF16 写在 [qwen3/README.md](qwen3/README.md)。

`run_qwen3_fsdp_mxfp4_pretrain.sh` 是 FSDP2 全参预训练，不在编号里。8×MI350X、GBS 16、seq 8192、step 21–250 的中位数：加上 `--aiter-attn --lumen-norm --fuse-rope` 是 1,310 ms，只开 MXFP4 是 1,844 ms，BF16 是 1,951 ms。

`examples/scripts/train_pretrain.sh` 是第 12 行的训练本体，接受 `MODEL=llama2_7b|llama31_8b|qwen3_8b` 和 `PRECISION=bf16|fp8|mxfp4`。MXFP4 只有 Qwen3-8B 有 Docker 包装；另外两个模型要自己 export `LUMEN_ROOT`、`MEGATRON_ROOT`、`RESULTS_ROOT`、`TOKENIZER_PATH` 再直接调这个脚本。`DRY_RUN=1` 只打印 `torchrun` 命令。

## CI 映射

Lumen GPU shard 会使用 `atomrl/docker/Dockerfile.lumen` 和 AITER `main` 构建镜像。
第 1–12 行直接在 runner 上调用原有 launcher，让内层容器拿到真实 host 路径；
第 13 行直接在同一个容器里执行。CI 会从共享模型目录查找 Llama-2-7b-hf（第 4 行）和
Qwen3-8B（第 11、13 行），找不到就报错且不会下载大模型；NumPy 和 Alpaca
JSONL smoke 数据由 CI 确定性生成。每个容器都把当前 checkout 挂到
`/workspace/ATOM`，并只用 `--no-deps` 刷新根目录 editable 安装。

---

## 20 step，8×MI350X

下面的表格来自历史 runbook 镜像，仅用于对比。micro-batch 1，global batch 8，
序列 2048，seed 1234。步时是第 2–20 step 的中位数。

| # | 步时 | 第 20 step loss | val loss |
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
