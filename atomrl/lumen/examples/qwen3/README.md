# Qwen3-8B LoRA SFT — PyTorch FSDP + Lumen FP8 blockwise2d

Supervised fine-tuning of **Qwen3-8B** with LoRA on **8 GPUs**, using PyTorch
FSDP for sharding and **Lumen** for FP8 **blockwise2d** linear quantization. Same
recipe as the Llama2-7B FSDP `fp8_blockwise2d` run (`../llama2`): `LumenConfig.enable`
applies FP8 blockwise2d linear quant + LoRA; attention and norm stay BF16.

## Layout

| Path | Purpose |
|---|---|
| `train_qwen3_fsdp.py` | Training script (FSDP + `LumenConfig.enable`; `--mode bf16|fp8_blockwise2d|mxfp4`) |
| `run_qwen3_fsdp.sh` | Docker launcher for 8 GPUs |
| `scripts/download_model.py` | Fetch the HF model checkpoint |
| `scripts/download_dataset.py` | Fetch an alpaca-style dataset as jsonl |
| `requirements.txt` | Python deps (`transformers`, `peft`, `datasets`, ...) |
| `results/` | Saved training logs + performance notes/report |

## Data

The script consumes alpaca-style jsonl rows `{instruction, input, output}`,
rendered through the **Qwen3 chat template** with an **answer-only loss mask**
(loss only on assistant tokens). The Llama2 gov_report `.npy` cannot be reused —
it is pre-tokenized with the Llama2 tokenizer.

```bash
python scripts/download_model.py   --model_name Qwen/Qwen3-8B --output_dir /data/Qwen3-8B
python scripts/download_dataset.py --dataset_name tatsu-lab/alpaca --output_dir /data/alpaca
```

> The reference run used a local Chinese alpaca set
> (`alpaca_zh-{train,valid}-general.jsonl`); any `{instruction, input, output}`
> jsonl works.

## Run

```bash
HOST_MODEL=/data/Qwen3-8B HOST_DATA=/data/alpaca \
TRAIN_FILE=train.jsonl VAL_FILE=validation.jsonl \
  bash run_qwen3_fsdp.sh
```

Overridable env: `HOST_MODEL`, `HOST_DATA`, `HOST_RESULTS`, `TRAIN_FILE`,
`VAL_FILE`, `SEQ_LENGTH` (default 2048), `MAX_STEPS` (200), `EVAL_INTERVAL` (50),
`PYTORCH_TUNABLEOP_ENABLED` (0), plus the optimization flags below.

Direct `torchrun` (inside the container):

```bash
torchrun --nproc_per_node=8 train_qwen3_fsdp.py \
  --model-name-or-path /model-qwen3 \
  --train-data-path /data/train.jsonl --val-data-path /data/validation.jsonl \
  --seq-length 2048 --max-steps 200 --eval-interval 50 --seed 1234
```

MXFP4 mode requires gfx950. `run_qwen3_fsdp.sh` is the BF16 / FP8 blockwise2d
launcher and rejects `MODE=mxfp4`. On gfx950, LoRA SFT is `run_qwen3_fsdp_mxfp4_sft.sh` and
full-parameter pretraining is `run_qwen3_fsdp_mxfp4_pretrain.sh`. Packed-FP4
parameter all-gather is opt-in and requires FSDP2:

```bash
torchrun --nproc_per_node=8 train_qwen3_fsdp.py \
  --model-name-or-path /model-qwen3 \
  --train-data-path /data/train.jsonl \
  --mode mxfp4 --fsdp-version 2 --fsdp-mxfp4-comm
```

Without `--fsdp-mxfp4-comm`, MXFP4 still quantizes Linear compute but FSDP
communicates the BF16 master weights.

A weight is eligible for FP4 communication only if it is 2D, MXFP4-patched, and
shaped so each rank's row shard stays block-aligned (`N % (32 * world_size) == 0`
and `K % 32 == 0`); Qwen3-8B's `lm_head` fails this and stays BF16. If the flag
matches no weight at all, `apply_fsdp2` raises rather than quietly all-gathering
BF16 under a config whose logs read as compressed.

### Full-parameter pretraining

`run_qwen3_fsdp_mxfp4_pretrain.sh` runs full-parameter (non-LoRA) causal-LM
pretraining on raw text or `{"text": ...}` jsonl:

```bash
MODEL_PATH=/data/Qwen3-8B TRAIN_DATA_PATH=/data/c4_train.jsonl \
PRECISION=mxfp4 SEQ_LEN=8192 TRAIN_STEPS=50 \
  bash examples/qwen3/run_qwen3_fsdp_mxfp4_pretrain.sh
```

Overridable env: `PRECISION` (`mxfp4|bf16`), `MXFP4_COMM`, `NPROC`, `MBS`, `GBS`,
`SEQ_LEN`, `TRAIN_STEPS`, `LR`, `WARMUP_STEPS`, `INIT_FROM_SCRATCH`,
`TRAIN_SAMPLES`, `RESULTS_DIR`. `TRAIN_SAMPLES` defaults to what the requested
step count consumes, so startup does not tokenize a corpus far larger than the
run needs. `--task pretrain` requires `--lora-rank 0` and defaults the MXFP4
recipe's BF16 tail to the last 5 layers, matching the Megatron runs.

Any FSDP trainer that enables MXFP4 must call
`atomrl.lumen.models.fsdp.register_quant_optimizer_hooks` after building its optimizer.
The MXFP4 weight cache otherwise falls back to keying on the parameter's
`_version`, which under FSDP tracks a reused all-gather buffer rather than the
sharded parameter the optimizer updates — every quantized layer then keeps
training against its step-0 weights, with no error and a loss and grad norm that
simply stop moving.

## Optimizations

The base recipe above is the **unoptimized FP8 baseline** (~2.78 s/step, ~61% slower
than BF16 — overhead-bound). A series of **lossless** (loss bit-identical) and
**memory-for-speed** optimizations bring it down to **~568 ms/step** (faster than BF16).
Full analysis: `results/qwen3_fp8_perf_notes.md` (not included in this repository) and the
visual report `results/qwen3_fp8_optimization_report.html` (not included in this repository);
per-operator breakdown in `results/qwen3_fsdp2_operator_breakdown.xlsx` (not included in this repository).

| Env flag | Effect | Type |
|---|---|---|
| `CACHE_FROZEN_WEIGHT=1` | Quantize the frozen LoRA base weight to FP8 **once** (not every forward); also caches the DGrad transpose and skips the frozen-weight WGrad | lossless |
| `BPRESHUFFLE=1` | Pre-shuffle the cached FP8 weight into an engine-friendly layout → blockscale GEMM ~2.6× (needs `CACHE_FROZEN_WEIGHT`) | lossless |
| `AITER_ATTN=1` | AITER CK FMHA v3 attention instead of PyTorch AOTriton SDPA | lossless |
| `LUMEN_NORM=1` | Fused RMSNorm (`Qwen3RMSNorm` → `LumenRMSNorm`) | lossless |
| `FUSE_ROPE=1` | Fused RoPE (HF `apply_rotary_pos_emb` → AITER autograd RoPE) | lossless |
| `GRAD_CKPT=0` | Disable gradient checkpointing — no backward recompute (needs activation memory) | memory-for-speed |
| `SHARDING=shard_grad_op` | ZeRO-2: params stay resident, no backward all-gather (pair with `GRAD_CKPT=0`) | memory-for-speed |
| `LIMIT_ALL_GATHERS=0` `FORWARD_PREFETCH=1` | Loosen FSDP all-gather throttling / prefetch | memory-for-speed |
| `FSDP_VERSION=2` | Use FSDP2 (`fully_shard`, per-parameter sharding) | — |
| `FSDP_FP8_PARAM_STORAGE=1` | Store the frozen base as FP8 **in the shard** (all-gather FP8, not BF16) — for large models that can't cache the full FP8 weight (e.g. 70B); needs FSDP2. Not the fastest 8B path | memory |

Fastest 8B config (memory permitting):

```bash
FSDP_VERSION=2 CACHE_FROZEN_WEIGHT=1 BPRESHUFFLE=1 GRAD_CKPT=0 \
SHARDING=shard_grad_op LIMIT_ALL_GATHERS=0 FORWARD_PREFETCH=1 \
AITER_ATTN=1 LUMEN_NORM=1 FUSE_ROPE=1 MODE=fp8_blockwise2d \
HOST_MODEL=/data/Qwen3-8B HOST_DATA=/data/alpaca \
  bash run_qwen3_fsdp.sh
```

| stage | step time |
|---|---|
| FP8 baseline | 2784 ms |
| + frozen-weight cache + skip WGrad | 1912 ms |
| + no grad-ckpt + SHARD_GRAD_OP | 1542 ms |
| + bpreshuffle GEMM (FSDP1 best) | 960 ms |
| + FSDP2 + AITER attn + fused RMSNorm/RoPE | 634 ms |
| + RMSNorm-bwd large-M-small-N kernel (AITER) | 603 ms |
| + FP8 GEMM gfx942 tune (AITER) | 578 ms |
| + zero-copy RoPE layout (Lumen) | **568 ms** |

> The last three are: an AITER RMSNorm-bwd kernel (q/k per-head norm row-parallel
> specialization, 37→9 ms/step), gfx942 bpreshuffle GEMM tuning (FP8 GEMM 295→277 ms),
> and a Lumen zero-copy RoPE layout (drops ~2.3 GB/step of contiguous copies). The two
> AITER items ship in the pinned official `/opt/aiter` checkout (effective once built into the image); the RoPE
> change is in `lumen/` (default-on). All numerically equivalent / exact.

When memory is tight (large model / batch / seq), drop `GRAD_CKPT=0` and `SHARDING`;
the lossless lib-level flags alone still give ~−31%.

## Recipe

| Item | Value |
|---|---|
| Sharding | FSDP1 `FULL_SHARD` (or FSDP2 `fully_shard` via `FSDP_VERSION=2`), `use_orig_params=True`, DP=8 |
| Precision | FP8 e4m3 **blockwise2d** (128×128 tiles) on linears; attention/norm BF16 |
| LoRA | rank 16, alpha 32, dropout 0.1 — q/k/v/o/gate/up/down (~0.53% trainable) |
| Seq length / micro-batch / global batch | 2048 / 1 / 8 |
| Optimizer / schedule | AdamW (β=0.9,0.95, eps=1e-5, wd=1e-4) / cosine→0, grad clip 0.3 |
| LR / steps | 4e-4 / 200 |

## Reference result (8×MI308X)

| step | val_loss (answer-only) |
|---|---|
| 50 | 1.5968 |
| 100 | 1.6086 |
| 150 | 1.5771 |
| 200 | 1.5741 |

0 NaN; val_loss is **bit-identical** across all optimization stages (the optimizations
change only kernel selection / scheduling, not the math). Logs:

| Log | Config |
|---|---|
| `results/qwen3_fsdp_train_bf16_lora_8b.log` (not included in this repository) | BF16 reference (~1.73 s/step) |
| `results/qwen3_fsdp1_train_fp8_blockwise2d_lora_8b.log` (not included in this repository) | FSDP1 FP8 baseline (~2.78 s/step) |
| `results/qwen3_fsdp2_train_fp8_blockwise2d_best_lora_8b.log` (not included in this repository) | FSDP2 fully optimized — RMSNorm kernel + gfx942 GEMM tune + zero-copy RoPE (~568 ms/step) |

## Megatron Pretraining (Qwen3-8B, 8×MI325X)

`run_pretrain_qwen3_8b.sh` is a self-contained launcher that runs Qwen3-8B in
either BF16 or FP8 delayed/hybrid with all validated Lumen fusion
optimizations, using mock data (no dataset download needed).

### Launch

```bash
# FP8 delayed/hybrid (default) — 50 steps on 8 GPUs
bash examples/qwen3/run_pretrain_qwen3_8b.sh

# BF16
PRECISION=bf16 bash examples/qwen3/run_pretrain_qwen3_8b.sh

# Override common knobs (defaults shown)
PRECISION=fp8 MBS=2 GBS=128 SEQ_LEN=8192 TRAIN_STEPS=50 \
    bash examples/qwen3/run_pretrain_qwen3_8b.sh
```

### Environment overrides

| Variable | Default | Notes |
|----------|---------|-------|
| `PRECISION` | `fp8` | `fp8` (delayed/hybrid) or `bf16` |
| `IMAGE` | `lumen:dev` | Docker image |
| `MBS` / `GBS` | `2` / `128` | micro / global batch size |
| `SEQ_LEN` | `8192` | sequence length |
| `TRAIN_STEPS` | `50` | training iterations |
| `TOKENIZER_DIR` | `examples/qwen3/tokenizer` | HuggingFace tokenizer dir; fetched from `Qwen/Qwen3-8B-Base` on first run |
| `RESULTS_DIR` | `examples/qwen3/results` | logs + mock data output |

FP8 mode enables the validated forward optimizations (fused quant+amax, fused
cast-transpose, fused SwiGLU/norm quant, transpose cache, weight-quant-once,
etc.) plus `--linear-fp8 --fp8-format hybrid --linear-fp8-scaling delayed`.
BF16 keeps only the precision-agnostic fusions (SwiGLU, residual-norm).

### Model config

36 layers · hidden 4096 · FFN 12288 · 32 heads · GQA 8 KV groups · kv-channels 128 · RoPE base 1e6 · RMSNorm · SwiGLU.

### Reference results (50 steps, steady-state step time)

| Precision | Log | step time |
|-----------|-----|-----------|
| BF16 | `results/qwen3_8b_pretrain_bf16.log` (not included in this repository) | ~14.6 s |
| FP8 delayed | `results/qwen3_8b_pretrain_fp8_delayed.log` (not included in this repository) | ~10.3 s |

### Lumen vs TransformerEngine (8×MI325X)

Same config (MBS=2, GBS=128, seq=8192, TP=1, PP=1), steady-state step time and
peak allocated memory:

| Precision | Framework | step time | peak memory |
|-----------|-----------|-----------|-------------|
| BF16 | TransformerEngine | ~15.54 s | 133.8 GB |
| BF16 | Lumen | ~14.90 s | 138.5 GB |
| BF16 | Δ | −4.1% | +3.5% |
| FP8 delayed | TransformerEngine | ~11.60 s | 137.8 GB |
| FP8 delayed | Lumen | ~10.44 s | 138.2 GB |
| FP8 delayed | Δ | −10.0% | +0.3% |
