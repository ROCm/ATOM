# LLaMA 3.1 Pretraining

Pretraining LLaMA 3.1 (8B) with FP8 hybrid training and MXFP8 attention, aligned with MLPerf LLM pretraining config.

## Quick Start

```bash
# 1. Prepare data and model checkpoint
bash examples/llama31/scripts/prepare_data_and_model.sh

# 2. Run training — Megatron backend (default)
BACKEND=megatron bash examples/llama31/run_pretrain.sh

# 2. Or: FSDP backend (no Megatron dependency)
BACKEND=fsdp bash examples/llama31/run_pretrain.sh
```

The entry point (`pretrain_llama31.py`) selects the backend via `--backend megatron|fsdp`.

## CLI Flags

| Feature | CLI Flag | Default |
|---------|----------|---------|
| Model size | `SIZE=8b` (env var) | 8b |
| MXFP8 attention | `--lumen-fp8-quant-type mxfp8` | mxfp8 |
| FP8 training | `--linear-fp8` | enabled |
| Amax algorithm | `--linear-fp8-amax-algo most_recent` | most_recent |
| Amax history | `--linear-fp8-amax-history 4` | 4 |
| Learning rate | `MAX_LR=8e-4` (env var) | 8e-4 |
| Cosine LR warmup | `LR_WARMUP_STEPS=128` | 128 |
| GQA (8 KV heads) | auto from SIZE | 32 heads / 8 KV groups |

See `run_pretrain.sh` for all environment variables.

## Standalone Pretraining Launcher (Llama3.1-8B, 8×MI325X)

`run_pretrain_llama31_8b.sh` is a self-contained launcher that runs Llama3.1-8B
in either BF16 or FP8 delayed/hybrid with all validated Lumen fusion
optimizations, using mock data (no dataset download needed).

It applies SOURCE patches with ``--tag llama`` via
``examples/dsv4/patch_megatron_source.py`` before ``torchrun``.
The non-Docker ``run_pretrain.sh`` Megatron path does the same when ``MEGATRON_ROOT`` is set.

### Launch

```bash
# FP8 delayed/hybrid (default) — 50 steps on 8 GPUs
bash examples/llama31/run_pretrain_llama31_8b.sh

# BF16
PRECISION=bf16 bash examples/llama31/run_pretrain_llama31_8b.sh

# Override common knobs (defaults shown)
PRECISION=fp8 MBS=2 GBS=128 SEQ_LEN=8192 TRAIN_STEPS=50 \
    bash examples/llama31/run_pretrain_llama31_8b.sh
```

### Environment overrides

| Variable | Default | Notes |
|----------|---------|-------|
| `PRECISION` | `fp8` | `fp8` (delayed/hybrid) or `bf16` |
| `IMAGE` | `lumen:dev` | Docker image |
| `MBS` / `GBS` | `2` / `128` | micro / global batch size |
| `SEQ_LEN` | `8192` | sequence length |
| `TRAIN_STEPS` | `50` | training iterations |
| `TOKENIZER_DIR` | `examples/llama31/tokenizer` | HuggingFace tokenizer dir; fetched from public `NousResearch/Meta-Llama-3.1-8B` on first run |
| `RESULTS_DIR` | `examples/llama31/results` | logs + mock data output |

FP8 mode enables the validated forward optimizations (fused quant+amax, fused
cast-transpose, fused SwiGLU/norm quant, transpose cache, weight-quant-once,
etc.) plus `--linear-fp8 --fp8-format hybrid --linear-fp8-scaling delayed`.
BF16 keeps only the precision-agnostic fusions (SwiGLU, residual-norm).

### Model config

32 layers · hidden 4096 · FFN 14336 · 32 heads · GQA 8 KV groups · RoPE base 5e5 · RMSNorm · SwiGLU.

### Reference results (50 steps, steady-state step time)

| Precision | Log | step time |
|-----------|-----|-----------|
| BF16 | `results/llama31_8b_pretrain_bf16.log` (not included in this repository) | ~13.7 s |
| FP8 delayed | `results/llama31_8b_pretrain_fp8_delayed.log` (not included in this repository) | ~9.8 s |

### Lumen vs TransformerEngine (8×MI325X)

Same config (MBS=2, GBS=128, seq=8192, TP=1, PP=1), steady-state step time and
peak allocated memory:

| Precision | Framework | step time | peak memory |
|-----------|-----------|-----------|-------------|
| BF16 | TransformerEngine | ~13.94 s | 122.1 GB |
| BF16 | Lumen | ~13.66 s | 133.1 GB |
| BF16 | Δ | −2.1% | +9.0% |
| FP8 delayed | TransformerEngine | ~10.73 s | 126.1 GB |
| FP8 delayed | Lumen | ~10.00 s | 133.4 GB |
| FP8 delayed | Δ | −6.8% | +5.8% |
