# Qwen3-8B-FP8 (block-128) on RX 9070 XT (gfx1201) via ROCm/ATOM

Verified, all-Triton, cudagraph-on path. Same backend setup as the
[Ministral-3-8B recipe](./Ministral-3-8B.md).

## Model

[`Qwen/Qwen3-8B-FP8`](https://huggingface.co/Qwen/Qwen3-8B-FP8) —
official Qwen release, FineGrainedFP8 quant with
`weight_block_size=[128, 128]`, `activation_scheme="dynamic"`.
36 layers, hidden=4096, head_dim=128, num_q_heads=32, num_kv_heads=8 (GQA),
vocab=151936.

```bash
hf download Qwen/Qwen3-8B-FP8 \
  --local-dir /mnt/sda1/carhuang/models/Qwen3-8B-FP8
```

## Required env

```bash
export ATOM_USE_UNIFIED_ATTN=1   # route through TritonMHABackend (aiter triton unified_attention)
export ATOM_USE_TRITON_GEMM=1
export AITER_ROPE_NATIVE_BACKEND=1
export AITER_LOG_LEVEL=WARNING
export ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT=0
export ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT=0
export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0
```

`ATOM_LM_HEAD_FP8=1` (default on) lazily quantizes lm_head to FP8;
set `=0` to keep the BF16 hipBLASLt path.

## Required CLI flags

- `--level 0` — torch.compile not supported with this backend
- `--kv_cache_dtype bf16` — FP8 KV is TODO
- `-tp 1` — TP > 1 not exercised

CUDAGraph capture works at all default decode batch sizes.

## OpenAI-compatible server

```bash
python3 -m atom.entrypoints.openai_server \
  --model /mnt/sda1/carhuang/models/Qwen3-8B-FP8 \
  --level 0 --kv_cache_dtype bf16 \
  --max-model-len 16384 \
  --server-port 30000
```

## gsm8k via lm_eval (5-shot, generate-until)

```bash
OPENAI_API_KEY=dummy lm_eval \
  --model local-completions \
  --model_args model=/mnt/sda1/carhuang/models/Qwen3-8B-FP8,base_url=http://localhost:30000/v1/completions,tokenizer=/mnt/sda1/carhuang/models/Qwen3-8B-FP8,tokenized_requests=False,max_length=4096,num_concurrent=4 \
  --tasks gsm8k --num_fewshot 5 --batch_size 1 --limit 50
```

## Verified results on RX 9070 XT (gfx1201, 16 GB), BF16 KV

| ISL / OSL | Mode | TTFT (ms) | TPOT (ms) | Output tok/s |
|---|---|---:|---:|---:|
| 18 / 80   | cudagraph | 48  | **20.7** | 38 |
| 549 / 256 | cudagraph | 801 | **21.7** | **40.4** |
| 549 / 256 | eager     | 428 | 25.2 | 38 |

gsm8k 5-shot, n=50:

| Mode | strict | flex |
|---|---:|---:|
| eager     | 0.88 ± 0.05 | 0.88 ± 0.05 |
| cudagraph | **0.86 ± 0.05** | **0.86 ± 0.05** |
