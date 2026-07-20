# Ministral-3-8B-Instruct-2512 on gfx1201 (RX 9070 XT)

Run `mistralai/Ministral-3-8B-Instruct-2512` (natively FP8) on a single
RDNA4 GPU. ATOM runs attention and GEMM through Triton
(`ATOM_USE_UNIFIED_ATTN=1`, `ATOM_USE_TRITON_GEMM=1`); the KV-cache write,
RoPE and norms use native aiter HIP kernels.

> **Navi (gfx1201) prerequisite:** aiter must be built for the arch — see
> [ROCm/aiter#3846](https://github.com/ROCm/aiter/issues/3846). Short-term
> fix: build aiter from source with `GPU_ARCHS=gfx1201` (a native build on
> the card does this automatically).

## Model

[`mistralai/Ministral-3-8B-Instruct-2512`](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) — gated, requires accepting the license on the model page and setting `HF_TOKEN`.

```bash
hf download mistralai/Ministral-3-8B-Instruct-2512 \
  --local-dir /mnt/sda1/carhuang/models/Ministral-3-8B-Instruct-2512
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

## Required CLI flags

- `--level 0` — torch.compile not supported with this backend
- `--block-size 64` — required with `ATOM_USE_UNIFIED_ATTN=1` + bf16 KV (the engine asserts `block_ratio == 1`; default 16 fails)
- `--kv_cache_dtype bf16` — FP8 KV is TODO
- `-tp 1` — multi-GPU not exercised (blocked on host `iommu=pt`)

CUDAGraph capture works at all default decode batch sizes.

## Smoke test

```bash
python3 -m atom.examples.simple_inference \
  --model /path/to/Ministral-3-8B-Instruct-2512 \
  --level 0 -tp 1 --kv_cache_dtype bf16 --block-size 64 \
  --max-model-len 16384 --max-tokens 32 \
  --gpu-memory-utilization 0.85
```

## OpenAI-compatible server

```bash
python3 -m atom.entrypoints.openai_server \
  --model /path/to/Ministral-3-8B-Instruct-2512 \
  --level 0 --kv_cache_dtype bf16 --block-size 64 \
  --max-model-len 16384 \
  --server-port 30000
```

## gsm8k via lm_eval (5-shot, generate-until)

```bash
OPENAI_API_KEY=dummy lm_eval \
  --model local-completions \
  --model_args model=/path/to/Ministral-3-8B-Instruct-2512,base_url=http://localhost:30000/v1/completions,tokenizer=/path/to/Ministral-3-8B-Instruct-2512,tokenized_requests=False,max_length=4096,num_concurrent=2 \
  --tasks gsm8k --num_fewshot 5 --batch_size 1
```

## Verified results on RX 9070 XT (gfx1201, 16 GB)

cudagraph default capture set, BF16 KV, single GPU:

| concurrency | ISL / OSL | TTFT (ms) | TPOT (ms) | Output tok/s | gsm8k 5-shot strict / flex (n=200) |
|---:|---|---:|---:|---:|:---:|
| 1   | 1024 / 1024 | 170 | **21.9** | 45.0 | — |
| 4   | 1024 / 1024 | 212 | 23.2 | 152 | 0.780 / 0.785 |
| 16  | 512 / 256   | 285 | 31.0 | 421 | 0.715 / 0.725 |
| 32  | 256 / 128   | 355 | 36.2 | 665 | 0.735 / 0.740 |
| 128 | 64 / 64     | 360 | 66.4 | 1543 | — |

Eager baseline: 0.785 / 0.785. All cudagraph results within ±0.030 stderr.

## Known caveats

- 238 `activation_scale` checkpoint tensors are silently dropped during
  load (harmless — the FP8 GEMM dequantizes weights and ignores
  per-channel input scale).
- `compute_block_bytes` logs a cosmetic 100% pool-size mismatch at boot.
- `--max-model-len` must accommodate the Mistral chat template
  (~540 tokens).
- TP > 1 needs `iommu=pt amd_iommu=on` on the host kernel cmdline.
