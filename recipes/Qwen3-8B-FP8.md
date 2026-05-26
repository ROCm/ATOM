# Qwen3-8B-FP8 (block-128) on RX 9070 XT (gfx1201) via ROCm/ATOM

Verified, all-Triton, cudagraph-on path. Mirrors the Ministral-3-8B recipe.

## Model

[`Qwen/Qwen3-8B-FP8`](https://huggingface.co/Qwen/Qwen3-8B-FP8) — official Qwen
release, **FineGrainedFP8** quant with `weight_block_size=[128, 128]`,
`activation_scheme="dynamic"`. 36 layers, hidden=4096, head_dim=128,
num_q_heads=32, num_kv_heads=8 (GQA), vocab=151936.

```bash
hf download Qwen/Qwen3-8B-FP8 \
  --local-dir /mnt/sda1/carhuang/models/Qwen3-8B-FP8
```

## Required setup (run once per fresh container)

### 1. Force the triton attention backend

```bash
export ATOM_USE_UNIFIED_ATTN=1
```

Required on gfx1201 — the default `AiterBackend` calls
`torch.ops.aiter.unified_attention_with_output_base` (HIP), and the
prebuilt `.so` files in `rocm/atom-dev:latest` ship code objects only
for gfx94x/95x, so the kernel launch SIGSEGVs on gfx1201 at first
forward. `ATOM_USE_UNIFIED_ATTN=1` routes through `TritonMHABackend`
which uses aiter triton `unified_attention` (JIT-compiled per arch).

### 2. Alias gfx1250 GEMM tuning configs

aiter ships **zero** gfx1201 GEMM tuned configs. Without aliasing the
gfx1250 ones to gfx1201, the autotuner falls back to a default that is
**~50% slower** at 8B-class shapes (Mistral TPOT 22 ms with this step,
32.5 ms without — verified end-to-end on `rocm/atom-dev:latest` digest
`sha256:b704d9a8...`). Run once after starting the container:

```bash
bash scripts/gfx1201/setup_aiter_configs.sh
```

This creates 24 symlinks from `gfx1201-*.json` to `gfx1250-*.json` in
`/app/aiter-test/aiter/ops/triton/configs/gemm/`. Idempotent. The Qwen3
`gemm_a16w8_blockscale` path overrides its config in code (see
`atom/model_ops/linear.py`) so it works even without this step, but
Mistral-3 needs it for full perf.


## Optional perf env: lm_head FP8 (gfx1201)

`ATOM_LM_HEAD_FP8=1` (default on) lazily quantizes the
lm_head weight to per-row FP8 on first forward and routes it through the same
triton FP8 GEMM as qkv/o/gate_up/down. Halves the lm_head weight bandwidth
(vocab × hidden × 2 → 1 byte/elem). Combined with the per-shape
`gemm_a8w8` retune and the Triton Q/K RoPE reshape (all in commit
`gfx1201: speed up native triton decode path`), end-to-end measured
**+10-19% TPOT across BS=1..16** with **no accuracy loss**:

| Model | BS=1 | BS=8 | BS=16 | gsm8k n=200 |
|---|---:|---:|---:|---:|
| Ministral-3-8B | 22.1 → **18.4 ms** | 26.5 → **21.6 ms** | 30.8 → **27.6 ms** | 0.765 → **0.83** |
| Qwen3-8B-FP8 | 21.7 → **18.5 ms** | 24.0 → **21.6 ms** | 28.8 → **23.4 ms** | 0.925 → **0.90** |

Set `ATOM_LM_HEAD_FP8=0` to opt out (preserves the BF16 hipBLASLt
lm_head path). Skipped automatically when lm_head shares storage with
embed_tokens (tied-embeddings models).

## Required env (gfx1201)

```bash
export ATOM_USE_TRITON_GEMM=1
export AITER_LOG_LEVEL=WARNING
export AITER_ROPE_NATIVE_BACKEND=1
export ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT=0
export ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT=0
export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0
export HIP_VISIBLE_DEVICES=1   # GPU 1 by convention on this host
```

## OpenAI-compatible server

```bash
python3 -m atom.entrypoints.openai_server \
  --model /mnt/sda1/carhuang/models/Qwen3-8B-FP8 \
  --level 0 --kv_cache_dtype bf16 \
  --max-model-len 16384 \
  --server-port 30000
```

## Required CLI flags

* `--level 0` — torch.compile (`--level 3`) not supported by this backend.
* `--kv_cache_dtype bf16` — FP8 KV is a TODO.
* `-tp 1` — TP > 1 not exercised.

CUDAGraph capture works at all default decode batch sizes
`[1, 2, 4, 8, 16, 32, 48, 64, 128, 256, 512]`. Use `--enforce-eager` only for
debugging.

## gsm8k via lm_eval (5-shot, generate-until)

```bash
OPENAI_API_KEY=dummy lm_eval \
  --model local-completions \
  --model_args model=/mnt/sda1/carhuang/models/Qwen3-8B-FP8,base_url=http://localhost:30000/v1/completions,tokenizer=/mnt/sda1/carhuang/models/Qwen3-8B-FP8,tokenized_requests=False,max_length=4096,num_concurrent=4 \
  --tasks gsm8k --num_fewshot 5 --batch_size 1 --limit 50
```

## Verified results on RX 9070 XT (gfx1201, 16 GB), GPU 1, BF16 KV

### Performance (single-stream)

| ISL / OSL | Mode | TTFT (ms) | TPOT (ms) | Output tok/s |
|---|---|---:|---:|---:|
| 18 / 80   | cudagraph | 48 | **20.7** | 38 |
| 549 / 256 | cudagraph | 801 | **21.7** | **40.4** |
| 549 / 256 | eager     | 428 | 25.2 | 38 |

### Accuracy (gsm8k 5-shot, n=50)

| Mode | strict-match | flexible-extract |
|---|---:|---:|
| eager | 0.88 ± 0.05 | 0.88 ± 0.05 |
| **cudagraph** | **0.86 ± 0.05** | **0.86 ± 0.05** |

Reference: vLLM/H100 reports ~0.83 for Qwen3-8B; we are within stderr.

### Side-by-side vs Ministral-3-8B-Instruct (same GPU, same flags)

| | Ministral-3-8B (per-Tensor FP8) | **Qwen3-8B-FP8 (block-128 FP8)** |
|---|---:|---:|
| TPOT cudagraph (ms) | 22 | **20.7** |
| Output tok/s | 45 | 40 |
| gsm8k flex (n=50) | 0.815 | **0.86** |
| Chat template OK with OpenClaw / multi-system harnesses | ❌ strict alternation | **✅ lenient + native tool calling** |
| VRAM | ~13.5 GB | ~14 GB |

Qwen3 matches Mistral-3 on perf and beats it on accuracy; recommended as the
agent-stack backend going forward.

## How the gfx1201 path works (all Triton, no torch reference)

| Op | Kernel |
|---|---|
| FP8 GEMM (per-Tensor, `o_proj`, `lm_head` etc. when applicable) | aiter triton `gemm_a8w8` |
| **FP8 GEMM (block-128, all Qwen3 layers)** | **aiter triton `gemm_a16w8_blockscale` (PREQUANT=False)** |
| Dynamic per-token FP8 quant of `x` | n/a — `gemm_a16w8_blockscale` casts FP8 weight → BF16 inside the kernel and runs `tl.dot(bf16, bf16)`, so `x` stays BF16 (no activation quant needed) |
| RMSNorm (incl. Qwen3 q_norm/k_norm) | triton `RMSNorm` |
| SiLU+Mul | triton `SiluAndMul` |
| Paged attention decode + prefill | aiter triton `unified_attention` via `TritonMHABackend` |
| KV-cache write | triton kernel (handles -1 sentinels in-kernel) |
| RoPE | aiter triton `get_rope` |

### Why `gemm_a16w8_blockscale`, not `gemm_a8w8_blockscale`?

Triton on this gfx1201 build does not implement `tl.dot(fp8, fp8)` — the assertion
`only int8 supported!` fires for FP8 lhs. So the standard
`gemm_a8w8_blockscale_preshuffle` kernel (which expects FP8 inputs on both sides)
JIT-fails. The `gemm_a16w8_blockscale` kernel sidesteps this by casting the FP8
weight to BF16 at load time inside the kernel, then doing `tl.dot(bf16, bf16)`
which Triton does support. We pay one extra load-time cast but keep the FP8
weight in DRAM (no activation quant overhead on the host either).

### Custom config to fit gfx1201's 64 KiB shared mem

The shipped `gfx1201-GEMM-A16W8_BLOCKSCALE.json` picks `BLOCK_N=256` which needs
~98 KiB shared mem and JIT-fails. We override at the call site:

```python
{
    "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16, "cache_modifier": None, "NUM_KSPLIT": 1,
}
```

Shared mem usage: a (32×128×bf16×stages2) = 16 K + b (64×128×bf16×stages2) = 32 K
+ acc (32×64×fp32) = 8 K → ~57 K, fits.

### Critical gotchas (from the debug journey)

1. **`d_dtypes['fp8'] == torch.uint8`** in aiter — FP8 weights are stored as raw
   uint8 bytes with e4m3fn semantics. Always `weight.view(torch.float8_e4m3fn)`
   before passing to a kernel that does `b.to(bf16)`, otherwise the cast decodes
   bytes 0–255 as integers and you get garbage outputs.
2. **`weight_block_size: [128, 128]` parses to a `QuantType.per_128x128` enum
   that has zero consumers** in `linear.py` GEMM dispatch — the existing per_1x128
   code path handles the `(out//128, in//128)` scale grid correctly, so we
   re-route in `quant_spec.py:307`.
3. **Disable `shuffle_weights()` for `per_1x128` on gfx1201** — preshuffle is for
   the `gemm_a8w8_blockscale_preshuffle` kernel which we cannot use here. Our
   `gemm_a16w8_blockscale` wants the plain `(N, K)` layout.

## Reproduction summary

```bash
git checkout carhuang/qwen3_8b_gfx1201
hf download Qwen/Qwen3-8B-FP8 --local-dir /mnt/sda1/carhuang/models/Qwen3-8B-FP8
# (env vars + serve cmd above; cudagraph default)
# Smoke: curl /v1/chat/completions, max_tokens=80, temperature=0
# Accuracy: lm_eval gsm8k 5-shot --limit 50  → 0.86 / 0.86
# Perf: ATOM's usage block returns ttft_s and tpot_s per request
```
