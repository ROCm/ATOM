# Ministral-3-8B-Instruct-2512 on gfx1201 (RX 9070 XT)

This recipe describes running `mistralai/Ministral-3-8B-Instruct-2512`
(natively FP8 trained) on a single RDNA4 GPU using ATOM's
`TORCH_NATIVE_ATTENTION` backend. The backend is selected automatically
when ATOM detects gfx1201; on other archs it does nothing.

## Why not the default AITER path?

The AITER package shipped in `rocm/atom-dev:latest` ships prebuilt HIP
`.so` files only for gfx94x/95x. Loading any of those modules on
gfx1201 segfaults with `No compatible code objects found for: gfx1201`.
The torch-native backend bypasses the prebuilt path:

| Op | Backend on gfx1201 |
|---|---|
| Per-tensor FP8 GEMM (qkv/o/gate_up/down proj) | **aiter triton `gemm_a8w8`** (JIT-compiled, ~360× faster than torch dequant) |
| Paged attention **prefill** | **aiter triton `context_attention_fwd`** (JIT-compiled; 2.2× faster per-call than torch SDPA; handles GQA internally) |
| Paged attention **decode** | **aiter triton `paged_attention_decode`** (JIT-compiled; ~20% e2e speedup) |
| **KV cache write** | **in-tree triton kernel** (handles -1 sentinels in-kernel; ~12× faster than torch advanced indexing; no GPU→CPU sync — CUDAGraph-capturable) |
| Unquantized BF16 linear (Reasoning checkpoints) | torch `F.linear` (gfx1201 fallback) |
| RMSNorm (with/without residual) | torch RMSNorm fallback |
| SiLU + Mul (SwiGLU) | `forward_native` (existing torch path) |
| Mixed Gumbel sampler | torch Gumbel-max + argmax |
| YaRN-scaled RoPE | `forward_native` via `AITER_ROPE_NATIVE_BACKEND=1` |

## One-shot image setup (per fresh container)

Aiter ships per-arch tuned GEMM configs but only for gfx94x/95x/1250.
Symlink the gfx1250 (sibling RDNA4) configs as gfx1201 placeholders:

```bash
cd /app/aiter-test/aiter/ops/triton/configs/gemm
for f in gfx1250-*.json; do
  ln -s "$f" "gfx1201-${f#gfx1250-}"
done
```

This is the only image-side setup. Everything else is in the repo.

## Required env vars

```bash
export ATOM_USE_TRITON_GEMM=1
export AITER_LOG_LEVEL=WARNING
export AITER_ROPE_NATIVE_BACKEND=1
export ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT=0
export ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT=0
export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0
```

## Required CLI flags

* `--enforce-eager --level 0` — CUDAGraph capture is not yet supported
  by the torch-native backend.
* `--kv_cache_dtype bf16` — FP8 KV is a TODO; only BF16 is wired up.
* `-tp 1` — multi-GPU TP not exercised against this backend yet.

## Smoke test

```bash
python3 -m atom.examples.simple_inference \
  --model /path/to/Ministral-3-8B-Instruct-2512 \
  --enforce-eager --level 0 -tp 1 --kv_cache_dtype bf16 \
  --max-model-len 4096 --max-tokens 32 \
  --gpu-memory-utilization 0.85
```

## OpenAI-compatible server

```bash
python3 -m atom.entrypoints.openai_server \
  --model /path/to/Ministral-3-8B-Instruct-2512 \
  --enforce-eager --level 0 --kv_cache_dtype bf16 \
  --max-model-len 4096 \
  --server-port 30000
```

## gsm8k via lm_eval (5-shot, generate-until)

```bash
OPENAI_API_KEY=dummy lm_eval \
  --model local-completions \
  --model_args model=/path/to/Ministral-3-8B-Instruct-2512,base_url=http://localhost:30000/v1/completions,tokenizer=/path/to/Ministral-3-8B-Instruct-2512,tokenized_requests=False,max_length=4096,num_concurrent=2 \
  --tasks gsm8k --num_fewshot 5 --batch_size 1
```

### Verified results on RX 9070 XT (gfx1201, 16 GB)

Best end-to-end with the **full triton stack** (FP8 GEMM + paged
attention decode + flash-attention prefill):

| Setup | n | strict-match | flexible-extract |
|---|---:|---:|---:|
| gsm8k 5-shot, n=200 | 200 | **0.785** | **0.785** |

Sits at the top of Mistral's published Ministral-3-8B gsm8k range
(~75–80% 5-shot).

**Accuracy evolution** (gsm8k 5-shot, n=200):

| Stack | strict | flex |
|---|---:|---:|
| Torch fallback | 0.765 | 0.770 |
| + triton FP8 GEMM | 0.765 | 0.770 |
| + triton paged_attention_decode | 0.765 | 0.770 |
| + triton context_attention_fwd (prefill) | **0.785** | **0.785** |

**Throughput evolution** (gsm8k 5-shot, num_concurrent=4):

| Backend | TPOT (5-tok prompt) | TTFT (5-tok prompt) | sec/problem |
|---|---:|---:|---:|
| Torch fallback (pre-triton) | 0.28 s/tok | 0.7 s | ~21 |
| + triton FP8 GEMM | 0.038 s/tok | 0.16 s | ~2.1 |
| + triton paged_attention_decode | 0.042 s/tok* | 0.54 s | ~1.7 |
| + triton context_attention_fwd | 0.044 s/tok* | **0.23 s** | ~1.4 |

\* TPOT for very short prompts is dominated by Python overhead; per-call
benchmarks show triton paged_attention_decode is 1.8× faster than torch
SDPA at gsm8k context lengths (500–1500 tokens).

Full gsm8k (1319 problems) extrapolates to ~30 min wall time at
`num_concurrent=4`.

Remaining perf headroom worth pursuing:

- **CUDAGraph capture** is still disabled (`--enforce-eager --level 0`).
  The torch-native backend doesn't yet implement
  `build_for_cudagraph_capture`; wiring it would shave Python launch
  overhead from each step.
- **FP8 KV cache**: BF16 KV today; would halve KV memory and shave
  some bandwidth on long-context decode.

## Known caveats

* 238 `activation_scale` checkpoint tensors are silently dropped during
  load. Harmless because the FP8 GEMM fallback dequantizes weights to
  BF16 and ignores per-channel input scale, but worth fixing if FP8
  native compute ever lands.
* `compute_block_bytes` reports a placeholder pool size. The KV pool is
  allocated correctly but the engine logs a 100% mismatch warning at
  boot. Cosmetic — KV writes/reads work end-to-end.
* `--max-model-len` must accommodate the chat-templated prompt (the
  Mistral system prompt is ~540 tokens).
