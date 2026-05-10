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
| **RMSNorm** (with/without residual) | **in-tree triton kernel** (~6.6× faster than torch fallback) |
| **SiLU+Mul** (SwiGLU) | **in-tree triton kernel** (chunked, handles non-pow2 D=14336; ~3.1× faster than torch `forward_native`) |
| Unquantized BF16 linear (Reasoning checkpoints) | torch `F.linear` (gfx1201 fallback) |
| Mixed-Gumbel sampler | torch (called once per token, not on hot path) |
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

* `--level 0` — torch.compile (`--level 3`) is not supported; ATOM's
  `VllmBackend` is single-use for this backend.
* `--kv_cache_dtype bf16` — FP8 KV is a TODO; only BF16 is wired up.
* `-tp 1` — multi-GPU TP not exercised against this backend yet.

CUDAGraph capture is supported for **decode at bs ≤ 2 only**. Pass
`--cudagraph-capture-sizes "[1,2]"` to opt in. Larger captured batches
(bs ≥ 4) currently corrupt logits at replay (see Known caveats); the
engine falls back to eager for any decode batch outside the captured
set, so concurrency above 2 still works — it just doesn't get the
graph speedup. Use `--enforce-eager` to disable cudagraph entirely.

## Smoke test

```bash
python3 -m atom.examples.simple_inference \
  --model /path/to/Ministral-3-8B-Instruct-2512 \
  --level 0 -tp 1 --kv_cache_dtype bf16 \
  --max-model-len 4096 --max-tokens 32 \
  --gpu-memory-utilization 0.85 \
  --cudagraph-capture-sizes "[1,2]"
```

## OpenAI-compatible server

```bash
python3 -m atom.entrypoints.openai_server \
  --model /path/to/Ministral-3-8B-Instruct-2512 \
  --level 0 --kv_cache_dtype bf16 \
  --max-model-len 4096 \
  --server-port 30000 \
  --cudagraph-capture-sizes "[1,2]"
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

**CUDAGraph at bs ≤ 2 + fused FP8 quant** (single-prompt latency,
single-token bench, "The capital of France is", max_tokens=64):

| Stack | TPOT | TTFT |
|---|---:|---:|
| Eager (pre-cudagraph) | 0.034 s/tok | 0.21 s |
| Eager (after FP8 fused-quant + cached w_scale) | 0.032 s/tok | 0.24 s |
| CUDAGraph `[1,2]` (pre-fused-quant) | 0.025 s/tok | 0.06 s |
| **CUDAGraph `[1,2]` + fused-quant + cached w_scale** | **0.022 s/tok** | **0.07 s** |

Cumulative vs the original eager baseline: **35% TPOT reduction** and
**3× TTFT reduction**. gsm8k accuracy preserved across both wins:

| Stack | strict | flex |
|---|---:|---:|
| Eager baseline | 0.785 | 0.785 |
| CUDAGraph `[1,2]` | 0.765 | 0.765 |
| CUDAGraph `[1,2]` + fused-quant | 0.78 | 0.78 |

(All within ±0.030 stderr at n=200, num_concurrent=2.)

Remaining perf headroom worth pursuing:

- **CUDAGraph at bs ≥ 3**: captured graphs at decode bs ≥ 3 corrupt
  the first decode-step logits (see Known caveats). Root cause is
  unidentified; investigation ruled out v1/v2 dispatch, prewarm,
  capture-stream alignment, JIT-during-capture, FP8 GEMM split-K
  configs, and lm_head capture. Eager-mode multi-seq decode is fine
  (gsm8k 0.785 at concurrent=4) — only the captured-graph replay at
  bs ≥ 3 corrupts. Symptom is consistent with sglang#1558 / sglang#19799
  (triton + cudagraph + ROCm). Concurrency above 2 still works via the
  engine's eager fallback path; just no graph speedup.
- **TP=2**: blocked at host kernel level — RCCL needs `iommu=pt` (and
  `amd_iommu=on`) on the GRUB cmdline for cross-GPU P2P. Without that
  every multi-rank `nccl_init` fails with `HIP failure: invalid device
  ordinal`. Fix is host-side: edit `/etc/default/grub`, regen, reboot.
  Once unblocked, TP=2 lets the BF16 8B Reasoning variant fit (16.6 GB
  weights → 8.3 GB / GPU); see "TP=2 (Reasoning-8B)" caveat.
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
* **CUDAGraph at decode bs ≥ 3 is broken**: captured graphs at bs=3,4,8
  all emit a wrong logit at the first decode step after prefill, almost
  always sampling EOS or a stop token. bs=1 and bs=2 captured graphs
  are correct. Eager mode at the same bs is correct (gsm8k 5-shot 0.785
  at concurrent=4). Investigated and ruled out as causes: v1 vs v2
  pa_decode dispatch (v1-only forced is also broken at bs ≥ 3); the
  prewarm helper (engine reaches capture without it; bs ≥ 3 still
  breaks); JIT during capture (capture itself succeeds, eager works);
  capture-stream alignment (warmup now on `gc.stream`, twice, per the
  SGLang/PyTorch idiom); FP8 GEMM split-K configs (`_get_config`
  returns NUM_KSPLIT=1 across all our (M, N, K) so no per-bs binary
  divergence); lm_head being captured (`logits_in_graph=False` also
  broken). Symptom is consistent with sglang#1558 / sglang#19799 and
  pytorch#155684 (HIP graph capture is silent on illegal-during-capture
  ops). Workaround: `--cudagraph-capture-sizes "[1,2]"`. Concurrency
  > 2 still works via eager fallback.
* **TP=2 not yet usable on this host**: `nccl_init` for world_size > 1
  fails with `HIP failure: invalid device ordinal` and a warning that
  `iommu=pt` is missing from the kernel command line. RCCL needs
  `iommu=pt amd_iommu=on` on the host GRUB cmdline to set up cross-GPU
  P2P. `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` does not help — RCCL
  fails before it gets to the transport choice. Fix is host-side:
  ```
  # /etc/default/grub
  GRUB_CMDLINE_LINUX_DEFAULT="... iommu=pt amd_iommu=on"
  # then update-grub && reboot
  ```
  Once that's in, TP=2 should work and lets the BF16 Ministral-3-8B-
  Reasoning model (16.6 GB) split across 2 × 16 GB gfx1201s. Without
  it, only single-GPU FP8 / 3B-BF16 models fit.
