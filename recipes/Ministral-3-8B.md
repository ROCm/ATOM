# Ministral-3-8B-Instruct-2512 on gfx1201 (RX 9070 XT)

This recipe describes running `mistralai/Ministral-3-8B-Instruct-2512`
(natively FP8 trained) on a single RDNA4 GPU using ATOM's
`NATIVE_TRITON_ATTENTION` backend. The backend is selected automatically
when ATOM detects gfx1201; on other archs it does nothing.

## Why not the default AITER path?

The AITER package shipped in `rocm/atom-dev:latest` ships prebuilt HIP
`.so` files only for gfx94x/95x. Loading any of those modules on
gfx1201 segfaults with `No compatible code objects found for: gfx1201`.
The gfx1201 triton backend bypasses the prebuilt path:

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
  the first decode-step logits (see Known caveats). Bisected as far as
  possible from outside the engine: every individual triton kernel
  (RMSNorm, SiluMul, kv-write, gemm_a8w8, pa_decode_v1, pa_decode_v2)
  passes a standalone capture-then-replay test at bs=4 with bitwise-
  identical output, AND a 36-layer chained version of those kernels in
  a single graph also passes at bs=4 — so the bug is not in any kernel
  or in their composition under cudagraph. Bypassing RoPE entirely
  does not fix it either (eager-no-rope → 0.633, cg-no-rope → 0.067 on
  gsm8k n=30 nc=4; cudagraph still degrades the model far below the
  RoPE-less baseline). The bug must therefore be in some interaction
  between ATOM's engine flow and the captured replay that doesn't
  reproduce in standalone — finding it would need engine-level
  intermediate-state diffing per layer, which is out of scope here.
- **TP=2**: blocked at host kernel level — both RCCL and aiter's
  CustomAllreduce fall over on the same root cause: HIP IPC requires
  `iommu=pt` (and `amd_iommu=on`) on the GRUB cmdline. PyNcclCommunicator
  init fails with `HIP error: invalid kernel file`; CustomAllreduce
  init then fails one step later with
  `hipIpcOpenMemHandle ... HIP error (invalid device pointer)`.
  `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` does NOT help (failure is
  before transport choice). Fix is host-side: edit `/etc/default/grub`,
  regen, reboot. Once unblocked, TP=2 lets the BF16 8B Reasoning
  variant fit (16.6 GB weights → 8.3 GB / GPU).
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
  at concurrent=4).

  Investigated and ruled out as causes:
  - **v1 vs v2 pa_decode dispatch**: v1-only forced is also broken at
    bs ≥ 3; bs ≥ 3 captured graphs replay incorrectly under both.
  - **The prewarm helper**: engine reaches capture without it; bs ≥ 3
    still breaks. (We keep the prewarm anyway since it follows the
    SGLang / PyTorch idiom and is a robustness belt-and-suspenders.)
  - **JIT-during-capture**: capture itself succeeds and the eager path
    works fine at the same bs and same shapes.
  - **Capture-stream alignment**: warmup now runs on `gc.stream`, twice,
    per the SGLang / PyTorch CUDA-graphs idiom; bug persists.
  - **FP8 GEMM split-K configs**: `_get_config` returns `NUM_KSPLIT=1`
    across all our (M, N, K) so there's no per-bs binary divergence in
    `gemm_a8w8`.
  - **lm_head being captured**: `logits_in_graph=False` also broken at
    bs ≥ 3.
  - **Standalone capture-replay of every triton kernel** (RMSNorm,
    SiluMul, kv-write, gemm_a8w8, pa_decode_v1, pa_decode_v2) at
    bs=1..8: every kernel passes bitwise-identically.
  - **Standalone 36-layer chained kernels** (full Mistral decoder
    depth) at bs=4 captured + replayed: passes bitwise-identically.
  - **RoPE bypass**: turning off RoPE entirely in ATOM still leaves
    the cudagraph bs ≥ 3 path broken (eager-no-rope = 0.633 vs
    cg-no-rope = 0.067 on gsm8k n=30 nc=4) — RoPE isn't the cause.

  Conclusion: the bug is in some interaction between ATOM's full
  engine flow at runtime and the captured-graph replay that doesn't
  reproduce in standalone tests. Running it down would need
  intermediate-state diffing per layer in the live engine — out of
  scope here. Symptom is consistent with sglang#1558 / sglang#19799 and
  pytorch#155684 (HIP graph capture is silent on illegal-during-capture
  ops). Workaround: `--cudagraph-capture-sizes "[1,2]"`. Concurrency
  > 2 still works via eager fallback.
* **TP=2 not yet usable on this host**: tried both transport paths;
  both fail on the same root cause — HIP IPC needs `iommu=pt` on the
  host kernel cmdline.

  - **RCCL / PyNcclCommunicator**: fails with `HIP failure: invalid
    device ordinal` and a `Missing "iommu=pt" from kernel command line`
    warning. `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` does NOT help
    (the failure is before RCCL chooses a transport).
  - **aiter CustomAllreduce** (the IPC-handle-based fast path that
    bypasses RCCL): also fails, one step later, with
    `hipIpcOpenMemHandle ... HIP error (invalid device pointer)`. It
    needs the same iommu=pt that RCCL does.

  Fix is host-side (requires reboot):
  ```
  # /etc/default/grub
  GRUB_CMDLINE_LINUX_DEFAULT="... iommu=pt amd_iommu=on"
  # then update-grub && reboot
  ```
  Once that's in, TP=2 should work and lets the BF16 Ministral-3-8B-
  Reasoning model (16.6 GB) split across 2 × 16 GB gfx1201s. Without
  it, only single-GPU FP8 / 3B-BF16 models fit.
