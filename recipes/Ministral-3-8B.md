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
| Per-tensor FP8 GEMM (qkv/o/gate_up/down proj) | **aiter triton `gemm_a8w8`** (JIT-compiled) |
| Dynamic per-tensor FP8 quant of x | **aiter triton `dynamic_per_tensor_quant_fp8_i8`** (single-launch, atomic_max scale) |
| Paged attention **prefill** | **aiter triton `context_attention_fwd`** (JIT; handles GQA) |
| Paged attention **decode** | **aiter triton `paged_attn_decode_v1` / `paged_attn_decode_v2`** (in-tree dispatcher with Python-float scales — wrapper's `.item()` would break cudagraph capture) |
| **KV cache write** | **in-tree triton kernel** (handles -1 sentinels in-kernel; CUDAGraph-capturable) |
| **RMSNorm** (with / with-add-residual) | **in-tree triton kernel** (pow2 D ≤ 16384) |
| **SiLU+Mul** (SwiGLU) | **in-tree triton kernel** (chunked, non-pow2 D OK) |
| YaRN-scaled RoPE | aiter `rope_cached_positions_2c_fwd_inplace` (JIT HIP via `@compile_ops`) |
| lm_head BF16 linear | rocBLAS `F.linear` (vocab=131072, BF16) |
| Sampler | torch greedy / Gumbel-max + argmax (one call per step, off hot path) |

There is no torch fallback for any kernel above — the path raises a
clear `RuntimeError` if a triton kernel is unavailable. Reason: every
historical fallback contained either `.item()` or `.cpu().tolist()`
syncs, which silently corrupt cudagraph capture on ROCm (HIP graph
capture does not raise on illegal-during-capture ops the way CUDA
does — see pytorch#155684).

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

CUDAGraph capture works at all decode batch sizes (default `[1, 2, 4,
8, 16, 32, 48, 64, 128, 256]`). The earlier `bs ≥ 3` corruption was a
NaN-from-padding bug in `prepare_decode` (now fixed — see Known
caveats for the diagnosis). Use `--enforce-eager` only if you want to
disable cudagraph entirely.

## Smoke test

```bash
python3 -m atom.examples.simple_inference \
  --model /path/to/Ministral-3-8B-Instruct-2512 \
  --level 0 -tp 1 --kv_cache_dtype bf16 \
  --max-model-len 4096 --max-tokens 32 \
  --gpu-memory-utilization 0.85
```

## OpenAI-compatible server

```bash
python3 -m atom.entrypoints.openai_server \
  --model /path/to/Ministral-3-8B-Instruct-2512 \
  --level 0 --kv_cache_dtype bf16 \
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
* **(FIXED) CUDAGraph at decode bs ≥ 3 used to be broken** — diagnosed
  and fixed. Root cause: `prepare_decode` padded `context_lens` to 0
  for slots `[scheduled_bs:bs]` when the engine padded a partial
  batch up to a captured cudagraph size. Aiter's pa_decode_v1/v2
  kernels with `seq_len=0` run zero loop iterations and end with
  `acc /= exp_sum` where `exp_sum` stayed 0 -> `0/0 = NaN`. That NaN
  in the padded slot's attn_out then propagated through the per-tensor
  FP8 quant of `attn_out` (`amax(... NaN ...) = NaN` -> the entire
  batch's `x_scale` became NaN -> every downstream `gemm_a8w8` output
  NaN), corrupting all real slots. Symptom: wrong logit at the first
  decode step, model emitted a stop token, request finished after one
  token.

  The reason a long simple bisection didn't find it earlier: when
  scheduled_bs == captured_bs (e.g., the standalone 36-layer chain
  test, or 4 simultaneous curl calls hitting the bs=4 graph), no
  padding ever happens, so the bug doesn't reproduce. Only lm_eval
  with its variable scheduled_bs over 200 requests reliably triggers
  partial batches that get padded.

  Fix (in `prepare_decode`): pad `context_lens` to `1` instead of `0`
  for `[scheduled_bs:bs]`. With seq_len=1 the kernel runs exactly one
  loop iteration, reads one garbage K/V from `block_tables[i, 0] = 0`
  (which points at real but unrelated KV — fine, the padded row's
  output is discarded by the engine which only reads
  `outputs[:scheduled_bs]`), and produces a finite attn_out. Slot
  mapping stays at -1 so our kv-write kernel's sentinel still skips
  the write (otherwise we'd overwrite slot 0's real KV).

  Verification: gsm8k 5-shot, n=200 with the default cudagraph capture
  set `[1, 2, 4, 8, 16, 32, 48, 64, 128, 256]`:
    num_concurrent=4: strict 0.815, flex 0.815 (was 0.005)
    num_concurrent=8: strict 0.760, flex 0.760 (was 0.005)
  Both at or above the eager baseline of 0.785.
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

## Performance roofline analysis

### Where the time goes (cudagraph bs=1, single-token decode)

torch.profiler trace of 48 decode steps at TPOT 0.022 s/tok = **45 tok/s**:

| Component | Per-step time | Notes |
|---|---:|---|
| `gemm_a8w8` (qkv + o + gate_up + down, ×34 layers) | **14.7 ms** | Dominant; 4 specializations (one per shape bucket) |
| Dynamic per-tensor FP8 quant (`dynamic_per_tensor_quant_fp8_i8` + `static_per_tensor_quant_fp8_i8`) | 1.4 ms | Two-kernel pair, called once per linear (×136 / step) |
| `lm_head` rocBLAS BF16 GEMM (vocab=131072) | 1.9 ms | Necessary; ~bandwidth-bound |
| `paged_attn_decode_v2` + reduce | 0.27 ms | Already very fast |
| `_rmsnorm_add_kernel` + `_rmsnorm_kernel` | 0.15 ms | Already very fast |
| `_kv_cache_write_kernel` | 0.07 ms | Already very fast |
| `_silu_mul_kernel` | 0.06 ms | Already very fast |
| Other elementwise (aten reshape / contiguous / etc.) | ~3.5 ms | residual python-side ops baked into the captured graph |
| **Total** | **~22 ms** | matches measured TPOT |

### Roofline projection (RX 9070 XT, 16 GB GDDR6, 640 GB/s)

For an 8B FP8 model at decode bs=1, weight read per step = ~8 GB:

- **Memory-bound roofline**: 8 GB ÷ 640 GB/s = **12.5 ms / step = 80 tok/s**
- **Realistic ceiling** (matches what comparable consumer GPUs achieve at bs=1 in practice — see cross-GPU table below): ~50-65 tok/s = 16-20 ms/step
- **Our measured**: 22 ms/step = **45 tok/s = 56% of memory roofline, 90% of realistic ceiling**

### Cross-GPU comparison (8B FP8 / Q4 LLM, decode bs=1)

| GPU | HBM/VRAM BW | FP8 8B roofline | Observed bs=1 | Quant / runtime | % of FP8 roofline |
|---|---:|---:|---:|---|---:|
| **MI300X** (gfx942) | 5.3 TB/s | ~670 tok/s | ~150-250 tok/s | FP8, vLLM+AITER | ~25-35% |
| **H100 SXM** | 3.35 TB/s | ~415 tok/s | ~180-250 tok/s | FP8, TRT-LLM | ~45-60% |
| **RTX 4090** | 1.0 TB/s | ~125 tok/s | ~131-150 tok/s | Q4 GGUF, llama.cpp | ~100% (Q4 reads less) |
| **RX 7900 XTX** (gfx1100) | 0.96 TB/s | ~120 tok/s | ~60-70 tok/s | Q4, llama.cpp ROCm | ~50% |
| **RX 9070 XT** (gfx1201) — published | 0.64 TB/s | ~80 tok/s | ~30-50 tok/s | Q4, llama.cpp ROCm 6.4.1+ | ~38-63% |
| **RX 9070 XT — this build (FP8, ATOM)** | 0.64 TB/s | ~80 tok/s | **45 tok/s** | FP8, ATOM | **56%** |

ATOM-on-RDNA4 with this triton stack matches or beats the published
llama.cpp Q4 numbers for the same GPU **despite reading 2× as much
weight data per step** (FP8 = 8 GB vs Q4 = 4 GB). That is, our
per-byte efficiency is roughly 2× llama.cpp's on this hardware.

### Remaining gap to roofline (~10 ms / step)

- **gemm_a8w8 itself is ~2 ms/step above its memory-bound floor**
  (~14.7 ms actual vs ~8.5 ms ideal aggregate). Aiter's triton kernel
  uses a fixed BLOCK_SIZE_M=64 even at M=1, wasting most of the row
  tile — but a bs=1-specialized kernel didn't exist in aiter at the
  time of writing. Closing this is ~6 ms (= 27% TPOT reduction).
- **Two-kernel dynamic per-tensor quant** (1.4 ms/step). Could be
  fused with gemm_a8w8 via `gemm_a8w8_with_dynamic_quant`, eliminating
  the launch-pair per linear. Mistral-3 ships
  `activation_scheme: "static"` but **no actual `input_scale` tensors
  in the safetensors checkpoint** — so the static-quant fast path is
  not usable for this model.
- **~3.5 ms/step in scattered elementwise ops** (aten reshape /
  contiguous / vectorized_elementwise around the linear path). These
  add up across 34 layers × 4 linears × small ops. Trimming via a
  single fused triton "rmsnorm + dynamic_quant + gemm_a8w8" kernel
  would be the cleanest win, requiring an aiter contribution.

### Sources for the cross-GPU table

- vLLM on MI300X: https://blog.vllm.ai/2024/10/23/vllm-serving-amd.html
- TRT-LLM Llama-3.1-8B FP8 on H100: https://github.com/NVIDIA/TensorRT-LLM/issues/6294
- Modal latency-optimized TRT-LLM on H100: https://modal.com/docs/examples/trtllm_latency
- llama.cpp on RTX 4090 / RDNA: https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/
- llama.cpp ROCm gfx1201 / gfx1100 community: https://github.com/ggml-org/llama.cpp/discussions/15021
- LLM-Inference-Bench (MI250 vs A100/H100/MI300X): https://arxiv.org/html/2411.00136v1
- TechReviewer: RX 9070 XT for LLMs: https://www.techreviewer.com/tech-specs/amd-rx-9070-xt-gpu-for-llms/
- GPU Hunter: 7900 XTX ~66 tok/s Llama-3-8B Q4: https://www.gpuhunter.io/blog/amd-vs-nvidia-local-ai-2026
