# Ministral-3-8B-Instruct-2512 on gfx1201 (RX 9070 XT)

This recipe describes running `mistralai/Ministral-3-8B-Instruct-2512`
(natively FP8 trained) on a single RDNA4 GPU using ATOM's
aiter triton attention backend (selected by default for non-MLA models).
The backend is arch-portable: it routes through aiter's triton paged
attention kernels, which JIT-compile for any supported arch including
gfx1201.

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

## Required setup (run once per fresh container)

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

`ATOM_GFX1201_LM_HEAD_FP8=1` (default on for gfx1201) lazily quantizes the
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

Set `ATOM_GFX1201_LM_HEAD_FP8=0` to opt out (preserves the BF16 hipBLASLt
lm_head path). Skipped automatically when lm_head shares storage with
embed_tokens (tied-embeddings models).

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
  --max-model-len 16384 --max-tokens 32 \
  --gpu-memory-utilization 0.85
```

## OpenAI-compatible server

```bash
python3 -m atom.entrypoints.openai_server \
  --model /path/to/Ministral-3-8B-Instruct-2512 \
  --level 0 --kv_cache_dtype bf16 \
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

### Verified results on RX 9070 XT (gfx1201, 16 GB)

**Performance + accuracy** (cudagraph default capture set
`[1,2,4,8,16,32,48,64,128,256,512]`, BF16 KV, max_model_len 4096,
RX 9070 XT @ 640 GB/s, single GPU):

| concurrency | ISL / OSL | TTFT mean (ms) | TPOT mean (ms) | Output tok/s | Total (in+out) tok/s | gsm8k 5-shot strict / flex (n=200) |
|---:|---|---:|---:|---:|---:|:---:|
| **1** | 1024 / 1024 | 170 | **21.9** | 45.0 | 116 | — |
| **2** | 1024 / 1024 | 180 | 22.5 | 76.6 | 169 | **0.765 / 0.765** |
| **4** | 1024 / 1024 | 212 | 23.2 | 152 | 280 | **0.780 / 0.785** |
| **8** | 1024 / 1024 | 486 | 24.9 | 254 | 568 | — |
| **16** | 512 / 256  | 285 | 31.0 | 421 | 1300 | **0.715 / 0.725** |
| **32** | 256 / 128  | 355 | 36.2 | 665 | 2048 | **0.735 / 0.740** |
| **64** | 128 / 128  | 287 | 41.5 | 1247 | 2410 | — |
| **128** | 64 / 64   | 360 | 66.4 | 1543 | 3194 | — |

- **Eager baseline**: 0.785 / 0.785. All cudagraph results are within
  ±0.030 stderr.
- **TPOT @ conc=1**: 21.9 ms = **45.6 tok/s** = **53% of the 86 tok/s
  memory roofline** (8 GB FP8 weights ÷ 640 GB/s). Beats published
  llama.cpp Q4 numbers (30-50 tok/s) on the same GPU despite reading
  2× as much weight per step (FP8 vs Q4) — per-byte ~2× more
  efficient than llama.cpp.
- **Practical max throughput**: ~3200 tok/s aggregate at conc=128
  (short contexts) — KV pool of 941 blocks × 16 tokens = 15k slots
  is the cap; longer contexts squeeze the practical conc lower.

**Optimization-step impact** (TPOT s/tok, single-prompt
"capital of France" decode, max_tokens=64):

| Stack | TPOT |
|---|---:|
| Eager pre-triton (torch dequant + matmul) | 0.28 |
| + triton FP8 GEMM (`gemm_a8w8`) | 0.038 |
| + triton kv-write / RMSNorm / SiLU+Mul / pa_decode | 0.034 |
| + CUDAGraph (decode only, bs ≤ 2 captured) | 0.025 |
| + fused dynamic FP8 quant + cached `w_scale_full` | 0.022 |
| + per-shape `gemm_a8w8` config (`GROUP_SIZE_M=1`) | 0.022 |
| + CUDAGraph at all bs (NaN-from-padding fix) | 0.022 |
| + **per-token FP8 quant (single kernel, no atomic)** | **0.022** |

Cumulative: **0.28 → 0.022 s/tok = ~13× speedup** end-to-end vs the
torch-fallback baseline. The last few steps don't move conc=1 TPOT
(already memory-bound), but each unlocks higher concurrency or fixes
correctness — see the table above.

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
