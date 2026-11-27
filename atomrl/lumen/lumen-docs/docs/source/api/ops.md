# atomrl.lumen.ops

Stateless functional API backed by [AITER](https://github.com/ROCm/aiter) kernels. Each operator family uses an **ASM → CK → Triton** fallback chain, selected at runtime via `@lru_cache` dispatch probes in `lumen/ops/dispatch.py`.

```{note}
Most users do not need to call these ops directly — the module layer and `quant.enable` handle dispatch automatically. Use the ops API for custom operator pipelines or benchmarking.
```

## Dispatch Architecture

All Lumen ops follow the same pattern:

1. **Probe** — `dispatch.py` lazily checks AITER kernel availability via ~30 `@lru_cache` probes
2. **Dispatch** — Each op has a preferred backend order (ASM → CK → Triton → fallback)
3. **Fallback** — If a fused kernel is unavailable, Lumen decomposes into simpler AITER primitives

## Attention

```python
from atomrl.lumen.ops.attention import attention, attention_fp8_quant
```

Two entry-point functions:

| Function | Purpose |
|----------|---------|
| `attention()` | BF16 attention (no FP8 quantization on QKV) |
| `attention_fp8_quant()` | FP8 quantized attention (QKV quantized to FP8) |

### BF16 Dispatch (`attention()`)

| Backend | Forward | Backward |
|---------|---------|----------|
| `aiter_csrc` (default) | CK FlashAttention | CK autograd backward |
| `aiter_triton` (fallback) | Triton FlashAttention | Triton backward |

On `RuntimeError`, CK path automatically falls back to Triton.

### FP8 Dispatch (`attention_fp8_quant()`)

| `quant_type` | Forward | Backward |
|--------------|---------|----------|
| `blockwise` / `dynamic` / `delayed` / `per_token` | Triton FP8 blockwise quantization + attention | Triton backward |
| `mxfp8` | Triton MXFP8 quantization + attention (requires gfx950+) | Triton MXFP8 backward |
| `blockwise2d` | Triton blockwise2d FP8 quantization + attention | Triton backward |
| `none` | Delegates to `attention()` (BF16) | BF16 backward |

### Context Parallelism

Both functions support Context Parallelism via all-to-all (A2A) when `cp_param_bundle` is provided. A2A redistributes KV across CP ranks before/after attention computation.

**Environment variable:** `LUMEN_ATTN_KERNEL_BACKEND` (`auto` / `csrc` / `triton`) controls backend selection.

## Quantized Linear & GEMM

```python
from atomrl.lumen.ops.quantize import fp8_linear, fp8_quantize, fp8_dequantize
from atomrl.lumen.ops.gemm import grouped_gemm, grouped_gemm_wgrad
```

### Linear

| Function | Backend Priority | Scaling Modes |
|----------|-----------------|---------------|
| `fp8_linear` | hipBLASLt → CK → Triton | per-tensor, per-token, blockwise, MXFP8, delayed, dynamic, static |

Fused **quant → GEMM → dequant** pipeline. The 7 scaling modes cover all combinations of weight/activation quantization granularity.

### Grouped GEMM (MoE)

| Scaling Type | Forward Kernel | Weight Gradient |
|--------------|----------------|-----------------|
| BF16 | `aiter.ops.triton.gemm.grouped.gmm` | `ptgmm` |
| FP8 per-tensor | `moe_gemm_a8w8` (Triton) | AITER BF16 GEMM |
| FP8 blockwise | `moe_gemm_a8w8_blockscale` (Triton) | AITER BF16 GEMM |
| FP8 per-token | Per-expert `gemm_a8w8_per_token_scale` | AITER BF16 GEMM |
| MXFP8 | Per-expert `gemm_a8w8_mxfp8_bpreshuffle` | AITER BF16 GEMM |

### Quantization Ops

| Function | Description |
|----------|-------------|
| `fp8_quantize` | Quantize tensor to FP8 — per-tensor (C++), blockwise (Triton), MXFP8 (Triton) |
| `fp8_dequantize` | Dequantize FP8 tensor back to BF16/FP32 |

## Normalization

```python
from atomrl.lumen.ops.normalization import rms_norm, layer_norm, fused_rms_norm_fp8
```

| Function | Backends | Fused Variants |
|----------|----------|----------------|
| `rms_norm` | ASM → CK → Triton | `fused_rms_norm_fp8` (RMSNorm + FP8 quantization in one kernel) |
| `layer_norm` | ASM → CK → Triton | `fused_layernorm_fp8` (LayerNorm + FP8 quantization) |

Fused norm+quant variants eliminate the intermediate BF16 write between normalization and quantization.

## Fused MLP

```python
from atomrl.lumen.ops.mlp import fused_gated_mlp, fused_mlp, fused_gated_mlp_fp8_store, fused_mlp_fp8_store
```

| Function | Description | Fused Path |
|----------|-------------|------------|
| `fused_gated_mlp` | Gated MLP: `down(act(gate(x)) * up(x))` | AITER Triton single-kernel (no bias) |
| `fused_mlp` | Ungated MLP: `down(act(up(x)))` | AITER Triton single-kernel (no bias) |
| `fused_gated_mlp_fp8_store` | Gated MLP + FP8 activation storage | Decomposed GEMM + FP8 save |
| `fused_mlp_fp8_store` | Ungated MLP + FP8 activation storage | Decomposed GEMM + FP8 save |

**FP8 activation store**: Activations saved as uint8 (1 byte) instead of BF16 (2 bytes) for ~50% activation memory reduction in the backward pass. Forward GEMMs are BF16; quantization happens only for storage.

**Supported activations**: `swiglu`, `geglu`, `reglu`, `gelu`, `relu`, `silu`

Compatible with `torch.compile`.

## MoE Routing

```python
from atomrl.lumen.ops.moe import fused_topk, fused_permute, fused_unpermute
```

| Function | Kernel | Description |
|----------|--------|-------------|
| `fused_topk` | ASM (`topk_softmax`) | Fused softmax + top-k expert gating |
| `fused_permute` | HIP/CK (`moe_sorting_fwd`) | Token-to-expert sort + pad |
| `fused_unpermute` | ASM (`moe_sum`) | Scatter expert outputs back with weighted sum |

The MoE pipeline: **gating** → **permute** → **grouped GEMM** → **unpermute**.

## RoPE

```python
from atomrl.lumen.ops.rope import fused_rope_fwd, fused_rope_bwd
```

Fused rotary position embedding (Triton only). Supports layouts: **SBHD**, **THD**, **2D** (vision), **3D** (video).

## Cross-Entropy

```python
from atomrl.lumen.ops.cross_entropy import vocab_parallel_cross_entropy
```

Vocab-parallel cross-entropy with online softmax + Triton kernel. Optional SDMA all-gather for distributed vocabulary.

## SDMA

```python
from atomrl.lumen.ops.sdma import AllgatherSdma, AllreduceSdma
```

Low-level SDMA collective wrappers backed by `mori.ccl`. Float32 only. Used for TP-integrated communication via `SdmaTpComm` module.
