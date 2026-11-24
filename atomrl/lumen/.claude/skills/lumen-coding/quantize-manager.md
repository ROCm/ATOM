# Quantize Manager Guide

## Overview

Lumen's FP8 quantization is managed by `ScalingManager` (in `lumen/quantize/scaling_manager.py`) and configured via `QuantConfig`. This guide covers scaling types, the blockwise2d dual behavior, amax history, and the scale misalignment rule.

## Scaling Types

| Type | GEMM Backends | FP8 Backward? | Key |
|------|---------------|---------------|-----|
| `delayed` | hipBLASLt -> CK -> Triton | Full FP8 bwd OK | Default TE-compatible path |
| `dynamic` | hipBLASLt -> CK -> Triton | Full FP8 bwd OK | Per-tensor, compute scale at runtime |
| `per_token` | Triton only | No FP8 wgrad | Per-row `(M,1)` scales, Lumen-only |
| `blockwise` | CK -> Triton | No FP8 wgrad | 1D block scales `(M, ceil(K/bs))` |
| `blockwise2d` | CK -> Triton (same kernels as `blockwise`) | No FP8 wgrad | Dual behavior (see below) |
| `mxfp8` | Triton only | No FP8 wgrad | gfx950+, MXFP8 block scales |
| `none` | BF16 GEMM | N/A | No quantization |

### Scale Misalignment Rule

**Only per-tensor scalar scales (delayed/dynamic/none) survive `weight.t()`**. Per-token `(N,1)`, blockwise `(N,K/bs)`, and mxfp8 block scales become misaligned after transpose -- FP8 backward restricted to `["delayed", "dynamic", "none"]`.

```python
if scaling_type in ("blockwise", "blockwise2d", "mxfp8", "per_token"):
    # FP8 wgrad disabled: scales misalign after weight transpose
    fp8_wgrad = False
```

### `blockwise` vs `blockwise2d` -- important `in` check

All `scaling_type == "blockwise"` checks in the codebase use `in ("blockwise", "blockwise2d")` across: `linear.py`, `rmsnorm.py`, `layernorm.py`, `grouped_gemm.py`, `scaling_manager.py`.

## blockwise2d Dual Behavior

| Context | Quantization | Scale Shape | Kernel |
|---------|-------------|-------------|--------|
| Linear (GEMM) | 1D block (identical to `blockwise`) | `(M, ceil(K/bs))` | `gemm_blockscale` (CK/Triton) |
| Attention | True 2D block via `Blockwise2DScaleManager` | `[B, H, S//bm, D//bn]` | `AttentionTritonBlockwise2DFunction` |

**Attention path:** `Blockwise2DScaleManager` caches FP8 Q/K/V tensors and their scales across forward/backward. Backward skips re-quantization and reuses dO scale across iterations. Triton kernel internally receives 1D block scales (`quantize_block_fp8` for Q/K, per-tensor for V).

**Linear path:** Routes through `_quant_blockwise` and `gemm_blockscale` -- identical to plain `blockwise`.

## ScalingManager Lifecycle

1. **Construction:** Created by `quant.enable(model, config=QuantConfig(...), dp_group=...)`.
2. **Forward:** `ScalingManager.quantize_fwd(tensor, scaling_type)` -> returns `(fp8_tensor, scale)`.
3. **Amax tracking (delayed):** `.item()` extracts amax from tensor -> updates `amax_history` buffer -> `amax_history_len` and `AmaxAlgo` (most_recent / max) control the sliding window.
4. **Scale computation (delayed):** `scale = max(amax_history) / fp8_max * (2 ** margin)`.
5. **Backward:** `ScalingManager.quantize_bwd(grad, scaling_type)` for gradient quantization. For blockwise2d linear, uses `quantize_bwd_delayed` (delayed per-tensor grad scaling).

## QuantConfig

```python
from atomrl.lumen.quantize.config import QuantConfig, ScalingType, AmaxAlgo

config = QuantConfig(
    scaling_type=ScalingType.BLOCKWISE,
    fp8_format="hybrid",          # e4m3 fwd, e5m2 bwd
    history_len=1024,
    amax_algo=AmaxAlgo.MAX,
    margin=0,
    fp8_wgrad=True,               # auto-disabled for non-scalar scales
    first_last_layers_bf16=True,  # keep first/last layers in BF16
    block_size=128,               # for blockwise/mxfp8
)
```

Key fields: `scaling_type`, `fp8_format` (e4m3/e5m2/hybrid), `history_len`, `amax_algo`, `margin`, `fp8_wgrad`, `first_last_layers_bf16`, `block_size`, `grad_quant_type`.

## FP8 Dtypes

- **e4m3:** `torch.float8_e4m3fn` (fnuz variant on gfx94x). Forward path default.
- **e5m2:** `torch.float8_e5m2` (fnuz on gfx94x). Used in backward for hybrid format.
- **Hybrid:** e4m3 forward, e5m2 backward.

Derive constants from spec:
```python
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
```

## Fused Norm+Quant

Available for all scaling types in both RMSNorm and LayerNorm. The norm kernel fuses normalization with FP8 quantization, avoiding materialization of the BF16 intermediate.

| Op | Fused variant | Scaling types |
|----|--------------|---------------|
| RMSNorm | `rmsnorm_blockwise`, `rmsnorm_mxfp8`, `rmsnorm_delayed`, `rmsnorm_dynamic` | All |
| LayerNorm | Same pattern | All |

## Grouped GEMM FP8 Quantization

For blockwise grouped GEMM (MoE):

- **Activations:** `quant_fp8_blockwise_impl(lhs, fp8_dtype, axis=1, block_size=128)` -> `lhs_fp8 + lhs_scales`
- **Weights:** Per-expert 2D block quant -> `rhs_fp8 [E, K, N] + rhs_scales [E, ceil(K/128), ceil(N/128)]`
- **RoutingData:** Built from `group_sizes` via `_build_routing_data_from_group_sizes`

## Common Pitfalls

1. **Double quantization:** Never pass already-FP8 tensors to `ScalingManager.quantize*()` -- `.abs()` is not implemented for FP8 dtypes. If tensors are pre-quantized, call the kernel directly (see blockwise2d attention fix).

2. **blockwise2d check:** Always use `in ("blockwise", "blockwise2d")`, not `== "blockwise"`.

3. **FP8 wgrad assumption:** Don't assume FP8 backward is available -- check `fp8_wgrad` flag, which is auto-set based on scaling type.

4. **Amax `.item()` in compile mode:** The `.item()` call in delayed scaling causes a graph break under `torch.compile`. In compile mode, amax update is deferred (see [torch-compile.md](torch-compile.md)).
