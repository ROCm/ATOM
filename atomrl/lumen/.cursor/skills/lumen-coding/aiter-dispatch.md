# AITER Dispatch Guide

## Overview

Lumen dispatches all GPU kernel calls through AITER (AMD Inference and Training Engine Runtime). Lumen never owns CK or ASM kernels directly — it owns only Triton kernels under `lumen/kernels/`. AITER is always available on AMD targets.

## `try_backends()` — Core Dispatch

```python
from atomrl.lumen.ops.dispatch import try_backends, build_fallback_chain, Backend

chain = build_fallback_chain({
    Backend.ASM: _asm_fn,       # preferred: lowest latency on gfx94x
    Backend.CK: _ck_fn,         # fallback: ASM requires K%64==0
    Backend.TRITON: _triton_fn, # last resort: always available
})
return try_backends(chain, x, op_name="my_op")
```

**Priority order:** ASM → CK → Triton → torch (last resort, not for production)

**Exception handling:** Catches `RuntimeError`, `NotImplementedError`, `TypeError`, `ValueError`, Triton `CompilationError`, `OutOfResources`.

**Synchronization:** `torch.cuda.synchronize()` after every successful GPU kernel call, before attempting fallback (skipped in `torch.compile` mode — see [torch-compile.md](torch-compile.md)).

## AITER Imports

AITER is pinned in the supported Lumen image. Import the required public
kernel API directly; do not add availability probes or guard imports with
private capability checks.

```python
# GOOD: the supported image guarantees this public contract.
from aiter.ops.shuffle import shuffle_weight

shuffled = shuffle_weight(weight, layout=(16, 16))
```

```python
# BAD: AtomRL must not duplicate AITER's availability or implementation.
try:
    from aiter.ops.shuffle import shuffle_weight
except ImportError:
    shuffle_weight = local_shuffle_copy
```

This also forbids local "faster" copies of AITER layout transforms such as
MXFP4 weight/scale shuffle. Performance changes belong in AITER so every caller
uses one implementation and one contract.

## `LUMEN_BACKEND` Env Var Override

```bash
LUMEN_BACKEND=triton pytest tests/  # force Triton-only chain
```

`build_fallback_chain` reads `os.environ.get("LUMEN_BACKEND")` and returns a single-entry chain for that backend. Raises `RuntimeError` if the requested backend is not available.

## FP8 GEMM Backend Dispatch

| Scaling Type | Backends (priority) | Tuning Config |
|---|---|---|
| delayed/dynamic | hipBLASLt → CK (`gemm_a8w8_CK`) → Triton | `a8w8_tuned_gemm.csv` |
| per_token | Triton only (`gemm_a8w8_per_token_scale`) | None |
| blockwise/blockwise2d | CK (`gemm_a8w8_blockscale`) → Triton | `a8w8_blockscale_tuned_gemm.csv` |
| mxfp8 | Triton only (`gemm_mxfp8`) | None |

## BF16 GEMM

All GEMM kernels use TN layout: `Y = A @ W^T`, weight is always `(N, K)`.

- `tuned_gemm` / `hipb_mm` / `F.linear` require matching dtypes — cast after dequant
- Triton `gemm_a16w16`: BLOCK_SIZE_K >= 128 on gfx942
- ASM: requires K % 64 == 0 and N % 64 == 0
- Backward dgrad: pass `weight.t().contiguous()` as `w`

## FP8 Attention Dispatch

| Quant Type | Autograd Function | Scale Shape |
|---|---|---|
| `blockwise` | `AttentionTritonFunction` | 1D: `[B, H, ceil(S/bm)]` |
| `blockwise2d` | `AttentionTritonBlockwise2DFunction` | 2D: `[B, H, S//bm, D//bn]` |
| `mxfp8` | `AttentionTritonMXFP8Function` | MXFP8 block scales |

BF16 attention routes through AITER CK (`FlashAttnFunc`) as primary, Triton as fallback.

## Grouped GEMM

- BF16: `aiter.ops.triton.gmm`
- FP8 per-token: per-expert `gemm_a8w8_per_token_scale`
- MXFP8: per-expert `gemm_a8w8_mxfp8_bpreshuffle`
- FP8 blockwise: `moe_gemm_a8w8_blockscale` with `RoutingData` (built from `group_sizes`)
- Sequential fallback: iterates per-group when no fused kernel
- wgrad: `ptgmm` (permuted-tensor GMM)

## RoPE Dispatch (AITER Triton, no PyTorch fallback)

| Priority | Kernel | Layout | Use Case |
|----------|--------|--------|----------|
| 1 | `rope_cached_thd_positions_2c_fwd` | THD | Q+K pair (one launch) |
| 2 | `rope_cached_fwd` | SBHD | Single tensor RoPE |

Layout adaptation: Lumen BHSD → AITER SBHD/THD via permute before/after kernel call. Extended API includes `rope_fwd_2d` (vision) and `rope_fwd_3d` (video).

## MoE Fused Dispatch

| Component | Kernel | Notes |
|-----------|--------|-------|
| Token sorting | `moe_align_block_size_triton` | Triton block-padded sort |
| TopK | `aiter.topk_softmax` | ASM-backed |
| Permute | `aiter.moe_sorting_fwd` | |
| Unpermute | `aiter.moe_sum` | |

## Adding a New Op

1. Import the required public AITER kernel API directly.
2. Build fallback chain in the op function: `build_fallback_chain({Backend.ASM: ..., Backend.CK: ..., Backend.TRITON: ...})`
3. Call `try_backends(chain, *args, op_name="<name>")`
4. Add per-lambda comments explaining why each fallback exists
5. Add tests comparing against PyTorch reference using `compute_snr`
6. Reject the change if it introduces an AITER probe, private API, or local
   kernel/layout/shuffle implementation.
