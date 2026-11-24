# Lumen Project

## Mission

Lumen is a drop-in replacement library for NVIDIA TransformerEngine (TE) within Megatron-Core, targeting AMD GPUs (gfx94x / gfx950). Every module, op, and kernel exists to provide feature-equivalent alternatives to TE's interfaces so that Megatron-Core can run on AMD hardware without modifying its own source code. All API design, testing, and feature prioritization should be evaluated against this goal.

## Code Boundary: Lumen vs AITER

New GPU kernels (Triton, CK, ASM) belong in the official [ROCm/aiter](https://github.com/ROCm/aiter) repository, not in Lumen. Lumen only implements the **dispatch and calling logic** — choosing the right backend, marshalling arguments, and handling fallback. If a feature requires a new kernel, add it to AITER and expose it through AITER's Python API; then call it from Lumen's ops layer via the standard `try_backends()` dispatch.

## First-Principles Code Generation

When generating code for Lumen, always reason from first principles — decompose the problem to its fundamental truths, then build the solution upward. Never copy-paste patterns blindly or code by analogy.

1. **Clarify the problem before writing code.** State what the code must achieve in one sentence. If you cannot, the problem is not yet understood.
2. **Identify the constraints.** For GPU kernels: memory bandwidth, register pressure, wave occupancy, data layout. For Python modules: API contract, backward compatibility, composability.
3. **Derive the solution from constraints**, not from "how others did it." Each design choice (tiling size, scaling strategy, dispatch order) must trace back to a hardware fact or a mathematical property.
4. **Eliminate unnecessary abstraction.** Add a layer only when it removes duplication across >=2 call sites or encapsulates a non-obvious invariant. Premature abstraction obscures the reasoning chain.

### Anti-Patterns

- Magic numbers without derivation (e.g., `scale = 448.0` instead of `amax / torch.finfo(torch.float8_e4m3fn).max`)
- Abstraction with only one subclass — use a plain function
- Copying another framework's API shape blindly — shape API by Lumen's own requirements

## Coding Conventions

### No TransformerEngine (TE) replacement language

Do NOT describe Lumen modules as "drop-in replacements" for TransformerEngine or reference TE classes in docstrings/comments.

```python
# BAD
"""Drop-in replacement for TE's ``TEColumnParallelLinear``."""
"""Mirrors ``TESpecProvider`` — returns Lumen classes instead of TE classes."""

# GOOD
"""Column-parallel linear using Lumen GEMM."""
"""Provides Lumen module classes for Megatron-Core layer specs."""
```

Exception: actual code that references TE class names (e.g. `TEGroupedMLP` in imports or runtime logic) is fine — only remove TE references from documentation and comments.

### Fallback Logging

Every fallback path must log why and comment the intent:

```python
def my_op(x):
    try:
        return _asm_path(x)
    except RuntimeError as e:
        logger.warning("my_op: ASM failed (%s), falling back to Triton", e)
        return _triton_path(x)
```

`try_backends()` chains need per-lambda comments. Conditional fallbacks need comments explaining why.

## Test Conventions

### FP8 / GPU tests MUST run on CUDA

FP8 ops (`torch.float8_e4m3fn`, `torch.float8_e4m3fnuz`) require CUDA. Tests using FP8 activation store, FP8 quantization, or any Lumen FP8 path must create tensors on CUDA and move modules to CUDA.

### Leaf tensor rule for gradient SNR tests

When comparing gradients between a Lumen op and a reference, the reference weight tensor MUST be a leaf. Multiplying after `requires_grad=True` creates a non-leaf tensor whose `.grad` is always `None`:

```python
# BAD — w is non-leaf (has grad_fn=MulBackward), w.grad stays None
w = torch.randn(128, 64, requires_grad=True) * 0.02

# GOOD — multiply first, then enable grad tracking
w = (torch.randn(128, 64) * 0.02).requires_grad_(True)
```

### Forward signature: always pass ALL positional args

Megatron-compatible modules require `attention_mask` as a positional argument. Always pass it, even in benchmarks:

```python
# BAD
attn(q, k, v)

# GOOD
attn(q, k, v, attention_mask=None)
```

### FP8 backend auto-mapping

`LumenAttention` defaults to `backend_type="aiter_csrc"`. When `quant_type` is set (FP8 mode), the module auto-maps `aiter_csrc -> aiter_triton`. Tests that want FP8 attention need not set `backend_type` explicitly.

### Tensor dimensions must match parallel partitioning

`LumenRowParallelLinear(in_features, out_features, tp_size=N)` with `input_is_parallel=True` expects input of size `in_features / tp_size`.

### MXFP8 quant_block_size must divide head_dim

Triton MXFP8 kernel requires `padded_head_dim % quant_block_size == 0`. When parametrizing `quant_block_size`, use a head_dim divisible by all values.

### Assert gradients before computing SNR

Always verify gradients are non-None before passing to `compute_snr`.

## AITER Backend Constraints

### Dispatch Architecture

Lumen uses `try_backends()` in `lumen/ops/dispatch.py` to fall through backends on failure. It catches `RuntimeError`, `NotImplementedError`, `TypeError`, `ValueError`, Triton `CompilationError`, and `OutOfResources`. Always call `torch.cuda.synchronize()` after GPU kernels to surface asynchronous errors before fallback.

### BF16 GEMM

- **dtype matching**: `tuned_gemm` / `hipb_mm` / `F.linear` require both inputs to have the same dtype. After dequantizing FP8 weights, always cast back to target dtype.
- **Triton `gemm_a16w16`**: BLOCK_SIZE_K >= 128 required on gfx942.
- **ASM kernels**: Require K % 64 == 0 and N % 64 == 0.
- **TN layout**: All BF16/FP8 GEMM kernels compute Y = A @ W^T. Weight `w` is always (N, K).

### FP8 Quantized GEMM Dispatch

| scaling_type | Backends (priority order) | Tuning |
|---|---|---|
| delayed/dynamic | hipBLASLt -> CK (`gemm_a8w8_CK`) -> Triton | CK reads `a8w8_tuned_gemm.csv` |
| per_token | Triton only (`gemm_a8w8_per_token_scale`) | No tuned config |
| blockwise | CK (`gemm_a8w8_blockscale`) -> Triton | CK reads `a8w8_blockscale_tuned_gemm.csv` |
| mxfp8 | Triton only (`gemm_mxfp8`) | No tuned config |

### FP8 Backward Scale Misalignment

Per-tensor scalar scales (delayed/dynamic) survive weight transposition. Per-token, blockwise, and mxfp8 block scales become misaligned after `weight_fp8.t()`. Backward with full FP8 quantization is restricted to `["delayed", "dynamic", "none"]` scaling types.

### Normalization

- **Backward**: Prefer Triton when `requires_grad=True` (CK fwd doesn't save intermediates for bwd).
- **Fused norm+quant**: Available for all scaling types in both LayerNorm and RMSNorm.
- **ASM norm**: `layernorm2d_with_add_asm` available for LayerNorm only (no RMSNorm ASM).

### General Rules

- AITER is a pinned runtime dependency of the Lumen image; import the required
  public kernel API directly.
- Use `try_backends()` only for supported kernel implementations, not for
  dependency-availability probing.
- Do not probe AITER imports/symbols/private attributes or copy its capability
  tables.
- Do not reimplement AITER kernels, layouts, MXFP4 shuffles, or communication
  primitives in Lumen; land the implementation in AITER and call its public API.

## Skills

Domain-specific guides are available as Claude skills in `.claude/skills/`:

- **lumen-coding** — Architecture, dispatch, FP8 conventions, code review
- **lumen-benchmark** — GPU timing, distributed setup, overlap patterns
- **lumen-test** — Test architecture, accuracy metrics, templates
- **atomrl** — AtomRL framework integration and execution checklists
- **lumen-sdma** — SDMA vs NCCL backend decision guide
