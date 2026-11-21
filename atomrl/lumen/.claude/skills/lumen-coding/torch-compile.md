# torch.compile Compatibility Guide

**Design doc:** `Lumen/docs/plans/2026-03-20-torch-compile-design.md`
**Test plan:** `Lumen/docs/plans/2026-03-20-torch-compile-test-plan.md`

## Goal

Make Lumen's forward and backward passes compatible with `torch.compile`:
- **Ideal:** `fullgraph=True` (single compiled graph, no breaks)
- **Pragmatic:** minimal graph breaks -- each op internally compilable; breaks only at distributed comm boundaries

## 6 Blockers Identified

| # | Blocker | Severity | Location |
|---|---------|----------|----------|
| 1 | `try_backends` try/except dispatch | Critical | `dispatch.py` (all ops) |
| 2 | `torch.cuda.synchronize()` on success path | Critical | Inside `try_backends` |
| 3 | 15 unregistered `autograd.Function` subclasses | Major | MLP, norm, CE, CP, SDMA |
| 4 | Data-dependent `.item()` calls | Major | `grouped_gemm.py`, `scaling_manager.py`, `fused_moe.py` |
| 5 | Manual CUDA stream usage | Moderate | SDMA overlap, HIP graphs, CPU offload |
| 6 | Runtime try/except import guards | Minor | `attention.py`, `grouped_gemm.py` |

## Design: Compile Guard Dual-Mode

Core principle: `torch.compiler.is_compiling()` branches between compile mode (fixed backend, no try/except, no sync) and eager mode (full fallback chain, unchanged).

### `try_backends` Changes

```python
def try_backends(backends, *args, op_name="op", **kwargs):
    if torch.compiler.is_compiling():
        _, fn = backends[0]  # direct call, no fallback
        return fn(*args, **kwargs)

    # Eager: original fallback logic unchanged
    for backend, fn in backends:
        try:
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            return result
        except _catchable as exc:
            logger.warning(...)
    raise RuntimeError(...)
```

**Key insight:** In compile mode, `backends[0]` is the highest-priority available backend, pre-filtered by `build_fallback_chain`. Backend availability is determined at import time.

### `LUMEN_BACKEND` Env Var

```bash
LUMEN_BACKEND=triton  # force single-backend chain
```

### autograd.Function Registration

**Phase 1:** `allow_in_graph` for all 15 unregistered Functions (opaque to Dynamo but no graph break).

**Phase 3:** Migrate high-frequency Functions to `torch.library.custom_op` + `register_fake` for Inductor integration.

### Triton Kernel Registration

Register via `torch.library.triton_op` + `wrap_triton` for Inductor visibility:

```python
@triton_op("lumen::quant_fp8_blockwise_impl", mutates_args={"output", "scale"})
def quant_fp8_blockwise_impl(data, output, scale, block_size, fp8_max):
    wrap_triton(_quant_fp8_blockwise_kernel[grid])(...)
```

### `.item()` Elimination

| Location | Fix |
|----------|-----|
| `grouped_gemm.py`: `int(group_sizes.sum().item())` | Use symbolic size or pre-compute |
| `scaling_manager.py`: amax `.item()` | Keep in eager only; defer in compile mode |
| `fused_moe.py`: seed `.item()` | Use `torch.randint` directly |

### SDMA Exclusion

SDMA paths are intentionally excluded via `@torch.compiler.disable`:

```python
@torch.compiler.disable
def sdma_allgather(tensor, group, ...):
    ...
```

## Implementation Phases

| Phase | Goal | Exit Criteria |
|-------|------|---------------|
| **P1** (Week 1) | Dispatch dual-mode + Function registration | `fullgraph=False` runs; no graph breaks in `try_backends`; eager unchanged |
| **P2** (Week 2) | `triton_op` registration + `.item()` elimination | `fullgraph=True` works single-GPU fwd (Triton backend); 0 `.item()` in compile path |
| **P3** (Week 3) | `custom_op` migration for high-frequency ops | `fullgraph=True` end-to-end transformer block; `opcheck` passes |
| **P4** (Ongoing) | Hardening, CI, benchmarks | Compile tests on every PR; benchmark baselines |

## Testing

Tests live in `tests/compile/`:

```python
def test_try_backends_compile_direct_path():
    """Under fullgraph=True, try_backends calls backends[0] directly."""
    ...

def test_transformer_block_compiles():
    """Full block (attention + norm + linear) compiles into 1 graph."""
    counter = CompileCounter()
    compiled = torch.compile(block, backend=counter, fullgraph=True)
    compiled(x)
    assert counter.frame_count == 1
```

## Writing Compile-Compatible Code

When adding new ops:

1. **No try/except in hot path** -- use `try_backends()` which handles compile mode
2. **No `.item()` / `.shape` branching on dynamic values** -- use symbolic sizes or compile-time constants
3. **No `torch.cuda.synchronize()`** -- handled by `try_backends()` in eager mode
4. **Register autograd.Function** -- add `allow_in_graph` or use `custom_op`
5. **Register Triton kernels** -- use `triton_op` + `wrap_triton`
6. **SDMA/stream ops** -- wrap with `@torch.compiler.disable`
