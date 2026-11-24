---
name: lumen-test
description: "Use when writing, reviewing, or planning tests for Lumen ops, modules, or integration. Covers test architecture, accuracy metrics (SNR thresholds), templates, naming conventions, and anti-patterns."
---

# Lumen Test Guide

## Mission

Every Lumen feature, kernel, or module must have tests before it can be marked "Supported". Tests compare against pure-PyTorch references using SNR / `check_close` metrics and follow the conventions below.

## Directory Guide

| You are testing... | Directory | Pattern |
|------------------|-----------|---------|
| A function in `lumen/ops/` | `tests/ops/` | Compare vs PyTorch reference using `compute_snr` |
| An `nn.Module` in `lumen/modules/` | `tests/modules/` | Mock distributed primitives, test construction + fwd + bwd |
| Megatron/FSDP integration | `tests/models/` | Mock `MegatronConfig`, test patching + CLI args |
| `QuantConfig` / `ScalingManager` | `tests/quantize/` | Config parsing, scale computation, amax history |
| Utilities (graphs, checkpoint) | `tests/utils/` | Lifecycle, context managers |
| Multi-GPU collectives | `tests/env/` | NCCL/RCCL spawn-based |
| `torch.compile` compat | `tests/compile/` | `CompileCounter`, `fullgraph=True`, SNR vs eager |

## Shared Infrastructure

### References (`tests/ops/conftest.py`)

| Reference | Use for |
|-----------|---------|
| `attention_ref(q, k, v, sm_scale, causal)` | MHA (BSHD, f32 compute) |
| `rmsnorm_ref(x, weight, eps)` | RMSNorm |
| `layernorm_ref(x, weight, bias, eps)` | LayerNorm |
| `grouped_gemm_ref(lhs, rhs, group_sizes, bias)` | Per-expert GEMM |
| `cross_entropy_ref(logits, target, ...)` | Cross-entropy |
| `fp8_quant_dequant_ref(tensor, fp8_dtype)` | Per-tensor FP8 round-trip |
| `fp8_blockwise_quant_dequant_ref(tensor, block_size, fp8_dtype)` | Blockwise FP8 round-trip |

No shared reference? Write a `_`-prefixed reference **in the test file**, not conftest.

### Accuracy Metrics

| Metric | Use | Call |
|--------|-----|------|
| `compute_snr(ref, test)` | Primary -- SNR in dB | `from conftest import compute_snr` |
| `check_close(a, b, ...)` | Element-wise with outlier tolerance (5%) | `from conftest import check_close` |
| `torch.testing.assert_close` | Strict -- non-quantized / BF16 paths | stdlib |

### SNR Thresholds

| Path | Forward | Backward dX | Backward dW |
|------|:-------:|:-----------:|:-----------:|
| BF16 attention | >= 20 dB | >= 15 dB | >= 15 dB |
| BF16 causal bwd | -- | >= 10 dB | >= 10 dB |
| FP8 blockwise | >= 15 dB | >= 8 dB | -- |
| FP8 delayed/dynamic | >= 12 dB | >= 6 dB | >= 6 dB |
| FP8 per_token | >= 12 dB | -- | -- |
| MXFP8 | >= 10 dB | >= 5 dB | -- |
| BF16 GEMM (none) | >= 25 dB | >= 20 dB | -- |
| Fused MLP | `assert_close` | `assert_close` | -- |

When adding relaxed thresholds, add a comment explaining why.

### Config Dataclasses & Fixtures

```python
from conftest import AttnConfig, NormConfig, LinearConfig

CONFIGS = [AttnConfig(128, 128, 8, 8, 64, 64)]
CONFIG_IDS = [repr(c) for c in CONFIGS]

@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_something(config): ...
```

| Fixture | Scope | Behavior |
|---------|-------|----------|
| `seed_rng` | autouse (`tests/ops/`) | `torch.manual_seed(0)` + `torch.cuda.manual_seed(0)` |
| `device` | function | Returns `"cuda"` |

## Templates

### Op-level (`tests/ops/`)

```python
"""Tests for atomrl.lumen.ops.<module>: <brief>."""
import pytest, torch
from conftest import SomeConfig, some_ref, compute_snr
import atomrl.lumen.ops.<module> as ops

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

CONFIGS = [SomeConfig(...)]

@_CUDA
@pytest.mark.parametrize("config", CONFIGS, ids=[repr(c) for c in CONFIGS])
def test_op_fwd(config):
    x = torch.randn(..., device="cuda", dtype=torch.bfloat16)
    ref = some_ref(x, ...)
    out = ops.some_op(x, ...)
    assert compute_snr(ref, out) > 20

@_CUDA
@pytest.mark.parametrize("config", CONFIGS, ids=[repr(c) for c in CONFIGS])
def test_op_fwd_bwd(config):
    x_ref = torch.randn(..., device="cuda", requires_grad=True)
    out_ref = some_ref(x_ref, ...); out_ref.sum().backward()
    x = torch.randn(..., device="cuda", requires_grad=True)
    out = ops.some_op(x, ...); out.sum().backward()
    assert compute_snr(out_ref, out) > 20
    assert compute_snr(x_ref.grad, x.grad) > 15
```

### Module-level (`tests/modules/`)

```python
@_CUDA
class TestSomeModule:
    def test_construction(self):
        m = SomeModule(64, 128)
        assert m.weight.shape == (128, 64)

    def test_forward_shape(self):
        m = SomeModule(64, 128).to("cuda")
        out = m(torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16))
        assert out.shape == (2, 16, 128)

    def test_backward_runs(self):
        m = SomeModule(64, 128).to("cuda")
        x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None and m.weight.grad is not None
```

### Multi-GPU spawn pattern

See [reference.md](reference.md) for the full `mp.spawn` + `_worker` pattern and SDMA test setup.

## Hardware Skip Markers

```python
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
# AITER is part of the supported test image; do not probe or skip it here.
_GFX950 = pytest.mark.skipif(not _is_gfx950(), reason="gfx950+ required")
_MULTI_GPU = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need >=2 GPUs")
```

## Scaling Type Parametrization

```python
ALL_SCALING_TYPES = ["delayed", "dynamic", "per_token", "blockwise", "blockwise2d", "mxfp8", "none"]
BWD_SCALING_TYPES = ["delayed", "dynamic", "none"]  # only scalar scales survive .t()
ATTN_FP8_TYPES = ["blockwise", "blockwise2d", "dynamic", "delayed", "per_token"]
ATTN_MXFP8 = ["mxfp8"]  # gfx950+ only
```

## Naming Conventions

| Pattern | Meaning |
|---------|---------|
| `test_<op>_fwd` | Forward-only correctness |
| `test_<op>_fwd_bwd` | Forward + backward |
| `test_<op>_<variant>` | Specific variant (causal, gqa, fp8) |
| `test_<op>_edge_<case>` | Edge case |
| `test_construction` | Module construction, shapes |
| `test_forward_shape` | Output shape |
| `test_backward_runs` | Backward completes, grads exist |
| `test_matches_reference` | SNR or assert_close vs reference |

## Anti-Patterns

```python
# BAD: CPU when op needs CUDA
x = torch.randn(2, 16, 64)  # defaults to CPU

# BAD: shape-only check
assert out.shape == q.shape  # proves nothing about correctness

# BAD: backward without verifying grads
out.sum().backward()  # test ends -- no grad check

# BAD: block_size > seqlen
attention_fp8_quant(q, k, v, quant_block_size=256, block_m_fwd=256)  # seqlen=128 -> FAILS

# BAD: autograd.grad outside enable_grad
with torch.enable_grad():
    act_val = act_fn(x.detach().requires_grad_(True))
act_grad = torch.autograd.grad(act_val.sum(), x)  # FAILS -- outside enable_grad
```

## What to Test per Feature

For detailed per-feature test requirements -> see [reference.md](reference.md).

## Coverage Gaps

For the complete test inventory and coverage gap list -> see [reference.md](reference.md).
