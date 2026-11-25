# Lumen Coding Reference

## TE → Lumen Module Mapping

| TE Class | Lumen Class | Notes |
|----------|-------------|-------|
| `TEColumnParallelLinear` | `LumenColumnParallelLinear` | |
| `TERowParallelLinear` | `LumenRowParallelLinear` | |
| `TELayerNormColumnParallelLinear` | `LumenLayerNormLinear` | |
| `TEGroupedLinear` | `LumenGroupedLinear` | MoE expert GEMM |
| `TEFusedMLP` | `LumenFusedMLP` / `LumenGatedMLP` | Gate+up+activation+down |
| `TEActivationOp` | `_ACTIVATION_TO_AITER` | SwiGLU/GeGLU/ReGLU |
| `TEDotProductAttention` | `LumenDotProductAttention` | Flash attention |
| `TEDotProductAttentionMLA` | `LumenDotProductAttentionMLA` | Multi-Latent Attention |
| `TESpecProvider` | `LumenSpecProvider` | Megatron layer specs |
| `te.pytorch.RMSNorm` | `LumenRMSNorm` | CK/Triton |
| `te.pytorch.LayerNorm` | `LumenLayerNorm` | CK/Triton |
| `make_graphed_callables()` | `LumenGraphedCallable` | HIP Graphs |
| `te_checkpoint` | `lumen_checkpoint` | FP8-aware activation recompute |
| `get_cpu_offload_context` | `CPUOffloadManager` | Optimizer state offload |

---

## Feature Parity Scorecard

| Category | Features | Supported | Lumen-Only | Missing | Partial | Deferred |
|----------|:--------:|:---------:|:----------:|:-------:|:-------:|:--------:|
| Attention | 13 | 10 | 0 | 1 | 1 | 1 |
| FP8 Quantization | 15 | 12 | 2 | 0 | 0 | 1 |
| Linear/GEMM | 11 | 9 | 0 | 0 | 2 | 0 |
| Normalization | 4 | 4 | 0 | 0 | 0 | 0 |
| Cross-Entropy | 4 | 3 | 1 | 0 | 0 | 0 |
| Communication | 5 | 2 | 1 | 0 | 2 | 0 |
| Advanced | 9 | 7 | 0 | 1 | 1 | 0 |
| **Total** | **~61** | **~47 (77%)** | **4** | **~2** | **~6** | **~2** |

### Remaining Work

**Missing (2):** CP A2A+P2P, FP8 Padding/Unpadding

**Partial (5):** TP Comm-GEMM Overlap (needs chunk pipelining), FP8 Param All-Gather (not wired), Delay wgrad (not wired), Sliding Window (Triton missing), MoE TopK/Aux Loss (aux loss at Megatron level)

**Deferred (2):** FP4 (no AMD HW), Softmax Variants (AITER hard-codes vanilla)

### Lumen-Only Features

1. **Blockwise2D quantization** — 2D block FP8 scaling for attention; linear uses 1D blockscale
2. **Per-Token FP8 scaling** — per-row FP8 quantization
3. **SDMA-based TP collectives** — System DMA on MI300X shared memory
4. **SDMA cross-entropy** — SDMA-routed all-gather in parallel cross-entropy

---

## Normalization

- Backward: prefer Triton when `requires_grad=True` (CK fwd doesn't save intermediates for CK bwd)
- Fused norm+quant: available for all scaling types in both LayerNorm and RMSNorm
- ASM: `layernorm2d_with_add_asm` for LayerNorm only (no RMSNorm ASM)

## Cross Entropy

- Triton kernel: `BLOCK_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(V))`
- Large vocab may need chunking (shared memory limit)
- SDMA all-gather route available (`use_sdma=True`, Lumen-only)

## Communication

| Feature | Status | Notes |
|---------|--------|-------|
| TP All-Gather / Reduce-Scatter | Supported | NCCL + SDMA option |
| TP Comm-GEMM Overlap | **Partial** | SDMA async, no chunk pipeline |
| Sequence Parallelism | Supported | |
| SDMA Collectives | Supported | **Lumen-only** (MI300X shared mem) |
| FP8 Param All-Gather | **Partial** | Manager exists, not wired |

---

## Submission Checklist

- [ ] Constants trace to hardware facts, math properties, or documented requirements
- [ ] No unnecessary abstraction layers
- [ ] Graceful degradation (fallback backends, clear error messages)
- [ ] Fallback paths logged — every `try/except`, `try_backends()`, or conditional fallback emits `logger.warning()`
- [ ] Fallback comments — `try_backends()` chains have per-lambda comments
- [ ] Performance-critical path free of Python-level overhead
- [ ] No TE replacement language in docstrings
- [ ] Tests use `compute_snr` or `check_close` against references
- [ ] Fallback paths tested — at least one test disables primary backend and verifies fallback
- [ ] AITER imports guarded by probe functions
