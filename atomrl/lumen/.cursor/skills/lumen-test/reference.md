# Lumen Test Reference

## Test Inventory by Directory

### tests/ops/ (12 files, ~4,060 lines)

| File | Lines | Tests | What it covers |
|------|:-----:|:-----:|----------------|
| `conftest.py` | 243 | — | Shared references, metrics, fixtures, config dataclasses |
| `test_attention.py` | 523 | 22 | BF16/FP8/MXFP8 attention fwd+bwd, causal, GQA, cross-attn, return_lse |
| `test_attention_cp_a2a_sdma.py` | 320 | 6 | CP A2A layout helpers, SDMA vs NCCL golden, performance |
| `test_cp_p2p.py` | 77 | 6 | Online softmax update, existence checks (no distributed ring test) |
| `test_cross_entropy.py` | 344 | 9 | Parallel CE fwd+bwd, label smoothing, reduce loss, CG capturable |
| `test_dispatch.py` | 194 | 13 | Backend enum, fallback order, try_backends chain |
| `test_fused_rope.py` | 277 | 17 | RoPE assertions, probes, layout adapters, correctness vs reference |
| `test_grouped_gemm.py` | 221 | 6 | Grouped GEMM fwd, bias, wgrad, FP8 scaling, zero groups |
| `test_linear.py` | 361 | 7 | FP8 linear fwd+bwd (all 7 scaling types), bias, weight-only, bf16 wgrad |
| `test_moe_routing.py` | 280 | 14 | fused_topk/permute/unpermute assertions + correctness |
| `test_normalization.py` | 682 | 10 | RMSNorm/LayerNorm fwd+bwd, fused quant (all scaling types), MXFP8 |
| `test_quantize.py` | 458 | 10 | FP8 tensorwise, blockwise, MXFP8 vs torchao references |
| `test_sdma.py` | 965 | 19 | SDMA context, allgather, allreduce, all2all, distributed, performance |

### tests/modules/ (17 files, ~3,700 lines)

| File | Lines | Tests | What it covers |
|------|:-----:|:-----:|----------------|
| `test_attention_module.py` | 739 | 28 | LumenAttention construction, forward, FP8, blockwise2D, benchmark |
| `test_attention_mla_module.py` | 265 | 10 | MLA construction, forward, benchmark |
| `test_attention_megatron_module.py` | 304 | 10 | LumenDotProductAttention (Megatron), FP8 forward SNR |
| `test_comm_overlap.py` | 229 | 7 | SdmaTpComm async, column/row parallel overlap path selection |
| `test_cross_entropy.py` | 63 | 3 | lumen_parallel_cross_entropy dispatch, SDMA flag (mocked) |
| `test_delay_wgrad.py` | 55 | 4 | _DeferredWgrad defer/execute/accumulate |
| `test_fp8_activation_store.py` | 77 | 5 | Gated + ungated MLP FP8 store: shape, backward, approximate match |
| `test_fused_mlp.py` | 154 | 10 | Fused MLP ops + LumenFusedMLP/LumenGatedMLP modules |
| `test_grad_accum_fusion.py` | 198 | 8 | Gradient accumulation fusion correctness |
| `test_grouped_linear.py` | 272 | 15 | LumenGroupedLinear construction, forward, state_dict, FP8, benchmark |
| `test_layernorm_linear.py` | 168 | 8 | LumenLayerNormLinear construction, forward, FP8 |
| `test_parallel_linear.py` | 272 | 12 | Column/Row parallel linear construction, forward, backward, benchmark |
| `test_quantize_module.py` | 144 | 12 | LumenLinear construction, forward, backward, benchmark |
| `test_sdma_comm.py` | 409 | 12 | SdmaTpContext/Comm unit tests, distributed, performance |

### tests/models/ (4 files, ~2,630 lines)

| File | Lines | Tests | What it covers |
|------|:-----:|:-----:|----------------|
| `test_megatron.py` | 1506 | 90+ | Megatron compat norms, patching, CLI args, FP8 training, LoRA, benchmarks |
| `test_fsdp.py` | 820 | 60+ | FSDP norms, patching, golden output, FP8 training, LoRA |
| `test_fsdp2.py` | 64 | 5 | FSDP2 CLI args, apply function exists, basic forward |
| `test_utils.py` | 241 | 15 | safe_add_argument, peek_backend, sha256, HF download |

### tests/quantize/ (4 files, ~1,100 lines)

| File | Lines | Tests | What it covers |
|------|:-----:|:-----:|----------------|
| `test_scaling_manager.py` | 708 | 55+ | ScalingManager lifecycle, amax history, quantize, FP8 param, blockwise2D |
| `test_config.py` | 200 | 28 | QuantFormat, ScalingType, AmaxAlgo, QuantConfig parsing |
| `test_fp8_params.py` | 89 | 8 | FP8 param quantize/dequant, FP8ParamManager hooks |

### tests/core/ (3 files, ~313 lines)

| File | Lines | Tests | What it covers |
|------|:-----:|:-----:|----------------|
| `test_grad_quant.py` | 156 | 9 | Grad quant types, FP8/MXFP8 round-trip, FP4 not-implemented |
| `test_float8.py` | 111 | 7 | FP8 dtype detection, OCP support, e4m3/e5m2 validity |
| `test_utils.py` | 46 | 4 | Core utility functions |

### tests/utils/ (3 files, ~273 lines)

| File | Lines | Tests | What it covers |
|------|:-----:|:-----:|----------------|
| `test_cpu_offload.py` | 84 | 7 | CPUOffloadManager/Context lifecycle |
| `test_hip_graphs.py` | 88 | 7 | LumenGraphedCallable/Module, make_graphed_callables |
| `test_checkpoint.py` | 101 | 6 | FP8ScalingContext, lumen_checkpoint |

### tests/env/ (1 file, ~664 lines)

| File | Lines | Tests | What it covers |
|------|:-----:|:-----:|----------------|
| `test_nccl.py` | 664 | 8 | NCCL/RCCL barrier, allreduce, broadcast, allgather, reduce_scatter, P2P |

---

## Feature → Test Mapping

### Attention

| Feature | Status | Test File | Coverage |
|---------|:------:|-----------|:--------:|
| Flash Attention | Supported | `test_attention.py` | Full |
| FP8 DPA | Supported | `test_attention.py` (fp8 tests) | Full |
| FP8 MHA | Supported | `test_attention_module.py` | Full |
| MLA | Supported | `test_attention_mla_module.py` | Full |
| CP P2P | Supported | `test_cp_p2p.py` | **Thin** — no distributed test |
| CP A2A | Supported | `test_attention_cp_a2a_sdma.py` | Full |
| CP A2A+P2P | Missing | — | None |
| Sliding Window | Partial | — | **None** |
| Attention Bias | Supported | — | **None** |
| ALiBi Slopes | Supported | — | **None** |
| GQA | Supported | `test_attention.py` (gqa tests) | Full |
| Return LSE | Supported | `test_attention.py` (return_lse tests) | Full |

### FP8 Quantization

| Feature | Status | Test File | Coverage |
|---------|:------:|-----------|:--------:|
| Delayed | Supported | `test_scaling_manager.py`, `test_linear.py` | Full |
| Dynamic | Supported | `test_quantize.py`, `test_linear.py` | Full |
| Blockwise | Supported | `test_quantize.py`, `test_linear.py` | Full |
| Blockwise2D | Lumen-only | `test_scaling_manager.py`, `test_attention_module.py` | Full |
| MXFP8 | Supported | `test_quantize.py`, `test_attention.py` | Full |
| Per-Token | Lumen-only | `test_linear.py` (parametrized) | Adequate |
| Hybrid (e4m3/e5m2) | Supported | `test_config.py` (config only) | **Thin** — no GEMM test |
| FP4 | Deferred | `test_grad_quant.py` (NotImplementedError) | N/A |
| Amax History | Supported | `test_scaling_manager.py` | Full |
| FP8 Margin | Supported | `test_config.py` | Full |
| FP8 wgrad | Supported | `test_linear.py` | Full |
| Reduce Amax | Supported | `test_megatron.py` (config path) | **Thin** — no distributed |
| First/Last BF16 | Supported | `test_megatron.py`, `test_fsdp.py` | Adequate |

### Linear / GEMM

| Feature | Status | Test File | Coverage |
|---------|:------:|-----------|:--------:|
| Column Parallel | Supported | `test_parallel_linear.py` | Full |
| Row Parallel | Supported | `test_parallel_linear.py` | Full |
| LayerNorm+Linear | Supported | `test_layernorm_linear.py` | Full |
| Grouped Linear | Supported | `test_grouped_linear.py`, `test_grouped_gemm.py` | Full |
| Fused MLP | Supported | `test_fused_mlp.py` | Full |
| Fused Activation | Supported | `test_fused_mlp.py` | Full |
| FP8 GEMM | Supported | `test_linear.py` | Full |
| FP8 Activation Store | Supported | `test_fp8_activation_store.py` | **Thin** |
| TP Comm-GEMM Overlap | Partial | `test_comm_overlap.py` | **Unit-only** |
| Grad Accum Fusion | Supported | `test_grad_accum_fusion.py` | Full |
| Delay wgrad | Partial | `test_delay_wgrad.py` | **Thin** |

### Normalization

| Feature | Status | Test File | Coverage |
|---------|:------:|-----------|:--------:|
| RMSNorm | Supported | `test_normalization.py` | Full |
| LayerNorm | Supported | `test_normalization.py` | Full |
| Zero-Centered Gamma | Supported | `test_layernorm_linear.py` | Adequate |
| Fused Norm+Quant | Supported | `test_normalization.py` | Full |

### Cross-Entropy

| Feature | Status | Test File | Coverage |
|---------|:------:|-----------|:--------:|
| Parallel CE | Supported | `test_cross_entropy.py` (ops + module) | Full |
| CG Capturable | Supported | `test_cross_entropy.py` | Full |
| Label Smoothing | Supported | `test_cross_entropy.py` | Full |
| SDMA CE | Lumen-only | `test_cross_entropy.py` (module, mocked) | **Thin** |

### Communication

| Feature | Status | Test File | Coverage |
|---------|:------:|-----------|:--------:|
| TP AG/RS | Supported | `test_nccl.py`, `test_sdma.py` | Full |
| TP Comm-GEMM Overlap | Partial | `test_comm_overlap.py` | **Unit-only** |
| Sequence Parallelism | Supported | `test_parallel_linear.py` (flag check) | **Thin** |
| SDMA | Lumen-only | `test_sdma.py` | Full |
| FP8 Param AG | Partial | `test_fp8_params.py` | **Local only** |

### Advanced / Misc

| Feature | Status | Test File | Coverage |
|---------|:------:|-----------|:--------:|
| HIP Graphs | Supported | `test_hip_graphs.py` | Full |
| TE Checkpoint | Supported | `test_checkpoint.py` | Full |
| CPU Offload | Supported | `test_cpu_offload.py` | Full |
| Fused RoPE | Supported | `test_fused_rope.py` | Full |
| MoE Permute/Unpermute | Partial | `test_moe_routing.py` | Full |
| MoE TopK / Aux Loss | Partial | `test_moe_routing.py`, `test_moe_fused_router.py` | AITER top-k; PyTorch aux-loss formula |
| FP8 Padding/Unpadding | Missing | — | None |
| FSDP2 | Supported | `test_fsdp2.py` | **Thin** |
| LoRA | Supported | `test_megatron.py`, `test_fsdp.py` | Full |

---

## What to Test per Feature Category

### Attention

| Feature | Required Tests |
|---------|---------------|
| New attention variant | Forward SNR vs `attention_ref`, backward dQ/dK/dV SNR, causal mode, GQA |
| Sliding window (`window_size`) | Forward correctness with window mask, compare masked vs full attention |
| Attention bias | All `core_attention_bias_type` variants, forward SNR, backward SNR |
| ALiBi slopes | Forward with slopes tensor, compare vs manual bias addition |
| Return LSE | Check `lse` shape and dtype, verify `exp(lse)` matches row softmax sums |
| CP (context parallelism) | Multi-GPU spawn, output matches non-CP reference (same global input) |

### FP8 Quantization

| Feature | Required Tests |
|---------|---------------|
| New scaling type | Forward GEMM SNR, backward dX SNR, round-trip quant→dequant vs `fp8_quant_dequant_ref` |
| Hybrid (e4m3/e5m2) | Forward uses e4m3, backward uses e5m2, verify dtype of intermediate tensors |
| Amax/delayed scale | `delayed_scale_ref` comparison, history buffer length, algorithm (most_recent/max) |
| FP8 param all-gather | Distributed: quant on each rank, all-gather, dequant, compare to BF16 all-gather |

### Linear / GEMM

| Feature | Required Tests |
|---------|---------------|
| New GEMM path | Forward SNR vs `torch.mm`, backward dX and dW SNR, parametrize all scaling types |
| Fused MLP | `torch.testing.assert_close` vs decomposed reference, backward correctness |
| TP overlap | Multi-GPU: verify output matches non-overlapped path, measure overlap ratio |
| Grad accum fusion | Compare accumulated gradient vs sequential accumulation |

### Normalization

| Feature | Required Tests |
|---------|---------------|
| Norm op | Forward+backward SNR vs `rmsnorm_ref`/`layernorm_ref`, parametrize shapes |
| Fused norm+quant | Forward output SNR, verify quantized output dtype, scale shape per scaling type |
| Zero-centered gamma | Forward with `zero_centered_gamma=True`, compare vs `(1 + gamma) * norm(x)` |

### Communication

| Feature | Required Tests |
|---------|---------------|
| SDMA collective | Multi-GPU: output matches NCCL reference, dtype variants (bf16, fp32) |
| Sequence parallelism | Verify SP scatter/gather round-trip preserves tensor values |
| SDMA cross-entropy | Multi-GPU: SDMA path output matches non-SDMA reference |

### Module

| Feature | Required Tests |
|---------|---------------|
| Any `nn.Module` | Construction (shapes, defaults), forward shape, backward runs, FP8 variant |
| Checkpoint/recompute | Verify recomputation same output, memory reduction |
| CPU offload | Hooks registered, tensors moved to CPU, restored on forward |
| HIP graphs | Capture succeeds, replayed output matches eager output |

---

## Distributed Test Pattern

### Multi-GPU spawn

```python
import functools, os, torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

_MULTI_GPU = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need ≥2 GPUs"
)

def _worker(rank, world_size, fn, *args):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    device = torch.device("cuda", rank)
    torch.cuda.set_device(rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank,
                            world_size=world_size, device_id=device)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()

@_MULTI_GPU
def test_distributed_op():
    def _check(rank, world_size):
        ...
    mp.spawn(_worker, args=(2, _check), nprocs=2, join=True)
```

### SDMA tests

Also register the process group for mori:
```python
torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
```

### TP/SP mocking (module tests)

```python
@mock.patch("atomrl.lumen.modules.parallel_linear._get_tp_group", return_value=None)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_size", return_value=1)
@mock.patch("atomrl.lumen.modules.parallel_linear._pg_rank", return_value=0)
class TestLumenColumnParallelLinear: ...
```

---

## Coverage Gaps (as of 2026-03-18)

### P0 — Supported but untested

- Sliding window attention (`window_size` parameter)
- Attention bias (`core_attention_bias_type` variants)
- ALiBi slopes (csrc + Triton backends)
- CP P2P distributed ring test (only existence checks)

### P1 — Plan compliance

- End-to-end training smoke test (short loop, Lumen FP8 vs BF16)
- Hybrid FP8 (e4m3 fwd / e5m2 bwd) GEMM round-trip
- MoE auxiliary loss
- Multi-GPU SP/TP integration tests

### P2 — Hardening thin tests

- `test_fp8_activation_store.py` — needs multi-dtype, large-tensor, SNR comparison
- `test_fsdp2.py` — needs FP8 integration, multi-GPU
- `test_delay_wgrad.py` — needs wiring into `LumenRowParallelLinear`
- `test_cp_p2p.py` — needs actual distributed ring send/recv
