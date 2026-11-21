# atomrl.lumen.modules

Drop-in `nn.Module` replacements for standard PyTorch and Megatron-LM layers with TP/SP/FP8 integration.

```{note}
Most users do not need to use these modules directly — `quant.enable(model)` patches the model automatically. Use the module API when you need fine-grained control over specific layers.
```

## Module Summary

| Module | Source | Wraps Op | TP Support | FP8 Support |
|--------|--------|----------|------------|-------------|
| `LumenColumnParallelLinear` | `parallel_linear.py` | Quantized Linear | Column-parallel | All scaling modes |
| `LumenRowParallelLinear` | `parallel_linear.py` | Quantized Linear | Row-parallel | All scaling modes |
| `LumenLayerNormLinear` | `layernorm_linear.py` | Norm + Linear | Column-parallel | Fused norm+quant |
| `LumenDotProductAttention` | `attention.py` | Attention | — | BF16 / FP8 / MXFP8 |
| `LumenDotProductAttentionMLA` | `attention_mla.py` | Attention (MLA) | — | BF16 / FP8 |
| `LumenFusedMLP` | `fused_mlp.py` | Fused ungated MLP | — | FP8 activation store |
| `LumenGatedMLP` | `fused_mlp.py` | Fused gated MLP | — | FP8 activation store |
| `LumenGroupedLinear` | `grouped_linear.py` | Grouped GEMM | — | All scaling modes |
| `LumenColumnParallelGroupedLinear` | `grouped_linear.py` | Grouped GEMM | Column-parallel | All scaling modes |
| `LumenRowParallelGroupedLinear` | `grouped_linear.py` | Grouped GEMM | Row-parallel | All scaling modes |
| `LumenRMSNorm` | `normalization.py` | RMSNorm | — | Fused quant |
| `LumenLayerNorm` | `normalization.py` | LayerNorm | — | Fused quant |
| `SdmaTpComm` | `sdma_comm.py` | SDMA collectives | TP AG/RS | float32 only |
| `CommOverlapLinear` | `comm_overlap.py` | Communication overlap | — | AG/GEMM + GEMM/RS |

## Attention

`LumenDotProductAttention` wraps the attention ops with module-level state management:

```python
from atomrl.lumen.modules import LumenDotProductAttention

attn = LumenDotProductAttention(
    num_attention_heads=32,
    kv_channels=128,
    attn_mask_type=AttnMaskType.causal,
    fp8_enabled=True,
)
```

- Dispatches to CK (csrc) or Triton based on `LUMEN_ATTN_KERNEL_BACKEND`
- Supports BF16, FP8 (blockwise, dynamic, delayed, per-token), MXFP8, blockwise2d
- Context Parallelism (A2A and P2P) for long sequences

`LumenDotProductAttentionMLA` provides Multi-Latent Attention for architectures like DeepSeek-V2.

## Linear

Tensor-parallel linear layers with FP8 GEMM:

```python
from atomrl.lumen.modules import LumenColumnParallelLinear, LumenRowParallelLinear

col_linear = LumenColumnParallelLinear(hidden_size, ffn_size, fp8_enabled=True)
row_linear = LumenRowParallelLinear(ffn_size, hidden_size, fp8_enabled=True)
```

All 7 FP8 scaling modes are supported (per-tensor, per-token, blockwise, MXFP8, delayed, dynamic, static).

## Fused MLP

```python
from atomrl.lumen.modules import LumenGatedMLP, LumenFusedMLP

mlp = LumenGatedMLP(hidden_size, ffn_size, activation="swiglu", fp8_store=True)
```

- Fused path dispatches to AITER Triton single-kernel (when no bias)
- FP8 activation store reduces backward memory by ~50%
- Supported activations: `swiglu`, `geglu`, `reglu`, `gelu`, `relu`, `silu`

**CLI flags**: `--lumen-fused-mlp`, `--lumen-fp8-activation-store`

## MoE

```python
from atomrl.lumen.modules import LumenGroupedLinear, LumenColumnParallelGroupedLinear
```

Per-expert linear with FP8 quantization and TP variants for Mixture-of-Experts architectures.

**CLI flags**: `--moe-grouped-gemm`, `--moe-use-legacy-grouped-gemm`

## Communication Overlap

`CommOverlapLinear` provides AG/GEMM and GEMM/RS overlap for hiding all-gather and reduce-scatter latency behind computation. Uses `NcclCommBackend` or `SdmaCommBackend`.
