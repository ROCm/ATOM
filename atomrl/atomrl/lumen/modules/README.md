# Lumen Modules

`nn.Module` wrappers for attention, linear, and normalization layers.

## FP8 Attention (module API)

```python
from atomrl.lumen.modules import LumenAttention

# BF16 attention via AITER CK (default)
attn = LumenAttention(causal=True, backend_type="aiter_csrc")

# FP8 blockwise attention via AITER Triton
attn = LumenAttention(causal=True, backend_type="aiter_triton_fp8", quant_type="blockwise")

# MXFP8 attention via Triton (gfx950, configurable block sizes)
attn = LumenAttention(
    causal=True,
    backend_type="aiter_triton",
    quant_type="mxfp8",
    block_m_fwd=64, block_n_fwd=64,     # forward pass tile sizes
    block_m_dq_bwd=64, block_n_dq_bwd=64,  # backward dQ tile sizes
    block_m_dkv_bwd=64, block_n_dkv_bwd=64, # backward dKV tile sizes
    quant_block_size=32,
)

output = attn(q, k, v)       # q, k, v: [B, S, H, D]
```
