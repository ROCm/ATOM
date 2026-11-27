# Lumen Ops

Stateless functional API for attention, quantization, and normalization.

## Functional API

```python
from atomrl.lumen.ops.attention import attention, attention_fp8_quant

# Standard attention — auto-selects aiter_csrc if available, else aiter_triton
output = attention(q, k, v, causal=True, backend_type="auto")

# FP8 blockwise attention (Triton only)
output = attention_fp8_quant(q, k, v, causal=True, quant_type="blockwise")

# MXFP8 attention (Triton only)
output = attention_fp8_quant(q, k, v, causal=True, quant_type="mxfp8",
                             quant_block_size=32)

# Context Parallelism (All-to-All)
output = attention(q, k, v, causal=True,
                   cp_param_bundle={"cp_group": cp_group, "cp_comm_type": "a2a"})
```
