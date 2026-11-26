# Quick Start

This page shows how to enable Lumen's model-training optimizations on an existing model with minimal code changes.

## Unified Training Optimization

Lumen's main entry point is `LumenConfig.enable()`. A single configuration coordinates quantization, AITER kernel patches, model-level patches, and distributed-training optimizations without changing the model's public training loop.

```python
from atomrl.lumen import LumenConfig

# 1. Build or load your model as usual
model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf")

# 2. Configure the training optimization stack
lumencfg = LumenConfig(
    format="fp8_e4m3",
    scaling="delayed",
    fp8_attn="dpa",          # AITER-optimized attention
    lumen_norm=True,         # model-level norm patch
    lumen_linear=True,       # AITER linear/GEMM patch
    fused_mlp=True,
    fp8_param_gather=False,  # set True with a distributed DP group
)
lumencfg.enable(model)

# 3. Train as usual — Lumen applies the configured optimizations
output = model(input_ids)
loss = output.loss
loss.backward()
optimizer.step()
```

For distributed training, set `fp8_param_gather=True` and pass the backend's data-parallel process group as `dp_group`.

### Quantization-only Shorthand

For experiments that need only low-precision linear patching, the lower-level `quant.enable` API remains available:

```python
import atomrl.lumen.quantize as quant

quant.enable(model, format="fp8_e4m3", scaling="delayed")
```

## FP8 Attention (Module API)

For more fine-grained control, use the module API to replace specific layers:

```python
from atomrl.lumen.modules import LumenAttention, LumenDotProductAttention
```

See {doc}`/api/modules` for the full list of drop-in module replacements.

## Functional API

For custom kernels and operator-level control:

```python
from atomrl.lumen.ops.attention import fp8_flash_attention_fwd
from atomrl.lumen.ops.quantize import fp8_quantize, fp8_dequantize
```

See {doc}`/api/ops` for the stateless functional API backed by AITER kernels.

## Training Backends

Lumen integrates with two distributed training backends:

| Backend | Use case | Documentation |
|---------|----------|---------------|
| **FSDP / FSDP2** | HuggingFace ecosystem, medium scale | {doc}`/advance/distributed` |
| **Megatron-LM** | Large-scale pre-training, TP/PP/EP | {doc}`/advance/distributed` |

See {doc}`/examples/llama2_sft` and {doc}`/examples/llama31_pretrain` for end-to-end training examples.
