# Enable Blockwise Linear Training with `LumenConfig.enable`

This file provides two minimal examples showing how to enable blockwise FP8 linear training through `LumenConfig(...).enable(model)`.

Core settings:
- `format="fp8_e4m3"`
- `scaling="delayed"` (enables FP8 quantization path; `"none"` disables quantized linear)
- `block_size=128` (blockwise granularity)
- `quantize_activation=True` and `fp8_wgrad=True`

> API aligned with `atomrl/atomrl/lumen/config.py`: `manager, model = cfg.enable(model, dp_group=..., backend="auto")`

## 1) Megatron example

```python
from __future__ import annotations

import torch
import torch.distributed as dist
from atomrl.lumen.config import LumenConfig


def enable_blockwise_linear_for_megatron(model):
    # In Megatron, a data-parallel group is usually available.
    # WORLD is used here for a minimal example.
    dp_group = dist.group.WORLD if dist.is_initialized() else None

    cfg = LumenConfig(
        # Tier-1: blockwise linear FP8
        format="fp8_e4m3",
        scaling="delayed",
        block_size=128,
        quantize_activation=True,
        fp8_wgrad=True,
        # Common training features (optional)
        fp8_param_manager=True,
        fp8_weight_cache=True,
        fused_mlp=True,
        fused_rope=True,
        hf_attn_patch=True,
    )

    manager, model = cfg.enable(model, dp_group=dp_group, backend="auto")
    return manager, model
```

## 2) FSDP example

```python
from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from transformers import AutoModelForCausalLM
from atomrl.lumen.config import LumenConfig


def build_fsdp_model_with_blockwise_linear(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # dp_group can also be passed in FSDP mode.
    # If reduce_amax is false, this group is ignored.
    dp_group = dist.group.WORLD if dist.is_initialized() else None

    cfg = LumenConfig(
        format="fp8_e4m3",
        scaling="delayed",
        block_size=128,
        quantize_activation=True,
        fp8_wgrad=True,
        fp8_weight_cache=True,
        hf_attn_patch=True,
        lumen_norm=True,
    )

    manager, model = cfg.enable(model, dp_group=dp_group, backend="auto")

    # Typical order: apply Lumen first, then shard with FSDP.
    if dist.is_initialized():
        fully_shard(model)

    return manager, model
```

## Notes

- If you only want BF16 kernel patching without FP8 quantization, set `scaling="none"` and use options like `lumen_linear=True` or `lumen_norm=True`.
- If you want per-step amax reduction across DP ranks, set `reduce_amax=True` and pass the correct `dp_group`.
