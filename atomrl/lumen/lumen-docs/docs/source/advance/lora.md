# LoRA Support

Lumen supports **Low-Rank Adaptation (LoRA)** for parameter-efficient fine-tuning, combined with FP8 quantized training on both Megatron-LM and FSDP backends.

## Overview

LoRA freezes the pre-trained weight matrix and injects trainable low-rank decomposition matrices, reducing the number of trainable parameters while retaining model quality. Combined with Lumen's FP8 training:

- **Full model weights** are stored in FP8 (frozen), saving ~50% memory
- **LoRA adapters** (low-rank A and B matrices) are trained in BF16 for numerical stability
- **FP8 forward/backward** runs through the frozen base model, while LoRA updates flow through the BF16 path

## Enabling LoRA

### With quant.enable

```python
import atomrl.lumen.quantize as quant

quant.enable(model, format="fp8_e4m3", scaling="delayed")

# Then apply LoRA through your preferred method (PEFT, Megatron, manual)
```

### Megatron Backend

```bash
python train.py \
    --quant-format fp8_e4m3 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules qkv_proj,o_proj,gate_proj,up_proj,down_proj
```

### FSDP Backend

```bash
python train.py \
    --quant-format fp8_e4m3 \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32
```

## Configuration

| Parameter | Description | Typical values |
|-----------|-------------|----------------|
| `lora-r` | Rank of the low-rank matrices | 8, 16, 32, 64 |
| `lora-alpha` | Scaling factor (effective LR multiplier = alpha/r) | 16, 32, 64 |
| `lora-target-modules` | Which layers to apply LoRA to | `qkv_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `lora-dropout` | Dropout on LoRA paths | 0.0–0.1 |

## Memory Impact

LoRA dramatically reduces the trainable parameter count and optimizer state size. Combined with FP8, this enables fine-tuning very large models on limited hardware:

| Model | Full FP8 SFT (8× MI300X) | FP8 + LoRA r=16 (8× MI300X) |
|-------|---------------------------|-------------------------------|
| LLaMA-2 7B | Fits comfortably | Single GPU viable |
| LLaMA-2 70B | Requires 8 GPUs | 2–4 GPUs viable |

## Checkpoint Handling

LoRA adapters are saved separately from the base model weights. Lumen preserves the original checkpoint format, so:

1. Base model weights remain in their original format (no FP8 checkpoints needed)
2. LoRA adapters are saved as standard PyTorch state dicts
3. Merging adapters back into the base model follows standard LoRA practice
