# LLaMA-2 SFT

This example demonstrates supervised fine-tuning (SFT) of LLaMA-2 models (7B–70B) with Lumen's FP8 quantized training on AMD Instinct MI300X.

## Overview

| Setting | Value |
|---------|-------|
| Model | LLaMA-2 7B / 13B / 70B |
| Task | Supervised fine-tuning (SFT) |
| Precision | FP8 E4M3 (delayed scaling) |
| Backends | Megatron-LM, FSDP |
| Hardware | AMD Instinct MI300X (1–8 GPUs) |
| Features | FP8 attention, packed sequences, LoRA, early stopping |

## Quick Launch

```bash
cd examples/llama2/

# Megatron backend — 8 GPUs
bash scripts/run_megatron.sh

# FSDP backend — 8 GPUs
bash scripts/run_fsdp.sh
```

## Enabling FP8

The training scripts call `quant.enable` before entering the training loop:

```python
import atomrl.lumen.quantize as quant

quant.enable(model, format="fp8_e4m3", scaling="delayed")
```

This patches all eligible linear and attention layers for FP8 execution. No model code changes are needed.

## LoRA Support

To train with LoRA instead of full fine-tuning:

```bash
bash scripts/run_megatron.sh --lora-r 16 --lora-alpha 32
```

See {doc}`/advance/lora` for details on LoRA configuration.

## Configuration Reference

### Megatron Backend

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-layers` | model-dependent | Number of transformer layers |
| `--hidden-size` | model-dependent | Hidden dimension |
| `--tensor-model-parallel-size` | 1 | Tensor parallelism degree |
| `--pipeline-model-parallel-size` | 1 | Pipeline parallelism degree |
| `--quant-format` | `fp8_e4m3` | Quantization format |
| `--scaling` | `delayed` | Scaling strategy |
| `--lora-r` | 0 (disabled) | LoRA rank |

### FSDP Backend

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fsdp-sharding-strategy` | `FULL_SHARD` | FSDP sharding mode |
| `--quant-format` | `fp8_e4m3` | Quantization format |
| `--gradient-checkpointing` | `true` | Activation checkpointing |

## Source

Full example code: [`examples/llama2/`](https://github.com/ROCm/ATOM/tree/main/atomrl/lumen/examples/llama2)
