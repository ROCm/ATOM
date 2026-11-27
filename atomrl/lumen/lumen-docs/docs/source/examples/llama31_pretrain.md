# LLaMA 3.1 Pre-training

This example demonstrates pre-training LLaMA 3.1 8B with Lumen's FP8 hybrid training and MXFP8 attention on AMD Instinct MI300X, aligned with MLPerf Training configurations.

## Overview

| Setting | Value |
|---------|-------|
| Model | LLaMA 3.1 8B |
| Task | Pre-training (causal language modeling) |
| Precision | FP8 hybrid (E4M3 forward, E5M2 backward) + MXFP8 attention |
| Backend | Megatron-LM |
| Hardware | AMD Instinct MI300X (multi-node) |
| Reference | MLPerf Training benchmark alignment |

## Quick Launch

```bash
cd examples/llama31/

# Megatron backend — multi-GPU
bash scripts/run_pretrain.sh
```

## FP8 Hybrid Training

LLaMA 3.1 pre-training uses **hybrid** FP8 precision — E4M3 for forward activations and E5M2 for backward gradients:

```python
from atomrl.lumen.quantize import QuantConfig, QuantFormat, ScalingType

config = QuantConfig(
    format=QuantFormat.HYBRID,          # E4M3 forward + E5M2 backward
    scaling=ScalingType.DELAYED,
    history_len=16,
    quantize_activation=True,
    quantize_grad="fp8",
)
quant.enable(model, config=config)
```

## MXFP8 Attention

For attention layers, MXFP8 (Microscaling FP8) provides finer-grained per-block scaling:

```python
config = QuantConfig(
    format=QuantFormat.MXFP8,
    scaling=ScalingType.BLOCKWISE,
)
```

## Distributed Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Tensor parallelism | 8 | One model shard per GPU within a node |
| Pipeline parallelism | configurable | Cross-node pipeline stages |
| Data parallelism | configurable | Across pipeline replicas |
| FP8 all-gather | enabled | ~2× communication bandwidth saving |
| Compute-comm overlap | enabled | AG ↔ forward, RS ↔ backward |

## Performance

FP8 training on MI300X achieves up to **2× peak FLOPS** over BF16 for large GEMM operations. End-to-end training speedup depends on the compute-to-communication ratio:

| Scenario | Estimated speedup vs BF16 |
|----------|---------------------------|
| Compute-bound (large batch, few GPUs) | 1.5–1.7× |
| Communication-heavy (many GPUs, aggressive sharding) | 1.3–1.5× |

## Source

Full example code: [`examples/llama31/`](https://github.com/ROCm/ATOM/tree/main/atomrl/lumen/examples/llama31)
