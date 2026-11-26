# Distributed Training

Lumen integrates with two distributed training backends: **FSDP** (PyTorch native) and **Megatron-LM** (NVIDIA). The same `quant.enable(model)` call works with both — Lumen detects the active backend and applies the appropriate FP8 parameter lifecycle.

## Backend Comparison

| Feature | FSDP / FSDP2 | Megatron-LM |
|---------|--------------|-------------|
| **Ecosystem** | HuggingFace Transformers + Accelerate | NVIDIA NeMo / standalone |
| **Parallelism** | Data parallel + full shard | TP + PP + DP + EP |
| **Best for** | Quick prototyping, medium scale, community models | Large-scale pre-training, MoE, production |
| **Lumen FP8 support** | Full (weights, activations, all-gather) | Full (weights, activations, all-gather) |
| **LoRA** | Supported | Supported |

## FSDP Backend

### How Lumen Integrates

With FSDP (`FULL_SHARD`), each GPU holds a shard of the model parameters. During forward/backward, parameters are gathered via all-gather. Lumen optimizes this path:

1. **FP8 parameter storage** — Shards are stored as FP8, reducing per-GPU memory
2. **FP8 all-gather** — Parameters are gathered as uint8, cutting communication volume by ~50%
3. **Overlap** — All-gather is overlapped with forward computation; reduce-scatter is overlapped with backward

### Usage

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import atomrl.lumen.quantize as quant

model = build_model()
quant.enable(model, format="fp8_e4m3", scaling="delayed")
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

### FSDP2

Lumen also supports PyTorch's FSDP2 API, which provides finer-grained control over sharding:

```python
from torch.distributed._composable.fsdp import fully_shard

quant.enable(model, format="fp8_e4m3", scaling="delayed")
fully_shard(model)
```

## Megatron-LM Backend

### How Lumen Integrates

Lumen provides a shared Megatron stack (`lumen/models/megatron.py`) that:

1. **Patches the model spec** — Replaces standard Megatron layers with Lumen's FP8 equivalents via spec patching (no model code changes)
2. **Manages FP8 buffers** — Contiguous FP8 parameter and gradient buffers aligned with Megatron's distributed optimizer
3. **Handles communication** — FP8 all-gather within TP groups, overlap with compute

### Parallelism Strategies

| Strategy | Flag | Description |
|----------|------|-------------|
| Tensor Parallelism (TP) | `--tensor-model-parallel-size` | Splits layers across GPUs within a node |
| Pipeline Parallelism (PP) | `--pipeline-model-parallel-size` | Splits layers across nodes in stages |
| Data Parallelism (DP) | `--data-parallel-size` | Replicates model across groups |
| Expert Parallelism (EP) | `--expert-model-parallel-size` | Distributes MoE experts via MORI-EP |

### Usage

```bash
python train.py \
    --tensor-model-parallel-size 8 \
    --quant-format fp8_e4m3 \
    --scaling delayed \
    --fp8-all-gather
```

## FP8 Communication

### All-Gather

In standard BF16 training, all-gather communicates 2 bytes per parameter. With FP8:

- Parameters are communicated as **1 byte (uint8)** + a small scale tensor
- This achieves approximately **48% bandwidth saving**
- Scales are broadcast separately (negligible overhead for large models)

### Compute-Communication Overlap

Lumen supports overlapping communication with computation:

| Overlap | Phase | Effect |
|---------|-------|--------|
| AG ↔ Forward | All-gather of next layer overlaps with forward of current layer | Hides gather latency |
| RS ↔ Backward | Reduce-scatter of current layer's gradients overlaps with backward of next layer | Hides scatter latency |

Enable with:

```python
quant.enable(model, format="fp8_e4m3", scaling="delayed")
# Overlap is enabled by default when using Lumen's distributed modules
```

## Hardware Requirements

| Configuration | Minimum hardware |
|---------------|-----------------|
| Single-GPU experimentation | 1× MI300X (192 GB HBM3) |
| 7B–13B full fine-tuning | 1–2× MI300X |
| 70B full fine-tuning | 8× MI300X |
| 70B+ pre-training | Multi-node MI300X cluster |
| MoE models | Multi-node with high-bandwidth interconnect |
