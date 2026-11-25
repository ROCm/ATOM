# Lumen Models

Training utilities and model definitions. Every model supports two independent training stacks, selected via `--backend megatron|fsdp`.

## Megatron stack (`atomrl.lumen.models.megatron`)

Uses [Megatron-LM](https://github.com/ROCm/Megatron-LM) as the training backbone. Lumen injects itself at four points:

1. **Aiter Attention** — `LumenDotProductAttention` for Megatron (the same as `DotProductAttention`) in every transformer layer via spec patching. Supports AITER backends.
2. **FP8 quantized linear** — `quant.enable(model)` patches all `ColumnParallelLinear` / `RowParallelLinear` layers with FP8 forward/backward, preserving Megatron's sequence-parallel all-gather / reduce-scatter communication.
3. **Distributed management** — FP8/BF16 contiguous param & grad buffers, distributed optimizer with shard + all-gather, FP8 param-gather (uint8 communication, 2x memory saving), and communication-computation overlap (all-gather ↔ forward, reduce-scatter ↔ backward).
4. **MORI communication** — Native RDMA + GPU communication via MORI-CCL (all-gather, reduce-scatter, all-reduce) and MORI-EP (MoE expert dispatch), with device-side RDMA for FP8 payloads.

Supports full parallelism: TP, PP, CP (All-to-All), VP, SP, distributed optimizer.

## FSDP stack (`atomrl.lumen.models.fsdp`)

Uses PyTorch FSDP + HuggingFace Transformers. No Megatron dependency. Lumen provides:

1. **FP8 quantized training** — same `quant.enable(model)` API, patches all `nn.Linear` layers.
2. **LoRA** — via HuggingFace PEFT (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
3. **FP8 state management** — `reset_fp8_state()` with FSDP-aware `.module` unwrapping.

## Usage

```python
# Megatron backend — full Megatron parallelism
from atomrl.lumen.models.llama2.megatron import (
    lumen_gpt_builder, forward_step, apply_fp8_training,
    apply_lora, add_finetune_args, train_valid_test_datasets_provider,
)

# FSDP backend — lightweight, no Megatron dependency
from atomrl.lumen.models.llama2.fsdp import FSDPTrainer, get_args
args = get_args()
trainer = FSDPTrainer(args)
trainer.train()
```
