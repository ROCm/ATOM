# Speculative Decoding Draft Distillation (SDDD)

SDDD trains lightweight draft models for speculative decoding by distilling teacher hidden states. The draft model learns to approximate the teacher's internal representations, enabling speculative decoding that accelerates inference without sacrificing output quality.

## Architecture

```
Training GPUs                          Inference GPUs
FSDP2 Eagle3/DFlash draft model  <---  Teacher model (ATOM / vLLM)
  BF16 training                          Quantized inference (FP8/MXFP8/MXFP4)
       ^                                       |
  Mooncake/MORI RDMA  <---  hidden_states transfer
```

The pipeline splits available GPUs between teacher inference and draft model training:

- **Inference GPUs**: Run the teacher model with tensor parallelism, processing dataset samples and extracting hidden states at each transformer layer.
- **Training GPUs**: Train the draft model via FSDP2, receiving teacher hidden states through Mooncake or MORI RDMA transfers.

## Draft Architectures

### Eagle3

Eagle3 uses iterative refinement to predict the teacher's next-token hidden states. It takes the teacher's hidden states from the previous position and refines them through a single transformer layer to predict the next position's hidden state.

### DFlash

DFlash uses block-causal masking to train on multiple positions simultaneously. It enables efficient per-position train/eval metrics and supports `eval_from_cache` for faster evaluation during training.

## Hidden-State Transfer

AtomRL supports two transfer backends:

| Backend | Protocol | Use Case |
|---------|----------|----------|
| **Mooncake** | Async RDMA / TCP | ATOM inference with batched scheduling and KV cache management |
| **MORI-IO** | GPU Direct P2P RDMA | Direct ATOM inference with minimal latency |

## Two-Phase Training

Following the [lightseekorg recipe](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3), SDDD uses a two-phase training strategy:

| Phase | Steps | Dataset | Description |
|-------|-------|---------|-------------|
| Phase 1 | 0 - 20K | Large general corpus | Foundation training on broad data distribution |
| Phase 2 | 20K - 40K | Mixed domain samples | Domain diversity (VL, CN, tool-call, agent, writing) |

## Supported Models

| Teacher Model | Draft Architecture | Inference Backend | Transfer |
|---|---|---|---|
| Kimi K2.5 (1T MoE) | Eagle3 | ATOM / vLLM | Mooncake / MORI |
| Qwen3-8B | Eagle3 | vLLM | MORI |

## Example Configurations

See the runnable examples:

- `examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/` — Kimi K2.5 teacher on ATOM (online MXFP4), MI350

For a step-by-step walkthrough, see {doc}`/examples/sddd_training`.
