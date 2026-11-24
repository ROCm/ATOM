# Welcome to AtomRL

**AtomRL** is an AMD native **Post-Training Framework for LLMs**, powered by [Lumen](https://github.com/ROCm/ATOM/tree/main/atomrl/lumen) (quantized training) and [ATOM](https://github.com/ROCm/ATOM) (optimized inference). AtomRL supports both **RL post-training** (GRPO, DAPO, PPO) and **Speculative Decoding Draft Distillation (SDDD)** on AMD GPUs.

## News

- **[2026/11]** AtomRL now supports **one-step off-policy and fully async training** on the Ray controller path — the rollout keeps generating on GPUs of its own while the trainer trains; 2.7× faster on Qwen3-30B-A3B DAPO (example 6) at comparable accuracy. See [Asynchronous Training](advance/async_training.md)
- **[2026/05]** AtomRL now supports **Speculative Decoding Draft Distillation (SDDD)** — Eagle3 draft distillation for Kimi K2.5 and Qwen3-8B on MI350 with [Mooncake](https://github.com/kvcache-ai/Mooncake)/MORI RDMA transfer
- **[2026/04]** AtomRL now supports **fully async training** — inspired by [VERL](https://github.com/verl-project/verl), decouples rollout and training for up to 2.7x throughput improvement
- **[2026/03]** AtomRL now supports **MoE [R3](https://arxiv.org/abs/2510.11370) router alignment** — Rollout Routing Replay for stable MoE RL training
- **[2026/03]** AtomRL now supports the **[FP8-RL](https://arxiv.org/abs/2601.18150) quantization stack** — FP8 rollout + training with TIS/MIS importance-sampling correction

## Features

- **AMD-Native**: Built for ROCm with [AITER](https://github.com/ROCm/aiter) kernels (ASM / CK / Triton) and [MORI](https://github.com/ROCm/mori) communication (RDMA + GPU collective ops, MoE expert dispatch)
- **Quantization End-to-End**: Quantized rollout (FP8, MXFP8, MXFP4) and quantized training with importance-sampling rollout correction (TIS/MIS) — up to 44% throughput gain over BF16
- **MoE-Stable RL**: Rollout Routing Replay (R3) aligns train/inference routers to prevent MoE training collapse
- **Speculative Decoding Draft Distillation (SDDD)**: Train Eagle3/DFlash draft models via teacher hidden-state distillation with [Mooncake](https://github.com/kvcache-ai/Mooncake)/MORI RDMA transfer
- **Flexible Backends**: FSDP2 or Megatron-Core training, ATOM/vLLM inference with TP/EP/DP parallelism

## Supported Models

### RL Models

| Model Family | Architecture | Dense/MoE | FP8 Rollout | FP8 Training | R3 |
|---|---|---|---|---|---|
| Llama 3.x | `LlamaForCausalLM` | Dense | Yes | Yes | N/A |
| Qwen3 | `Qwen3ForCausalLM` | Dense | Yes | Yes | N/A |
| Qwen3-MoE | `Qwen3MoeForCausalLM` | MoE | Yes | Yes | Yes |
| DeepSeek V2/V3 | `DeepseekV3ForCausalLM` | MoE | Yes | Yes | Yes |
| Mixtral | `MixtralForCausalLM` | MoE | Yes | Yes | Yes |
| GLM-4-MoE | `Glm4MoeForCausalLM` | MoE | Yes | Yes | Yes |

### SDDD Models

| Teacher Model | Draft Architecture | Inference Backend | Transfer |
|---|---|---|---|
| Kimi K2.5 (1T MoE) | Eagle3 | ATOM / vLLM | Mooncake / MORI |
| Qwen3-8B | Eagle3 | vLLM | MORI |

```{toctree}
---
maxdepth: 2
caption: Getting Started
---
quickstart/install
quickstart/quick_start
```

```{toctree}
---
maxdepth: 2
caption: Architecture
---
architecture
```

```{toctree}
---
maxdepth: 2
caption: Examples
---
examples/grpo_training
examples/dapo_training
examples/ppo_training
examples/fp8_training
examples/sddd_training
```

```{toctree}
---
maxdepth: 2
caption: Advanced Features
---
advance/fp8_quantization
advance/moe_r3
advance/sddd
advance/distributed
advance/async_training
advance/algorithms
```

```{toctree}
---
maxdepth: 2
caption: API Reference
---
api/config
api/protocol
api/algorithms
api/quantization
api/moe
```
