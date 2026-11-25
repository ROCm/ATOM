# Welcome to Lumen

**Lumen**: Light Unified Model Engine for AITER.

Lumen is an **AMD GPU model-training optimization engine built on [AITER](https://github.com/ROCm/aiter)**. It combines AITER-optimized training kernels with model-level integrations for quantized pretraining, fine-tuning, and post-training.

Lumen manages the **quantized training lifecycle** — the vertical path a low-precision tensor takes through forward, backward, optimizer, and communication — and delegates optimizer execution, data loading, and distributed orchestration to the training backend.

Lumen is **non-invasive** with:

- **One-line enablement** — `quant.enable(model)` patches an existing model for quantized training without changing model code or checkpoint format.
- **Unified `QuantConfig`** — A single configuration object governs format, scaling strategy, amax algorithm, activation/gradient quantization, and all precision-related knobs.
- **Backend-agnostic** — The same quantization semantics work across **FSDP**, **FSDP2**, and **Megatron-LM** backends, so switching parallelism strategies does not require re-engineering precision paths. RL training (GRPO, PPO, DAPO) works with both **SGLang** and **vLLM** rollout engines on ROCm.

Lumen is **fast** with:

- **FP8 formats** — FP8 E4M3, E5M2, MXFP8, and Hybrid precision with up to **2× peak FLOPS** over BF16 on AMD Instinct MI300X.
- **[AITER kernels](https://github.com/ROCm/aiter)** — High-performance fused operators for attention, GEMM, normalization, RoPE, MoE, MLP, cross-entropy, and quantization with ASM → CK → Triton fallback.
- **[MORI communication](https://github.com/ROCm/mori)** — Optional native RDMA + GPU communication with FP8 parameter all-gather (~2× bandwidth saving) and compute–communication overlap.

---

```{toctree}
:maxdepth: 2
:caption: Getting Started

quickstart/install
quickstart/quick_start
```

```{toctree}
:maxdepth: 2
:caption: Architecture

architecture
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples/llama2_sft
examples/llama31_pretrain
```

```{toctree}
:maxdepth: 2
:caption: Advanced Features

advance/fp8_training
advance/lora
advance/distributed
advance/feature_catalog
advance/patch_registry
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/quantize
api/modules
api/ops
```
