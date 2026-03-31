# ATOM as vLLM Plugin Backend: Ecosystem Co-Evolution for AMD LLM Inference

Authors:  Zejun Chen, Hattie Wu, Lingpeng Jin, Carlus Huang, Chuan Li, Peng Sun, Emad Barsoum

## 1. Introduction

LLM inference faces a persistent dilemma: balancing hardware-specific optimization and framework compatibility. To unlock the full performance potential of AMD Instinct GPUs, in-depth hardware-aware kernel engineering is a must; yet production-grade LLM serving is overwhelmingly built on [vLLM](https://github.com/vllm-project/vllm) — the de facto industry standard—thanks to its robust scheduling mechanisms, sophisticated memory management, and universal API compatibility.


ATOM, a high-performance inference engine purpose-built for AMD Instinct GPUs, resolves this conflict with its dual-mode architecture. It can run as a standalone inference server or integrate seamlessly into vLLM as a plugin backend, delivering AMD-native model and kernel optimizations without any modifications to vLLM’s core codebase.

Not a fork, nor a replacement for vLLM, the ATOM vLLM plugin acts as a collaborative bridge connecting AMD’s hardware innovation to the open-source vLLM ecosystem, rooted in the spirit of co-evolution rather than competition. As the universal standard for LLM serving, vLLM forms the backbone of inference infrastructure for startups and hyperscalers alike, who rely on its mature APIs, continuous batching capabilities, and full suite of operational tools. Switching serving frameworks entails steep learning curves, migration risks, and heavy operational overhead—users should never have to choose between a trusted production framework and full hardware performance.

Integrating ATOM as a vLLM plugin creates a win-win for all stakeholders, with five core advantages:


- **Zero learning curve:** Full compatibility with existing vLLM commands, APIs, and end-to-end workflows. ATOM runs transparently in the background, requiring no new tools or complex configurations—only delivering enhanced kernel performance while preserving a consistent user experience.

- **Instant access to AMD innovation:** Leverage cutting-edge AMD hardware features (e.g., FP4 on MI355X, rack-scale inference on MI400) and top-tier kernel optimizations (e.g., AITER fused attention, custom AllReduce) out of the box, without waiting for upstream integration into the main vLLM codebase. This drastically shortens the time-to-value for new AMD GPU chips.

- **Agile innovation sandbox:** A fast validation layer for new technical ideas, hardware enablement, and kernel library testing (e.g., AITER). The plugin aligns flexibly with AMD’s product roadmap—including new GPU releases, FP8/FP4 precision support, and next-gen attention mechanisms—unconstrained by vLLM’s upstream release cycles. 

- **vLLM as production-grade foundation for ROCm:** As the community-standard serving framework, vLLM provides the enterprise-grade stability, broad model coverage, and production-critical features needed to deploy ROCm-based infrastructure at scale.

- **Mature optimizations upstreamed for all:** ATOM serves as a temporary proving ground for new optimizations; once stabilized, kernels, optimization strategies, and new features are upstreamed to vLLM’s native ROCm backend, benefiting the entire ROCm user community and strengthening the open-source LLM ecosystem.

This plugin enables a closed, iterative innovation cycle: **new hardware/ideas/libraries** → **rapid validation via ATOM** → **upstream integration into vLLM core when mature** → **universal access for all ROCm users**. This approach accelerates the delivery of AMD’s hardware advantages to end users through ATOM, while long-term technical improvements flow back to the broader open-source community via vLLM. The remainder of this post dives into the design, architecture, and technical implementation of the ATOM vLLM plugin system.


## 2. Architecture Overview

Framework development teams (vLLM) focus on cross-backend scheduling, batching logic, and API design; hardware teams (AMD) drive kernel-level optimizations but do not need to rewrite an entire serving framework from scratch. vLLM’s native plugin registration mechanism bridges this divide—and ATOM adheres to the same established integration pattern used by other accelerator vendors, enabling a clean three-layer separation of concerns that clarifies responsibilities across the stack:

| Layer | Responsibility |
|-------|---------------|
| **vLLM** | Request scheduling, KV cache management, continuous batching, OpenAI-compatible API |
| **ATOM Plugin** | Platform registration, optimized model implementation, attention backends routing, kernel-level optimization tuning |
| **AITER** | Low-level GPU kernels — fused MoE, flash attention, quantized GEMM, RoPE fusion |

For a comprehensive technical reference—covering configuration translation, attention integration internals, installation steps, and environment variable settings — refer to the [ATOM vLLM Plugin Backend Guide](https://rocm.github.io/ATOM/docs/vllm_plugin_backend_guide.html).

The ATOM vLLM plugin system is composed of four interconnected subsystems that form a cohesive architecture stack:

![ATOM vLLM Plugin Architecture Stack](atom_vllm_architecture_stack.svg)

## 3. Execution Details

The following diagram illustrates the end-to-end execution flow—from `vllm serve` startup through OOT plugin discovery, model construction, and a single serving step—showing how vLLM and ATOM interact at each stage:

![vLLM execution flow with ATOM OOT plugin](atom_vllm_oot_injection.svg)

The execution flow is divided into four core phases: **plugin discovery** (Steps 1–5), **attention backend selection** (Steps 6–7), **model construction** (Steps 8–9), and **inference serving** (Steps 10–11). The following subsections break down the key technical implementation details for each phase.

### 3.1. Entry Point Registration (Steps 1–5)

The ATOM plugin is activated through Python's standard `entry_points` mechanism, with the following registration configuration:

```toml
[project.entry-points."vllm.platform_plugins"]
atom = "atom.plugin.vllm.register:register_platform"

[project.entry-points."vllm.general_plugins"]
atom_model_registry = "atom.plugin.vllm.register:register_model"
```

- `register_platform()` returns the `ATOMPlatform` class (step 3); 
- `register_model()` overrides vLLM's model registry with ATOM's optimized wrappers (step 5). Both hooks are no-ops when `ATOM_DISABLE_VLLM_PLUGIN=1` is set.

### 3.2. Attention Backend Selection (Steps 6–7)

`ATOMPlatform` extends `RocmPlatform` and overrides `get_attn_backend_cls()` method to route all attention computations to AITER-backed optimized implementations:

```python
class ATOMPlatform(RocmPlatform):
    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads):
        if attn_selector_config.use_mla:
            return "atom.model_ops.attentions.aiter_mla.AiterMLABackend"
        return "atom.model_ops.attentions.aiter_attention.AiterBackend"
```

Two specialized attention backends are supported:

- **AiterBackend (MHA)** — Translates vLLM's `CommonAttentionMetadata` into a three-phase format (decode / extend / prefill) with chunk-based context processing, optimized for standard multi-head attention workloads.
- **AiterMLABackend (MLA)** — Purpose-built for DeepSeek V2/V3-style latent attention. It features fused QK-RoPE-Cache-Update operations, batched FP4/FP8 GEMM for V-projection, persistent metadata buffers for CUDAGraph acceleration, and distributed context parallelism. When the plugin is active, ATOM patches vLLM’s `MLAAttention.forward_impl` method at import time, delegating all MLA computations to its optimized implementation.

### 3.3. Model Construction and Weight Loading (Steps 8–9)

The `ATOMModelBase` wrapper fully implements vLLM's model interface while delegating all core computations to ATOM's native model implementations. It handles three critical tasks:

- **Config translation** —  Seamlessly converts vLLM’s `VllmConfig` to ATOM’s native `Config`, preserving CUDAGraph settings while applying ATOM’s optimized compile policies for AMD GPUs.
- **Model construction** — Instantiates the ATOM model class and initializes AITER’s distributed backend for multi-GPU/rack-scale inference.
- **Weight loading** — Uses ATOM’s `load_model_in_plugin_mode()` method to load models in ATOM-specific formats and support AMD-optimized quantization schemes, ensuring optimal weight utilization on Instinct GPUs.

### 3.4. Inference Serving (Steps 10–11)

Once the model is constructed and weights are loaded, vLLM drives the serving loop. During each forward pass, vLLM’s scheduler assembles a batch of requests and invokes `ATOMModelBase.forward()`, which delegates to ATOM’s native model implementation. The attention layers execute through ATOM’s AITER-backed kernels (AiterBackend or AiterMLABackend), while vLLM retains full control over request scheduling, KV cache allocation, and output sampling. This clear separation ensures that ATOM handles the performance-critical compute path while vLLM manages the production-critical serving infrastructure.

## 4. Supported Models

The ATOM vLLM plugin backend supports both LLMs and VLMs through a unified serving pipeline, covering text-only LLM models and Qwen3.5-based conditional-generation VLM models (including both dense and Mixture-of-Experts (MoE) variants). The table below lists the supported model architectures, types, representative examples, and their corresponding ATOM model classes:

| Architecture | Type | Representative Models | ATOM Model Class |
|-------------|------|----------------------|-----------------|
| Qwen3MoeForCausalLM | MoE | Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 | `atom.models.qwen3_moe` |
| DeepseekV3ForCausalLM | MoE (MLA) | deepseek-ai/DeepSeek-R1-0528 (FP8), amd/DeepSeek-R1-0528-MXFP4, amd/Kimi-K2-Thinking-MXFP4 | `atom.models.deepseek_v2` |
| GptOssForCausalLM | MoE | openai/gpt-oss-120b | `atom.models.gpt_oss` |
| Glm4MoeForCausalLM | MoE (MLA) | zai-org/GLM-4.7-FP8 | `atom.models.glm4_moe` |
| Qwen3NextForCausalLM | Hybrid MoE | Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 | `atom.models.qwen3_next` |
| Qwen3_5ForConditionalGeneration | Dense (Text/VLM) | Qwen/Qwen3.5-35B-A3B-FP8 | `atom.models.qwen3_5` |
| Qwen3_5MoeForConditionalGeneration | MoE (Text/VLM) | Qwen/Qwen3.5-397B-A17B-FP8 | `atom.models.qwen3_5` |

> **Note:** Kimi-K2 (`amd/Kimi-K2-Thinking-MXFP4`) shares the DeepSeek V3-style MLA+MoE architecture and is served through the same `DeepseekV3ForCausalLM` pathway with `--trust-remote-code`.

For step-by-step deployment guides—including Docker environment setup, server launch commands, performance benchmarking, and accuracy validation—refer to the [ATOM vLLM Recipes](https://github.com/ROCm/ATOM/tree/main/recipes/atom_vllm).

### Quick Start

Pull the pre-built Docker image and launch a vLLM server with ATOM in just two commands:

```bash
docker pull rocm/atom-dev:vllm-latest

vllm serve ${model} \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --gpu_memory_utilization 0.9
```

ATOM activates automatically when installed alongside vLLM—no additional configuration is needed. For the full set of server options, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

> **Version compatibility:** The ATOM vLLM plugin is tested against vLLM v0.17.x. Pre-built Docker images for specific vLLM versions are available on [Docker Hub](https://hub.docker.com/r/rocm/atom-dev/tags?name=vllm).

## 5. Performance Characteristics

To enable transparent performance and quality tracking in production-like environments, ATOM provides a live benchmark dashboard (**Dashboard URL:** [ATOM vLLM Benchmark Dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/#backend=ATOM-vLLM)).

The dashboard serves as a single pane of glass for monitoring and validating the plugin’s performance, with core features including:

- **Throughput vs Latency:** Compare output throughput and end-to-end latency metrics under different load conditions to evaluate efficiency-quality tradeoffs.
- **Performance Trends:** Track real-time performance changes over time, enabling quick identification of regressions or improvements after kernel updates, model changes, or runtime upgrades.
- **Accuracy Monitoring:** Review benchmarked model quality metrics alongside performance data, ensuring optimization decisions balance inference speed and model correctness.
- **Data Export:** Export benchmark artifacts (via Data & Trace / Download JSON) for offline analysis, experiment reproducibility, and CI/CD report integration.

In practice, the dashboard is an essential tool for release validation: after every ATOM or vLLM plugin update, teams can quickly verify that throughput, latency, and model accuracy remain within expected production ranges.


## 6. Conclusion

The ATOM vLLM plugin proves that hardware-specific optimization and framework compatibility are not mutually exclusive. By leveraging vLLM’s out-of-the-box plugin mechanism, ATOM delivers AMD-native kernel optimizations—including fused attention, quantized GEMM, and optimized MoE routing—while preserving the full feature set of vLLM that production LLM deployments rely on.

Beyond immediate performance gains, the plugin’s architecture serves as a critical proving ground for AMD’s hardware and software innovations: optimizations validated in ATOM’s plugin mode are gradually upstreamed to vLLM’s native ROCm backend, benefiting the entire ROCm and open-source LLM community. For end users, this means immediate access to the latest AMD hardware capabilities without waiting for slow upstream integration cycles—creating a virtuous cycle of co-evolution between AMD’s hardware innovation and the vLLM serving ecosystem.

## References

- [ATOM Documentation](https://rocm.github.io/ATOM/docs/)
- [ATOM vLLM Plugin Backend Guide](https://rocm.github.io/ATOM/docs/vllm_plugin_backend_guide.html)
- [RFC: Enable ATOM as vLLM out-of-tree Platform](https://github.com/ROCm/ATOM/issues/201)
- [ATOM Repository](https://github.com/ROCm/ATOM)
- [AITER — AMD Inference Tensor Engine for ROCm](https://github.com/ROCm/aiter)
- [ATOM vLLM Recipes](https://github.com/ROCm/ATOM/tree/main/recipes/atom_vllm)
- [Docker Hub — ATOM + vLLM Images](https://hub.docker.com/r/rocm/atom-dev/tags?name=vllm)
