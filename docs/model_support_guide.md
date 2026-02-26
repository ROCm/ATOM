# ATOM Model Support Guide

ATOM (AiTer Optimized Model) is AMD's lightweight LLM inference engine built on AITER kernels for ROCm/HIP GPUs. This guide covers the supported model architectures, weight loading, and how to add new models.

## Quick Reference

The model registry lives in `atom/model_engine/model_runner.py` as `support_model_arch_dict`:

```python
support_model_arch_dict = {
    "Qwen3ForCausalLM": "atom.models.qwen3.Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM": "atom.models.qwen3_moe.Qwen3MoeForCausalLM",
    "LlamaForCausalLM": "atom.models.llama.LlamaForCausalLM",
    "MixtralForCausalLM": "atom.models.mixtral.MixtralForCausalLM",
    "DeepseekV3ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "DeepseekV32ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "GptOssForCausalLM": "atom.models.gpt_oss.GptOssForCausalLM",
    "Glm4MoeForCausalLM": "atom.models.glm4_moe.Glm4MoeForCausalLM",
}
```

ATOM resolves the HuggingFace `architectures` field from a model's `config.json` against this dictionary. If the architecture string matches a key, ATOM imports and instantiates the corresponding class.

---

## 1. Supported Model Architectures

| HF Architecture | ATOM Module | ATOM Class | MoE | MLA | Key Features |
|---|---|---|---|---|---|
| `Qwen3ForCausalLM` | `atom.models.qwen3` | `Qwen3ForCausalLM` | No | No | GQA, QK norm, RoPE |
| `Qwen3MoeForCausalLM` | `atom.models.qwen3_moe` | `Qwen3MoeForCausalLM` | Yes | No | GQA, QK norm, FusedMoE, sparse+dense layer mixing, QK norm+RoPE+cache+quant fusion |
| `LlamaForCausalLM` | `atom.models.llama` | `LlamaForCausalLM` | No | No | GQA, RoPE, fused RMSNorm+quant, fused SiLU+mul+quant |
| `MixtralForCausalLM` | `atom.models.mixtral` | `MixtralForCausalLM` | Yes | No | GQA, RoPE, FusedMoE with TP sharding |
| `DeepseekV3ForCausalLM` | `atom.models.deepseek_v2` | `DeepseekV2ForCausalLM` | Yes | Yes | MLA attention, LoRA-compressed QKV, FusedMoE with shared experts, FP4/FP8 fused kernels |
| `DeepseekV32ForCausalLM` | `atom.models.deepseek_v2` | `DeepseekV2ForCausalLM` | Yes | Yes | Same as above with V3.2 index-based top-k routing |
| `GptOssForCausalLM` | `atom.models.gpt_oss` | `GptOssForCausalLM` | Yes | No | GQA, RoPE, sliding window attention (every other layer), attention sinks, bias in QKV and MoE |
| `Glm4MoeForCausalLM` | `atom.models.glm4_moe` | `Glm4MoeForCausalLM` | Yes | No | GQA, partial RoPE (0.5 factor), QK norm, shared+routed experts, sigmoid scoring, grouped top-k |

**Note:** `DeepSeekMTP` (`atom.models.deepseek_mtp.DeepSeekMTP`) is not in the registry -- it is used exclusively as a speculative draft model for DeepSeek multi-token prediction and is loaded separately.

---

## 2. Model Architecture Details

### Qwen3 (`Qwen3ForCausalLM`)

- **Architecture:** Dense transformer with Grouped-Query Attention (GQA).
- **Layer structure:** `Qwen3DecoderLayer` containing `Qwen3Attention` + `Qwen3MLP`.
- **Attention:** `QKVParallelLinear` for fused QKV projection, per-head QK RMSNorm (`q_norm`, `k_norm`), RoPE, `RowParallelLinear` for output projection.
- **MLP:** `MergedColumnParallelLinear` for gate+up projection, SiLU activation, `RowParallelLinear` for down projection.
- **Normalization:** RMSNorm on input and post-attention.

### Qwen3-MoE (`Qwen3MoeForCausalLM`)

- **Architecture:** Mixture-of-Experts transformer with GQA.
- **Layer structure:** `Qwen3MoeDecoderLayer` containing `Qwen3MoeAttention` + either `Qwen3MoeSparseMoeBlock` (MoE layers) or `Qwen3MoeMLP` (dense layers, controlled by `mlp_only_layers` and `decoder_sparse_step`).
- **Attention:** Same QKV structure as Qwen3 with QK norm. Supports QK norm + RoPE + cache + quant fusion when `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION` is set -- this precomputes a joint `cos_sin_cache` and passes `q_norm`/`k_norm` to the `Attention` module.
- **MoE:** `FusedMoE` with `ReplicatedLinear` gate router. Supports allreduce+RMSNorm fusion (`ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION`).
- **Normalization:** RMSNorm with optional fused allreduce.

### Llama (`LlamaForCausalLM`)

- **Architecture:** Dense transformer with GQA. Covers Llama 2/3 and compatible architectures (InternLM, Mistral-Nemo via optional `head_dim`).
- **Layer structure:** `LlamaDecoderLayer` containing `LlamaAttention` + `LlamaMLP`.
- **Attention:** `QKVParallelLinear`, RoPE (NeoX or original style based on GGUF), per-layer sliding window support via `layer_types` config.
- **MLP:** `MergedColumnParallelLinear` for gate+up, SiLU+mul activation, `RowParallelLinear` for down.
- **Fused optimizations:** Controlled by environment variables:
  - `ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT` -- fuses RMSNorm with FP8/MXFP4 quantization.
  - `ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT` -- fuses SiLU+mul activation with quantization.
- **Pipeline parallelism:** Full PP support with `PPMissingLayer` placeholders and `IntermediateTensors` for cross-stage communication. Supports auxiliary hidden state extraction for speculative decoding.

### Mixtral (`MixtralForCausalLM`)

- **Architecture:** Sparse Mixture-of-Experts with GQA.
- **Layer structure:** `MixtralDecoderLayer` containing `MixtralAttention` + `MixtralMoE`.
- **Attention:** Standard GQA with `QKVParallelLinear`, RoPE (NeoX style), `RowParallelLinear`.
- **MoE:** `MixtralMoE` wraps `ReplicatedLinear` gate + `FusedMoE`. Experts are sharded across TP ranks with full reduce. Gate checkpoint names use `w1`/`w2`/`w3` convention (mapped to `gate_proj`/`down_proj`/`up_proj`).
- **Normalization:** RMSNorm.

### DeepSeek V2/V3 (`DeepseekV2ForCausalLM`)

- **Architecture:** MoE transformer with Multi-head Latent Attention (MLA).
- **Layer structure:** `DeepseekV2DecoderLayer` containing `DeepseekV2MLAAttention` + either `DeepseekV2MoE` (MoE layers) or `DeepseekV2MLP` (dense layers).
- **MLA Attention:** Uses LoRA-compressed QKV (`q_lora_rank`, `kv_lora_rank`), separate `qk_nope_head_dim` and `qk_rope_head_dim` for non-positional and rotary-embedded components. Backed by `MLAModules` from `atom.model_ops.attention_mla`.
- **MoE:** `DeepseekV2MoE` with routed + shared experts. Supports shared expert fusion (`is_rocm_aiter_fusion_shared_expert_enabled`), routed scaling factor fusion (`is_rocm_aiter_fuse_routed_scaling_factor`), and grouped top-k routing.
- **Fused optimizations:**
  - `ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION` -- fuses input RMSNorm with FP8/FP4 quantization.
  - `ATOM_ENABLE_DS_QKNORM_QUANT_FUSION` -- fuses QK norm with quantization.
  - `ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION` -- fuses allreduce with RMSNorm.
  - Dedicated Triton kernels for FP8 MQA logits (`fp8_mqa_logits`), paged MQA logits (`deepgemm_fp8_paged_mqa_logits`), and fused RMSNorm+quantization (`_fuse_rmsnorm_quant`).
- **V3.2 extension:** `DeepseekV32ForCausalLM` is an alias. The `DeepseekV2Model` detects V3.2 via `config.index_topk` and allocates an `topk_indices_buffer` for index-based routing.
- **Note:** `DeepseekV3ForCausalLM` is a subclass of `DeepseekV2ForCausalLM` (pass-through, no override).

### DeepSeek MTP (`DeepSeekMTP`)

- **Architecture:** Multi-Token Prediction draft model for speculative decoding.
- **Layer structure:** `DeepSeekMultiTokenPredictor` containing one or more `DeepSeekMultiTokenPredictorLayer`, each with `enorm` (embedding norm), `hnorm` (hidden state norm), `eh_proj` (linear projection joining embedded+hidden), `mtp_block` (a `DeepseekV2DecoderLayer`), and a `SharedHead` (norm + LM head).
- **Usage:** Not registered in `support_model_arch_dict`. Loaded separately with `spec_decode=True` in `load_model()`, which invokes `rewrite_spec_layer_name()` to remap MTP weight names (e.g., adding `.mtp_block.` prefix for transformer layer weights, remapping `embed_tokens` to top-level).
- **MTP layers start** at `config.num_hidden_layers` (i.e., the layer indices following the main model layers).

### GPT-OSS (`GptOssForCausalLM`)

- **Architecture:** MoE transformer with GQA and alternating sliding window attention.
- **Layer structure:** `TransformerBlock` containing `OAIAttention` + `MLPBlock`.
- **Attention:** `OAIAttention` with bias on QKV and output projections, attention sinks (learnable per-head parameters), and sliding window applied on even-indexed layers only.
- **MoE:** `MLPBlock` wraps `ReplicatedLinear` router (with bias) + `FusedMoE` with SwiGLU activation and bias support. Custom `weights_mapping` translates checkpoint names (`gate_up_proj_blocks` to `w13_weight`, etc.).
- **Normalization:** RMSNorm with eps=1e-5, post-attention norm uses `x_pad_to_multiple=256`.
- **Pipeline parallelism:** Supports auxiliary hidden state layers for EAGLE3 speculative decoding (`get_eagle3_aux_hidden_state_layers`).

### GLM4-MoE (`Glm4MoeForCausalLM`)

- **Architecture:** MoE transformer with GQA, shared + routed experts, partial RoPE.
- **Layer structure:** `Glm4MoeDecoderLayer` containing `Glm4MoeAttention` + either `Glm4MoE` (MoE layers, from `first_k_dense_replace` onward) or `Glm4MoeMLP` (dense layers).
- **Attention:** `Glm4MoeAttention` with optional QK norm (`use_qk_norm`), partial rotary factor of 0.5.
- **MoE:** `Glm4MoE` with sigmoid scoring, `e_score_correction_bias`, grouped top-k routing (`n_group`, `topk_group`), routed scaling factor. Shared experts handled separately or fused into `FusedMoE` via `is_rocm_aiter_fusion_shared_expert_enabled()`. Expert parallelism (EP) support built in.
- **Inherits:** `Glm4MixtureOfExperts` mixin for MoE metadata management and expert load balancing (EPLB) support.

---

## 3. Weight Loading

Weight loading is handled by `load_model()` in `atom/model_loader/loader.py`.

### Function Signature

```python
def load_model(
    model: nn.Module,
    model_name_or_path: str,
    hf_config: AutoConfig,
    load_dummy: bool = False,
    spec_decode: bool = False,
):
```

### Loading Flow

1. **SafeTensors iteration:** `safetensors_weights_iterator()` discovers and iterates over all `*.safetensors` files in the model directory (or downloads them from HuggingFace Hub via `download_weights_from_hf()`). Duplicate files are filtered using the `model.safetensors.index.json` weight map. Memory-mapped loading is used by default; set `ATOM_DISABLE_MMAP=true` to disable.

2. **Weight name rewriting:** Each weight name goes through several transformations:
   - `weight_scale_inv` is renamed to `weight_scale`.
   - Model-specific `weights_mapping` (e.g., GPT-OSS maps `gate_up_proj_blocks` to `w13_weight`).
   - For speculative decoding (`spec_decode=True`), MTP layer weights are rewritten via `rewrite_spec_layer_name()`.
   - Shared expert fusion: when enabled, `mlp.shared_experts` is remapped to `mlp.experts.<n_routed_experts>` so the shared expert is loaded as the last expert in the `FusedMoE` module.

3. **Packed module resolution:** The `packed_modules_mapping` dict on each model class defines how HuggingFace checkpoint weight names map to ATOM's fused parameter names. For example, Llama maps:
   ```python
   "q_proj": ("qkv_proj", "q"),
   "k_proj": ("qkv_proj", "k"),
   "v_proj": ("qkv_proj", "v"),
   "gate_proj": ("gate_up_proj", 0),
   "up_proj": ("gate_up_proj", 1),
   ```
   Each packed parameter has a `weight_loader` attribute that knows how to shard and place the weight into the correct slice.

4. **Expert parameter loading:** If the model has a `get_expert_mapping()` method, expert weights are loaded using `FusedMoE.make_expert_params_mapping()`, which generates (param_name, weight_name, expert_id, shard_id) tuples. This handles per-expert sharding across TP ranks.

5. **TP sharding:** Parallel linear layers (`ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`) have custom `weight_loader` methods that automatically select the correct shard for the current TP rank during loading. The default fallback `default_weight_loader` handles simple cases where weights need to be sliced by TP rank.

6. **Concurrent loading:** All weight loading calls are submitted to a `ThreadPoolExecutor` for parallel execution.

7. **Post-processing:** After all weights are loaded, `process_weights_after_loading()` is called on each module (e.g., for weight pre-shuffling, scale computation), and `quant_method.process_weights_after_loading()` is invoked for quantized modules. For `FusedMoEMethodBase`, `init_prepare_finalize()` is also called.

### Layers Beyond `num_hidden_layers`

Weights for layers with index >= `config.num_hidden_layers` are skipped during normal loading. These layers (MTP layers) are only loaded when `spec_decode=True`.

---

## 4. Adding a New Model

Follow these steps to add support for a new model architecture:

### Step 1: Create the Model File

Create a new file in `atom/models/`, e.g., `atom/models/my_model.py`. Follow the existing patterns:

```python
from atom.config import Config, QuantizationConfig
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils.decorators import support_torch_compile
```

### Step 2: Implement Layer Classes

Each model typically defines three core module classes:

1. **Attention module** (e.g., `MyModelAttention`):
   - Initialize `QKVParallelLinear` for query/key/value.
   - Initialize `RowParallelLinear` for output projection.
   - Set up rotary embeddings via `aiter.rotary_embedding.get_rope()`.
   - Create `Attention` from `atom.model_ops.base_attention`.

2. **MLP module** (e.g., `MyModelMLP`):
   - Use `MergedColumnParallelLinear` for gate+up projections.
   - Use `RowParallelLinear` for down projection.
   - For MoE models, use `FusedMoE` from `atom.model_ops.moe`.

3. **Decoder layer** (e.g., `MyModelDecoderLayer`):
   - Combine attention + MLP with RMSNorm layers.
   - Implement the forward pass with residual connections.

### Step 3: Implement the Model and CausalLM Classes

1. **Backbone model** (e.g., `MyModel`):
   - Decorate with `@support_torch_compile`.
   - Initialize `VocabParallelEmbedding`, decoder layers via `make_layers()`, and final `RMSNorm`.
   - Support pipeline parallelism with `PPMissingLayer` and `IntermediateTensors`.

2. **CausalLM wrapper** (e.g., `MyModelForCausalLM`):
   - Define `packed_modules_mapping` to map checkpoint weight names to ATOM's fused parameter names.
   - Initialize the backbone model and `ParallelLMHead`.
   - Implement `forward()` (returns hidden states) and `compute_logits()` (returns logits via `lm_head`).
   - If the model uses MoE, implement `get_expert_mapping()` returning `FusedMoE.make_expert_params_mapping(...)`.

### Step 4: Register the Model

Add an entry to `support_model_arch_dict` in `atom/model_engine/model_runner.py`:

```python
support_model_arch_dict = {
    ...
    "MyModelForCausalLM": "atom.models.my_model.MyModelForCausalLM",
}
```

The key must exactly match the `architectures` field in the HuggingFace model's `config.json`.

### Step 5: Handle Weight Loading

Ensure your `packed_modules_mapping` correctly maps all checkpoint weight names that differ from ATOM's internal names. Common patterns:

| Checkpoint Name | ATOM Parameter | Shard ID |
|---|---|---|
| `q_proj` | `qkv_proj` | `"q"` |
| `k_proj` | `qkv_proj` | `"k"` |
| `v_proj` | `qkv_proj` | `"v"` |
| `gate_proj` | `gate_up_proj` | `0` |
| `up_proj` | `gate_up_proj` | `1` |

For MoE models, add `get_expert_mapping()` to delegate to `FusedMoE.make_expert_params_mapping()` with the correct gate/down/up projection names and expert count.

If the checkpoint uses non-standard weight names (like GPT-OSS), define a `weights_mapping` class attribute to rename them at load time.

---

## 5. Model-Specific Optimizations

### Llama: Fused RMSNorm+Quant and SiLU+Mul+Quant

Llama supports two AITER Triton fused kernel optimizations:

- **`ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT`**: Fuses the RMSNorm normalization with FP8 or MXFP4 quantization in a single kernel call. Applied to both `input_layernorm` and `post_attention_layernorm`. Eliminates an extra read/write pass over the hidden states.

- **`ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT`**: Fuses the SiLU activation, element-wise multiply, and quantization in the MLP. The `SiluAndMul` module receives the `fused_quant=True` flag and the quant config, producing quantized output directly for the down projection.

Both are controlled by environment variables and read from `atom.utils.envs`.

### DeepSeek V2/V3: MLA + Fused Input Norm + QK Norm Fusion

DeepSeek models use Multi-head Latent Attention (MLA) with LoRA-compressed projections (`q_lora_rank`, `kv_lora_rank`). Several fusion optimizations are available:

- **`ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION`**: Fuses the input RMSNorm with quantization. Implemented via `_fuse_rmsnorm_quant()` which dispatches to either `_fuse_rmsnorm_fp4_quant()` or `_fused_rms_fp8_group_quant()` based on the quant dtype. When enabled, the allreduce+RMSNorm fusion is disabled for `input_layernorm` but kept for `post_attention_layernorm`.

- **`ATOM_ENABLE_DS_QKNORM_QUANT_FUSION`**: Fuses the Q/K LoRA layernorm with quantization via `_fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4()` or the FP8 variant, which performs the fused QKV-A projection, RMSNorm on Q and KV components, and quantization in a single fused operation.

- **`ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION`**: Fuses tensor-parallel allreduce with RMSNorm.

- **FP8 MQA logits**: `fp8_mqa_logits` and `deepgemm_fp8_paged_mqa_logits` implement FP8-precision attention score computation for MLA decode.

- **FP4 support**: MXFP4 quantized GEMM kernels (`gemm_afp4wfp4_preshuffle`, `gemm_a16wfp4_preshuffle`) and FP4 block-scale BMM via `is_rocm_aiter_fp4bmm_enabled()`.

### Qwen3-MoE: QK Norm + RoPE + Cache + Quant Fusion

When `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION` is enabled, the `Qwen3MoeAttention` module:
1. Precomputes a joint `cos_sin_cache` by concatenating cosine and sine RoPE caches.
2. Passes `q_norm` and `k_norm` directly to the `Attention` module.
3. The attention backend then fuses QK normalization, RoPE application, KV cache write, and optional quantization into a single kernel pass.

Additionally, `ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION` fuses allreduce with RMSNorm for both attention output and MoE output, reducing communication overhead.

### MTP: DeepSeek Multi-Token Prediction

The `DeepSeekMTP` model serves as a speculative draft model:
- Each `DeepSeekMultiTokenPredictorLayer` takes the previous hidden state and the next token's embedding, normalizes both (`enorm`, `hnorm`), concatenates them, and passes through a linear projection (`eh_proj`) followed by a standard `DeepseekV2DecoderLayer`.
- The `SharedHead` provides per-layer norm + LM head for logit computation.
- For FP4 quantized main models, MTP blocks fall back to non-FP4 quantization config to maintain draft model accuracy.

---

## Source Files

| File | Description |
|------|-------------|
| `atom/model_engine/model_runner.py` | Model registry (`support_model_arch_dict`) and `ModelRunner` class |
| `atom/models/llama.py` | Llama model: `LlamaForCausalLM`, `LlamaModel`, `LlamaDecoderLayer`, `LlamaAttention`, `LlamaMLP` |
| `atom/models/qwen3.py` | Qwen3 model: `Qwen3ForCausalLM`, `Qwen3Model`, `Qwen3DecoderLayer`, `Qwen3Attention`, `Qwen3MLP` |
| `atom/models/qwen3_moe.py` | Qwen3-MoE model: `Qwen3MoeForCausalLM`, `Qwen3MoeModel`, `Qwen3MoeDecoderLayer`, `Qwen3MoeAttention`, `Qwen3MoeSparseMoeBlock`, `Qwen3MoeMLP` |
| `atom/models/deepseek_v2.py` | DeepSeek V2/V3 model: `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`, `DeepseekV2Model`, `DeepseekV2DecoderLayer`, `DeepseekV2MLAAttention`, `DeepseekV2MoE`, `DeepseekV2MLP` |
| `atom/models/deepseek_mtp.py` | DeepSeek MTP draft model: `DeepSeekMTP`, `DeepSeekMultiTokenPredictor`, `DeepSeekMultiTokenPredictorLayer`, `SharedHead` |
| `atom/models/mixtral.py` | Mixtral model: `MixtralForCausalLM`, `MixtralModel`, `MixtralDecoderLayer`, `MixtralAttention`, `MixtralMoE` |
| `atom/models/gpt_oss.py` | GPT-OSS model: `GptOssForCausalLM`, `GptOssModel`, `TransformerBlock`, `OAIAttention`, `MLPBlock` |
| `atom/models/glm4_moe.py` | GLM4-MoE model: `Glm4MoeForCausalLM`, `Glm4MoeModel`, `Glm4MoeDecoderLayer`, `Glm4MoeAttention`, `Glm4MoE`, `Glm4MoeMLP` |
| `atom/models/utils.py` | Model utilities: `IntermediateTensors`, `PPMissingLayer`, `make_layers`, `maybe_prefix`, `extract_layer_index`, `should_ignore_layer`, `get_quant_config_for_layer` |
| `atom/model_loader/loader.py` | Weight loading: `load_model`, `safetensors_weights_iterator`, `default_weight_loader` |
| `atom/model_loader/weight_utils.py` | Weight utilities: `download_weights_from_hf`, `set_weight_attrs`, `filter_duplicate_safetensors_files` |
