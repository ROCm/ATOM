# ATOM Model Operations Guide

ATOM (AiTer Optimized Model) wraps AITER kernels with model-level abstractions for LLM inference on AMD ROCm/HIP GPUs. This guide documents every operator class in `atom/model_ops/`, their AITER kernel mappings, quantization paths, and fused kernel chains.

---

## Quick Reference

| ATOM Class | File | AITER Kernel / Import | Purpose |
|---|---|---|---|
| `LinearBase` | `linear.py` | `tgemm.mm`, `gemm_a8w8`, `gemm_a8w8_bpreshuffle`, `gemm_a8w8_blockscale_bpreshuffle`, `gemm_a4w4` | Quantized linear dispatch |
| `ColumnParallelLinear` | `linear.py` | (inherits `LinearBase`) | Column-sharded TP linear |
| `RowParallelLinear` | `linear.py` | (inherits `LinearBase`) | Row-sharded TP linear |
| `QKVParallelLinear` | `linear.py` | (inherits `ColumnParallelLinear`) | Fused Q/K/V projection |
| `MergedColumnParallelLinear` | `linear.py` | (inherits `LinearBase`) | Merged gate+up projection |
| `Attention` | `base_attention.py` | `unified_attention_with_output_base` (custom op) | Unified attention entry |
| MHA `Attention` | `attention_mha.py` | `flash_attn_varlen_func`, `pa_fwd_asm`, `pa_persistent_fwd`, `pa_decode_gluon` | Multi-head attention |
| `MLAAttention` | `attention_mla.py` | `mla_decode_fwd`, `mla_prefill_fwd`, `concat_and_cache_mla`, `fused_qk_rope_concat_and_cache_mla` | Multi-head latent attention |
| `FusedMoE` | `moe.py` | `aiter.fused_moe.fused_moe`, `asm_moe` | Mixture of experts |
| `RMSNorm` | `layernorm.py` | `rmsnorm2d_fwd`, `rmsnorm2d_fwd_with_add`, `fused_add_rmsnorm_pad` | RMS normalization |
| `LayerNorm` | `layernorm.py` | `layernorm2d_fwd`, `layernorm2d_fwd_with_add` | Layer normalization |
| `SiluAndMul` | `activation.py` | `aiter.silu_and_mul` | SiLU gated activation |
| `VocabParallelEmbedding` | `embed_head.py` | `F.embedding` + TP all-reduce | Vocab embedding |
| `ParallelLMHead` | `embed_head.py` | `tgemm.mm` + `tensor_model_parallel_all_gather` | LM output head |
| `RotaryEmbedding` | `rotary_embedding.py` | `aiter.rope_cached_positions_2c_fwd_inplace` | Rotary position embedding |
| `Sampler` | `sampler.py` | `aiter.mixed_sample_outer_exponential`, `aiter.ops.triton.topk.topk`, `aiter.ops.triton.softmax.softmax` | Token sampling |
| `RejectionSampler` | `rejection_sampler.py` | Triton `rejection_greedy_sample_kernel` | Speculative decoding |

---

## 1. AITER Integration Overview

ATOM is a thin model-level inference engine. Every compute-heavy operation delegates to an AITER kernel. The general pattern is:

1. An ATOM `nn.Module` owns model weights and configuration.
2. Its `forward()` method selects the appropriate AITER function based on quantization type, parallelism settings, and phase (prefill vs. decode).
3. Results are optionally reduced across tensor-parallel (TP) or data-parallel (DP) groups.

### AITER Kernel Mapping Table

| ATOM Wrapper | AITER Function / Import Path | Backend Type |
|---|---|---|
| `LinearBase.forward` (No quant) | `aiter.tuned_gemm.tgemm.mm` | hipBLASLt |
| `LinearBase.forward` (per_Tensor FP8) | `aiter.tuned_gemm.tgemm.mm` with scales | hipBLASLt |
| `LinearBase.forward` (per_Token INT8) | `aiter.gemm_a8w8` | CK |
| `LinearBase.forward` (per_Token FP8) | `aiter.gemm_a8w8_bpreshuffle` | CK |
| `LinearBase.forward` (per_1x128 FP8) | `aiter.gemm_a8w8_blockscale_bpreshuffle` | CK |
| `LinearBase.forward` (per_1x32 MXFP4) | `aiter.gemm_a4w4` | CK |
| MHA prefill | `aiter.flash_attn_varlen_func` | ASM / CK |
| MHA decode (ASM) | `aiter.pa_fwd_asm` | ASM |
| MHA decode (persistent ASM) | `aiter.pa_persistent_fwd` | ASM |
| MHA decode (Triton) | `aiter.ops.triton.gluon.pa_decode_gluon` | Triton |
| MHA prefill (Triton unified) | `aiter.ops.triton.unified_attention.unified_attention` | Triton |
| MLA decode | `aiter.mla.mla_decode_fwd` | ASM |
| MLA prefill | `aiter.mla.mla_prefill_fwd` | ASM |
| MLA KV cache | `aiter.concat_and_cache_mla` | CK |
| RoPE | `aiter.rope_cached_positions_2c_fwd_inplace` | Triton |
| RMSNorm | `aiter.rmsnorm2d_fwd` | CK |
| SiLU+Mul | `aiter.silu_and_mul` | CK |
| TopK routing | `aiter.topk_softmax`, `aiter.grouped_topk`, `aiter.biased_grouped_topk` | CK |
| Sampling | `aiter.mixed_sample_outer_exponential` | CK |
| FusedMoE | `aiter.fused_moe.fused_moe` | CK |
| ASM MoE | `aiter.fused_moe_bf16_asm.asm_moe` | ASM |
| Quantization | `aiter.get_hip_quant(QuantType)` | CK / Triton |

---

## 2. Linear Operations

All linear layers inherit from `LinearBase` in `atom/model_ops/linear.py`.

### 2.1 Class Hierarchy

```
LinearBase (nn.Module)
  +-- ReplicatedLinear          # No TP sharding
  |     +-- MergedReplicatedLinear
  +-- ColumnParallelLinear      # tp_dim=0, shard output
  |     +-- QKVParallelLinear   # Fused Q/K/V with per-head sharding
  +-- MergedColumnParallelLinear # tp_dim=0, merged gate+up
  +-- RowParallelLinear          # tp_dim=1, shard input, optional all-reduce
```

### 2.2 Quantization Dispatch

`LinearBase.forward()` dispatches to different GEMM kernels based on `QuantType`:

| `QuantType` | Weight dtype | GEMM Kernel | Scale Shape |
|---|---|---|---|
| `No` | BF16/FP16 | `tgemm.mm` (hipBLASLt) | None |
| `per_Tensor` | FP8 | `tgemm.mm` with `scale_a`, `scale_b` | `[num_partitions, 1]` |
| `per_Token` (INT8) | INT8 | `gemm_a8w8` | `[output_size, 1]` |
| `per_Token` (FP8) | FP8 | `gemm_a8w8_bpreshuffle` | `[output_size, 1]` |
| `per_1x128` | FP8 | `gemm_a8w8_blockscale_bpreshuffle` | `[ceil(N/128), ceil(K/128)]` |
| `per_1x32` | MXFP4 (`fp4x2`) | `gemm_a4w4` | `[N, ceil(K/32)]` (e8m0) |

When `x_scale` is not provided, the input is dynamically quantized via `get_hip_quant(quant_type)`.

### 2.3 Tensor Parallel Sharding

- **ColumnParallelLinear** (`tp_dim=0`): Shards weight rows (output dimension) across GPUs. Each GPU owns `output_size / tp_size` rows.
- **RowParallelLinear** (`tp_dim=1`): Shards weight columns (input dimension). If `reduce_results=True`, output is all-reduced across TP group.
- **QKVParallelLinear**: Extends `ColumnParallelLinear` with per-head sharding. Q heads are evenly divided; KV heads are either divided or replicated when `num_kv_heads < tp_size`.
- **MergedColumnParallelLinear**: Handles gate and up projections merged into a single weight with `output_sizes` as a list (e.g., `[intermediate_size, intermediate_size]`).

### 2.4 Weight Processing

After loading, `process_weights_after_loading()` handles:
- **e4m3fn to e4m3fnuz normalization** (AMD FP8 format conversion).
- **Weight reshuffling** via `shuffle_weights()` for pre-shuffled GEMM kernels.
- **Scale reshuffling** via `fp4_utils.e8m0_shuffle()` for MXFP4 block scales.
- **Per-tensor requantization** via `requantize_with_max_scale()` when multiple output partitions have separate scales.

---

## 3. Attention Operations

### 3.1 Base: `Attention` (`base_attention.py`)

The top-level `Attention` class in `base_attention.py` is a dispatcher. It:

1. Selects the backend via `get_attn_backend()` from `atom/utils/selector.py`.
2. Instantiates the backend's implementation class (`impl_cls`).
3. Registers itself in `compilation_config.static_forward_context` under `layer_name`.
4. On `forward()`, calls `torch.ops.aiter.unified_attention_with_output_base`, which is a custom op decorated with `@mark_spliting_op` -- this prevents `torch.compile` from tracing into attention internals, enabling full-graph capture.

Backend selection logic (in `selector.py`):

| Condition | Backend Class | Implementation |
|---|---|---|
| `use_mla=True` | `AiterMLABackend` | `MLAAttention` from `attention_mla.py` |
| `use_mla=False` | `AiterBackend` | `Attention` from `attention_mha.py` |

### 3.2 Multi-Head Attention (`attention_mha.py`)

The MHA `Attention` class handles standard models (Llama, Qwen3, Mixtral, etc.).

**Forward flow:**

1. Reshape Q, K, V to `[num_tokens, num_heads, head_dim]`.
2. Apply RoPE + KV cache write via `rope_cache()`.
3. Dispatch to the appropriate backend via `dispatch_backend()`.

**RoPE + KV cache paths:**

| Condition | Kernel Chain |
|---|---|
| `q_norm` + `k_norm` + `rotary_emb` present | `fused_qk_norm_rope_cache_quant_shuffle` (single fused kernel for QK norm, RoPE, cache write, optional FP8 quant) |
| Triton path (`sliding_window != -1` or `head_dim != 128`) + `rotary_emb` | `fused_qk_rope_reshape_and_cache` (Triton fused RoPE + reshape + cache) |
| ASM path + `rotary_emb` | `rotary_emb(position, q, k)` then `reshape_and_cache` or `reshape_and_cache_with_pertoken_quant` |

**Attention dispatch:**

| Phase | Condition | Method | AITER Kernel |
|---|---|---|---|
| Prefill | Always | `prefill_attention` | `aiter.flash_attn_varlen_func` |
| Decode | `use_triton_attn=True` | `paged_attention_triton` | `torch.ops.aiter.pa_decode_gluon` |
| Decode | `block_size == 1024` | `paged_attention_persistent_asm` | `aiter.pa_persistent_fwd` |
| Decode | Default | `paged_attention_asm` | `aiter.pa_fwd_asm` |

The `use_triton_attn` flag is set when `sliding_window != -1` or `head_dim != 128`.

### 3.3 Multi-head Latent Attention (`attention_mla.py`)

`MLAAttention` implements DeepSeek's MLA with a compressed KV representation. Key data structures:

```python
@dataclass
class MLAModules:
    q_lora_rank: Optional[int]
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    qk_head_dim: int
    v_head_dim: int
    rotary_emb: torch.nn.Module
    q_proj: Optional[torch.nn.Module]
    kv_b_proj: torch.nn.Module
    o_proj: torch.nn.Module
    indexer: Optional[torch.nn.Module]
```

**Forward flow:**

1. If prefill and not sparse: Standard MHA-style prefill with `flash_attn_varlen_func`, preceded by `kv_b_proj` GEMM to produce K_nope and V from compressed `kv_c_normed`.
2. Otherwise: Fused Q projection + K up-projection via batched FP8/FP4 BMM (`_q_proj_and_k_up_proj`), then:
   - `fused_qk_rope_concat_and_cache_mla` writes to KV cache.
   - Decode: `mla_decode_fwd` (ASM persistent MLA kernel).
   - Prefill (sparse): `mla_prefill_fwd`.
3. V up-projection + O projection via batched BMM (`_v_up_proj_and_o_proj`).

**Batched GEMM backends for MLA projections:**

| Condition | Kernel |
|---|---|
| `ATOM_USE_TRITON_MXFP4_BMM=True` | `batched_gemm_a16wfp4` (Triton FP4 BMM) |
| Default | `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant` (Triton FP8 BMM) |

**Prefill GEMM optimizations** (for `kv_b_proj`):

| Condition | Kernel |
|---|---|
| `ATOM_USE_TRITON_GEMM=True` + FP4 weights | `fused_gemm_afp4wfp4_preshuffle_split_cat` (GEMM + split K/V + cat rope in one kernel) |
| `ATOM_USE_TRITON_GEMM=True` + FP8 weights | `fused_gemm_a8w8_blockscale_preshuffle_split_cat` |
| Default | `kv_b_proj(kv_c_normed)` then manual split + cat |

### 3.4 Backend Abstraction (`attentions/backends.py`)

The `AttentionBackend` abstract class defines three required methods:

- `get_name()` -- Returns backend identifier string.
- `get_builder_cls()` -- Returns the `AttentionMetadataBuilder` subclass.
- `get_impl_cls()` -- Returns the attention implementation class.

`CommonAttentionBuilder` provides shared metadata preparation (slot mapping, block tables, cumulative sequence lengths) used by both `AiterBackend` and `AiterMLABackend`.

### 3.5 KV Cache Operations

| Operation | AITER Kernel | Used By |
|---|---|---|
| Standard KV cache write | `aiter.reshape_and_cache` | MHA (BF16 KV) |
| FP8 KV cache write | `aiter.reshape_and_cache_with_pertoken_quant` | MHA (FP8 KV) |
| MLA KV cache write | `aiter.concat_and_cache_mla` | MLA prefill |
| Fused QK RoPE + MLA cache | `aiter.fused_qk_rope_concat_and_cache_mla` | MLA decode |

---

## 4. Mixture of Experts (MoE)

### 4.1 `FusedMoE` Class (`moe.py`)

`FusedMoE` is the top-level MoE module. It handles:
- Expert routing via `select_experts()`.
- Weight creation and quantization dispatch via `quant_method`.
- Tensor/Expert/Data parallelism via `FusedMoEParallelConfig`.
- Optional shared expert fusion and MORI communication.

**Constructor parameters:**
```python
FusedMoE(
    num_experts: int,        # Global number of experts
    top_k: int,              # Experts per token
    hidden_size: int,        # Input hidden dimension
    intermediate_size: int,  # Expert intermediate dimension
    reduce_results: bool,    # Whether to all-reduce output
    renormalize: bool,       # Renormalize routing weights
    use_grouped_topk: bool,  # Use grouped top-k (DeepSeek)
    activation: ActivationType,  # Silu, Gelu, Swiglu, etc.
    ...
)
```

### 4.2 Quantization Methods

`FusedMoE` selects a `quant_method` at construction time:

| Quant Config | Method Class | GEMM Kernel |
|---|---|---|
| `QuantType.No` | `UnquantizedFusedMoEMethod` | `aiter.fused_moe.fused_moe` |
| FP8 (`dtypes.fp8`) | `Fp8MoEMethod` | `aiter.fused_moe.fused_moe` with quant_type |
| FP8 compressed-tensors | `CompressedTensorsFp8MoEMethod` | `aiter.fused_moe.fused_moe` or `asm_moe` |
| MXFP4 (`dtypes.fp4x2`) | `Mxfp4MoEMethod` | `aiter.fused_moe.fused_moe` or Triton `triton_kernel_moe_forward` |

The ASM MoE path (`asm_moe` from `aiter.fused_moe_bf16_asm`) is used by FP8 methods and supports `a16` mode where activations remain in BF16/FP16 while weights are FP8/INT8.

### 4.3 TopK Routing (`topK.py`)

| Routing Function | AITER Kernel | Used For |
|---|---|---|
| `rocm_aiter_topk_softmax` | `aiter.topk_softmax` | Standard top-k (Mixtral) |
| `rocm_aiter_grouped_topk` | `aiter.grouped_topk` | Grouped top-k (DeepSeek) |
| `rocm_aiter_biased_grouped_topk` | `aiter.biased_grouped_topk` | Biased grouped top-k (DeepSeek V3) |

**Shared expert fusion:** When `is_rocm_aiter_fusion_shared_expert_enabled()` returns `True`, the top-k buffers are extended with shared expert IDs appended after routed expert IDs. This allows shared expert computation to be fused into the same MoE kernel call. The metadata is initialized via `init_aiter_topK_meta_data()`.

### 4.4 `FusedMoEParallelConfig`

```python
@dataclass
class FusedMoEParallelConfig:
    tp_size: int       # Tensor parallel size
    dp_size: int       # Data parallel size
    ep_size: int       # Expert parallel size
    tp_rank: int
    dp_rank: int
    ep_rank: int
    use_ep: bool       # Whether expert parallelism is active
    local_ep_size: int # Local EP size (GPUs per node * TP)
```

Key properties:
- `use_all2all_kernels`: `True` when `dp_size > 1`, EP is enabled, and MORI is available.
- `use_mori_kernels`: Always `True` (currently).

### 4.5 MORI Integration (`fused_moe/mori_prepare_finalize.py`)

MORI (MoE Router Infrastructure) provides all-to-all communication kernels for expert parallelism. `MoriPrepareAndFinalize` implements:

- `prepare()`: Dispatches tokens to remote experts via `mori_op.dispatch()`. Optionally quantizes activations to FP8 before dispatch.
- `finalize()`: Combines expert outputs via `mori_op.combine()` and copies results back.

The `FusedMoEModularKernel` orchestrates the prepare-compute-finalize pipeline.

### 4.6 MoE Quantization Config (`fused_moe/config.py`)

`FusedMoEQuantConfig` describes activation and weight quantization for MoE layers:

```python
@dataclass
class FusedMoEQuantConfig:
    _a1: FusedMoEQuantDesc   # First activation (input to gate_up)
    _a2: FusedMoEQuantDesc   # Second activation (input to down_proj)
    _w1: FusedMoEQuantDesc   # gate_up_proj weights
    _w2: FusedMoEQuantDesc   # down_proj weights
```

Factory functions:
- `fp8_w8a8_moe_quant_config()` -- FP8 weights and activations.
- `mxfp4_w4a16_moe_quant_config()` -- MXFP4 weights, unquantized activations.
- `FUSED_MOE_UNQUANTIZED_CONFIG` -- No quantization.

### 4.7 Triton MoE Fallback (`fused_moe_triton.py`)

`triton_kernel_moe_forward()` provides a Triton-based MoE path using the `triton_kernels` library. It uses `routing()` for expert assignment and `matmul_ogs()` for the expert GEMM. This path is currently used for MXFP4 MoE on GFX94x hardware.

---

## 5. Normalization

### 5.1 `RMSNorm` (`layernorm.py`)

`RMSNorm` supports multiple forward paths depending on configuration flags:

| Condition | Kernel / Path | Returns |
|---|---|---|
| `x_pad_to_multiple > 0`, no residual | `fused_rmsnorm_pad_` (Triton `fused_add_rmsnorm_pad`) | Padded output |
| `x_pad_to_multiple > 0`, with residual | `fused_add_rmsnorm_pad_` | (output, residual) |
| `fused_allreduce=True` and `tp_size > 1` | `tensor_model_parallel_fused_allreduce_rmsnorm` | (output, residual) |
| `fused_quant=True` and `x_scale` provided | `fused_rms_fp8_per_tensor_static_quant` | (FP8 output, scale) |
| `fused_quant=True` and `per_1x32` | `fused_rms_mxfp4_quant` | (MXFP4 output, scale) |
| Default, no residual | `rmsnorm2d_fwd` | Output |
| Default, with residual | `rmsnorm2d_fwd_with_add` | (output, residual) |

Constructor parameters:
```python
RMSNorm(
    dim: int,
    eps: float = 1e-6,
    x_pad_to_multiple: int = 0,
    fused_allreduce: bool = False,
    fused_quant: bool = False,
    quant_config: Optional[QuantizationConfig] = None,
)
```

### 5.2 `LayerNorm` (`layernorm.py`)

`LayerNorm` wraps `layernorm2d_fwd` and `layernorm2d_fwd_with_add` (with bias support):

```python
LayerNorm(dim: int, eps: float = 1e-6)
```

- Without residual: `layernorm2d_fwd(x, weight, bias, eps)`
- With residual: `layernorm2d_fwd_with_add(out, x, residual, residual_out, weight, bias, eps)`

---

## 6. Activation Functions

### 6.1 `SiluAndMul` (`activation.py`)

`SiluAndMul` computes `SiLU(x_first_half) * x_second_half`. It splits the last dimension in half.

| Condition | Kernel | Output |
|---|---|---|
| `fused_quant=True` + `x_scale` provided (FP8) | `fused_silu_mul_fp8_per_tensor_static_quant` | `(FP8 output, scale)` |
| `fused_quant=True` + `per_1x32` (MXFP4) | `fused_reduce_act_mul_and_mxfp4_quant` (via `mxfp4_act_mul_quant_fuse`) | `(MXFP4 output, scale)` |
| Default | `aiter.silu_and_mul(out, x)` | BF16 output |

Constructor:
```python
SiluAndMul(
    fused_quant: bool = False,
    quant_config: Optional[QuantizationConfig] = None,
)
```

---

## 7. Embedding & Output Head

### 7.1 `VocabParallelEmbedding` (`embed_head.py`)

Partitions the vocabulary across TP ranks. Each rank holds `num_embeddings / tp_size` rows.

**Forward:**
1. Mask input token IDs to this rank's partition range `[vocab_start_idx, vocab_end_idx)`.
2. `F.embedding()` on local partition.
3. Zero out out-of-range positions.
4. `all_reduce()` across TP group.

### 7.2 `ParallelLMHead` (`embed_head.py`)

Extends `VocabParallelEmbedding` for the output projection. Key differences:

- **Forward** extracts only the last token per sequence during prefill (via `cu_seqlens_q[1:] - 1`).
- Uses `tgemm.mm(x, self.weight, self.bias)` for the logit computation (not `F.linear`).
- Calls `tensor_model_parallel_all_gather()` to gather logits across TP ranks.

---

## 8. Rotary Position Embedding (RoPE)

### 8.1 `RotaryEmbedding` (`rotary_embedding.py`)

Precomputes cos/sin caches at initialization and applies RoPE in-place.

**Constructor:**
```python
RotaryEmbedding(
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool = True,
    dtype: Optional[torch.dtype] = None,
)
```

**Forward:** Calls `aiter.rope_cached_positions_2c_fwd_inplace(query_, key_, cos, sin, positions, rotate_style, ...)` which applies RoPE to Q and K tensors in-place using precomputed caches indexed by position IDs.

### 8.2 `get_rope()` Factory

```python
get_rope(head_size, rotary_dim, max_position, base, rope_scaling=None)
```

Returns a cached `RotaryEmbedding` instance. Currently `rope_scaling` must be `None`.

### 8.3 Integration in Attention

- **MHA** (`attention_mha.py`): RoPE is applied during the `rope_cache()` phase, either via the fused `fused_qk_norm_rope_cache_quant_shuffle` kernel, via `fused_qk_rope_reshape_and_cache`, or via standalone `rotary_emb(position, q, k)`.
- **MLA** (`attention_mla.py`): RoPE is applied to `q_pe` and `k_rope` tensors. During decode, this is fused into `fused_qk_rope_concat_and_cache_mla`. During prefill, it is applied via `self.rotary_emb(positions, prefill_q_pe, k_rope)`.

---

## 9. Sampling

### 9.1 `Sampler` (`sampler.py`)

Unified sampling supporting both greedy (temperature=0) and random (temperature>0) sampling in a single kernel call.

**Forward:**
```python
def forward(self, logits, temperatures) -> sampled_tokens:
    mixed_sample_outer_exponential(sampled_tokens, logits, exponential, temperatures, eps)
```

`aiter.mixed_sample_outer_exponential` performs temperature-scaled exponential sampling: it divides logits by temperature, then uses the Gumbel-max trick with pre-generated exponential random variates.

**Fallback methods** (currently unreachable due to early return):
- `greedy_sample()`: `aiter.ops.triton.topk.topk(logits, 1)`
- `random_sample()`: `aiter.ops.triton.softmax.softmax(logits)` followed by exponential sampling and `topk`.

### 9.2 `RejectionSampler` (`rejection_sampler.py`)

Implements rejection sampling for speculative decoding (MTP). Given draft token IDs and target model logits:

1. Computes `target_argmax = target_logits.argmax(dim=-1)`.
2. Runs a Triton kernel `rejection_greedy_sample_kernel` that sequentially compares draft tokens against target argmax, accepting until first mismatch.
3. On full acceptance, appends the bonus token.
4. Returns `(output_token_ids, num_bonus_tokens)`.

---

## 10. Fused Kernel Chains

ATOM uses fused kernels to reduce memory traffic by combining multiple operations into a single kernel launch.

| Fused Operation | Components | Controlled By | AITER Kernel |
|---|---|---|---|
| RMSNorm + FP8 quant | RMSNorm, per-tensor FP8 static quant | `RMSNorm(fused_quant=True)` + `x_scale` | `fused_rms_fp8_per_tensor_static_quant` |
| RMSNorm + MXFP4 quant | RMSNorm, per-1x32 MXFP4 quant | `RMSNorm(fused_quant=True)` + `QuantType.per_1x32` | `fused_rms_mxfp4_quant` |
| RMSNorm + add + pad | Residual add, RMSNorm, output padding | `RMSNorm(x_pad_to_multiple>0)` | `fused_add_rmsnorm_pad` |
| AllReduce + RMSNorm | TP all-reduce, RMSNorm | `RMSNorm(fused_allreduce=True)` | `tensor_model_parallel_fused_allreduce_rmsnorm` |
| SiLU + mul + FP8 quant | SiLU activation, multiply, FP8 quant | `SiluAndMul(fused_quant=True)` + `x_scale` | `fused_silu_mul_fp8_per_tensor_static_quant` |
| SiLU + mul + MXFP4 quant | SiLU activation, multiply, MXFP4 quant | `SiluAndMul(fused_quant=True)` + `QuantType.per_1x32` | `fused_reduce_act_mul_and_mxfp4_quant` |
| QK norm + RoPE + cache + quant | Q/K norm, RoPE, KV cache write, optional FP8 quant, weight shuffle | `q_norm` + `k_norm` + `rotary_emb` all present | `fused_qk_norm_rope_cache_quant_shuffle` |
| RoPE + reshape + cache | RoPE, K reshape, KV cache write | Triton attention path | `fused_qk_rope_reshape_and_cache` |
| QK RoPE + MLA cache | Q RoPE, KV concat, MLA cache write, FP8 quant | MLA decode path | `fused_qk_rope_concat_and_cache_mla` |
| GEMM + split + cat (FP4) | KV_b_proj GEMM, split K_nope/V, cat K_rope | `ATOM_USE_TRITON_GEMM=True` + FP4 weights | `fused_gemm_afp4wfp4_preshuffle_split_cat` |
| GEMM + split + cat (FP8) | KV_b_proj GEMM, split K_nope/V, cat K_rope | `ATOM_USE_TRITON_GEMM=True` + FP8 weights | `fused_gemm_a8w8_blockscale_preshuffle_split_cat` |
| FP8 BMM + RoPE + cache (MLA) | Batched FP8 BMM, RoPE, MLA KV cache write | MLA decode with FP8 | `fused_fp8_bmm_rope_cat_and_cache_mla` |
| FP4 BMM + RoPE + cache (MLA) | Batched FP4 BMM, RoPE, MLA KV cache write | MLA decode with MXFP4 | `fused_fp4_bmm_rope_cat_and_cache_mla` |

---

## Source Files

### `atom/model_ops/`

| File | Description |
|---|---|
| `linear.py` | `LinearBase`, `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear`, `ReplicatedLinear`, `MergedReplicatedLinear` |
| `activation.py` | `SiluAndMul` with fused FP8/MXFP4 quantization |
| `layernorm.py` | `RMSNorm`, `LayerNorm` with fused allreduce/quant/pad variants |
| `base_attention.py` | Top-level `Attention` dispatcher with custom op registration |
| `attention_mha.py` | MHA implementation: prefill (flash), decode (ASM/Triton paged attention) |
| `attention_mla.py` | `MLAAttention`, `MLAModules` -- DeepSeek MLA with compressed KV |
| `moe.py` | `FusedMoE`, `FusedMoEParallelConfig`, `UnquantizedFusedMoEMethod`, `Fp8MoEMethod`, `Mxfp4MoEMethod`, `CompressedTensorsFp8MoEMethod` |
| `fused_moe_triton.py` | `triton_kernel_moe_forward` -- Triton MoE via `triton_kernels` library |
| `embed_head.py` | `VocabParallelEmbedding`, `ParallelLMHead` |
| `rotary_embedding.py` | `RotaryEmbedding`, `get_rope` |
| `topK.py` | `rocm_aiter_topk_softmax`, `rocm_aiter_grouped_topk`, `init_aiter_topK_meta_data` |
| `sampler.py` | `Sampler` -- unified greedy/random sampling |
| `rejection_sampler.py` | `RejectionSampler` -- speculative decoding rejection sampling |
| `base_config.py` | `QuantizeMethodBase` abstract class |
| `utils.py` | Helper utilities: `shuffle_weights`, `normalize_e4m3fn_to_e4m3fnuz`, `per_tensor_dequantize`, etc. |

### `atom/model_ops/attentions/`

| File | Description |
|---|---|
| `backends.py` | `AttentionBackend`, `AttentionMetadataBuilder`, `CommonAttentionBuilder`, `AttentionImpl` abstract classes |
| `aiter_attention.py` | `AiterBackend`, `AiterAttentionMetadataBuilder` -- MHA backend with persistent ASM paged attention support |
| `aiter_mla.py` | `AiterMLABackend`, `AiterMLAMetadataBuilder` -- MLA backend with sparse attention support |

### `atom/model_ops/fused_moe/`

| File | Description |
|---|---|
| `config.py` | `FusedMoEConfig`, `FusedMoEQuantConfig`, `FusedMoEQuantDesc`, `GroupShape`, factory functions (`fp8_w8a8_moe_quant_config`, `mxfp4_w4a16_moe_quant_config`) |
| `modular_kernel.py` | `FusedMoEModularKernel`, `FusedMoEPrepareAndFinalize`, `ExpertTokensMetadata` -- modular MoE kernel pipeline |
| `mori_prepare_finalize.py` | `MoriPrepareAndFinalize` -- MORI all-to-all dispatch/combine for expert parallelism |
| `utils.py` | MoE utility functions |

### `atom/utils/`

| File | Description |
|---|---|
| `selector.py` | `get_attn_backend()` -- selects `AiterBackend` or `AiterMLABackend` based on `use_mla` flag |
