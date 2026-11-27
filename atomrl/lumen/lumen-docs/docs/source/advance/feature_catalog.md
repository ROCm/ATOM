# Lumen Feature Catalog

Last updated: 04/09/2026.

A comprehensive inventory of all Lumen features organized by domain: **Quantization**, **Distributed Training**, and **LoRA**. Each section provides a capability overview table (for team leads and architects) followed by API reference subsections with function signatures, parameters, and return types (for developers).

---

## 1. Quantization Features

### 1.1 Capability Overview

| Feature | Description | CLI Flag / Entry Point | Scaling Types | Status |
|---------|-------------|------------------------|---------------|--------|
| `quant.enable()` | One-call FP8 enablement -- patches all `nn.Linear` modules for FP8 training | `quant.enable(model)` | All | SUPPORTED |
| FP8 Quantization Kernels | Per-tensor, blockwise, MXFP8 quantization primitives | `atomrl.lumen.ops.quantize.ops` | delayed, dynamic, per_token, blockwise, mxfp8 | SUPPORTED |
| FP8 Linear (Forward + Backward) | Quantized linear with 7 scaling modes, full forward/dgrad/wgrad dispatch | `--linear-fp8` | delayed, dynamic, per_token, blockwise, mxfp8, blockwise2d, none | SUPPORTED |
| MXFP4 Linear (Forward + Backward) | Dedicated MXFP4 linear training path with 32-element block scales | `--linear-fp4` | mxfp4 | SUPPORTED |
| Fused Norm + Quantize | RMSNorm/LayerNorm with fused FP8 output -- single kernel pass | `--lumen-norm` | delayed, dynamic, per_token, blockwise, mxfp8 | SUPPORTED |
| FP8 Attention | FP8 quantized QKV in dot-product attention (DPA) or multi-head attention (MHA) | `--lumen-fp8-attn dpa\|mha` | blockwise, dynamic, delayed, per_token, mxfp8, blockwise2d | SUPPORTED |
| FP8 Grouped GEMM (MoE) | FP8 expert GEMM for MoE layers, fused and sequential fallback | `--moe-grouped-gemm` | delayed, dynamic, per_token, blockwise, mxfp8 | SUPPORTED |
| FP8 Activation Store | Save MLP activations as FP8 uint8 in backward pass (~50% memory reduction) | `--lumen-fp8-activation-store` | dynamic (quantize to FP8) | SUPPORTED |
| FP8 Weight Gradients | FP8 GEMM in wgrad path (MXFP8 only; others forced BF16) | `--linear-fp8-wgrad` | mxfp8 | SUPPORTED |
| FP8 Param All-Gather (Layer 1) | Store parameters in FP8 with lazy re-quant on access + optimizer hook | `--lumen-fp8-param-gather` | -- | SUPPORTED |
| FP8 Param Manager (FSDP2) | In-place FP8 param storage for FSDP2, -25% VRAM (no offload) or -5% (offload) | `FP8_PARAM_MANAGER=1` | -- | SUPPORTED |
| FP8 Param Manager (Megatron) | On-the-fly BF16→FP8 quantization for Megatron ColumnParallel/RowParallel, -29% VRAM | `FP8_PARAM_MANAGER=1` | -- | SUPPORTED |
| Scaling Manager | Amax history tracking, delayed scaling lifecycle, cross-rank sync, warmup + reset | Internal (`ScalingManager`) | delayed | SUPPORTED |
| Blockwise2D Quantization | 2D block FP8 scaling for attention and linear backward (Lumen-only) | `--linear-fp8-scaling blockwise2d` | blockwise2d | LUMEN-ONLY |
| Per-Token FP8 Scaling | Per-row FP8 quantization for attention (Lumen-only) | `scaling_type=per_token` | per_token | LUMEN-ONLY |
| FP8 Checkpoint | Save/restore FP8 scaling state across checkpoints | `lumen_checkpoint` | All | SUPPORTED |
| First/Last Layers BF16 | Keep first and last transformer layers in BF16 for numerical stability | `first_last_layers_bf16` | -- | SUPPORTED |

### 1.2 API Reference

#### `quant.enable()`

```python
quant.enable(
    model: torch.nn.Module,
    config: QuantConfig | None = None,
) -> torch.nn.Module
```

Patches all `nn.Linear` layers in the model for FP8 training in one call. Replaces linear modules with Lumen's quantized linear implementation, wires up the `ScalingManager`, and registers optimizer hooks for FP8 parameter management.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | (required) | The model to patch for FP8 training |
| `config` | `QuantConfig \| None` | `None` | Quantization configuration. If `None`, uses default delayed scaling |

**Returns:** The patched model (same object, modified in-place).

**Example:**

```python
import atomrl.lumen
from atomrl.lumen.quantize import QuantConfig

model = build_my_model()
atomrl.lumen.quant.enable(model, QuantConfig(scaling_type="blockwise"))
```

#### `QuantConfig`

```python
class QuantConfig:
    scaling_type: str = "delayed"       # delayed | dynamic | per_token | blockwise | blockwise2d | mxfp8
    fp8_format: str = "hybrid"          # hybrid (e4m3 fwd / e5m2 bwd) | e4m3 | e5m2
    amax_history_len: int = 1024        # Number of steps to track for amax history
    amax_algo: str = "most_recent"      # most_recent | max
    margin: float = 0.0                 # FP8 scaling margin
    fp8_wgrad: bool = False             # Enable FP8 weight gradients (MXFP8 only)
    first_last_layers_bf16: bool = False # Keep first/last layers in BF16
    reduce_amax: bool = True            # Cross-rank amax synchronization
```

Configuration dataclass controlling all FP8 quantization behavior. Passed to `quant.enable()` or used internally by Lumen modules.

#### `ScalingManager`

```python
class ScalingManager:
    def __init__(self, config: QuantConfig, dp_group=None, tp_group=None): ...
    def quantize(self, tensor: Tensor) -> Tuple[Tensor, Tensor]: ...
    def quantize_bwd_delayed(self, tensor: Tensor) -> Tuple[Tensor, Tensor]: ...
    def step(self) -> None: ...
    def enable_fp8_params(self, model: nn.Module) -> None: ...
```

Manages the FP8 scaling lifecycle: amax history collection, scale computation, delayed vs dynamic dispatch, cross-rank amax reduction, and warmup/reset logic.

| Method | Description |
|--------|-------------|
| `quantize(tensor)` | Quantize a tensor to FP8 using current scaling state; returns `(fp8_tensor, scale)` |
| `quantize_bwd_delayed(tensor)` | Delayed per-tensor quantization for backward pass (blockwise2d linear) |
| `step()` | Update amax history and recompute scales (call once per training step) |
| `enable_fp8_params(model)` | Enable FP8 parameter storage with lazy re-quant and optimizer hooks |

#### FP8 Quantization Primitives

```python
# Per-tensor (delayed / dynamic)
static_per_tensor_quant(tensor, scale, fp8_dtype) -> Tensor   # C++ kernel
per_tensor_quant_hip(tensor) -> Tuple[Tensor, Tensor]          # CK/HIP
per_tensor_quant_triton(tensor) -> Tuple[Tensor, Tensor]       # Triton fallback

# Per-token
per_token_quant_hip(tensor) -> Tuple[Tensor, Tensor]           # CK/HIP

# Blockwise
quant_fp8_blockwise_impl(tensor, block_size) -> Tuple[Tensor, Tensor]  # Triton

# MXFP8
convert_to_mxfp8(tensor) -> Tuple[Tensor, Tensor]             # Triton
convert_from_mxfp8(tensor, scale) -> Tensor                    # Triton
```

**Source:** `lumen/ops/quantize/ops.py`

#### FP8 Linear

```python
class LumenColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, *, bias=True,
                 scaling_type="delayed", tp_group=None, sp=False, ...): ...

class LumenRowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, *, bias=True,
                 scaling_type="delayed", tp_group=None, sp=False, ...): ...
```

**Forward dispatch:** Input quantization -> FP8 GEMM -> optional bias. 7 scaling modes with backend priority: hipBLASLt -> CK -> Triton.

**Backward dispatch:** dgrad uses same GEMM as forward; wgrad forced BF16 (except MXFP8).

#### Fused Norm + Quantize

```python
class LumenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, scaling_type=None): ...
    def forward(self, x: Tensor) -> Tensor | Tuple[Tensor, Tensor]: ...

class LumenLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, scaling_type=None): ...
    def forward(self, x: Tensor) -> Tensor | Tuple[Tensor, Tensor]: ...
```

When `scaling_type` is set, the norm operation fuses FP8 quantization into the same kernel call, avoiding an extra memory round-trip. Returns `(normed_fp8, scale)` when fused, or just `normed_bf16` when `scaling_type=None`.

#### FP8 Attention

```python
# BF16 attention
attention(
    query, key, value, *,
    attn_mask=None, dropout_p=0.0, is_causal=False,
    backend_type="auto",       # auto | aiter_csrc | aiter_triton
    cp_param_bundle=None,
    return_lse=False,
) -> Tensor | Tuple[Tensor, Tensor]

# FP8 attention
attention_fp8_quant(
    query, key, value, *,
    quant_type="blockwise",    # blockwise | dynamic | delayed | per_token | mxfp8 | blockwise2d | none
    scaling_manager=None,
    cp_param_bundle=None,
    return_lse=False,
) -> Tensor | Tuple[Tensor, Tensor]
```

**Source:** `lumen/ops/attention/attention.py`

#### FP8 Grouped GEMM (MoE)

```python
grouped_gemm(
    lhs: Tensor,           # [total_tokens, K]
    rhs: Tensor,           # [num_experts, N, K]
    group_sizes: Tensor,   # [num_experts]
    scaling_type: str = "none",
    ...
) -> Tensor               # [total_tokens, N]

grouped_gemm_wgrad(
    grad_output: Tensor,
    input_tensor: Tensor,
    group_sizes: Tensor,
    scaling_type: str = "none",
) -> Tensor
```

**Source:** `lumen/ops/gemm/grouped_gemm.py`

#### FP8 Activation Store

```python
fused_gated_mlp_fp8_store(
    x: Tensor, w_up: Tensor, w_gate: Tensor, w_down: Tensor,
    activation: str,
    bias_up=None, bias_gate=None, bias_down=None,
    fp8_dtype=torch.float8_e4m3fn,
) -> Tensor

fused_mlp_fp8_store(
    x: Tensor, w_up: Tensor, w_down: Tensor,
    activation: str,
    bias_up=None, bias_down=None,
    fp8_dtype=torch.float8_e4m3fn,
) -> Tensor
```

Forward pass uses BF16 GEMM; activations are quantized to FP8 uint8 + scale before `save_for_backward`. Backward dequantizes and computes gradients in BF16. ~50% activation memory reduction.

**Source:** `lumen/ops/mlp/fused_mlp.py`

---

## 2. Distributed Features

### 2.1 Capability Overview

| Feature | Description | CLI Flag / Entry Point | Status |
|---------|-------------|------------------------|--------|
| Column-Parallel Linear | TP column-parallel linear with FP8 support (all scaling modes) | `LumenColumnParallelLinear` | SUPPORTED |
| Row-Parallel Linear | TP row-parallel linear with FP8 support (all scaling modes) | `LumenRowParallelLinear` | SUPPORTED |
| LayerNorm + Linear (fused) | Fused norm -> quantize -> column-parallel linear in one module | `LumenLayerNormLinear` | SUPPORTED |
| Sequence Parallelism (SP) | Sequence-parallel scatter/gather integrated into TP linear modules | Built-in to TP modules (`sp=True`) | SUPPORTED |
| TP Comm-GEMM Overlap (SDMA) | Async SDMA-based all-gather/reduce-scatter overlapped with GEMM | `--lumen-sdma` | SUPPORTED |
| TP Comm-GEMM Overlap (Pipeline) | NCCL chunk pipelining with double-buffered staging (BF16 only) | `--lumen-tp-comm-overlap-mode pipeline` | PARTIAL |
| SDMA Collectives | mori SDMA for intra-node all-gather / all-reduce on MI300X shared memory | `--lumen-sdma` | LUMEN-ONLY |
| SDMA Cross-Entropy | SDMA-routed all-gather in vocab-parallel cross-entropy | `--lumen-sdma-cross-entropy` | LUMEN-ONLY |
| Context Parallelism -- P2P | Ring send/recv for context parallelism | `cp_comm_type="p2p"` | SUPPORTED |
| Context Parallelism -- A2A | All-to-all context parallelism with optional SDMA | `cp_comm_type="a2a"` | SUPPORTED |
| FSDP | FSDP integration with FP8 -- non-invasive wrap via `apply_fp8_training()` | `apply_fp8_training(model, args)` | SUPPORTED |
| FSDP2 | PyTorch `fully_shard` integration | `apply_fsdp2` / `--fsdp-version 2` | SUPPORTED |
| Amax Reduction (DP) | Cross-rank amax synchronization for FP8 delayed scaling | `reduce_amax` with `dp_group` | SUPPORTED |
| Gradient Accumulation Fusion | Fuse weight gradient accumulation into backward pass | `--lumen-gradient-accumulation-fusion` | SUPPORTED |
| Delay wgrad Compute | Defer weight gradient computation for better comm-compute overlap | `--lumen-delay-wgrad` | SUPPORTED |

### 2.2 API Reference

#### TP Parallel Linear Modules

```python
class LumenColumnParallelLinear(nn.Module):
    """Column-parallel linear with FP8 quantization and optional SDMA comm overlap."""
    def __init__(self, input_size: int, output_size: int, *,
                 bias: bool = True,
                 scaling_type: str = "delayed",
                 tp_group = None,
                 sp: bool = False,
                 sdma_comm: SdmaTpComm | None = None,
                 gradient_accumulation_fusion: bool = False): ...
    def forward(self, x: Tensor) -> Tensor: ...

class LumenRowParallelLinear(nn.Module):
    """Row-parallel linear with FP8 quantization and optional SDMA comm overlap."""
    def __init__(self, input_size: int, output_size: int, *,
                 bias: bool = True,
                 scaling_type: str = "delayed",
                 tp_group = None,
                 sp: bool = False,
                 sdma_comm: SdmaTpComm | None = None,
                 gradient_accumulation_fusion: bool = False): ...
    def forward(self, x: Tensor) -> Tensor: ...
```

**Source:** `lumen/modules/parallel_linear.py`

#### LayerNorm + Linear (Fused)

```python
class LumenLayerNormLinear(nn.Module):
    """Fused LayerNorm -> FP8 quantize -> column-parallel linear."""
    def __init__(self, input_size: int, output_size: int, *,
                 eps: float = 1e-5,
                 scaling_type: str = "delayed",
                 zero_centered_gamma: bool = False,
                 tp_group = None, sp: bool = False): ...
    def forward(self, x: Tensor) -> Tensor: ...
```

**Source:** `lumen/modules/layernorm_linear.py`

#### SDMA Communication

```python
class SdmaAllgather:
    """Wrapper around mori.ccl.AllgatherSdma. Supports uint32/int32/float32/float64."""
    def __init__(self, ctx: SdmaContext): ...
    def __call__(self, input: Tensor, output: Tensor) -> None: ...

class SdmaAllreduce:
    """Wrapper around mori.ccl.AllreduceSdma. Supports uint32/int32/float32/float16/bfloat16."""
    def __init__(self, ctx: SdmaContext): ...
    def __call__(self, input: Tensor, output: Tensor) -> None: ...

def sdma_allgather_max(local_amax: Tensor, ctx: SdmaContext) -> Tensor:
    """All-gather + element-wise max for cross-rank amax sharing."""

class SdmaTpComm:
    """TP-integrated SDMA: async all-gather for column-parallel, reduce-scatter for row-parallel."""
    def __init__(self, tp_group, sdma_ctx: SdmaContext): ...
    def allgather_async(self, input: Tensor) -> Tensor: ...
    def reduce_scatter(self, input: Tensor) -> Tensor: ...
```

**Source:** `lumen/ops/sdma.py`, `lumen/modules/sdma_comm.py`

#### FSDP / FSDP2 Integration

```python
def apply_fp8_training(model: nn.Module, args) -> nn.Module:
    """Non-invasive FP8 enablement for FSDP-wrapped models.
    Patches norms, linears, and wires ScalingManager before FSDP wrap."""

def apply_fsdp2(model: nn.Module, **kwargs) -> nn.Module:
    """Wrap model with PyTorch fully_shard (FSDP2)."""
```

**Source:** `lumen/models/fsdp.py`

#### Context Parallelism

```python
# P2P (ring)
attention(..., cp_param_bundle=CpParamBundle(comm_type="p2p", ...)) -> Tensor

# A2A (all-to-all)
attention(..., cp_param_bundle=CpParamBundle(comm_type="a2a", ...)) -> Tensor
attention_fp8_quant(..., cp_param_bundle=CpParamBundle(comm_type="a2a", ...)) -> Tensor
```

Context parallelism splits the sequence dimension across ranks. P2P uses ring send/recv; A2A uses all-to-all with optional SDMA transport. Both work with BF16 and FP8 attention.

#### Comm-GEMM Overlap (Pipeline)

```python
class PipelinedAllgatherGemm:
    """Double-buffered NCCL all-gather overlapped with chunked GEMM (BF16)."""
    def __init__(self, tp_group, num_chunks: int = 4): ...
    def __call__(self, input: Tensor, weight: Tensor) -> Tensor: ...

class PipelinedGemmReduceScatter:
    """Double-buffered GEMM overlapped with NCCL reduce-scatter (BF16)."""
    def __init__(self, tp_group, num_chunks: int = 4): ...
    def __call__(self, input: Tensor, weight: Tensor) -> Tensor: ...
```

**CLI:** `--lumen-tp-comm-overlap-mode pipeline`

**Limitation:** BF16 only; FP8 pipeline overlap is pending.

---

## 3. LoRA Features

### 3.1 Capability Overview

| Feature | Description | Entry Point | Status |
|---------|-------------|-------------|--------|
| LoRA (FSDP path) | HuggingFace PEFT LoRA adapters with FSDP/FSDP2 distributed training | `apply_lora(model, args)` when `lora_rank > 0` | SUPPORTED |
| LoRA (Megatron path) | Megatron-Core LoraAdapter integration for TP-aware LoRA | Via Megatron config | SUPPORTED |
| LoRA + FP8 | LoRA adapters on top of FP8-quantized base model | `quant.enable(model)` then `apply_lora(model, args)` | SUPPORTED |

### 3.2 API Reference

#### LoRA Application (FSDP Path)

```python
def apply_lora(model: nn.Module, args) -> nn.Module:
    """Apply HuggingFace PEFT LoRA adapters when args.lora_rank > 0.
    Compatible with both FSDP and FSDP2 wrapping."""
```

**CLI:** `--lora-rank N`

**Typical usage with FP8:**

```python
import atomrl.lumen
from atomrl.lumen.quantize import QuantConfig

model = build_model()

# Step 1: Enable FP8 on the base model
atomrl.lumen.quant.enable(model, QuantConfig(scaling_type="blockwise"))

# Step 2: Apply LoRA adapters (LoRA weights stay in BF16)
apply_lora(model, args)  # args.lora_rank = 32

# Step 3: Wrap with FSDP
model = FSDP(model, ...)
```

## 4. Cross-Cutting Features

Features that span multiple domains and are used across quantization, distributed, and LoRA workflows.

| Feature | Module / Function | Description | CLI Flag | Status |
|---------|-------------------|-------------|----------|--------|
| HIP Graphs | `LumenGraphedCallable`, `LumenGraphedModule` | Graph-capture training steps for reduced kernel launch overhead | -- | SUPPORTED |
| CPU Offload | `CPUOffloadManager`, `get_cpu_offload_context` | Async GPU->CPU->GPU activation offload via `saved_tensors_hooks` | `--lumen-cpu-offload` | SUPPORTED |
| TE Checkpointing | `lumen_checkpoint`, `_FP8ScalingContext` | Save/restore FP8 scaling state across activation checkpoints | -- | SUPPORTED |
| Fused RoPE | `apply_rotary_pos_emb`, `fused_rope` | AITER Triton fused rotary position embeddings (text, vision, video) | `--lumen-fused-rope` | SUPPORTED |
| Fused MLP | `LumenFusedMLP`, `LumenGatedMLP` | Fused gated/ungated MLP via AITER Triton (BF16 GEMM + fused activation) | `--lumen-fused-mlp` | SUPPORTED |
| Fused MoE Routing | `fused_topk`, `fused_permute`, `fused_unpermute` | AITER ASM/HIP fused MoE gating, token sort, scatter-back | -- | SUPPORTED |
| Parallel Cross-Entropy | `lumen_parallel_cross_entropy` | Vocab-parallel cross-entropy with Triton kernel + optional SDMA | -- | SUPPORTED |

### 4.1 API Reference

#### Fused RoPE

```python
apply_rotary_pos_emb(x, cos, sin, interleaved=False) -> Tensor
    # x: [B, H, S, D]

fused_rope(q, k, cos, sin, interleaved=False) -> Tuple[Tensor, Tensor]
    # Fused Q+K RoPE in one call

apply_rotary_pos_emb_2d(x, cos_h, sin_h, cos_w, sin_w, img_height, img_width) -> Tensor
    # 2D RoPE for vision models

apply_rotary_pos_emb_3d(x, grid_sizes, freqs, sp_size=1, sp_rank=0) -> Tensor
    # 3D RoPE for video models
```

**Source:** `lumen/ops/rope.py`

#### Fused MLP

```python
fused_gated_mlp(x, w_up, w_gate, w_down, activation, ...) -> Tensor
fused_mlp(x, w_up, w_down, activation, ...) -> Tensor
```

**Supported activations:** `swiglu`, `geglu`, `reglu`, `gelu`, `relu`, `silu`

**Source:** `lumen/ops/mlp/fused_mlp.py`

#### Fused MoE Routing

```python
fused_topk(logits, k, softmax_first=True) -> Tuple[Tensor, Tensor]
    # Returns (weights: float32, indices: int64) of shape [num_tokens, k]

fused_permute(tokens, indices, weights, num_experts, block_size=32) -> Tuple
    # Returns (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)

fused_unpermute(expert_output, sort_order, num_tokens, k) -> Tensor
    # Scatter expert outputs back to original token order
```

**Source:** `lumen/ops/moe/fused_routing.py`

#### Parallel Cross-Entropy

```python
lumen_parallel_cross_entropy(
    logits: Tensor,       # [B, SQ, V]
    targets: Tensor,      # [B, SQ]
    *,
    label_smoothing: float = 0.0,
    use_sdma: bool = False,
    dist_group = None,
    is_cg_capturable: bool = False,
) -> Tensor
```

**Source:** `lumen/ops/cross_entropy.py`

#### CPU Offload

```python
class CPUOffloadManager:
    """Async GPU->CPU->GPU activation offload via torch.autograd.graph.saved_tensors_hooks."""
    def __init__(self, enabled: bool = True): ...
    def __enter__(self): ...
    def __exit__(self, *args): ...

def get_cpu_offload_context(enabled: bool = True) -> ContextManager:
    """Returns a context manager for CPU activation offload."""
```

---

## 5. CLI Flag Quick Reference

A flat, searchable table of all Lumen CLI flags.

| CLI Flag | Feature | Category | Values / Notes |
|----------|---------|----------|----------------|
| `--linear-fp8` | FP8 Linear GEMM (forward + dgrad) | Quantization | Enable FP8 linear |
| `--linear-fp4` | MXFP4 Linear GEMM (forward + dgrad + wgrad) | Quantization | Dedicated MXFP4 enable switch; mutually exclusive with `--linear-fp8` |
| `--linear-fp8-wgrad` | FP8 weight gradient GEMM | Quantization | MXFP8 only |
| `--linear-fp8-scaling` | FP8 scaling type for linear | Quantization | `delayed\|dynamic\|per_token\|blockwise\|blockwise2d\|mxfp8` |
| `--lumen-fp8-attn` | FP8 attention | Quantization | `none\|dpa\|mha` |
| `--lumen-fp8-activation-store` | FP8 activation storage in MLP backward | Quantization | Boolean |
| `--lumen-fp8-param-gather` | FP8 parameter storage + lazy re-quant | Quantization | Boolean |
| `--lumen-norm` | Fused norm (RMSNorm/LayerNorm) via AITER | Quantization | Boolean |
| `--lumen-fused-mlp` | Fused MLP via AITER Triton | Cross-Cutting | Boolean |
| `--lumen-fused-rope` | Fused RoPE via AITER Triton | Cross-Cutting | Boolean |
| `--lumen-sdma` | SDMA for TP communication | Distributed | Boolean |
| `--lumen-sdma-cross-entropy` | SDMA all-gather in cross-entropy | Distributed | Boolean |
| `--lumen-tp-comm-overlap-mode` | TP comm-GEMM overlap mode | Distributed | `sdma\|pipeline` |
| `--lumen-gradient-accumulation-fusion` | Fuse wgrad accumulation | Distributed | Boolean |
| `--lumen-delay-wgrad` | Defer weight gradient computation | Distributed | Boolean |
| `--lumen-cpu-offload` | CPU activation offload | Cross-Cutting | Boolean |
| `--use-sdma` | Enable mori SDMA (alias) | Distributed | Boolean |
| `--moe-grouped-gemm` | Grouped GEMM for MoE | Quantization | Boolean |
| `--lora-rank` | LoRA adapter rank | LoRA | Integer (recommended >= 32 for RL) |
| `--fsdp-version` | FSDP version | Distributed | `1\|2` |
| `--grad-quant-type` | Gradient quantization type | Quantization | `fp8\|mxfp8` |
| `--warmup-steps` | BF16 warmup steps before FP8 + scaling reset | Quantization | Integer |
