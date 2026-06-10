# MiniMax-M2: fuse AllReduce + RMSNorm + per-group quant for `input_layernorm`

Date: 2026-06-10

## Goal

In `MiniMaxM2DecoderLayer`, when `input_layernorm` already performs a fused
AllReduce + RMSNorm (TP > 1, `layer_idx > 0`) **and** the downstream
`self_attn.qkv_proj` requires per-group-quantized FP8 input
(`quant_type == per_1x128`, `params_dtype == fp8`), fuse the activation
quantization into the same kernel. The fused kernel then emits
`(fp8_activation, scale)`, which `qkv_proj` consumes directly, skipping its
internal quantization step.

This removes a separate per-group quant kernel launch per decoder layer on the
TP critical path.

## Scope

- Use the **per-group** (`per_1x128`) fused variant only
  (`tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group`). Other quant
  types fall back to today's behavior.
- Activate **only** when `tp_size > 1` and `residual is not None`
  (i.e. `layer_idx > 0`), matching the existing `input_layernorm`
  `fused_allreduce` gating. No new env var; auto-detected from `qkv_proj`'s
  quant config, with the decoder layer deciding and passing a flag to `RMSNorm`.
- Out of scope: the `post_attention_layernorm → block_sparse_moe` path (MoE gate
  runs in fp32 and `experts` quantize internally); MTP / spec-decode head;
  models other than MiniMax-M2.

## Key constraint: scale layout

The current aiter per-group fused kernel writes the scale **row-major**
(`scale_out[tidx * num_groups + group_id]`, shape `(M, num_groups)`). This is
consumed correctly by the non-preshuffle `gemm_a8w8_blockscale`, but the
preshuffle GEMM (`gemm_a8w8_blockscale_preshuffle`, used when
`ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1`) expects **column-major**
`(num_groups, M).view(M, num_groups)`.

Resolution (decided): extend the aiter kernel to support an optional
`transpose_scale` flag that writes the scale column-major, so the fusion works
in both preshuffle and non-preshuffle modes. `transpose_scale=False` is the
default everywhere and reproduces today's exact behavior.

## Part A — aiter kernel change (repo: `/home/gyu_qle/ganyi/aiter`)

Add an optional `transpose_scale` parameter (default `false`) threaded through
the entire per-group AR+RMSNorm+quant call chain. When `true`, the per-group
scale is written column-major: `scale_out[group_id * m + tidx]` instead of
`scale_out[tidx * num_groups + group_id]`.

Files / functions to edit:

1. `csrc/include/custom_all_reduce.cuh`
   - `ar_fusion_epilogue_per_group` (~L1665): add params `int m, bool transpose_scale`;
     change the scale-write index (~L1734) to
     `scale_out[transpose_scale ? (group_id * m + tidx) : (tidx * num_groups + group_id)] = scale;`
   - All **three** per-group kernels — `allreduce_fusion_kernel_1stage_per_group`
     (~L1850), `allreduce_fusion_kernel_2stage_per_group` (~L2556),
     `allreduce_fusion_kernel_split_per_group` — add `int m, bool transpose_scale`
     params and forward to the epilogue. `m = size / hidden_dim` (token count);
     `tidx` is the global token index already tracked per kernel.
   - All **three** launchers — `allreduce_fusion_kernel_{1stage,2stage,split}_per_group_launcher`
     — add `bool transpose_scale`, pass the already-computed `m` and the flag.
   - `dispatchFusedAllReduceRMSNormQuantPerGroup` (~L3884): add
     `bool transpose_scale=false`; forward through the `DISPATCH_AR_FUSION_PG_KERNEL`
     macro to all three launchers.

2. `csrc/kernels/custom_all_reduce.cu`
   - `fused_allreduce_rmsnorm_quant_per_group` (~L623): add `bool transpose_scale`
     param; forward to both bf16 and fp16 `dispatchFusedAllReduceRMSNormQuantPerGroup`
     calls.

3. Python bindings (default `transpose_scale=False` so existing callers are
   unaffected):
   - `aiter/dist/device_communicators/custom_all_reduce.py`:
     `fused_ar_rms_per_group_quant` and `custom_fused_ar_rms_per_group_quant` —
     add `transpose_scale`; when `True`, allocate `scale_out` as `(num_groups, M)`
     and `.view(M, num_groups)` (column-major storage), and pass the flag to the
     C++ op `ops.fused_allreduce_rmsnorm_quant_per_group`.
   - `aiter/dist/device_communicators/communicator_cuda.py`:
     `fused_allreduce_rmsnorm_quant_per_group` and `fused_allreduce_rmsnorm_quant`
     — plumb `transpose_scale`.
   - `aiter/dist/parallel_state.py`: `fused_allreduce_rmsnorm_quant` and
     `fused_allreduce_rmsnorm_quant_per_group` — add `transpose_scale` kwarg.
   - `aiter/dist/communication_op.py`:
     `tensor_model_parallel_fused_allreduce_rmsnorm_quant` and
     `tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group` — add
     `transpose_scale` kwarg.

Build: aiter JIT recompiles the `custom_all_reduce` module on next import. No
manual AOT build is triggered by this work.

**Invariant:** with `transpose_scale=False` everywhere by default, every existing
caller produces byte-identical output to today.

## Part B — ATOM `RMSNorm` (`atom/model_ops/layernorm.py`)

Add a combined branch in `RMSNorm.forward`, taken when:

```
self.fused_allreduce and self.tp_size > 1 and self.use_fused_quant
and residual is not None and self.quant_type.value == _QV_PER_1X128
```

Behavior:

```python
from aiter.dist.communication_op import (
    tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group,
)
fp8, residual_out, scale = tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(
    x.contiguous(), residual, self.weight, self.eps,
    group_size=128, transpose_scale=self._aiter_transpose_scale,
)
return (fp8, scale), residual_out
```

- Reuse the existing `self._aiter_transpose_scale` (already resolved at init as
  `per_1x128 and ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE`) so the emitted scale
  layout matches whatever GEMM `qkv_proj` runs.
- This branch is placed **before** the existing plain `fused_allreduce` branch.
  When `use_fused_quant` is False, the existing bf16 `fused_allreduce` path is
  unchanged.
- `import` of the new symbol is added alongside the existing
  `tensor_model_parallel_fused_allreduce_rmsnorm` import.

## Part C — `atom/models/minimax_m2.py`

1. `MiniMaxM2DecoderLayer.__init__`: after `self.self_attn` is constructed,
   detect the qkv quant requirement and pass it to `input_layernorm`:

   ```python
   qkv = self.self_attn.qkv_proj
   qkv_group_quant = (
       qkv.quant_type.value == QuantType.per_1x128.value
       and qkv.params_dtype == dtypes.fp8
   )
   fuse_in = ENABLE_ALLREDUCE_RMSNORM_FUSION and self.layer_idx > 0
   self.input_layernorm = RMSNorm(
       config.hidden_size,
       eps=config.rms_norm_eps,
       fused_allreduce=fuse_in,
       fused_quant=fuse_in and qkv_group_quant,
       quant_config=quant_config,
       prefix=f"{prefix}.self_attn.qkv_proj",
   )
   ```

   - Passing `quant_config` + the **qkv_proj** prefix lets `RMSNorm` resolve
     `quant_type=per_1x128` and set `_aiter_transpose_scale` for the consumer's
     GEMM. (`fused_quant` is what enables the combined kernel; `quant_type`
     resolved from the prefix selects per-group.)
   - `post_attention_layernorm` is unchanged.
   - Requires importing `QuantType` (from `aiter`) and `dtypes` (from
     `aiter` / `aiter.utility.dtypes`) at the top of the model file.

2. `MiniMaxM2DecoderLayer.forward`: thread the optional scale through:

   ```python
   if residual is None:
       residual = hidden_states
       hidden_states = self.input_layernorm(hidden_states)
       x_scale = None
   else:
       hidden_states, residual = self.input_layernorm(hidden_states, residual)
       x_scale = None
       if isinstance(hidden_states, tuple):
           hidden_states, x_scale = hidden_states
   hidden_states = self.self_attn(
       positions=positions, hidden_states=hidden_states,
       hidden_states_scale=x_scale,
   )
   ```

   - `layer_idx == 0` (residual is None) path is unchanged: RMSNorm returns a
     plain bf16 tensor, `x_scale` stays None.

3. `MiniMaxM2Attention.forward`: add `hidden_states_scale: Optional[torch.Tensor] = None`
   and forward it: `qkv = self.qkv_proj(hidden_states, hidden_states_scale)`.
   `LinearBase.forward(x, x_scale)` already skips internal quant when `x_scale`
   is provided and dispatches to the per_1x128 GEMM.

## Data flow (fused path, layer_idx > 0, TP > 1, qkv per_1x128 fp8)

```
prev layer MoE out (un-all-reduced) ─┐
                                     ▼
input_layernorm:  AR + RMSNorm + per_group_quant  (single aiter kernel)
   → ((fp8_x, scale), residual_out)
                                     ▼
self_attn.qkv_proj(fp8_x, scale)  → gemm_a8w8_blockscale[_preshuffle]
   (no internal quant kernel)
                                     ▼
... attn ... o_proj (reduce_results=False) ...
                                     ▼
post_attention_layernorm: AR + RMSNorm (bf16, unchanged)
```

## Error handling / edge cases

- `tp_size == 1`: combined branch not taken (gated on `tp_size > 1`); falls
  through to existing non-AR logic → `qkv_proj` quantizes internally as today.
- `layer_idx == 0` / `residual is None`: combined branch not taken; bf16 output,
  `qkv_proj` quantizes internally.
- `qkv_proj` not per_1x128 fp8 (e.g. bf16 or per_1x32): `fused_quant=False`,
  `input_layernorm` stays on the plain bf16 `fused_allreduce` path.
- Online quant: if qkv_proj is online-requantized to per_1x128, `RMSNorm`'s
  existing `online_quantize_activation` hook already realigns its emitted
  `quant_type` / `_aiter_transpose_scale`; the combined branch reads the
  post-load state.

## Testing

- aiter: extend `op_tests/multigpu_tests/test_fused_ar_rms_per_group_quant.py`
  to sweep `transpose_scale=True`, reading the column-major scale and comparing
  the dequantized result against the existing row-major reference.
- ATOM: smoke-test `RMSNorm` combined-branch output shape `((x, scale), residual)`
  when the flags are set (AITER mocked, per `tests/` conventions).
- End-to-end: `simple_inference.py` smoke + `lm_eval` gsm8k on MiniMax-M2 with
  TP (e.g. `-tp 8`), comparing accuracy against the non-fused baseline, per the
  project accuracy workflow. Run with both
  `ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE` on and off to validate both scale
  layouts.
```

