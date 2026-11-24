# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Megatron-LM IMPORT-phase compatibility patches (runtime monkey-patches).

Registered under :class:`~atomrl.lumen.patches.PatchPhase`.IMPORT. Loaded via
``import atomrl.lumen.patches.runtime`` (or the backward-compatible
``atomrl.lumen.models.megatron_patches`` re-export shim).
"""

import math
import os

import torch
from torch.autograd.function import once_differentiable

_FUSED_SWIGLU_QUANT = os.environ.get("LUMEN_FUSED_SWIGLU_QUANT", "0") == "1"


# ── 1. FusedLayerNorm → Lumen RMSNorm / LayerNorm ──────────────────────────


def install_fused_layer_norm():
    """Replace Megatron's ``FusedLayerNorm`` with a Lumen-compatible wrapper.

    Must run **before** ``GPTModel`` or ``TransformerBlock`` is imported,
    because those modules capture ``FusedLayerNorm`` at class-definition time.
    """
    from megatron.core.fusions import fused_layer_norm as _fln_mod

    from atomrl.lumen.ops.normalization import LumenLayerNorm, LumenRMSNorm

    class _FusedLayerNormCompat(torch.nn.Module):
        def __new__(cls, config, hidden_size, eps=1e-6, **kwargs):
            return object.__new__(cls)

        def __init__(self, config, hidden_size, eps=1e-6, **kwargs):
            super().__init__()
            norm_type = getattr(config, "normalization", "LayerNorm")
            if norm_type == "RMSNorm":
                self._norm = LumenRMSNorm(hidden_size, eps=eps)
            else:
                self._norm = LumenLayerNorm(hidden_size, eps=eps)
            self.weight = self._norm.weight

        def forward(self, x):
            return self._norm(x)

    _FusedLayerNormCompat.__name__ = "FusedLayerNorm"
    _fln_mod.FusedLayerNorm = _FusedLayerNormCompat


# ── 2. LanguageModule checkpoint save guard ─────────────────────────────────


def install_language_module_checkpoint_guard():
    """Guard ``LanguageModule.sharded_state_dict`` against missing
    ``output_layer.weight`` when LoRA wraps that layer."""
    from megatron.core.models.common.language_module import language_module as _lm_mod

    def _patched_sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        assert not sharded_offsets, "Unexpected sharded offsets"
        from megatron.core.transformer.module import MegatronModule

        sharded_state_dict = MegatronModule.sharded_state_dict(
            self, prefix, sharded_offsets, metadata
        )
        first_stage_word_emb_key = f"{prefix}embedding.word_embeddings.weight"
        output_layer_weight_key = f"{prefix}output_layer.weight"
        output_layer_bias_key = f"{prefix}output_layer.bias"

        if self.share_embeddings_and_output_weights:
            self.tie_embeddings_and_output_weights_state_dict(
                sharded_state_dict, output_layer_weight_key, first_stage_word_emb_key
            )
        elif self.post_process and output_layer_weight_key in sharded_state_dict:
            sharded_state_dict[output_layer_weight_key].allow_shape_mismatch = True

        if self.post_process and output_layer_bias_key in sharded_state_dict:
            sharded_state_dict[output_layer_bias_key].allow_shape_mismatch = True

        return sharded_state_dict

    _lm_mod.LanguageModule.sharded_state_dict = _patched_sharded_state_dict


# ── 3. Memory-mapped checkpoint loading ─────────────────────────────────────


class _TorchLoadProxy:
    """Module-local torch view with an isolated ``load`` implementation."""

    def __init__(self, torch_module, load):
        self._torch_module = torch_module
        self.load = load

    def __getattr__(self, name):
        return getattr(self._torch_module, name)


def install_mmap_checkpoint():
    """Inject ``mmap=True`` into Megatron's ``torch.load`` calls.

    Prevents 8 ranks from each loading a full checkpoint into CPU RAM.
    """
    import megatron.training.checkpointing as _ckpt_mod

    if getattr(_ckpt_mod, "_lumen_mmap_patched", False):
        return

    _orig_torch_load = torch.load

    def _mmap_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        if args and isinstance(args[0], (str, os.PathLike)):
            kwargs.setdefault("mmap", True)
        return _orig_torch_load(*args, **kwargs)

    for mod_name in (
        "megatron.training.checkpointing",
        "megatron.core.dist_checkpointing.strategies.common",
    ):
        try:
            mod = __import__(mod_name, fromlist=["torch"])
        except ImportError:
            continue
        if getattr(mod, "_lumen_mmap_patched", False):
            continue
        mod.torch = _TorchLoadProxy(torch, _mmap_torch_load)
        mod._lumen_mmap_patched = True

    _ckpt_mod._lumen_mmap_patched = True


# ── 4. requires_grad fix for LoRA + activation checkpointing ───────────────


def install_requires_grad_fix():
    """Force ``hidden_states.requires_grad_(True)`` after ``make_viewless_tensor``.

    With LoRA + activation checkpointing, frozen embeddings produce
    ``hidden_states.requires_grad=False``, breaking the backward chain
    (grad_norm=0).
    """
    from megatron.core.transformer import transformer_block as _tb_mod
    from megatron.core.utils import make_viewless_tensor as _orig_mvt

    if getattr(_tb_mod, "_lumen_requires_grad_patched", False):
        return

    def _make_viewless_with_grad(inp, requires_grad, keep_graph):
        out = _orig_mvt(inp=inp, requires_grad=requires_grad, keep_graph=keep_graph)
        if not out.requires_grad and torch.is_grad_enabled():
            out = out.detach().requires_grad_(True)
        return out

    _tb_mod.make_viewless_tensor = _make_viewless_with_grad
    _tb_mod._lumen_requires_grad_patched = True


# ── 5. SwiGLU FP8 dtype + chunked backward ─────────────────────────────────


def install_swiglu_fp8():
    """Replace Megatron's ``SwiGLUFunction`` with a version that:

    1. Uses ``float8_e4m3fnuz`` on MI300X (instead of hard-coded ``float8_e4m3fn``).
    2. Processes the backward in 1024-row chunks when FP8 input store is active,
       reducing peak transient memory from 896 MiB to 112 MiB for Llama-70B.
    3. When ``LUMEN_FUSED_SWIGLU_QUANT=1``, runs AITER's fused SiLU+mul+FP8 quant
       after SwiGLU forward so the next FP8 GEMM can skip separate activation quant.
    """
    try:
        from megatron.core.fusions import fused_bias_swiglu as _swiglu_mod
    except ImportError:
        return

    if getattr(_swiglu_mod, "_lumen_fp8_dtype_patched", False):
        return

    from atomrl.lumen.quantize.config import _get_float8_e4m3

    _fp8_e4m3 = _get_float8_e4m3()

    class _PatchedSwiGLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, fp8_input_store=False, cpu_offload_input=False):
            if fp8_input_store:
                input_for_backward = input.to(_fp8_e4m3)
            else:
                input_for_backward = input
            if cpu_offload_input:
                input_for_backward.activation_offloading = True
            ctx.save_for_backward(input_for_backward)
            ctx.ori_input_dtype = input.dtype
            ctx.fp8_input_store = fp8_input_store
            bf16_out = _swiglu_mod.swiglu(input)
            if _FUSED_SWIGLU_QUANT:
                try:
                    from atomrl.lumen.models._swiglu_fp8_fuse import (
                        try_fused_swiglu_fp8,
                    )

                    try_fused_swiglu_fp8(input, bf16_out)
                except (AttributeError, RuntimeError):
                    pass
            return bf16_out

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            input_saved = ctx.saved_tensors[0]
            if not ctx.fp8_input_store:
                return _swiglu_mod.swiglu_back(grad_output, input_saved), None, None

            dtype = ctx.ori_input_dtype
            M = input_saved.shape[0]
            chunk_sz = min(1024, M)
            n_chunks = (M + chunk_sz - 1) // chunk_sz
            result = torch.empty(
                M,
                input_saved.shape[1],
                dtype=dtype,
                device=input_saved.device,
            )
            for i in range(n_chunks):
                s = i * chunk_sz
                e = min(s + chunk_sz, M)
                inp_chunk = input_saved[s:e].to(dtype)
                grad_chunk = grad_output[s:e]
                result[s:e] = _swiglu_mod.swiglu_back(grad_chunk, inp_chunk)
                del inp_chunk
            return result, None, None

    _swiglu_mod.SwiGLUFunction = _PatchedSwiGLUFunction
    _swiglu_mod._lumen_fp8_dtype_patched = True


# ── 5b. Fused SwiGLU activation via Triton ────────────────────────────────

_FUSED_SWIGLU = os.environ.get("LUMEN_FUSED_SWIGLU", "0") == "1"


def install_fused_swiglu_triton():
    """Replace Megatron's ``swiglu`` / ``swiglu_back`` with single-kernel
    Triton fused implementations.

    The original ``@jit_fuser`` implementations expand into 6-8 separate
    elementwise kernels per call (chunk, silu, sigmoid, mul, add, cat).
    The Triton kernels handle the full forward and backward in one launch
    each, saving ~400-600ms/step at LLaMA-70B scale.

    Controlled by ``LUMEN_FUSED_SWIGLU=1``.
    """
    if not _FUSED_SWIGLU:
        return

    try:
        from megatron.core.fusions import fused_bias_swiglu as _swiglu_mod
    except ImportError:
        return

    if getattr(_swiglu_mod, "_lumen_triton_swiglu_patched", False):
        return

    from atomrl.lumen.ops.fused_swiglu import fused_swiglu, fused_swiglu_backward

    _swiglu_mod.swiglu = fused_swiglu
    _swiglu_mod.swiglu_back = fused_swiglu_backward
    _swiglu_mod._lumen_triton_swiglu_patched = True


# ── 5c. MLP FP8 activation storage for SwiGLU ─────────────────────────────

_MLP_FP8_STORE = os.environ.get("LUMEN_MLP_FP8_STORE", "0") == "1"


def install_mlp_fp8_store():
    """Replace the decomposed SwiGLU in ``MLP.forward`` with a custom autograd
    function that stores the fc1 output in FP8 (1 byte/element) instead of BF16
    (2 bytes/element), recomputing the SwiGLU during backward.

    Memory savings per non-recomputed layer (Llama2-70B, seq=8192, MBS=1):
      Before: ~1.3 GB for SwiGLU intermediates in BF16
      After:  ~0.06 GB for fc1 input in FP8 + ~0.22 GB for fc2 input in FP8
      Saving: ~1.0 GB/layer → ~59 GB for 59 non-recomputed layers (ACL=21)

    Controlled by ``LUMEN_MLP_FP8_STORE=1``.

    Only affects the non-fused GLU path (``bias_swiglu_fusion=False``, which is
    what Lumen uses).  The fused bias_swiglu path already has its own FP8 store
    via ``activation_func_fp8_input_store``.
    """
    if not _MLP_FP8_STORE:
        return

    try:
        from megatron.core.transformer import mlp as _mlp_mod
    except ImportError:
        return

    if getattr(_mlp_mod, "_lumen_mlp_fp8_store_patched", False):
        return

    from atomrl.lumen.quantize.config import _get_float8_e4m3

    _fp8_e4m3 = _get_float8_e4m3()
    _MLP = _mlp_mod.MLP
    _orig_forward = _MLP.forward

    from torch.nn.functional import sigmoid as _F_sigmoid
    from torch.nn.functional import silu as _F_silu

    from atomrl.lumen.ops.quantize.linear import _fp8_store_activation

    class _SwiGLU_FP8Store(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fc1_output, activation_func, clamp_value, glu_linear_offset):
            x_gate, x_linear = torch.chunk(fc1_output, 2, dim=-1)
            if clamp_value is not None:
                x_gate = x_gate.clamp(min=None, max=clamp_value)
                x_linear = x_linear.clamp(min=-clamp_value, max=clamp_value)
            activated = activation_func(x_gate) * (x_linear + glu_linear_offset)

            fc1_flat = fc1_output.reshape(-1, fc1_output.shape[-1]).contiguous()
            # Use _fp8_store_activation which dispatches via fast quant path
            # (CK single kernel for E4M3) instead of the 2-kernel Triton
            # dynamic_per_tensor_quant_fp8_i8.
            qx, scale_out = _fp8_store_activation(fc1_flat, _fp8_e4m3)
            fc1_fp8 = qx.view(torch.uint8)
            inv_scale = scale_out

            ctx.save_for_backward(fc1_fp8, inv_scale)
            ctx._orig_shape = fc1_output.shape
            ctx._clamp_value = clamp_value
            ctx._glu_linear_offset = glu_linear_offset
            ctx._activation_func = activation_func
            ctx._is_silu = (
                activation_func is _F_silu
                or activation_func is torch.nn.functional.silu
            )
            return activated

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            fc1_fp8, inv_scale = ctx.saved_tensors
            clamp_value = ctx._clamp_value
            activation_func = ctx._activation_func
            glu_linear_offset = ctx._glu_linear_offset
            orig_shape = ctx._orig_shape

            fc1_recon = (fc1_fp8.view(_fp8_e4m3).to(torch.float32) * inv_scale).to(
                grad_output.dtype
            )
            fc1_recon = fc1_recon.reshape(orig_shape)
            x_gate, x_linear = torch.chunk(fc1_recon, 2, dim=-1)
            if clamp_value is not None:
                x_gate = x_gate.clamp(min=None, max=clamp_value)
                x_linear = x_linear.clamp(min=-clamp_value, max=clamp_value)

            if ctx._is_silu:
                sig = _F_sigmoid(x_gate)
                gate_activated = x_gate * sig
                act_grad = sig * (1.0 + x_gate * (1.0 - sig))
            else:
                with torch.enable_grad():
                    x_gate_g = x_gate.detach().requires_grad_(True)
                    act_val = activation_func(x_gate_g)
                    act_grad = torch.autograd.grad(
                        act_val,
                        x_gate_g,
                        torch.ones_like(act_val),
                        retain_graph=False,
                    )[0]
                gate_activated = activation_func(x_gate)

            up_val = x_linear + glu_linear_offset

            grad_fc1 = torch.empty_like(fc1_recon)
            N = grad_fc1.shape[-1] // 2
            grad_fc1[..., :N] = grad_output * up_val * act_grad
            grad_fc1[..., N:] = grad_output * gate_activated
            return grad_fc1, None, None, None

    def _mlp_forward_with_fp8_store(self, hidden_states, per_token_scale=None):
        if (
            not self.training
            or self.config.bias_activation_fusion
            or self.config.use_te_activation_func
            or not self.config.gated_linear_unit
        ):
            return _orig_forward(self, hidden_states, per_token_scale)

        from megatron.core.utils import nvtx_range_pop, nvtx_range_push

        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")

        nvtx_range_push(suffix="activation")
        if bias_parallel is not None:
            intermediate_parallel = intermediate_parallel + bias_parallel

        intermediate_parallel = _SwiGLU_FP8Store.apply(
            intermediate_parallel,
            self.config.activation_func,
            self.config.activation_func_clamp_value,
            self.config.glu_linear_offset,
        )

        if per_token_scale is not None:
            original_dtype = intermediate_parallel.dtype
            intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(
                -1
            )
            intermediate_parallel = intermediate_parallel.to(original_dtype)
        nvtx_range_pop(suffix="activation")

        nvtx_range_push(suffix="linear_fc2")
        output, output_bias = self.linear_fc2(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc2")
        return output, output_bias

    _MLP.forward = _mlp_forward_with_fp8_store
    _mlp_mod._lumen_mlp_fp8_store_patched = True


# ── 6. MLP-only recompute ──────────────────────────────────────────────────


def install_mlp_recompute():
    """Wrap the MLP sublayer of non-recomputed ``TransformerLayer`` with
    ``tensor_parallel.checkpoint`` when ``LUMEN_MLP_RECOMPUTE=1``.

    This frees intermediate BF16 tensors (FC1_out, SwiGLU_out) that
    would otherwise stay alive in the autograd graph (~1.3 GiB/layer).

    ``LUMEN_MLP_RECOMPUTE_LAYERS`` controls how many non-recomputed layers
    get MLP-only recompute (default: all).  Set to e.g. ``20`` to only
    recompute MLP for the first 20 non-recomputed layers, balancing memory
    savings (~1.3 GiB/layer) vs compute overhead (~30ms/layer).
    """
    import os

    try:
        from megatron.core.transformer import transformer_layer as _tl_mod
    except ImportError:
        return

    if getattr(_tl_mod, "_lumen_mlp_recompute_patched", False):
        return

    try:
        from megatron.core.transformer.identity_op import IdentityOp as _IdentityOp
    except ImportError:
        _IdentityOp = type(None)

    _OrigTransformerLayer = _tl_mod.TransformerLayer
    _orig_forward = _OrigTransformerLayer.forward

    _mlp_recompute_count = [0]
    _max_mlp_recompute = int(os.environ.get("LUMEN_MLP_RECOMPUTE_LAYERS", "9999"))

    def _patched_forward(self, *args, **kwargs):
        if os.environ.get("LUMEN_MLP_RECOMPUTE", "0") != "1" or not self.training:
            return _orig_forward(self, *args, **kwargs)

        if not hasattr(self, "_lumen_mlp_recompute_installed"):
            self._lumen_mlp_recompute_installed = True
            should_recompute = (
                hasattr(self, "mlp")
                and not isinstance(self.mlp, _IdentityOp)
                and not getattr(self, "recompute_mlp", False)
                and _mlp_recompute_count[0] < _max_mlp_recompute
            )
            if should_recompute:
                _orig_mlp_fwd = self.mlp.forward
                _mlp_recompute_count[0] += 1

                def _recompute_mlp_forward(hidden_states, *a, **kw):
                    from megatron.core import tensor_parallel

                    return tensor_parallel.checkpoint(
                        _orig_mlp_fwd, False, hidden_states, *a, **kw
                    )

                self.mlp.forward = _recompute_mlp_forward

        return _orig_forward(self, *args, **kwargs)

    _OrigTransformerLayer.forward = _patched_forward
    _tl_mod._lumen_mlp_recompute_patched = True


# ── 7. LoRA checkpoint key remapping ────────────────────────────────────────


def remap_lora_state_dict(module, state_dict):
    """Remap checkpoint keys for LoRA-wrapped layers.

    Transforms ``decoder.layers.N.linear_qkv.weight`` →
    ``decoder.layers.N.linear_qkv.base_layer.weight`` to match the model's
    state dict when LoRA wrapping adds a ``.base_layer.`` level.

    Returns the (possibly modified) state dict.
    """
    import re

    inner = module
    while hasattr(inner, "module"):
        inner = inner.module

    inner_keys = set(inner.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if len(ckpt_keys.intersection(inner_keys)) >= len(ckpt_keys):
        return state_dict

    lora_parents = set()
    for ik in inner_keys:
        m = re.match(r"(.+)\.base_layer\.weight$", ik)
        if m:
            lora_parents.add(m.group(1))
    if not lora_parents:
        return state_dict

    new_sd = {}
    mapped = 0
    for ck, cv in state_dict.items():
        parts = ck.rsplit(".", 1)
        if len(parts) == 2 and parts[0] in lora_parents:
            new_sd[f"{parts[0]}.base_layer.{parts[1]}"] = cv
            mapped += 1
        else:
            new_sd[ck] = cv

    if mapped > 0:
        try:
            from megatron.training import print_rank_0

            print_rank_0(f"> Lumen: remapped {mapped} LoRA base_layer checkpoint keys")
        except ImportError:
            pass
    return new_sd


# ── Fused RoPE (apex → Megatron rope_utils) ────────────────────────────────

_FUSED_ROPE_INSTALLED = False


def install_fused_rope():
    """Register apex fused RoPE into Megatron's rope_utils.

    Enables ``apply_rope_fusion=True`` without TransformerEngine.  The apex
    kernel delegates to AITER on ROCm (MI300X), matching TE's fused path.

    Megatron passes ``interleaved=`` kwarg that apex doesn't accept, so we
    wrap to strip unsupported keywords.
    """
    global _FUSED_ROPE_INSTALLED
    if _FUSED_ROPE_INSTALLED:
        return
    try:
        from apex.transformer.functional.fused_rope import (
            fused_apply_rotary_pos_emb as _apex_rope,
        )
        from megatron.core.models.common.embeddings import rope_utils

        def _compat_fused_rope(t, freqs, **kwargs):
            return _apex_rope(t, freqs)

        rope_utils.fused_apply_rotary_pos_emb = _compat_fused_rope
        _FUSED_ROPE_INSTALLED = True
    except ImportError:
        pass


# ── 8. Force activation recompute during eval ───────────────────────────────


def install_eval_recompute():
    """Keep activation checkpointing active during ``model.eval()`` so that
    evaluation forward passes use the same 21-layer recompute pattern as
    training instead of storing all 80 layers' activations.

    Without this, the eval forward pass at 96%+ GPU memory creates a
    different allocation pattern that fragments the ROCm block cache.
    The fragmentation persists after ``model.train()`` resumes, causing
    a permanent +20% step-time regression.

    Controlled by ``LUMEN_EVAL_RECOMPUTE=1``.
    """
    if os.environ.get("LUMEN_EVAL_RECOMPUTE", "0") != "1":
        return

    try:
        from megatron.core.transformer import transformer_block as _tb_mod
    except ImportError:
        return

    if getattr(_tb_mod, "_lumen_eval_recompute_patched", False):
        return

    _OrigTransformerBlock = _tb_mod.TransformerBlock
    _orig_forward = _OrigTransformerBlock.forward

    def _forward_with_eval_recompute(self, *args, **kwargs):
        if not self.training and self.config.recompute_granularity == "full":
            self.training = True
            try:
                return _orig_forward(self, *args, **kwargs)
            finally:
                self.training = False
        return _orig_forward(self, *args, **kwargs)

    _OrigTransformerBlock.forward = _forward_with_eval_recompute
    _tb_mod._lumen_eval_recompute_patched = True


# ── 9. Post-eval memory cache clear ─────────────────────────────────────────


def install_post_eval_cache_clear():
    """Call ``torch.cuda.empty_cache()`` after every evaluation run to
    reset the ROCm allocator's cached block pool.

    This prevents fragmented eval-phase blocks from polluting the
    training allocation pattern.  There is a brief re-warmup cost on the
    first post-eval training step, but subsequent steps return to
    pre-eval speed.

    Controlled by ``LUMEN_POST_EVAL_CACHE_CLEAR=1``.
    """
    if os.environ.get("LUMEN_POST_EVAL_CACHE_CLEAR", "0") != "1":
        return

    try:
        import megatron.training.training as _train_mod
    except ImportError:
        return

    if getattr(_train_mod, "_lumen_post_eval_cache_clear_patched", False):
        return

    _orig_evaluate = _train_mod.evaluate
    _do_rewarm = os.environ.get("LUMEN_POST_EVAL_REWARM", "0") == "1"
    _rewarm_mode = os.environ.get(
        "LUMEN_POST_EVAL_REWARM_MODE",
        os.environ.get("LUMEN_POST_EVAL_STRATEGY", "gc_only"),
    )

    def _evaluate_with_cache_clear(*args, **kwargs):
        import gc

        result = _orig_evaluate(*args, **kwargs)
        gc.collect()
        if _rewarm_mode == "empty_cache":
            torch.cuda.empty_cache()
            if _do_rewarm:
                _post_eval_rewarm()
        else:
            torch.cuda.synchronize()
        return result

    _train_mod.evaluate = _evaluate_with_cache_clear
    _train_mod._lumen_post_eval_cache_clear_patched = True


def _post_eval_rewarm():
    """Allocate and free training-shaped tensors after eval + empty_cache to
    re-prime the ROCm allocator with the training block layout.

    This forces the allocator to create block splits that match training
    tensor sizes, so the first real training step after eval doesn't
    suffer from suboptimal allocations.

    Controlled by ``LUMEN_POST_EVAL_REWARM=1`` (requires POST_EVAL_CACHE_CLEAR).
    """
    try:
        from megatron.training import get_args
    except ImportError:
        return

    args = get_args()
    seq_len = getattr(args, "seq_length", 8192)
    mbs = getattr(args, "micro_batch_size", 1)
    hidden = getattr(args, "hidden_size", 8192)

    shapes = [
        (mbs * seq_len, hidden),
        (mbs * seq_len, hidden * 4),
        (mbs * seq_len, hidden * 2),
        (mbs, seq_len),
    ]
    buffers = []
    for shape in shapes:
        buffers.append(torch.empty(shape, dtype=torch.bfloat16, device="cuda"))
    del buffers
    torch.cuda.synchronize()


# ── 10. Fused residual-add + RMSNorm + optional Norm+Quant bypass ───────────

_FUSED_RESIDUAL_NORM = os.environ.get("LUMEN_FUSED_RESIDUAL_NORM", "0") == "1"


def install_fused_residual_norm():
    """Replace ``TransformerLayer._forward_attention`` and ``_forward_mlp``
    with Lumen-optimised versions that:

    1. **Deferred BDA** (``LUMEN_FUSED_RESIDUAL_NORM=1``): When
       ``hidden_dropout=0`` and cross-attention is ``IdentityOp``, defers the
       self-attention BDA add to ``_forward_mlp`` and fuses it with RMSNorm
       using ``atomrl.lumen.ops.fused_residual_norm``.
    """
    if not _FUSED_RESIDUAL_NORM:
        return

    try:
        from megatron.core.transformer import transformer_layer as _tl_mod
    except ImportError:
        return

    if getattr(_tl_mod, "_lumen_fused_residual_norm_patched", False):
        return

    from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp

    _OrigTL = _tl_mod.TransformerLayer
    _orig_init = _OrigTL.__init__
    _orig_fwd_attn = _OrigTL._forward_attention
    _orig_fwd_mlp = _OrigTL._forward_mlp

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)

        _can_defer = (
            _FUSED_RESIDUAL_NORM
            and self.hidden_dropout == 0.0
            and isinstance(self.cross_attention, IdentityOp)
            and isinstance(self.cross_attn_bda, IdentityFuncOp)
        )
        self._lumen_can_fuse_bda_norm = _can_defer and not isinstance(
            self.pre_mlp_layernorm, IdentityOp
        )

        from atomrl.lumen.modules.layernorm_linear import LumenLayerNormLinear

        _fc1 = getattr(getattr(self, "mlp", None), "linear_fc1", None)
        _fc1_base = getattr(_fc1, "base_layer", _fc1)
        self._lumen_can_defer_bda_to_lnlinear = (
            _can_defer
            and isinstance(self.pre_mlp_layernorm, IdentityOp)
            and isinstance(_fc1_base, LumenLayerNormLinear)
        )

        # Defer MLP BDA to the NEXT layer's linear_qkv (LumenLayerNormLinear)
        # via TLS _set_pending_residual.
        # Skip: last layer (no next layer), checkpointed layers (TLS lost
        # across checkpoint boundaries during activation recomputation).
        _qkv = getattr(getattr(self, "self_attention", None), "linear_qkv", None)
        _qkv_base = getattr(_qkv, "base_layer", _qkv)
        _recompute_n = getattr(self.config, "recompute_num_layers", None) or 0
        _is_full_recompute = (
            getattr(self.config, "recompute_granularity", None) == "full"
        )
        _in_ckpt_region = _is_full_recompute and self.layer_number <= _recompute_n
        self._lumen_defer_mlp_bda = (
            self._lumen_can_defer_bda_to_lnlinear
            and isinstance(_qkv_base, LumenLayerNormLinear)
            and self.layer_number < self.config.num_layers
            and not _in_ckpt_region
        )

        self._lumen_deferred_bda = None

    def _do_input_layernorm(self, hidden_states):
        """input_layernorm (optionally checkpointed)."""
        if self.recompute_input_layernorm:
            from megatron.core import tensor_parallel

            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            return self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states
            )
        return self.input_layernorm(hidden_states)

    def _do_pre_mlp_layernorm(self, hidden_states):
        """pre_mlp_layernorm (optionally checkpointed)."""
        if self.recompute_pre_mlp_layernorm:
            from megatron.core import tensor_parallel

            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            return self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
        return self.pre_mlp_layernorm(hidden_states)

    def _patched_fwd_attn(self, hidden_states, *args, **kwargs):
        _sa_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("context", "context_mask")
        }

        if self._lumen_can_fuse_bda_norm and not self.recompute_pre_mlp_layernorm:
            from megatron.core.utils import nvtx_range_pop, nvtx_range_push

            residual = hidden_states
            input_layernorm_output = _do_input_layernorm(self, hidden_states)

            nvtx_range_push(suffix="self_attention")
            attention_output_with_bias = self.self_attention(
                input_layernorm_output,
                *args,
                **_sa_kwargs,
            )
            nvtx_range_pop(suffix="self_attention")

            if self.recompute_input_layernorm:
                self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                    attention_output_with_bias[0]
                )

            nvtx_range_push(suffix="self_attn_bda")
            self._lumen_deferred_bda = (attention_output_with_bias, residual)
            x, bias = attention_output_with_bias
            hidden_states = (x + bias) if bias is not None else x
            nvtx_range_pop(suffix="self_attn_bda")

            residual = hidden_states
            pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(
                hidden_states
            )
            attention_output_with_bias = self.cross_attention(
                pre_cross_attn_layernorm_output,
                attention_mask=kwargs.get("context_mask"),
                key_value_states=kwargs.get("context"),
                inference_context=kwargs.get("inference_context"),
            )
            context = None
            if (
                isinstance(attention_output_with_bias, dict)
                and "context" in attention_output_with_bias
            ):
                context = attention_output_with_bias["context"]
            with self.bias_dropout_add_exec_handler():
                hidden_states = self.cross_attn_bda(
                    self.training, self.config.bias_dropout_fusion
                )(attention_output_with_bias, residual, self.hidden_dropout)
            return hidden_states, context

        if self._lumen_can_defer_bda_to_lnlinear:
            from megatron.core.utils import nvtx_range_pop, nvtx_range_push

            residual = hidden_states
            input_layernorm_output = _do_input_layernorm(self, hidden_states)

            nvtx_range_push(suffix="self_attention")
            attention_output_with_bias = self.self_attention(
                input_layernorm_output,
                *args,
                **_sa_kwargs,
            )
            nvtx_range_pop(suffix="self_attention")

            # If previous layer deferred its MLP BDA via _set_pending_residual(),
            # linear_qkv fused (hidden_states + pending_residual) and stored the
            # result in residual_out.  Use that as the true residual.
            from atomrl.lumen.modules.layernorm_linear import _pop_residual_out

            _fused_residual = _pop_residual_out()
            if _fused_residual is not None:
                residual = _fused_residual

            if self.recompute_input_layernorm:
                self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                    attention_output_with_bias[0]
                )

            nvtx_range_push(suffix="self_attn_bda")
            self._lumen_deferred_bda = (attention_output_with_bias, residual)
            x, bias = attention_output_with_bias
            hidden_states = (x + bias) if bias is not None else x
            nvtx_range_pop(suffix="self_attn_bda")

            residual = hidden_states
            pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(
                hidden_states
            )
            attention_output_with_bias = self.cross_attention(
                pre_cross_attn_layernorm_output,
                attention_mask=kwargs.get("context_mask"),
                key_value_states=kwargs.get("context"),
                inference_context=kwargs.get("inference_context"),
            )
            context = None
            if (
                isinstance(attention_output_with_bias, dict)
                and "context" in attention_output_with_bias
            ):
                context = attention_output_with_bias["context"]
            with self.bias_dropout_add_exec_handler():
                hidden_states = self.cross_attn_bda(
                    self.training, self.config.bias_dropout_fusion
                )(attention_output_with_bias, residual, self.hidden_dropout)
            return hidden_states, context

        self._lumen_deferred_bda = None
        return _orig_fwd_attn(self, hidden_states, *args, **kwargs)

    def _patched_fwd_mlp(self, hidden_states, inference_context=None):
        from megatron.core.utils import (
            make_viewless_tensor,
            nvtx_range_pop,
            nvtx_range_push,
        )

        _deferred = self._lumen_deferred_bda
        _fused_ok = False
        _defer_to_lnlinear = False

        if _deferred is not None:
            self._lumen_deferred_bda = None

            if self._lumen_can_defer_bda_to_lnlinear:
                from atomrl.lumen.modules.layernorm_linear import _set_pending_residual

                _x_with_bias, _orig_residual = _deferred
                x, bias = _x_with_bias
                hidden_states = (x + bias) if bias is not None else x
                _set_pending_residual(_orig_residual)
                _defer_to_lnlinear = True
            else:
                from atomrl.lumen.ops.fused_residual_norm import (
                    deferred_bda_add,
                    rmsnorm_from_module,
                )

                _x_with_bias, _orig_residual = _deferred
                hidden_states = deferred_bda_add(_x_with_bias, _orig_residual)
                residual = hidden_states
                pre_mlp_layernorm_output = rmsnorm_from_module(
                    hidden_states, self.pre_mlp_layernorm
                )
                _fused_ok = True

        if not _fused_ok and not _defer_to_lnlinear:
            residual = hidden_states
            pre_mlp_layernorm_output = _do_pre_mlp_layernorm(self, hidden_states)

        nvtx_range_push(suffix="mlp")
        if _defer_to_lnlinear:
            mlp_output_with_bias = self.mlp(hidden_states)
            from atomrl.lumen.modules.layernorm_linear import _pop_residual_out

            residual = _pop_residual_out()
        elif self.recompute_mlp:
            if self.config.fp8:
                from megatron.core import tensor_parallel
                from megatron.core.extensions.transformer_engine import te_checkpoint

                mlp_output_with_bias = te_checkpoint(
                    self.mlp,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    pre_mlp_layernorm_output,
                )
            else:
                from megatron.core import tensor_parallel

                mlp_output_with_bias = tensor_parallel.checkpoint(
                    self.mlp, False, pre_mlp_layernorm_output
                )
        else:
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )
        nvtx_range_pop(suffix="mlp")

        if self._lumen_defer_mlp_bda:
            # Defer MLP BDA: store residual in TLS for the next layer's
            # linear_qkv (LumenLayerNormLinear) to fuse add+norm+quant.
            from atomrl.lumen.modules.layernorm_linear import _set_pending_residual

            _set_pending_residual(residual)
            x, bias = mlp_output_with_bias
            hidden_states = (x + bias) if bias is not None else x
        else:
            nvtx_range_push(suffix="mlp_bda")
            with self.bias_dropout_add_exec_handler():
                hidden_states = self.mlp_bda(
                    self.training, self.config.bias_dropout_fusion
                )(mlp_output_with_bias, residual, self.hidden_dropout)
            nvtx_range_pop(suffix="mlp_bda")

        output = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )
        return output

    _OrigTL.__init__ = _patched_init
    _OrigTL._forward_attention = _patched_fwd_attn
    _OrigTL._forward_mlp = _patched_fwd_mlp
    _tl_mod._lumen_fused_residual_norm_patched = True


# ── SplitAlongDim zero-copy backward ───────────────────────────────────────


class _LumenSplitAlongDim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixed_x, split_dim, split_size_or_sections):
        ctx.split_dim = split_dim
        ctx.split_size_or_sections = split_size_or_sections
        return torch.split(mixed_x, split_size_or_sections, dim=split_dim)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not grad_outputs:
            return None, None, None
        split_sizes = ctx.split_size_or_sections
        if isinstance(split_sizes, int):
            split_sizes = [split_sizes] * len(grad_outputs)
        dims = len(grad_outputs[0].shape)
        split_dim = (ctx.split_dim + dims) % dims
        noop_ok = True
        strides = grad_outputs[0].stride()
        data_ptr = grad_outputs[0].untyped_storage().data_ptr()
        shape = list(grad_outputs[0].shape)
        for i, tensor in enumerate(grad_outputs):
            expected_shape = list(shape)
            expected_shape[split_dim] = split_sizes[i]
            offset_size = sum(split_sizes[:i]) * math.prod(shape[split_dim + 1 :])
            if (
                tensor.stride() != strides
                or list(tensor.shape) != expected_shape
                or tensor.untyped_storage().data_ptr() != data_ptr
                or tensor.storage_offset() != offset_size
            ):
                noop_ok = False
                break
        if noop_ok:
            ret = torch.Tensor().to(
                device=grad_outputs[0].device, dtype=grad_outputs[0].dtype
            )
            new_shape = list(shape)
            new_shape[split_dim] = sum(split_sizes)
            ret.set_(
                grad_outputs[0].untyped_storage(),
                grad_outputs[0].storage_offset(),
                new_shape,
                strides,
            )
            return ret, None, None
        return torch.cat(grad_outputs, dim=split_dim), None, None


def install_split_along_dim():
    try:
        import megatron.core.extensions.transformer_engine as _te_mod
    except ImportError:
        return
    if getattr(_te_mod, "_lumen_split_along_dim_patched", False):
        return
    _te_mod.SplitAlongDim = _LumenSplitAlongDim.apply
    _te_mod._lumen_split_along_dim_patched = True
    try:
        import megatron.core.transformer.attention as _attn_mod

        _attn_mod.SplitAlongDim = _LumenSplitAlongDim.apply
    except ImportError:
        pass


# ── DistributedOptimizer _foreach_copy_ patch ────────────────────────────


def install_optimizer_patches():
    """Batch per-param copy loops in DistributedOptimizer with _foreach_copy_.

    Replaces two methods that iterate over hundreds of BF16 params one by one:
      _copy_model_grads_to_main_grads  : N BF16→FP32 grad copies → 1 batched call
      _copy_main_params_to_model_params: N FP32→BF16 param copies → 1 batched call

    FP32 param groups keep original per-param assignment (no allocation overhead).
    Idempotent; safe to call multiple times.
    """
    try:
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    except ImportError:
        return

    if getattr(DistributedOptimizer, "_lumen_foreach_copy_patched", False):
        return

    import torch
    from megatron.core.fp8_utils import is_float8tensor, quantize_param_shard

    def _patched_copy_grads_to_main(self):
        if self.is_stub_optimizer:
            return
        if self.ddp_config.use_megatron_fsdp:
            return

        # BF16 model params → FP32 main grads, batched with _foreach_copy_
        def copy_float16_grads(model_groups, shard_main_groups):
            bf16_srcs, fp32_dsts = [], []
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):
                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()
                    shard_model_grad = model_param.main_grad.view(-1)[
                        param_range.start : param_range.end
                    ]
                    if (
                        shard_main_param.grad is None
                        or shard_main_param.grad.shape != shard_model_grad.shape
                        or shard_main_param.grad.dtype != torch.float32
                    ):
                        shard_main_param.grad = torch.empty(
                            shard_model_grad.shape,
                            dtype=torch.float32,
                            device=shard_model_grad.device,
                        )
                    bf16_srcs.append(shard_model_grad)
                    fp32_dsts.append(shard_main_param.grad)
            if bf16_srcs:
                torch._foreach_copy_(fp32_dsts, bf16_srcs)

        # FP32 model params → FP32 main grads, original per-param assignment
        def copy_fp32_grads(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):
                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()
                    shard_model_grad = model_param.main_grad.view(-1)[
                        param_range.start : param_range.end
                    ]
                    shard_main_param.grad = shard_model_grad.float()

        # Precision-aware optimizer uses decoupled_grad instead of grad
        def copy_decoupled_grads(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):
                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()
                    shard_model_grad = model_param.main_grad.view(-1)[
                        param_range.start : param_range.end
                    ]
                    shard_main_param.decoupled_grad = shard_model_grad

        if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            copy_decoupled_grads(self.model_float16_groups, self.shard_float16_groups)
            copy_decoupled_grads(self.model_fp32_groups, self.shard_fp32_groups)
        else:
            copy_float16_grads(
                self.model_float16_groups, self.shard_fp32_from_float16_groups
            )
            copy_fp32_grads(self.model_fp32_groups, self.shard_fp32_groups)

    def _patched_copy_main_to_model(self):
        if self.is_stub_optimizer:
            return
        if self.ddp_config.use_megatron_fsdp:
            for model_chunk in self.model_chunks:
                model_chunk.param_and_grad_buffer.copy_main_weights_to_model_weights()
            return
        if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            return

        quantize_param_shard(
            *self._get_fp8_params_and_shard_fp32_from_fp8(), self.data_parallel_group
        )

        # FP32 main params → BF16 model buffers, batched with _foreach_copy_
        def copy_group_params(shard_main_groups, model_groups):
            fp32_srcs, model_dsts = [], []
            for shard_main_group, model_group in zip(shard_main_groups, model_groups):
                for shard_main_param, model_param in zip(shard_main_group, model_group):
                    if is_float8tensor(model_param):
                        continue
                    param_range_map = self._get_model_param_range_map(model_param)
                    world_range = param_range_map["gbuf_world_in_bucket"]
                    assert world_range.size == shard_main_param.nelement()
                    gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                    model_param_buffer = (
                        self.buffers[gbuf_index].buckets[bucket_id].param_data
                    )
                    shard_model_param = model_param_buffer.view(-1)[
                        world_range.start : world_range.end
                    ]
                    fp32_srcs.append(shard_main_param)
                    model_dsts.append(shard_model_param)
            if fp32_srcs:
                torch._foreach_copy_(model_dsts, fp32_srcs)

        copy_group_params(
            self.shard_fp32_from_float16_groups, self.model_float16_groups
        )
        copy_group_params(self.shard_fp32_groups, self.model_fp32_groups)

    DistributedOptimizer._copy_model_grads_to_main_grads = _patched_copy_grads_to_main
    DistributedOptimizer._copy_main_params_to_model_params = _patched_copy_main_to_model
    DistributedOptimizer._lumen_foreach_copy_patched = True


_cross_entropy_patched = False


def install_cross_entropy(**_kwargs) -> None:
    """Monkey-patch Megatron's cross-entropy so the GPTModel loss computation
    goes through Lumen's Triton kernel.

    ``LanguageModule.compute_language_model_loss`` dispatches through
    ``te_parallel_cross_entropy`` when ``cross_entropy_loss_fusion`` is
    enabled.  We replace that symbol with ``lumen_parallel_cross_entropy``
    (same signature) and also wrap it as ``vocab_parallel_cross_entropy``
    for the non-fusion fallback path.
    """
    global _cross_entropy_patched
    if _cross_entropy_patched:
        return

    from atomrl.lumen.modules.cross_entropy import lumen_parallel_cross_entropy

    try:
        import megatron.core.models.common.language_module.language_module as _lm_mod

        _lm_mod.te_parallel_cross_entropy = lumen_parallel_cross_entropy
    except (ImportError, AttributeError):
        pass

    try:
        import megatron.core.extensions.transformer_engine as _te_ext

        _te_ext.te_parallel_cross_entropy = lumen_parallel_cross_entropy
    except (ImportError, AttributeError):
        pass

    def _vocab_parallel_ce_adapter(logits, labels, label_smoothing=0.0):
        from megatron.core import parallel_state

        tp_group = parallel_state.get_tensor_model_parallel_group()
        return lumen_parallel_cross_entropy(logits, labels, tp_group)

    try:
        import megatron.core.tensor_parallel as _tp_mod
        import megatron.core.tensor_parallel.cross_entropy as _ce_mod

        _tp_mod.vocab_parallel_cross_entropy = _vocab_parallel_ce_adapter
        _ce_mod.vocab_parallel_cross_entropy = _vocab_parallel_ce_adapter
    except (ImportError, AttributeError):
        pass

    _cross_entropy_patched = True
    from megatron.training import print_rank_0

    print_rank_0("> Patched cross-entropy with Lumen Triton kernel")


# ── Patch registry (Phase 1) ────────────────────────────────────────────────

from atomrl.lumen.patches.registry import PatchPhase, register_patch


def _env_flag(name: str, default: str = "0"):
    return lambda: os.environ.get(name, default) == "1"


register_patch(
    "fused_layer_norm",
    PatchPhase.IMPORT,
    description="FusedLayerNorm → Lumen RMSNorm/LayerNorm",
    tags=frozenset({"core", "norm", "megatron"}),
)(install_fused_layer_norm)

register_patch(
    "language_module_checkpoint_guard",
    PatchPhase.IMPORT,
    description="Guard LanguageModule.sharded_state_dict for LoRA output_layer",
    tags=frozenset({"core", "checkpoint", "lora", "megatron"}),
)(install_language_module_checkpoint_guard)

register_patch(
    "mmap_checkpoint",
    PatchPhase.IMPORT,
    description="Inject mmap=True into Megatron torch.load calls",
    tags=frozenset({"core", "checkpoint", "megatron"}),
)(install_mmap_checkpoint)

register_patch(
    "requires_grad_fix",
    PatchPhase.IMPORT,
    description="Fix requires_grad for LoRA + activation checkpointing",
    tags=frozenset({"core", "lora", "recompute", "megatron"}),
)(install_requires_grad_fix)

register_patch(
    "swiglu_fp8",
    PatchPhase.IMPORT,
    description="ROCm FP8 SwiGLU dtype + chunked backward",
    tags=frozenset({"core", "fp8", "fusion", "megatron"}),
)(install_swiglu_fp8)

register_patch(
    "fused_swiglu_triton",
    PatchPhase.IMPORT,
    description="AITER Triton fused SwiGLU",
    enabled=_env_flag("LUMEN_FUSED_SWIGLU"),
    tags=frozenset({"fusion", "fp8", "megatron"}),
)(install_fused_swiglu_triton)

register_patch(
    "mlp_fp8_store",
    PatchPhase.IMPORT,
    description="Store MLP SwiGLU intermediates in FP8",
    enabled=_env_flag("LUMEN_MLP_FP8_STORE"),
    tags=frozenset({"fp8", "mlp", "megatron"}),
)(install_mlp_fp8_store)

register_patch(
    "mlp_recompute",
    PatchPhase.IMPORT,
    description="MLP-only recompute for non-recomputed TransformerLayers",
    tags=frozenset({"recompute", "mlp", "megatron"}),
)(install_mlp_recompute)

register_patch(
    "fused_rope",
    PatchPhase.IMPORT,
    description="Apex fused RoPE into megatron.core.models.common.rope_utils",
    tags=frozenset({"fusion", "rope", "megatron"}),
)(install_fused_rope)

register_patch(
    "eval_recompute",
    PatchPhase.IMPORT,
    description="Keep activation checkpointing active during model.eval()",
    enabled=_env_flag("LUMEN_EVAL_RECOMPUTE"),
    tags=frozenset({"recompute", "eval", "megatron"}),
)(install_eval_recompute)

register_patch(
    "post_eval_cache_clear",
    PatchPhase.IMPORT,
    description="empty_cache / gc after Megatron evaluate()",
    enabled=_env_flag("LUMEN_POST_EVAL_CACHE_CLEAR"),
    tags=frozenset({"eval", "memory", "megatron"}),
)(install_post_eval_cache_clear)

register_patch(
    "fused_residual_norm",
    PatchPhase.IMPORT,
    description="Deferred BDA + fused residual+RMSNorm on TransformerLayer",
    enabled=_env_flag("LUMEN_FUSED_RESIDUAL_NORM"),
    tags=frozenset({"fusion", "norm", "megatron"}),
)(install_fused_residual_norm)

register_patch(
    "optimizer_patches",
    PatchPhase.IMPORT,
    description="DistributedOptimizer batched _foreach_copy_",
    tags=frozenset({"optimizer", "megatron"}),
)(install_optimizer_patches)

register_patch(
    "split_along_dim",
    PatchPhase.IMPORT,
    description="Lumen SplitAlongDim shim for TE attention",
    default=False,
    tags=frozenset({"te", "attention", "megatron"}),
)(install_split_along_dim)

register_patch(
    "cross_entropy",
    PatchPhase.IMPORT,
    description="Lumen Triton parallel cross-entropy for --lumen-cross-entropy",
    default=False,
    tags=frozenset({"core", "fusion", "lumen", "megatron"}),
)(install_cross_entropy)


# ── Public API ──────────────────────────────────────────────────────────────


def install_all():
    """Apply every default Megatron compatibility patch.

    Must be called **before** ``GPTModel`` / ``TransformerBlock`` is imported.
    Safe to call multiple times (each installer is idempotent).

    Delegates to :func:`atomrl.lumen.patches.apply_patches` for the IMPORT phase.
    """
    from atomrl.lumen.patches import apply_patches

    apply_patches(PatchPhase.IMPORT, default_only=True)

    # SDMA DP gradient all-reduce (replaces NCCL when --use-sdma is set)
    # Disabled: the sdma_dp_grad_reduce module was never committed on this
    # branch (only the import line exists). NCCL grad reduce is used instead.
    # from atomrl.lumen.models.sdma_dp_grad_reduce import install_sdma_dp_grad_reduce
    # install_sdma_dp_grad_reduce()
