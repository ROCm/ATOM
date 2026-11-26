# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Unified Lumen configuration — single dataclass covering quantization,
attention, norm replacement, and execution/fusion features.

Usage::

    from atomrl.lumen.config import LumenConfig

    # Programmatic — flat kwargs, one call
    cfg = LumenConfig(format="fp8_e4m3", scaling="delayed",
                      fp8_attn="dpa", lumen_norm=True, fused_mlp=True)
    cfg.enable(model)

    # From CLI argparse namespace
    cfg = LumenConfig.from_args(args)
    cfg.enable(model, dp_group=dp_group)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields

logger = logging.getLogger(__name__)


def _rank0_print(msg: str) -> None:
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except (AssertionError, RuntimeError):
        pass
    logger.info(msg)


# ---------------------------------------------------------------------------
# Arg-name mapping: LumenConfig field → argparse attribute name(s)
# ---------------------------------------------------------------------------

_ARG_MAP: dict[str, tuple[str, ...]] = {
    "format": ("linear_fp8_format", "lumen_fp8_format"),
    "scaling": ("linear_fp8_scaling",),
    "block_size": ("linear_fp8_block_size",),
    "amax_algo": ("linear_fp8_amax_algo",),
    "margin": ("linear_fp8_margin",),
    "reduce_amax": ("linear_fp8_reduce_amax",),
    "history_len": ("linear_fp8_amax_history",),
    "quantize_activation": ("linear_fp8_activation",),
    "fp8_wgrad": ("linear_fp8_wgrad",),
    "cache_frozen_weight": ("linear_fp8_cache_frozen_weight",),
    "bpreshuffle_gemm": ("linear_fp8_bpreshuffle",),
    "quantize_grad": ("grad_quant_type",),
    "quantize_output_layer": (
        "linear_fp8_quantize_output_layer",
        "quantize_output_layer",
    ),
    "first_last_layers_bf16": ("first_last_layers_bf16",),
    "num_layers_at_start_in_bf16": ("num_layers_at_start_in_bf16",),
    "num_layers_at_end_in_bf16": ("num_layers_at_end_in_bf16",),
    "num_layers": ("num_layers",),
    "use_sdma": ("use_sdma",),
    "fp8_attn": ("lumen_fp8_attn",),
    "attn_backend": ("lumen_attn_backend",),
    "attn_quant_type": ("lumen_fp8_quant_type",),
    "lumen_norm": ("lumen_norm",),
    "hf_attn_patch": ("hf_attn_patch",),
    "lumen_linear": ("lumen_linear",),
    "aiter_conv": ("lumen_aiter_conv", "aiter_conv"),
    "dit_sdpa_efficient": ("lumen_dit_sdpa_efficient",),
    "dit_sdpa_triton_forward": ("lumen_dit_sdpa_triton_forward",),
    "diffusers_rmsnorm_fuse": ("lumen_diffusers_rmsnorm_fuse",),
    "local_shard_adamw": ("lumen_local_shard_adamw",),
    "fused_mlp": ("lumen_fused_mlp",),
    "fp8_activation_store": ("lumen_fp8_activation_store",),
    "cpu_offload": ("lumen_cpu_offload",),
    "delay_wgrad": ("lumen_delay_wgrad",),
    "gradient_accumulation_fusion": ("lumen_gradient_accumulation_fusion",),
    "fp8_param_gather": ("lumen_fp8_param_gather",),
    "fp8_weight_cache": ("lumen_fp8_weight_cache",),
    "fused_rope": ("lumen_fused_rope",),
    "hip_graphs": ("lumen_hip_graphs",),
    "fp8_checkpoint": ("lumen_fp8_checkpoint",),
    "fp8_param_manager": ("fp8_param_manager",),
    "lora_rank": ("lora_rank",),
    "lora_alpha": ("lora_alpha",),
    "lora_dropout": ("lora_dropout",),
    "use_8bit_adam": ("use_8bit_adam",),
}


def check_linear_quant_exclusive(linear_fp8, linear_fp4) -> None:
    """Refuse a config that asks for FP8 and FP4 linears at once.

    Lives here rather than only inside ``from_args`` because not every consumer
    goes through it: the RL dataclasses are constructed directly and several
    call sites read ``args.linear_fp4`` off the namespace. With the check in one
    place they all reach it, and the two flags are also mutually exclusive at
    the Megatron parser so the common case fails before any model is built.
    """
    if linear_fp8 and linear_fp4:
        raise ValueError(
            "--linear-fp8 and --linear-fp4 are mutually exclusive: FP4 selects the "
            "fixed MXFP4 recipe and cannot share the FP8 format selector. Pick one."
        )


@dataclass
class LumenConfig:
    """Unified configuration for all Lumen training features.

    Groups functionality behind a single flat interface:

    * **Tier 0 — Weight storage & adapters:** ``fp8_param_manager``,
      ``lora_rank`` / ``lora_alpha`` / ``lora_dropout``.  Applied first
      (FP8ParamManager before LoRA) so adapter weights stay BF16.
    * **Tier 1 — Linear quantization:** ``format``, ``scaling``, ``block_size``, etc.
      These are forwarded to :class:`~atomrl.lumen.quantize.QuantConfig`.
    * **Tier 2 — Attention FP8 & norms:** ``fp8_attn``, ``attn_backend``,
      ``attn_quant_type``, ``lumen_norm``.
    * **Tier 3 — Execution / fusion:** ``fused_mlp``, ``cpu_offload``,
      ``delay_wgrad``, ``hip_graphs``, etc.
    * **Optimizer hint:** ``use_8bit_adam`` — not applied to the model,
      read by the trainer to select 8-bit Adam from bitsandbytes.
    """

    # -- Tier 1: Linear quantization (forwarded to QuantConfig) --
    format: str = "fp8_e4m3"
    scaling: str = "delayed"
    # ``None`` means the format default: 128 for FP8 block recipes and the
    # spec-mandated 32 for MXFP4. Keeping the sentinel is what lets an explicit
    # invalid MXFP4 value reach QuantConfig and fail instead of being silently
    # rewritten to 32.
    block_size: int | None = None
    amax_algo: str = "max"
    margin: int = 0
    reduce_amax: bool = False
    history_len: int = 16
    quantize_activation: bool = True
    fp8_wgrad: bool = True
    cache_frozen_weight: bool = False
    bpreshuffle_gemm: bool = False
    quantize_grad: str | None = None
    quantize_output_layer: bool = False
    first_last_layers_bf16: bool = False
    num_layers_at_start_in_bf16: int = 1
    num_layers_at_end_in_bf16: int = 1
    num_layers: int = 0
    use_sdma: bool = False

    # -- Tier 2: Attention FP8 --
    fp8_attn: str = "none"
    attn_backend: str = "auto"
    attn_quant_type: str = "blockwise"

    # -- Tier 2: Norm replacement --
    lumen_norm: bool = False

    # -- Tier 2: Attention patching --
    hf_attn_patch: bool = False

    # -- Tier 2: Linear GEMM patching (BF16) --
    lumen_linear: bool = False

    # -- Tier 2: Conv2d/Conv3d (AITER forward, PyTorch backward) --
    aiter_conv: bool = False

    # -- Tier 2: Diffusion-model attention and norm routing --
    dit_sdpa_efficient: bool = False
    dit_sdpa_triton_forward: bool = False
    diffusers_rmsnorm_fuse: bool = False

    # -- Tier 3: FSDP2 local-shard optimizer fast path --
    local_shard_adamw: bool = False

    # -- Tier 3: Execution / fusion --
    fused_mlp: bool = False
    fp8_activation_store: bool = False
    cpu_offload: bool = False
    delay_wgrad: bool = False
    gradient_accumulation_fusion: bool = False
    fp8_param_gather: bool = False
    fp8_weight_cache: bool = False
    fused_rope: bool = False
    hip_graphs: bool = False
    fp8_checkpoint: bool = False

    # -- Tier 0: FP8 weight storage (applied before everything else) --
    fp8_param_manager: bool = False

    # -- Tier 0: LoRA via HuggingFace PEFT (after FP8ParamManager) --
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # -- Optimizer hint (not applied to model, read by trainer) --
    use_8bit_adam: bool = False

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def quant_config(self):
        """Build the inner :class:`~atomrl.lumen.quantize.QuantConfig`."""
        from atomrl.lumen.quantize import QuantConfig
        from atomrl.lumen.quantize.config import MXFP4_BLOCK_SIZE

        # Hand-built configs use the format default, while an explicit value is
        # preserved so QuantConfig can enforce MXFP4's fixed 32-element blocks.
        if self.block_size is None:
            block_size = MXFP4_BLOCK_SIZE if self.format == "mxfp4" else 128
        else:
            block_size = self.block_size

        return QuantConfig.from_str(
            format=self.format,
            scaling=self.scaling,
            block_size=block_size,
            amax_algo=self.amax_algo,
            margin=self.margin,
            reduce_amax=self.reduce_amax,
            history_len=self.history_len,
            quantize_activation=self.quantize_activation,
            fp8_wgrad=self.fp8_wgrad,
            cache_frozen_weight=self.cache_frozen_weight,
            bpreshuffle_gemm=self.bpreshuffle_gemm,
            quantize_grad=self.quantize_grad,
            quantize_output_layer=self.quantize_output_layer,
            first_last_layers_bf16=self.first_last_layers_bf16,
            num_layers_at_start_in_bf16=self.num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16=self.num_layers_at_end_in_bf16,
            num_layers=self.num_layers,
            use_sdma=self.use_sdma,
            fp8_dpa=self.fp8_attn in ("dpa", "mha"),
            fp8_mha=self.fp8_attn == "mha",
        )

    # -----------------------------------------------------------------------
    # Orchestrated enablement
    # -----------------------------------------------------------------------

    @property
    def has_any_features(self) -> bool:
        """Return True if any Lumen feature is enabled."""
        return (
            self.quant_config.is_quantized
            or self.lumen_norm
            or self.hf_attn_patch
            or self.lumen_linear
            or self.aiter_conv
            or self.dit_sdpa_efficient
            or self.dit_sdpa_triton_forward
            or self.diffusers_rmsnorm_fuse
            or self.local_shard_adamw
            or self.fp8_param_manager
            or self.lora_rank > 0
            or self.fused_mlp
            or self.fp8_activation_store
            or self.cpu_offload
            or self.fp8_param_gather
            or self.fp8_weight_cache
            or self.fp8_checkpoint
            or self.fused_rope
            or self.hip_graphs
            or self.delay_wgrad
            or self.gradient_accumulation_fusion
        )

    def enable(self, model, *, dp_group=None, backend: str = "auto"):
        """Apply all Lumen features to *model* in the correct order.

        Orchestration:
          0a. FP8ParamManager — quantize linear weights to FP8 storage
          0b. LoRA (PEFT) — wrap linears with trainable adapters
          1.  Norm patching (before quant so new norm modules get patched)
          2.  Pre-quant module flags (delay_wgrad, grad-accum fusion, etc.)
          3.  ``quant.enable()`` — low-precision linear patching
          4.  Post-quant features (fp8_checkpoint, fp8_param_gather)
          5.  Attach config to model for downstream reads

        FP8ParamManager runs **before** LoRA so that only base ``nn.Linear``
        weights are quantized; the LoRA adapter weights (``lora_A``, ``lora_B``)
        created afterwards stay in BF16 and remain trainable.

        Returns:
            ``(manager, model)`` — the :class:`~atomrl.lumen.quantize.ScalingManager`
            (or ``None``) and the model (may be a new PEFT wrapper).
        """

        if self.fp8_param_manager and self.format == "mxfp4":
            raise ValueError(
                "fp8_param_manager cannot be combined with MXFP4: it replaces "
                "nn.Linear weights with FP8 storage before MXFP4 quantization, "
                "which requires BF16/FP32 masters. Disable fp8_param_manager; "
                "MXFP4 already caches packed FP4 weights per master-weight version."
            )
        if self.dit_sdpa_triton_forward and not self.dit_sdpa_efficient:
            raise ValueError(
                "dit_sdpa_triton_forward requires dit_sdpa_efficient so its "
                "forward uses the matching efficient backward"
            )

        # 0a. FP8 param storage (replaces weight.data with FP8, freezes)
        fp8pm_mgr = None
        if self.fp8_param_manager:
            fp8pm_mgr = self._apply_fp8_param_manager(model)

        # 0b. LoRA adapters (PEFT)
        if self.lora_rank > 0:
            model = self._apply_lora(model)

        # 1. Norm patching
        if self.lumen_norm:
            self._patch_norms(model)

        # 1b. Attention patching (monkey-patch F.scaled_dot_product_attention)
        if self.hf_attn_patch:
            self._patch_sdpa()

        # 1c. Linear GEMM patching (BF16 AITER Triton)
        if self.lumen_linear:
            self._patch_linear(model)

        if self.aiter_conv:
            self._patch_conv(model)

        if self.dit_sdpa_efficient:
            from atomrl.lumen.ops.attention.hf_patch import prefer_efficient_backend

            prefer_efficient_backend()
        if self.dit_sdpa_triton_forward:
            from atomrl.lumen.ops.attention.hf_patch import install_triton_forward

            install_triton_forward()
        if self.diffusers_rmsnorm_fuse:
            from atomrl.lumen.ops.normalization.rmsnorm import patch_diffusers_rmsnorm

            patch_diffusers_rmsnorm()
        if self.local_shard_adamw:
            from atomrl.lumen.quantize.local_adamw import install_local_shard_adamw

            install_local_shard_adamw()

        # 2. Pre-quant module attributes
        self._apply_pre_quant(model)

        # 3. Linear quantization
        qcfg = self.quant_config
        manager = None
        if qcfg.is_quantized:
            import torch.distributed as dist

            import atomrl.lumen.quantize as quant

            if dp_group is None and qcfg.reduce_amax and dist.is_initialized():
                dp_group = dist.group.WORLD
            manager = quant.enable(
                model,
                config=qcfg,
                backend=backend,
                dp_group=dp_group if qcfg.reduce_amax else None,
            )

        # 4. Post-quant features
        self._apply_post_quant(model, manager)

        # 5. Attach config + FP8ParamManager reference
        model._lumen_config = self
        if fp8pm_mgr is not None:
            model._fp8_param_manager = fp8pm_mgr

        self._log_summary(manager)
        return manager, model

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _apply_fp8_param_manager(self, model):
        """Replace linear weights with FP8 storage via FP8ParamManager.

        Targets ``nn.Linear`` and Megatron ``ColumnParallelLinear`` /
        ``RowParallelLinear``.  Must run before LoRA wrapping so that
        adapter weights stay BF16/trainable.
        """
        import torch

        from atomrl.lumen.quantize.fp8_params import FP8ParamManager

        mgr = FP8ParamManager(fp8_dtype=torch.float8_e4m3fn)
        count = mgr.quantize_params(model)
        hooks = mgr.register_dequant_hooks(model)
        _rank0_print(
            f"> FP8ParamManager: quantized {count} params, registered {hooks} hooks"
        )
        return mgr

    def _apply_lora(self, model):
        """Apply LoRA adapters via HuggingFace PEFT and freeze the base model.

        When FP8ParamManager is active, PEFT's ``_move_adapter_to_device_of_base_layer``
        casts LoRA adapter weights to FP8 (matching the quantized base weight dtype).
        We fix this by re-casting all LoRA adapter parameters back to BF16 so they
        remain trainable with standard arithmetic.
        """
        import torch
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=list(self.lora_target_modules),
        )
        model = get_peft_model(model, peft_config)

        if self.fp8_param_manager:
            fixed = 0
            for name, param in model.named_parameters():
                if "lora_" in name and param.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                    torch.float8_e5m2,
                    torch.float8_e5m2fnuz,
                ):
                    param.data = param.data.to(torch.bfloat16)
                    param.requires_grad_(True)
                    fixed += 1
            if fixed:
                _rank0_print(f"> Fixed {fixed} LoRA params cast to FP8 by PEFT -> BF16")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        _rank0_print(
            f"> LoRA applied (rank={self.lora_rank}, alpha={self.lora_alpha}) — "
            f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
        )
        return model

    def _patch_norms(self, model) -> None:
        import torch

        from atomrl.lumen.ops.normalization import LumenLayerNorm, LumenRMSNorm

        count = 0
        for _name, module in model.named_modules():
            for attr_name, child in list(module.named_children()):
                cls_name = type(child).__name__
                if cls_name in (
                    "RMSNorm",
                    "LlamaRMSNorm",
                    "MistralRMSNorm",
                    "Qwen2RMSNorm",
                    "Qwen3RMSNorm",
                ):
                    hidden = child.weight.shape[0]
                    eps = getattr(
                        child, "eps", getattr(child, "variance_epsilon", 1e-6)
                    )
                    target_dtype = (
                        child.weight.dtype
                        if not child.weight.is_meta
                        else torch.bfloat16
                    )
                    repl = LumenRMSNorm(
                        hidden, eps=eps, grad_quant_type=self.quantize_grad
                    )
                    if not child.weight.is_meta:
                        repl.weight.data.copy_(child.weight.data)
                    repl.to(target_dtype)
                    if child.weight.is_meta:
                        repl.to(device="meta")
                    setattr(module, attr_name, repl)
                    count += 1
                elif cls_name in ("LayerNorm",):
                    hidden = (
                        child.weight.shape[0]
                        if child.weight is not None
                        else child.normalized_shape[0]
                    )
                    eps = getattr(child, "eps", 1e-5)
                    repl = LumenLayerNorm(
                        hidden, eps=eps, grad_quant_type=self.quantize_grad
                    )
                    is_meta = child.weight is not None and child.weight.is_meta
                    target_dtype = (
                        child.weight.dtype
                        if (child.weight is not None and not is_meta)
                        else torch.bfloat16
                    )
                    if child.weight is not None and not is_meta:
                        repl.weight.data.copy_(child.weight.data)
                    if (
                        hasattr(child, "bias")
                        and child.bias is not None
                        and repl.bias is not None
                        and not is_meta
                    ):
                        repl.bias.data.copy_(child.bias.data)
                    repl.to(target_dtype)
                    if is_meta:
                        repl.to(device="meta")
                    setattr(module, attr_name, repl)
                    count += 1

        if count:
            _rank0_print(f"> Replaced {count} norm modules with Lumen implementations")

    def _patch_sdpa(self) -> None:
        """Monkey-patch ``F.scaled_dot_product_attention`` with AITER attention."""
        from atomrl.lumen.ops.attention.hf_patch import patch_sdpa

        patch_sdpa()

    @staticmethod
    def _auto_tune_bf16_gemm(model) -> None:
        """Placeholder — BF16 GEMM uses torch.mm fallback (safe on MI350).

        hipBLASLt and ASM JIT both SIGABRT on MI350 gfx950, so we skip
        tuned GEMM injection and let ``tuned_gemm.gemm_a16w16`` fall back
        to ``torch.mm`` (which internally uses hipBLASLt via PyTorch).
        """

    def _patch_linear(self, model) -> None:
        """Replace ``nn.Linear`` forward with AITER GEMM (BF16, autograd-safe).

        Runs ``_auto_tune_bf16_gemm`` first to ensure hipBLASLt tuning
        configs exist, then patches forward with ``dispatch_gemm``.
        """
        import torch
        from torch import nn

        self._auto_tune_bf16_gemm(model)

        from atomrl.lumen.ops.quantize.linear import dispatch_gemm

        class _GemmA16W16Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, weight, bias):
                ctx.save_for_backward(input, weight)
                ctx.has_bias = bias is not None
                output = dispatch_gemm(input, weight, scaling_type="none", bias=bias)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                input, weight = ctx.saved_tensors
                grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
                input_flat = input.reshape(-1, input.shape[-1])
                grad_input = grad_flat @ weight
                grad_input = grad_input.reshape(input.shape)
                grad_weight = grad_flat.t() @ input_flat
                grad_bias = grad_flat.sum(0) if ctx.has_bias else None
                return grad_input, grad_weight, grad_bias

        count = 0
        for _name, module in model.named_modules():
            if isinstance(module, nn.Linear):

                def _make_aiter_forward(mod):
                    def _aiter_forward(input):
                        return _GemmA16W16Fn.apply(input, mod.weight, mod.bias)

                    return _aiter_forward

                module.forward = _make_aiter_forward(module)
                count += 1

        if count:
            _rank0_print(
                f"> Replaced {count} nn.Linear forward with AITER GEMM (ASM→HIP→Triton)"
            )

    def _patch_conv(self, model) -> None:
        """Route supported Conv2d and Conv3d modules through Lumen."""
        from atomrl.lumen.ops.conv import patch_conv

        count_2d, count_3d = patch_conv(model)
        if count_2d or count_3d:
            _rank0_print(
                f"> Replaced {count_2d} Conv2d and {count_3d} Conv3d forwards "
                "with Lumen AITER convolution"
            )

    def _apply_pre_quant(self, model) -> None:
        """Set module attributes that must exist before ``quant.enable()``.

        ``delay_wgrad`` and ``gradient_accumulation_fusion`` are only
        meaningful on Lumen-native parallel linear types.

        ``fp8_activation_store`` is set on both Lumen-native types **and**
        standard ``nn.Linear`` so that ``_replace_forward`` captures the
        flag and passes it into ``QuantizedLinearFunction``.
        """
        from torch import nn

        has_lumen_only = self.delay_wgrad or self.gradient_accumulation_fusion
        has_act_store = self.fp8_activation_store

        if not (has_lumen_only or has_act_store):
            return

        lumen_count = 0
        if has_lumen_only:
            try:
                from atomrl.lumen.modules.grouped_linear import LumenGroupedLinear
                from atomrl.lumen.modules.layernorm_linear import LumenLayerNormLinear
                from atomrl.lumen.modules.parallel_linear import (
                    LumenColumnParallelLinear,
                    LumenRowParallelLinear,
                )

                lumen_types = (
                    LumenColumnParallelLinear,
                    LumenRowParallelLinear,
                    LumenLayerNormLinear,
                    LumenGroupedLinear,
                )
            except ImportError:
                lumen_types = ()

            for module in model.modules():
                if lumen_types and isinstance(module, lumen_types):
                    if self.delay_wgrad and hasattr(module, "delay_wgrad"):
                        module.delay_wgrad = True
                    if self.gradient_accumulation_fusion and hasattr(
                        module, "gradient_accumulation_fusion"
                    ):
                        module.gradient_accumulation_fusion = True
                    if has_act_store:
                        module.fp8_activation_store = True
                    lumen_count += 1

        linear_count = 0
        if has_act_store:
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module.fp8_activation_store = True
                    linear_count += 1

        opts = []
        if self.delay_wgrad:
            opts.append("delay_wgrad")
        if self.gradient_accumulation_fusion:
            opts.append("gradient_accumulation_fusion")
        if has_act_store:
            opts.append("fp8_activation_store")
        total = lumen_count + linear_count
        if total:
            _rank0_print(
                f"> Pre-quant optimizations ({', '.join(opts)}) applied to "
                f"{total} modules ({lumen_count} Lumen, {linear_count} nn.Linear)"
            )

    def _apply_post_quant(self, model, manager) -> None:
        """Apply features that require the ScalingManager from ``quant.enable()``."""
        if self.fp8_checkpoint:
            if manager is not None:
                self._enable_fp8_checkpoint(manager)
            else:
                _rank0_print("> WARNING: fp8_checkpoint requires FP8 quantization")

        # LUMEN_WEIGHT_QUANT_ONCE registers weights for the FP8 param cache so the
        # weight is quantized (cast+transpose+amax) once per optimizer step and
        # reused across micro-batches, rather than re-quantized every forward.
        # Invalidation is driven by the optimizer post-step hook
        # (mark_fp8_params_stale), NOT tensor._version — see
        # register_fp8_param_optimizer_hook. This does NOT enable FP8 DP gather
        # (that is a separate forward-comm decision gated on fp8_param_gather).
        _weight_quant_once = os.environ.get("LUMEN_WEIGHT_QUANT_ONCE", "0") == "1"
        if self.fp8_param_gather or _weight_quant_once:
            if manager is not None:
                manager.enable_fp8_params(model)
                _tag = (
                    "FP8 param gather"
                    if self.fp8_param_gather
                    else "per-step weight quant"
                )
                _rank0_print(f"> {_tag} enabled ({manager.num_fp8_params} params)")
            else:
                _rank0_print("> WARNING: fp8 param cache requires FP8 quantization")

        if self.fp8_weight_cache:
            if manager is not None:
                import atomrl.lumen.quantize as quant

                count = quant.store_weights_fp8(model)
                _rank0_print(f"> FP8 weight cache enabled ({count} layers cached)")
            else:
                _rank0_print("> WARNING: fp8_weight_cache requires FP8 quantization")

    @staticmethod
    def _enable_fp8_checkpoint(manager) -> None:
        """Monkey-patch Megatron checkpoint to preserve FP8 scaling state.

        No-op if Megatron is not installed.
        """
        try:
            import megatron.core.tensor_parallel as tp_module
            import megatron.core.tensor_parallel.random as tp_random
        except ImportError:
            return

        from atomrl.lumen.utils.checkpoint import _FP8ScalingContext

        if hasattr(tp_module, "_lumen_fp8_checkpoint_patched"):
            return

        _original = tp_random.checkpoint

        def _patched(function, distribute_saved_activations, *args):
            ctx = _FP8ScalingContext()
            ctx.save(manager)
            orig_fn = function

            def wrapped(*a, **kw):
                ctx.restore(manager)
                return orig_fn(*a, **kw)

            return _original(wrapped, distribute_saved_activations, *args)

        tp_random.checkpoint = _patched
        tp_module.checkpoint = _patched
        tp_module._lumen_fp8_checkpoint_patched = True
        tp_module._lumen_fp8_checkpoint_original = _original
        _rank0_print("> FP8-aware activation checkpointing enabled")

    def _log_summary(self, manager) -> None:
        parts = [f"format={self.format}", f"scaling={self.scaling}"]
        if self.fp8_param_manager:
            parts.append("fp8_param_manager")
        if self.lora_rank > 0:
            parts.append(f"lora_rank={self.lora_rank}")
        if self.fp8_attn != "none":
            parts.append(f"fp8_attn={self.fp8_attn}")
        if self.lumen_norm:
            parts.append("lumen_norm")
        if self.hf_attn_patch:
            parts.append("hf_attn_patch")
        if self.lumen_linear:
            parts.append("lumen_linear")
        if self.aiter_conv:
            parts.append("aiter_conv")
        if self.dit_sdpa_efficient:
            parts.append("dit_sdpa_efficient")
        if self.dit_sdpa_triton_forward:
            parts.append("dit_sdpa_triton_forward")
        if self.diffusers_rmsnorm_fuse:
            parts.append("diffusers_rmsnorm_fuse")
        if self.use_8bit_adam:
            parts.append("use_8bit_adam")
        tier3 = [
            name
            for name in (
                "fused_mlp",
                "cpu_offload",
                "delay_wgrad",
                "gradient_accumulation_fusion",
                "fp8_param_gather",
                "fp8_weight_cache",
                "fused_rope",
                "hip_graphs",
                "fp8_checkpoint",
                "fp8_activation_store",
                "local_shard_adamw",
            )
            if getattr(self, name, False)
        ]
        if tier3:
            parts.append(f"features=[{', '.join(tier3)}]")
        _rank0_print(f"> LumenConfig enabled ({', '.join(parts)})")

    # -----------------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------------

    @classmethod
    def from_args(cls, args) -> LumenConfig:
        """Build from an argparse ``Namespace`` (replaces manual ``getattr`` blocks).

        Iterates :data:`_ARG_MAP` to find matching attributes on *args*.
        Unknown / missing attributes are silently skipped (defaults apply).

        ``linear_fp8`` and ``linear_fp4`` are independent, mutually exclusive
        public gates. ``linear_fp4`` selects the fixed MXFP4 recipe; MXFP4 is
        deliberately not accepted through the FP8 format selector.
        """
        linear_fp8_enabled = getattr(args, "linear_fp8", None)
        linear_fp4_enabled = getattr(args, "linear_fp4", False)

        check_linear_quant_exclusive(linear_fp8_enabled, linear_fp4_enabled)

        kwargs: dict = {}
        for field_name, arg_names in _ARG_MAP.items():
            for arg_name in arg_names:
                val = getattr(args, arg_name, None)
                if val is not None:
                    kwargs[field_name] = val
                    break

        if linear_fp4_enabled:
            kwargs["format"] = "mxfp4"
            kwargs["scaling"] = "blockwise"
            kwargs["block_size"] = 32
        elif linear_fp8_enabled:
            if kwargs.get("format") == "mxfp4":
                raise ValueError(
                    "MXFP4 must be enabled with --linear-fp4, not --linear-fp8-format"
                )
        elif linear_fp8_enabled is False:
            kwargs["scaling"] = "none"

        return cls(**kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> LumenConfig:
        """Build from a plain dict (e.g. YAML / JSON config)."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)
