# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FSDP2 training backend: real HF model loading + Lumen FP8 + FSDP2 sharding."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from torch import nn

from atomrl.moe.router_precision import enable_fp32_moe_router

logger = logging.getLogger(__name__)


class _TinyLM(nn.Module):
    """Compact causal LM for tests and offline development."""

    def __init__(
        self, vocab_size: int = 32000, dim: int = 256, n_layers: int = 2
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(dim, nhead=4, batch_first=True)
                for _ in range(n_layers)
            ]
        )
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


def _patch_hf_attention_with_lumen() -> None:
    """Patch HF sdpa attention to use AITER CK kernels via Lumen.

    Uses ``atomrl.lumen.ops.attention.hf_patch`` which replaces the ``sdpa``
    entry in HF's ``ALL_ATTENTION_FUNCTIONS`` with an AITER-backed
    implementation that supports both forward and backward.
    """
    try:
        from atomrl.lumen.ops.attention.hf_patch import patch_hf_sdpa

        patch_hf_sdpa()
    except ImportError:
        logger.warning(
            "Lumen HF attention patch not available; using default SDPA/AOTriton."
        )
    except Exception:
        logger.exception("Lumen HF attention patch failed")


def _install_packing_attention() -> None:
    """Install the varlen-aware wrapper used by the packed forward.

    Independent of the Lumen AITER patch: it wraps whatever is registered as
    HF ``sdpa`` at call time and only diverges inside a ``PackingContext``,
    where it dispatches to ``aiter.flash_attn_varlen_func`` with ``cu_seqlens``.
    Without it a packed micro-batch degrades to one causal block over the whole
    buffer, so sequences attend across each other's boundaries and multi-sequence
    packing is unusable. Must run after ``patch_hf_sdpa`` when that is enabled,
    so the Lumen kernel becomes the non-packed fallback.
    """
    try:
        from atomrl.engine.training.packing import patch_attention_for_packing

        patch_attention_for_packing()
    except (AttributeError, RuntimeError) as exc:
        logger.warning("Packing attention patch failed: %s", exc)


# Architectures transformers ships as eager-only. DeepSeek-V4 sets
# `_supports_sdpa = False` (head_dim 512 exceeds FlashAttention's 256 cap, and
# torch SDPA carries no per-head learnable sink term), so the hardcoded
# `attn_implementation="sdpa"` below would raise before any weight is read.
_EAGER_ONLY_MODEL_TYPES = frozenset({"deepseek_v4"})


def _patch_deepseek_v4_grouped_linear() -> None:
    """Replace `DeepseekV4GroupedLinear`'s batched GEMM with one plain GEMM per group.

    Stock `forward` hands `torch.bmm` two non-contiguous views. On gfx950 / ROCm 7.0
    the `x` one -- whose batch stride is smaller than the matrix extent, so the
    groups interleave in memory -- is read out of bounds once the token count
    reaches ~2040: a hard memory fault in isolation, silent inf/NaN in the model.
    See rocm-gfx950-strided-bmm-oob-issue.md.

    The obvious fix, `.contiguous()` on both operands, trades one bug for another:
    it fixes the forward, but making `w` contiguous makes the *backward* produce
    NaN in 85 of 102 gradients (measured at 1024 tokens, where the forward bug is
    not even in play; `.contiguous()` on `x` alone is harmless). Rather than rely
    on which strides happen to dodge both, drop the batched GEMM: 8 separate 2D
    GEMMs were clean in every direction and at every length tested, and match the
    stock gradients bit-for-bit where stock is correct (max grad norm 18.5 either
    way at 1024 tokens). Cost is 8 kernel launches per call instead of 1.

    Must run before `from_pretrained`: under a device_map, accelerate stores the
    hooked forward as an instance attribute that shadows the class method, so a
    later class-level patch is silently inert.
    """
    try:
        from transformers.models.deepseek_v4 import modeling_deepseek_v4 as m
    except ImportError:
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim)  # (groups, out, in)
        xs = x.reshape(-1, self.n_groups, hidden_dim)  # (tokens, groups, in)
        outs = [
            torch.nn.functional.linear(xs[:, g].contiguous(), w[g])
            for g in range(self.n_groups)
        ]
        return torch.stack(outs, dim=1).reshape(*input_shape, self.n_groups, -1)

    m.DeepseekV4GroupedLinear.forward = forward
    logger.info("Patched DeepseekV4GroupedLinear.forward (per-group GEMM)")


def _unify_param_dtype(model: nn.Module, torch_dtype: torch.dtype, rank: int) -> None:
    """Force one floating dtype across the model, because FSDP2 requires it.

    ``_init_mp_dtypes`` asserts every parameter in an FSDP unit shares a dtype
    ("FSDP expects uniform original parameter dtype"). Two things break that for
    DeepSeek-V4: its ``_keep_in_fp32_modules_strict`` pins the hyper-connection
    params, attention sinks, position biases and norms to fp32, and the FP8
    dequantize path emits the config's bf16 for everything else, ignoring the
    ``torch_dtype`` asked for here. The result is a bf16/fp32 mix in the same
    decoder layer.

    Casting up to the requested dtype (fp32 for the master weights) is the
    direction that loses nothing. Models that already load uniform -- Qwen3 and
    everything else on this path -- skip the cast entirely.
    """
    dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
    if len(dtypes) <= 1 and dtypes <= {torch_dtype}:
        return
    logger.info(
        "[rank %d] Mixed parameter dtypes %s; casting to %s for FSDP2.",
        rank,
        sorted(str(d) for d in dtypes),
        torch_dtype,
    )
    model.to(torch_dtype)


def _hf_load_overrides(model_name: str) -> dict[str, Any]:
    """Per-architecture `from_pretrained` kwargs, derived from the checkpoint config.

    Two cases so far, both DeepSeek-V4:
      * eager-only architectures, which reject the default `sdpa`;
      * natively FP8 block-quantized checkpoints. transformers would otherwise
        load them as FP8 parameters, and `FineGrainedFP8HfQuantizer.is_trainable`
        is False -- the actor would come up untrainable. `dequantize=True` folds
        the per-block scales in at load time and yields ordinary dense weights.
    """
    from transformers import AutoConfig

    overrides: dict[str, Any] = {}
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except (ImportError, OSError, ValueError) as exc:
        logger.warning(
            "Could not read HF config for %s (%s); using defaults.", model_name, exc
        )
        return overrides

    model_type = getattr(cfg, "model_type", "")
    if model_type in _EAGER_ONLY_MODEL_TYPES:
        overrides["attn_implementation"] = "eager"
    if model_type == "deepseek_v4":
        _patch_deepseek_v4_grouped_linear()

    qcfg = getattr(cfg, "quantization_config", None)
    if isinstance(qcfg, dict) and qcfg.get("quant_method") == "fp8":
        from transformers import FineGrainedFP8Config

        kwargs = {k: v for k, v in qcfg.items() if k not in ("quant_method", "fmt")}
        overrides["quantization_config"] = FineGrainedFP8Config(
            dequantize=True, **kwargs
        )

    if overrides:
        logger.info(
            "HF load overrides for %s (%s): %s",
            model_name,
            model_type,
            sorted(overrides),
        )
    return overrides


_DTYPE_ALIASES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def _resolve_dtype(value: Any, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """Resolve a string or torch.dtype config value to a torch.dtype."""
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        try:
            return _DTYPE_ALIASES[value.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype string: {value!r}") from exc
    raise TypeError(f"Cannot resolve dtype from {value!r}")


def _load_hf_model(
    model_name: str, torch_dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    """Load a HuggingFace causal LM with gradient checkpointing.

    All ranks load the full model from disk to ensure identical weights
    before FSDP2 sharding. Model files on /dev/shm make this fast.
    """
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM

    rank = dist.get_rank() if dist.is_initialized() else 0

    # The Lumen/AITER attention patch replaces HF "sdpa" with an AITER varlen
    # flash-attn kernel. It targets packed (varlen) inputs; on plain padded
    # batches (e.g. the Ray controller path, which does not pack) its backward
    # can fault on ROCm. Allow opting out to fall back to pure torch SDPA,
    # matching the vanilla verl BF16 baseline.
    if os.environ.get("LUMEN_DISABLE_HF_ATTN_PATCH", "0") == "1":
        logger.info(
            "[rank %d] HF attention patch disabled; using pure torch SDPA.", rank
        )
    else:
        _patch_hf_attention_with_lumen()

    # Always installed: the packed forward needs varlen isolation regardless of
    # whether the Lumen AITER kernel is in play.
    _install_packing_attention()

    logger.info(
        "[rank %d] Loading HF model: %s (dtype=%s)", rank, model_name, torch_dtype
    )
    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
    }
    load_kwargs.update(_hf_load_overrides(model_name))
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    _unify_param_dtype(model, torch_dtype, rank)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # MoE only: a BF16 router makes top-k expert selection disagree with the
    # rollout engine on ~6% of (token, layer) decisions, which shows up as a
    # large rollout_corr/kl. Must stay in sync with the rollout-side patch in
    # atomrl/engine/inference/vllm_colocate_worker_ext.py.
    enable_fp32_moe_router(model)

    logger.info(
        "[rank %d] Model ready: %s (%d params)",
        rank,
        model_name,
        sum(p.numel() for p in model.parameters()),
    )
    return model


def _apply_lumen_fp8(model: nn.Module, quant_config: dict[str, Any]) -> nn.Module:
    """Apply Lumen optimizations to the model before FSDP2 sharding.

    Supports both FP8 and BF16 Lumen features (norm, fused_mlp, fused_rope).
    """
    if not quant_config:
        return model

    fp8_enabled = quant_config.get("fp8") or os.environ.get("LUMEN_FP8", "0") == "1"
    fp8_pm = (
        quant_config.get("fp8_param_manager")
        or os.environ.get("FP8_PARAM_MANAGER", "0") == "1"
    )
    lumen_norm = (
        quant_config.get("lumen_norm") or os.environ.get("LUMEN_NORM", "0") == "1"
    )
    fused_mlp = (
        quant_config.get("fused_mlp") or os.environ.get("LUMEN_FUSED_MLP", "0") == "1"
    )
    fused_rope = (
        quant_config.get("fused_rope") or os.environ.get("LUMEN_FUSED_ROPE", "0") == "1"
    )
    # ATOM rollout op-alignment (verl FP8 recipe: LUMEN_ROLLOUT=ATOM). Patches
    # norm/sdpa/linear/mlp with AITER kernels so the training-side forward matches
    # the AITER-based vLLM rollout. Works in pure BF16 (no FP8 quant).
    lumen_rollout = quant_config.get("rollout") or os.environ.get("LUMEN_ROLLOUT", "")

    if not (
        fp8_enabled or fp8_pm or lumen_norm or fused_mlp or fused_rope or lumen_rollout
    ):
        return model

    # Alignment mode: delegate the exact FP8 model modification to verl's
    # ``_maybe_apply_lumen`` (same LumenConfig.enable + lm_head BF16 restore),
    # so the native actor's FP8 training path is byte-identical to verl's
    # validated recipe. Enabled with ATOMRL_FP8_VIA_VERL=1 (needs verl importable
    # + MODEL_NAME env set). Keeps the native verl-free default for BF16/rollout.
    if os.environ.get("ATOMRL_FP8_VIA_VERL", "0") == "1":
        try:
            from verl.utils.fsdp_utils import _maybe_apply_lumen

            _maybe_apply_lumen(model, forward_only=False)
            logger.info("FP8 applied via verl _maybe_apply_lumen (aligned recipe).")
            return model
        except Exception:
            logger.exception("verl _maybe_apply_lumen delegation failed")
            return model

    try:
        from atomrl.lumen.config import LumenConfig

        # Respect LUMEN_DISABLE_HF_ATTN_PATCH: the native Ray path runs a validated
        # packed single-sequence pure-SDPA forward and installs its own packing
        # attention wrapper. Letting Lumen re-patch SDPA on top corrupts the
        # padded/packed attention layout (garbage logits -> exploding grad_norm /
        # rollout_corr/kl). Keep FP8 *linear* + norm while leaving attention to
        # the native forward unless the user explicitly opts into Lumen attention.
        hf_attn_patch = os.environ.get("LUMEN_DISABLE_HF_ATTN_PATCH", "0") != "1"
        kwargs = {
            "fp8_param_manager": bool(fp8_pm),
            "lumen_norm": bool(lumen_norm),
            "fused_mlp": bool(fused_mlp),
            "fused_rope": bool(fused_rope),
            "hf_attn_patch": hf_attn_patch,
            "fp8_weight_cache": quant_config.get("fp8_weight_cache", False),
        }
        if lumen_rollout:
            kwargs["rollout"] = lumen_rollout
        if fp8_enabled:
            # FP8 mode: enable quantized linear and related features. Attention
            # FP8 knobs (fp8_attn/attn_quant_type/attn_backend) are forwarded so
            # that ``LUMEN_FP8_ATTN=mha`` etc. take effect — matching verl's
            # ``_maybe_apply_lumen`` (per_block_fp8 recipe uses mha/blockwise).
            kwargs.update(
                scaling=os.environ.get("LUMEN_FP8_SCALING", "delayed"),
                format=os.environ.get("LUMEN_FP8_FORMAT", "fp8_e4m3"),
                block_size=int(os.environ.get("LUMEN_FP8_BLOCK_SIZE", "128")),
                fp8_activation_store=os.environ.get("LUMEN_FP8_ACTIVATION_STORE", "0")
                == "1",
                fp8_param_gather=os.environ.get("LUMEN_FP8_PARAM_GATHER", "0") == "1",
                fp8_attn=os.environ.get("LUMEN_FP8_ATTN", "none"),
                attn_quant_type=os.environ.get("LUMEN_FP8_QUANT_TYPE", "blockwise"),
                attn_backend=os.environ.get("LUMEN_ATTN_BACKEND", "auto"),
                fp8_wgrad=os.environ.get(
                    "LUMEN_FP8_WGRAD",
                    os.environ.get("FP8_WGRAD", "1"),
                )
                != "0",
            )
        else:
            # BF16 mode: no FP8 quantized linear (scaling="none"); ATOM/norm/attn
            # patches still apply with AITER BF16 kernels.
            kwargs.update(
                scaling="none",
                quantize_activation=False,
                fp8_wgrad=False,
            )
        cfg = LumenConfig(**kwargs)
        _manager, model = cfg.enable(model)

        # verl-aligned (fsdp_utils._maybe_apply_lumen): restore lm_head to BF16.
        # The lm_head FP8 blockscale GEMM overflows INT32 on the huge vocab GEMM
        # (vocab ~152k x hidden 4096) -> garbage logits -> exploding grad_norm /
        # near-zero entropy / policy collapse. Keep every other Linear in FP8 but
        # run lm_head in BF16. Only relevant when FP8 quantized linear is active.
        if fp8_enabled:
            from torch import nn

            restored = 0
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Linear) and "lm_head" in name:
                    if getattr(mod, "_quant_enabled", False):
                        mod._quant_enabled = False
                    mod.forward = nn.Linear.forward.__get__(mod, nn.Linear)
                    restored += 1
            if restored:
                logger.info(
                    "Restored %d lm_head Linear(s) to BF16 (CK blockscale INT32 overflow)",
                    restored,
                )

        logger.info(
            "Lumen optimizations applied (fp8=%s, fp8pm=%s, norm=%s, fused_mlp=%s, "
            "fused_rope=%s, rollout=%s)",
            fp8_enabled,
            fp8_pm,
            lumen_norm,
            fused_mlp,
            fused_rope,
            lumen_rollout or "(none)",
        )
    except ImportError:
        logger.warning(
            "atomrl.lumen package not installed; skipping Lumen optimizations."
        )
    except Exception:
        logger.exception("Lumen enable() failed")

    return model


def _apply_fsdp2_sharding(
    model: nn.Module,
    fsdp_config: dict[str, Any],
    fp8_linear: bool = False,
) -> nn.Module:
    """Apply PyTorch FSDP2 to the model.

    Strategy is controlled by ``fsdp_config["strategy"]``:
      - ``"full_shard"`` (default): Full FSDP2 sharding on all layers + root.
      - ``"replicate"``: Pure DDP-like gradient all-reduce via composable replicate.
    """
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        # No process group: no sharding, but the model still has to live on the
        # device the rest of the trainer uses.
        logger.warning("torch.distributed not initialized; returning unsharded model.")
        return model.to(local_device) if torch.cuda.is_available() else model

    rank = dist.get_rank()

    model.to(local_device)

    strategy = fsdp_config.get("strategy", "full_shard").lower()

    if strategy == "replicate":
        from torch.distributed._composable.replicate import replicate

        replicate(
            model, device_mesh=dist.init_device_mesh("cuda", (dist.get_world_size(),))
        )
        logger.info(
            "[rank %d] FSDP2 replicate applied (pure DDP, reduce_dtype=float32)",
            rank,
        )
        return model

    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

    from atomrl.engine.training.fsdp_chunk_cat_fallback import (
        install_chunk_cat_fallback,
    )

    install_chunk_cat_fallback()

    param_dtype = _resolve_dtype(fsdp_config.get("param_dtype"), torch.bfloat16)
    reduce_dtype = _resolve_dtype(fsdp_config.get("reduce_dtype"), torch.float32)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )

    reshard = fsdp_config.get("reshard_after_forward", True)
    offload_policy = None
    if fsdp_config.get("param_offload", False):
        from torch.distributed._composable.fsdp import CPUOffloadPolicy

        offload_policy = CPUOffloadPolicy(pin_memory=True)

    for module in model.modules():
        if hasattr(module, "layers") and isinstance(module.layers, nn.ModuleList):
            for layer in module.layers:
                fully_shard(
                    layer,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard,
                    offload_policy=offload_policy,
                )

    fully_shard(
        model,
        mp_policy=mp_policy,
        reshard_after_forward=reshard,
        offload_policy=offload_policy,
    )

    logger.info(
        "[rank %d] FSDP2 full_shard applied (fp8_reduce=%s, offload=%s)",
        rank,
        fp8_linear,
        fsdp_config.get("param_offload", False),
    )
    return model


def set_requires_gradient_sync(model: nn.Module, requires_sync: bool) -> None:
    """Toggle gradient sync on all FSDP2 units in *model*.

    For gradient accumulation: set to ``False`` on intermediate
    micro-batches so that FSDP2 skips the reduce-scatter in backward,
    letting gradients accumulate locally.  Set back to ``True`` on the
    last micro-batch of each accumulation group so that the final
    backward performs the reduce-scatter.
    """
    from torch.distributed._composable.fsdp import FSDPModule

    if isinstance(model, FSDPModule):
        model.set_requires_gradient_sync(requires_sync, recurse=True)
    else:
        for mod in model.modules():
            if isinstance(mod, FSDPModule):
                mod.set_requires_gradient_sync(requires_sync, recurse=True)
                break


def set_reshard_after_forward(model: nn.Module, reshard: bool) -> None:
    """Toggle ``reshard_after_forward`` on all FSDP2 units in *model*.

    During generation, setting this to ``False`` keeps parameters
    all-gathered across decode steps, eliminating one all-gather per
    layer per token.  Restore to ``True`` before training so that
    memory stays bounded.

    Uses the official ``FSDPModule.set_reshard_after_forward`` API
    (PyTorch >= 2.6).
    """
    from torch.distributed._composable.fsdp import FSDPModule

    if isinstance(model, FSDPModule):
        model.set_reshard_after_forward(reshard, recurse=True)
    else:
        for mod in model.modules():
            if isinstance(mod, FSDPModule):
                mod.set_reshard_after_forward(reshard, recurse=True)
                break


class FSDP2Backend:
    """Build a model and optionally wrap it with PyTorch FSDP2."""

    @staticmethod
    def build_model(model_name: str, config: dict[str, Any] | None = None) -> nn.Module:
        """Construct the trainable policy network.

        If ``model_name`` points to a real HF model (local path or hub ID),
        loads it. Otherwise falls back to ``_TinyLM`` for testing.
        """
        cfg = config or {}

        # Apply Liger kernel monkey-patches BEFORE model loading
        if cfg.get("use_liger", False):
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_llama

                apply_liger_kernel_to_llama()
                logger.info("Applied Liger kernel optimizations.")
            except ImportError:
                logger.warning(
                    "use_liger=True but liger-kernel not installed. Skipping."
                )

        if cfg.get("use_tiny_lm", False) or not model_name:
            arch = cfg.get("tiny_lm", {})
            vocab = int(arch.get("vocab_size", 32000))
            dim = int(arch.get("dim", 256))
            n_layers = int(arch.get("n_layers", 2))
            logger.info(
                "FSDP2Backend.build_model: TinyLM (vocab=%d, dim=%d)", vocab, dim
            )
            return _TinyLM(vocab_size=vocab, dim=dim, n_layers=n_layers)

        model_dtype = _resolve_dtype(cfg.get("model_dtype"), torch.bfloat16)
        return _load_hf_model(model_name, torch_dtype=model_dtype)

    @staticmethod
    def apply_lumen_optimizations(
        model: nn.Module, quant_config: dict[str, Any] | None
    ) -> nn.Module:
        """Apply Lumen FP8/norm optimizations before FSDP2 wrapping."""
        return _apply_lumen_fp8(model, quant_config or {})

    @staticmethod
    def apply_fsdp2(model: nn.Module, fsdp_config: dict[str, Any] | None) -> nn.Module:
        """Shard ``model`` with FSDP2 ``fully_shard``."""
        if not fsdp_config or not fsdp_config.get("enabled", True):
            return model
        fp8_linear = os.environ.get("LUMEN_FP8", "0") == "1"
        return _apply_fsdp2_sharding(model, fsdp_config, fp8_linear=fp8_linear)
