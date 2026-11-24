# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""
atomrl.lumen.quantize — low-precision training lifecycle for AMD GPUs.

Supports FP8 (E4M3 / E5M2), MXFP8, MXFP4, and FP4 formats.

Usage::

    import atomrl.lumen.quantize as quant
    from atomrl.lumen.quantize import QuantConfig, QuantFormat, ScalingType

    # Configure quantization
    config = QuantConfig(format=QuantFormat.FP8_E4M3,
                         scaling=ScalingType.DELAYED)

    # Non-invasive: patch existing model, no module replacement
    quant.enable(model, config=config)

    # Or use string shorthand
    quant.enable(model, format="fp8_e4m3", scaling="delayed")

    # Check backend availability
    backend = quant.get_attention_backend()   # "aiter_csrc" or "aiter_triton"
"""

import logging
import os as _os
import re

import torch
from torch import nn

from atomrl.lumen.ops.quantize import (
    convert_from_mxfp8,
    convert_to_mxfp8,
    dequant_fp8_tensorwise_impl,
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
    quant_fp8_tensorwise_impl,
)
from atomrl.lumen.quantize.comm_tensor import Blockwise2DFP8Gathered, FP8CommTensor
from atomrl.lumen.quantize.config import (
    AmaxAlgo,
    QuantConfig,
    QuantFormat,
    ScalingType,
    get_fp8_max,
    get_fp8_max_bwd,
)
from atomrl.lumen.quantize.descriptor import FP8Descriptor
from atomrl.lumen.quantize.optimizer_manager import (
    FP32MasterWeightOptimizer,
    get_scaling_manager,
)
from atomrl.lumen.quantize.scaling_manager import (
    GRAD_QUANT_TYPES,
    ScalingManager,
)

__all__ = [
    "GRAD_QUANT_TYPES",
    "AmaxAlgo",
    "Blockwise2DFP8Gathered",
    "FP8CommTensor",
    "FP8Descriptor",
    "FP32MasterWeightOptimizer",
    "QuantConfig",
    "QuantFormat",
    "QuantizedLinearFunction",
    "ScalingManager",
    "ScalingType",
    "assert_mxfp4_arch_supported",
    "convert_from_mxfp8",
    "convert_to_mxfp8",
    "dequant_fp8_tensorwise_impl",
    "disable",
    "enable",
    "get_attention_backend",
    "get_fp8_max",
    "get_fp8_max_bwd",
    "get_quant_backend",
    "get_scaling_manager",
    "is_under_bf16_prefix",
    "quant_fp8_blockwise_impl",
    "quant_fp8_blockwise_segment_m_impl",
    "quant_fp8_tensorwise_impl",
    "quantized_linear",
    "register_fp8_weight_optimizer_hooks",
    "register_mxfp4_weight_optimizer_hooks",
    "store_weights_fp8",
    "store_weights_fp8_blockwise2d",
]

logger = logging.getLogger(__name__)


def __getattr__(name):
    if name not in ("QuantizedLinearFunction", "quantized_linear"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from atomrl.lumen.ops.quantize.linear import (
        QuantizedLinearFunction,
        quantized_linear,
    )

    return {
        "QuantizedLinearFunction": QuantizedLinearFunction,
        "quantized_linear": quantized_linear,
    }[name]


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def get_attention_backend(prefer: str = "auto") -> str:
    """Determine which attention backend to use.

    Args:
        prefer: One of ``"auto"``, ``"aiter_csrc"``, ``"aiter_triton"``.

    Returns:
        ``"aiter_csrc"`` or ``"aiter_triton"``
    """
    if prefer == "aiter_triton":
        return "aiter_triton"

    return "aiter_csrc"


def get_quant_backend(prefer: str = "auto") -> str:
    """Determine which quantization backend to use.

    Args:
        prefer: One of ``"auto"``, ``"aiter"``, ``"triton"``.

    Returns:
        ``"aiter"`` or ``"triton"``
    """
    if prefer == "triton":
        return "triton"
    return "aiter"


# ---------------------------------------------------------------------------
# Quantization enablement
# ---------------------------------------------------------------------------

# MXFP4 is supported on gfx950 (MI350/MI355) and nowhere else.
#
# The FP4 conversion has two implementations. gfx950 has the arithmetic in
# hardware -- ``v_cvt_scalef32_[sr_]pk_fp4_{f32,bf16}``, which round correctly
# and whose SR is unbiased by construction -- and that is the path this feature
# was built, measured and trained on. Every other architecture lands in the
# Triton software fallback in ``atomrl.lumen.kernels.mxfp4._pack_fp4``, which is not
# equivalent to it:
#
#   - SR's dither is one-sided. ``tl.randint4x`` returns signed int32, so the
#     noise lands in [-0.5, 0.5) and ``(noise - 0.5) * 0.01`` is never
#     positive: a debiasing term with mean -0.005, and about 200x too small.
#   - RTN breaks ties away from even. All seven boundaries use ``>=``, and
#     E2M1's even codes are 0.0/1.0/2.0/4.0, so 0.25, 1.25, 2.50 and 5.00 all
#     round the wrong way. They err in the same direction, so they do not
#     cancel -- the result is systematic magnitude inflation.
#   - The dither pattern repeats every BLOCK_M x BLOCK_N, because the Philox
#     offset carries no program id. Rounding errors at the same position in
#     each tile are then fully correlated, which is exactly the correlation the
#     reduction along K is supposed to average away.
#   - A block whose amax is subnormal is off by 2x: the scale is clamped up for
#     packing, but the unclamped byte is what gets stored.
#
# Fixing those is a numerics change that needs the precision harness behind it,
# and until it has one, a fallback with no test that forces ``use_asm=False``
# must not be presented as MI300X support. So the support matrix is gfx950, and
# this is where it is enforced rather than discovered as a quiet accuracy loss.
_MXFP4_SUPPORTED_ARCHS = ("gfx950",)

_MXFP4_ARCH_OVERRIDE_ENV = "LUMEN_MXFP4_ALLOW_UNVALIDATED_ARCH"


def assert_mxfp4_arch_supported() -> None:
    """Refuse MXFP4 on an architecture whose FP4 conversion is not validated.

    A no-op when the format is not MXFP4, when no GPU is visible (the arch is
    not knowable then, and nothing will run either), or when
    ``LUMEN_MXFP4_ALLOW_UNVALIDATED_ARCH=1`` opts in deliberately.
    """
    if _os.environ.get(_MXFP4_ARCH_OVERRIDE_ENV) == "1":
        return
    if not torch.cuda.is_available():
        return

    from atomrl.lumen.ops.quantize.ops import triton_arch

    arch = triton_arch()
    # An unreadable arch is not evidence of an unsupported one; the GEMM
    # dispatch will still refuse a chip that has no FP4 kernel.
    if not arch or arch in _MXFP4_SUPPORTED_ARCHS:
        return

    raise RuntimeError(
        f"MXFP4 is supported on {'/'.join(_MXFP4_SUPPORTED_ARCHS)} and this is "
        f"{arch}. The hardware FP4 conversion instructions only exist on gfx950; "
        "elsewhere the Triton software fallback takes over, and its rounding is "
        "known to be wrong (one-sided SR dither, ties rounded away from even, a "
        "dither pattern that repeats per tile, and a 2x error on subnormal-amax "
        "blocks). It has no test that exercises it, so it is not MI300X support. "
        f"Set {_MXFP4_ARCH_OVERRIDE_ENV}=1 to run it anyway, for kernel work "
        "rather than for training."
    )


def enable(
    model,
    config: QuantConfig | None = None,
    *,
    format: str = "fp8_e4m3",
    scaling: str = "delayed",
    backend: str = "auto",
    recipe: str | None = None,
    dp_group=None,
    **kwargs,
) -> ScalingManager:
    """Non-invasive: patch existing model's ``nn.Linear`` layers with FP8
    quantized forward/backward.

    Every ``nn.Linear`` in *model* gets a forward hook that quantizes
    input and weight, runs an FP8 GEMM, and dequantizes the output — all
    transparently.  The training loop does not need to change.

    Args:
        model: The ``nn.Module`` to patch.
        config: A :class:`QuantConfig`. If provided, ``format``/``scaling``
                kwargs are ignored.
        format: Shorthand format string (ignored when *config* is given).
        scaling: Shorthand scaling string (ignored when *config* is given).
        backend: ``"auto"``, ``"aiter"``, or ``"triton"``.
        recipe: Legacy alias — maps to *scaling*. Deprecated.
        dp_group: Data-parallel process group for ``reduce_amax``.
            Required when ``config.reduce_amax=True``.
        **kwargs: Forwarded to :class:`QuantConfig` (e.g. ``block_size``).

    Returns:
        The :class:`ScalingManager` instance attached to the model.
    """
    if config is None:
        if recipe is not None:
            scaling = recipe
        config = QuantConfig.from_str(format=format, scaling=scaling, **kwargs)

    if config.format == QuantFormat.MXFP4:
        assert_mxfp4_arch_supported()

    resolved_backend = get_quant_backend(backend)
    manager = ScalingManager(config)

    if config.reduce_amax and dp_group is not None:
        manager.set_dp_group(dp_group)

    _patch_linear_layers(model, manager, resolved_backend, config)
    return manager


def _get_megatron_linear_types():
    """Return Megatron's parallel linear types if available, else empty tuple."""
    try:
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        return ColumnParallelLinear, RowParallelLinear
    except ImportError:
        return ()


_LAYER_INDEX_RE = re.compile(r"layers\.(\d+)\b")

# Module leaf names of the vocab-projection (output) layer across HF / Megatron.
_OUTPUT_LAYER_NAMES = ("lm_head", "output_layer")


def _is_output_layer(name: str) -> bool:
    """True if *name* is the model's output/vocab-projection layer.

    Matched by the module path's leaf (``...lm_head`` or ``...output_layer``)
    so both HuggingFace (``lm_head``) and Megatron (``output_layer``) hit.
    """
    leaf = name.rsplit(".", 1)[-1]
    return leaf in _OUTPUT_LAYER_NAMES


def _build_bf16_skip_prefixes(
    model: nn.Module,
    config: QuantConfig,
) -> set[str]:
    """Return a set of module-name prefixes whose transformer layers should
    stay in BF16 (not be FP8-patched).

    Strategy:
    1. Walk the model looking for modules that expose a ``layer_number``
       attribute (Megatron ``TransformerLayer`` — 1-indexed global, correct
       even under pipeline parallelism).
    2. If none are found (HuggingFace / FSDP models), fall back to extracting
       the layer index from the module path (``layers.<N>``).
    """
    if not config.first_last_layers_bf16:
        return set()

    bf16_start = config.num_layers_at_start_in_bf16
    bf16_end = config.num_layers_at_end_in_bf16
    total = config.num_layers

    if total <= 0:
        # ``total - bf16_end`` goes non-positive, so _should_skip answers True
        # for every index and the caller keeps the whole model in BF16 with
        # nothing said. The Megatron native path guards this at its call site;
        # putting it here covers the generic path too, which has the same hole.
        logger.warning(
            "first_last_layers_bf16 is set but num_layers=%d, so the BF16 tail "
            "cannot be located; quantizing every layer instead of none. Set "
            "num_layers on the quant config to use this option.",
            total,
        )
        return set()

    def _should_skip(global_idx: int) -> bool:
        return global_idx < bf16_start or global_idx >= total - bf16_end

    prefixes: set[str] = set()

    # --- Strategy 1: Megatron layer_number (global, 1-indexed) ---
    for name, module in model.named_modules():
        layer_num = getattr(module, "layer_number", None)
        if layer_num is not None and isinstance(layer_num, int):
            global_idx = layer_num - 1
            if _should_skip(global_idx):
                prefixes.add(name)

    if prefixes:
        return prefixes

    # --- Strategy 2: path-based detection (HF / FSDP) ---
    layer_prefixes: dict[str, int] = {}
    for name, _ in model.named_modules():
        m = _LAYER_INDEX_RE.search(name)
        if m:
            prefix = name[: m.end()]
            idx = int(m.group(1))
            if prefix not in layer_prefixes or idx < layer_prefixes[prefix]:
                layer_prefixes[prefix] = idx

    if not layer_prefixes:
        return set()

    detected_max = max(
        int(m.group(1))
        for name, _ in model.named_modules()
        for m in [_LAYER_INDEX_RE.search(name)]
        if m
    )
    effective_total = total if total > 0 else detected_max + 1

    for name, _ in model.named_modules():
        m = _LAYER_INDEX_RE.search(name)
        if m:
            idx = int(m.group(1))
            if idx < bf16_start or idx >= effective_total - bf16_end:
                prefixes.add(name[: m.end()])

    return prefixes


def is_under_bf16_prefix(name: str, bf16_prefixes: set[str]) -> bool:
    """Whether ``name`` is one of the BF16 layers, or lives inside one.

    A plain ``startswith`` is wrong here: the prefixes are layer module paths
    ending in an index, so ``decoder.layers.1`` also prefixes ``decoder.layers.10``
    through ``.19``. That silently drags every one of those layers into BF16, and
    only for the configurations where a skipped index happens to prefix another
    one -- a tail of 5 in 36 layers (indices 31-35) is clean, while a *start* of 2
    pulls in 10-19 as well. Requiring the boundary dot makes the match exact.
    """
    return any(name == p or name.startswith(p + ".") for p in bf16_prefixes)


def _patch_linear_layers(
    model: nn.Module,
    manager: ScalingManager,
    backend: str,
    config: QuantConfig,
) -> None:
    """Hook quantized dispatch into every ``nn.Linear`` layer.

    Each layer gets a unique ``tensor_id`` derived from its module path so that
    :class:`ScalingManager` tracks independent amax histories per layer (fixes
    the shared-``"weight"`` bug where all layers polluted a single deque).

    Also handles Megatron's ``ColumnParallelLinear`` and ``RowParallelLinear``
    which do not inherit ``nn.Linear`` but expose the same ``.weight`` attribute.

    Gradient quantization is handled by the :class:`ScalingManager` itself
    (configured via ``config.quantize_grad``).
    """
    fp8_dtype = config.torch_dtype or torch.float8_e4m3fn
    block_size = config.block_size
    quant_act = config.quantize_activation
    fp8_wgrad = config.fp8_wgrad
    scaling_type = config.recipe

    megatron_types = _get_megatron_linear_types()
    quantizable_types = (nn.Linear,) + megatron_types

    bf16_prefixes = _build_bf16_skip_prefixes(model, config)

    count = 0
    skipped = 0
    for name, module in model.named_modules():
        if isinstance(module, quantizable_types):
            if bf16_prefixes and is_under_bf16_prefix(name, bf16_prefixes):
                skipped += 1
                continue

            if not config.quantize_output_layer and _is_output_layer(name):
                skipped += 1
                continue

            if "lora_" in name:
                skipped += 1
                continue

            tensor_id = f"{name}.weight" if name else "weight"
            module._quant_manager = manager
            module._quant_backend = backend
            module._quant_enabled = True
            module._quant_tensor_id = tensor_id
            module._quant_scaling_type = scaling_type
            # Decide frozen-weight caching NOW (before any FSDP wrap): PEFT has
            # already frozen the base weight, so requires_grad is reliable here.
            # Under FSDP the gathered view can falsely report requires_grad=True.
            _w = getattr(module, "weight", None)
            # Patch-time frozen fact (PEFT froze base, pre-FSDP → reliable). Used to
            # skip the frozen weight's (discarded) WGrad even without the FP8 cache
            # (e.g. 70B FSDP), gated at runtime by LUMEN_SKIP_FROZEN_WGRAD.
            module._lumen_frozen = _w is not None and not _w.requires_grad
            module._lumen_cache_frozen = (
                getattr(config, "cache_frozen_weight", False) and module._lumen_frozen
            )
            # bpreshuffle GEMM only makes sense with the frozen-weight cache (the
            # shuffle is amortized once).
            module._lumen_bpreshuffle = (
                getattr(config, "bpreshuffle_gemm", False)
                and module._lumen_cache_frozen
            )

            is_megatron = megatron_types and isinstance(module, megatron_types)
            _replace_forward(
                module,
                manager,
                backend,
                fp8_dtype,
                block_size,
                tensor_id,
                quant_act,
                fp8_wgrad,
                is_megatron,
                scaling_type,
            )
            count += 1

    act_str = "weight+activation" if quant_act else "weight-only"
    grad_quant_type = config.quantize_grad
    grad_str = f"+grad({grad_quant_type})" if grad_quant_type else ""
    bf16_str = f", bf16_layers_skipped={skipped}" if skipped else ""
    logger.info(
        "Quantization enabled on %d nn.Linear layers "
        "(backend=%s, format=%s, scaling=%s, amax_algo=%s, %s%s%s)",
        count,
        backend,
        config.format.value,
        config.scaling.value,
        config.amax_algo.value,
        act_str,
        grad_str,
        bf16_str,
    )


def _maybe_cache_frozen_weight(module, scaling_type, fp8_dtype, block_size):
    """Quantize a *frozen* weight to FP8 once and cache it on the module.

    When ``cache_frozen_weight`` is on, the base (LoRA-frozen) weight's FP8
    quant never changes, yet ``quantized_linear`` re-quantizes it every forward
    (and again on each gradient-checkpointing recompute) — ~43 GB/step of copies
    at 8B.  Compute it once here (in the forward, where FSDP has gathered the
    full weight) and stash ``_fp8_weight_data`` / ``_fp8_weight_scale``; the
    existing ``fp8_weight_cache`` path then feeds it straight to the GEMM.

    Guards: only for frozen weights, only when the gathered weight is the full
    2D tensor (skip FSDP-sharded views), and only for scalings whose cached
    weight scale matches what the GEMM expects (blockwise2d's 2D tile scale).
    """
    # _lumen_cache_frozen is set at patch time (before FSDP wrap) and already
    # encodes "frozen base weight" — do NOT re-check weight.requires_grad here:
    # under FSDP a frozen view can report requires_grad=True (flat-param taint).
    if not getattr(module, "_lumen_cache_frozen", False):
        return
    if getattr(module, "_fp8_weight_data", None) is not None:
        return
    weight = getattr(module, "weight", None)
    if weight is None or weight.dim() != 2:
        return
    # FSDP may expose a flattened/sharded view mid-init; require K divisible by
    # block_size so the blockwise quantizer's tiling is valid (full weight).
    if scaling_type == "blockwise2d" and (
        weight.shape[0] % block_size or weight.shape[1] % block_size
    ):
        return
    if scaling_type not in ("blockwise2d", "blockwise"):
        return  # per-tensor caches go through store_weights_fp8 separately
    from atomrl.lumen.ops.quantize.linear import quantize_input

    try:
        desc = quantize_input(
            weight.detach().contiguous(),
            scaling_type,
            fp8_dtype,
            block_size,
            None,
            getattr(module, "_quant_tensor_id", "weight"),
            is_weight=True,
        )
    except (AssertionError, RuntimeError) as e:
        logger.warning(
            "cache_frozen_weight: quant failed (%s); will re-quant per fwd", e
        )
        return
    module._fp8_weight_data = desc.data
    module._fp8_weight_scale = desc.scale
    # The weight is frozen → its WGrad is discarded; mark it so the backward can
    # skip the whole WGrad (dequant→requant + transpose + GEMM). Attached to the
    # (stable) cache tensor so the forward threads it onto ctx.
    desc.data._lumen_skip_wgrad = True
    # Pre-shuffle frozen weight into (N//16, K*16) layout for the Triton preshuffle
    # GEMM kernel.  One-time cost amortized over all forwards.
    if getattr(module, "_lumen_bpreshuffle", False) and scaling_type in (
        "blockwise",
        "blockwise2d",
    ):
        from aiter.ops.shuffle import shuffle_weight

        try:
            N_w, K_w = desc.data.shape
            wsh = shuffle_weight(desc.data, layout=(16, 16)).reshape(
                N_w // 16, K_w * 16
            )
            desc.data._lumen_wsh = wsh
        except RuntimeError as e:
            logger.warning(
                "preshuffle cache failed (%s); will use standard blockscale", e
            )
    # Also cache the transposed FP8 weight + scale (frozen → constant) so the
    # blockwise2d DGrad reuses it instead of doing weight_data.t().contiguous()
    # (~7.6 GB/step of copies) every backward.
    if scaling_type == "blockwise2d":
        try:
            data_t = desc.data.t().contiguous()
            desc.data._lumen_wt = (data_t, desc.scale.t().contiguous())
        except (AssertionError, RuntimeError):
            pass


def _mxfp4_cached_weight(
    module,
    weight,
    wcache,
    wscale,
    scaling_type,
    fp8_dtype,
    block_size,
    gemm_rows=None,
):
    """Quantize an MXFP4 weight once per optimizer step, not once per micro-batch.

    MXFP4 weight quantization is round-to-nearest, so every micro-batch of a
    gradient accumulation step re-derives byte-identical FP4 data, packed
    transpose and transposed scales from an unchanged weight.
    ``register_mxfp4_weight_optimizer_hooks`` drops the cache when
    ``optimizer.step()`` moves the master weights, so a stale cache cannot
    outlive the weight it came from.

    ``gemm_rows`` is the row count of the GEMMs this weight is about to serve,
    which is what decides whether either operand can be stored pre-shuffled.

    Returns the (data, scale) pair to hand the GEMM, unchanged when the caller
    already has a weight cache of its own or the format is not MXFP4.
    """
    if (
        wcache is not None
        or scaling_type != "mxfp4"
        or _os.environ.get("LUMEN_MXFP4_DISABLE_WEIGHT_CACHE") == "1"
    ):
        return wcache, wscale

    from atomrl.lumen.ops.quantize.linear import quantize_input
    from atomrl.lumen.ops.quantize.ops import transpose_packed_fp4

    cached = getattr(module, "_mxfp4_w_cache", None)
    # Optimizers update Parameters in-place and increment ``_version``. Keying
    # the cache on that generation makes the generic ``quant.enable(nn.Linear)``
    # path correct even when its training loop cannot register Lumen's optional
    # post-step hook. The hook remains useful for eagerly releasing old buffers,
    # but correctness must not depend on framework-specific optimizer wiring.
    weight_version = weight._version
    if (
        cached is not None
        and getattr(module, "_mxfp4_w_cache_version", None) == weight_version
    ):
        return cached[0], cached[1]

    desc = quantize_input(
        weight.contiguous(),
        "mxfp4",
        fp8_dtype,
        block_size,
        None,
        None,
        is_weight=True,
    )
    data, scale = desc.data, desc.scale

    data_t = transpose_packed_fp4(data)
    fwd_scale, dgrad_scale = scale, scale.t().contiguous()

    data._mxfp4_wt_cached = (data_t, dgrad_scale)
    module._mxfp4_w_cache = (data, fwd_scale)
    module._mxfp4_w_cache_version = weight_version
    return data, fwd_scale


def _replace_forward(
    module,
    manager,
    backend,
    fp8_dtype,
    block_size,
    tensor_id,
    quantize_activation,
    fp8_wgrad,
    is_megatron,
    scaling_type="delayed",
):
    """Replace the module's forward method with an FP8-quantized version.

    Unlike ``register_forward_hook``, this prevents the original (BF16) linear
    from running at all, saving both compute and peak memory — critical for
    70B-class models under FSDP.
    """
    original_forward = module.forward

    module._lumen_scaling_manager = manager
    module._lumen_scaling_type = scaling_type
    module._lumen_fp8_dtype = fp8_dtype
    module._lumen_act_tensor_id = tensor_id.replace(".weight", ".activation")

    _delay_wgrad = getattr(module, "delay_wgrad", False)
    _deferred_wgrad = getattr(module, "_deferred_wgrad", None) if _delay_wgrad else None
    _gaf = getattr(module, "gradient_accumulation_fusion", False)
    _fp8_act_store = getattr(module, "fp8_activation_store", False)

    def _get_pre_quant_weight():
        """Return (fp8_data, gemm_scale) if weight is stored in FP8, else None."""
        import torch as _torch

        w = module.weight
        if hasattr(w, "_fp8_desc"):
            if w._fp8_desc.data.data_ptr() != w.data.data_ptr():
                w._fp8_desc.invalidate_transpose()
                return None
            # blockwise/blockwise2d GEMM wants the DIRECT (2D) dequant scale; per-tensor
            # (delayed/dynamic, scalar) wants the reciprocal.
            _s = w._fp8_desc.scale
            _gs = _s if (isinstance(_s, _torch.Tensor) and _s.numel() > 1) else 1.0 / _s
            return (w._fp8_desc.data, _gs)
        if hasattr(w, "_fp8_scale") and w.dtype in (
            _torch.float8_e4m3fn,
            _torch.float8_e4m3fnuz,
            _torch.float8_e5m2,
        ):
            _s = w._fp8_scale
            _gs = _s if (isinstance(_s, _torch.Tensor) and _s.numel() > 1) else 1.0 / _s
            return (w.data, _gs)
        return None

    _act_tensor_id = tensor_id.replace(".weight", ".activation")

    if not is_megatron:

        def quant_forward(input_tensor, *args, **kwargs):
            from atomrl.lumen.ops.quantize.linear import quantized_linear

            _maybe_cache_frozen_weight(module, scaling_type, fp8_dtype, block_size)
            w = module.weight
            _wcache = getattr(module, "_fp8_weight_data", None)
            _wscale = getattr(module, "_fp8_weight_scale", None)
            if isinstance(w, Blockwise2DFP8Gathered):
                _wcache, _wscale = w._fp8, w._scale
                w._lumen_frozen = True
            elif getattr(module, "_lumen_frozen", False):
                w._lumen_frozen = True
            _wcache, _wscale = _mxfp4_cached_weight(
                module,
                w,
                _wcache,
                _wscale,
                scaling_type,
                fp8_dtype,
                block_size,
                gemm_rows=input_tensor.numel() // input_tensor.shape[-1],
            )
            return quantized_linear(
                input_tensor,
                w,
                module.bias,
                scaling_manager=manager,
                backend=backend,
                scaling_type=scaling_type,
                fp8_dtype=fp8_dtype,
                block_size=block_size,
                tensor_id=tensor_id,
                quantize_activation=quantize_activation,
                fp8_wgrad=fp8_wgrad,
                gradient_accumulation_fusion=_gaf,
                delay_wgrad=_delay_wgrad,
                deferred_wgrad=_deferred_wgrad,
                fp8_activation_store=_fp8_act_store,
                fp8_weight_cache=_wcache,
                fp8_weight_scale=_wscale,
                pre_quantized_weight=_get_pre_quant_weight(),
                activation_tensor_id=_act_tensor_id,
            )

    else:

        def quant_forward(input_tensor, *args, **kwargs):
            from atomrl.lumen.ops.quantize.linear import quantized_linear

            skip_bias_add = getattr(module, "skip_bias_add", False)
            bias = getattr(module, "bias", None)
            bias_for_gemm = None if skip_bias_add else bias

            is_row_parallel = getattr(module, "input_is_parallel", False)
            seq_parallel = getattr(module, "sequence_parallel", False)
            tp_group = getattr(module, "tp_group", None)

            # Cache frozen base weight FP8 quant on first call (blockwise2d avoids
            # 896 MiB re-allocation on every forward + gradient-checkpoint recompute).
            _maybe_cache_frozen_weight(module, scaling_type, fp8_dtype, block_size)
            _wcache = getattr(module, "_fp8_weight_data", None)
            _wscale = getattr(module, "_fp8_weight_scale", None)

            if seq_parallel and not is_row_parallel:
                from megatron.core.tensor_parallel.mappings import (
                    gather_from_sequence_parallel_region,
                )

                input_tensor = gather_from_sequence_parallel_region(
                    input_tensor,
                    tensor_parallel_output_grad=True,
                    group=tp_group,
                )

            # After the gather, so the row count handed to the weight cache is the
            # one the GEMMs will see; it decides which backend layout to build for.
            _wcache, _wscale = _mxfp4_cached_weight(
                module,
                module.weight,
                _wcache,
                _wscale,
                scaling_type,
                fp8_dtype,
                block_size,
                gemm_rows=input_tensor.numel() // input_tensor.shape[-1],
            )

            result = quantized_linear(
                input_tensor,
                module.weight,
                bias_for_gemm,
                scaling_manager=manager,
                backend=backend,
                scaling_type=scaling_type,
                fp8_dtype=fp8_dtype,
                block_size=block_size,
                tensor_id=tensor_id,
                quantize_activation=quantize_activation,
                fp8_wgrad=fp8_wgrad,
                gradient_accumulation_fusion=_gaf,
                delay_wgrad=_delay_wgrad,
                deferred_wgrad=_deferred_wgrad,
                fp8_activation_store=_fp8_act_store,
                fp8_weight_cache=_wcache,
                fp8_weight_scale=_wscale,
                pre_quantized_weight=_get_pre_quant_weight(),
                activation_tensor_id=_act_tensor_id,
            )

            if is_row_parallel and seq_parallel:
                from megatron.core.tensor_parallel.mappings import (
                    reduce_scatter_to_sequence_parallel_region,
                )

                result = reduce_scatter_to_sequence_parallel_region(
                    result,
                    group=tp_group,
                )
            elif is_row_parallel:
                try:
                    from megatron.core.tensor_parallel.mappings import (
                        reduce_from_tensor_model_parallel_region,
                    )

                    result = reduce_from_tensor_model_parallel_region(result)
                except ImportError:
                    pass
            elif getattr(module, "gather_output", False):
                try:
                    from megatron.core.tensor_parallel.mappings import (
                        gather_from_tensor_model_parallel_region,
                    )

                    result = gather_from_tensor_model_parallel_region(result)
                except ImportError:
                    pass

            output_bias = bias if skip_bias_add else None
            return result, output_bias

    module._original_forward = original_forward
    module.forward = quant_forward


def store_weights_fp8(
    model: nn.Module,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> int:
    """Pre-quantize patched ``nn.Linear`` weights to FP8 and cache them.

    Must be called **after** :func:`enable`.  Each patched layer's BF16
    weight is quantized to FP8 and stored as a non-parameter buffer
    ``module._fp8_weight_data`` alongside its scale ``module._fp8_weight_scale``.
    The original BF16 ``nn.Parameter`` is kept for DDP gradient sync and
    the optimizer.

    The ``quant_forward`` path detects ``_fp8_weight_data`` and feeds
    the cached FP8 weight directly to the GEMM, **skipping the per-forward
    re-quantization** that ``quant.enable`` normally performs.

    After each ``optimizer.step()`` the BF16 master weight has been
    updated, so :func:`register_fp8_weight_optimizer_hooks` re-quantizes
    the master into the FP8 cache to keep it in sync.

    Returns:
        Number of linear layers with cached FP8 weights.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    count = 0
    for _name, module in model.named_modules():
        if not getattr(module, "_quant_enabled", False):
            continue
        # This cache holds per-tensor FP8 with a scalar scale, and the forward
        # hands whatever is in it straight to the GEMM: _mxfp4_cached_weight
        # passes a populated cache through untouched, and quantize_input is
        # skipped. An MXFP4 GEMM then receives e4m3 bytes where it expects
        # packed FP4 and a scalar where it expects an E8M0 tile grid, and dies
        # unpacking an empty shape several frames from the cause. Refuse here,
        # where the feature and the format are both still named.
        if getattr(module, "_quant_scaling_type", None) == "mxfp4":
            raise ValueError(
                f"{_name or type(module).__name__}: the FP8 weight cache "
                "(LUMEN_FP8_WEIGHT_CACHE=1, or fp8_weight_cache on LumenConfig) "
                "does not apply to MXFP4. It stores per-tensor FP8 with a scalar "
                "scale, which no MXFP4 GEMM can read. Nothing is lost by turning "
                "it off: MXFP4 caches its own FP4 weight per optimizer step "
                "unconditionally (atomrl.lumen.quantize._mxfp4_cached_weight, opt out "
                "with LUMEN_MXFP4_DISABLE_WEIGHT_CACHE=1). It could not be filled "
                "from here anyway -- the operand layout depends on the backend "
                "measured for the consuming GEMM shape, which is not known until "
                "that shape's first call."
            )
        weight = getattr(module, "weight", None)
        if weight is None or not isinstance(weight, nn.Parameter):
            continue
        if hasattr(module, "_fp8_weight_data"):
            continue

        with torch.no_grad():
            amax = weight.data.abs().amax().clamp(min=1e-12)
            scale = (amax / fp8_max).float()
            fp8_data = (
                (weight.data.float() * (1.0 / scale))
                .clamp(-fp8_max, fp8_max)
                .to(fp8_dtype)
            )

        module.register_buffer("_fp8_weight_data", fp8_data, persistent=False)
        module.register_buffer("_fp8_weight_scale", scale, persistent=False)
        module._fp8_weight_dtype = fp8_dtype
        count += 1

    logger.info("store_weights_fp8: cached %d linear weights in %s", count, fp8_dtype)
    return count


def store_weights_fp8_blockwise2d(
    model: nn.Module,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    block_size: int = 128,
) -> int:
    """Store **frozen** blockwise2d base weights as FP8 in-place (param-storage).

    For each patched, frozen ``nn.Linear`` whose dims are block-aligned, quantize
    the weight ONCE to FP8 with a 2D (N/block, K/block) scale, replace
    ``weight.data`` with the FP8 tensor (BF16 freed → ~half the param bytes), and
    stash the FP8 data + 2D scale as ``_fp8_weight_data`` / ``_fp8_weight_scale``
    buffers. The ``quant_forward`` path then feeds these straight to
    ``QuantizedLinearFunction`` (correct blockwise2d fwd + backward, no per-step
    weight re-quantization), and (under FSDP) the param all-gathers as FP8.

    A state_dict pre/post hook dequantizes FP8→BF16 so checkpoints stay BF16.
    Only blockwise2d; per-tensor uses :func:`store_weights_fp8`.

    Returns: number of weights stored in FP8.
    """
    from atomrl.lumen.ops.quantize.linear import _quant_blockwise2d_weight

    count = 0
    for _name, module in model.named_modules():
        if not getattr(module, "_quant_enabled", False):
            continue
        w = getattr(module, "weight", None)
        if w is None or not isinstance(w, nn.Parameter) or w.dim() != 2:
            continue
        if w.requires_grad:  # only frozen base weights (LoRA recipe)
            continue
        if getattr(module, "_fp8_weight_data", None) is not None:
            continue
        if w.shape[0] % block_size or w.shape[1] % block_size:
            continue
        with torch.no_grad():
            fp8, scale2d = _quant_blockwise2d_weight(
                w.data.contiguous(), fp8_dtype, block_size
            )
            # Route through the (tested) fp8_weight_cache → QuantizedLinearFunction
            # path (correct blockwise2d backward + skip-wgrad/DGrad-transpose), NOT
            # FP8StoredLinearFunction (whose blockwise2d bwd falls back to per-tensor).
            module.register_buffer("_fp8_weight_data", fp8, persistent=False)
            module.register_buffer("_fp8_weight_scale", scale2d, persistent=False)
            module._fp8_weight_dtype = fp8_dtype
            module._lumen_frozen = True  # so WGrad is skipped (grad discarded)
            w._fp8_original_dtype = w.dtype
            w.data = fp8  # free the BF16 master (param.data → FP8)
            w._fp8_storage_scale = scale2d
        count += 1

    # state_dict: emit dequantized BF16 weight, keep FP8 at runtime.
    def _pre_save(mod, prefix, keep_vars):
        from atomrl.lumen.ops.quantize.gemm_primitives import _dequant_fp8_weight

        for p in mod._parameters.values():
            if p is None or not hasattr(p, "_fp8_storage_scale"):
                continue
            p._sd_backup = p.data
            p.data = _dequant_fp8_weight(p.data, p._fp8_storage_scale, block_size).to(
                getattr(p, "_fp8_original_dtype", torch.bfloat16)
            )

    def _post_save(mod, state_dict, prefix, local_metadata):
        for p in mod._parameters.values():
            if p is not None and hasattr(p, "_sd_backup"):
                p.data = p._sd_backup
                del p._sd_backup

    for mod in model.modules():
        if any(
            p is not None and hasattr(p, "_fp8_storage_scale")
            for p in mod._parameters.values()
        ):
            mod.register_state_dict_pre_hook(_pre_save)
            mod.register_state_dict_post_hook(_post_save)

    logger.info(
        "store_weights_fp8_blockwise2d: stored %d frozen weights in %s",
        count,
        fp8_dtype,
    )
    return count


def register_fp8_weight_optimizer_hooks(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Register a post-step hook on *optimizer* to refresh FP8 weight caches.

    After ``optimizer.step()`` updates the BF16 master weights, this hook
    re-quantizes each patched layer's weight to FP8 and writes it into the
    ``_fp8_weight_data`` buffer so the next forward uses fresh cached data.

    **Must be called after at least one forward pass** (or after explicitly
    calling :func:`store_weights_fp8` / :func:`store_weights_fp8_blockwise2d`)
    so that ``_fp8_weight_data`` is already present on each module. The hook
    snapshots which modules have the cache at registration time; layers whose
    cache is built lazily on the first forward will not be tracked if this
    function is called before any forward has run.
    """
    modules_with_cache = [
        m
        for m in model.modules()
        if hasattr(m, "_fp8_weight_data") and hasattr(m, "_fp8_weight_scale")
    ]

    if not modules_with_cache:
        logger.warning(
            "register_fp8_weight_optimizer_hooks: no FP8 cached modules found — "
            "call this function after at least one forward pass or after store_weights_fp8()"
        )
        return

    logger.info(
        "register_fp8_weight_optimizer_hooks: managing %d FP8 cached layers",
        len(modules_with_cache),
    )

    def _post_step(opt, args, kwargs):
        with torch.no_grad():
            for m in modules_with_cache:
                w = m.weight
                fp8_dt = getattr(m, "_fp8_weight_dtype", torch.float8_e4m3fn)
                fp8_max_val = torch.finfo(fp8_dt).max
                amax = w.data.abs().amax().clamp(min=1e-12)
                scale = (amax / fp8_max_val).float()
                fp8_data = (
                    (w.data.float() * (1.0 / scale))
                    .clamp(-fp8_max_val, fp8_max_val)
                    .to(fp8_dt)
                )
                m._fp8_weight_data.copy_(fp8_data)
                m._fp8_weight_scale.copy_(scale)

    optimizer.register_step_post_hook(_post_step)


def register_mxfp4_weight_optimizer_hooks(
    model,
    optimizer,
) -> None:
    """Register a post-step hook to invalidate MXFP4 weight caches.

    MXFP4 weight quantization (RTN, deterministic) is cached on each patched
    module, or on the weight Parameter for native Lumen linears, across
    micro-batches within a gradient accumulation step. After ``optimizer.step()``
    updates BF16 master weights, this hook clears both cache locations so the
    next forward re-quantizes from the updated weights.

    Without this the cached FP4 weight is never invalidated, so forward and
    DGrad keep using the step-0 weights for the whole run — the loss flattens
    and nothing raises.

    Args:
        model: A module, or the list of model chunks Megatron builds under
            virtual pipeline parallelism.
        optimizer: Any optimizer. Megatron's wrappers are not
            ``torch.optim.Optimizer`` subclasses and are handled too.
    """
    chunks = list(model) if isinstance(model, (list, tuple)) else [model]

    def _invalidate():
        for chunk in chunks:
            for m in chunk.modules():
                if hasattr(m, "_mxfp4_w_cache"):
                    del m._mxfp4_w_cache
                    if hasattr(m, "_mxfp4_w_cache_version"):
                        del m._mxfp4_w_cache_version
                # Native Lumen parallel linears cache on their Parameter because
                # they call the shared _do_gemm helper rather than the patched
                # module forward above. Every parameter the module owns has to be
                # swept, not just ``.weight``: a grouped MoE layer holds its
                # experts as weight0..weightN and hands them to the linear's
                # forward one at a time, so the cache lands on a Parameter that
                # is not reachable under that name. Those entries were never
                # cleared, leaving the experts quantized from the step-0 master
                # weights for the whole run while the dense layers updated -- and
                # nothing raises, the loss just stops moving.
                for param in m._parameters.values():
                    if param is not None and hasattr(param, "_mxfp4_w_cache"):
                        del param._mxfp4_w_cache
                        if hasattr(param, "_mxfp4_w_cache_version"):
                            del param._mxfp4_w_cache_version

    # Megatron's ChainedOptimizer / DistributedOptimizer are not
    # torch.optim.Optimizer subclasses and lack register_step_post_hook, so
    # wrap step() in that case (same approach as
    # ScalingManager.register_fp8_optimizer_hook).
    if hasattr(optimizer, "register_step_post_hook"):
        optimizer.register_step_post_hook(lambda _opt, _a, _k: _invalidate())
        logger.info("register_mxfp4_weight_optimizer_hooks: registered post-step hook")
    else:
        _orig_step = optimizer.step

        def _wrapped_step(*args, **kwargs):
            result = _orig_step(*args, **kwargs)
            _invalidate()
            return result

        optimizer.step = _wrapped_step
        logger.info(
            "register_mxfp4_weight_optimizer_hooks: wrapped optimizer.step() "
            "(no register_step_post_hook)"
        )


def disable(model: nn.Module) -> None:
    """Remove FP8 quantized forward from all patched layers."""
    for module in model.modules():
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward
            module._quant_enabled = False
