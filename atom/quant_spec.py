# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Typed quantization specification and parser registry.

This module introduces:
- :class:`LayerQuantConfig` — a frozen dataclass for type-safe, immutable
  layer quant descriptions.
- :class:`ParsedQuantConfig` — structured output of parsing ``quantization_config``
  from a HuggingFace ``PretrainedConfig``.
- A parser registry (:func:`register_quant_parser`, :func:`get_quant_parser`) so
  new quantizer back-ends (Quark, compressed-tensors, …) can each provide their
  own parsing logic without bloating ``config.py``.
"""

from __future__ import annotations

import functools
import importlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch


class _LazyAiterAttr:
    """One AITER attribute, resolved on first use rather than at import.

    `atom.config` imports this module, and Python imports a package before its
    submodule, so a plain `from aiter import QuantType` here makes *reading a
    dataclass* require the AITER build. That is why the unit-test suite used to
    replace `atom.config` with a hand-written stand-in, which then drifted from
    the real thing and silently disabled test modules.

    Only the name is deferred. Every attribute read returns the genuine AITER
    object, so `quant_type` values compare and behave exactly as before -- this
    is not a stand-in and never answers when AITER is missing.
    """

    __slots__ = ("_attr", "_module", "_value")

    def __init__(self, module: str, attr: str):
        self._module = module
        self._attr = attr
        self._value = None

    def _resolve(self) -> Any:
        if self._value is None:
            self._value = getattr(importlib.import_module(self._module), self._attr)
        return self._value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)


QuantType = _LazyAiterAttr("aiter", "QuantType")
d_dtypes = _LazyAiterAttr("aiter.utility.dtypes", "d_dtypes")

# Logical quant dtype used for dispatch. NVFP4 is a composite format rather
# than a torch.dtype: its checkpoint stores packed uint8 weights, FP8-E4M3
# group scales, and FP32 global scales.
#
# It must stay unequal to every torch.dtype. NVFP4 shares QuantType.per_1x32
# with MXFP4.
NVFP4_DTYPE = "nvfp4"
NVFP4_GROUP_SIZE = 16

# ──────────────────────────────────────────────────────────────────────
# Typed layer-level spec
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayerQuantConfig:
    """Immutable description of how a single layer (or default) is quantized."""

    # `default_factory`, not `default`: a plain default is evaluated when the
    # class is created, which would resolve the lazy AITER name at import time
    # and undo the deferral above. The factory runs per instantiation instead.
    quant_type: QuantType = field(default_factory=lambda: QuantType.No)
    # Usually a torch.dtype; composite formats use a logical dispatch marker
    # (see NVFP4_DTYPE before comparing this field against a torch.dtype).
    quant_dtype: Any = torch.bfloat16
    is_dynamic: bool = True
    quant_method: str | None = None
    # Source weight blocks, distinct from the activation grouping in QuantType.
    weight_block_size: tuple[int, int] | None = None
    # An explicit activation contract, e.g. native W4A8 rather than W4A4.
    activation_dtype: Any = None

    @property
    def is_quantized(self) -> bool:
        return self.quant_type != QuantType.No

    @classmethod
    def no_quant(cls, dtype: Any = torch.bfloat16) -> LayerQuantConfig:
        """Convenience: unquantized spec with a given storage dtype."""
        return cls(quant_type=QuantType.No, quant_dtype=dtype)


def should_skip_online_quant(cur_type, cur_dtype, online_cfg) -> bool:
    """Skip online re-quant when the layer is excluded (No) or already in target.

    Shared by ``LinearBase.online_quantize_weight``, ``FusedMoE._online_quant``
    and ``RMSNorm.online_quantize_activation``: re-quantizing is a no-op (and may
    corrupt already-quantized weights) when the online target is ``No`` or the
    layer already matches the target ``(quant_type, quant_dtype)``.
    """
    return online_cfg.quant_type == QuantType.No or (
        cur_type == online_cfg.quant_type and cur_dtype == online_cfg.quant_dtype
    )


def will_online_requant(
    quant_config,
    prefix: str,
    source_quant_type,
    source_quant_dtype,
) -> bool:
    """Whether *this* layer's weight gets replaced by online re-quantization.

    ``quant_config.online_quant`` is a whole-model flag: it says the run was
    launched with ``--online_quant_config``, not that any given layer is
    affected. A layer named in the online ``exclude_layer`` list, or one whose
    source already is the online target, keeps its checkpoint weight. Anything
    that decides a layer's *format* -- which parameters to allocate, which GEMM
    to dispatch -- has to ask per layer, because the two answers differ for
    every model whose online config excludes part of the graph.
    """
    if quant_config is None or not getattr(quant_config, "online_quant", False):
        return False
    online_cfg = quant_config.get_layer_quant_config(prefix, use_online_quant=True)
    return not should_skip_online_quant(
        source_quant_type, source_quant_dtype, online_cfg
    )


def validate_nvfp4_online_target(quant_config, prefix: str) -> None:
    """Reject an NVFP4 source layer that online quantization will not convert.

    NVFP4 is load-only and MXFP4 is its only runtime format, so every NVFP4
    layer needs a per_1x32/fp4x2 online target. Layers call this when they are
    built, so a recipe that leaves one out fails before any weight is loaded.
    """
    if getattr(quant_config, "online_quant", False):
        target = quant_config.get_layer_quant_config(prefix, use_online_quant=True)
        if target.quant_type.value == QuantType.per_1x32.value and (
            target.quant_dtype == d_dtypes.get("fp4x2")
        ):
            return
        if target.quant_type.value == QuantType.No.value:
            reason = "this layer is excluded from online quantization or has no target"
        else:
            reason = (
                f"this layer's online target is "
                f"{target.quant_type.name}/{target.quant_dtype}"
            )
    else:
        reason = "no online quantization config was given"
    raise ValueError(
        f"{prefix}: NVFP4 checkpoint weights cannot run directly and must be "
        f"converted to MXFP4 online, but {reason}. Give this layer an mxfp4 "
        "target in --online_quant_config."
    )


def validate_nvfp4_global_scales(scale_2, prefix: str, what: str) -> None:
    """Reject an NVFP4 global scale that was never loaded from the checkpoint.

    ``weight_scale_2`` is allocated zeroed up front and only filled if the
    checkpoint actually carries a ``*_weight_scale_2`` tensor for the layer. If
    it does not, dequantization reads the placeholder and silently produces an
    all-zero weight that degrades accuracy with no error anywhere -- which is
    why both allocation sites zero-fill. Every value has to be finite
    and strictly positive: NVFP4 global scales are amax ratios, so zero, a
    negative, a NaN or an inf all mean "not loaded", not "unusual checkpoint".
    """
    if scale_2 is None:
        raise RuntimeError(
            f"{prefix}: NVFP4 {what} is missing. NVFP4 is a two-level format "
            "and cannot be dequantized without its global scale."
        )
    if not torch.all(torch.isfinite(scale_2) & (scale_2 > 0)):
        raise RuntimeError(
            f"{prefix}: NVFP4 {what} holds non-positive or non-finite values "
            f"({scale_2.flatten().tolist()[:8]}...), which means the checkpoint "
            "did not provide it. An NVFP4 checkpoint must carry a "
            "*_weight_scale_2 tensor for every quantized layer."
        )


def should_stream_online_quant(
    quant_config,
    prefix: str,
    source_quant_type,
    source_quant_dtype,
) -> bool:
    """Return whether this source can be streamed to a distinct online target."""
    # Attention stays in the post-load pass because MLA post-processing reads
    # an already-loaded and processed kv_b_proj.
    # Imported lazily to avoid a module-load cycle (envs/quark pull in config
    # which can pull in quant_spec).
    from atom.quantization.quark.utils import can_dequant_weight_online
    from atom.utils import envs

    if not envs.ATOM_ONLINE_QUANT_STREAMING:
        return False
    # can_dequant_weight_online recognizes only torch dtypes; NVFP4 is decoded
    # by its own source decoder, `dequantize_nvfp4`.
    if source_quant_dtype != NVFP4_DTYPE and not can_dequant_weight_online(
        source_quant_type, source_quant_dtype
    ):
        return False
    return will_online_requant(
        quant_config, prefix, source_quant_type, source_quant_dtype
    )


# ──────────────────────────────────────────────────────────────────────
# Structured parsed config
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ParsedQuantConfig:
    """Result of parsing a ``quantization_config`` dict."""

    global_spec: LayerQuantConfig = field(default_factory=LayerQuantConfig)
    # Pattern specs as list of (pattern, spec) tuples to preserve order
    layer_pattern_specs: list[tuple[str, LayerQuantConfig]] = field(
        default_factory=list
    )
    exclude_layers: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Parser registry
# ──────────────────────────────────────────────────────────────────────

_PARSER_REGISTRY: dict[str, type[QuantConfigParser]] = {}


class QuantConfigParser(ABC):
    """Base class for quantization config parsers."""

    @abstractmethod
    def parse(self, hf_quant_config: dict) -> ParsedQuantConfig:
        """Parse a ``quantization_config`` dict into :class:`ParsedQuantConfig`."""
        ...


def register_quant_parser(name: str):
    """Decorator: register a parser class under *name*."""

    def wrapper(cls: type[QuantConfigParser]):
        _PARSER_REGISTRY[name] = cls
        return cls

    return wrapper


def get_quant_parser(method_name: str) -> QuantConfigParser:
    """Return an instance of the parser for *method_name*.

    Falls back to the ``_generic`` parser if no specific one is registered.
    """
    cls = _PARSER_REGISTRY.get(method_name) or _PARSER_REGISTRY.get("_generic")
    if cls is None:
        raise ValueError(
            f"No quant config parser registered for {method_name!r} "
            f"and no _generic fallback available."
        )
    return cls()


# ──────────────────────────────────────────────────────────────────────
# Built-in parsers
# ──────────────────────────────────────────────────────────────────────


# -- helpers ----------------------------------------------------------------


@functools.cache
def _qscheme_to_quant_type() -> dict[str, QuantType]:
    """qscheme string -> AITER quant type, built on first use.

    A module-level dict literal would read the lazy AITER names at import time,
    which is exactly what the deferral above exists to avoid. Cached, so the
    lookup stays a dict access after the first call.
    """
    return {
        "per_channel": QuantType.per_Token,
        "per_tensor": QuantType.per_Tensor,
        "per_group": QuantType.per_1x32,
        "per_block": QuantType.per_1x128,
    }


def _parse_quant_type(qscheme: str | None) -> QuantType:
    if qscheme is None:
        return QuantType.No
    return _qscheme_to_quant_type().get(qscheme, QuantType.No)


def _parse_quant_dtype(dtype_str: str | None) -> Any:
    if dtype_str is None:
        return torch.bfloat16
    # Normalise e.g. "fp8_e4m3" -> "fp8", "fp4_e2m1" -> "fp4"
    key = re.sub(r"_e\d+m\d+.*", "", dtype_str)
    # Direct lookup
    result = d_dtypes.get(key)
    if result is not None:
        return result
    # Try common suffixed variants: fp4 -> fp4x2, int4 -> int4x2, etc.
    for suffix in ("x2", "x4"):
        result = d_dtypes.get(key + suffix)
        if result is not None:
            return result
    return torch.bfloat16


def _parse_is_dynamic(input_tensors: dict | list | None) -> bool:
    if input_tensors is None:
        return True
    if isinstance(input_tensors, list):
        # Sequential scale quantization (NVFP4) has a dynamic FP4 first stage
        # followed by a static FP8 scale-quant stage. Activation dynamism is
        # defined by the first stage.
        if not input_tensors:
            return True
        input_tensors = input_tensors[0]
    return input_tensors.get("is_dynamic", True)


def _quark_weight_spec(
    weight: dict | list,
    input_tensors: dict | list | None,
) -> tuple[dict, Any | None]:
    """Return the primary weight stage and an optional logical dtype override.

    ATOM historically accepted only one Quark weight-spec dictionary. Quark's
    NVFP4 export uses two sequential stages for both weights and activations:

    1. packed FP4 E2M1, per-group with group_size=16;
    2. FP8 E4M3 per-tensor quantization of the first-stage scales.

    Keep the AITER quant type at per_1x32 for loader compatibility (AITER has
    no per_1x16 enum), but use ``NVFP4_DTYPE`` as the logical ``quant_dtype``
    so allocation and execution cannot confuse this format with MXFP4.
    """
    weight_is_multi_stage = isinstance(weight, list) and len(weight) > 1
    input_is_multi_stage = isinstance(input_tensors, list) and len(input_tensors) > 1
    if not weight_is_multi_stage and not input_is_multi_stage:
        if isinstance(weight, dict):
            return weight, None
        if (
            isinstance(weight, list)
            and len(weight) == 1
            and isinstance(weight[0], dict)
        ):
            return weight[0], None
        raise ValueError(
            "A single-stage Quark weight config must be a dictionary or a "
            "one-entry list containing a dictionary."
        )

    if (
        not isinstance(weight, list)
        or len(weight) != 2
        or not all(isinstance(stage, dict) for stage in weight)
        or not isinstance(input_tensors, list)
        or len(input_tensors) != 2
        or not all(isinstance(stage, dict) for stage in input_tensors)
    ):
        raise ValueError(
            "Unsupported Quark multi-stage quantization config. ATOM recognizes "
            "only NVFP4 with two dictionary stages in both `weight` and "
            "`input_tensors`."
        )

    weight_fp4, weight_scale_quant = weight
    input_fp4, input_scale_quant = input_tensors
    is_nvfp4 = (
        str(weight_fp4.get("dtype", "")).lower().startswith("fp4")
        and weight_fp4.get("qscheme") == "per_group"
        and weight_fp4.get("group_size") == NVFP4_GROUP_SIZE
        and not weight_fp4.get("is_dynamic")
        and str(input_fp4.get("dtype", "")).lower().startswith("fp4")
        and input_fp4.get("qscheme") == "per_group"
        and input_fp4.get("group_size") == NVFP4_GROUP_SIZE
        and input_fp4.get("is_dynamic") is True
        and str(weight_scale_quant.get("dtype", "")).lower().startswith("fp8_e4m3")
        and weight_scale_quant.get("qscheme") == "per_tensor"
        and not weight_scale_quant.get("is_dynamic")
        and str(input_scale_quant.get("dtype", "")).lower().startswith("fp8_e4m3")
        and input_scale_quant.get("qscheme") == "per_tensor"
        and not input_scale_quant.get("is_dynamic")
    )
    if not is_nvfp4:
        raise ValueError(
            "Unsupported Quark multi-stage quantization config. ATOM recognizes "
            "only NVFP4: static FP4 weights and dynamic FP4 activations with "
            "per-group(group_size=16), each followed by static FP8-E4M3 "
            "per-tensor scale quantization."
        )
    return weight_fp4, NVFP4_DTYPE


def _build_quark_layer_spec(layer_dict: dict) -> LayerQuantConfig:
    """Build a :class:`LayerQuantConfig` from a single Quark per-layer dict."""
    weight = layer_dict.get("weight", {}) or {}
    input_tensors = layer_dict.get("input_tensors")
    weight, quant_dtype_override = _quark_weight_spec(weight, input_tensors)
    return LayerQuantConfig(
        quant_type=_parse_quant_type(weight.get("qscheme")),
        quant_dtype=(
            quant_dtype_override
            if quant_dtype_override is not None
            else _parse_quant_dtype(weight.get("dtype"))
        ),
        is_dynamic=_parse_is_dynamic(input_tensors),
        quant_method="quark",
    )


# -- Quark ------------------------------------------------------------------


@register_quant_parser("quark")
class QuarkParser(QuantConfigParser):
    """Parser for Quark-style ``quantization_config``."""

    def parse(self, hf_quant_config: dict) -> ParsedQuantConfig:
        global_dict = hf_quant_config.get("global_quant_config") or {}
        layer_dict = hf_quant_config.get("layer_quant_config") or {}
        exclude = list(hf_quant_config.get("exclude") or [])

        global_spec = (
            _build_quark_layer_spec(global_dict) if global_dict else LayerQuantConfig()
        )

        pattern_specs: list[tuple[str, LayerQuantConfig]] = []
        for pattern, cfg in layer_dict.items():
            pattern_specs.append((pattern, _build_quark_layer_spec(cfg)))

        return ParsedQuantConfig(
            global_spec=global_spec,
            layer_pattern_specs=pattern_specs,
            exclude_layers=exclude,
        )


# -- NVIDIA ModelOpt --------------------------------------------------------


def _build_modelopt_layer_spec(
    quant_algo: str, layer_info: dict[str, Any]
) -> LayerQuantConfig:
    quant_algo = quant_algo.upper()
    if quant_algo == "NVFP4":
        group_size = layer_info.get("group_size", NVFP4_GROUP_SIZE)
        if group_size != NVFP4_GROUP_SIZE:
            raise ValueError(
                "ATOM supports ModelOpt NVFP4 only with "
                f"group_size={NVFP4_GROUP_SIZE}, got {group_size}."
            )
        return LayerQuantConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=NVFP4_DTYPE,
            is_dynamic=True,
            quant_method="modelopt",
        )
    if quant_algo == "MXFP8":
        return LayerQuantConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=d_dtypes.get("fp8"),
            is_dynamic=True,
            quant_method="modelopt",
        )
    raise ValueError(
        f"Unsupported ModelOpt mixed-precision quant_algo {quant_algo!r}; "
        "ATOM currently supports NVFP4 and MXFP8 entries."
    )


@register_quant_parser("modelopt")
class ModelOptParser(QuantConfigParser):
    """Parse NVIDIA ModelOpt mixed MXFP8/NVFP4 checkpoints."""

    _EXPERT_LAYER_RE = re.compile(
        r"^(?P<parent>.+\.experts)\.\d+\." r"(?:w1|w2|w3|gate_proj|down_proj|up_proj)$"
    )

    def parse(self, hf_quant_config: dict) -> ParsedQuantConfig:
        config = hf_quant_config.get("quantization", hf_quant_config)
        quant_algo = str(config.get("quant_algo", "")).upper()
        if quant_algo != "MIXED_PRECISION":
            # MIXED_PRECISION is the only ModelOpt shape ATOM has a checkpoint
            # to test against. The generic heuristics cannot stand in for the
            # others: they substring-match the "fp4" inside "nvfp4" and report
            # group-32 MXFP4, and they read neither `quantized_layers` nor
            # `exclude_modules`, so the fallback answers confidently and wrong.
            raise ValueError(
                f"Unsupported ModelOpt quant_algo {quant_algo!r}. ATOM reads "
                "ModelOpt checkpoints only in the MIXED_PRECISION form, whose "
                "`quantized_layers` entries are NVFP4 or MXFP8."
            )

        quantized_layers = config.get("quantized_layers")
        if not isinstance(quantized_layers, dict) or not quantized_layers:
            raise ValueError(
                "ModelOpt MIXED_PRECISION requires a non-empty "
                "`quantized_layers` mapping."
            )

        pattern_specs: list[tuple[str, LayerQuantConfig]] = []
        expert_specs: dict[str, LayerQuantConfig] = {}
        for layer_name, layer_info in quantized_layers.items():
            if not isinstance(layer_info, dict):
                raise TypeError(
                    f"ModelOpt layer {layer_name!r} must map to a dictionary."
                )
            layer_algo = str(layer_info.get("quant_algo", "")).upper()
            spec = _build_modelopt_layer_spec(layer_algo, layer_info)
            expert_match = self._EXPERT_LAYER_RE.fullmatch(layer_name)
            if expert_match is None:
                pattern_specs.append((layer_name, spec))
                continue

            parent = expert_match.group("parent")
            previous = expert_specs.setdefault(parent, spec)
            if previous != spec:
                raise ValueError(
                    f"Mixed ModelOpt quantization inside fused experts {parent!r}."
                )

        # FusedMoE asks for one config at the experts container, whereas
        # ModelOpt records one entry per expert and projection.
        pattern_specs.extend(expert_specs.items())
        exclude = list(config.get("exclude_modules") or [])
        return ParsedQuantConfig(
            global_spec=LayerQuantConfig(),
            layer_pattern_specs=pattern_specs,
            exclude_layers=exclude,
        )


# -- Online quantization ----------------------------------------------------


@register_quant_parser("online_quant")
class QuarkOnlineParser(QuantConfigParser):
    """Parser for Quark-style online ``quantization_config``."""

    def parse(self, online_quant_config: dict) -> ParsedQuantConfig:
        """Parse the user-facing online quantization dict and populate
        ``online_global_qconfig_dict``, ``online_layer_qconfig_dict``,
        and ``online_exclude_layers_list``.

        Supported format strings:
        - ``"ptpc_fp8"``        — per-tensor-per-channel FP8
        - ``"per_block_fp8"``   — per-block FP8 (128x128; alias of per_block128_fp8)
        - ``"per_block128_fp8"``— per-block FP8, 128x128 block
        - ``"mxfp4"``           — microscaling FP4 (block size 32)
        - ``"mxfp8"``           — microscaling FP8 (block size 32)
        """
        if not isinstance(online_quant_config, dict):
            raise TypeError("online_quant_config must be a dict parsed from JSON.")

        # Explicit table of supported online quant formats -> (QuantType, dtype_str).
        # per_block / per_block128 are aliases (128 is the default block size).
        FORMAT_MAP = {
            "ptpc_fp8": (QuantType.per_Token, "fp8"),
            "per_block_fp8": (QuantType.per_1x128, "fp8"),
            "per_block128_fp8": (QuantType.per_1x128, "fp8"),
            "mxfp4": (QuantType.per_1x32, "fp4"),
            "mxfp8": (QuantType.per_1x32, "fp8"),
        }

        def _parse_online_quant_format(quant_format_str: str) -> LayerQuantConfig:
            quant_format_str = quant_format_str.strip().lower()
            if quant_format_str not in FORMAT_MAP:
                raise ValueError(
                    f"Unsupported online quant format: '{quant_format_str}'. "
                    f"Expected one of: {sorted(FORMAT_MAP)}."
                )
            quant_type, dtype_str = FORMAT_MAP[quant_format_str]

            dtype_str = dtype_str.split("_")[0]
            if dtype_str.endswith("4"):
                dtype_str += "x2"
            quant_dtype = d_dtypes.get(dtype_str)
            if quant_dtype is None:
                raise ValueError(
                    f"Unsupported online quant dtype: '{dtype_str}' "
                    f"(from '{quant_format_str}')"
                )

            return LayerQuantConfig(
                quant_type=quant_type,
                quant_dtype=quant_dtype,
                is_dynamic=True,
                quant_method="quark",
            )

        global_quant_str = online_quant_config.get("global_quant_config", "")
        if global_quant_str:
            online_global_qconfig_dict = _parse_online_quant_format(global_quant_str)
        else:
            online_global_qconfig_dict = LayerQuantConfig()

        layer_quant_dict = online_quant_config.get("layer_quant_config", {})
        layer_pattern_specs: list[tuple[str, LayerQuantConfig]] = []
        if isinstance(layer_quant_dict, dict):
            for layer_pattern, quant_str in layer_quant_dict.items():
                layer_pattern_specs.append(
                    (layer_pattern, _parse_online_quant_format(quant_str))
                )

        exclude_layers = online_quant_config.get("exclude_layer", [])
        if isinstance(exclude_layers, str):
            online_exclude_layers_list = [exclude_layers] if exclude_layers else []
        elif isinstance(exclude_layers, list):
            online_exclude_layers_list = exclude_layers
        else:
            online_exclude_layers_list = []
        return ParsedQuantConfig(
            global_spec=online_global_qconfig_dict,
            layer_pattern_specs=layer_pattern_specs,
            exclude_layers=online_exclude_layers_list,
        )


# -- Generic (compressed-tensors, GPTQ, AWQ, …) ----------------------------


@register_quant_parser("_generic")
class GenericParser(QuantConfigParser):
    """Fallback parser that uses heuristics for compressed-tensors, etc."""

    # Regex patterns for identifying quantization types from config keys/values
    _DTYPE_PATTERNS: ClassVar[dict[str, str]] = {
        r"fp8|float8": "fp8",
        r"fp4|float4|mxfp4": "fp4x2",
        r"int8|w8a8": "int8",
        r"int4|w4a16|gptq|awq": "int4x2",
    }

    @staticmethod
    @functools.cache
    def _qtype_patterns() -> dict[str, QuantType]:
        """Regex -> AITER quant type, built on first use.

        A class-level dict literal runs when the class is created, i.e. at
        import time, which would resolve the lazy AITER names and defeat the
        deferral. Same reason as `_qscheme_to_quant_type`.
        """
        return {
            r"block|per_block|blockwise|1x128": QuantType.per_1x128,
            r"per_channel|channel|per_token|token": QuantType.per_Token,
            r"per_tensor|tensor": QuantType.per_Tensor,
            r"per_group|group": QuantType.per_1x32,
        }

    def parse(self, hf_quant_config: dict) -> ParsedQuantConfig:
        quant_method = hf_quant_config.get("quant_method", "")
        config_str = str(hf_quant_config).lower()

        quant_dtype = self._infer_dtype(hf_quant_config, config_str)
        quant_type = self._infer_qtype(hf_quant_config, config_str)
        if quant_method == "fbgemm_fp8":
            # HF fbgemm_fp8 checkpoints store FP8 weights plus per-tensor
            # weight_scale tensors, while activations are dynamically quantized.
            # The config does not spell out a qscheme, so text heuristics would
            # otherwise leave dense linears as QuantType.No and run GEMM on raw
            # FP8 weights without scales.
            quant_type = QuantType.per_Tensor
        # MXFP4 (fp4x2) uses microscaling with 1x32 block scaling by definition
        if quant_dtype == d_dtypes.get("fp4x2") and quant_type not in (
            QuantType.per_1x32,
            QuantType.per_1x128,
        ):
            quant_type = QuantType.per_1x32
        weight_block_size = hf_quant_config.get("weight_block_size")
        # `activation_scheme: static` ships precomputed input_scales in the
        # checkpoint, so the activation quant is NOT dynamic (load the scales);
        # `dynamic` (or unspecified) quantizes activations at runtime.
        act_scheme = (hf_quant_config.get("activation_scheme") or "").lower()
        default_dynamic = act_scheme != "static"
        is_dynamic = hf_quant_config.get("is_dynamic", default_dynamic)
        # Each quantizer uses a different key for excluded layers:
        # Quark -> "exclude", compressed-tensors -> "ignore",
        # gpt-oss/HF transformers -> "modules_to_not_convert",
        # MiMo-V2-Flash/HF transformers -> "ignored_layers"
        exclude = list(
            hf_quant_config.get("ignore")
            or hf_quant_config.get("modules_to_not_convert")
            or hf_quant_config.get("exclude")
            or hf_quant_config.get("ignored_layers")
            or []
        )

        global_spec = LayerQuantConfig(
            quant_type=quant_type,
            quant_dtype=quant_dtype,
            is_dynamic=is_dynamic,
            quant_method=quant_method or None,
            weight_block_size=(
                tuple(weight_block_size)
                if isinstance(weight_block_size, (list, tuple))
                and len(weight_block_size) == 2
                else None
            ),
        )

        return ParsedQuantConfig(global_spec=global_spec, exclude_layers=exclude)

    def _infer_dtype(self, cfg: dict, config_str: str) -> Any:
        # Check explicit fields first
        for key in ("weight_dtype", "activation_dtype", "dtype"):
            val = cfg.get(key)
            if val and isinstance(val, str):
                parsed = _parse_quant_dtype(val)
                if parsed != torch.bfloat16:
                    return parsed
        # Check compressed-tensors config_groups (type + num_bits encoding)
        config_groups = cfg.get("config_groups")
        if isinstance(config_groups, dict):
            for group in config_groups.values():
                if not isinstance(group, dict):
                    continue
                weights = group.get("weights") or {}
                wtype = weights.get("type", "")
                num_bits = weights.get("num_bits")
                if wtype == "float" and num_bits == 8:
                    return d_dtypes.get("fp8", torch.bfloat16)
                if wtype == "float" and num_bits == 4:
                    return d_dtypes.get("fp4x2", torch.bfloat16)
                if wtype == "int" and num_bits == 8:
                    return d_dtypes.get("i8", torch.bfloat16)
        # Fall back to regex heuristics
        for pattern, dtype_key in self._DTYPE_PATTERNS.items():
            if re.search(pattern, config_str):
                return d_dtypes.get(dtype_key, torch.bfloat16)
        return torch.bfloat16

    def _infer_qtype(self, cfg: dict, config_str: str) -> QuantType:
        # Prefer explicit HF/compressed-tensors block size over text heuristics
        # so MXFP8 1x32 and blockscale 1x128/128x128 are not conflated.
        if "weight_block_size" in cfg:
            wbs = cfg.get("weight_block_size")
            if wbs is None:
                return QuantType.per_Tensor
            if isinstance(wbs, (list, tuple)) and len(wbs) >= 2:
                try:
                    m, n = int(wbs[0]), int(wbs[1])
                except (TypeError, ValueError):
                    m = n = None
                if (m, n) == (1, 128):
                    return QuantType.per_1x128
                if (m, n) == (128, 128):
                    # per_128x128 enum has no consumers in linear.py / GEMM dispatch yet;
                    # the per_1x128 path already allocates a (out//128, in//128)
                    # scale grid which is exactly the (128, 128) block layout.
                    return QuantType.per_1x128
                if (m, n) in ((1, 32), (32, 32)):
                    return QuantType.per_1x32
                return QuantType.per_1x128
        # Check explicit fields
        for key in ("quant_type", "quantization_type", "scheme"):
            val = cfg.get(key)
            if val and isinstance(val, str):
                for pattern, qtype in self._qtype_patterns().items():
                    if re.search(pattern, val.lower()):
                        return qtype
        # Check compressed-tensors config_groups for weight strategy
        config_groups = cfg.get("config_groups")
        if isinstance(config_groups, dict):
            for group in config_groups.values():
                if not isinstance(group, dict):
                    continue
                weights = group.get("weights") or {}
                strategy = weights.get("strategy", "")
                if strategy:
                    mapped = _qscheme_to_quant_type().get(strategy)
                    if mapped is None:
                        mapped = _qscheme_to_quant_type().get(f"per_{strategy}")
                    if mapped is not None:
                        return mapped
        # Fall back to regex heuristics on full config string
        for pattern, qtype in self._qtype_patterns().items():
            if re.search(pattern, config_str):
                return qtype
        # Bare compressed-tensors / vLLM fp8 (no weight_block_size, no config_groups,
        # no explicit scheme/strategy) — e.g. Llama-3.1-8B-Instruct-FP8-KV with
        # {"quant_method":"fp8","activation_scheme":"static"}. `activation_scheme`
        # distinguishes per-tensor (static) from per-token (dynamic); default fp8 to
        # per_Tensor so the weight/input scales actually get applied.
        quant_method = (cfg.get("quant_method") or "").lower()
        if quant_method in ("fp8", "float8") or re.search(r"fp8|float8", config_str):
            act_scheme = (cfg.get("activation_scheme") or "").lower()
            return (
                QuantType.per_Token if act_scheme == "dynamic" else QuantType.per_Tensor
            )
        return QuantType.No
