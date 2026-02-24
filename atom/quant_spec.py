# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Per-layer quantization specification and parser registry.

This module introduces:
- ``LayerQuantSpec``: a frozen dataclass describing how a single layer
  should be quantized at init-time and (optionally) post-load-time.
- ``ParsedQuantConfig``: the output of a quant-method parser, containing
  a global spec, per-layer overrides, and the raw exclude list.
- ``QuantConfigParser`` ABC + ``_QUANT_PARSERS`` registry so new
  quant methods can be added without touching the core class.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from aiter import QuantType
from aiter.utility.dtypes import d_dtypes

logger = logging.getLogger("atom")

# ---------------------------------------------------------------------------
# LayerQuantSpec
# ---------------------------------------------------------------------------

# Sentinel for "no quant" — used as the default
_NO_QUANT_SPEC: Optional["LayerQuantSpec"] = None  # set after class def


@dataclass(frozen=True)
class LayerQuantSpec:
    """Immutable description of how a single linear layer is quantized.

    By making this a frozen dataclass we get:
    - Safe sharing between layers (immutable)
    - Easy equality checks and hashing
    - A single place to extend when new quantization knobs are needed
    """

    quant_type: QuantType = QuantType.No
    quant_dtype: torch.dtype = torch.bfloat16
    is_dynamic: bool = True
    quant_method: Optional[str] = None

    # Checkpoint format may differ from runtime quant_dtype.
    # e.g. MXFP4 checkpoints store bf16 source weights that are
    # quantized in-place during process_weights_after_loading.
    checkpoint_dtype: Optional[torch.dtype] = None

    # Extra flags for specialised paths (e.g. triton-gemm selection,
    # pre-shuffle hints, etc.).  Using a dict keeps LayerQuantSpec
    # extensible without adding a new field for every niche flag.
    flags: dict[str, Any] = field(default_factory=dict)

    # -- convenience predicates -------------------------------------------

    @property
    def is_quantized(self) -> bool:
        return self.quant_type != QuantType.No

    @property
    def needs_online_quant(self) -> bool:
        """True when the checkpoint stores BF16 weights that must be
        quantized in ``process_weights_after_loading``."""
        return self.checkpoint_dtype is not None and self.checkpoint_dtype != self.quant_dtype

    @classmethod
    def no_quant(cls) -> "LayerQuantSpec":
        """Canonical "no quantization" spec — BF16 weights, no scales."""
        global _NO_QUANT_SPEC
        if _NO_QUANT_SPEC is None:
            _NO_QUANT_SPEC = cls()
        return _NO_QUANT_SPEC


# ---------------------------------------------------------------------------
# ParsedQuantConfig — output of a parser
# ---------------------------------------------------------------------------


@dataclass
class ParsedQuantConfig:
    """Result returned by every ``QuantConfigParser.parse()`` call.

    Attributes:
        global_spec: The default spec applied to every layer unless
            overridden.
        layer_specs: Optional per-layer overrides keyed by layer prefix.
            When a layer's prefix is not in this dict the ``global_spec``
            is used.
        exclude_layers: The raw exclude-layer list from the HF config,
            kept for backward-compat logging / debugging.
    """

    global_spec: LayerQuantSpec
    layer_specs: dict[str, LayerQuantSpec] = field(default_factory=dict)
    exclude_layers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------


class QuantConfigParser(ABC):
    """Abstract base for quant-method parsers.

    To register a new parser:

        @register_quant_parser("my_method")
        class MyParser(QuantConfigParser):
            def parse(self, raw_config: dict) -> ParsedQuantConfig:
                ...
    """

    @abstractmethod
    def parse(self, raw_config: dict) -> ParsedQuantConfig:
        """Parse a raw HF ``quantization_config`` dict into a
        ``ParsedQuantConfig``."""
        ...


_QUANT_PARSERS: dict[str, type[QuantConfigParser]] = {}


def register_quant_parser(method_name: str):
    """Decorator to register a ``QuantConfigParser`` subclass for a given
    ``quant_method`` string."""

    def wrapper(cls: type[QuantConfigParser]):
        if method_name in _QUANT_PARSERS:
            logger.warning(
                "Overwriting quant parser for method %r with %s",
                method_name,
                cls.__name__,
            )
        _QUANT_PARSERS[method_name] = cls
        return cls

    return wrapper


def get_quant_parser(method_name: str) -> QuantConfigParser:
    """Instantiate and return the parser for *method_name*.

    Falls back to ``GenericParser`` when no specific parser is registered.
    """
    cls = _QUANT_PARSERS.get(method_name)
    if cls is None:
        logger.warning(
            "No dedicated quant parser for method %r — falling back to GenericParser",
            method_name,
        )
        cls = _QUANT_PARSERS.get("__generic__", GenericParser)
    return cls()


# ---------------------------------------------------------------------------
# Shared parsing helpers
# ---------------------------------------------------------------------------

RE_QUANT_BLOCKSIZE = r"\'(?:group_size|weight_block_size)\'\:\s*(?:\[\n*)\s*(\d+),"
RE_QUANT_DTYPE = r"\'(?:d?type|weight_dtype|quant_method)\'\:\s*\'(\w+)\'"
RE_STATIC_QUANT = r"\'(?:activation_scheme)\'\:\s*\'(static)\'"


def _parse_quant_type(raw_str: str) -> QuantType:
    """Infer QuantType from the stringified HF config."""
    if "channel'," in raw_str:
        return QuantType.per_Token
    if group_size := re.search(RE_QUANT_BLOCKSIZE, raw_str):
        gs = int(group_size.group(1))
        assert gs in (32, 128), f"Unsupported group size {gs}"
        return QuantType.per_1x128 if gs == 128 else QuantType.per_1x32
    return QuantType.per_Tensor


def _parse_quant_dtype(raw_str: str) -> torch.dtype:
    """Infer torch dtype from stringified HF config."""
    m = re.search(RE_QUANT_DTYPE, raw_str)
    if m and m.group(1).lower() in [
        "fp8", "fp4", "int8", "int4", "fp8_e4m3", "mxfp4",
    ]:
        dtype = m.group(1).lower().split("_")[0]
        if dtype == "mxfp4":
            dtype = "fp4"
        if dtype.endswith("4"):
            dtype += "x2"
        return d_dtypes[dtype]

    bit_match = re.search(r"\'(?:num_)?bits\'\:\s*(\d+)", raw_str)
    if bit_match:
        bit = int(bit_match.group(1))
        dtype_match = re.search(RE_QUANT_DTYPE, raw_str)
        if dtype_match:
            dtype = dtype_match.group(1).lower()
            dtype_prefix = "i" if dtype.startswith("int") else "fp"
        else:
            dtype_prefix = "i"
        quant_dtype_str = f"{dtype_prefix}{bit}" if bit != 4 else f"{dtype_prefix}{bit}x2"
        result = d_dtypes.get(quant_dtype_str, None)
        if result is not None:
            return result

    raise ValueError(f"Cannot parse quant dtype from {raw_str}")


def _parse_is_dynamic(raw_str: str) -> bool:
    return not bool(re.search(RE_STATIC_QUANT, raw_str))


# ---------------------------------------------------------------------------
# Built-in parsers
# ---------------------------------------------------------------------------


@register_quant_parser("quark")
class QuarkParser(QuantConfigParser):
    """Parser for the ``quark`` quantization method (used by AMD MXFP4
    checkpoints, among others)."""

    def parse(self, raw_config: dict) -> ParsedQuantConfig:
        # Extract weight sub-config for dtype/quant_type parsing
        weight_cfg = raw_config.get("global_quant_config", {}).get("weight", {})
        weight_str = str(weight_cfg) if weight_cfg else str(raw_config)
        raw_str = str(raw_config)

        quant_dtype = _parse_quant_dtype(weight_str)
        is_dynamic = _parse_is_dynamic(raw_str)

        # Prefer structured group_size from the dict over regex
        group_size = weight_cfg.get("group_size", None) if weight_cfg else None
        if isinstance(group_size, list):
            group_size = group_size[0]
        if group_size in (32, 128):
            quant_type = QuantType.per_1x128 if group_size == 128 else QuantType.per_1x32
        else:
            quant_type = _parse_quant_type(weight_str)

        # quark uses "exclude" for the exclude-layers key
        exclude_layers = raw_config.get("exclude", []) or []

        # For FP4 dtype, force per_1x32
        if quant_dtype == d_dtypes["fp4x2"]:
            quant_type = QuantType.per_1x32

        global_spec = LayerQuantSpec(
            quant_type=quant_type,
            quant_dtype=quant_dtype,
            is_dynamic=is_dynamic,
            quant_method="quark",
        )

        return ParsedQuantConfig(
            global_spec=global_spec,
            exclude_layers=exclude_layers,
        )


@register_quant_parser("compressed-tensors")
class CompressedTensorsParser(QuantConfigParser):
    """Parser for the ``compressed-tensors`` quantization method."""

    def parse(self, raw_config: dict) -> ParsedQuantConfig:
        raw_str = str(raw_config)
        quant_type = QuantType.per_Token  # compressed-tensors → per_Token
        quant_dtype = _parse_quant_dtype(raw_str)
        is_dynamic = _parse_is_dynamic(raw_str)

        # compressed-tensors uses "ignore" for exclude layers
        exclude_layers = raw_config.get("ignore", []) or []

        if quant_dtype == d_dtypes["fp4x2"]:
            quant_type = QuantType.per_1x32

        global_spec = LayerQuantSpec(
            quant_type=quant_type,
            quant_dtype=quant_dtype,
            is_dynamic=is_dynamic,
            quant_method="compressed-tensors",
        )

        return ParsedQuantConfig(
            global_spec=global_spec,
            exclude_layers=exclude_layers,
        )


@register_quant_parser("__generic__")
class GenericParser(QuantConfigParser):
    """Fallback parser used when no method-specific parser is registered."""

    def parse(self, raw_config: dict) -> ParsedQuantConfig:
        raw_str = str(raw_config)
        quant_method = raw_config.get("quant_method", None)
        quant_type = _parse_quant_type(raw_str)
        quant_dtype = _parse_quant_dtype(raw_str)
        is_dynamic = _parse_is_dynamic(raw_str)

        if quant_dtype == d_dtypes["fp4x2"]:
            quant_type = QuantType.per_1x32

        # Best-effort: try "ignore" first, then "exclude"
        exclude_layers = raw_config.get("ignore", None) or raw_config.get("exclude", []) or []

        logger.warning(
            "Using generic quant parser for method %r — "
            "please verify exclude_layers key is correct.",
            quant_method,
        )

        global_spec = LayerQuantSpec(
            quant_type=quant_type,
            quant_dtype=quant_dtype,
            is_dynamic=is_dynamic,
            quant_method=quant_method,
        )

        return ParsedQuantConfig(
            global_spec=global_spec,
            exclude_layers=exclude_layers,
        )
