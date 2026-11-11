# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Where a block's weights come from: mono-packed copies or tensors bound as they are.

A block's emit code reads each weight arg in one *layout*, fixed at build time. ``Block.weight_specs()`` declares, per
pointer-table arg, a ``WeightSpec`` (layout name, logical shape and dtype, base alignment, required attributes).
``bind(block, source)`` returns the block's pointer-table entries from ``Packed(layer)`` (``Block.pack`` of a
``QuantizedLayer``) or ``InPlace(tensors)`` (bound without a copy), after checking every tensor against its spec
(``check``: dtype, shape, contiguity, alignment, attributes, then the layout's own check from ``LAYOUTS``). A mismatch
raises ``LayoutError`` at bind time, before any launch reads garbage.

The MoE layouts ``aiter.a8w4.moe`` / ``aiter.e8m0.moe`` are aiter's A8W4 expert tensors (``aiter_moe``): their check
reads the first and the last expert and requires the nonzero 16 x 128 units to be exactly the used ones
(``WeightSpec.params`` ``gate_up``, ``n``, ``k``, ``experts``), which tells aiter's interleaved layout from others.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch


class LayoutError(ValueError):
    """A tensor does not match the layout its block reads."""


@dataclass(frozen=True)
class WeightSpec:
    """One weight arg of a block: what its emit code reads."""

    arg: str  # the pointer-table arg
    layout: str  # a ``LAYOUTS`` name
    shape: tuple[int, ...]  # the logical shape in that layout (elements of ``dtype``)
    dtypes: tuple[
        torch.dtype, ...
    ]  # accepted element types (same bytes, e.g. uint8 / fp4x2)
    align: int = 16  # base address alignment in bytes the loads need
    attrs: tuple[
        tuple[str, object], ...
    ] = ()  # (attribute, value) the tensor must carry
    params: tuple[
        tuple[str, object], ...
    ] = ()  # (name, value) the layout's own check reads


LAYOUTS: dict[str, Callable[[torch.Tensor, WeightSpec], None]] = {}


def layout(name: str):
    """Register ``fn(tensor, spec)`` (raise ``LayoutError``) as the check of layout ``name``."""

    def deco(fn):
        LAYOUTS[name] = fn
        return fn

    return deco


def _dtypes(*names: str) -> tuple[torch.dtype, ...]:
    return tuple(getattr(torch, n) for n in names if hasattr(torch, n))


U8 = (torch.uint8,)
FP4X2 = _dtypes("uint8", "float4_e2m1fn_x2")
E8M0 = _dtypes("uint8", "float8_e8m0fnu")


@layout("mono.w8.gemv")
def _w8_gemv(t, spec):
    """``gemv.pack_weight``'s MXFP8 weight bytes (``Linear(weight_format="mxfp8")``), uint8 flat [N * K]."""
    if t.dim() != 1 or t.numel() % (16 * 128):
        raise LayoutError(
            f"{spec.arg}: {spec.layout} is flat whole 16 x 128 units, got {tuple(t.shape)}"
        )


@layout("mono.e8m0.gemv")
def _e8m0_gemv(t, spec):
    if t.dim() != 1 or t.numel() % 64:
        raise LayoutError(
            f"{spec.arg}: {spec.layout} is flat whole 64-code units, got {tuple(t.shape)}"
        )


@layout("bf16")
def _bf16(t, spec):
    pass


@layout("mono.bf16.gemv16")
def _bf16_gemv16(t, spec):
    if t.dim() != 1 or t.numel() % (16 * 128):
        raise LayoutError(
            f"{spec.arg}: {spec.layout} is flat whole 16 x 128 units, got {tuple(t.shape)}"
        )


@layout("mono.bf16.gemv16m")
def _bf16_gemv16m(t, spec):
    if t.dim() != 1 or t.numel() % (16 * 128):
        raise LayoutError(
            f"{spec.arg}: {spec.layout} is flat whole 16 x 128 units, got {tuple(t.shape)}"
        )


@layout("f32")
def _f32(t, spec):
    if t.dim() != 1:
        raise LayoutError(
            f"{spec.arg}: {spec.layout} is a flat fp32 vector, got {tuple(t.shape)}"
        )


@layout("aiter.a8w4.moe")
def _a8w4_moe(t, spec):
    e, n, kb = spec.shape
    if n % 32 or (kb * 2) % 256:
        raise LayoutError(
            f"{spec.arg}: {spec.layout} needs N % 32 == 0 and K % 256 == 0, got {spec.shape}"
        )
    _moe_units(t, spec, e, n, kb * 2, "weights")


@layout("aiter.e8m0.moe")
def _e8m0_moe(t, spec):
    if t.dim() != 2 or len(spec.shape) != 2:
        raise LayoutError(
            f"{spec.arg}: {spec.layout} is [E * N, K / 32], got {tuple(t.shape)}"
        )
    p = dict(spec.params)
    e = p.get("experts")
    if e is None or spec.shape[0] % e or (spec.shape[0] // e) % 32 or spec.shape[1] % 8:
        raise LayoutError(
            f"{spec.arg}: {spec.layout} needs params experts and N % 32, K % 256 == 0, got {spec}"
        )
    _moe_units(t, spec, e, spec.shape[0] // e, spec.shape[1] * 32, "scales")


def _moe_units(t, spec, e, n_stored, k_stored, what):
    """The used units of the first and the last expert hold data, the padding is zero."""
    from atom.models.qwen38.mono.kernels.weights import aiter_moe

    p = dict(spec.params)
    if not {"gate_up", "n", "k"} <= set(p):
        raise LayoutError(
            f"{spec.arg}: {spec.layout} needs params gate_up, n, k, got {spec.params}"
        )
    want = aiter_moe.used_units(p["gate_up"], p["n"], p["k"], n_stored, k_stored)
    for x in sorted({0, e - 1}):
        if what == "weights":
            got = aiter_moe.unit_mask(t, x).cpu()
        else:
            got = aiter_moe.scale_mask(t, e, x).cpu()
        if not torch.equal(got, want):
            bad = (got != want).nonzero()[:4].tolist()
            raise LayoutError(
                f"{spec.arg}: expert {x}'s {what} are not aiter's interleaved A8W4 layout of a used "
                f"[{p['n']}, {p['k']}] in [{n_stored}, {k_stored}] (gate_up {p['gate_up']}): units "
                f"(16-row block, 128-K step) {bad} hold data where padding belongs, or the reverse"
            )


def check(t: torch.Tensor, spec: WeightSpec) -> None:
    """Raise ``LayoutError`` unless ``t`` is what ``spec`` says its block reads."""
    if spec.layout not in LAYOUTS:
        raise LayoutError(
            f"{spec.arg}: unknown layout {spec.layout!r} (have {sorted(LAYOUTS)})"
        )
    if t.dtype not in spec.dtypes:
        raise LayoutError(
            f"{spec.arg}: {spec.layout} wants {spec.dtypes}, got {t.dtype}"
        )
    if tuple(t.shape) != tuple(spec.shape):
        raise LayoutError(
            f"{spec.arg}: {spec.layout} shape {tuple(spec.shape)}, got {tuple(t.shape)}"
        )
    if not t.is_contiguous():
        raise LayoutError(
            f"{spec.arg}: {spec.layout} must be contiguous, got strides {t.stride()}"
        )
    if t.data_ptr() % spec.align:
        raise LayoutError(f"{spec.arg}: base address not {spec.align}-byte aligned")
    for name, want in spec.attrs:
        if getattr(t, name, None) != want:
            raise LayoutError(
                f"{spec.arg}: {spec.layout} needs {name} = {want!r}, got {getattr(t, name, None)!r}"
            )
    LAYOUTS[spec.layout](t, spec)


class WeightSource:
    """Supplies a block's weight args (``tensors(block) -> {arg: tensor}``)."""

    in_place = False


class Packed(WeightSource):
    """mono-packed copies of a ``QuantizedLayer``'s weights (``block.pack(layer)``)."""

    def __init__(self, layer):
        self.layer = layer

    def tensors(self, block):
        return block.pack(self.layer)


class InPlace(WeightSource):
    """Tensors owned elsewhere (Atom's parameters), keyed by the block's arg names, bound
    without a copy."""

    in_place = True

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.t = dict(tensors)

    def tensors(self, block):
        missing = [s.arg for s in block.weight_specs() if s.arg not in self.t]
        if missing:
            raise LayoutError(f"{block.name}: no in-place tensor for {missing}")
        return {s.arg: self.t[s.arg] for s in block.weight_specs()}


def bind(block, source: WeightSource) -> dict[str, torch.Tensor]:
    """``block``'s weight args from ``source``, every one checked against ``block.weight_specs()``."""
    got = source.tensors(block)
    specs = block.weight_specs()
    for spec in specs:
        if spec.arg not in got:
            raise LayoutError(
                f"{block.name}: {type(source).__name__} gave no {spec.arg}"
            )
        check(got[spec.arg], spec)
    if source.in_place:
        for spec in specs:
            if got[spec.arg].data_ptr() != source.t[spec.arg].data_ptr():
                raise LayoutError(
                    f"{block.name}: {spec.arg} was copied, not bound in place"
                )
    return got
