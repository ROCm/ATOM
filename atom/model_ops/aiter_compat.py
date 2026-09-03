"""Thin adapters over aiter entry points whose signatures are mid-rename.

ATOM tracks aiter main, but the released ATOM images freeze an aiter from
their build day. When aiter renames a keyword, models written against main
stop loading on the shipped image with a bare `TypeError`. Each adapter here
keeps one call site working on both spellings and is deleted once the oldest
supported image carries the new one.
"""

from __future__ import annotations

import functools
import inspect

from aiter.ops.inverse_rope_group_quant import (
    inverse_rope_group_quant as _aiter_inverse_rope_group_quant,
)
from torch import Tensor

# aiter replaced the boolean `scale_shuffle` with the string `scale_layout`.
# The row layouts are identical. Legacy `scale_shuffle=True` used a different
# padded-K shape from the new `mfma_tile` contract, so it is not exposed here.
_LEGACY_SCALE_LAYOUTS = {
    "row": False,
}


@functools.cache
def _inverse_rope_scale_keyword() -> str:
    """Detect the installed aiter spelling without failing module import."""
    try:
        signature = inspect.signature(_aiter_inverse_rope_group_quant)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot inspect aiter inverse_rope_group_quant; expected either "
            "a scale_layout or scale_shuffle keyword."
        ) from exc

    for keyword in ("scale_layout", "scale_shuffle"):
        if keyword in signature.parameters:
            return keyword
    raise RuntimeError(
        "Unsupported aiter inverse_rope_group_quant signature "
        f"{signature}; expected either scale_layout or scale_shuffle."
    )


def inverse_rope_group_quant(
    o: Tensor,
    positions: Tensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
    num_groups: int,
    quant_group_size: int = 128,
    scale_layout: str = "row",
    x_fp8: Tensor | None = None,
    x_scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """`aiter.ops.inverse_rope_group_quant` that accepts `scale_layout` always."""
    common_kwargs = {
        "num_groups": num_groups,
        "quant_group_size": quant_group_size,
        "x_fp8": x_fp8,
        "x_scale": x_scale,
    }
    if _inverse_rope_scale_keyword() == "scale_layout":
        return _aiter_inverse_rope_group_quant(
            o,
            positions,
            cos_cache,
            sin_cache,
            scale_layout=scale_layout,
            **common_kwargs,
        )
    try:
        scale_shuffle = _LEGACY_SCALE_LAYOUTS[scale_layout]
    except KeyError as exc:
        raise NotImplementedError(
            "The compatibility adapter only maps the row layout on legacy "
            f'aiter; scale_layout="{scale_layout}" requires the current API.'
        ) from exc
    return _aiter_inverse_rope_group_quant(
        o,
        positions,
        cos_cache,
        sin_cache,
        scale_shuffle=scale_shuffle,
        **common_kwargs,
    )
