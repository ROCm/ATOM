"""Thin adapters over aiter entry points whose signatures are mid-rename.

ATOM tracks aiter main, but the released ATOM images freeze an aiter from
their build day. When aiter renames a keyword, models written against main
stop loading on the shipped image with a bare `TypeError`. Each adapter here
keeps one call site working on both spellings and is deleted once the oldest
supported image carries the new one.
"""

from __future__ import annotations

import inspect

from aiter.ops.inverse_rope_group_quant import (
    inverse_rope_group_quant as _aiter_inverse_rope_group_quant,
)

# aiter replaced the boolean `scale_shuffle` with the string `scale_layout`,
# which also gained layouts the boolean could not express. `scale_shuffle=False`
# and `scale_layout="row"` are the same row-major `[S, G, Ks]` scale.
_TAKES_SCALE_LAYOUT = (
    "scale_layout" in inspect.signature(_aiter_inverse_rope_group_quant).parameters
)


def inverse_rope_group_quant(*args, scale_layout: str = "row", **kwargs):
    """`aiter.ops.inverse_rope_group_quant` that accepts `scale_layout` always."""
    if _TAKES_SCALE_LAYOUT:
        return _aiter_inverse_rope_group_quant(
            *args, scale_layout=scale_layout, **kwargs
        )
    if scale_layout != "row":
        raise NotImplementedError(
            f"The installed aiter predates scale_layout and only emits the "
            f'row-major scale; scale_layout="{scale_layout}" needs aiter main.'
        )
    return _aiter_inverse_rope_group_quant(*args, scale_shuffle=False, **kwargs)
