# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where a checkpoint image's bytes live in a Qwen3.8 QSA arena.

Same arithmetic as ``mha_page_unit_geometry``: one contiguous region per
field per layer, unit stride equal to one scheduler block. The arena is
layer-major, so a block's k, v and index rows are not one slice; the copy
walks the regions in field order, which is also the order
``get_kv_transfer_tensors`` publishes.
"""

from __future__ import annotations

from atom.model_ops.attentions.pool_layout.entry_arena import EntryField


class QsaRegionPool:
    """``region_tensors()`` over one ``LayerMajorArena``, for the MHA helpers."""

    def __init__(self, arena) -> None:
        rows: list[tuple[str, object]] = []
        for field in arena.fields:
            view = arena.view(field.name)
            for layer in range(field.layers):
                rows.append((f"qsa.{field.name}.{layer}", view[layer]))
        self._rows = rows

    def region_tensors(self):
        return self._rows


def qsa_published_page_bytes(fields: list[EntryField]) -> int:
    """Bytes one scheduler block publishes, alignment gaps excluded.

    ``entry_bytes_for`` also bills the padding between fields. Those gaps are
    not a region, and ``lmcache_mp`` requires the checkpoint unit to equal the
    published sum.
    """

    return sum(field.bytes_per_entry for field in fields)
