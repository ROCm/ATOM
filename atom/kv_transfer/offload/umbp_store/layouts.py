# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The layouts the UMBP offload moves, and the codec for each.

Copied from the Mooncake Store offload's ``layouts`` module; a shared
layer is planned. One row is the codec id folded into the object namespace
and the builder the worker calls. The scheduler and the worker both read
it, so a new layout is one entry rather than a branch in each caller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.dense.kv_byte_codec import DenseKVByteCodec
from atom.kv_transfer.offload.hybrid.m3 import codec as m3_codec
from atom.kv_transfer.offload.page_codec import pure_page_codec

# Bumped when the dense codec's per-chunk byte layout changes.
DENSE_CODEC_ID = "dense-opaque-block-v1"

CodecBuilder = Callable[[Any, Any, int | None, bool], Any]


@dataclass(frozen=True)
class UMBPLayout:
    """One layout this connector moves."""

    codec_id: Callable[[], str]
    build: CodecBuilder


def _dense_codec_id() -> str:
    return DENSE_CODEC_ID


def _m3_codec_id() -> str:
    return m3_codec.M3_CODEC_ID


def _build_dense(
    kv_caches: Any,
    _transfer_tensors: Any,
    num_blocks: int | None,
    permit_per_request_state: bool,
) -> DenseKVByteCodec:
    return DenseKVByteCodec(
        kv_caches,
        num_blocks=num_blocks,
        permit_per_request_state=permit_per_request_state,
    )


def _build_m3(
    _kv_caches: Any,
    transfer_tensors: Any,
    num_blocks: int | None,
    _permit_per_request_state: bool,
) -> Any:
    # Complete PAGE regions, the NSA index cache included.
    return pure_page_codec(transfer_tensors, num_blocks)


UMBP_LAYOUTS: dict[str, UMBPLayout] = {
    "dense": UMBPLayout(_dense_codec_id, _build_dense),
    "m3": UMBPLayout(_m3_codec_id, _build_m3),
}


def require_umbp_layout(name: str) -> UMBPLayout:
    """The row for `name`, or the refusal the startup check reports."""
    try:
        return UMBP_LAYOUTS[name]
    except KeyError:
        raise ValueError(
            "the UMBP offload moves only the dense KV layout or M3 PAGE "
            f"layout; this model resolves to {name!r}"
        ) from None


def umbp_codec_id(config: Any) -> str:
    """The codec version folded into this server's object namespace."""
    layout = require_umbp_layout(offcfg.select_offload_layout(config))
    return layout.codec_id()


def build_umbp_codec(
    config: Any,
    kv_caches: Any,
    transfer_tensors: Any,
    num_blocks: int | None,
    *,
    permit_per_request_state: bool,
) -> Any:
    """Build the codec for the layout `config` resolves to."""
    layout = require_umbp_layout(offcfg.select_offload_layout(config))
    return layout.build(
        kv_caches,
        transfer_tensors,
        num_blocks,
        permit_per_request_state,
    )
