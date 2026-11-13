# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The layouts the Mooncake Store offload moves, and the codec for each.

One row is the codec id folded into the Store namespace and the builder the
worker calls. The scheduler and the worker both read it, so a new layout is
one entry rather than a branch in each caller. Codec modules are imported
here; Triton stays lazy inside the DSV4 codec.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import atom.kv_transfer.offload.hybrid.m3.codec as m3_codec
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.capabilities import replays_state
from atom.kv_transfer.offload.dense.kv_byte_codec import DenseKVByteCodec
from atom.kv_transfer.offload.metadata_lease import METADATA_GPU_RESERVE_BYTES
from atom.kv_transfer.offload.page_codec import pure_page_codec

# Bumped when the dense codec's per-chunk byte layout changes.
DENSE_CODEC_ID = "dense-opaque-block-v1"
# V4.1 stores PAGE regions only; request state is rebuilt by replay, so its
# objects must not share a namespace with dense or M3 bytes.
REPLAY_PAGE_CODEC_ID = "v41-replay-page-v1"

CodecBuilder = Callable[[Any, Any, int | None, bool], Any]


@dataclass(frozen=True)
class MooncakeLayout:
    """One layout this connector moves."""

    codec_id: Callable[[], str]
    build: CodecBuilder
    # PAGE codecs upload their block IDs through a per-thread metadata lease.
    metadata_leases: bool = False


def _dense_codec_id() -> str:
    return DENSE_CODEC_ID


def _m3_codec_id() -> str:
    return m3_codec.M3_CODEC_ID


def _replay_codec_id() -> str:
    return REPLAY_PAGE_CODEC_ID


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
    return pure_page_codec(transfer_tensors, num_blocks)


def _build_replay_pages(
    _kv_caches: Any,
    transfer_tensors: Any,
    num_blocks: int | None,
    _permit_per_request_state: bool,
) -> Any:
    """V4.1 PAGE bytes. The replay contract is what rebuilds request state."""
    if not replays_state(transfer_tensors):
        raise ValueError("Mooncake Store V4.1 requires the PAGE replay contract")
    # A P/D leg also publishes the live SLOT. Store prefixes carry only PAGEs,
    # so the codec is built on a copy without it (as the MP tier does).
    return pure_page_codec(transfer_tensors.without_slots(), num_blocks)


MOONCAKE_LAYOUTS: dict[str, MooncakeLayout] = {
    "dense": MooncakeLayout(_dense_codec_id, _build_dense),
    "m3": MooncakeLayout(_m3_codec_id, _build_m3, metadata_leases=True),
    "dsv41": MooncakeLayout(
        _replay_codec_id, _build_replay_pages, metadata_leases=True
    ),
}


def require_mooncake_layout(name: str) -> MooncakeLayout:
    """The row for `name`, or the refusal the startup check reports."""
    try:
        return MOONCAKE_LAYOUTS[name]
    except KeyError:
        raise ValueError(
            "the Mooncake Store offload moves the dense KV layout, M3 PAGE "
            f"layout, and V4.1's replay PAGEs; this model resolves to {name!r}"
        ) from None


def metadata_reserve_bytes(config: Any) -> int:
    """HBM one transfer thread's metadata lease pool may hold."""
    layout = MOONCAKE_LAYOUTS.get(offcfg.select_offload_layout(config))
    return METADATA_GPU_RESERVE_BYTES if layout and layout.metadata_leases else 0


def mooncake_codec_id(config: Any) -> str:
    """The codec version folded into this server's Store namespace."""
    layout = require_mooncake_layout(offcfg.select_offload_layout(config))
    return layout.codec_id()


def build_mooncake_codec(
    config: Any,
    kv_caches: Any,
    transfer_tensors: Any,
    num_blocks: int | None,
    *,
    permit_per_request_state: bool,
) -> Any:
    """Build the codec for the layout `config` resolves to."""
    layout = require_mooncake_layout(offcfg.select_offload_layout(config))
    return layout.build(
        kv_caches,
        transfer_tensors,
        num_blocks,
        permit_per_request_state,
    )
