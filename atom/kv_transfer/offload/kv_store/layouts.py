# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The layouts a KV store offload moves, and the codec for each.

One row is the codec id folded into the store namespace and the builder the
worker calls. The scheduler and the worker both read it, so a new layout is
one entry rather than a branch in each caller. A backend moves the subset its
:class:`StoreLayouts` names. Codec modules are imported here; Triton stays lazy
inside the DSV4 codec.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import atom.kv_transfer.offload.hybrid.m3.codec as m3_codec
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.capabilities import replays_state
from atom.kv_transfer.offload.dense.kv_byte_codec import DenseKVByteCodec
from atom.kv_transfer.offload.page_codec import pure_page_codec

# Bumped when the dense codec's per-chunk byte layout changes.
DENSE_CODEC_ID = "dense-opaque-block-v1"
# V4.1 stores PAGE regions only; request state is rebuilt by replay, so its
# objects must not share a namespace with dense or M3 bytes.
REPLAY_PAGE_CODEC_ID = "v41-replay-page-v1"

CodecBuilder = Callable[[Any, Any, int | None, bool], Any]


@dataclass(frozen=True)
class StoreLayout:
    """One layout a KV store offload moves."""

    codec_id: Callable[[], str]
    build: CodecBuilder


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
    # Complete PAGE regions, the NSA index cache included.
    return pure_page_codec(transfer_tensors, num_blocks)


def _build_replay_pages(
    _kv_caches: Any,
    transfer_tensors: Any,
    num_blocks: int | None,
    _permit_per_request_state: bool,
) -> Any:
    """V4.1 PAGE bytes. The replay contract is what rebuilds request state."""
    if not replays_state(transfer_tensors):
        raise ValueError("V4.1 in a KV store requires the PAGE replay contract")
    # A P/D leg also publishes the live SLOT. Store prefixes carry only PAGEs,
    # so the codec is built on a copy without it (as the MP tier does).
    return pure_page_codec(transfer_tensors.without_slots(), num_blocks)


STORE_LAYOUTS: dict[str, StoreLayout] = {
    "dense": StoreLayout(_dense_codec_id, _build_dense),
    "m3": StoreLayout(_m3_codec_id, _build_m3),
    "dsv41": StoreLayout(_replay_codec_id, _build_replay_pages),
}


class StoreLayouts:
    """The layouts one backend moves: a subset of :data:`STORE_LAYOUTS`."""

    def __init__(self, names: Iterable[str], *, refusal: str) -> None:
        self.names = tuple(names)
        unknown = [name for name in self.names if name not in STORE_LAYOUTS]
        if unknown:
            raise ValueError(f"no store layout named {unknown}")
        # What the startup check says the backend moves, before "; this
        # model resolves to ...".
        self._refusal = refusal

    def require(self, name: str) -> StoreLayout:
        """The row for `name`, or the refusal the startup check reports."""
        if name not in self.names:
            raise ValueError(f"{self._refusal}; this model resolves to {name!r}")
        return STORE_LAYOUTS[name]

    def codec_id(self, config: Any) -> str:
        """The codec version folded into this server's store namespace."""
        return self.require(offcfg.select_offload_layout(config)).codec_id()

    def build_codec(
        self,
        config: Any,
        kv_caches: Any,
        transfer_tensors: Any,
        num_blocks: int | None,
        *,
        permit_per_request_state: bool,
    ) -> Any:
        """Build the codec for the layout `config` resolves to."""
        layout = self.require(offcfg.select_offload_layout(config))
        return layout.build(
            kv_caches,
            transfer_tensors,
            num_blocks,
            permit_per_request_state,
        )


def check_store_engine(
    config: Any, chunk_tokens: int, *, offload: str, chunk_setting: str
) -> None:
    """Refuse DCP and a chunk that is not a whole number of KV blocks.

    A DCP rank's pages cover a virtual block that the chunk keys do not name.
    ``offload`` names the backend in the errors ("the Mooncake Store
    offload"), ``chunk_setting`` its chunk setting.

    Raises:
        ValueError: DCP > 1, or the chunk is not a whole number of KV blocks.
    """
    dcp = int(config.decode_context_parallel_size or 1)
    if dcp > 1:
        raise ValueError(
            f"{offload} does not support decode context parallelism "
            f"(decode_context_parallel_size={dcp})"
        )
    block_size = offcfg._strict_integer(
        "KV cache block size", config.kv_cache_block_size, minimum=1
    )
    if chunk_tokens % block_size:
        raise ValueError(
            f"{chunk_setting}={chunk_tokens} must be a multiple "
            f"of the KV cache block size {block_size}"
        )


__all__ = [
    "DENSE_CODEC_ID",
    "REPLAY_PAGE_CODEC_ID",
    "STORE_LAYOUTS",
    "StoreLayout",
    "StoreLayouts",
    "check_store_engine",
]
