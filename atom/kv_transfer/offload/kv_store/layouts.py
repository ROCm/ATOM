# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The layouts a KV store offload moves, and the codec for each.

One row is the codec id folded into the store namespace and the builder the
worker calls. The scheduler and the worker both read it, so a new layout is
one entry rather than a branch in each caller. A backend moves the subset its
:class:`StoreLayouts` names. Under a draft window, a layout's row is its split
(:data:`DRAFT_WINDOW_LAYOUTS`): target objects without the draft's PAGE regions,
which :func:`build_draft_codec` packs into objects of their own. Codec modules
are imported here; Triton stays lazy inside the DSV4 codec.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import copy
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
# M3 under a draft window: a namespace of its own, so no key names unsplit bytes.
DRAFT_WINDOW_M3_CODEC_ID = "m3-page-regions-draft-window-v1"
# The `PageUnitSpec.namespace` of a draft's own PAGE regions.
DRAFT_PAGE_NAMESPACE = "draft"

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


def _draft_window_m3_codec_id() -> str:
    return DRAFT_WINDOW_M3_CODEC_ID


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


def split_draft_pages(transfer_tensors: Any) -> tuple[Any, Any | None]:
    """Copies of `transfer_tensors` with the target's PAGE regions and with the
    draft's, each in publication order; the draft's is None when it has none."""
    pages = list(transfer_tensors.pages)
    is_draft = [
        getattr(page.unit_spec, "namespace", None) == DRAFT_PAGE_NAMESPACE
        for page in pages
    ]
    if not any(is_draft):
        return transfer_tensors, None
    target, draft = copy(transfer_tensors), copy(transfer_tensors)
    target.pages = [page for page, own in zip(pages, is_draft) if not own]
    draft.pages = [page for page, own in zip(pages, is_draft) if own]
    return target, draft


def _build_m3_target(
    _kv_caches: Any,
    transfer_tensors: Any,
    num_blocks: int | None,
    _permit_per_request_state: bool,
) -> Any:
    return pure_page_codec(split_draft_pages(transfer_tensors)[0], num_blocks)


def build_draft_codec(transfer_tensors: Any, num_blocks: int | None) -> Any | None:
    """The codec of a draft window's draft objects; None for a rank whose
    transfer tensors hold no draft regions."""
    draft = split_draft_pages(transfer_tensors)[1]
    return None if draft is None else pure_page_codec(draft, num_blocks)


STORE_LAYOUTS: dict[str, StoreLayout] = {
    "dense": StoreLayout(_dense_codec_id, _build_dense),
    "m3": StoreLayout(_m3_codec_id, _build_m3),
    "dsv41": StoreLayout(_replay_codec_id, _build_replay_pages),
}
# The layouts a draft window splits, by their target objects.
DRAFT_WINDOW_LAYOUTS: dict[str, StoreLayout] = {
    "m3": StoreLayout(_draft_window_m3_codec_id, _build_m3_target),
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

    def require(self, name: str, *, draft_window: bool = False) -> StoreLayout:
        """The row for `name` (its split under a draft window), or the refusal
        the startup check reports."""
        if name not in self.names:
            raise ValueError(f"{self._refusal}; this model resolves to {name!r}")
        if not draft_window:
            return STORE_LAYOUTS[name]
        if name not in DRAFT_WINDOW_LAYOUTS:
            raise ValueError(
                f"a draft window splits the {', '.join(DRAFT_WINDOW_LAYOUTS)} "
                f"layout only; this model resolves to {name!r}"
            )
        return DRAFT_WINDOW_LAYOUTS[name]

    def codec_id(self, config: Any, *, draft_window: bool = False) -> str:
        """The codec version folded into this server's store namespace."""
        name = offcfg.select_offload_layout(config)
        return self.require(name, draft_window=draft_window).codec_id()

    def build_codec(
        self,
        config: Any,
        kv_caches: Any,
        transfer_tensors: Any,
        num_blocks: int | None,
        *,
        permit_per_request_state: bool,
        draft_window: bool = False,
    ) -> Any:
        """Build the codec for the layout `config` resolves to."""
        name = offcfg.select_offload_layout(config)
        layout = self.require(name, draft_window=draft_window)
        return layout.build(
            kv_caches,
            transfer_tensors,
            num_blocks,
            permit_per_request_state,
        )


__all__ = [
    "DENSE_CODEC_ID",
    "DRAFT_PAGE_NAMESPACE",
    "DRAFT_WINDOW_LAYOUTS",
    "DRAFT_WINDOW_M3_CODEC_ID",
    "REPLAY_PAGE_CODEC_ID",
    "STORE_LAYOUTS",
    "StoreLayout",
    "StoreLayouts",
    "build_draft_codec",
    "split_draft_pages",
]
