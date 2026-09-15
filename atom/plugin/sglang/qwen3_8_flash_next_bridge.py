# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang ForwardBatch → Native Qwen3.8-Flash-Next QSA / PLE metadata.

Compute stays in Native ATOM (#2048): QSA, indexer, GDN, hyper-connection, PLE.
This module only translates the current step's page tables into the structs
those kernels already read. Geometry is never cached across steps.

First knife: eager text. Decode CUDA-graph capture must not D2H-sync
(`.item()`). Prefill graphs are not required. Decode-graph replay refreshes
QSA page tables through persistent buffers filled in
`init_forward_metadata_out_graph` (see `prepare_flash_decode_graph_metadata`).
PLE table stays in Native; do not enable SGLang --ple-offload-embedding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from atom.plugin.sglang.attention_backend.backend_resolver import (
    real_batch_size,
    resolve_attn_backend,
    resolve_mamba_req_pool,
)
from atom.plugin.sglang.deepseek_v4_bridge import _build_block_tables

logger = logging.getLogger(__name__)

# Dummy / graph-padding rows must not write QSA caches. Native kernels treat -1
# as no-write (same idea as the Qwen3.5 SGLang sentinel in #2067).
_NO_WRITE = -1


class _FlashDecodeGraphBuffers:
    """Persistent QSA tensors whose addresses are baked into decode CUDA graphs.

    SGLang decode CUDA graphs capture kernel pointer operands. Metadata built
    with fresh allocations inside ``model.forward`` becomes stale on replay
    because Python does not re-run. These buffers are filled in-place from
    ``init_forward_metadata_out_graph`` before every capture/replay, and
    ``build_qsa_metadata`` returns views into the same storage so capture and
    replay share addresses.
    """

    def __init__(self) -> None:
        self.max_bs = 0
        self.max_tokens = 0
        self.max_pages = 0
        self.ngram_context_len = 1
        self.device: torch.device | None = None
        self.block_tables: torch.Tensor | None = None
        self.slot_mapping: torch.Tensor | None = None
        self.compressed_slot_mapping: torch.Tensor | None = None
        self.token_to_req: torch.Tensor | None = None
        self.logical_positions: torch.Tensor | None = None
        self.seq_lens: torch.Tensor | None = None
        self.query_start_loc: torch.Tensor | None = None
        self.ngram_context: torch.Tensor | None = None
        self.state_indices_in: torch.Tensor | None = None
        self.state_indices_out: torch.Tensor | None = None
        self.has_initial_state: torch.Tensor | None = None
        self.max_seq_len = 1
        self.active = False
        self.last_qsa: Qwen3_8FlashNextQSAMetadata | None = None
        self.last_ple: Qwen3_8FlashNextPLEMetadata | None = None
        self.model: Any | None = None
        self.atom_config: Any | None = None
        self._logged_block_size = False

    def ensure(
        self,
        *,
        max_bs: int,
        max_tokens: int,
        max_pages: int,
        device: torch.device,
        ngram_context_len: int = 1,
    ) -> None:
        max_bs = max(int(max_bs), 1)
        max_tokens = max(int(max_tokens), max_bs)
        max_pages = max(int(max_pages), 1)
        ngram_context_len = max(int(ngram_context_len), 1)
        need = (
            self.block_tables is None
            or self.device != device
            or self.max_bs < max_bs
            or self.max_tokens < max_tokens
            or self.max_pages < max_pages
            or self.ngram_context_len < ngram_context_len
            or self.query_start_loc is None
        )
        if not need:
            return
        # CUDA graphs bake buffer *addresses*. Growing after a capture leaves
        # older graphs pointing at freed storage — always allocate once to the
        # high-water mark and never replace live buffers mid-serve if possible.
        grew_after_init = self.block_tables is not None
        self.max_bs = max(self.max_bs, max_bs)
        self.max_tokens = max(self.max_tokens, max_tokens)
        self.max_pages = max(self.max_pages, max_pages)
        self.ngram_context_len = max(self.ngram_context_len, ngram_context_len)
        self.device = device
        if grew_after_init:
            logger.warning(
                "Flash decode graph QSA buffers grew after init "
                "(bs=%s tokens=%s pages=%s); existing CUDA graphs may be stale",
                self.max_bs,
                self.max_tokens,
                self.max_pages,
            )
        self.block_tables = torch.zeros(
            (self.max_bs, self.max_pages), dtype=torch.int32, device=device
        )
        self.slot_mapping = torch.full(
            (self.max_tokens,), _NO_WRITE, dtype=torch.int64, device=device
        )
        self.compressed_slot_mapping = torch.full(
            (self.max_tokens,), _NO_WRITE, dtype=torch.int64, device=device
        )
        self.token_to_req = torch.zeros(
            (self.max_tokens,), dtype=torch.int32, device=device
        )
        self.logical_positions = torch.full(
            (self.max_tokens,), _NO_WRITE, dtype=torch.int64, device=device
        )
        self.seq_lens = torch.zeros((self.max_bs,), dtype=torch.int32, device=device)
        self.query_start_loc = torch.zeros(
            (self.max_bs + 1,), dtype=torch.int32, device=device
        )
        self.ngram_context = torch.zeros(
            (self.max_bs, self.ngram_context_len), dtype=torch.int64, device=device
        )
        self.state_indices_in = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device=device
        )
        self.state_indices_out = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device=device
        )
        self.has_initial_state = torch.ones(
            (self.max_bs,), dtype=torch.bool, device=device
        )
        self.active = True

    def preallocate_for_decode_graph(
        self,
        *,
        max_bs: int,
        max_pages: int,
        device: torch.device,
        max_tokens: int | None = None,
        ngram_context_len: int = 1,
    ) -> None:
        """Reserve persistent capacity before any CUDA-graph capture."""
        self.ensure(
            max_bs=max_bs,
            max_tokens=max_tokens if max_tokens is not None else max_bs,
            max_pages=max_pages,
            device=device,
            ngram_context_len=ngram_context_len,
        )

    def materialize_qsa(
        self,
        *,
        block_tables: torch.Tensor,
        slot_mapping: torch.Tensor,
        compressed_slot_mapping: torch.Tensor,
        token_to_req: torch.Tensor,
        logical_positions: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> Qwen3_8FlashNextQSAMetadata:
        bs = int(seq_lens.shape[0])
        num_tokens = int(slot_mapping.shape[0])
        pages = int(block_tables.shape[1])
        # Never grow page width from a live (possibly mis-sized) table after
        # preallocate — a wrong block_size=16 build is 128 pages vs 32.
        target_pages = self.max_pages if self.max_pages > 0 else pages
        self.ensure(
            max_bs=bs,
            max_tokens=num_tokens,
            max_pages=target_pages,
            device=block_tables.device,
        )
        assert self.block_tables is not None
        assert self.slot_mapping is not None
        assert self.compressed_slot_mapping is not None
        assert self.token_to_req is not None
        assert self.logical_positions is not None
        assert self.seq_lens is not None
        self.block_tables.zero_()
        self.slot_mapping.fill_(_NO_WRITE)
        self.compressed_slot_mapping.fill_(_NO_WRITE)
        self.token_to_req.zero_()
        self.logical_positions.fill_(_NO_WRITE)
        self.seq_lens.zero_()

        # Always expose the full preallocated page width so every capture
        # bucket bakes the same page-table shape; pad shorter builds with 0.
        out_pages = self.max_pages
        copy_pages = min(pages, out_pages)
        self.block_tables[:bs, :copy_pages].copy_(
            block_tables[:, :copy_pages].to(dtype=torch.int32)
        )
        self.slot_mapping[:num_tokens].copy_(slot_mapping.to(dtype=torch.int64))
        self.compressed_slot_mapping[:num_tokens].copy_(
            compressed_slot_mapping.to(dtype=torch.int64)
        )
        self.token_to_req[:num_tokens].copy_(token_to_req.to(dtype=torch.int32))
        self.logical_positions[:num_tokens].copy_(
            logical_positions.to(dtype=torch.int64)
        )
        self.seq_lens[:bs].copy_(seq_lens.to(dtype=torch.int32))
        self.max_seq_len = max(int(max_seq_len), 1)
        self.last_qsa = Qwen3_8FlashNextQSAMetadata(
            block_tables=self.block_tables[:bs, :out_pages],
            slot_mapping=self.slot_mapping[:num_tokens],
            compressed_slot_mapping=self.compressed_slot_mapping[:num_tokens],
            token_to_req=self.token_to_req[:num_tokens],
            logical_positions=self.logical_positions[:num_tokens],
            seq_lens=self.seq_lens[:bs],
            max_seq_len=self.max_seq_len,
        )
        return self.last_qsa

    def materialize_ple(
        self,
        *,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        ngram_state: torch.Tensor,
        state_indices_in: torch.Tensor,
        state_indices_out: torch.Tensor,
        has_initial_state: torch.Tensor,
        conv_state: torch.Tensor,
        num_reqs: int,
        is_prefill: bool,
        max_query_len: int,
    ) -> Qwen3_8FlashNextPLEMetadata:
        bs = int(num_reqs)
        ngram_w = int(ngram_context.shape[1]) if ngram_context.dim() == 2 else 1
        self.ensure(
            max_bs=bs,
            max_tokens=max(bs, self.max_tokens),
            max_pages=max(self.max_pages, 1),
            device=query_start_loc.device,
            ngram_context_len=ngram_w,
        )
        assert self.query_start_loc is not None
        assert self.ngram_context is not None
        assert self.state_indices_in is not None
        assert self.state_indices_out is not None
        assert self.has_initial_state is not None

        self.query_start_loc.zero_()
        self.state_indices_in.zero_()
        self.state_indices_out.zero_()
        self.has_initial_state.fill_(True)

        qsl = query_start_loc.to(dtype=torch.int32).reshape(-1)
        copy_q = min(int(qsl.numel()), bs + 1)
        self.query_start_loc[:copy_q].copy_(qsl[:copy_q])
        self.ngram_context[:bs, :ngram_w].copy_(
            ngram_context[:bs, :ngram_w].to(dtype=torch.int64)
        )
        self.state_indices_in[:bs].copy_(
            state_indices_in[:bs].to(dtype=torch.int32)
        )
        self.state_indices_out[:bs].copy_(
            state_indices_out[:bs].to(dtype=torch.int32)
        )
        self.has_initial_state[:bs].copy_(
            has_initial_state[:bs].to(dtype=torch.bool)
        )
        self.last_ple = Qwen3_8FlashNextPLEMetadata(
            query_start_loc=self.query_start_loc[: bs + 1],
            ngram_context=self.ngram_context[:bs, : self.ngram_context_len],
            ngram_state=ngram_state,
            state_indices_in=self.state_indices_in[:bs],
            state_indices_out=self.state_indices_out[:bs],
            has_initial_state=self.has_initial_state[:bs],
            conv_state=conv_state,
            num_reqs=bs,
            is_prefill=is_prefill,
            max_query_len=max(int(max_query_len), 1),
        )
        return self.last_ple


_DECODE_GRAPH = _FlashDecodeGraphBuffers()


def _is_flash_next_config(atom_config: Any) -> bool:
    hf = _hf_text_config(atom_config)
    model_type = str(getattr(hf, "model_type", "") or "")
    if model_type.startswith("qwen4_exp"):
        return True
    arch = getattr(atom_config, "architectures", None) or getattr(
        getattr(atom_config, "hf_config", None), "architectures", None
    )
    if not arch:
        return False
    return any("Qwen4Exp" in str(a) or "FlashNext" in str(a) for a in arch)


def _use_decode_graph_buffers(forward_batch: Any) -> bool:
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return False
    if not mode.is_decode_or_idle():
        return False
    return _DECODE_GRAPH.active or _is_capturing()

def _is_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _host_int(tensor: torch.Tensor, fallback: int) -> int:
    """Read a 0-d / 1-element GPU tensor on the host unless a graph is capturing."""
    if tensor.numel() == 0:
        return fallback
    if _is_capturing():
        return fallback
    return int(tensor.reshape(-1)[0].item())


@dataclass
class Qwen3_8FlashNextQSAMetadata:
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    compressed_slot_mapping: torch.Tensor
    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int


@dataclass
class Qwen3_8FlashNextPLEMetadata:
    query_start_loc: torch.Tensor
    ngram_context: torch.Tensor
    ngram_state: torch.Tensor
    state_indices_in: torch.Tensor
    state_indices_out: torch.Tensor
    has_initial_state: torch.Tensor
    conv_state: torch.Tensor
    num_reqs: int
    is_prefill: bool
    max_query_len: int


def _hf_text_config(atom_config: Any) -> Any:
    hf = getattr(atom_config, "hf_config", atom_config)
    return getattr(hf, "text_config", None) or hf


def _block_size(forward_batch: Any, atom_config: Any) -> int:
    """Resolve SGLang page size for QSA block tables.

    Must match the KV pool page size (server ``--page-size``). Preferring a
    smaller HF/atom ``block_size`` (e.g. 16) yields OOB page ids on long decode
    and HSA faults under CUDA-graph replay.
    """
    candidates: list[Any] = []
    try:
        from sglang.srt.server_args import get_global_server_args

        candidates.append(getattr(get_global_server_args(), "page_size", None))
    except Exception:  # noqa: BLE001
        pass
    pool = getattr(forward_batch, "token_to_kv_pool", None) or getattr(
        forward_batch, "token_to_kv_pool_allocator", None
    )
    req_pool = getattr(forward_batch, "req_to_token_pool", None)
    candidates.extend(
        (
            getattr(forward_batch, "page_size", None),
            getattr(pool, "page_size", None),
            getattr(req_pool, "page_size", None),
            getattr(atom_config, "page_size", None),
            getattr(atom_config, "kv_cache_block_size", None),
        )
    )
    for candidate in candidates:
        if candidate:
            return int(candidate)
    return 64


def _compress_ratio(atom_config: Any) -> int:
    return int(getattr(_hf_text_config(atom_config), "indexer_compress_ratio", 4))


def _req_to_token_pool(forward_batch: Any) -> Any:
    backend = resolve_attn_backend(forward_batch)
    linear = getattr(backend, "full_attn_backend", None) or getattr(
        backend, "attn_backend", None
    )
    return (
        getattr(forward_batch, "req_to_token_pool", None)
        or getattr(backend, "req_to_token_pool", None)
        or getattr(linear, "req_to_token_pool", None)
        or resolve_mamba_req_pool(forward_batch, backend)
    )


def _seq_lens(forward_batch: Any, device: torch.device) -> torch.Tensor:
    seq = getattr(forward_batch, "seq_lens", None)
    bs = int(getattr(forward_batch, "batch_size", 0) or 0)
    if torch.is_tensor(seq):
        seq = seq.to(device=device, dtype=torch.int32)[:bs]
    else:
        seq = torch.ones((bs,), dtype=torch.int32, device=device)
    live_bs = real_batch_size(forward_batch)
    if live_bs < seq.shape[0]:
        # CUDA-graph pad rows keep seq_len_fill_value (usually 1) and a
        # finished request's page table. Zero them so QSA does not score
        # or write freed pages.
        seq = seq.clone()
        seq[live_bs:] = 0
    return seq


def _query_start_loc(forward_batch: Any, num_tokens: int, device: torch.device) -> torch.Tensor:
    mode = forward_batch.forward_mode
    bs = int(forward_batch.batch_size)
    live_bs = real_batch_size(forward_batch)
    if mode.is_decode_or_idle():
        loc = torch.arange(0, bs + 1, dtype=torch.int32, device=device)
        loc[live_bs + 1 :] = live_bs
        return loc
    if mode.is_extend():
        loc = torch.empty((bs + 1,), dtype=torch.int32, device=device)
        if live_bs:
            loc[:live_bs] = forward_batch.extend_start_loc[:live_bs].to(
                dtype=torch.int32
            )
            loc[live_bs:] = (
                forward_batch.extend_start_loc[live_bs - 1]
                + forward_batch.extend_seq_lens[live_bs - 1]
            ).to(dtype=torch.int32)
        else:
            loc.fill_(0)
        return loc
    return torch.tensor([0, num_tokens], dtype=torch.int32, device=device)


def _token_to_req_and_logical(
    *,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = positions.device
    token_to_req = torch.zeros((num_tokens,), dtype=torch.int32, device=device)
    logical = torch.full((num_tokens,), _NO_WRITE, dtype=torch.int64, device=device)
    if query_start_loc.numel() < 2 or num_tokens <= 0:
        return token_to_req, logical
    token_ids = torch.arange(num_tokens, device=device)
    ends = query_start_loc[1:]
    req = torch.searchsorted(ends, token_ids, right=True)
    valid = token_ids < query_start_loc[-1]
    token_to_req = torch.where(valid, req.to(torch.int32), token_to_req)
    pos = positions.reshape(-1)[:num_tokens].to(torch.int64)
    logical = torch.where(valid, pos, logical)
    return token_to_req, logical


def _compressed_slot_mapping(
    *,
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    block_size: int,
    compress_ratio: int,
) -> torch.Tensor:
    storage_block = max(block_size // compress_ratio, 1)
    logical = logical_positions
    compressed_pos = torch.div(
        logical.clamp_min(0), compress_ratio, rounding_mode="floor"
    )
    logical_block = torch.div(
        compressed_pos, storage_block, rounding_mode="floor"
    ).clamp_(0, max(block_tables.shape[1] - 1, 0))
    requests = token_to_req.long().clamp_(0, max(block_tables.shape[0] - 1, 0))
    physical = block_tables[requests, logical_block].long()
    compressed_slots = physical * storage_block + compressed_pos.remainder(storage_block)
    closes_group = (logical >= 0) & ((logical + 1).remainder(compress_ratio) == 0)
    return torch.where(
        closes_group & (slot_mapping >= 0) & (physical >= 0),
        compressed_slots,
        torch.full_like(compressed_slots, _NO_WRITE),
    )


def build_qsa_metadata(
    atom_config: Any,
    forward_batch: Any,
    positions: torch.Tensor,
) -> Qwen3_8FlashNextQSAMetadata | None:
    pool = _req_to_token_pool(forward_batch)
    if pool is None or not hasattr(pool, "req_to_token"):
        logger.debug("Flash QSA bridge: no req_to_token pool; skip QSA metadata")
        return None

    device = positions.device
    bs = int(forward_batch.batch_size)
    num_tokens = int(positions.reshape(-1).numel())
    block_size = _block_size(forward_batch, atom_config)
    compress_ratio = _compress_ratio(atom_config)
    if block_size % compress_ratio:
        raise ValueError(
            f"page-size / block-size ({block_size}) must be divisible by "
            f"indexer_compress_ratio ({compress_ratio})"
        )

    live_bs = real_batch_size(forward_batch)
    seq_lens = _seq_lens(forward_batch, device)[:bs]
    hf = _hf_text_config(atom_config)
    indexer_budget = int(getattr(hf, "indexer_budget", 2048) or 2048)
    ctx_len = indexer_budget
    try:
        from sglang.srt.server_args import get_global_server_args

        args = get_global_server_args()
        ctx_len = max(
            ctx_len,
            int(getattr(args, "context_length", 0) or 0),
            int(getattr(args, "max_model_len", 0) or 0),
        )
    except Exception:  # noqa: BLE001
        pass
    # Host max_seq_len is a Triton constexpr. Pin to serving context so a 12k
    # replay does not exceed the captured scoring / page-table width.
    pin_for_graph = _is_capturing() or _DECODE_GRAPH.active
    max_seq_len = (
        ctx_len
        if pin_for_graph or seq_lens.numel() == 0
        else int(seq_lens.max().item())
    )
    req_pool_indices = forward_batch.req_pool_indices[:bs]
    if live_bs < bs:
        # Pad rows keep the just-finished request's pool index after 4→3
        # (or any bucket pad). Do not gather that row's pages.
        req_pool_indices = req_pool_indices.clone()
        req_pool_indices[live_bs:] = 0
    # Native QSA requires page_table width * compressed_rows >= indexer_budget
    # groups even on short warmup sequences. SGLang's current-seq table is too
    # narrow (one 64-token page => 16 compressed slots vs 512 top-k).
    table_tokens = max(max_seq_len, indexer_budget)
    req_to_token = getattr(pool, "req_to_token", None)
    if torch.is_tensor(req_to_token) and req_to_token.dim() >= 2:
        table_tokens = min(table_tokens, int(req_to_token.shape[1]))
    block_tables = _build_block_tables(
        pool, req_pool_indices, table_tokens, block_size
    ).to(device=device)

    slot_mapping = getattr(forward_batch, "out_cache_loc", None)
    if not torch.is_tensor(slot_mapping):
        slot_mapping = torch.full(
            (num_tokens,), _NO_WRITE, dtype=torch.int64, device=device
        )
    else:
        slot_mapping = slot_mapping.reshape(-1)[:num_tokens].to(
            device=device, dtype=torch.int64
        )
        if slot_mapping.numel() < num_tokens:
            pad = torch.full(
                (num_tokens - slot_mapping.numel(),),
                _NO_WRITE,
                dtype=torch.int64,
                device=device,
            )
            slot_mapping = torch.cat([slot_mapping, pad], dim=0)
        slot_mapping = torch.where(
            slot_mapping < 0, torch.full_like(slot_mapping, _NO_WRITE), slot_mapping
        )

    query_start_loc = _query_start_loc(forward_batch, num_tokens, device)
    token_to_req, logical = _token_to_req_and_logical(
        query_start_loc=query_start_loc,
        positions=positions,
        num_tokens=num_tokens,
    )
    if query_start_loc.numel():
        token_ids = torch.arange(num_tokens, device=device)
        slot_mapping = torch.where(
            token_ids < query_start_loc[-1],
            slot_mapping,
            torch.full_like(slot_mapping, _NO_WRITE),
        )
    if live_bs < bs:
        # Decode is one token per row. Drop pad-token cache writes even if
        # leftover out_cache_loc still holds a freed page id.
        pad_tokens = min(num_tokens, bs)
        if pad_tokens > live_bs:
            slot_mapping = slot_mapping.clone()
            slot_mapping[live_bs:pad_tokens] = _NO_WRITE
            logical = logical.clone()
            logical[live_bs:pad_tokens] = _NO_WRITE
            block_tables = block_tables.clone()
            block_tables[live_bs:].zero_()

    compressed = _compressed_slot_mapping(
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        token_to_req=token_to_req,
        logical_positions=logical,
        block_size=block_size,
        compress_ratio=compress_ratio,
    )
    max_seq_len = max(max_seq_len, 1)
    # Decode CUDA graphs bake Triton constexprs from this host integer. Pin to
    # the serving context (not just indexer_budget) so 12k replay fits.
    if _use_decode_graph_buffers(forward_batch) or _is_capturing():
        max_seq_len = max(max_seq_len, indexer_budget, ctx_len)
        return _DECODE_GRAPH.materialize_qsa(
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            compressed_slot_mapping=compressed,
            token_to_req=token_to_req,
            logical_positions=logical,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
        )
    return Qwen3_8FlashNextQSAMetadata(
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        compressed_slot_mapping=compressed,
        token_to_req=token_to_req,
        logical_positions=logical,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
    )


def _ple_state_pool_slots(forward_batch: Any, idx: torch.Tensor | None) -> int:
    """Native allocates PLE conv state for the whole per-req pool, not max_bs.

    Decode CUDA graphs bake ``conv_state``'s address. Growing after capture
    frees that storage; undersizing it makes a recycled mamba slot OOB on the
    next eager prefill (the conc=4 wave-2 HSA).
    """
    slots = max(int(_DECODE_GRAPH.max_bs), 16)
    backend = resolve_attn_backend(forward_batch)
    pool = resolve_mamba_req_pool(forward_batch, backend)
    if pool is not None:
        mapping = getattr(pool, "req_index_to_mamba_index_mapping", None)
        if torch.is_tensor(mapping) and mapping.numel():
            slots = max(slots, int(mapping.numel()))
        for key in ("size", "max_num_reqs", "mamba_size"):
            val = getattr(pool, key, None)
            if val:
                slots = max(slots, int(val))
    req_pool = _req_to_token_pool(forward_batch)
    if req_pool is not None:
        for key in ("size", "max_num_reqs"):
            val = getattr(req_pool, key, None)
            if val:
                slots = max(slots, int(val))
    if torch.is_tensor(idx) and idx.numel() and not _is_capturing():
        slots = max(slots, int(idx.clamp(min=0).max().item()) + 1)
    return slots


def _ensure_ple_conv_state(model: Any, atom_config: Any, num_slots: int) -> torch.Tensor:
    hf = _hf_text_config(atom_config)
    state_len = (int(getattr(hf, "ple_conv_kernel_size", 4)) - 1) * int(
        getattr(hf, "ngram_size", 3)
    )
    channels = int(hf.hidden_size) * int(getattr(hf, "hc_count", 4))
    existing = getattr(model, "_atom_ple_conv_state", None)
    if (
        existing is not None
        and existing.shape[0] >= num_slots
        and existing.shape[1] == channels
        and existing.shape[2] == state_len
    ):
        return existing
    # Never replace a live graph-captured conv_state. Clamp callers instead.
    if existing is not None and _DECODE_GRAPH.active:
        logger.warning(
            "PLE conv_state already captured at %s slots; refusing grow to %s",
            int(existing.shape[0]),
            num_slots,
        )
        return existing
    device = next(model.parameters()).device
    dtype = getattr(atom_config, "torch_dtype", None) or torch.bfloat16
    state = torch.zeros(
        (max(num_slots, 1), channels, state_len), dtype=dtype, device=device
    )
    model._atom_ple_conv_state = state
    return state



def _ensure_ple_ngram_state(model: Any, atom_config: Any, num_slots: int) -> torch.Tensor:
    hf = _hf_text_config(atom_config)
    width = max(int(getattr(hf, "ngram_size", 3)) - 1, 1)
    eos = getattr(hf, "eos_token_id", 0)
    eos_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)
    existing = getattr(model, "_atom_ple_ngram_state", None)
    if (
        existing is not None
        and existing.shape[0] >= num_slots
        and existing.shape[1] == width
    ):
        return existing
    if existing is not None and _DECODE_GRAPH.active:
        logger.warning(
            "PLE ngram_state already captured at %s slots; refusing grow to %s",
            int(existing.shape[0]),
            num_slots,
        )
        return existing
    device = next(model.parameters()).device
    state = torch.full(
        (max(num_slots, 1), width), eos_id, dtype=torch.int64, device=device
    )
    model._atom_ple_ngram_state = state
    return state

def build_ple_metadata(
    atom_config: Any,
    forward_batch: Any,
    positions: torch.Tensor,
    *,
    model: Any,
    input_ids: torch.Tensor | None,
    gdn_metadata: Any,
) -> Qwen3_8FlashNextPLEMetadata | None:
    hf = _hf_text_config(atom_config)
    if not getattr(hf, "ple_layer_ids", None):
        return None
    if gdn_metadata is None:
        return None
    idx = getattr(gdn_metadata, "non_spec_state_indices_tensor", None)
    if idx is None:
        return None

    ngram_context_len = max(int(getattr(hf, "ngram_size", 3)) - 1, 0)
    device = positions.device
    bs = int(forward_batch.batch_size)
    num_tokens = int(positions.reshape(-1).numel())
    query_start_loc = _query_start_loc(forward_batch, num_tokens, device)
    is_prefill = bool(forward_batch.forward_mode.is_extend())
    eos = getattr(hf, "eos_token_id", 0)
    eos_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)

    context = torch.full(
        (bs, max(ngram_context_len, 1)), eos_id, dtype=torch.int64, device=device
    )
    if torch.is_tensor(input_ids) and ngram_context_len and is_prefill:
        ids = input_ids.reshape(-1)
        starts = query_start_loc[:-1]
        for req in range(bs):
            lo = int(starts[req].item())
            for offset in range(ngram_context_len):
                src = lo - ngram_context_len + offset
                if 0 <= src < ids.numel():
                    context[req, offset] = ids[src].to(torch.int64)

    live_bs = real_batch_size(forward_batch)
    idx = idx[:bs].to(device=device, dtype=torch.int32)
    idx_in = getattr(gdn_metadata, "non_spec_state_indices_in_tensor", None)
    if idx_in is None:
        idx_in = idx
    else:
        idx_in = idx_in[:bs].to(device=device, dtype=torch.int32)
    if live_bs < bs:
        idx = idx.clone()
        idx[live_bs:] = -1
        if idx_in.data_ptr() == idx.data_ptr():
            idx_in = idx
        else:
            idx_in = idx_in.clone()
            idx_in[live_bs:] = -1

    num_slots = _ple_state_pool_slots(forward_batch, idx)
    conv_state = _ensure_ple_conv_state(model, atom_config, num_slots)
    ngram_state = _ensure_ple_ngram_state(model, atom_config, num_slots)
    last_slot = max(int(conv_state.shape[0]) - 1, 0)
    if last_slot >= 0:
        idx = torch.where(idx < 0, idx, idx.clamp(max=last_slot))
        idx_in = torch.where(idx_in < 0, idx_in, idx_in.clamp(max=last_slot))
    prefix = getattr(forward_batch, "extend_prefix_lens", None)
    if is_prefill and torch.is_tensor(prefix):
        has_initial = prefix[:bs] > 0
    else:
        has_initial = torch.ones((bs,), dtype=torch.bool, device=device)
    if live_bs < bs:
        has_initial = has_initial.clone()
        has_initial[live_bs:] = False

    max_query_len = (
        1
        if (not is_prefill or _is_capturing() or query_start_loc.numel() <= 1)
        else int((query_start_loc[1:] - query_start_loc[:-1]).max().item())
    )
    if _use_decode_graph_buffers(forward_batch) or _is_capturing():
        _DECODE_GRAPH.model = model
        _DECODE_GRAPH.atom_config = atom_config
        return _DECODE_GRAPH.materialize_ple(
            query_start_loc=query_start_loc,
            ngram_context=context,
            ngram_state=ngram_state,
            state_indices_in=idx_in,
            state_indices_out=idx,
            has_initial_state=has_initial,
            conv_state=conv_state,
            num_reqs=bs,
            is_prefill=is_prefill,
            max_query_len=max_query_len,
        )
    return Qwen3_8FlashNextPLEMetadata(
        query_start_loc=query_start_loc,
        ngram_context=context,
        ngram_state=ngram_state,
        state_indices_in=idx_in,
        state_indices_out=idx,
        has_initial_state=has_initial,
        conv_state=conv_state,
        num_reqs=bs,
        is_prefill=is_prefill,
        max_query_len=max_query_len,
    )


def attach_flash_metadata(attn_md: Any, qsa: Any, ple: Any) -> Any:
    """Stamp Native Flash fields onto the live AttentionMetaData object."""
    attn_md.qsa_metadata = qsa
    attn_md.ple_metadata = ple
    # Temporary smoke diagnostic — remove after SGLang FP8 greedy passes.
    # Log every attach until PLE is non-None once (GDN wiring debug).
    if ple is None or not getattr(attach_flash_metadata, "_logged_ok", False):
        logger.warning(
            "Flash metadata attach: qsa=%s ple=%s "
            "(qsa None => QSA zeros; ple None => PLE skipped)",
            None if qsa is None else type(qsa).__name__,
            None if ple is None else type(ple).__name__,
        )
        if ple is not None:
            attach_flash_metadata._logged_ok = True  # type: ignore[attr-defined]
    return attn_md


def bind_qsa_caches(model: Any, forward_batch: Any, atom_config: Any) -> None:
    """Bind QSA K/V + indexer caches.

    Native #2048 uses a compact paged pool ``[blocks, block_size, heads, dim]``
    with no AITER shuffle. SGLang's leftover token pool is far larger (~2.6M
    tokens here); duplicating it as dedicated Native tensors OOMs MI308.
    Main K/V therefore views the SGLang buffer when it is already
    ``[tokens, heads, dim]`` with ``tokens % block_size == 0`` (same bytes as
    Native ``[pages, block, heads, dim]``). Indexer raw/compressed stay
    plugin-owned, as in #2048.
    """
    if getattr(model, "_atom_flash_qsa_bound", False):
        return
    qsa_layers = [
        mod
        for mod in model.modules()
        if getattr(mod, "is_qsa_attention", False)
    ]
    if not qsa_layers:
        return

    pool = getattr(forward_batch, "token_to_kv_pool", None)
    if pool is None:
        backend = resolve_attn_backend(forward_batch)
        pool = getattr(backend, "token_to_kv_pool", None) or getattr(
            getattr(backend, "full_attn_backend", None), "token_to_kv_pool", None
        )
    block_size = _block_size(forward_batch, atom_config)
    compress_ratio = _compress_ratio(atom_config)
    hf = _hf_text_config(atom_config)
    index_head_dim = int(getattr(hf, "indexer_head_dim", 128))
    kv_heads = max(int(getattr(qsa_layers[0], "num_kv_heads", 2)), 1)
    head_dim = int(getattr(qsa_layers[0], "head_dim", 256))

    num_pages = None
    for attr in ("num_pages", "page_num"):
        val = getattr(pool, attr, None)
        if val:
            num_pages = int(val)
            break
    if num_pages is None:
        num_tokens = getattr(pool, "size", None)
        if num_tokens:
            num_pages = max((int(num_tokens) + block_size - 1) // block_size, 1)
    if num_pages is None:
        k0 = getattr(qsa_layers[0], "k_cache", None)
        if torch.is_tensor(k0) and k0.dim() >= 2:
            num_pages = int(k0.shape[0])
    if not num_pages:
        logger.warning("Flash QSA bridge: cannot size indexer caches yet")
        return

    device = next(model.parameters()).device
    raw = torch.zeros(
        (len(qsa_layers), num_pages, block_size, 1, index_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    compressed = torch.zeros(
        (
            len(qsa_layers),
            num_pages,
            max(block_size // compress_ratio, 1),
            1,
            index_head_dim,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    used_pool = 0
    pool_shape = None
    for i, layer in enumerate(qsa_layers):
        k_cache = None
        v_cache = None
        getter = getattr(pool, "get_kv_buffer", None) if pool is not None else None
        if callable(getter):
            try:
                k_buf, v_buf = getter(int(getattr(layer, "layer_num", i)))
            except Exception:  # noqa: BLE001
                k_buf = v_buf = None
            if torch.is_tensor(k_buf) and torch.is_tensor(v_buf) and k_buf.dim() == 3:
                tokens, heads, dim = k_buf.shape
                pages = tokens // block_size
                if pages > 0 and tokens == pages * block_size:
                    k_cache = k_buf.view(pages, block_size, heads, dim)
                    v_cache = v_buf.view(pages, block_size, heads, dim)
                    used_pool += 1
                    pool_shape = tuple(k_buf.shape)
        if k_cache is None:
            k_cache = torch.zeros(
                (num_pages, block_size, kv_heads, head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            v_cache = torch.zeros_like(k_cache)
        layer.bind_caches(k_cache, v_cache, raw[i], compressed[i], None)
    model._atom_flash_qsa_raw = raw
    model._atom_flash_qsa_compressed = compressed
    model._atom_flash_qsa_bound = True
    logger.info(
        "Bound %s QSA layers pages=%s block=%s compress=%s "
        "sglang_pool_view=%s/%s pool_shape=%s (Native layout = paged view of [T,H,D])",
        len(qsa_layers),
        num_pages,
        block_size,
        compress_ratio,
        used_pool,
        len(qsa_layers),
        pool_shape,
    )


def _decode_graph_capacity(atom_config: Any, forward_batch: Any) -> tuple[int, int, torch.device]:
    """High-water sizes so capture never reallocates persistent QSA buffers."""
    device = getattr(forward_batch, "device", None)
    if device is None:
        pos = getattr(forward_batch, "positions", None)
        device = pos.device if torch.is_tensor(pos) else torch.device("cuda")
    bs = int(getattr(forward_batch, "batch_size", 0) or 0)
    # Prefer SGLang's configured decode capture ceiling when present.
    graph_bs = 0
    for key in ("cuda_graph_max_bs_decode", "cuda_graph_max_bs"):
        val = getattr(forward_batch, key, None)
        if val is None:
            continue
        try:
            graph_bs = max(graph_bs, int(val))
        except (TypeError, ValueError):
            pass
    if graph_bs <= 0:
        try:
            from sglang.srt.server_args import get_global_server_args

            args = get_global_server_args()
            graph_bs = int(
                getattr(args, "cuda_graph_max_bs_decode", None)
                or getattr(args, "cuda_graph_max_bs", 0)
                or 0
            )
        except Exception:  # noqa: BLE001
            graph_bs = 0
    max_bs = max(bs, graph_bs, 8)
    hf = _hf_text_config(atom_config)
    indexer_budget = int(getattr(hf, "indexer_budget", 2048) or 2048)
    block_size = _block_size(forward_batch, atom_config)
    # Page table must cover the full context, not just indexer_budget.
    # QSA top-k still keeps only ``indexer_budget`` tokens, but those tokens
    # can sit anywhere in a 12k sequence; a 32-page (2k) table would clamp
    # logical_page and drop the recent context.
    ctx_len = indexer_budget
    try:
        from sglang.srt.server_args import get_global_server_args

        args = get_global_server_args()
        ctx_len = max(
            ctx_len,
            int(getattr(args, "context_length", 0) or 0),
            int(getattr(args, "max_model_len", 0) or 0),
        )
    except Exception:  # noqa: BLE001
        pass
    max_pages = max((ctx_len + block_size - 1) // block_size, 1)
    return max_bs, max_pages, torch.device(device)


def _reserve_qsa_graph_workspace(
    hf: Any, *, max_bs: int, device: torch.device
) -> None:
    """Pin grow-once QSA scratch so CUDA-graph decode never owns ephemeral slabs.

    Native ``Qwen3_8FlashNextIndexer.token_topk`` is ``indexer_budget`` (2048),
    so ``block_topk = budget / compress_ratio`` (512). Undersizing this buffer
    truncates the top-k write under graph replay and HSA-faults after a long
    prefill reclaims the capture-time ``torch.empty`` logits slab.
    """
    from atom.model_ops.qwen4_exp import qsa_graph_workspace as _gws

    indexer_budget = int(getattr(hf, "indexer_budget", 2048) or 2048)
    compress_ratio = int(
        getattr(hf, "indexer_compress_ratio", None)
        or getattr(hf, "compress_ratio", None)
        or 4
    )
    # Match Native indexer: token_topk == indexer_budget (not a separate topk).
    token_topk = int(
        getattr(hf, "indexer_topk", None)
        or getattr(hf, "token_topk", None)
        or indexer_budget
    )
    # Score the full context (compressed groups), not only indexer_budget.
    # 12k tokens / compress 4 → 3k columns, rounded up to a power of two.
    ctx_len = indexer_budget
    try:
        from sglang.srt.server_args import get_global_server_args

        args = get_global_server_args()
        ctx_len = max(
            ctx_len,
            int(getattr(args, "context_length", 0) or 0),
            int(getattr(args, "max_model_len", 0) or 0),
        )
    except Exception:  # noqa: BLE001
        pass
    max_columns = max(int(ctx_len) // max(compress_ratio, 1), 1)
    max_columns = 1 << (max_columns - 1).bit_length()
    block_topk = max(token_topk // max(compress_ratio, 1), 1)
    # Workspace holds both indexer pooled keys and attention outputs.
    head_dim = max(
        int(getattr(hf, "head_dim", 256) or 256),
        int(getattr(hf, "indexer_head_dim", 128) or 128),
    )
    num_q_heads = int(getattr(hf, "num_attention_heads", 32) or 32)
    tp = 1
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        tp = max(int(get_tensor_model_parallel_world_size()), 1)
        num_q_heads = max(num_q_heads // tp, 1)
    except Exception:  # noqa: BLE001
        pass
    _gws.reserve(
        max_tokens=max_bs,
        max_columns=max_columns,
        max_block_topk=block_topk,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        device=device,
    )
    # GDN / hyper-connection / grouped RMSNorm still torch.empty inside the
    # captured decode forward unless this second workspace is reserved before
    # capture. Without it, greedy graph replay + long eager prefill reclaims
    # those slabs and the first long-decode replay HSA-faults.
    from atom.model_ops.qwen4_exp import flash_decode_graph_workspace as _fws

    hidden = int(getattr(hf, "hidden_size", 2048) or 2048)
    hc_count = int(getattr(hf, "hc_count", 4) or 4)
    linear_v_heads = max(
        int(getattr(hf, "linear_num_value_heads", 32) or 32) // tp, 1
    )
    head_v_dim = int(getattr(hf, "linear_value_head_dim", 128) or 128)
    expand_width = token_topk + max(compress_ratio, 1) - 1
    # QKVG last-dim split views need pinned contiguous staging under graph
    # decode (q_dim / kv_dim after TP). RoPE scratch covers attn + indexer.
    attn_heads = max(int(getattr(hf, "num_attention_heads", 32) or 32) // tp, 1)
    kv_heads = int(getattr(hf, "num_key_value_heads", attn_heads) or attn_heads)
    if kv_heads >= tp:
        kv_heads = max(kv_heads // tp, 1)
    else:
        kv_heads = 1
    attn_head_dim = int(getattr(hf, "head_dim", 256) or 256)
    q_dim = attn_heads * attn_head_dim
    kv_dim = kv_heads * attn_head_dim
    indexer_head_dim = int(getattr(hf, "indexer_head_dim", 128) or 128)
    indexer_n_heads = int(getattr(hf, "indexer_n_heads", 4) or 4)
    rope_dim = max(attn_head_dim, indexer_head_dim)
    k_heads = max(int(getattr(hf, "linear_num_key_heads", 16) or 16) // tp, 1)
    head_k_dim = int(getattr(hf, "linear_key_head_dim", 128) or 128)
    head_norm_rows = max_bs * max(attn_heads, indexer_n_heads, 1)
    _fws.reserve(
        max_tokens=max_bs,
        hidden=hidden,
        hc_count=hc_count,
        v_heads=linear_v_heads,
        head_v_dim=head_v_dim,
        expand_width=expand_width,
        device=device,
        dtype=torch.bfloat16,
        q_dim=q_dim,
        kv_dim=kv_dim,
        rope_dim=rope_dim,
        head_norm_rows=head_norm_rows,
        head_norm_dim=rope_dim,
        k_heads=k_heads,
        head_k_dim=head_k_dim,
        ple_state_len=(int(getattr(hf, "ple_conv_kernel_size", 4)) - 1)
        * int(getattr(hf, "ngram_size", 3)),
        moe_topk=int(getattr(hf, "num_experts_per_tok", None)
                     or getattr(hf, "moe_topk", 10) or 10),
        moe_intermediate=int(getattr(hf, "moe_intermediate_size", 640) or 640),
    )
    if not getattr(_reserve_qsa_graph_workspace, "_logged", False):
        logger.info(
            "QSA graph workspace reserved: tokens<=%s cols<=%s block_topk<=%s "
            "head_dim=%s q_heads=%s enabled=%s",
            max_bs,
            max_columns,
            block_topk,
            head_dim,
            num_q_heads,
            _gws.is_enabled(),
        )
        logger.info(
            "Flash decode graph workspace reserved: tokens<=%s hidden=%s hc=%s "
            "v_heads=%s head_v_dim=%s expand_w=%s q_dim=%s kv_dim=%s rope_dim=%s "
            "head_norm_rows=%s k_heads=%s ple_state_len=%s enabled=%s",
            max_bs,
            hidden,
            hc_count,
            linear_v_heads,
            head_v_dim,
            expand_width,
            q_dim,
            kv_dim,
            rope_dim,
            head_norm_rows,
            k_heads,
            (int(getattr(hf, "ple_conv_kernel_size", 4)) - 1)
            * int(getattr(hf, "ngram_size", 3)),
            _fws.is_enabled(),
        )
        _reserve_qsa_graph_workspace._logged = True  # type: ignore[attr-defined]


def prepare_flash_decode_graph_metadata(
    forward_batch: Any,
    in_capture: bool = False,
    *,
    atom_config: Any | None = None,
    **kwargs: Any,
) -> Qwen3_8FlashNextQSAMetadata | None:
    """Fill persistent QSA buffers outside the CUDA graph (capture + replay).

    Called from ``ATOMAttnBackendForSgl.init_forward_metadata_out_graph``. On
    replay this is the only chance to refresh page tables before
    ``graph.replay()``.
    """
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not mode.is_decode_or_idle():
        return None
    if atom_config is None:
        try:
            from atom.config import get_current_atom_config

            atom_config = get_current_atom_config()
        except Exception:  # noqa: BLE001
            return None
    if atom_config is None or not _is_flash_next_config(atom_config):
        return None

    positions = getattr(forward_batch, "positions", None)
    if not torch.is_tensor(positions):
        bs = int(getattr(forward_batch, "batch_size", 0) or 0)
        device = getattr(forward_batch, "device", None) or torch.device("cuda")
        positions = torch.zeros((max(bs, 1),), dtype=torch.int64, device=device)

    # Force graph-buffer path for this decode step (capture or replay).
    _DECODE_GRAPH.active = True
    max_bs, max_pages, device = _decode_graph_capacity(atom_config, forward_batch)
    block_size = _block_size(forward_batch, atom_config)
    hf = _hf_text_config(atom_config)
    ngram_context_len = max(int(getattr(hf, "ngram_size", 3)) - 1, 1)
    _DECODE_GRAPH.preallocate_for_decode_graph(
        max_bs=max_bs,
        max_pages=max_pages,
        device=device,
        ngram_context_len=ngram_context_len,
    )
    # Reserve QSA decode workspaces before capture so Triton pointer operands
    # never alias caching-allocator slabs reused by long eager prefills.
    # Native indexer.token_topk == indexer_budget (not a separate topk field).
    try:
        _reserve_qsa_graph_workspace(hf, max_bs=max_bs, device=device)
    except Exception as exc:  # noqa: BLE001
        logger.warning("QSA graph workspace reserve failed: %s", exc)

    qsa = build_qsa_metadata(atom_config, forward_batch, positions)
    # PLE must refresh *after* HybridLinearAttnBackend runs the GDN/linear
    # child (full-attn out_graph runs first). See refresh_flash_decode_graph_ple
    # installed from register.py. Capture-time PLE still comes from model.forward.
    # Replay must not .item() / INFO-log: each D2H sync is a full device drain
    # and Native ATOM decode (TPOT ~26ms) has none of this.
    debug_qsa = in_capture or logger.isEnabledFor(logging.DEBUG)
    if qsa is not None and debug_qsa:
        try:
            bt = qsa.block_tables
            nonzero_pages = int((bt > 0).sum().item()) if torch.is_tensor(bt) else -1
            seq_max = (
                int(qsa.seq_lens.max().item())
                if torch.is_tensor(qsa.seq_lens) and qsa.seq_lens.numel()
                else -1
            )
            slot0 = (
                int(qsa.slot_mapping.reshape(-1)[0].item())
                if torch.is_tensor(qsa.slot_mapping) and qsa.slot_mapping.numel()
                else -1
            )
        except Exception:  # noqa: BLE001
            nonzero_pages, seq_max, slot0 = -1, -1, -1
        logger.info(
            "Flash decode graph QSA %s: seq_max=%s pages_nonzero=%s "
            "page_width=%s slot0=%s max_seq_len=%s",
            "capture" if in_capture else "replay",
            seq_max,
            nonzero_pages,
            int(qsa.block_tables.shape[1]) if torch.is_tensor(qsa.block_tables) else -1,
            slot0,
            qsa.max_seq_len,
        )
    if in_capture:
        logger.info(
            "Flash decode CUDA-graph QSA buffers ready: "
            "block_size=%s bs<=%s tokens<=%s pages<=%s max_seq_len=%s",
            block_size,
            _DECODE_GRAPH.max_bs,
            _DECODE_GRAPH.max_tokens,
            _DECODE_GRAPH.max_pages,
            _DECODE_GRAPH.max_seq_len,
        )
        if block_size != 64 and max_pages > 64:
            logger.warning(
                "Flash QSA block_size=%s yields pages<=%s; "
                "expected page-size 64 → pages<=32 for indexer_budget=2048. "
                "Wrong block_size causes OOB page ids on long decode.",
                block_size,
                max_pages,
            )
    return qsa


def refresh_flash_decode_graph_ple(
    forward_batch: Any,
    in_capture: bool = False,
) -> bool:
    """Copy fresh GDN ``mamba_cache_indices`` into persistent PLE graph buffers.

    Must run after ``linear_attn_backend.init_forward_metadata_out_graph`` so
    indices already hold this step's values. Replay never re-enters
    ``model.forward`` Python, so this is the only chance to refresh PLE.
    """
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not mode.is_decode_or_idle():
        return False
    if not _DECODE_GRAPH.active and not _is_capturing():
        return False
    model = getattr(_DECODE_GRAPH, "model", None)
    atom_config = getattr(_DECODE_GRAPH, "atom_config", None)
    if model is None:
        return False
    if atom_config is None:
        try:
            from atom.config import get_current_atom_config

            atom_config = get_current_atom_config()
        except Exception:  # noqa: BLE001
            return False
    if atom_config is None or not _is_flash_next_config(atom_config):
        return False

    positions = getattr(forward_batch, "positions", None)
    if not torch.is_tensor(positions):
        bs = int(getattr(forward_batch, "batch_size", 0) or 0)
        device = getattr(forward_batch, "device", None) or torch.device("cuda")
        positions = torch.zeros((max(bs, 1),), dtype=torch.int64, device=device)

    try:
        backend = resolve_attn_backend(forward_batch)
        # GDN indices live on the linear child — not full_attn_backend.
        gdn_backend = getattr(backend, "linear_attn_backend", None)
        if gdn_backend is None:
            gdn_backend = backend
        fm = getattr(gdn_backend, "forward_metadata", None)
        # Do NOT bool-evaluate CUDA tensors (``t1 or t2`` / ``getattr(...) or``).
        # A multi-element tensor raises
        # "Boolean value of Tensor with more than one value is ambiguous".
        idx = None
        if fm is not None:
            idx = getattr(fm, "mamba_cache_indices", None)
            if idx is None:
                idx = getattr(fm, "gdn_cache_indices", None)
        if idx is None:
            logger.debug(
                "Flash PLE graph refresh: no mamba_cache_indices on %s",
                type(gdn_backend).__name__,
            )
            return False
        build_ple_metadata(
            atom_config,
            forward_batch,
            positions,
            model=model,
            input_ids=None,
            gdn_metadata=SimpleNamespace(
                non_spec_state_indices_tensor=idx,
                non_spec_state_indices_in_tensor=None,
            ),
        )
        bs = int(getattr(forward_batch, "batch_size", 0) or 0)
        # Do not .item() positions on replay — that D2H-syncs every decode token.
        if in_capture or logger.isEnabledFor(logging.DEBUG):
            logger.info(
                "Flash decode graph PLE refresh ok (in_capture=%s): bs=%s",
                in_capture,
                bs,
            )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Flash decode graph PLE refresh failed (in_capture=%s): %s",
            in_capture,
            exc,
            exc_info=True,
        )
        return False


def flash_metadata_from_forward_batch(
    atom_config: Any,
    forward_batch: Any,
    positions: torch.Tensor,
    *,
    model: Any,
    input_ids: torch.Tensor | None,
    gdn_metadata: Any,
) -> SimpleNamespace:
    """One-step translation. Call every forward; do not reuse across steps."""
    bind_qsa_caches(model, forward_batch, atom_config)
    qsa = build_qsa_metadata(atom_config, forward_batch, positions)
    ple = build_ple_metadata(
        atom_config,
        forward_batch,
        positions,
        model=model,
        input_ids=input_ids,
        gdn_metadata=gdn_metadata,
    )
    return SimpleNamespace(qsa_metadata=qsa, ple_metadata=ple)
