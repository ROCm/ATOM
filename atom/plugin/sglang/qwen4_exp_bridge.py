# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang ForwardBatch → Native Qwen3.8-Flash-Next QSA / PLE metadata.

Compute stays in Native ATOM (#2048): QSA, indexer, GDN, hyper-connection, PLE.
This module only translates the current step's page tables into the structs
those kernels already read.

Decode CUDA-graph contract matches Native ATOM FULL graph:

- Capture bakes *addresses* of persistent QSA buffers into ``graph.replay()``.
- Replay (and capture) fill those buffers *outside* the graph, the way Native
  ``prepare_decode`` does: host ``build_batch_ids`` + H2D + Triton
  ``qsa_compressed_slots``. No GPU ``arange`` / ``searchsorted`` / ``where``
  rebuild, and ``model.forward`` must not re-run that rebuild while capturing
  (otherwise the eager aten ops get recorded into the graph).
- Prefill stays eager. Do not enable SGLang --ple-offload-embedding.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from atom.model_ops.attentions.qwen4_exp_attn import (
    Qwen4ExpPLEMetadata,
    Qwen4ExpQSAMetadata,
)
from atom.model_ops.attentions.token_layout.batch_ids import build_batch_ids
from atom.plugin.sglang.attention_backend.backend_resolver import (
    real_batch_size,
    resolve_attn_backend,
    resolve_mamba_req_pool,
)
from atom.utils import CpuGpuBuffer

logger = logging.getLogger(__name__)

# Dummy / graph-padding rows must not write QSA caches. Native kernels treat -1
# as no-write (same idea as the Qwen3.5 SGLang sentinel in #2067).
_NO_WRITE = -1


def _server_args() -> Any | None:
    try:
        from sglang.srt.server_args import get_global_server_args

        return get_global_server_args()
    except Exception:  # noqa: BLE001
        return None


def _cpu_gpu_i32(size: int, device: torch.device) -> CpuGpuBuffer:
    try:
        return CpuGpuBuffer(size, dtype=torch.int32, device=device, pin_memory=True)
    except Exception:  # noqa: BLE001
        return CpuGpuBuffer(size, dtype=torch.int32, device=device, pin_memory=False)


def _is_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


class _Qwen4ExpDecodeGraphBuffers:
    """Persistent QSA tensors whose addresses are baked into decode CUDA graphs.

    SGLang decode CUDA graphs capture kernel pointer operands. Metadata built
    with fresh allocations inside ``model.forward`` becomes stale on replay
    because Python does not re-run. These buffers are filled in-place from
    ``init_forward_metadata_out_graph`` before every capture/replay, and
    ``build_qsa_metadata`` returns views into the same storage so capture and
    replay share addresses.

    PLE slot indices are *not* copied here. Capture binds
    ``state_indices_*`` to Hybrid GDN's static ``mamba_cache_indices`` the way
    Native ``_ple_state_slots`` aliases the GDN CpuGpuBuffer. Replay's linear
    ``out_graph`` overwrites that tensor in place.
    """

    def __init__(self) -> None:
        self.max_bs = 0
        self.max_tokens = 0
        self.max_pages = 0
        self.device: torch.device | None = None
        self.block_tables: torch.Tensor | None = None
        self.slot_mapping: torch.Tensor | None = None
        self.compressed_slot_mapping: torch.Tensor | None = None
        self.token_to_req: torch.Tensor | None = None
        self.logical_positions: torch.Tensor | None = None
        self.seq_lens: torch.Tensor | None = None
        self.has_initial_state: torch.Tensor | None = None
        self.max_seq_len = 1
        self.active = False
        self.last_qsa: Qwen4ExpQSAMetadata | None = None
        self.last_ple: Qwen4ExpPLEMetadata | None = None
        self._token_to_req_buf: CpuGpuBuffer | None = None
        self._req_pool_indices: torch.Tensor | None = None

    def ensure(
        self,
        *,
        max_bs: int,
        max_tokens: int,
        max_pages: int,
        device: torch.device,
    ) -> None:
        max_bs = max(int(max_bs), 1)
        max_tokens = max(int(max_tokens), max_bs)
        max_pages = max(int(max_pages), 1)
        need = (
            self.block_tables is None
            or self._token_to_req_buf is None
            or self._req_pool_indices is None
            or self.has_initial_state is None
            or self.device != device
            or self.max_bs < max_bs
            or self.max_tokens < max_tokens
            or self.max_pages < max_pages
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
        self._token_to_req_buf = _cpu_gpu_i32(self.max_tokens, device)
        self.token_to_req = self._token_to_req_buf.gpu
        self._req_pool_indices = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device=device
        )
        self.logical_positions = torch.full(
            (self.max_tokens,), _NO_WRITE, dtype=torch.int64, device=device
        )
        self.seq_lens = torch.zeros((self.max_bs,), dtype=torch.int32, device=device)
        self.has_initial_state = torch.ones(
            (self.max_bs,), dtype=torch.bool, device=device
        )
        self.active = True

    def view_qsa(
        self,
        *,
        bs: int,
        num_tokens: int,
        max_seq_len: int,
    ) -> Qwen4ExpQSAMetadata:
        """Return slices of the persistent buffers already written in place."""
        assert self.block_tables is not None
        assert self.slot_mapping is not None
        assert self.compressed_slot_mapping is not None
        assert self.token_to_req is not None
        assert self.logical_positions is not None
        assert self.seq_lens is not None
        self.max_seq_len = max(int(max_seq_len), 1)
        self.last_qsa = Qwen4ExpQSAMetadata(
            block_tables=self.block_tables[:bs, : self.max_pages],
            slot_mapping=self.slot_mapping[:num_tokens],
            compressed_slot_mapping=self.compressed_slot_mapping[:num_tokens],
            token_to_req=self.token_to_req[:num_tokens],
            logical_positions=self.logical_positions[:num_tokens],
            seq_lens=self.seq_lens[:bs],
            max_seq_len=self.max_seq_len,
        )
        return self.last_qsa

    def bind_graph_ple(
        self,
        *,
        query_start_loc: torch.Tensor,
        ngram_state: torch.Tensor,
        state_indices: torch.Tensor,
        conv_state: torch.Tensor,
        batch_size: int,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> Qwen4ExpPLEMetadata:
        """Point PLE metadata at GDN static slot tensors (Native aliasing).

        ``state_indices_*`` and ``query_start_loc`` must be views of Hybrid
        GDN's cuda-graph buffers. Do not copy or clone them: the captured
        graph bakes these addresses, and linear ``out_graph`` fills them.
        """
        bs = int(batch_size)
        if self.has_initial_state is None:
            if _is_capturing():
                raise RuntimeError(
                    "Flash PLE graph buffers must be preallocated before capture"
                )
            self.ensure(
                max_bs=bs,
                max_tokens=max(bs, self.max_tokens),
                max_pages=max(self.max_pages, 1),
                device=query_start_loc.device,
            )
        assert self.has_initial_state is not None
        self.last_ple = Qwen4ExpPLEMetadata(
            query_start_loc=query_start_loc[: bs + 1],
            ngram_state=ngram_state,
            state_indices_in=state_indices[:bs],
            state_indices_out=state_indices[:bs],
            has_initial_state=self.has_initial_state[:bs],
            conv_state=conv_state,
            num_accepted_tokens=num_accepted_tokens,
        )
        return self.last_ple


_DECODE_GRAPH = _Qwen4ExpDecodeGraphBuffers()


def _pin_decode_graph(forward_batch: Any) -> bool:
    """Decode step that must write the persistent CUDA-graph QSA buffers."""
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not mode.is_decode_or_idle():
        return False
    return _DECODE_GRAPH.active or _is_capturing()


def _is_qwen4_exp_config(atom_config: Any) -> bool:
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


def _linear_static_ple_slots(
    forward_batch: Any, gdn_metadata: Any
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """GDN cuda-graph slot tensors PLE must alias (Native ``_ple_state_slots``).

    Prefer the linear child's ``forward_metadata`` so we never bind a clone
    that ``_build_gdn_metadata`` made for pad rows.
    """
    backend = resolve_attn_backend(forward_batch)
    if backend is None:
        return None
    linear = getattr(backend, "linear_attn_backend", None) or backend
    fm = getattr(linear, "forward_metadata", None)
    idx = getattr(fm, "mamba_cache_indices", None) if fm is not None else None
    qsl = getattr(fm, "query_start_loc", None) if fm is not None else None
    if idx is None or qsl is None:
        logger.debug(
            "Flash graph PLE: no static GDN slots (linear=%s has_gdn_md=%s)",
            type(linear).__name__,
            gdn_metadata is not None,
        )
        return None
    # Bind the static cuda-graph tensors themselves. A clone (pad-row fixup in
    # _build_gdn_metadata) would bake a throwaway address into the graph.
    return qsl, idx


def _hf_text_config(atom_config: Any) -> Any:
    hf = getattr(atom_config, "hf_config", atom_config)
    return getattr(hf, "text_config", None) or hf


def _indexer_budget(atom_config: Any) -> int:
    return int(getattr(_hf_text_config(atom_config), "indexer_budget", 2048) or 2048)


def _block_size(forward_batch: Any, atom_config: Any) -> int:
    """Resolve SGLang page size for QSA block tables.

    Must match the KV pool page size (server ``--page-size``). Preferring a
    smaller HF/atom ``block_size`` (e.g. 16) yields OOB page ids on long decode
    and HSA faults under CUDA-graph replay.
    """
    args = _server_args()
    pool = getattr(forward_batch, "token_to_kv_pool", None) or getattr(
        forward_batch, "token_to_kv_pool_allocator", None
    )
    req_pool = getattr(forward_batch, "req_to_token_pool", None)
    for candidate in (
        getattr(args, "page_size", None) if args is not None else None,
        getattr(forward_batch, "page_size", None),
        getattr(pool, "page_size", None),
        getattr(req_pool, "page_size", None),
        getattr(atom_config, "page_size", None),
        getattr(atom_config, "kv_cache_block_size", None),
    ):
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


def _query_start_loc(
    forward_batch: Any, num_tokens: int, device: torch.device
) -> torch.Tensor:
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


def _num_tokens_from_positions(positions: torch.Tensor) -> int:
    """Token count for QSA/PLE. A 3-row mRoPE tensor is ``[3, tokens]``."""
    if positions.ndim == 2 and positions.shape[0] in (1, 3):
        return int(positions.shape[-1])
    return int(positions.reshape(-1).numel())


def _sequence_index_positions(positions: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """1-D sequence index for QSA grouping. Do not flatten 3-row mRoPE.

    Flash forward must pass ``forward_batch.positions`` (1-D). Native RoPE
    uses the separate 3-row ``mrope_positions``. Flattening ``[3, N]`` would
    triple the token count and treat the T-axis as the sequence index.
    """
    if positions.ndim == 2 and positions.shape[0] in (1, 3):
        pos = positions[0, :num_tokens]
    else:
        pos = positions.reshape(-1)[:num_tokens]
    return pos.to(torch.int64)


def _token_to_req_and_logical(
    *,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill fallback: token→req from cu_seqlens (not used on decode graph)."""
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
    pos = _sequence_index_positions(positions, num_tokens)
    logical = torch.where(valid, pos, logical)
    return token_to_req, logical


def _compressed_slots_native(
    slot_mapping: torch.Tensor,
    logical_positions: torch.Tensor,
    compress_ratio: int,
    out: torch.Tensor,
) -> None:
    """Native ATOM formula: floor(physical_slot / ratio) on complete groups.

    ``qsa_compressed_slots`` is the GPU kernel Native ``prepare_decode`` runs
    before ``graph.replay()``. CPU tests use the same expression.
    """
    if slot_mapping.numel() == 0:
        return
    if slot_mapping.is_cuda:
        from atom.model_ops.qwen4_exp.ops.qsa import qsa_compressed_slots

        qsa_compressed_slots(slot_mapping, logical_positions, compress_ratio, out)
        return
    closes = (
        (logical_positions >= 0)
        & ((logical_positions + 1) % compress_ratio == 0)
        & (slot_mapping >= 0)
    )
    out.copy_(torch.where(closes, slot_mapping // compress_ratio, _NO_WRITE))


def _fill_block_tables_into(
    out: torch.Tensor,
    *,
    pool: Any,
    req_pool_indices: torch.Tensor,
    table_tokens: int,
    block_size: int,
    live_bs: int,
) -> None:
    """Gather live rows into ``out[:bs]``; zero pad rows. No max_bs fill."""
    req_to_token = pool.req_to_token
    bs = int(req_pool_indices.shape[0])
    live = max(min(int(live_bs), bs), 0)
    max_blocks = max(1, (int(table_tokens) + block_size - 1) // block_size)
    pages = min(max_blocks, int(out.shape[1]))
    out[:bs].zero_()
    if live == 0 or pages == 0:
        return
    gathered = req_to_token[req_pool_indices[:live], : pages * block_size : block_size]
    gathered = gathered.to(dtype=out.dtype) // block_size
    copy_pages = min(int(gathered.shape[1]), pages)
    out[:live, :copy_pages].copy_(gathered[:, :copy_pages])


def _clamp_table_tokens(pool: Any, table_tokens: int) -> int:
    req_to_token = getattr(pool, "req_to_token", None)
    if torch.is_tensor(req_to_token) and req_to_token.dim() >= 2:
        return min(table_tokens, int(req_to_token.shape[1]))
    return table_tokens


def _eager_block_tables(
    *,
    pool: Any,
    req_pool_indices: torch.Tensor,
    table_tokens: int,
    block_size: int,
    live_bs: int,
    device: torch.device,
) -> torch.Tensor:
    pages = max(1, (int(table_tokens) + block_size - 1) // block_size)
    tables = torch.zeros(
        (int(req_pool_indices.shape[0]), pages), dtype=torch.int32, device=device
    )
    _fill_block_tables_into(
        tables,
        pool=pool,
        req_pool_indices=req_pool_indices,
        table_tokens=table_tokens,
        block_size=block_size,
        live_bs=live_bs,
    )
    return tables


def _eager_slot_mapping(
    forward_batch: Any, num_tokens: int, device: torch.device
) -> torch.Tensor:
    out = torch.full((num_tokens,), _NO_WRITE, dtype=torch.int64, device=device)
    loc = getattr(forward_batch, "out_cache_loc", None)
    if not torch.is_tensor(loc) or loc.numel() == 0 or num_tokens <= 0:
        return out
    src = loc.reshape(-1)[:num_tokens].to(device=device, dtype=torch.int64)
    out[: src.numel()] = torch.where(src < 0, torch.full_like(src, _NO_WRITE), src)
    return out


def _serving_context_len(indexer_budget: int) -> int:
    ctx_len = indexer_budget
    args = _server_args()
    if args is None:
        return ctx_len
    return max(
        ctx_len,
        int(getattr(args, "context_length", 0) or 0),
        int(getattr(args, "max_model_len", 0) or 0),
    )


def _fill_qsa_like_native(
    *,
    forward_batch: Any,
    positions: torch.Tensor,
    pool: Any,
    block_size: int,
    compress_ratio: int,
    live_bs: int,
    bs: int,
    num_tokens: int,
    indexer_budget: int,
    ctx_len: int,
) -> Qwen4ExpQSAMetadata:
    """Native ``_build_qsa_metadata`` / ``prepare_decode`` for SGLang decode graph.

    Host ``build_batch_ids`` + H2D for token→req, GPU gather of live page-table
    rows, Triton ``qsa_compressed_slots`` in place. Persistent addresses stay
    the ones baked into the CUDA graph.
    """
    device = positions.device
    max_pages = _DECODE_GRAPH.max_pages if _DECODE_GRAPH.max_pages > 0 else 1
    _DECODE_GRAPH.ensure(
        max_bs=bs,
        max_tokens=num_tokens,
        max_pages=max_pages,
        device=device,
    )
    assert _DECODE_GRAPH.block_tables is not None
    assert _DECODE_GRAPH.slot_mapping is not None
    assert _DECODE_GRAPH.compressed_slot_mapping is not None
    assert _DECODE_GRAPH.logical_positions is not None
    assert _DECODE_GRAPH.seq_lens is not None
    assert _DECODE_GRAPH._req_pool_indices is not None

    table_tokens = _clamp_table_tokens(pool, max(ctx_len, indexer_budget))

    req_idx = _DECODE_GRAPH._req_pool_indices[:bs]
    src_idx = forward_batch.req_pool_indices[:bs]
    if src_idx.dtype != torch.int32:
        src_idx = src_idx.to(dtype=torch.int32)
    req_idx.copy_(src_idx)
    if live_bs < bs:
        req_idx[live_bs:].zero_()
    _fill_block_tables_into(
        _DECODE_GRAPH.block_tables,
        pool=pool,
        req_pool_indices=req_idx,
        table_tokens=table_tokens,
        block_size=block_size,
        live_bs=live_bs,
    )

    live_tokens = min(num_tokens, live_bs)
    slot = _DECODE_GRAPH.slot_mapping[:num_tokens]
    slot.fill_(_NO_WRITE)
    out_loc = getattr(forward_batch, "out_cache_loc", None)
    if torch.is_tensor(out_loc) and live_tokens:
        loc = out_loc.reshape(-1)[:live_tokens]
        slot[: loc.numel()].copy_(loc.to(dtype=torch.int64))

    logical = _DECODE_GRAPH.logical_positions[:num_tokens]
    logical.fill_(_NO_WRITE)
    pos = _sequence_index_positions(positions, num_tokens)
    copy_pos = min(int(pos.numel()), live_tokens)
    if copy_pos:
        logical[:copy_pos].copy_(pos[:copy_pos])

    seq = _DECODE_GRAPH.seq_lens[:bs]
    src_seq = getattr(forward_batch, "seq_lens", None)
    if torch.is_tensor(src_seq):
        seq.copy_(src_seq[:bs].to(device=device, dtype=torch.int32))
    else:
        seq.fill_(1)
    if live_bs < bs:
        seq[live_bs:] = 0
        slot[live_bs:num_tokens] = _NO_WRITE
        logical[live_bs:num_tokens] = _NO_WRITE

    if _DECODE_GRAPH._token_to_req_buf is not None:
        if live_bs == 0:
            _DECODE_GRAPH._token_to_req_buf.np[:num_tokens] = _NO_WRITE
        else:
            build_batch_ids(
                np.ones(live_bs, dtype=np.int64),
                pad_to=num_tokens,
                pad=_NO_WRITE,
                out=_DECODE_GRAPH._token_to_req_buf.np,
            )
        _DECODE_GRAPH._token_to_req_buf.copy_to_gpu(num_tokens)
    else:
        ttr = _DECODE_GRAPH.token_to_req[:num_tokens]
        ttr.fill_(_NO_WRITE)
        if live_bs:
            ttr[:live_bs] = torch.arange(live_bs, dtype=torch.int32, device=device)

    _compressed_slots_native(
        slot,
        logical,
        compress_ratio,
        _DECODE_GRAPH.compressed_slot_mapping[:num_tokens],
    )
    max_seq_len = max(ctx_len, indexer_budget, 1)
    return _DECODE_GRAPH.view_qsa(bs=bs, num_tokens=num_tokens, max_seq_len=max_seq_len)


def build_qsa_metadata(
    atom_config: Any,
    forward_batch: Any,
    positions: torch.Tensor,
) -> Qwen4ExpQSAMetadata | None:
    pool = _req_to_token_pool(forward_batch)
    if pool is None or not hasattr(pool, "req_to_token"):
        logger.debug("Flash QSA bridge: no req_to_token pool; skip QSA metadata")
        return None

    device = positions.device
    bs = int(forward_batch.batch_size)
    num_tokens = _num_tokens_from_positions(positions)
    block_size = _block_size(forward_batch, atom_config)
    compress_ratio = _compress_ratio(atom_config)
    if block_size % compress_ratio:
        raise ValueError(
            f"page-size / block-size ({block_size}) must be divisible by "
            f"indexer_compress_ratio ({compress_ratio})"
        )

    live_bs = real_batch_size(forward_batch)
    indexer_budget = _indexer_budget(atom_config)
    ctx_len = _serving_context_len(indexer_budget)
    if _pin_decode_graph(forward_batch):
        return _fill_qsa_like_native(
            forward_batch=forward_batch,
            positions=positions,
            pool=pool,
            block_size=block_size,
            compress_ratio=compress_ratio,
            live_bs=live_bs,
            bs=bs,
            num_tokens=num_tokens,
            indexer_budget=indexer_budget,
            ctx_len=ctx_len,
        )

    seq_lens = _seq_lens(forward_batch, device)[:bs]
    # Decode-graph capture pins scoring width to the engine context so every
    # replay has a constant grid. Eager prefill/extend must use the live batch:
    # `_DECODE_GRAPH.active` stays True after capture and would otherwise make a
    # 4096-token prefill score `context_length` (e.g. 131072) on every QSA layer.
    if _is_capturing() or seq_lens.numel() == 0:
        max_seq_len = ctx_len
    else:
        max_seq_len = int(seq_lens.max().item())
    table_tokens = _clamp_table_tokens(pool, max(max_seq_len, indexer_budget))
    block_tables = _eager_block_tables(
        pool=pool,
        req_pool_indices=forward_batch.req_pool_indices[:bs],
        table_tokens=table_tokens,
        block_size=block_size,
        live_bs=live_bs,
        device=device,
    )
    slot_mapping = _eager_slot_mapping(forward_batch, num_tokens, device)
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
        pad_tokens = min(num_tokens, bs)
        if pad_tokens > live_bs:
            slot_mapping = slot_mapping.clone()
            slot_mapping[live_bs:pad_tokens] = _NO_WRITE
            logical = logical.clone()
            logical[live_bs:pad_tokens] = _NO_WRITE

    compressed = torch.empty_like(slot_mapping)
    _compressed_slots_native(slot_mapping, logical, compress_ratio, compressed)
    return Qwen4ExpQSAMetadata(
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        compressed_slot_mapping=compressed,
        token_to_req=token_to_req,
        logical_positions=logical,
        seq_lens=seq_lens,
        max_seq_len=max(max_seq_len, 1),
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
        slots = _max_int_attrs(pool, ("size", "max_num_reqs", "mamba_size"), slots)
    slots = _max_int_attrs(
        _req_to_token_pool(forward_batch), ("size", "max_num_reqs"), slots
    )
    if torch.is_tensor(idx) and idx.numel() and not _is_capturing():
        slots = max(slots, int(idx.clamp(min=0).max().item()) + 1)
    return slots


def _max_int_attrs(obj: Any, keys: tuple[str, ...], current: int) -> int:
    if obj is None:
        return current
    for key in keys:
        val = getattr(obj, key, None)
        if val:
            current = max(current, int(val))
    return current


def _grow_once_tensor(
    model: Any,
    attr: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    fill: int | None = None,
) -> torch.Tensor:
    """Allocate a persistent model buffer once; refuse grow after CUDA-graph capture."""
    existing = getattr(model, attr, None)
    rows, *tail = shape
    if (
        existing is not None
        and existing.shape[0] >= rows
        and list(existing.shape[1:]) == list(tail)
    ):
        return existing
    if existing is not None and _DECODE_GRAPH.active:
        logger.warning(
            "%s already captured at %s slots; refusing grow to %s",
            attr,
            int(existing.shape[0]),
            rows,
        )
        return existing
    device = next(model.parameters()).device
    if fill is None:
        tensor = torch.zeros(shape, dtype=dtype, device=device)
    else:
        tensor = torch.full(shape, fill, dtype=dtype, device=device)
    setattr(model, attr, tensor)
    return tensor


def _ensure_ple_states(
    model: Any, atom_config: Any, num_slots: int
) -> tuple[torch.Tensor, torch.Tensor]:
    hf = _hf_text_config(atom_config)
    ngram = max(int(getattr(hf, "ngram_size", 3)), 1)
    rows = max(num_slots, 1)
    conv_state = _grow_once_tensor(
        model,
        "_atom_ple_conv_state",
        shape=(
            rows,
            int(hf.hidden_size) * int(getattr(hf, "hc_count", 4)),
            (int(getattr(hf, "ple_conv_kernel_size", 4)) - 1) * ngram,
        ),
        dtype=getattr(atom_config, "torch_dtype", None) or torch.bfloat16,
    )
    eos = getattr(hf, "eos_token_id", 0)
    ngram_state = _grow_once_tensor(
        model,
        "_atom_ple_ngram_state",
        shape=(rows, max(ngram - 1, 1)),
        dtype=torch.int64,
        fill=int(eos[0] if isinstance(eos, (list, tuple)) else eos),
    )
    return conv_state, ngram_state


def _build_decode_graph_ple_metadata(
    atom_config: Any,
    forward_batch: Any,
    *,
    model: Any,
    gdn_metadata: Any,
) -> Qwen4ExpPLEMetadata | None:
    """Bind PLE to GDN static buffers. No copies — graph replay updates them."""
    slots = _linear_static_ple_slots(forward_batch, gdn_metadata)
    if slots is None:
        return None
    query_start_loc, idx = slots
    bs = int(forward_batch.batch_size)
    num_slots = _ple_state_pool_slots(forward_batch, idx)
    conv_state, ngram_state = _ensure_ple_states(model, atom_config, num_slots)
    ple = _DECODE_GRAPH.bind_graph_ple(
        query_start_loc=query_start_loc,
        ngram_state=ngram_state,
        state_indices=idx,
        conv_state=conv_state,
        batch_size=bs,
        num_accepted_tokens=getattr(gdn_metadata, "num_accepted_tokens", None),
    )
    if not getattr(_build_decode_graph_ple_metadata, "_logged", False):
        logger.info(
            "Flash decode graph PLE aliases GDN mamba_cache_indices "
            "(bs=%s idx_ptr=%s qsl_ptr=%s)",
            bs,
            int(idx.data_ptr()),
            int(query_start_loc.data_ptr()),
        )
        _build_decode_graph_ple_metadata._logged = True  # type: ignore[attr-defined]
    return ple


def build_ple_metadata(
    atom_config: Any,
    forward_batch: Any,
    positions: torch.Tensor,
    *,
    model: Any,
    gdn_metadata: Any,
) -> Qwen4ExpPLEMetadata | None:
    hf = _hf_text_config(atom_config)
    if not getattr(hf, "ple_layer_ids", None):
        return None
    if _pin_decode_graph(forward_batch):
        return _build_decode_graph_ple_metadata(
            atom_config, forward_batch, model=model, gdn_metadata=gdn_metadata
        )
    if gdn_metadata is None:
        return None
    idx = getattr(gdn_metadata, "non_spec_state_indices_tensor", None)
    if idx is None:
        return None

    device = positions.device
    bs = int(forward_batch.batch_size)
    num_tokens = _num_tokens_from_positions(positions)
    query_start_loc = _query_start_loc(forward_batch, num_tokens, device)
    is_prefill = bool(forward_batch.forward_mode.is_extend())

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
    conv_state, ngram_state = _ensure_ple_states(model, atom_config, num_slots)
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

    return Qwen4ExpPLEMetadata(
        query_start_loc=query_start_loc,
        ngram_state=ngram_state,
        state_indices_in=idx_in,
        state_indices_out=idx,
        has_initial_state=has_initial,
        conv_state=conv_state,
        num_accepted_tokens=getattr(gdn_metadata, "num_accepted_tokens", None),
    )


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
    if getattr(model, "_atom_qwen4_exp_qsa_bound", False):
        return
    qsa_layers = [
        mod for mod in model.modules() if getattr(mod, "is_qsa_attention", False)
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
    model._atom_qwen4_exp_qsa_raw = raw
    model._atom_qwen4_exp_qsa_compressed = compressed
    model._atom_qwen4_exp_qsa_bound = True
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


def _decode_graph_capacity(
    atom_config: Any, forward_batch: Any
) -> tuple[int, int, torch.device]:
    """High-water sizes so capture never reallocates persistent QSA buffers."""
    device = getattr(forward_batch, "device", None)
    if device is None:
        pos = getattr(forward_batch, "positions", None)
        device = pos.device if torch.is_tensor(pos) else torch.device("cuda")
    bs = int(getattr(forward_batch, "batch_size", 0) or 0)
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
        args = _server_args()
        if args is not None:
            graph_bs = int(
                getattr(args, "cuda_graph_max_bs_decode", None)
                or getattr(args, "cuda_graph_max_bs", 0)
                or 0
            )
    max_bs = max(bs, graph_bs, 8)
    indexer_budget = _indexer_budget(atom_config)
    block_size = _block_size(forward_batch, atom_config)
    # Page table must cover the full context, not just indexer_budget.
    # QSA top-k still keeps only ``indexer_budget`` tokens, but those tokens
    # can sit anywhere in a 12k sequence; a 32-page (2k) table would clamp
    # logical_page and drop the recent context.
    ctx_len = _serving_context_len(indexer_budget)
    max_pages = max((ctx_len + block_size - 1) // block_size, 1)
    return max_bs, max_pages, torch.device(device)


def prepare_qwen4_exp_decode_graph_metadata(
    forward_batch: Any,
    in_capture: bool = False,
    *,
    atom_config: Any | None = None,
) -> Qwen4ExpQSAMetadata | None:
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
    if atom_config is None or not _is_qwen4_exp_config(atom_config):
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
    _DECODE_GRAPH.ensure(
        max_bs=max_bs,
        max_tokens=max_bs,
        max_pages=max_pages,
        device=device,
    )

    qsa = build_qsa_metadata(atom_config, forward_batch, positions)
    # QSA page tables only. PLE aliases Hybrid GDN's static mamba_cache_indices
    # during warmup/capture forward (after this out_graph returns and the
    # linear child has filled those slots). Replay does not rebuild PLE.
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


def qwen4_exp_metadata_from_forward_batch(
    atom_config: Any,
    forward_batch: Any,
    positions: torch.Tensor,
    *,
    model: Any,
    gdn_metadata: Any,
) -> SimpleNamespace:
    """One-step translation. Call every forward; do not reuse across steps.

    During CUDA-graph *capture* Native does not rebuild metadata inside
    ``model()``: ``prepare_decode`` already wrote the persistent buffers.
    Rebuilding here would record the eager aten chain into the graph.

    PLE ``state_indices_*`` alias GDN's static ``mamba_cache_indices`` (bound
    on the warmup forward before capture). Replay's linear out_graph overwrites
    that storage in place; there is no extra PLE copy.
    """
    bind_qsa_caches(model, forward_batch, atom_config)
    if _is_capturing() and _DECODE_GRAPH.last_qsa is not None:
        ple = _DECODE_GRAPH.last_ple
        if ple is None:
            ple = _build_decode_graph_ple_metadata(
                atom_config, forward_batch, model=model, gdn_metadata=gdn_metadata
            )
        return SimpleNamespace(
            qsa_metadata=_DECODE_GRAPH.last_qsa,
            ple_metadata=ple,
        )
    return SimpleNamespace(
        qsa_metadata=build_qsa_metadata(atom_config, forward_batch, positions),
        ple_metadata=build_ple_metadata(
            atom_config,
            forward_batch,
            positions,
            model=model,
            gdn_metadata=gdn_metadata,
        ),
    )
