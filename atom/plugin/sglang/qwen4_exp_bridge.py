# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang ForwardBatch → Native Qwen3.8-Flash-Next QSA / PLE metadata.

Compute stays in Native ATOM (#2048): QSA, indexer, GDN, hyper-connection, PLE.
This module only translates the current step's page tables into the structs
those kernels already read.

Decode / TARGET_VERIFY graph contract:

- Allocate persistent buffers for the serving batch ceiling, MTP token width
  and full context length before capture. Never replace captured storage.
- Fill QSA metadata outside the graph. Packed token IDs use request-major
  order; padded tokens and unused pages carry the Native ``-1`` sentinel.
- Attach the active backend's request pool to batches and graph replay views.
- Draft runs eagerly and must not overwrite the target's graph buffers.
- Lift speculative QSA context lengths to cover query positions beyond
  SGLang's accepted prefix, without changing scheduler-owned seq_lens.
- PLE aliases Hybrid GDN's static state indices. Prefill stays eager.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import torch

from atom.model_ops.attentions.qwen4_exp_attn import (
    Qwen4ExpPLEMetadata,
    Qwen4ExpQSAMetadata,
)
from atom.plugin.sglang.attention_backend.backend_resolver import (
    real_batch_size,
    resolve_attn_backend,
    resolve_mamba_req_pool,
)
from atom.utils.forward_context import get_forward_context

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
    Native ``_ple_state_slots`` aliases the GDN slot buffer. Replay's linear
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
        self.active = False
        self.last_qsa: Qwen4ExpQSAMetadata | None = None
        self.last_ple: Qwen4ExpPLEMetadata | None = None
        self.num_accepted_tokens: torch.Tensor | None = None

    def reset(self) -> None:
        """Drop persistent buffers. Tests only; never call after CUDA-graph capture."""
        self.__init__()

    def allocate_once(
        self,
        *,
        max_bs: int,
        max_tokens: int,
        max_pages: int,
        device: torch.device,
    ) -> None:
        """Allocate once at serving capacity; reject later growth/device changes.

        ``max_tokens`` includes the launch-time MTP width, even if the first
        capture is ordinary decode. ``max_pages`` covers the full context,
        since selected indexer tokens may occur anywhere in that context.
        """
        max_bs = max(int(max_bs), 1)
        max_tokens = max(int(max_tokens), max_bs)
        max_pages = max(int(max_pages), 1)
        need = (
            self.block_tables is None
            or self.token_to_req is None
            or self.has_initial_state is None
            or self.device != device
            or self.max_bs < max_bs
            or self.max_tokens < max_tokens
            or self.max_pages < max_pages
        )
        if not need:
            return
        if self.block_tables is not None:
            raise RuntimeError(
                "Flash QSA graph capacity/device mismatch: "
                f"allocated bs={self.max_bs} tokens={self.max_tokens} "
                f"pages={self.max_pages} device={self.device}; "
                f"requested bs={max_bs} tokens={max_tokens} "
                f"pages={max_pages} device={device}. "
                "Allocate the serving capacity before CUDA-graph capture."
            )
        self.max_bs = max_bs
        self.max_tokens = max_tokens
        self.max_pages = max_pages
        self.device = device
        self.block_tables = torch.zeros(
            (self.max_bs, self.max_pages), dtype=torch.int32, device=device
        )
        self.slot_mapping = torch.full(
            (self.max_tokens,), _NO_WRITE, dtype=torch.int64, device=device
        )
        self.compressed_slot_mapping = torch.full(
            (self.max_tokens,), _NO_WRITE, dtype=torch.int64, device=device
        )
        # Graph-baked GPU address. Packed ids are written in place; pad stays -1.
        self.token_to_req = torch.full(
            (self.max_tokens,), _NO_WRITE, dtype=torch.int32, device=device
        )
        self.logical_positions = torch.full(
            (self.max_tokens,), _NO_WRITE, dtype=torch.int64, device=device
        )
        self.seq_lens = torch.zeros((self.max_bs,), dtype=torch.int32, device=device)
        self.has_initial_state = torch.ones(
            (self.max_bs,), dtype=torch.bool, device=device
        )
        self.num_accepted_tokens = torch.ones(
            (self.max_bs,), dtype=torch.int32, device=device
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
        self.last_qsa = Qwen4ExpQSAMetadata(
            block_tables=self.block_tables[:bs, : self.max_pages],
            slot_mapping=self.slot_mapping[:num_tokens],
            compressed_slot_mapping=self.compressed_slot_mapping[:num_tokens],
            token_to_req=self.token_to_req[:num_tokens],
            logical_positions=self.logical_positions[:num_tokens],
            seq_lens=self.seq_lens[:bs],
            max_seq_len=max(int(max_seq_len), 1),
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
            raise RuntimeError("Flash PLE graph buffers must be allocated before bind")
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
# Draft stays eager (Flash mRoPE) but must not write the buffers baked into
# the target verify graph. A second set still keeps page tables and seq_lens
# on device, so a draft step does not allocate or sync seq_lens back to host.
_DRAFT_QSA = _Qwen4ExpDecodeGraphBuffers()


def _is_draft_forward() -> bool:
    """True while the NextN draft model owns this forward (``context.is_draft``)."""
    return bool(getattr(get_forward_context().context, "is_draft", False))


def _is_target_verify(forward_batch: Any) -> bool:
    mode = getattr(forward_batch, "forward_mode", None)
    return bool(
        mode is not None
        and callable(getattr(mode, "is_target_verify", None))
        and mode.is_target_verify()
    )


def _is_verify_or_draft_extend(forward_batch: Any) -> bool:
    """TARGET_VERIFY or DRAFT_EXTEND_V2: request-major equal-width
    ``tokens_per_req`` tokens.

    Check these modes explicitly; they are not ragged prefill.
    ``token_to_req`` / cu_seqlens must use ``draft_token_num``, not leftover
    ``extend_start_loc``. Decode (1 token/req) is neither.
    """
    if _is_target_verify(forward_batch):
        return True
    mode = getattr(forward_batch, "forward_mode", None)
    return bool(
        mode is not None
        and callable(getattr(mode, "is_draft_extend_v2", None))
        and mode.is_draft_extend_v2()
    )


def _use_decode_graph_buffers(forward_batch: Any) -> bool:
    """Decode / TARGET_VERIFY step that must write persistent CUDA-graph QSA buffers.

    Draft decode is eager (Flash mRoPE) and shares this process with the
    target. Writing the draft into ``_DECODE_GRAPH`` overwrites the addresses
    TARGET_VERIFY already captured.
    """
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or _is_draft_forward():
        return False
    if not (mode.is_decode_or_idle() or mode.is_target_verify()):
        return False
    if _DECODE_GRAPH.active and forward_batch.batch_size > _DECODE_GRAPH.max_bs:
        # SGLang falls back to eager above the graph ceiling. Build separate
        # metadata instead of writing beyond the captured buffers.
        return False
    return _DECODE_GRAPH.active or _is_capturing()


def _get_qsa_tokens_per_req(forward_batch: Any) -> int:
    """How many tokens this step actually fills per live request.

    Decode / idle is always 1. TARGET_VERIFY / DRAFT_EXTEND is
    ``draft_token_num`` (3 when ``speculative-num-steps=2``). Buffer
    *allocation* uses ``_get_qsa_graph_max_tokens_per_req`` (CLI width).
    """
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is not None and mode.is_decode_or_idle():
        return 1
    spec = getattr(forward_batch, "spec_info", None)
    for attr in ("draft_token_num", "num_tokens_per_req"):
        val = getattr(spec, attr, None)
        if val:
            return max(int(val), 1)
    return 1


def _get_qsa_graph_max_tokens_per_req() -> int:
    """Launch-time tokens/req for persistent QSA CUDA-graph buffers.

    After SGLang speculative init, TARGET_VERIFY width is
    ``speculative_num_draft_tokens`` (Flash chain: ``steps+1``). Fill still
    uses ``_get_qsa_tokens_per_req``.
    """
    args = _server_args()
    return max(int(getattr(args, "speculative_num_draft_tokens", 0) or 0), 1)


def bind_qsa_replay_batch(forward_batch: Any, backend: Any) -> Any:
    """Attach pool pointers that SGLang's CUDA-graph replay view drops.

    ``build_replay_fb_view`` is a ``SimpleNamespace`` with seq_lens / positions
    / req_pool_indices. Without ``req_to_token_pool``,
    QSA metadata cannot gather the current request page tables.
    """
    if getattr(forward_batch, "req_to_token_pool", None) is None:
        pool = getattr(backend, "req_to_token_pool", None)
        if pool is not None:
            forward_batch.req_to_token_pool = pool
    return forward_batch


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
    return any("Qwen4Exp" in str(a) for a in arch)


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
    """SGLang page table on the batch.

    Wrappers attach the active backend pool before model forward. Graph
    replay views use ``bind_qsa_replay_batch`` to attach it before metadata fill.
    """
    pool = forward_batch.req_to_token_pool
    if pool is None:
        raise RuntimeError(
            "Flash QSA requires req_to_token_pool on the batch; "
            "replay views must call bind_qsa_replay_batch first"
        )
    return pool


def _seq_lens(forward_batch: Any, device: torch.device) -> torch.Tensor:
    seq = getattr(forward_batch, "seq_lens", None)
    bs = int(getattr(forward_batch, "batch_size", 0) or 0)
    if torch.is_tensor(seq):
        # QSA may lift these lengths for speculative queries; never mutate
        # the scheduler's prefix lengths.
        seq = seq.to(device=device, dtype=torch.int32)[:bs].clone()
    else:
        seq = torch.ones((bs,), dtype=torch.int32, device=device)
    live_bs = real_batch_size(forward_batch)
    if live_bs < seq.shape[0]:
        # CUDA-graph pad rows keep seq_len_fill_value (usually 1) and a
        # finished request's page table. Zero them so QSA does not score
        # or write freed pages.
        seq[live_bs:] = 0
    return seq


def _lift_qsa_seq_lens(
    seq_lens: torch.Tensor,
    logical_positions: torch.Tensor,
    *,
    live_bs: int,
    tokens_per_req: int,
) -> None:
    """Raise QSA context lengths to include each request's speculative queries.

    Native QSA requires ``logical_pos < seq_lens[req]``, while SGLang verify
    and draft positions can extend beyond its accepted prefix lengths.
    Operate on a QSA-owned copy. Only complete live rows contribute; padded
    rows stay zero and ordinary decode already satisfies the bound.
    """
    tokens_per_req = max(int(tokens_per_req), 1)
    grouped = min(int(logical_positions.numel()), int(live_bs) * tokens_per_req)
    grouped //= tokens_per_req
    if grouped <= 0:
        return
    max_pos = (
        logical_positions[: grouped * tokens_per_req]
        .view(grouped, tokens_per_req)
        .clamp(min=0)
        .max(dim=1)
        .values
    )
    torch.maximum(
        seq_lens[:grouped],
        (max_pos + 1).to(dtype=seq_lens.dtype),
        out=seq_lens[:grouped],
    )


def _query_start_loc(
    forward_batch: Any, num_tokens: int, device: torch.device
) -> torch.Tensor:
    mode = forward_batch.forward_mode
    bs = int(forward_batch.batch_size)
    live_bs = real_batch_size(forward_batch)
    # Packed decode / TARGET_VERIFY / DRAFT_EXTEND must not fall through
    # to leftover prefill extend_start_loc.
    # Decode width is 1 via ``_get_qsa_tokens_per_req``.
    if mode.is_decode_or_idle() or _is_verify_or_draft_extend(forward_batch):
        tokens_per_req = _get_qsa_tokens_per_req(forward_batch)
        loc = torch.arange(
            0,
            tokens_per_req * bs + 1,
            tokens_per_req,
            dtype=torch.int32,
            device=device,
        )
        loc[live_bs + 1 :] = tokens_per_req * live_bs
        return loc
    if mode.is_extend():
        start = getattr(forward_batch, "extend_start_loc", None)
        lens = getattr(forward_batch, "extend_seq_lens", None)
        if torch.is_tensor(start) and torch.is_tensor(lens) and live_bs:
            loc = torch.empty((bs + 1,), dtype=torch.int32, device=device)
            loc[:live_bs] = start[:live_bs].to(dtype=torch.int32)
            loc[live_bs:] = (start[live_bs - 1] + lens[live_bs - 1]).to(
                dtype=torch.int32
            )
            return loc
        loc = torch.empty((bs + 1,), dtype=torch.int32, device=device)
        loc.fill_(0)
        if live_bs:
            loc[live_bs:] = num_tokens
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


def _token_to_req_from_packed(
    *,
    live_bs: int,
    tokens_per_req: int,
    num_tokens: int,
    device: torch.device,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed (equal-width) token→req. Ragged uses
    ``_token_to_req_from_query_start_loc``. Pad is ``-1``.

    Searchsorted-from-cu_seqlens defaults unmapped rows to 0, so a mixed
    TARGET_VERIFY / decode batch would score every leftover token against
    request 0's page table. Identical-length concurrent requests hide that
    (same context length); mixed lengths do not.

    Graph replay passes the baked ``token_to_req`` as ``out`` so the CUDA-graph
    address stays fixed. Eager allocates a fresh tensor.

    Decode (``tokens_per_req == 1``) is ``arange(n)``. Verify / draft-extend
    is ``arange(n) // tokens_per_req``. Pad is filled once; live rows are
    not written as ``-1`` then overwritten.
    """
    dst = (
        out[:num_tokens]
        if out is not None
        else torch.empty((num_tokens,), dtype=torch.int32, device=device)
    )
    tokens_per_req = max(int(tokens_per_req), 1)
    n = min(int(num_tokens), max(int(live_bs), 0) * tokens_per_req)
    if n > 0:
        torch.arange(n, out=dst[:n], dtype=torch.int32, device=dst.device)
        if tokens_per_req > 1:
            dst[:n].floor_divide_(tokens_per_req)
    if n < dst.numel():
        dst[n:].fill_(_NO_WRITE)
    return dst


def _qsa_logical_positions(
    positions: torch.Tensor,
    num_tokens: int,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Sequence index per token. Invalid rows stay ``_NO_WRITE``, not 0."""
    logical = torch.full(
        (num_tokens,), _NO_WRITE, dtype=torch.int64, device=positions.device
    )
    pos = _sequence_index_positions(positions, num_tokens)
    if pos.numel() == 0:
        return logical
    n = pos.numel()
    # ``valid`` is a prefix (packed live rows, or tokens before cu_seqlens[-1]).
    # Write the prefix in place; holes stay ``_NO_WRITE`` without a second tensor.
    logical[:n].copy_(pos)
    logical[:n].masked_fill_(~valid[:n], _NO_WRITE)
    return logical


def _token_to_req_from_query_start_loc(
    query_start_loc: torch.Tensor | None,
    num_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ragged prefill token→req. Unmapped rows are ``_NO_WRITE``, not 0."""
    if query_start_loc is None or query_start_loc.numel() < 2 or num_tokens <= 0:
        token_to_req = torch.full(
            (num_tokens,), _NO_WRITE, dtype=torch.int32, device=device
        )
        valid = torch.zeros((num_tokens,), dtype=torch.bool, device=device)
        return token_to_req, valid
    token_ids = torch.arange(num_tokens, device=device)
    valid = token_ids < query_start_loc[-1]
    token_to_req = torch.searchsorted(query_start_loc[1:], token_ids, right=True).to(
        torch.int32
    )
    return token_to_req.masked_fill(~valid, _NO_WRITE), valid


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
    seq_lens: torch.Tensor | None = None,
) -> None:
    """Gather live rows into ``out[:bs]``; unused pages and pad rows are ``-1``.

    SGLang's paged allocator reserves physical page 0 for dummy writes.
    Translate it to Native QSA's no-write sentinel in every column, then
    mask the unallocated tail using the request's context length.
    """
    req_to_token = pool.req_to_token
    bs = int(req_pool_indices.shape[0])
    live = max(min(int(live_bs), bs), 0)
    max_blocks = max(1, (int(table_tokens) + block_size - 1) // block_size)
    pages = min(max_blocks, int(out.shape[1]))
    out[:bs].fill_(_NO_WRITE)
    if live == 0 or pages == 0:
        return
    gathered = req_to_token[req_pool_indices[:live], : pages * block_size : block_size]
    gathered = gathered.to(dtype=out.dtype) // block_size
    gathered.masked_fill_(gathered <= 0, _NO_WRITE)
    copy_pages = min(int(gathered.shape[1]), pages)
    out[:live, :copy_pages].copy_(gathered[:, :copy_pages])
    if seq_lens is not None and live > 0 and copy_pages > 0:
        n_pages = (
            (seq_lens[:live].to(dtype=torch.int64) + (block_size - 1)) // block_size
        ).clamp(min=0, max=copy_pages)
        page_ids = torch.arange(copy_pages, device=out.device, dtype=torch.int64)
        invalid = page_ids.unsqueeze(0) >= n_pages.unsqueeze(1)
        out[:live, :copy_pages].masked_fill_(invalid, _NO_WRITE)


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
    seq_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    pages = max(1, (int(table_tokens) + block_size - 1) // block_size)
    tables = torch.empty(
        (int(req_pool_indices.shape[0]), pages),
        dtype=torch.int32,
        device=device,
    )
    _fill_block_tables_into(
        tables,
        pool=pool,
        req_pool_indices=req_pool_indices,
        table_tokens=table_tokens,
        block_size=block_size,
        live_bs=live_bs,
        seq_lens=seq_lens,
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


def _host_positive_max(values: Any) -> int | None:
    """Max of a host-side length list. GPU tensors are refused.

    ``Tensor.max().item()`` on a device tensor synchronizes the stream.
    Draft extend calls that right after target verify, so the sync waits
    out the verify graph.
    """
    if isinstance(values, torch.Tensor):
        if values.numel() == 0 or values.device.type != "cpu":
            return None
        return int(values.max().item())
    if values is None:
        return None
    try:
        return max(int(v) for v in values)
    except ValueError:
        return None


def _eager_qsa_max_seq_len(
    forward_batch: Any,
    seq_lens: torch.Tensor,
    *,
    tokens_per_req: int,
    ctx_len: int,
) -> int:
    """Page-table width without reading the GPU ``seq_lens`` copy.

    The graph path already pins this to the context length. Eager draft
    and verify lift lengths on device, so the host prefix plus the
    speculative width is enough and does not drain the previous step.
    """
    host_max = _host_positive_max(getattr(forward_batch, "seq_lens_cpu", None))
    if host_max is not None:
        extra = (
            int(tokens_per_req) if int(tokens_per_req) > 1 or _is_draft_forward() else 0
        )
        return max(host_max + extra, 1)
    if seq_lens.device.type == "cpu" and seq_lens.numel():
        return max(int(seq_lens.max().item()), 1)
    return max(int(ctx_len), 1)


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


def _ensure_draft_qsa(device: torch.device, block_size: int) -> None:
    """Allocate the draft page tables once, at serving capacity."""
    args = _server_args()
    max_bs = 32
    ctx = 1
    if args is not None:
        max_bs = max(int(getattr(args, "max_running_requests", 32) or 32), 1)
        ctx = max(
            int(getattr(args, "context_length", 0) or 0),
            int(getattr(args, "max_model_len", 0) or 0),
            1,
        )
    tokens_per_req = _get_qsa_graph_max_tokens_per_req()
    pages = max(1, (ctx + block_size - 1) // block_size)
    _DRAFT_QSA.allocate_once(
        max_bs=max_bs,
        max_tokens=max_bs * tokens_per_req,
        max_pages=pages,
        device=device,
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
    buffers: _Qwen4ExpDecodeGraphBuffers | None = None,
) -> Qwen4ExpQSAMetadata:
    """Native ``_build_qsa_metadata`` / ``prepare_decode`` for SGLang CUDA graphs.

    GPU packed token→req into the baked tensor, GPU gather of live page-table
    rows, Triton ``qsa_compressed_slots`` in place. Persistent addresses stay
    the ones baked into the CUDA graph.

    Decode is one token per request. TARGET_VERIFY is ``draft_token_num``
    tokens per request; pad-row tokens stay ``_NO_WRITE``.
    """
    device = positions.device
    tokens_per_req = _get_qsa_tokens_per_req(forward_batch)
    graph = _DECODE_GRAPH if buffers is None else buffers
    assert graph.block_tables is not None
    assert graph.slot_mapping is not None
    assert graph.compressed_slot_mapping is not None
    assert graph.logical_positions is not None
    assert graph.seq_lens is not None
    assert graph.token_to_req is not None

    table_tokens = _clamp_table_tokens(pool, max(ctx_len, indexer_budget))

    live_tokens = min(num_tokens, live_bs * tokens_per_req)
    # Native kernels consume the current bucket's views. Initialize its pad
    # tokens; storage beyond those views is not part of this replay.
    slot = graph.slot_mapping[:num_tokens]
    slot.fill_(_NO_WRITE)
    out_loc = getattr(forward_batch, "out_cache_loc", None)
    if torch.is_tensor(out_loc) and live_tokens:
        loc = out_loc.reshape(-1)[:live_tokens]
        if loc.dtype != slot.dtype:
            loc = loc.to(dtype=slot.dtype)
        slot[: loc.numel()].copy_(loc)

    logical = graph.logical_positions[:num_tokens]
    logical.fill_(_NO_WRITE)
    pos = _sequence_index_positions(positions, num_tokens)
    copy_pos = min(int(pos.numel()), live_tokens)
    if copy_pos:
        logical[:copy_pos].copy_(pos[:copy_pos])

    seq = graph.seq_lens[:bs]
    src_seq = getattr(forward_batch, "seq_lens", None)
    if torch.is_tensor(src_seq):
        src = src_seq[:bs]
        if src.dtype != seq.dtype or src.device != device:
            src = src.to(device=device, dtype=seq.dtype)
        seq.copy_(src)
    else:
        seq.fill_(1)
    if live_bs < bs:
        seq[live_bs:] = 0
    # Target decode is pos = seq_lens - 1; the lift is a no-op and a wasted
    # kernel on the captured graph. Draft decode is also 1 token/req, but its
    # position sits at seq_lens, so QSA must see max(pos)+1. Verify and
    # draft-extend are wider than 1.
    if tokens_per_req > 1 or _is_draft_forward():
        _lift_qsa_seq_lens(seq, logical, live_bs=live_bs, tokens_per_req=tokens_per_req)
    # Page tables after the seq_lens lift: draft slots that cross a page boundary must
    # stay visible, and mixed-length tails must not alias page 0.
    _fill_block_tables_into(
        graph.block_tables,
        pool=pool,
        # Gather runs outside the graph and reads only live requests. The
        # source indices need neither a persistent copy nor pad-row writes.
        req_pool_indices=forward_batch.req_pool_indices[:bs],
        table_tokens=table_tokens,
        block_size=block_size,
        live_bs=live_bs,
        seq_lens=seq,
    )

    _token_to_req_from_packed(
        live_bs=live_bs,
        tokens_per_req=tokens_per_req,
        num_tokens=num_tokens,
        device=device,
        out=graph.token_to_req,
    )

    _compressed_slots_native(
        slot,
        logical,
        compress_ratio,
        graph.compressed_slot_mapping[:num_tokens],
    )
    # Same constant as Native prepare_decode: the scored width is the engine
    # context, not the live sequence. Draft stays eager, matching Native mRoPE.
    max_seq_len = max(ctx_len, indexer_budget, 1)
    return graph.view_qsa(bs=bs, num_tokens=num_tokens, max_seq_len=max_seq_len)


def build_qsa_metadata(
    atom_config: Any,
    forward_batch: Any,
    positions: torch.Tensor,
) -> Qwen4ExpQSAMetadata | None:
    pool = _req_to_token_pool(forward_batch)

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
    # Draft metadata used to take the eager path: one host sync of seq_lens
    # and a fresh page-table allocation on every step. That drains the GPU
    # between the graphed verify and each eager draft, which is most of the
    # small-batch MTP gap versus Native. Keep a private buffer set instead.
    if _is_draft_forward():
        mode = getattr(forward_batch, "forward_mode", None)
        packed = bool(
            mode is not None
            and (mode.is_decode_or_idle() or _is_verify_or_draft_extend(forward_batch))
        )
        if packed:
            _ensure_draft_qsa(device, block_size)
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
                buffers=_DRAFT_QSA,
            )
    if _use_decode_graph_buffers(forward_batch):
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

    seq_lens = _seq_lens(forward_batch, device)
    tokens_per_req = _get_qsa_tokens_per_req(forward_batch)
    slot_mapping = _eager_slot_mapping(forward_batch, num_tokens, device)
    mode = getattr(forward_batch, "forward_mode", None)
    packed_step = bool(
        mode is not None
        and (mode.is_decode_or_idle() or _is_verify_or_draft_extend(forward_batch))
    )
    if packed_step:
        token_to_req = _token_to_req_from_packed(
            live_bs=live_bs,
            tokens_per_req=tokens_per_req,
            num_tokens=num_tokens,
            device=device,
        )
        logical = _qsa_logical_positions(positions, num_tokens, token_to_req >= 0)
        if live_bs < bs:
            pad_from = min(num_tokens, live_bs * tokens_per_req)
            if pad_from < num_tokens:
                slot_mapping[pad_from:] = _NO_WRITE
        # Target decode is pos = seq_lens - 1 (no-op). Draft decode is tpr=1
        # but sits at seq_lens; verify / draft-extend are tpr > 1.
        if tokens_per_req > 1 or _is_draft_forward():
            _lift_qsa_seq_lens(
                seq_lens, logical, live_bs=live_bs, tokens_per_req=tokens_per_req
            )
    else:
        query_start_loc = _query_start_loc(forward_batch, num_tokens, device)
        token_to_req, valid = _token_to_req_from_query_start_loc(
            query_start_loc, num_tokens, device
        )
        logical = _qsa_logical_positions(positions, num_tokens, valid)
        slot_mapping = slot_mapping.masked_fill(~valid, _NO_WRITE)
    # Decode-graph capture may pin width to context. Eager steps use the
    # host prefix; do not synchronize the GPU seq_lens copy.
    if _is_capturing() or seq_lens.numel() == 0:
        max_seq_len = ctx_len
    else:
        max_seq_len = _eager_qsa_max_seq_len(
            forward_batch,
            seq_lens,
            tokens_per_req=tokens_per_req,
            ctx_len=ctx_len,
        )
    table_tokens = _clamp_table_tokens(pool, max(max_seq_len, indexer_budget))
    block_tables = _eager_block_tables(
        pool=pool,
        req_pool_indices=forward_batch.req_pool_indices[:bs],
        table_tokens=table_tokens,
        block_size=block_size,
        live_bs=live_bs,
        device=device,
        seq_lens=seq_lens,
    )

    compressed = torch.empty_like(slot_mapping)
    _compressed_slots_native(slot_mapping, logical, compress_ratio, compressed)
    qsa = Qwen4ExpQSAMetadata(
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        compressed_slot_mapping=compressed,
        token_to_req=token_to_req,
        logical_positions=logical,
        seq_lens=seq_lens,
        max_seq_len=max(max_seq_len, 1),
    )
    return qsa


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
    # Host indices only. A GPU ``idx.max().item()`` drains verify, which is
    # still queued when eager draft builds PLE state. Pool capacity above
    # already matches Native's whole-pool allocation.
    if (
        torch.is_tensor(idx)
        and idx.numel()
        and not _is_capturing()
        and idx.device.type == "cpu"
    ):
        slots = max(slots, int(idx.max().item()) + 1)
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
    accepted = getattr(gdn_metadata, "num_accepted_tokens", None)
    if accepted is None and _is_target_verify(forward_batch):
        # Eager verify used ones; reuse the persistent ones-buffer so capture
        # does not allocate a fresh tensor whose address dies on replay.
        if _DECODE_GRAPH.num_accepted_tokens is None:
            raise RuntimeError(
                "Flash PLE graph buffers must be allocated before TARGET_VERIFY"
            )
        accepted = _DECODE_GRAPH.num_accepted_tokens[:bs]
    ple = _DECODE_GRAPH.bind_graph_ple(
        query_start_loc=query_start_loc,
        ngram_state=ngram_state,
        state_indices=idx,
        conv_state=conv_state,
        batch_size=bs,
        num_accepted_tokens=accepted,
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
    if _use_decode_graph_buffers(forward_batch):
        return _build_decode_graph_ple_metadata(
            atom_config, forward_batch, model=model, gdn_metadata=gdn_metadata
        )
    idx = getattr(gdn_metadata, "non_spec_state_indices_tensor", None)
    idx_in = getattr(gdn_metadata, "non_spec_state_indices_in_tensor", None)
    accepted = getattr(gdn_metadata, "num_accepted_tokens", None)
    if gdn_metadata is None:
        if not _is_target_verify(forward_batch):
            return None
        slots = _linear_static_ple_slots(forward_batch, None)
        if slots is None:
            return None
        _, idx = slots
        idx_in = idx
        accepted = torch.ones(
            (int(forward_batch.batch_size),), dtype=torch.int32, device=idx.device
        )
    if idx is None:
        return None

    device = positions.device
    bs = int(forward_batch.batch_size)
    num_tokens = _num_tokens_from_positions(positions)
    query_start_loc = _query_start_loc(forward_batch, num_tokens, device)
    is_prefill = bool(
        forward_batch.forward_mode.is_extend()
    ) and not _is_verify_or_draft_extend(forward_batch)

    live_bs = real_batch_size(forward_batch)
    idx = idx[:bs].to(device=device, dtype=torch.int32)
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
        num_accepted_tokens=accepted,
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
    is_draft = getattr(hf, "model_type", None) == "qwen4_exp_mtp"
    if is_draft and len(qsa_layers) != 1:
        raise ValueError("Flash MTP requires exactly one QSA draft layer")

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

    getter = getattr(pool, "get_kv_buffer", None)
    if not callable(getter):
        raise TypeError("Flash QSA requires a SGLang KV pool with get_kv_buffer")

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
    pool_shape = None
    for i, layer in enumerate(qsa_layers):
        # Native draft layer 48 belongs to a separate, single-layer SGLang
        # pool. Target layers retain their logical ids; the pool owns their
        # full_attention_layer_id_mapping. Never guess another target layer.
        layer_id = 0 if is_draft else int(layer.layer_num)
        try:
            k_buf, v_buf = getter(layer_id)
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Flash QSA KV mapping missing for layer {layer_id} "
                f"(draft={is_draft}, pool={type(pool).__name__})"
            ) from exc
        if (
            not torch.is_tensor(k_buf)
            or not torch.is_tensor(v_buf)
            or k_buf.ndim != 3
            or v_buf.shape != k_buf.shape
            or k_buf.shape[0] == 0
            or k_buf.shape[0] % block_size
        ):
            raise ValueError(
                f"Flash QSA layer {layer_id} requires matching paged K/V "
                f"buffers [tokens, heads, dim], with tokens divisible by {block_size}"
            )
        tokens, heads, dim = k_buf.shape
        pages = tokens // block_size
        k_cache = k_buf.view(pages, block_size, heads, dim)
        v_cache = v_buf.view(pages, block_size, heads, dim)
        pool_shape = tuple(k_buf.shape)
        layer.bind_caches(k_cache, v_cache, raw[i], compressed[i], None)
    model._atom_qwen4_exp_qsa_raw = raw
    model._atom_qwen4_exp_qsa_compressed = compressed
    model._atom_qwen4_exp_qsa_bound = True
    logger.info(
        "Bound %s QSA layers pages=%s block=%s compress=%s "
        "pool_shape=%s (Native layout = paged view of [T,H,D])",
        len(qsa_layers),
        num_pages,
        block_size,
        compress_ratio,
        pool_shape,
    )


def _get_cuda_graph_max_bs() -> int:
    """Use SGLang's resolved decode buckets; older versions use CLI fields."""
    args = _server_args()
    decode = getattr(getattr(args, "cuda_graph_config", None), "decode", None)
    buckets = getattr(decode, "bs", None)
    if buckets:
        graph_bs = max(buckets)
    elif getattr(decode, "max_bs", None) is not None:
        graph_bs = int(decode.max_bs)
    else:
        graph_bs = int(
            getattr(args, "cuda_graph_max_bs_decode", None)
            or getattr(args, "cuda_graph_max_bs", 0)
            or 0
        )
    running = int(getattr(args, "max_running_requests", 0) or 0)
    if running > 0:
        graph_bs = min(graph_bs, running) if graph_bs > 0 else running
    return graph_bs


def _decode_graph_capacity(
    atom_config: Any, forward_batch: Any
) -> tuple[int, int, int, torch.device]:
    """High-water sizes so capture never reallocates persistent QSA buffers.

    Returns ``(max_bs, max_tokens, max_pages, device)``. ``max_tokens`` is
    ``max_bs *`` CLI MTP width, not the current-step fill width.
    """
    device = getattr(forward_batch, "device", None)
    if device is None:
        pos = getattr(forward_batch, "positions", None)
        device = pos.device if torch.is_tensor(pos) else torch.device("cuda")
    bs = int(getattr(forward_batch, "batch_size", 0) or 0)
    graph_bs = _get_cuda_graph_max_bs()
    max_bs = max(bs, graph_bs, 8)
    max_tokens_per_req = _get_qsa_graph_max_tokens_per_req()
    max_tokens = max_bs * max_tokens_per_req
    indexer_budget = _indexer_budget(atom_config)
    block_size = _block_size(forward_batch, atom_config)
    # Page table must cover the full context, not just indexer_budget.
    # QSA top-k still keeps only ``indexer_budget`` tokens, but those tokens
    # can sit anywhere in a 12k sequence; a 32-page (2k) table would clamp
    # logical_page and drop the recent context.
    ctx_len = _serving_context_len(indexer_budget)
    max_pages = max((ctx_len + block_size - 1) // block_size, 1)
    return max_bs, max_tokens, max_pages, torch.device(device)


def _ensure_decode_graph_buffers(atom_config: Any, forward_batch: Any) -> None:
    """Allocate persistent QSA CUDA-graph buffers at CLI high-water, once."""
    max_bs, max_tokens, max_pages, device = _decode_graph_capacity(
        atom_config, forward_batch
    )
    _DECODE_GRAPH.allocate_once(
        max_bs=max_bs,
        max_tokens=max_tokens,
        max_pages=max_pages,
        device=device,
    )


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
    if mode is None:
        return None
    is_verify = _is_target_verify(forward_batch)
    if not (mode.is_decode_or_idle() or is_verify):
        return None
    if atom_config is None:
        try:
            from atom.config import get_current_atom_config

            atom_config = get_current_atom_config()
        except Exception:  # noqa: BLE001
            return None
    if atom_config is None or not _is_qwen4_exp_config(atom_config):
        return None

    tokens_per_req = _get_qsa_tokens_per_req(forward_batch)
    positions = getattr(forward_batch, "positions", None)
    if not torch.is_tensor(positions):
        bs = int(getattr(forward_batch, "batch_size", 0) or 0)
        device = getattr(forward_batch, "device", None) or torch.device("cuda")
        positions = torch.zeros(
            (max(bs, 1) * tokens_per_req,), dtype=torch.int64, device=device
        )

    # Allocation activates the graph-buffer path for capture and replay.
    _ensure_decode_graph_buffers(atom_config, forward_batch)
    block_size = _block_size(forward_batch, atom_config)

    qsa = build_qsa_metadata(atom_config, forward_batch, positions)
    # QSA page tables only. PLE aliases Hybrid GDN's static mamba_cache_indices
    # during warmup/capture forward (after this out_graph returns and the
    # linear child has filled those slots). Replay does not rebuild PLE.
    if in_capture:
        logger.info(
            "Flash %s CUDA-graph QSA buffers ready: "
            "block_size=%s bs<=%s tokens<=%s tokens_per_req=%s pages<=%s max_seq_len=%s",
            "target_verify" if is_verify else "decode",
            block_size,
            _DECODE_GRAPH.max_bs,
            _DECODE_GRAPH.max_tokens,
            tokens_per_req,
            _DECODE_GRAPH.max_pages,
            qsa.max_seq_len,
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
