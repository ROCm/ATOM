# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang ForwardBatch → Native Qwen3.8-Flash-Next QSA / PLE metadata.

Compute stays in Native ATOM (#2048): QSA, indexer, GDN, hyper-connection, PLE.
This module only translates the current step's page tables into the structs
those kernels already read. Geometry is never cached across steps.

First knife: eager text. No MTP, VLM, radix, speculative, or CUDA-graph
alignment. PLE table stays in Native; do not enable SGLang --ple-offload-embedding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from atom.plugin.sglang.attention_backend.backend_resolver import (
    resolve_attn_backend,
    resolve_mamba_req_pool,
)
from atom.plugin.sglang.deepseek_v4_bridge import _build_block_tables

logger = logging.getLogger(__name__)

# Dummy / graph-padding rows must not write QSA caches. Native kernels treat -1
# as no-write (same idea as the Qwen3.5 SGLang sentinel in #2067).
_NO_WRITE = -1


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
    for candidate in (
        getattr(forward_batch, "page_size", None),
        getattr(getattr(forward_batch, "token_to_kv_pool", None), "page_size", None),
        getattr(atom_config, "kv_cache_block_size", None),
        getattr(atom_config, "block_size", None),
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
    if torch.is_tensor(seq):
        return seq.to(device=device, dtype=torch.int32)
    return torch.ones(
        (int(forward_batch.batch_size),), dtype=torch.int32, device=device
    )


def _query_start_loc(forward_batch: Any, num_tokens: int, device: torch.device) -> torch.Tensor:
    mode = forward_batch.forward_mode
    bs = int(forward_batch.batch_size)
    if mode.is_decode_or_idle():
        return torch.arange(0, bs + 1, dtype=torch.int32, device=device)
    if mode.is_extend():
        loc = torch.empty((bs + 1,), dtype=torch.int32, device=device)
        loc[:bs] = forward_batch.extend_start_loc.to(dtype=torch.int32)
        loc[bs] = (
            forward_batch.extend_start_loc[-1] + forward_batch.extend_seq_lens[-1]
        ).to(dtype=torch.int32)
        return loc
    return torch.tensor([0, num_tokens], dtype=torch.int32, device=device)


def _token_to_req_and_logical(
    *,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_tokens: int,
    mapped_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = positions.device
    token_to_req = torch.zeros((num_tokens,), dtype=torch.int32, device=device)
    logical = torch.full((num_tokens,), _NO_WRITE, dtype=torch.int64, device=device)
    if mapped_tokens <= 0:
        return token_to_req, logical
    starts = query_start_loc[:-1]
    ends = query_start_loc[1:]
    for req in range(int(query_start_loc.numel()) - 1):
        lo = int(starts[req].item())
        hi = int(ends[req].item())
        hi = min(hi, mapped_tokens)
        if hi > lo:
            token_to_req[lo:hi] = req
            logical[lo:hi] = positions.reshape(-1)[lo:hi].to(torch.int64)
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

    seq_lens = _seq_lens(forward_batch, device)[:bs]
    max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() else 1
    req_pool_indices = forward_batch.req_pool_indices[:bs]
    # Native QSA requires page_table width * compressed_rows >= indexer_budget
    # groups even on short warmup sequences. SGLang's current-seq table is too
    # narrow (one 64-token page => 16 compressed slots vs 512 top-k).
    hf = _hf_text_config(atom_config)
    indexer_budget = int(getattr(hf, "indexer_budget", 2048) or 2048)
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
    mapped = int(query_start_loc[-1].item()) if query_start_loc.numel() else 0
    mapped = min(mapped, num_tokens)
    token_to_req, logical = _token_to_req_and_logical(
        query_start_loc=query_start_loc,
        positions=positions,
        num_tokens=num_tokens,
        mapped_tokens=mapped,
    )
    if mapped < num_tokens:
        slot_mapping[mapped:] = _NO_WRITE

    compressed = _compressed_slot_mapping(
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        token_to_req=token_to_req,
        logical_positions=logical,
        block_size=block_size,
        compress_ratio=compress_ratio,
    )
    return Qwen3_8FlashNextQSAMetadata(
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        compressed_slot_mapping=compressed,
        token_to_req=token_to_req,
        logical_positions=logical,
        seq_lens=seq_lens,
        max_seq_len=max(max_seq_len, 1),
    )


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
    device = next(model.parameters()).device
    dtype = getattr(atom_config, "torch_dtype", None) or torch.bfloat16
    state = torch.zeros(
        (max(num_slots, 1), channels, state_len), dtype=dtype, device=device
    )
    model._atom_ple_conv_state = state
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

    idx = idx[:bs].to(device=device, dtype=torch.int32)
    idx_in = getattr(gdn_metadata, "non_spec_state_indices_in_tensor", None)
    if idx_in is None:
        idx_in = idx
    else:
        idx_in = idx_in[:bs].to(device=device, dtype=torch.int32)

    conv_state = _ensure_ple_conv_state(
        model, atom_config, int(idx.max().item()) + 1 if idx.numel() else 1
    )
    prefix = getattr(forward_batch, "extend_prefix_lens", None)
    if is_prefill and torch.is_tensor(prefix):
        has_initial = prefix[:bs] > 0
    else:
        has_initial = torch.ones((bs,), dtype=torch.bool, device=device)

    return Qwen3_8FlashNextPLEMetadata(
        query_start_loc=query_start_loc,
        ngram_context=context,
        state_indices_in=idx_in,
        state_indices_out=idx,
        has_initial_state=has_initial,
        conv_state=conv_state,
        num_reqs=bs,
        is_prefill=is_prefill,
        max_query_len=int((query_start_loc[1:] - query_start_loc[:-1]).max().item())
        if query_start_loc.numel() > 1
        else 1,
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
    """Bind main K/V plus indexer raw/compressed keys on the same block table.

    Main K/V views come from SGLang's token pool when present. Indexer caches
    are plugin-owned BF16 tensors sized to that pool's page count.
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
        # SGLang MHATokenToKVPool.size is token capacity, not page count.
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
    for i, layer in enumerate(qsa_layers):
        k_cache = getattr(layer, "k_cache", None)
        v_cache = getattr(layer, "v_cache", None)
        getter = getattr(pool, "get_kv_buffer", None) if pool is not None else None
        if callable(getter):
            try:
                k_buf, v_buf = getter(int(getattr(layer, "layer_num", i)))
            except Exception:  # noqa: BLE001 - pool layout varies by SGLang version
                k_buf = v_buf = None
            if torch.is_tensor(k_buf) and torch.is_tensor(v_buf) and k_buf.dim() == 3:
                tokens, heads, dim = k_buf.shape
                pages = tokens // block_size
                if pages > 0 and tokens == pages * block_size:
                    k_cache = k_buf.view(pages, block_size, heads, dim)
                    v_cache = v_buf.view(pages, block_size, heads, dim)
                    used_pool += 1
        if k_cache is None:
            logger.warning(
                "Flash QSA layer %s: no SGLang KV buffer; allocating zeros",
                getattr(layer, "layer_num", i),
            )
            k_cache = torch.zeros(
                (num_pages, block_size, kv_heads, head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            v_cache = torch.zeros_like(k_cache)
        layer.bind_caches(
            k_cache,
            v_cache,
            raw[i],
            compressed[i],
            None,
        )
    model._atom_flash_qsa_raw = raw
    model._atom_flash_qsa_compressed = compressed
    model._atom_flash_qsa_bound = True
    logger.info(
        "Bound %s QSA layers to %s pages (block=%s, compress=%s); "
        "KV dtype=bf16; sglang_pool=%s/%s",
        len(qsa_layers),
        num_pages,
        block_size,
        compress_ratio,
        used_pool,
        len(qsa_layers),
    )


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
