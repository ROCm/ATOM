"""Native ATOM attention bridge for the Kimi-K3 DSpark draft on sglang-main.

The draft runs ATOM's ``KimiK3DSpark`` backbone so its MLA attention uses ATOM
kernels (parity with the target and with ATOM core / ATOM-vLLM). This module
translates sglang-main's DSpark block ``ForwardBatch`` into the *same* ATOM
``AttentionMetaData`` that ATOM core's ``DSparkProposer._propose_with_draft``
builds, and binds ATOM's per-layer ``mla_attn.kv_cache`` to the sibling bf16 MLA
KV pool that the DSpark draft model runner owns.

Design note (causality): ATOM does NOT use a special non-causal kernel. The
block attends over its whole self simply because the metadata makes every one of
the ``T`` per-request queries see the full ``context_len = prefix + T`` KV span
(the block's own ``T`` rows are written into the paged cache and counted in
``context_lens``). We mirror that exactly; there is no bespoke non-causal path.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import torch

logger = logging.getLogger(__name__)


_DSPARK_FRONTEND_CLS = None


def _dspark_attention_frontend_cls():
    """Native K3 MLA frontend for the draft, exposing ``kv_cache`` (resolved
    lazily to avoid importing the target bridge at module import time)."""
    global _DSPARK_FRONTEND_CLS
    if _DSPARK_FRONTEND_CLS is None:
        from atom.plugin.sglang.kimi_k3_bridge import SGLangATOMKimiK3Attention

        class _Frontend(SGLangATOMKimiK3Attention):
            @property
            def kv_cache(self):
                return self.impl.kv_cache

            @kv_cache.setter
            def kv_cache(self, value):
                self.impl.kv_cache = value

        _DSPARK_FRONTEND_CLS = _Frontend
    return _DSPARK_FRONTEND_CLS


@contextmanager
def kimi_k3_dspark_native_attention_construction():
    """Construct the DSpark draft's MLA attention as native ATOM MLA.

    In plugin mode ``atom.model_ops.base_attention.Attention`` resolves to the
    SGLang ``AttentionForSGLang`` frontend, which does not expose the native MLA
    surface (``impl``/``kv_cache``/``layer_num``, ``impl._pcp_write_full_kv``)
    that ``K3DSparkMLAAttention`` needs. Mirror the target's approach: swap the
    draft model's ``Attention`` symbol to a native MLA frontend for the duration
    of construction. The frontend is the target's ``SGLangATOMKimiK3Attention``
    (same 576-wide MLA, same ``(q_c, kv_c, k_pe, positions)`` call convention),
    extended with a ``kv_cache`` accessor the draft's ``write_context_kv`` uses.
    """
    from atom.models import kimi_k3_dspark

    previous = kimi_k3_dspark.Attention
    kimi_k3_dspark.Attention = _dspark_attention_frontend_cls()
    try:
        yield
    finally:
        kimi_k3_dspark.Attention = previous


def _iter_draft_mla_attn(model: Any):
    from atom.models.kimi_k3_dspark import K3DSparkMLAAttention

    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, K3DSparkMLAAttention):
            yield attn


def _resolve_draft_pools(forward_batch: Any):
    """Resolve the DSpark draft's sibling token/req KV pools for this forward."""
    token_pool = getattr(forward_batch, "token_to_kv_pool", None)
    req_pool = getattr(forward_batch, "req_to_token_pool", None)
    if token_pool is None or req_pool is None:
        try:
            from sglang.srt.model_executor.forward_context import (
                get_attn_backend,
                has_forward_context,
            )

            backend = get_attn_backend() if has_forward_context() else None
        except Exception:  # noqa: BLE001 - forward context optional
            backend = None
        if backend is not None:
            token_pool = token_pool or getattr(backend, "token_to_kv_pool", None)
            req_pool = req_pool or getattr(backend, "req_to_token_pool", None)
    return token_pool, req_pool


def _bind_sibling_pool(model: Any, token_pool: Any) -> None:
    """Bind each draft layer's ATOM MLA cache to the sibling bf16 pool.

    In plugin mode ``KimiK3DSpark`` is built with ``layer_offset=0`` (see
    prepare_model), so ``mla_attn.layer_num`` is the draft-local id 0..N-1 that
    sglang keyed the draft KV pool by -- bind directly, no offset remap.
    """
    if token_pool is None or not hasattr(token_pool, "get_kv_buffer"):
        raise RuntimeError("Kimi-K3 DSpark draft pool has no get_kv_buffer()")

    from atom.config import KVCacheTensor
    from atom.utils.forward_context import get_forward_context, set_kv_cache_data

    kv_cache_data = dict(getattr(get_forward_context(), "kv_cache_data", None) or {})
    for attn in _iter_draft_mla_attn(model):
        layer_num = int(attn.mla_attn.layer_num)
        k_buffer, _ = token_pool.get_kv_buffer(layer_num)
        # Expect [slots, kv_heads>=1, 576] MLA latent; use the first lane.
        k_cache = k_buffer[:, :1, :] if k_buffer.ndim == 3 else k_buffer
        attn.mla_attn.kv_cache = k_cache
        kv_cache_data[f"layer_{layer_num}"] = KVCacheTensor(
            layer_num=layer_num,
            k_cache=k_cache,
            v_cache=None,
            k_scale=None,
            v_scale=None,
        )
    set_kv_cache_data(kv_cache_data)
    get_forward_context().kv_cache_data = kv_cache_data


def _block_width(forward_batch: Any, bs: int) -> int:
    spec_info = getattr(forward_batch, "spec_info", None)
    gamma = int(getattr(spec_info, "draft_token_num", 0) or 0)
    if gamma > 0:
        return gamma
    total = int(forward_batch.input_ids.shape[0])
    if bs > 0 and total % bs == 0:
        return total // bs
    raise RuntimeError(
        f"Kimi-K3 DSpark cannot resolve block width (total={total}, bs={bs})."
    )


def _build_block_metadata(
    forward_batch: Any,
    positions: torch.Tensor,
    *,
    token_pool: Any,
    req_pool: Any,
):
    """Build the ATOM block ``AttentionMetaData``, mirroring ATOM core
    ``DSparkProposer._propose_with_draft`` step 2."""
    from atom.plugin.sglang.kimi_k3_bridge import _build_block_table
    from atom.utils.block_convert import kv_indices_generate_triton
    from atom.utils.forward_context import AttentionMetaData, AttnState

    device = positions.device
    page_size = int(token_pool.page_size)
    bs = int(forward_batch.batch_size)
    T = _block_width(forward_batch, bs)

    # prefix length per request (anchor + 1); the block adds T rows on top.
    prefix_lens = forward_batch.seq_lens[:bs].to(dtype=torch.int32)
    context_lens = (prefix_lens + T).to(dtype=torch.int32)

    # Every request contributes exactly T query rows -> constant ramp.
    cu_seqlens_q = torch.arange(
        0, (bs + 1) * T, step=T, dtype=torch.int32, device=device
    )
    # The block's T KV rows live at out_cache_loc; ATOM MLA writes them there and
    # context_lens counts them, so each query sees the whole block.
    slot_mapping = forward_batch.out_cache_loc[: bs * T]

    # During CUDA-graph capture host<->device syncs (``.item()``) are illegal, so
    # use static upper bounds (the pool's max context width). The MLA kernel is
    # driven by the per-request ``context_lens``/``kv_indptr`` counts, so an
    # over-estimated launch bound and an over-sized ``kv_indices`` are safe; the
    # eager path keeps exact values to avoid wasted work.
    capturing = torch.cuda.is_current_stream_capturing()
    static_max_k = int(req_pool.req_to_token.shape[1])
    if capturing:
        max_seqlen_k = static_max_k
        total_kv = bs * static_max_k
    else:
        max_seqlen_k = int(context_lens.max().item()) if bs else 0
        total_kv = int(context_lens.sum().item()) if bs else 0
    block_tables = _build_block_table(
        forward_batch,
        req_pool,
        seq_lens=context_lens,
        extend_lens=None,
        page_size=page_size,
        max_seq_len=max_seqlen_k,
    )

    metadata = AttentionMetaData(
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=T,
        max_seqlen_k=max_seqlen_k,
        min_seqlen_q=T,
        total_kv=total_kv,
        has_cached=False,
        dropout_p=0.0,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        state=AttnState.DECODE,
    )
    # The draft owns a bf16 sibling pool; use bf16 Q regardless of the target's
    # fp8 KV dtype (matches ATOM K3DSparkMLAAttention kv_cache_dtype="bf16").
    metadata.dtype_q = torch.bfloat16

    # CSR KV layout over prefix ++ block, generated exactly like ATOM core.
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1:].copy_(torch.cumsum(context_lens, dim=0))
    # ``kv_indices`` size = sum(context_lens); during capture that value cannot be
    # read host-side, so allocate the static upper bound (bs * max context). The
    # triton generator only writes the real ``kv_indptr``-counted entries.
    kv_indices_len = (
        bs * static_max_k if capturing else (int(kv_indptr[-1].item()) if bs else 0)
    )
    kv_indices = torch.zeros(kv_indices_len, dtype=torch.int32, device=device)
    kv_indices_generate_triton(
        block_tables, kv_indices, kv_indptr, page_size, max_seqlen_k
    )
    metadata.kv_indptr = kv_indptr
    metadata.kv_indices = kv_indices
    metadata.kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)
    # Pin every capture-time transient (all held by ``metadata``) so a later
    # smaller-bs capture cannot reuse their graph-mempool memory and clobber the
    # first-captured (max-bs) draft graph on replay.
    from atom.plugin.sglang.kimi_k3_spec_verify import keepalive_if_capturing

    keepalive_if_capturing(metadata)
    return metadata


def bind_kimi_k3_dspark_cache_views(model: Any, runtime: Any) -> None:
    """Bind the sibling bf16 MLA pool and publish the block metadata.

    Called from ``K3DSparkModel.forward`` inside the ATOM forward-context scope.
    """
    forward_batch = runtime.forward_batch
    if getattr(forward_batch.forward_mode, "is_idle", lambda: False)():
        return

    token_pool, req_pool = _resolve_draft_pools(forward_batch)
    if token_pool is None or req_pool is None:
        raise RuntimeError(
            "Kimi-K3 DSpark draft KV pools are not available "
            f"(token_pool={token_pool!r}, req_pool={req_pool!r})."
        )

    _bind_sibling_pool(model, token_pool)

    from atom.utils.forward_context import get_forward_context

    metadata = _build_block_metadata(
        forward_batch,
        runtime.positions,
        token_pool=token_pool,
        req_pool=req_pool,
    )
    ctx = get_forward_context()
    ctx.attn_metadata = metadata
    # The block pass is always decode-shaped (T queries per request), even on a
    # step where the target just prefilled. Force the decode branch, matching
    # ATOM core (DSparkProposer sets context.is_prefill = False).
    if getattr(ctx, "context", None) is not None:
        ctx.context.is_prefill = False
        ctx.context.is_draft = True


def write_kimi_k3_dspark_target_hidden_kv(
    model: Any,
    *,
    target_hidden: torch.Tensor | None = None,
    main_hidden: torch.Tensor | None = None,
    pool: Any = None,
    positions: torch.Tensor | None = None,
    cache_loc: torch.Tensor | None = None,
    swa_loc: torch.Tensor | None = None,
    cache_loc_2d: torch.Tensor | None = None,
    commit_lens: torch.Tensor | None = None,
    **_: Any,
) -> None:
    """Project verified target hidden states into the draft's sibling KV cache.

    Adapts sglang-main's ``TargetHiddenKvInjector`` call (either
    ``main_hidden``/``swa_loc`` or ``target_hidden``/``cache_loc``) to ATOM's
    ``KimiK3DSpark.write_context_kv(aux_concat, positions, slot_mapping)`` --
    identical semantics to ATOM core ``_propose_with_draft`` step 1.
    """
    hidden = target_hidden if target_hidden is not None else main_hidden
    slots = cache_loc if cache_loc is not None else swa_loc
    if hidden is None or slots is None or positions is None:
        raise RuntimeError(
            "write_kimi_k3_dspark_target_hidden_kv missing tensors "
            f"(hidden={hidden is not None}, slots={slots is not None}, "
            f"positions={positions is not None})."
        )
    n = int(positions.shape[0])
    hidden = hidden[:n]
    slots = slots.to(dtype=torch.int64)[:n]
    positions = positions.to(dtype=torch.int64)[:n]

    # The injector runs OUTSIDE any ATOM forward context (the target verify
    # context is already torn down), but KimiK3DSpark.write_context_kv reads
    # get_forward_context().context.is_dummy_run and writes through the draft's
    # bound MLA cache. Bind the sibling pool and establish a minimal decode
    # context here before the write.
    from atom.config import KVCacheTensor
    from atom.utils.forward_context import (
        Context,
        set_forward_context,
        set_kv_cache_data,
    )

    draft_pool = pool
    kv_cache_data = {}
    for attn in _iter_draft_mla_attn(model):
        layer_num = int(attn.mla_attn.layer_num)
        k_buffer, _ = draft_pool.get_kv_buffer(layer_num)
        k_cache = k_buffer[:, :1, :] if k_buffer.ndim == 3 else k_buffer
        attn.mla_attn.kv_cache = k_cache
        kv_cache_data[f"layer_{layer_num}"] = KVCacheTensor(
            layer_num=layer_num,
            k_cache=k_cache,
            v_cache=None,
            k_scale=None,
            v_scale=None,
        )
    set_kv_cache_data(kv_cache_data)
    atom_config = getattr(model, "atom_config", None)
    set_forward_context(
        attn_metadata=None,
        atom_config=atom_config,
        context=Context(positions=positions, is_dummy_run=False, is_draft=True),
    )
    with torch.inference_mode():
        model.write_context_kv(hidden, positions, slots)
