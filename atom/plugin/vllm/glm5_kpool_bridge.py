"""Forward-context bridge for GLM-5.3 pooled sparse indexing under vLLM."""

from __future__ import annotations

from contextlib import contextmanager
import logging

import torch
from vllm.forward_context import (
    get_forward_context as get_vllm_forward_context,
)

logger = logging.getLogger("atom")

from atom.utils.forward_context import (
    AttentionMetaData,
    AttnState,
    Context,
    reset_forward_context,
    set_forward_context,
)


def _is_stream_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _scheduler_caps(atom_config, positions: torch.Tensor) -> tuple[int, int]:
    vllm_config = getattr(
        getattr(atom_config, "plugin_config", None), "vllm_config", None
    )
    sched = getattr(vllm_config, "scheduler_config", None)
    max_reqs = int(getattr(sched, "max_num_seqs", 0) or 256)
    max_tokens = int(
        getattr(sched, "max_num_batched_tokens", 0) or max(int(positions.numel()), 2048)
    )
    return max(max_reqs, 1), max(max_tokens, int(positions.numel()), 1)


def _metadata_by_shape(metadata: dict):
    indexer = sparse = kda = None
    kda_layer_name = None
    for layer_name, value in metadata.items():
        if value is None:
            continue
        if (
            hasattr(value, "num_decode_tokens")
            and hasattr(value, "num_prefill_tokens")
            and hasattr(value, "query_start_loc")
            and hasattr(value, "slot_mapping")
            and not hasattr(value, "paged_kv_indices")
        ):
            indexer = value
        if (
            hasattr(value, "paged_kv_indices")
            and hasattr(value, "batch_id_per_q_token")
            and hasattr(value, "block_table")
        ):
            sparse = value
        if hasattr(value, "non_spec_state_indices_tensor"):
            kda = value
            kda_layer_name = layer_name
    return indexer, sparse, kda, kda_layer_name


def _ensure_workspaces(model, *, device, max_reqs: int, max_tokens: int, capturing: bool):
    ws = getattr(model, "_glm5_kpool_ws", None)
    need = (
        ws is None
        or int(ws["cu_k"].shape[0]) < max_reqs + 1
        or int(ws["sparse_cu_q"].shape[0]) < max_tokens + 1
    )
    if not need:
        return ws
    if capturing:
        return None
    model._glm5_kpool_ws = {
        "cu_k": torch.zeros(max_reqs + 1, dtype=torch.int32, device=device),
        "query_lens": torch.empty(max_reqs, dtype=torch.int32, device=device),
        "sparse_cu_q": torch.arange(
            max_tokens + 1, dtype=torch.int32, device=device
        ),
    }
    return model._glm5_kpool_ws


def _bind_tail_caches(model, vllm_context, kda_layer_name: str | None, capturing: bool) -> bool:
    if not kda_layer_name:
        return False
    layer = vllm_context.no_compile_layers.get(kda_layer_name)
    kv_cache = getattr(layer, "kv_cache", None)
    if not kv_cache or len(kv_cache) < 1 or kv_cache[0] is None:
        return False
    num_slots = int(kv_cache[0].shape[0])
    if num_slots <= 1:
        return False

    found = False
    for module in model.modules():
        if not hasattr(module, "kpool_tail_cache"):
            continue
        found = True
        current = module.kpool_tail_cache
        expected = (
            num_slots,
            2,
            int(module.index_kpool),
            int(module.head_dim),
        )
        if current is not None and tuple(current.shape) == expected:
            continue
        if capturing:
            return False
        module.kpool_tail_cache = torch.zeros(
            expected,
            dtype=torch.bfloat16,
            device=kv_cache[0].device,
        )
        logger.info("ATOM GLM5 k-pool: bound tail cache with %d state slots", num_slots)
    return found


def _build_metadata(
    indexer, sparse, kda, workspaces, *, index_kpool: int = 4
) -> tuple[AttentionMetaData, bool, int]:
    num_prefills = int(indexer.num_prefills)
    num_decodes = int(indexer.num_decodes)
    # Mixed batch (num_decodes > 0 and num_prefills > 0) arises from chunked
    # prefill. Decode requests each contribute 1 query token against a fully
    # cached KV, which is identical to an "extend" prefill (query_len < seq_len).
    # The kpool prefill path already handles that case via has_cached + cu_seqlens_k,
    # so we route the entire mixed batch through the prefill path with has_cached=True.
    is_mixed = num_prefills > 0 and num_decodes > 0
    is_prefill = num_prefills > 0 or is_mixed
    # Pure decode: num_decodes > 0 and num_prefills == 0
    is_pure_decode = num_decodes > 0 and num_prefills == 0
    if is_mixed:
        # Treat all requests as prefill-with-prefix.
        is_prefill = True
    num_reqs = int(indexer.num_reqs)
    num_tokens = int(indexer.num_actual_tokens)
    seq_lens = indexer.seq_lens[:num_reqs].to(torch.int32)
    cu_k = workspaces["cu_k"][: num_reqs + 1]
    cu_k.zero_()
    torch.cumsum(seq_lens, dim=0, out=cu_k[1:])

    query_start = indexer.query_start_loc[: num_reqs + 1]
    if query_start.dtype != torch.int32:
        query_start = query_start.to(torch.int32)
    query_lens = workspaces["query_lens"][:num_reqs]
    query_lens.copy_(query_start[1 : num_reqs + 1])
    query_lens.sub_(query_start[:num_reqs])

    if is_mixed:
        # Decode requests always have cached context (seq_len > query_len == 1).
        has_cached = True
    elif is_pure_decode:
        has_cached = False  # decode uses its own path; has_cached unused
    else:
        # Host ints only: `.item()` on a device tensor is an illegal D2H under capture.
        has_cached = bool(int(indexer.max_seq_len) > int(indexer.max_query_len))
    state = AttnState.PREFILL_PREFIX if has_cached else AttnState.PREFILL_NATIVE
    if not is_prefill:
        state = AttnState.DECODE

    kpool_total_pools = None
    if is_prefill:
        if is_mixed:
            # Sum complete pools from prefill chunks + decode requests.
            prefill_pools = 0
            if indexer.prefill is not None:
                prefill_pools = sum(
                    int(chunk.kpool_total_pools) for chunk in indexer.prefill.chunks
                )
            # Decode requests: each contributes seq_len // index_kpool complete pools.
            decode_pools = int(
                (seq_lens[:num_decodes].to(torch.int64) // index_kpool).sum().item()
            )
            kpool_total_pools = prefill_pools + decode_pools
        elif indexer.prefill is not None:
            kpool_total_pools = sum(
                int(chunk.kpool_total_pools) for chunk in indexer.prefill.chunks
            )

    metadata = AttentionMetaData(
        cu_seqlens_q=query_start,
        cu_seqlens_k=cu_k,
        max_seqlen_q=int(indexer.max_query_len),
        max_seqlen_k=int(indexer.max_seq_len),
        slot_mapping=indexer.slot_mapping[:num_tokens],
        context_lens=seq_lens,
        block_tables=sparse.block_table[:num_reqs],
        state=state,
        sparse_cu_seqlens_q=workspaces["sparse_cu_q"][: num_tokens + 1],
        sparse_kv_indptr=sparse.paged_kv_indptr[: num_tokens + 1],
        batch_id_per_q_token=sparse.batch_id_per_q_token[:num_tokens],
        has_cached=has_cached,
        kpool_total_pools=kpool_total_pools,
    )
    metadata.gdn_metadata = kda
    metadata.kpool_plugin_mode = True
    return metadata, is_prefill, num_reqs


@contextmanager
def atom_glm5_kpool_forward_context(
    *,
    model,
    atom_config,
    input_ids,
    positions,
):
    capturing = _is_stream_capturing()
    vllm_context = get_vllm_forward_context()
    metadata_dict = vllm_context.attn_metadata
    if not isinstance(metadata_dict, dict):
        metadata_dict = {}
    indexer, sparse, kda, kda_layer_name = _metadata_by_shape(metadata_dict)
    max_reqs, max_tokens = _scheduler_caps(atom_config, positions)
    workspaces = _ensure_workspaces(
        model,
        device=positions.device,
        max_reqs=max_reqs,
        max_tokens=max_tokens,
        capturing=capturing,
    )
    ready = (
        indexer is not None
        and sparse is not None
        and kda is not None
        and workspaces is not None
        and _bind_tail_caches(model, vllm_context, kda_layer_name, capturing)
    )
    if capturing and not getattr(model, "_glm5_kpool_capture_logged", False):
        logger.warning(
            "ATOM GLM5 k-pool capture: ready=%s indexer=%s sparse=%s "
            "kda=%s workspaces=%s tail_bound=%s",
            ready,
            indexer is not None,
            sparse is not None,
            kda is not None,
            workspaces is not None,
            any(
                getattr(module, "kpool_tail_cache", None) is not None
                for module in model.modules()
            ),
        )
        model._glm5_kpool_capture_logged = True
    if ready:
        hf_config = getattr(atom_config, "hf_config", None)
        _index_kpool = int(getattr(hf_config, "index_kpool", 4) or 4)
        try:
            metadata, is_prefill, scheduled_bs = _build_metadata(
                indexer, sparse, kda, workspaces, index_kpool=_index_kpool
            )
        except NotImplementedError:
            metadata = AttentionMetaData()
            is_prefill = False
            scheduled_bs = 0
            ready = False
    else:
        metadata = AttentionMetaData()
        is_prefill = False
        scheduled_bs = 0

    context = Context(
        positions=positions,
        is_prefill=is_prefill,
        is_dummy_run=not ready,
        scheduled_bs=scheduled_bs,
        scheduled_tokens=int(positions.numel()),
        running_bs=scheduled_bs,
        running_tokens=int(positions.numel()),
        input_ids=input_ids,
    )
    set_forward_context(
        attn_metadata=metadata,
        atom_config=atom_config,
        context=context,
        num_tokens=int(positions.numel()),
        in_hipgraph=capturing and ready,
    )
    try:
        yield
    finally:
        reset_forward_context()
