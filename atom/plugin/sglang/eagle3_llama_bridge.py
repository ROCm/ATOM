from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from typing import Any

import torch

from atom.plugin.sglang.models.minimax_m3 import SGLangATOMMiniMaxM3Attention
from atom.utils import envs

logger = logging.getLogger("atom")


def _is_stream_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - ROCm capture query may fail before init
        return False


@contextmanager
def eagle3_llama_native_attention_construction():
    """Construct EAGLE3 draft attention layers with ATOM native MHA."""

    from atom.models import eagle3_llama

    previous = eagle3_llama.Attention

    def _build_eagle3_attention(*args: Any, **kwargs: Any):
        return SGLangATOMEagle3Attention(*args, **kwargs)

    eagle3_llama.Attention = _build_eagle3_attention
    try:
        yield
    finally:
        eagle3_llama.Attention = previous


class SGLangATOMEagle3Attention(SGLangATOMMiniMaxM3Attention):
    """Use ATOM native dense MHA for EAGLE3 draft under SGLang plugin runtime."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from atom.config import get_current_atom_config

        atom_config = get_current_atom_config()
        layer_offset = int(getattr(atom_config, "sgl_atom_eagle3_layer_offset", 0) or 0)
        if layer_offset > 0 and int(kwargs.get("layer_num", 0) or 0) == 0:
            kwargs["layer_num"] = layer_offset
        super().__init__(*args, **kwargs)


def is_eagle3_llama_config(config: Any) -> bool:
    archs = getattr(config, "architectures", None) or []
    return any("LlamaForCausalLMEagle3" in str(arch) for arch in archs)


def maybe_get_eagle3_pools_from_sglang_batch(forward_batch=None):
    if forward_batch is None:
        return None, None
    token_to_kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
    req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)

    # SGLang v0.5.15 builds an inner ForwardBatch for each multi-step draft
    # attention backend. The inner view receives the pools, while the outer
    # ForwardBatch passed to the draft model no longer does. Recover them from
    # the active backend (or one of its per-step children) during graph capture.
    if token_to_kv_pool is None or req_to_token_pool is None:
        try:
            from sglang.srt.model_executor.forward_context import (
                get_attn_backend,
                has_forward_context,
            )

            backends = []
            if has_forward_context():
                active_backend = get_attn_backend()
                backends.append(active_backend)
                full_backend = getattr(active_backend, "full_attn_backend", None)
                if full_backend is not None:
                    backends.append(full_backend)
                backends.extend(getattr(active_backend, "attn_backends", None) or [])

            for backend in backends:
                token_to_kv_pool = token_to_kv_pool or getattr(
                    backend,
                    "_atom_token_to_kv_pool",
                    getattr(backend, "token_to_kv_pool", None),
                )
                req_to_token_pool = req_to_token_pool or getattr(
                    backend,
                    "_atom_req_to_token_pool",
                    getattr(backend, "req_to_token_pool", None),
                )
                if token_to_kv_pool is not None and req_to_token_pool is not None:
                    break
        except Exception:  # noqa: BLE001, S110 - optional across SGLang versions
            pass

    if token_to_kv_pool is None or req_to_token_pool is None:
        return None, None
    if getattr(forward_batch, "token_to_kv_pool", None) is None:
        forward_batch.token_to_kv_pool = token_to_kv_pool
    if getattr(forward_batch, "req_to_token_pool", None) is None:
        forward_batch.req_to_token_pool = req_to_token_pool
    return token_to_kv_pool, req_to_token_pool


def _page_size(token_to_kv_pool) -> int:
    return int(getattr(token_to_kv_pool, "page_size", 1))


def _seq_lens(forward_batch, bs: int) -> torch.Tensor:
    return forward_batch.seq_lens[:bs].to(dtype=torch.int32)


def _extend_lens(forward_batch, positions: torch.Tensor, bs: int) -> torch.Tensor:
    extend_lens = getattr(forward_batch, "extend_seq_lens", None)
    if extend_lens is not None:
        return extend_lens[:bs].to(device=positions.device, dtype=torch.int32)

    extend_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_lens_cpu is not None:
        return torch.as_tensor(
            extend_lens_cpu[:bs], dtype=torch.int32, device=positions.device
        )

    tokens_per_req = getattr(
        getattr(forward_batch, "spec_info", None), "num_tokens_per_req", None
    )
    if tokens_per_req is None:
        tokens_per_req = max(1, int(positions.numel()) // max(1, bs))
    return torch.full(
        (bs,), int(tokens_per_req), dtype=torch.int32, device=positions.device
    )


def _effective_decode_lens(
    forward_batch, bs: int, seq_lens: torch.Tensor
) -> torch.Tensor:
    spec_info = getattr(forward_batch, "spec_info", None)
    kv_indptr = getattr(spec_info, "kv_indptr", None)
    if torch.is_tensor(kv_indptr) and kv_indptr.numel() >= bs + 1:
        return (kv_indptr[1 : bs + 1] - kv_indptr[:bs]).to(
            device=seq_lens.device, dtype=torch.int32
        )

    backend_metadata = getattr(
        getattr(forward_batch, "attn_backend", None), "forward_metadata", None
    )
    backend_lens = getattr(backend_metadata, "kv_lens", None)
    if torch.is_tensor(backend_lens) and backend_lens.numel() >= bs:
        return backend_lens[:bs].to(device=seq_lens.device, dtype=torch.int32)

    return seq_lens


def _decode_lens_from_positions(
    positions: torch.Tensor | None, bs: int, seq_lens: torch.Tensor
) -> torch.Tensor | None:
    if not torch.is_tensor(positions) or positions.numel() < bs:
        return None
    # The EAGLE3 wrapper forwards draft tokens at positions + 1 to match native
    # ATOM. Keep the attention metadata's position-derived KV length in the same
    # coordinate system; kv_indptr remains the upper bound.
    return positions[:bs].to(device=seq_lens.device, dtype=torch.int32) + 1


def _atom_num_reject_tokens(
    forward_batch, bs: int, seq_lens: torch.Tensor
) -> torch.Tensor | None:
    spec_info = getattr(forward_batch, "spec_info", None)
    num_reject_tokens = getattr(
        spec_info, "_atom_sglang_eagle3_num_reject_tokens", None
    )
    if not torch.is_tensor(num_reject_tokens):
        # CUDA graph capture owns a fixed-address generic input buffer because
        # plugin-only Python attributes are not re-evaluated during replay.
        num_reject_tokens = getattr(spec_info, "num_reject_tokens", None)
    if not torch.is_tensor(num_reject_tokens) or num_reject_tokens.numel() < bs:
        return None
    return num_reject_tokens[:bs].to(device=seq_lens.device, dtype=torch.int32)


def _frontier_slot_mapping(
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Map each sequence's speculative frontier to its physical KV slot."""
    if context_lens.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=context_lens.device)
    logical_slots = (context_lens.to(dtype=torch.long) - 1).clamp_min_(0)
    block_indices = torch.div(
        logical_slots, int(page_size), rounding_mode="floor"
    ).unsqueeze(1)
    block_ids = (
        block_tables[: context_lens.numel()]
        .to(device=context_lens.device, dtype=torch.long)
        .gather(1, block_indices)
        .squeeze(1)
    )
    return block_ids * int(page_size) + torch.remainder(logical_slots, int(page_size))


def _block_tables_from_token_table(
    token_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    if max_seq_len is None:
        max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() else 0
    max_blocks = max(1, (max_seq_len + page_size - 1) // page_size)
    return (
        (token_table[:, : max_blocks * page_size : page_size] // page_size)
        .to(dtype=torch.int32)
        .contiguous()
    )


def _build_block_table(
    forward_batch,
    req_to_token_pool,
    *,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor | None,
    page_size: int,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    bs = int(forward_batch.batch_size)
    if max_seq_len is None:
        max_seq_len = int(seq_lens.max().item()) if bs else 0
    max_blocks = max(1, (max_seq_len + page_size - 1) // page_size)
    req_pool_indices = forward_batch.req_pool_indices[:bs]
    token_table = req_to_token_pool.req_to_token[
        req_pool_indices, : max_blocks * page_size
    ].clone()

    if extend_lens is not None:
        prefix_lens = seq_lens - extend_lens
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is not None:
            columns = torch.arange(token_table.shape[1], device=token_table.device)
            rel_pos = columns.unsqueeze(0) - prefix_lens.unsqueeze(1)
            mask = (rel_pos >= 0) & (rel_pos < extend_lens.unsqueeze(1))
            query_offsets = torch.cumsum(extend_lens, dim=0) - extend_lens
            src_idx = query_offsets.unsqueeze(1) + rel_pos
            max_src_idx = max(int(out_cache_loc.numel()) - 1, 0)
            src_idx = src_idx.clamp_min(0).clamp_max(max_src_idx).to(torch.long)
            src_values = out_cache_loc.gather(0, src_idx.reshape(-1)).view_as(src_idx)
            token_table = torch.where(mask, src_values, token_table)

    return _block_tables_from_token_table(
        token_table, seq_lens, page_size, max_seq_len=max_seq_len
    )


def _metadata_from_backend(
    forward_batch,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    bs: int,
    *,
    page_size: int,
):
    backend_metadata = getattr(
        getattr(forward_batch, "attn_backend", None), "forward_metadata", None
    )
    page_table = getattr(backend_metadata, "page_table", None)
    if not torch.is_tensor(page_table):
        return None

    from atom.utils.forward_context import AttentionMetaData, AttnState

    context_lens = _effective_decode_lens(forward_batch, bs, seq_lens)
    position_lens = _decode_lens_from_positions(positions, bs, seq_lens)
    if position_lens is not None:
        num_reject_tokens = _atom_num_reject_tokens(forward_batch, bs, seq_lens)
        context_lens = (
            position_lens + num_reject_tokens
            if num_reject_tokens is not None
            else position_lens
        )
    cu_q = torch.arange(0, bs + 1, dtype=torch.int32, device=seq_lens.device)
    if position_lens is not None:
        slot_mapping = _frontier_slot_mapping(page_table, context_lens, page_size)
    else:
        slot_mapping = getattr(forward_batch, "out_cache_loc", None)
    if torch.is_tensor(slot_mapping) and position_lens is None:
        slot_mapping = slot_mapping[:bs].to(dtype=torch.long)
    # Draft graph capture cannot read context_lens.max() back to the host. Use
    # the captured page-table width as the static K-length bound.
    max_seq_len = (
        int(page_table.shape[1]) * int(page_size)
        if _is_stream_capturing()
        else (int(context_lens.max().item()) if bs else 0)
    )
    md = AttentionMetaData(
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        max_seqlen_k=max_seq_len,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=page_table[:bs],
        state=AttnState.DECODE,
    )
    return md


def build_atom_eagle3_attention_metadata_from_sglang(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
):
    from atom.utils.forward_context import AttentionMetaData, AttnState

    bs = int(forward_batch.batch_size)
    page_size = _page_size(token_to_kv_pool)
    max_context_len = int(req_to_token_pool.req_to_token.shape[1])
    # EAGLE3 decode and draft-extend graphs need a static block-table width;
    # eager execution can derive the live bound from sequence lengths.
    capture_max_seq_len = max_context_len if _is_stream_capturing() else None
    forward_mode = forward_batch.forward_mode
    seq_lens = _seq_lens(forward_batch, bs)

    if forward_mode.is_decode_or_idle():
        backend_md = _metadata_from_backend(
            forward_batch, positions, seq_lens, bs, page_size=page_size
        )
        if backend_md is not None:
            return backend_md

        context_lens = _effective_decode_lens(forward_batch, bs, seq_lens)
        position_lens = _decode_lens_from_positions(positions, bs, seq_lens)
        if position_lens is not None:
            num_reject_tokens = _atom_num_reject_tokens(forward_batch, bs, seq_lens)
            context_lens = (
                position_lens + num_reject_tokens
                if num_reject_tokens is not None
                else position_lens
            )
        cu_q = torch.arange(0, bs + 1, dtype=torch.int32, device=positions.device)
        block_table = _build_block_table(
            forward_batch,
            req_to_token_pool,
            seq_lens=context_lens,
            extend_lens=None,
            page_size=page_size,
            max_seq_len=capture_max_seq_len,
        )
        slot_mapping = (
            _frontier_slot_mapping(block_table, context_lens, page_size)
            if position_lens is not None
            else getattr(forward_batch, "out_cache_loc", None)[:bs].to(torch.long)
        )
        max_seqlen_k = (
            max_context_len
            if _is_stream_capturing()
            else (int(context_lens.max().item()) if bs else 0)
        )
        md = AttentionMetaData(
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_table,
            state=AttnState.DECODE,
        )
        return md

    # Prefill / draft-extend paths are extend-like: seq_lens includes the new
    # tokens, extend_lens tells how much of each sequence is newly written.
    is_draft_extend_v2 = bool(
        getattr(forward_mode, "is_draft_extend_v2", lambda: False)()
    )
    if is_draft_extend_v2:
        tokens_per_req = getattr(
            getattr(forward_batch, "spec_info", None), "num_tokens_per_req", None
        )
        if tokens_per_req is None:
            if bs <= 0:
                raise RuntimeError(
                    "DRAFT_EXTEND_V2 requires a positive batch size to infer "
                    "num_tokens_per_req"
                )
            tokens_per_req = int(positions.numel()) // bs
        tokens_per_req = int(tokens_per_req)
        expected_num_positions = bs * tokens_per_req
        if tokens_per_req <= 0 or positions.numel() != expected_num_positions:
            raise RuntimeError(
                "Invalid DRAFT_EXTEND_V2 padded query layout: expected "
                "num_tokens_per_req > 0 and positions.numel() == "
                f"batch_size * num_tokens_per_req, but got batch_size={bs}, "
                f"num_tokens_per_req={tokens_per_req}, "
                f"positions.numel()={positions.numel()}"
            )
        # In SGLang v0.5.15, V2 extend_seq_lens may describe this step's
        # accepted/valid token count, while model inputs retain a fixed padded
        # tree width. Attention's QO layout must use num_tokens_per_req.
        extend_lens = torch.full(
            (bs,),
            tokens_per_req,
            dtype=torch.int32,
            device=positions.device,
        )
    else:
        extend_lens = _extend_lens(forward_batch, positions, bs)
        tokens_per_req = max(1, int(positions.numel()) // max(1, bs))
    cu_q = torch.zeros(bs + 1, dtype=torch.int32, device=positions.device)
    cu_q[1:] = torch.cumsum(extend_lens, dim=0)
    block_table = _build_block_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=seq_lens,
        extend_lens=extend_lens,
        page_size=page_size,
        max_seq_len=capture_max_seq_len,
    )
    # DRAFT_EXTEND_V2 uses a fixed padded Q width. Other captured extend shapes
    # also require static Q/K bounds because GPU scalar reads are not capturable.
    max_seqlen_q = (
        tokens_per_req
        if is_draft_extend_v2 or _is_stream_capturing()
        else (int(extend_lens.max().item()) if bs else 0)
    )
    max_seqlen_k = (
        max_context_len
        if _is_stream_capturing()
        else (int(seq_lens.max().item()) if bs else 0)
    )
    md = AttentionMetaData(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=getattr(forward_batch, "out_cache_loc", None),
        context_lens=(seq_lens if is_draft_extend_v2 else seq_lens - extend_lens),
        block_tables=block_table,
        state=AttnState.PREFILL_PREFIX,
    )
    return md


def _iter_eagle3_attention_layers(model):
    for module in model.modules():
        attn = getattr(module, "attn", None)
        impl = getattr(attn, "impl", None)
        if isinstance(attn, SGLangATOMEagle3Attention) and impl is not None:
            yield attn, impl


def bind_eagle3_llama_cache_views(model, token_to_kv_pool) -> bool:
    if token_to_kv_pool is None or not hasattr(token_to_kv_pool, "get_kv_buffer"):
        return False

    from atom.config import KVCacheTensor
    from atom.utils.forward_context import get_forward_context, set_kv_cache_data

    page_size = _page_size(token_to_kv_pool)
    pool_start_layer = int(getattr(token_to_kv_pool, "start_layer", 0) or 0)
    pool_layer_num = int(getattr(token_to_kv_pool, "layer_num", 0) or 0)
    kv_cache_data = {}
    bound = False
    for attn, impl in _iter_eagle3_attention_layers(model):
        layer_id = int(impl.layer_num)
        # SGLang draft workers allocate an independent one-layer draft KV pool
        # whose physical layer index can still start at 0. ATOM native EAGLE3
        # uses the logical layer id after target layers (e.g. 60), so bind the
        # physical pool slot while keeping kv_cache_data keyed by layer_id.
        if pool_layer_num == 1 and not (
            pool_start_layer <= layer_id < pool_start_layer + pool_layer_num
        ):
            pool_layer_id = pool_start_layer
        else:
            pool_layer_id = layer_id
        k_buffer, _v_buffer = token_to_kv_pool.get_kv_buffer(pool_layer_id)
        num_slots, num_kv_heads, head_dim = k_buffer.shape
        num_blocks = max(1, int(num_slots) // page_size)
        x = 16 // k_buffer.element_size()

        # SGLang's MHA KV pool is row-major [slot, kv_head, head_dim].  Native
        # ATOM EAGLE3 allocates a page-major draft cache
        # [block, block_size, kv_head, head_dim] and then reinterprets it as the
        # shuffle views below.  A direct view of SGLang's row-major pool has the
        # same shape but not the same page/head/dim memory layout, which breaks
        # later draft decode steps that read previously written draft KV.  Keep
        # an ATOM-native draft cache while reusing SGLang's slot ids/metadata.
        native_cache = getattr(model, "_atom_sglang_eagle3_native_kv_cache", None)
        cache_shape = (2, num_blocks, page_size, num_kv_heads, head_dim)
        if (
            native_cache is None
            or tuple(native_cache.shape) != cache_shape
            or native_cache.dtype != k_buffer.dtype
            or native_cache.device != k_buffer.device
        ):
            native_cache = torch.zeros(
                cache_shape, dtype=k_buffer.dtype, device=k_buffer.device
            )
            model._atom_sglang_eagle3_native_kv_cache = native_cache

        native_scale = getattr(model, "_atom_sglang_eagle3_native_kv_scale", None)
        scale_shape = (2, num_blocks, num_kv_heads, page_size)
        use_fp8_cache = k_buffer.element_size() == 1
        if use_fp8_cache and (
            native_scale is None
            or tuple(native_scale.shape) != scale_shape
            or native_scale.device != k_buffer.device
        ):
            native_scale = torch.zeros(
                scale_shape, dtype=torch.float32, device=k_buffer.device
            )
            model._atom_sglang_eagle3_native_kv_scale = native_scale

        k_cache = native_cache[0].view(
            num_blocks, num_kv_heads, head_dim // x, page_size, x
        )
        v_cache = native_cache[1].view(num_blocks, num_kv_heads, head_dim, page_size)
        if use_fp8_cache:
            k_scale = native_scale[0]
            v_scale = native_scale[1]
        elif hasattr(token_to_kv_pool, "get_kv_scale_buffer"):
            k_scale, v_scale = token_to_kv_pool.get_kv_scale_buffer(pool_layer_id)
        else:
            k_scale = None
            v_scale = None

        kv_cache_data[f"layer_{layer_id}"] = KVCacheTensor(
            layer_num=layer_id,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        attn.k_cache = k_cache
        attn.v_cache = v_cache
        attn.k_scale = k_scale
        attn.v_scale = v_scale
        bound = True

    if not bound:
        return False

    set_kv_cache_data(kv_cache_data)
    get_forward_context().kv_cache_data = kv_cache_data
    return True


def _patch_sglang_eagle3_state_lifecycle() -> None:
    """Keep ATOM EAGLE3 reject state aligned with SGLang V2 batch rows.

    The reject-count tensor is an ATOM plugin extension on ``EagleDraftInput``.
    SGLang's native ``filter_batch`` / ``merge_batch`` only know about the
    dataclass fields, so the extension otherwise keeps stale row order whenever
    requests are merged or removed.  This is invisible for a steady bs=1 batch
    but corrupts per-request draft context lengths under continuous batching.
    """

    try:
        from sglang.srt.managers.overlap_utils import FutureMap
        from sglang.srt.speculative.eagle_info import EagleDraftInput
    except Exception:  # noqa: BLE001 - optional across SGLang versions
        return

    if getattr(EagleDraftInput, "_atom_eagle3_state_lifecycle_patched", False):
        return

    state_attr = "_atom_sglang_eagle3_num_reject_tokens"
    original_filter_batch = EagleDraftInput.filter_batch
    original_merge_batch = EagleDraftInput.merge_batch
    try:
        filter_signature = inspect.signature(original_filter_batch)
        filter_has_legacy_flag = "has_been_filtered" in filter_signature.parameters
    except (TypeError, ValueError):
        filter_has_legacy_flag = False
    uses_pool_indexed_future_map = all(
        hasattr(FutureMap, name) for name in ("stash", "_resolve_spec_extras")
    )
    if uses_pool_indexed_future_map:
        from sglang.srt.managers.overlap_utils import RelayPayload

        original_relay_from_draft_input = RelayPayload.from_draft_input
        original_future_store = FutureMap.stash
        original_future_resolve = FutureMap._resolve_spec_extras
    else:
        original_future_lazy_init = FutureMap._lazy_init_buf
        original_future_store = FutureMap.store_to_map_for_new_batch
        original_future_resolve = FutureMap.resolve_future

    def _row_count(spec_info) -> int:
        future_indices = getattr(spec_info, "future_indices", None)
        indices = getattr(future_indices, "indices", None)
        if torch.is_tensor(indices):
            return int(indices.numel())
        topk_index = getattr(spec_info, "topk_index", None)
        if torch.is_tensor(topk_index):
            return int(topk_index.shape[0])
        return 0

    def _state_for_rows(state, rows: int, reference):
        device = (
            state.device
            if torch.is_tensor(state)
            else (
                reference.device if torch.is_tensor(reference) else torch.device("cpu")
            )
        )
        if rows <= 0:
            return torch.empty((0,), dtype=torch.int32, device=device)
        if not torch.is_tensor(state):
            return torch.zeros((rows,), dtype=torch.int32, device=device)
        state = state.reshape(-1).to(dtype=torch.int32)
        if state.numel() >= rows:
            return state[:rows]
        return torch.cat(
            [
                state,
                torch.zeros(
                    (rows - state.numel(),), dtype=torch.int32, device=state.device
                ),
            ]
        )

    def filter_batch(self, new_indices, *args, **kwargs):
        # With overlap scheduling, ``future_indices`` is the only payload that
        # is ready on the scheduler stream.  The tensors attached to
        # ``EagleDraftInput`` are produced on the forward stream and are
        # intentionally resolved lazily by ``FutureMap.resolve_future``.
        # Reading/indexing our plugin-only state here would introduce the same
        # cross-stream race that SGLang's native early-return avoids.
        has_been_filtered = bool(
            kwargs.get("has_been_filtered", filter_has_legacy_flag)
        )
        if args and isinstance(args[0], bool):
            has_been_filtered = args[0]
        state_is_deferred = getattr(self, "future_indices", None) is not None
        state_before = getattr(self, state_attr, None)
        ret = original_filter_batch(self, new_indices, *args, **kwargs)

        output_rows = _row_count(self)
        if state_is_deferred:
            # The authoritative state is already stored in the parallel
            # FutureMap buffer.  Drop this possibly-not-yet-ready tensor; the
            # filtered future indices will gather the correct rows on the
            # forward stream before the next model invocation.
            setattr(self, state_attr, None)
            return ret

        reference = getattr(self, "topk_index", None)
        if torch.is_tensor(state_before):
            state_before = state_before.reshape(-1)
            indices = new_indices.to(device=state_before.device, dtype=torch.long)
            if has_been_filtered and state_before.numel() == output_rows:
                state_after = state_before.to(dtype=torch.int32)
            elif indices.numel() == output_rows:
                state_after = state_before.index_select(0, indices).to(
                    dtype=torch.int32
                )
            else:
                state_after = _state_for_rows(state_before, output_rows, reference)
        else:
            state_after = _state_for_rows(None, output_rows, reference)
        setattr(self, state_attr, state_after)
        return ret

    def merge_batch(self, spec_info):
        left_rows = _row_count(self)
        right_rows = _row_count(spec_info)
        left_state = getattr(self, state_attr, None)
        right_state = getattr(spec_info, state_attr, None)
        left_reference = getattr(self, "topk_index", None)
        right_reference = getattr(spec_info, "topk_index", None)
        state_is_deferred = getattr(self, "future_indices", None) is not None

        ret = original_merge_batch(self, spec_info)

        if state_is_deferred:
            # ``original_merge_batch`` concatenated the two future-index lists.
            # Their payload rows, including the plugin state, remain in
            # FutureMap and must not be touched from the scheduler stream.
            setattr(self, state_attr, None)
            return ret

        left_state = _state_for_rows(left_state, left_rows, left_reference)
        right_state = _state_for_rows(right_state, right_rows, right_reference)
        if left_state.device != right_state.device:
            if left_rows == 0:
                left_state = left_state.to(right_state.device)
            else:
                right_state = right_state.to(left_state.device)
        merged_state = torch.cat([left_state, right_state])
        setattr(self, state_attr, merged_state)
        return ret

    def future_lazy_init(self, draft_input):
        ret = original_future_lazy_init(self, draft_input)
        if not hasattr(self, "_atom_eagle3_reject_tokens_buf"):
            self._atom_eagle3_reject_tokens_buf = torch.zeros(
                (self.future_buffer_len,), dtype=torch.int32, device=self.device
            )
        return ret

    def future_store(self, future_indices, draft_input):
        ret = original_future_store(self, future_indices, draft_input)
        interval = future_indices.interval
        if (
            interval is None
            or self.is_empty_slice(interval)
            or not hasattr(self, "_atom_eagle3_reject_tokens_buf")
        ):
            return ret
        rows = _row_count(draft_input)
        state = _state_for_rows(
            getattr(draft_input, state_attr, None),
            rows,
            getattr(draft_input, "topk_index", None),
        ).to(self._atom_eagle3_reject_tokens_buf.device)
        self._atom_eagle3_reject_tokens_buf[interval] = state
        return ret

    def future_resolve(self, model_worker_batch):
        ret = original_future_resolve(self, model_worker_batch)
        draft_input = getattr(model_worker_batch, "spec_info", None)
        future_indices = getattr(draft_input, "future_indices", None)
        indices = getattr(future_indices, "indices", None)
        if (
            draft_input is not None
            and torch.is_tensor(indices)
            and hasattr(self, "_atom_eagle3_reject_tokens_buf")
        ):
            resolved_state = self._atom_eagle3_reject_tokens_buf[indices].to(
                torch.int32
            )
            setattr(
                draft_input,
                state_attr,
                resolved_state,
            )
        return ret

    def relay_from_draft_input(cls, draft_input):
        payload = original_relay_from_draft_input(draft_input)
        setattr(payload, state_attr, getattr(draft_input, state_attr, None))
        return payload

    def pool_indexed_future_store(self, future_indices, payload):
        ret = original_future_store(self, future_indices, payload)
        # FutureMap is shared by ordinary generation and speculative decoding.
        # Do not create/copy EAGLE3-only state for non-EAGLE payloads: a tiny
        # CPU-to-GPU copy here synchronizes the forward stream and defeats
        # SGLang's overlap scheduler after every prefill.
        if not hasattr(payload, state_attr):
            return ret
        indices = future_indices
        if not torch.is_tensor(indices) or indices.numel() == 0:
            return ret
        if not hasattr(self, "_atom_eagle3_reject_tokens_buf"):
            self._atom_eagle3_reject_tokens_buf = torch.zeros(
                (self.req_pool_size,), dtype=torch.int32, device=self.device
            )
        rows = int(indices.numel())
        state = _state_for_rows(
            getattr(payload, state_attr, None),
            rows,
            getattr(payload, "topk_index", None),
        ).to(self._atom_eagle3_reject_tokens_buf.device)
        self._atom_eagle3_reject_tokens_buf[indices] = state
        return ret

    def pool_indexed_future_resolve(self, batch):
        ret = original_future_resolve(self, batch)
        draft_input = getattr(batch, "spec_info", None)
        indices = getattr(draft_input, "future_indices", None)
        if (
            draft_input is not None
            and torch.is_tensor(indices)
            and hasattr(self, "_atom_eagle3_reject_tokens_buf")
        ):
            setattr(
                draft_input,
                state_attr,
                self._atom_eagle3_reject_tokens_buf[indices].to(torch.int32),
            )
        return ret

    EagleDraftInput.filter_batch = filter_batch
    EagleDraftInput.merge_batch = merge_batch
    if uses_pool_indexed_future_map:
        RelayPayload.from_draft_input = classmethod(relay_from_draft_input)
        FutureMap.stash = pool_indexed_future_store
        FutureMap._resolve_spec_extras = pool_indexed_future_resolve
    else:
        FutureMap._lazy_init_buf = future_lazy_init
        FutureMap.store_to_map_for_new_batch = future_store
        FutureMap.resolve_future = future_resolve
    EagleDraftInput._atom_eagle3_state_lifecycle_patched = True


def _patch_sglang_eagle3_cuda_graph_reject_state() -> None:
    """Stage plugin reject state through a fixed-address draft graph buffer."""

    try:
        from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
            EAGLEDraftCudaGraphRunner,
        )
        from sglang.srt.speculative.eagle_info import EagleDraftInput
    except Exception:  # noqa: BLE001 - optional across SGLang versions
        return

    if getattr(
        EAGLEDraftCudaGraphRunner,
        "_atom_eagle3_reject_graph_state_patched",
        False,
    ):
        return

    state_attr = "_atom_sglang_eagle3_num_reject_tokens"
    graph_state_attr = "_atom_eagle3_reject_graph_buffer"
    active_capture_runner = {"runner": None}
    original_input_init = EagleDraftInput.__init__
    # v0.5.15 uses shape-based capture/execute; retain the older
    # batch-size/replay API as a compatibility fallback.
    uses_shape_graph_api = hasattr(EAGLEDraftCudaGraphRunner, "capture_one_shape")
    if uses_shape_graph_api:
        original_capture = EAGLEDraftCudaGraphRunner.capture_one_shape
        original_replay = EAGLEDraftCudaGraphRunner.execute
    else:
        original_capture = EAGLEDraftCudaGraphRunner.capture_one_batch_size
        original_replay = EAGLEDraftCudaGraphRunner.replay

    def _graph_state_buffer(runner):
        buffer = getattr(runner, graph_state_attr, None)
        max_bs = int(getattr(runner, "max_bs", 0) or 0)
        reference = getattr(getattr(runner, "buffers", None), "topk_p", None)
        if max_bs <= 0 or not torch.is_tensor(reference):
            raise RuntimeError("EAGLE3 draft graph inputs are not initialized")
        if (
            not torch.is_tensor(buffer)
            or buffer.numel() < max_bs
            or buffer.device != reference.device
        ):
            buffer = torch.zeros(
                (max_bs,),
                dtype=torch.int32,
                device=reference.device,
            )
            setattr(runner, graph_state_attr, buffer)
        return buffer

    def input_init(self, *args, **kwargs):
        original_input_init(self, *args, **kwargs)
        runner = active_capture_runner["runner"]
        if runner is None:
            return
        topk_p = getattr(self, "topk_p", None)
        if not torch.is_tensor(topk_p):
            return
        state = _graph_state_buffer(runner)[: int(topk_p.shape[0])]
        setattr(self, state_attr, state)
        self.num_reject_tokens = state

    def capture_one_batch_size(self, num_seqs, forward, stream_idx=0):
        buffer = _graph_state_buffer(self)
        buffer.zero_()
        previous_runner = active_capture_runner["runner"]
        active_capture_runner["runner"] = self
        try:
            return original_capture(
                self,
                num_seqs,
                forward,
                stream_idx=stream_idx,
            )
        finally:
            active_capture_runner["runner"] = previous_runner

    def capture_one_shape(self, *args, **kwargs):
        buffer = _graph_state_buffer(self)
        buffer.zero_()
        previous_runner = active_capture_runner["runner"]
        active_capture_runner["runner"] = self
        try:
            return original_capture(self, *args, **kwargs)
        finally:
            active_capture_runner["runner"] = previous_runner

    def replay(self, forward_batch):
        buffer = _graph_state_buffer(self)
        buffer.zero_()
        raw_bs = int(forward_batch.batch_size)
        spec_info = getattr(forward_batch, "spec_info", None)
        live_state = getattr(spec_info, state_attr, None)
        if not torch.is_tensor(live_state):
            live_state = getattr(spec_info, "num_reject_tokens", None)
        if torch.is_tensor(live_state):
            if live_state.numel() < raw_bs:
                raise RuntimeError(
                    "EAGLE3 reject graph state has fewer rows than the replay "
                    f"batch: state_rows={live_state.numel()} batch_size={raw_bs}"
                )
            buffer[:raw_bs].copy_(
                live_state[:raw_bs].to(
                    device=buffer.device,
                    dtype=buffer.dtype,
                )
            )
        return original_replay(self, forward_batch)

    EagleDraftInput.__init__ = input_init
    if uses_shape_graph_api:
        EAGLEDraftCudaGraphRunner.capture_one_shape = capture_one_shape
        EAGLEDraftCudaGraphRunner.execute = replay
    else:
        EAGLEDraftCudaGraphRunner.capture_one_batch_size = capture_one_batch_size
        EAGLEDraftCudaGraphRunner.replay = replay
    EAGLEDraftCudaGraphRunner._atom_eagle3_reject_graph_state_patched = True


def _patch_sglang_eagle3_draft_extend_compat() -> None:
    """Keep SGLang EAGLE3 draft-extend semantics aligned with ATOM."""

    try:
        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
    except Exception:  # noqa: BLE001 - optional across SGLang versions
        return

    if getattr(
        EagleDraftWorker,
        "_atom_eagle3_draft_extend_compat_patched",
        False,
    ):
        return

    original_draft_extend_for_prefill = EagleDraftWorker._draft_extend_for_prefill
    original_draft_extend_for_decode = EagleDraftWorker._draft_extend_for_decode

    def _draft_extend_for_prefill(
        self,
        batch,
        target_hidden_states,
        next_token_ids,
        mm_input_embeds=None,
    ):
        ret = original_draft_extend_for_prefill(
            self,
            batch,
            target_hidden_states,
            next_token_ids,
            mm_input_embeds=mm_input_embeds,
        )
        # A fresh request has no rejected draft suffix.  Set an explicit
        # per-request zero vector so merging it into an active decode batch
        # cannot inherit the previous batch's plugin-only reject state.
        ret._atom_sglang_eagle3_num_reject_tokens = torch.zeros(
            (len(batch.seq_lens),), dtype=torch.int32, device=next_token_ids.device
        )
        return ret

    def _draft_extend_for_decode(self, batch, batch_result):
        is_eagle3 = bool(
            getattr(
                getattr(self, "speculative_algorithm", None),
                "is_eagle3",
                lambda: False,
            )()
        )
        if not is_eagle3:
            return original_draft_extend_for_decode(self, batch, batch_result)

        patched_next_token_ids = None
        original_next_token_ids = getattr(batch_result, "next_token_ids", None)
        spec_info = getattr(batch, "spec_info", None)
        candidates = getattr(spec_info, "draft_token", None)
        accept_lens = getattr(batch_result, "accept_lens", None)
        num_draft_tokens = int(getattr(self, "speculative_num_draft_tokens", 0) or 0)

        if (
            torch.is_tensor(original_next_token_ids)
            and torch.is_tensor(candidates)
            and torch.is_tensor(accept_lens)
            and num_draft_tokens > 1
            and original_next_token_ids.numel()
            >= len(batch.seq_lens) * num_draft_tokens
            and candidates.numel() >= len(batch.seq_lens) * num_draft_tokens
        ):
            bs = len(batch.seq_lens)
            patched_next_token_ids = original_next_token_ids.clone()
            patched_view = patched_next_token_ids[: bs * num_draft_tokens].view(
                bs, num_draft_tokens
            )
            candidate_view = candidates[: bs * num_draft_tokens].view(
                bs, num_draft_tokens
            )
            accept_view = accept_lens[:bs].to(device=patched_view.device)
            # SGLang stores verify candidates as
            # [target_next, draft_1, ..., draft_k], while native ATOM's
            # draft-extend input is the one-column left rotation.
            rotated_candidates = torch.roll(candidate_view, shifts=-1, dims=1)
            columns = torch.arange(
                num_draft_tokens, device=patched_view.device
            ).unsqueeze(0)
            fill_mask = (columns >= accept_view.unsqueeze(1)) & (columns > 0)
            patched_view.copy_(torch.where(fill_mask, rotated_candidates, patched_view))
            batch_result.next_token_ids = patched_next_token_ids

        try:
            ret = original_draft_extend_for_decode(self, batch, batch_result)
        finally:
            if patched_next_token_ids is not None:
                batch_result.next_token_ids = original_next_token_ids

        accept_lens = getattr(batch_result, "accept_lens", None)
        next_draft_input = getattr(batch_result, "next_draft_input", None)
        if (
            next_draft_input is not None
            and torch.is_tensor(accept_lens)
            and num_draft_tokens > 1
        ):
            bs = len(batch.seq_lens)
            num_reject_tokens = (
                num_draft_tokens - accept_lens[:bs].to(dtype=torch.int32)
            ).clamp(min=0, max=num_draft_tokens - 1)
            next_draft_input._atom_sglang_eagle3_num_reject_tokens = num_reject_tokens
        return ret

    EagleDraftWorker._draft_extend_for_prefill = _draft_extend_for_prefill
    EagleDraftWorker._draft_extend_for_decode = _draft_extend_for_decode
    EagleDraftWorker._atom_eagle3_draft_extend_compat_patched = True


def _patch_sglang_eagle3_tp_verify_broadcast() -> None:
    """Optionally broadcast SGLang EAGLE3 verify outputs across TP ranks."""

    if not envs.ATOM_SGLANG_EAGLE3_TP_VERIFY_BROADCAST:
        return

    try:
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.distributed.parallel_state import get_attn_tp_group
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled
        from sglang.srt.speculative import eagle_worker_v2 as eagle_worker_module
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        if not hasattr(EagleVerifyInput, "sample"):
            from sglang.srt.speculative import eagle_utils
    except Exception as exc:
        logger.debug(
            "Skipping optional EAGLE3 TP verify broadcast patch: %s",
            exc,
            exc_info=True,
        )
        return

    if getattr(
        eagle_worker_module,
        "_atom_eagle3_tp_verify_broadcast_patched",
        False,
    ):
        return

    def _broadcast_sample_result(result):
        predict, accept_lens, accept_index = result
        tp_group = get_attn_tp_group() if is_dp_attention_enabled() else get_tp_group()
        if tp_group.world_size > 1:
            tp_group.broadcast(predict, src=0)
            tp_group.broadcast(accept_lens, src=0)
            tp_group.broadcast(accept_index, src=0)
        return predict, accept_lens, accept_index

    if hasattr(EagleVerifyInput, "sample"):
        original_sample = EagleVerifyInput.sample

        def sample(self, batch, logits_output, vocab_mask=None):
            return _broadcast_sample_result(
                original_sample(self, batch, logits_output, vocab_mask)
            )

        EagleVerifyInput.sample = sample
    else:
        original_eagle_sample = eagle_utils.eagle_sample

        def eagle_sample(*args, **kwargs):
            return _broadcast_sample_result(original_eagle_sample(*args, **kwargs))

        eagle_utils.eagle_sample = eagle_sample
        eagle_worker_module.eagle_sample = eagle_sample

    eagle_worker_module._atom_eagle3_tp_verify_broadcast_patched = True


def patch_sglang_eagle3_runtime_compat() -> None:
    """Install all ATOM runtime compatibility patches for SGLang EAGLE3."""

    _patch_sglang_eagle3_state_lifecycle()
    _patch_sglang_eagle3_cuda_graph_reject_state()
    _patch_sglang_eagle3_draft_extend_compat()
    _patch_sglang_eagle3_tp_verify_broadcast()
