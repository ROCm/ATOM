"""Scoped runtime adapter from SGLang batches to ATOM core."""

from __future__ import annotations

import copy
import logging
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from atom.models.utils import IntermediateTensors
from atom.plugin.sglang.runtime.context import bind_current_forward_batch

logger = logging.getLogger("atom.plugin.sglang.runtime.forward_context")


def _is_dummy_forward(forward_batch: ForwardBatch) -> bool:
    """Return whether an SGLang batch represents an empty/idle dummy run."""

    forward_mode = getattr(forward_batch, "forward_mode", None)
    return bool(
        forward_mode is not None
        and hasattr(forward_mode, "is_idle")
        and forward_mode.is_idle()
    )


def _pad_dummy_like(
    tensor: torch.Tensor | None,
    *,
    length: int,
    fill_value: float = 0,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    shape = (length, *tensor.shape[1:])
    return torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)


def _materialize_atom_dummy_forward(
    input_ids: torch.Tensor | None,
    positions: torch.Tensor | None,
    input_embeds: torch.Tensor | None,
    forward_batch: ForwardBatch,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    ForwardBatch,
]:
    """Convert an empty SGLang IDLE batch into ATOM-style dummy inputs."""
    if positions is not None:
        device = positions.device
    elif input_ids is not None:
        device = input_ids.device
    elif input_embeds is not None:
        device = input_embeds.device
    else:
        raise RuntimeError(
            "SGLang dummy forward materialization requires at least one of "
            "positions, input_ids, or input_embeds"
        )

    if positions is not None:
        dummy_positions_shape = (
            (3, 1) if positions.ndim == 2 and positions.shape[0] == 3 else (1,)
        )
        dummy_positions = positions.new_zeros(dummy_positions_shape)
    else:
        dummy_positions = torch.zeros((1,), dtype=torch.long, device=device)
    dummy_input_ids = (
        input_ids.new_zeros((1,))
        if input_ids is not None
        else torch.zeros((1,), dtype=torch.long, device=device)
    )
    dummy_input_embeds = _pad_dummy_like(input_embeds, length=1, fill_value=0)

    model_forward_batch = copy.copy(forward_batch)
    model_forward_batch.positions = dummy_positions
    model_forward_batch.batch_size = 1
    model_forward_batch.seq_lens_sum = 1
    seq_lens = getattr(forward_batch, "seq_lens", None)
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    model_forward_batch.seq_lens = (
        seq_lens.new_ones((1,))
        if torch.is_tensor(seq_lens)
        else torch.ones((1,), dtype=torch.int32, device=device)
    )
    model_forward_batch.seq_lens_cpu = (
        seq_lens_cpu.new_ones((1,))
        if torch.is_tensor(seq_lens_cpu)
        else torch.ones((1,), dtype=torch.int32, device="cpu")
    )

    return dummy_input_ids, dummy_positions, dummy_input_embeds, model_forward_batch


def _trim_hidden_states_for_output(hidden_states, num_tokens: int):
    if isinstance(hidden_states, IntermediateTensors):
        return hidden_states[:num_tokens]
    if torch.is_tensor(hidden_states):
        return hidden_states[:num_tokens]
    if isinstance(hidden_states, tuple):
        return tuple(
            tensor[:num_tokens] if torch.is_tensor(tensor) else tensor
            for tensor in hidden_states
        )
    return hidden_states


def _resolve_num_tokens_across_dp(
    atom_config: Any,
    forward_batch: ForwardBatch,
    num_tokens: int,
) -> torch.Tensor:
    """Resolve per-DP token counts for ATOM's CPU-side DPMetadata."""

    global_num_tokens_cpu = getattr(forward_batch, "global_num_tokens_cpu", None)
    if global_num_tokens_cpu is not None:
        num_tokens_across_dp = torch.tensor(
            global_num_tokens_cpu, dtype=torch.int32, device="cpu"
        )
    else:
        dp_size = atom_config.parallel_config.data_parallel_size
        global_num_tokens_gpu = getattr(forward_batch, "global_num_tokens_gpu", None)
        global_dp_buffer_len = getattr(forward_batch, "global_dp_buffer_len", None)
        is_static_same_shape_batch = (
            global_num_tokens_gpu is not None
            and global_dp_buffer_len == num_tokens * dp_size
        )
        if not is_static_same_shape_batch:
            raise RuntimeError(
                "[SGL+ATOM] SGLang dp-attention requires "
                "forward_batch.global_num_tokens_cpu unless the batch uses static "
                "same-shape DP metadata."
            )

        # Static batches, such as CUDA graph capture batches, may only keep
        # global token counts on GPU. Avoid GPU-to-CPU reads here and mirror
        # their same-shape layout directly for ATOM's CPU DPMetadata.
        num_tokens_across_dp = torch.full(
            (dp_size,), num_tokens, dtype=torch.int32, device="cpu"
        )

    # SGLang reports idle ranks as 0 tokens, but ATOM materializes every idle
    # rank as one physical dummy token. Normalize the complete vector on every
    # rank so collectives and DPMetadata use identical sizes.
    num_tokens_across_dp.clamp_min_(1)
    return num_tokens_across_dp


def _resolve_running_tokens_are_unified(
    atom_config: Any,
    forward_batch: ForwardBatch,
) -> bool:
    """Resolve the DP decode mode needed by ATOM TBO.

    This was added after skewed TBO prefill incorrectly inherited the Context
    default ``True`` and entered the uniform-decode MORI path, which could
    truncate variable-length buffers and eventually trigger a HIP error.
    Keep the legacy default when TBO is disabled.
    """

    if not atom_config.enable_tbo or not atom_config.enable_dp_attention:
        return True
    return not forward_batch.is_extend_in_batch


def _max_len_from_optional(cpu_lens, gpu_lens, default: int) -> int:
    if cpu_lens is not None:
        if isinstance(cpu_lens, torch.Tensor):
            return int(cpu_lens.max().item()) if cpu_lens.numel() else default
        return max((int(x) for x in cpu_lens), default=default)
    if gpu_lens is not None:
        return int(gpu_lens.max().item()) if gpu_lens.numel() else default
    return default


def _build_generic_attention_metadata(forward_batch: ForwardBatch, max_seqlen_q: int):
    """Build minimal ATOM metadata from SGLang batch fields for non-V4 models."""

    from atom.utils.forward_context import AttentionMetaData

    # GLM/DSA SGLang plugin attention kernels read detailed scheduling metadata
    # directly from SGLang's forward_batch.  ATOM's model-level PCP gate,
    # however, runs before those kernels and checks
    # get_forward_context().attn_metadata.max_seqlen_k in deepseek_v2._pcp_active().
    # Without this fallback metadata, max_seqlen_k stays at AttentionMetaData's
    # default 0 and long-prefill PCP never activates.
    forward_mode = forward_batch.forward_mode
    seq_lens = getattr(forward_batch, "seq_lens", None)
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
    extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)

    if not forward_mode.is_decode_or_idle():
        max_seqlen_q = _max_len_from_optional(
            extend_seq_lens_cpu, extend_seq_lens, max_seqlen_q
        )
    max_seqlen_k = _max_len_from_optional(seq_lens_cpu, seq_lens, 0)

    return AttentionMetaData(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        context_lens=seq_lens,
    )


def _slice_v4_graph_metadata_for_capture(
    attn_metadata: Any, *, num_tokens: int, bs: int
):
    """Narrow reusable V4 graph metadata to this capture bucket.

    The DSV4 fallback metadata is initialized at max graph size.  SGLang then
    captures smaller buckets (e.g. bs=248, tokens=496), so per-token arrays must
    be narrowed before model code reads them.
    """

    if attn_metadata is None:
        return None

    md = copy.copy(attn_metadata)

    def _slice_attr(name: str, n: int):
        value = getattr(md, name, None)
        if torch.is_tensor(value):
            setattr(md, name, value[:n])
        elif value is not None:
            try:
                setattr(md, name, value[:n])
            except Exception:  # noqa: BLE001, S110 - optional metadata field
                pass

    for name in (
        "batch_id_per_token",
        "slot_mapping",
        "kv_indices_swa",
        "kv_indices_csa",
        "kv_indices_hca",
        "kv_indices_extend",
        "kv_indices_prefix_swa",
        "kv_indices_prefix_csa",
        "kv_indices_prefix_hca",
        "skip_prefix_len_csa",
    ):
        _slice_attr(name, num_tokens)

    for name in (
        "kv_indptr_swa",
        "kv_indptr_csa",
        "kv_indptr_hca",
        "kv_indptr_extend",
        "kv_indptr_prefix_swa",
        "kv_indptr_prefix_csa",
        "kv_indptr_prefix_hca",
    ):
        _slice_attr(name, num_tokens + 1)

    for name in (
        "state_slot_mapping",
        "state_slot_mapping_cpu",
        "n_committed_csa_per_seq",
        "n_committed_csa_per_seq_cpu",
        "context_lens",
    ):
        _slice_attr(name, bs)

    block_tables = getattr(md, "block_tables", None)
    if torch.is_tensor(block_tables):
        md.block_tables = block_tables[:bs]

    for name in ("cu_seqlens_q", "cu_seqlens_k"):
        _slice_attr(name, bs + 1)

    indexer_meta = getattr(md, "indexer_meta", None)
    if isinstance(indexer_meta, dict):
        indexer_meta = dict(indexer_meta)
        for key in (
            "batch_id_per_token_gpu",
            "seq_base_per_token_gpu",
            "cu_starts_gpu",
            "cu_ends_gpu",
        ):
            value = indexer_meta.get(key)
            if torch.is_tensor(value):
                indexer_meta[key] = value[:num_tokens]
        value = indexer_meta.get("n_committed_per_seq_gpu")
        if torch.is_tensor(value):
            indexer_meta["n_committed_per_seq_gpu"] = value[:bs]
        md.indexer_meta = indexer_meta

    return md


def _is_current_stream_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - HIP compatibility fallback
        return False


def _get_sglang_attention_backend():
    try:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        return get_attn_backend()
    except Exception:  # noqa: BLE001 - SGLang version compatibility
        return None


def _build_glm52_dsa_metadata(
    atom_config: Any,
    forward_batch: ForwardBatch,
    positions: torch.Tensor,
):
    hf_config = getattr(atom_config, "hf_config", None)
    if _is_dummy_forward(forward_batch) or hf_config is None:
        return None
    if bool(getattr(forward_batch, "_atom_glm52_generic_draft_frontend", False)):
        return None

    from atom.plugin.sglang.runtime.model_arch import is_glm52_dsa_config

    if not is_glm52_dsa_config(hf_config):
        return None

    from atom.plugin.sglang.glm52_dsa_bridge import (
        build_atom_glm52_attention_metadata_from_sglang,
        get_draft_decode_sub_step,
        is_draft_decode_metadata,
    )

    is_capture_batch = _is_current_stream_capturing()
    from atom.plugin.sglang.runtime.attention_backend_resolver import (
        resolve_sglang_runtime,
    )

    runtime_objects = resolve_sglang_runtime(forward_batch)
    backend = runtime_objects.attn_backend
    token_to_kv_pool = runtime_objects.token_to_kv_pool
    req_to_token_pool = runtime_objects.req_to_token_pool
    is_draft_decode = is_draft_decode_metadata(forward_batch)
    is_target_verify = bool(
        getattr(forward_batch.forward_mode, "is_target_verify", lambda: False)()
    )
    is_plain_decode = (
        forward_batch.forward_mode.is_decode_or_idle() and not is_draft_decode
    )
    attn_metadata = (
        getattr(forward_batch, "atom_glm52_graph_metadata", None)
        if is_target_verify or is_plain_decode
        else None
    )
    if attn_metadata is None and is_plain_decode:
        attn_metadata = getattr(backend, "atom_glm52_graph_metadata", None)
    if attn_metadata is None and is_target_verify and is_capture_batch:
        batch_backend = getattr(forward_batch, "attn_backend", None)
        attn_metadata = getattr(batch_backend, "atom_glm52_graph_metadata", None)
    if attn_metadata is None and is_target_verify and is_capture_batch:
        attn_metadata = getattr(backend, "atom_glm52_graph_metadata", None)
    graph_cache = None
    graph_cache_key = None
    if is_draft_decode and token_to_kv_pool is not None:
        graph_cache = getattr(
            token_to_kv_pool, "_atom_glm52_draft_decode_graph_metadata", None
        )
        graph_cache_key = (
            int(forward_batch.batch_size),
            get_draft_decode_sub_step(forward_batch),
        )
        if is_capture_batch:
            cached_metadata = (
                graph_cache.get(graph_cache_key) if graph_cache is not None else None
            )
            if cached_metadata is None:
                raise RuntimeError(
                    "Missing fixed GLM-5.2 draft graph metadata for "
                    f"batch/substep={graph_cache_key}"
                )
            attn_metadata = cached_metadata
    elif attn_metadata is None and is_capture_batch and is_target_verify:
        from atom.plugin.sglang.attention_backend.glm52_dsa_backend import (
            ATOMGLM52DSABackendForSgl,
        )

        attn_metadata = ATOMGLM52DSABackendForSgl._last_atom_glm52_graph_metadata
    if (
        attn_metadata is None
        and token_to_kv_pool is not None
        and req_to_token_pool is not None
    ):
        if is_capture_batch:
            raise RuntimeError(
                "ATOM GLM-5.2 CUDA graph metadata was not initialized before capture"
            )
        attn_metadata = build_atom_glm52_attention_metadata_from_sglang(
            forward_batch,
            positions,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token_pool=req_to_token_pool,
            atom_config=atom_config,
        )
        if is_draft_decode:
            try:
                from sglang.srt.model_executor.cuda_graph_runner import (
                    get_is_capture_mode,
                )

                in_graph_warmup = get_is_capture_mode()
            except Exception:  # noqa: BLE001 - SGLang version compatibility
                in_graph_warmup = False
            if in_graph_warmup and graph_cache_key is not None:
                if graph_cache is None:
                    graph_cache = {}
                    token_to_kv_pool._atom_glm52_draft_decode_graph_metadata = (
                        graph_cache
                    )
                graph_cache[graph_cache_key] = attn_metadata
    return attn_metadata


def stage_glm52_draft_decode_graph_metadata(
    forward_batch: ForwardBatch,
    *,
    speculative_num_steps: int,
    topk: int,
) -> None:
    """Stage current draft-decode routing into fixed graph metadata tensors."""
    from atom.config import get_current_atom_config
    from atom.plugin.sglang.glm52_dsa_bridge import (
        build_atom_glm52_attention_metadata_from_sglang,
        clear_draft_decode_sub_step,
        set_draft_decode_sub_step,
    )
    from atom.plugin.sglang.runtime.attention_backend_resolver import (
        resolve_sglang_runtime,
    )

    runtime_objects = resolve_sglang_runtime(forward_batch)
    token_to_kv_pool = runtime_objects.token_to_kv_pool
    req_to_token_pool = runtime_objects.req_to_token_pool
    cache = getattr(token_to_kv_pool, "_atom_glm52_draft_decode_graph_metadata", None)
    if not cache:
        raise RuntimeError("GLM-5.2 draft-decode graph metadata cache is empty")

    bs = int(forward_batch.batch_size)
    original_out_cache_loc = forward_batch.out_cache_loc
    original_positions = forward_batch.positions
    out_cache_rows = (
        original_out_cache_loc.reshape(bs, int(topk), int(speculative_num_steps))
        .permute(2, 0, 1)
        .reshape(int(speculative_num_steps), -1)
    )
    tensor_fields = (
        "cu_seqlens_q",
        "cu_seqlens_k",
        "slot_mapping",
        "context_lens",
        "block_tables",
        "kv_indptr",
        "kv_indices",
        "kv_last_page_lens",
        "sparse_kv_indptr",
        "sparse_kv_last_page_lens",
        "sparse_cu_seqlens_q",
        "token_to_seq_idxs",
        "work_meta_data",
        "work_indptr",
        "work_info_set",
        "reduce_indptr",
        "reduce_final_map",
        "reduce_partial_map",
    )

    try:
        for sub_step in range(int(speculative_num_steps) - 1):
            set_draft_decode_sub_step(forward_batch, sub_step)
            forward_batch.out_cache_loc = out_cache_rows[sub_step]
            forward_batch.positions = original_positions + sub_step
            staged = build_atom_glm52_attention_metadata_from_sglang(
                forward_batch,
                forward_batch.positions,
                token_to_kv_pool=token_to_kv_pool,
                req_to_token_pool=req_to_token_pool,
                atom_config=get_current_atom_config(),
            )
            fixed = cache.get((bs, sub_step))
            if fixed is None:
                raise RuntimeError(
                    "Missing GLM-5.2 draft graph metadata for "
                    f"batch_size={bs}, sub_step={sub_step}"
                )
            for name in tensor_fields:
                source = getattr(staged, name, None)
                target = getattr(fixed, name, None)
                if not torch.is_tensor(source) or not torch.is_tensor(target):
                    continue
                if source.numel() > target.numel():
                    raise RuntimeError(
                        f"GLM-5.2 draft graph metadata field {name} exceeds "
                        f"capture capacity: runtime={tuple(source.shape)} "
                        f"capture={tuple(target.shape)}"
                    )
                target_flat = target.reshape(-1)
                source_flat = source.reshape(-1)
                target_flat.zero_()
                target_flat[: source_flat.numel()].copy_(source_flat)
    finally:
        clear_draft_decode_sub_step(forward_batch)
        forward_batch.out_cache_loc = original_out_cache_loc
        forward_batch.positions = original_positions


def _build_minimax_m3_metadata(
    atom_config: Any,
    forward_batch: ForwardBatch,
    positions: torch.Tensor,
):
    hf_config = getattr(atom_config, "hf_config", None)
    if _is_dummy_forward(forward_batch) or hf_config is None:
        return None

    from atom.plugin.sglang.minimax_m3_bridge import (
        build_atom_minimax_m3_attention_metadata_from_sglang,
        is_minimax_m3_config,
        maybe_get_minimax_m3_pools_from_sglang_batch,
    )

    if not is_minimax_m3_config(hf_config):
        return None

    token_to_kv_pool, req_to_token_pool = maybe_get_minimax_m3_pools_from_sglang_batch(
        forward_batch
    )
    if token_to_kv_pool is None or req_to_token_pool is None:
        return None

    return build_atom_minimax_m3_attention_metadata_from_sglang(
        forward_batch,
        positions,
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
        max_model_len=int(atom_config.max_model_len),
    )


def _build_deepseek_v4_metadata(forward_batch: ForwardBatch, positions: torch.Tensor):
    backend = None
    attn_metadata = getattr(forward_batch, "atom_v4_graph_metadata", None)
    from atom.plugin.sglang.deepseek_v4_bridge import (
        build_atom_v4_attention_metadata_from_sglang,
        maybe_get_proxy_pool_from_sglang_backend,
    )

    if attn_metadata is None:
        backend = _get_sglang_attention_backend()
        attn_metadata = getattr(backend, "atom_v4_graph_metadata", None)

    if attn_metadata is None:
        backend = getattr(forward_batch, "attn_backend", None)
        attn_metadata = getattr(backend, "atom_v4_graph_metadata", None)

    if attn_metadata is None and backend is not None:
        backend_forward_batch = getattr(backend, "forward_metadata", None)
        attn_metadata = getattr(backend_forward_batch, "atom_v4_graph_metadata", None)

    proxy_pool, req_to_token_pool = maybe_get_proxy_pool_from_sglang_backend()

    is_capture_batch = _is_current_stream_capturing()
    if attn_metadata is None and is_capture_batch:
        try:
            from atom.plugin.sglang.attention_backend.deepseek_v4_backend import (
                ATOMDeepseekV4BackendForSgl,
            )

            attn_metadata = ATOMDeepseekV4BackendForSgl._last_atom_v4_graph_metadata
            if attn_metadata is not None:
                attn_metadata = _slice_v4_graph_metadata_for_capture(
                    attn_metadata,
                    num_tokens=int(positions.shape[0]),
                    bs=int(forward_batch.batch_size),
                )
        except Exception:  # noqa: BLE001 - optional V4 backend
            attn_metadata = None

    if attn_metadata is None and getattr(proxy_pool, "is_atom_v4_proxy_pool", False):
        if is_capture_batch:
            raise RuntimeError(
                "ATOM DeepSeek-V4 CUDA graph metadata was not initialized before capture"
            )
        attn_metadata = build_atom_v4_attention_metadata_from_sglang(
            forward_batch,
            positions,
            proxy_pool=proxy_pool,
            req_to_token_pool=req_to_token_pool,
        )
    return attn_metadata


def _build_eagle3_llama_metadata(
    atom_config: Any, forward_batch: ForwardBatch, positions: torch.Tensor
):
    hf_config = getattr(atom_config, "hf_config", None)
    if _is_dummy_forward(forward_batch) or hf_config is None:
        return None

    from atom.plugin.sglang.eagle3_llama_bridge import (
        build_atom_eagle3_attention_metadata_from_sglang,
        is_eagle3_llama_config,
        maybe_get_eagle3_pools_from_sglang_batch,
    )

    if not is_eagle3_llama_config(hf_config):
        return None

    token_to_kv_pool, req_to_token_pool = maybe_get_eagle3_pools_from_sglang_batch(
        forward_batch
    )
    if token_to_kv_pool is None or req_to_token_pool is None:
        return None

    return build_atom_eagle3_attention_metadata_from_sglang(
        forward_batch,
        positions,
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
    )


def _build_kimi_k3_metadata(
    atom_config: Any, forward_batch: ForwardBatch, positions: torch.Tensor
):
    from atom.plugin.sglang.kimi_k3_bridge import (
        build_kimi_k3_attention_metadata,
        maybe_get_kimi_k3_pools,
    )

    attn_metadata = getattr(forward_batch, "atom_kimi_k3_graph_metadata", None)
    if attn_metadata is None:
        backend = _get_sglang_attention_backend()
        attn_metadata = getattr(backend, "atom_kimi_k3_graph_metadata", None)
    if attn_metadata is None and _is_current_stream_capturing():
        from atom.plugin.sglang.attention_backend.kimi_k3_backend import (
            ATOMKimiK3BackendForSgl,
        )

        attn_metadata = ATOMKimiK3BackendForSgl._last_atom_kimi_k3_graph_metadata

    token_to_kv_pool, req_to_token_pool = maybe_get_kimi_k3_pools(forward_batch)
    if token_to_kv_pool is None or req_to_token_pool is None:
        raise RuntimeError("Kimi-K3 SGLang pools are unavailable")
    if attn_metadata is None:
        attn_metadata = build_kimi_k3_attention_metadata(
            forward_batch,
            positions,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token_pool=req_to_token_pool,
        )

    from atom.plugin.sglang.attention_backend.attention_gdn import (
        SGLangGDNForwardContext,
    )

    attn_backend = SGLangGDNForwardContext._resolve_attn_backend(forward_batch)
    if forward_batch.forward_mode.is_decode_or_idle():
        full_attn_backend = getattr(attn_backend, "full_attn_backend", attn_backend)
        forward_metadata = getattr(full_attn_backend, "forward_metadata", None)
        kv_indices = getattr(forward_metadata, "kv_indices", None)
        if kv_indices is None:
            raise RuntimeError(
                "Kimi-K3 decode metadata has no KV indices; "
                f"backend={type(full_attn_backend).__name__}, "
                f"forward_metadata={forward_metadata is not None}"
            )
        attn_metadata.kv_indptr = forward_metadata.kv_indptr
        attn_metadata.kv_indices = kv_indices
        attn_metadata.kv_last_page_lens = getattr(
            forward_metadata, "kv_last_page_len", None
        )
        for name in (
            "work_meta_data",
            "work_info_set",
            "work_indptr",
            "reduce_indptr",
            "reduce_final_map",
            "reduce_partial_map",
            "num_kv_splits",
        ):
            setattr(attn_metadata, name, getattr(forward_metadata, name, None))

    linear_backend = SGLangGDNForwardContext._linear_attn_backend(attn_backend)
    attn_metadata.gdn_metadata = SGLangGDNForwardContext._build_gdn_metadata(
        forward_batch, linear_backend
    )
    return attn_metadata


def _set_atom_forward_context(
    atom_config: Any,
    forward_batch: ForwardBatch,
    positions: torch.Tensor,
) -> None:
    """Bridge SGLang batch metadata into ATOM's global forward context."""

    from atom.utils.forward_context import (
        Context,
        set_forward_context,
    )

    forward_mode = forward_batch.forward_mode
    is_target_verify = bool(getattr(forward_mode, "is_target_verify", lambda: False)())
    is_draft_extend = bool(
        getattr(forward_mode, "is_draft_extend", lambda **kwargs: False)(
            include_v2=True
        )
    )
    from atom.plugin.sglang.runtime.model_arch import resolve_model_arch_spec

    hf_config = getattr(atom_config, "hf_config", None)
    model_arch, model_adapter = resolve_model_arch_spec(hf_config)
    attn_metadata = None
    if model_adapter.build_forward_metadata is not None:
        try:
            attn_metadata = model_adapter.build_forward_metadata(
                atom_config, forward_batch, positions
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to build ATOM metadata for SGLang model "
                f"{model_arch or '<unknown>'}"
            ) from exc

    if attn_metadata is None:
        # Model-specific builders own their query geometry. This fallback value
        # is only used by generic metadata for ATOM-side MoE padding.
        from atom.plugin.sglang.runtime.model_arch import is_glm52_dsa_config

        is_glm52_dsa = is_glm52_dsa_config(getattr(atom_config, "hf_config", None))
        if is_glm52_dsa and is_target_verify:
            draft_token_num = int(
                getattr(getattr(forward_batch, "spec_info", None), "draft_token_num", 0)
                or 0
            )
            max_seqlen_q = max(1, draft_token_num)
        elif is_glm52_dsa and is_draft_extend:
            from atom.plugin.sglang.glm52_dsa_bridge import draft_extend_token_num

            max_seqlen_q = max(
                1,
                draft_extend_token_num(
                    forward_batch, positions, int(forward_batch.batch_size)
                ),
            )
        else:
            max_seqlen_q = 1 if forward_mode.is_decode_or_idle() else 0
        attn_metadata = _build_generic_attention_metadata(forward_batch, max_seqlen_q)
    batch_size = int(forward_batch.batch_size)
    is_dummy_run = _is_dummy_forward(forward_batch)
    is_prefill = forward_mode.is_prefill()
    if is_target_verify or is_draft_extend:
        from atom.utils.forward_context import AttnState

        verify_state = getattr(attn_metadata, "state", None)
        is_prefill = verify_state in (
            AttnState.PREFILL_PREFIX,
            AttnState.PREFILL_NATIVE,
        )
    # Qwen-VL mRoPE positions use [3, num_tokens], while ordinary position
    # tensors use [num_tokens]. The token dimension is therefore the last
    # dimension for mRoPE rather than the leading coordinate dimension.
    num_tokens = int(
        positions.shape[-1]
        if positions.ndim == 2 and positions.shape[0] == 3
        else positions.shape[0]
    )

    max_q = int(getattr(attn_metadata, "max_seqlen_q", 1) or 1)
    enable_dp_attention = bool(atom_config.enable_dp_attention)
    if enable_dp_attention:
        num_tokens_across_dp = _resolve_num_tokens_across_dp(
            atom_config, forward_batch, num_tokens
        )
        # Already TOKENS, and already the group max -- what MoE pads to.
        running_tokens = int(torch.max(num_tokens_across_dp).item())
    else:
        num_tokens_across_dp = None
        running_tokens = num_tokens if is_prefill else batch_size * max_q
    # Sequences, per the field's contract -- the TBO split divides this into
    # per-ubatch request counts, so handing it a token count would make each
    # ubatch `max_seqlen_q` times too wide. Prefill's value is unread there.
    running_bs = batch_size if is_prefill else max(1, running_tokens // max_q)

    running_tokens_are_unified = _resolve_running_tokens_are_unified(
        atom_config, forward_batch
    )
    context = Context(
        positions=positions,
        is_prefill=is_prefill,
        is_dummy_run=is_dummy_run,
        scheduled_bs=batch_size,
        scheduled_tokens=num_tokens,
        running_bs=running_bs,
        running_tokens=running_tokens,
        running_tokens_are_unified=running_tokens_are_unified,
    )
    set_forward_context(
        attn_metadata=attn_metadata,
        atom_config=atom_config,
        context=context,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    )


def _reset_atom_forward_context() -> None:
    from atom.utils.forward_context import reset_forward_context

    reset_forward_context()


@dataclass
class SGLangPluginRuntime:
    """Scoped adapter for running ATOM model code under SGLang plugin runtime.

    The adapter owns the temporary translation from SGLang's ``ForwardBatch`` to
    ATOM's process-local runtime state.  Callers should use the normalized
    ``input_ids``, ``positions``, ``input_embeds``, and ``forward_batch`` exposed
    by this object while inside the context.
    """

    atom_config: Any
    forward_batch: ForwardBatch
    positions: torch.Tensor
    input_ids: torch.Tensor | None = None
    input_embeds: torch.Tensor | None = None
    set_forward_context: bool = True
    _original_forward_batch: ForwardBatch = field(init=False, repr=False)
    _is_dummy_run: bool = field(init=False, default=False)
    _exit_stack: ExitStack = field(init=False, repr=False)

    def __enter__(self) -> SGLangPluginRuntime:  # noqa: PYI034
        self._original_forward_batch = self.forward_batch
        self._is_dummy_run = _is_dummy_forward(self.forward_batch)

        if self._is_dummy_run:
            (
                self.input_ids,
                self.positions,
                self.input_embeds,
                self.forward_batch,
            ) = _materialize_atom_dummy_forward(
                self.input_ids,
                self.positions,
                self.input_embeds,
                self.forward_batch,
            )

        self._exit_stack = ExitStack()
        self._exit_stack.enter_context(bind_current_forward_batch(self.forward_batch))
        if self.set_forward_context:
            _set_atom_forward_context(
                self.atom_config,
                self.forward_batch,
                self.positions,
            )
            self._exit_stack.callback(_reset_atom_forward_context)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._exit_stack.close()

    def trim_output(self, hidden_states):
        """Map ATOM-visible outputs back to SGLang-visible token count."""

        if self._is_dummy_run:
            return _trim_hidden_states_for_output(hidden_states, 0)
        return hidden_states
