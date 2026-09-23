from typing import Any

import torch


def real_batch_size(forward_batch: Any) -> int:
    """Live request count, excluding CUDA-graph / DP dummy rows.

    SGLang decode graphs pad to the next captured bucket. The replay view
    exposes that as ``num_padding`` (and sometimes ``_original_batch_size``
    for DP/MLP-sync). Dummy rows keep the previous request's
    ``req_pool_indices`` / page tables after a completion frees those pages;
    QSA/GDN must not read them.
    """
    bs = int(getattr(forward_batch, "batch_size", 0) or 0)
    real = bs
    orig = getattr(forward_batch, "_original_batch_size", None)
    if orig is not None:
        real = min(real, int(orig))
    pad = getattr(forward_batch, "num_padding", None)
    if pad:
        real = min(real, max(bs - int(pad), 0))
    return max(real, 0)


def resolve_attn_backend(forward_batch: Any) -> Any:
    try:
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            has_forward_context,
        )

        if has_forward_context():
            backend = get_attn_backend()
            if backend is not None:
                return backend
    except Exception:  # noqa: BLE001, S110 - forward context is optional
        pass

    return getattr(forward_batch, "attn_backend", None)


def resolve_mamba_req_pool(forward_batch: Any, linear_backend: Any) -> Any:
    token_pool = getattr(forward_batch, "token_to_kv_pool", None)
    candidates = (
        getattr(token_pool, "_atom_kimi_k3_req_pool", None),
        getattr(linear_backend, "req_to_token_pool", None),
        getattr(forward_batch, "req_to_token_pool", None),
    )
    for pool in candidates:
        if pool is not None and hasattr(pool, "get_mamba_indices"):
            return pool

    try:
        from sglang.srt.model_executor.forward_context import (
            get_req_to_token_pool,
            has_forward_context,
        )

        if has_forward_context():
            pool = get_req_to_token_pool()
            if pool is not None and hasattr(pool, "get_mamba_indices"):
                return pool
    except Exception:  # noqa: BLE001, S110 - forward context is optional
        pass
    return None


def reconstruct_linear_metadata(
    forward_batch: Any, linear_backend: Any
) -> tuple[torch.Tensor, torch.Tensor] | None:
    pool = resolve_mamba_req_pool(forward_batch, linear_backend)
    if pool is None:
        return None

    indices = pool.get_mamba_indices(forward_batch.req_pool_indices)
    translate = getattr(pool, "translate_mamba_indices", None)
    if translate is not None:
        indices = translate(indices)

    mode = forward_batch.forward_mode
    batch_size = forward_batch.batch_size
    live_bs = real_batch_size(forward_batch)
    device = indices.device
    if live_bs < indices.shape[0]:
        # Mark DP/MLP-sync and CUDA-graph padding rows so they cannot
        # read or write state (including a just-finished request's slot).
        indices = indices.clone()
        indices[live_bs:] = -1

    if mode.is_decode_or_idle():
        # Give each real decode request one token and every padded row zero tokens.
        query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        query_start_loc[live_bs + 1 :] = live_bs
    elif mode.is_extend():
        # Build variable-length query offsets using only real extend requests.
        query_start_loc = torch.empty(
            (batch_size + 1,), dtype=torch.int32, device=device
        )
        if live_bs:
            # End at the final real request instead of a synthetic padded row.
            query_start_loc[:live_bs] = forward_batch.extend_start_loc[:live_bs]
            end = (
                forward_batch.extend_start_loc[live_bs - 1]
                + forward_batch.extend_seq_lens[live_bs - 1]
            )
        else:
            # An empty real batch makes every synthetic row a zero-length query.
            end = 0
        query_start_loc[live_bs:] = end
    else:
        return None

    return query_start_loc, indices.to(dtype=torch.int32, device=device)
