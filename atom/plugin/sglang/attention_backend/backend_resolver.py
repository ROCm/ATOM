from typing import Any

import torch


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
    device = indices.device
    if mode.is_decode_or_idle():
        query_start_loc = torch.arange(
            0, batch_size + 1, dtype=torch.int32, device=device
        )
    elif mode.is_extend():
        query_start_loc = torch.empty(
            (batch_size + 1,), dtype=torch.int32, device=device
        )
        query_start_loc[:batch_size] = forward_batch.extend_start_loc
        query_start_loc[batch_size] = (
            forward_batch.extend_start_loc[-1] + forward_batch.extend_seq_lens[-1]
        )
    else:
        return None

    return query_start_loc, indices.to(dtype=torch.int32, device=device)
