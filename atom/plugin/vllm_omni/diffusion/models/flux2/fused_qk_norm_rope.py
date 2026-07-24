from collections.abc import Callable
import os
import torch

FusedQKNormRoPE2Way = Callable[..., None]


def _fusion_enabled() -> bool:
    return os.getenv("ATOM_FLUX2_FUSED_QK_NORM_ROPE", "1").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _load_fused_qk_norm_rope_2way() -> FusedQKNormRoPE2Way | None:
    try:
        from aiter import fused_qk_norm_rope_2way
    except ImportError:
        return None
    return fused_qk_norm_rope_2way


def _norm_weight(norm, dtype: torch.dtype) -> torch.Tensor | None:
    weight = getattr(norm, "weight", None)
    if weight is None or weight.dtype != dtype:
        return None
    return weight.contiguous()


def _norm_eps(norm) -> float:
    return float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))


def _as_2d_rotary(tensor: torch.Tensor) -> torch.Tensor | None:
    if tensor.dim() == 2:
        return tensor
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        return tensor[0]
    return None


def _pack_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    start: int,
    end: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    cos = _as_2d_rotary(cos)
    sin = _as_2d_rotary(sin)
    if cos is None or sin is None:
        return None
    if cos.shape != sin.shape or end > cos.shape[0]:
        return None
    return torch.cat(
        [cos[start:end].to(dtype), sin[start:end].to(dtype)], dim=-1
    ).contiguous()


def try_fused_qk_norm_rope_2way(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    norm_q,
    norm_k,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    is_interleaved: bool,
    encoder_query: torch.Tensor | None = None,
    encoder_key: torch.Tensor | None = None,
    norm_added_q=None,
    norm_added_k=None,
    fused_op: FusedQKNormRoPE2Way | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run aiter's fused RMSNorm+RoPE kernel when ATOM Flux2 shapes fit it.
    The 2-way kernel writes stream 0 followed by stream 1 on the token axis.
    For double-stream Flux2 attention, stream 0 is text/context and stream 1 is
    image. For single-stream attention, stream 1 is empty.
    """
    if not _fusion_enabled():
        return None
    if image_rotary_emb is None:
        return None
    if query.dim() != 4 or key.dim() != 4:
        return None
    if query.device != key.device or (fused_op is None and not query.is_cuda):
        return None
    if query.dtype != key.dtype:
        return None
    batch_size_q, num_tokens_q, num_heads_q, head_size = query.shape
    batch_size_k, num_tokens_k, num_heads_k, _ = key.shape
    if (
        batch_size_k != batch_size_q
        or num_tokens_k != num_tokens_q
        or key.shape[3] != head_size
        or head_size not in (64, 128, 256)
    ):
        return None
    batch_size = batch_size_q
    has_context = encoder_query is not None or encoder_key is not None
    if has_context:
        if encoder_query is None or encoder_key is None:
            return None
        if norm_added_q is None or norm_added_k is None:
            return None
        if encoder_query.dim() != 4 or encoder_key.dim() != 4:
            return None
        if encoder_query.dtype != query.dtype or encoder_key.dtype != key.dtype:
            return None
        if encoder_query.device != query.device or encoder_key.device != key.device:
            return None
        if encoder_query.shape[0] != batch_size or encoder_key.shape[0] != batch_size:
            return None
        if encoder_query.shape[2:] != (num_heads_q, head_size):
            return None
        if encoder_key.shape[2:] != (num_heads_k, head_size):
            return None
        if encoder_query.shape[1] != encoder_key.shape[1]:
            return None
        q0, k0 = encoder_query, encoder_key
        q1, k1 = query, key
        w_q0 = _norm_weight(norm_added_q, query.dtype)
        w_k0 = _norm_weight(norm_added_k, key.dtype)
        w_q1 = _norm_weight(norm_q, query.dtype)
        w_k1 = _norm_weight(norm_k, key.dtype)
        num_tokens0 = encoder_query.shape[1]
        num_tokens1 = num_tokens_q
    else:
        q0, k0 = query, key
        q1 = query.new_empty((batch_size, 0, num_heads_q, head_size))
        k1 = key.new_empty((batch_size, 0, num_heads_k, head_size))
        w_q0 = _norm_weight(norm_q, query.dtype)
        w_k0 = _norm_weight(norm_k, key.dtype)
        w_q1 = w_q0
        w_k1 = w_k0
        num_tokens0 = num_tokens_q
        num_tokens1 = 0
    if w_q0 is None or w_k0 is None or w_q1 is None or w_k1 is None:
        return None
    cos, sin = image_rotary_emb
    total_tokens = num_tokens0 + num_tokens1
    cos_sin0 = _pack_cos_sin(cos, sin, 0, num_tokens0, query.dtype)
    cos_sin1 = _pack_cos_sin(cos, sin, num_tokens0, total_tokens, query.dtype)
    if cos_sin0 is None or cos_sin1 is None:
        return None
    if cos_sin0.shape[-1] != head_size or cos_sin1.shape[-1] != head_size:
        return None
    fused_op = fused_op or _load_fused_qk_norm_rope_2way()
    if fused_op is None:
        return None
    out_q = torch.empty(
        (batch_size, total_tokens, num_heads_q, head_size),
        dtype=query.dtype,
        device=query.device,
    )
    out_k = torch.empty(
        (batch_size, total_tokens, num_heads_k, head_size),
        dtype=key.dtype,
        device=key.device,
    )
    fused_op(
        q0.contiguous(),
        k0.contiguous(),
        q1.contiguous(),
        k1.contiguous(),
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        batch_size,
        num_tokens0,
        num_tokens1,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        _norm_eps(norm_q),
        out_q,
        out_k,
    )
    return out_q, out_k
