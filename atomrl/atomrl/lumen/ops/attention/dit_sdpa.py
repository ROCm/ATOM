"""Faster kernels behind ``torch.nn.functional.scaled_dot_product_attention`` on ROCm.

For models that call SDPA directly -- diffusers' native attention processor,
VeOmni's ``WanTransformer3DModel`` -- rather than through a Lumen attention
module. Two process-wide switches, the second built on the first:

:func:`prefer_efficient_backend`
    Turns SDPA's flash backend off so torch dispatches to mem-efficient. On
    ROCm the two differ in the backward: flash runs AOTriton's split
    ``dk_dv`` + ``dq`` pair, efficient runs aiter's fused ``fmha_bwd``.
    Measured on MI350X at the Qwen-Image DiT's attention (B=1 H=24 S=4109
    D=128 BF16, non-causal): backward 3.022 -> 1.231 ms, forward unchanged, both
    52.9 dB against FP32 attention. **Not deterministic**: the fused backward
    reduces dQ with FP32 atomics (``a32`` in the kernel name), so repeats of an
    otherwise bit-reproducible run differ (0.56% in the Qwen-Image loss).

:func:`install_triton_forward`
    Keeps that backward and replaces the forward with aiter's Triton flash
    forward, 1.7-1.8x faster than AOTriton's ``attn_fwd`` at both DiT shapes
    (Qwen-Image 776 -> 437 us, Wan2.1 4661 -> 2782 us) at the same accuracy.
    aiter's Triton *backward* is slower than the fused one, which is why the
    two are spliced rather than switched wholesale. The seam is the
    log-sum-exp: the efficient backward recomputes the softmax from the
    forward's output and LSE, and aiter's Triton forward returns a natural-log
    LSE in the ``(B, H, S)`` layout that backward expects (1.9e-6 from an FP32
    reference, the same as the efficient forward's own). Only calls that are
    unambiguously a DiT's self/cross attention are routed -- no mask, no
    dropout, not causal, no GQA, BF16, head_dim 128, query length >=
    ``min_seq`` -- so a text encoder's causal GQA attention and a VAE's
    head_dim-384 attention take the stock path unchanged.

Both look the function up on ``torch.nn.functional`` at call time, which is how
diffusers and VeOmni call it, so rebinding the module attribute is enough.
"""

import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_orig_sdpa = None
_min_seq = 1024
_stats = {"routed": 0, "stock": 0}


def prefer_efficient_backend():
    """Disable SDPA's flash backend; keep mem-efficient, and math as the last resort."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    # Left on for shapes the efficient kernel rejects (a text encoder's GQA may).
    torch.backends.cuda.enable_math_sdp(True)
    logger.info("sdpa: flash backend disabled, mem-efficient preferred")


def _triton_forward(q, k, v, scale):
    """q/k/v in SDPA layout (B, H, S, D); returns out in the same layout, and LSE (B, H, Sq)."""
    from aiter.ops.triton.mha import flash_attn_func

    qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
    out, lse = flash_attn_func(
        qs,
        ks,
        vs,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
        return_lse=True,
        backend="triton",
    )
    return out.transpose(1, 2), lse


class TritonFwdEfficientBwd(torch.autograd.Function):
    """aiter Triton flash forward, SDPA mem-efficient backward, joined by the LSE."""

    @staticmethod
    def forward(ctx, q, k, v, scale):
        out, lse = _triton_forward(q, k, v, scale)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        # Dropout is 0, so the philox state is never read; any int64 scalar will do.
        zero = torch.zeros((), dtype=torch.int64)
        dq, dk, dv, _ = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
            grad_out.contiguous() if grad_out.stride(-1) != 1 else grad_out,
            q,
            k,
            v,
            None,
            out,
            lse,
            zero,
            zero,
            0.0,
            [True, True, True, False],
            False,
            scale=ctx.scale,
        )
        return dq, dk, dv, None


def _eligible(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa):
    return (
        attn_mask is None
        and dropout_p == 0.0
        and not is_causal
        and not enable_gqa
        and q.dim() == 4
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and q.shape[-1] == 128
        and k.shape[-1] == 128
        and v.shape[-1] == 128
        and q.shape[1] == k.shape[1] == v.shape[1]
        and q.shape[2] >= _min_seq
        and q.is_cuda
    )


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Drop-in for ``F.scaled_dot_product_attention``; see :func:`install_triton_forward`."""
    if _eligible(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
        _stats["routed"] += 1
        s = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
        if torch.is_grad_enabled() and (
            query.requires_grad or key.requires_grad or value.requires_grad
        ):
            return TritonFwdEfficientBwd.apply(query, key, value, s)
        return _triton_forward(query, key, value, s)[0]
    _stats["stock"] += 1
    return _orig_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def install_triton_forward(min_seq=1024):
    """Route eligible ``F.scaled_dot_product_attention`` calls. Idempotent.

    ``min_seq`` is the shortest query routed. The default keeps the 13-token
    text-conditioning attention of an image DiT on the stock path.
    """
    global _orig_sdpa, _min_seq
    _min_seq = int(min_seq)
    if _orig_sdpa is not None:
        return
    # Fail here rather than on the first routed call, mid-step.
    from aiter.ops.triton.mha import flash_attn_func  # noqa: F401

    _orig_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = scaled_dot_product_attention
    logger.info(
        "sdpa: aiter Triton forward + mem-efficient backward for no-mask, non-causal, BF16, "
        "head_dim-128 calls with seq >= %d",
        _min_seq,
    )


def uninstall_triton_forward():
    global _orig_sdpa
    if _orig_sdpa is not None:
        F.scaled_dot_product_attention = _orig_sdpa
        _orig_sdpa = None


def stats():
    """Calls routed to the Triton forward and calls left on the stock SDPA."""
    return dict(_stats)


__all__ = [
    "TritonFwdEfficientBwd",
    "install_triton_forward",
    "prefer_efficient_backend",
    "scaled_dot_product_attention",
    "stats",
    "uninstall_triton_forward",
]
