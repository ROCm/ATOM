"""Optional AITER FlyDSL GDN for the standalone Qwen4Exp backend.

ATOM exposes logical KV state views backed by VK storage. No per-token state
packing, pool-wide transpose, or device-to-host synchronization is required.
"""

import functools
import logging
import os

import torch
import triton

from .chunk_o import chunk_fwd_o
from .l2norm import l2norm_fwd

logger = logging.getLogger(__name__)


@functools.cache
def _log_dispatch(stage):
    logger.info("Qwen4Exp GDN %s: using AITER FlyDSL", stage)


@functools.cache
def backend(stage):
    value = os.getenv(f"ATOM_GDN_{stage.upper()}_BACKEND", "auto")
    if value not in ("auto", "triton", "flydsl"):
        raise ValueError(f"Invalid ATOM GDN {stage} backend: {value}")
    return value


@functools.cache
def ops():
    if torch.version.hip is None:
        return None
    try:
        from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode
        from aiter.ops.flydsl.linear_attention_prefill_kernels import (
            chunk_gated_delta_rule_fwd_h_flydsl_opt,
            gdn_prepare_flydsl_supported,
            gdn_prepare_fwd_flydsl,
        )
        from aiter.ops.prefill_batch_metadata import (
            build_gated_delta_rule_prefill_metadata,
        )

        return (
            flydsl_gdr_decode,
            gdn_prepare_fwd_flydsl,
            chunk_gated_delta_rule_fwd_h_flydsl_opt,
            build_gated_delta_rule_prefill_metadata,
            gdn_prepare_flydsl_supported,
        )
    except (ImportError, AttributeError, RuntimeError, OSError) as error:
        logger.warning("AITER FlyDSL GDN unavailable; keeping Triton: %s", error)
        return None


def build_prefill_metadata(lengths, cu_seqlens):
    if backend("prefill") == "triton" or ops() is None:
        return None
    lengths = tuple(int(n) for n in lengths)
    metadata = ops()[3](lengths, cu_seqlens=cu_seqlens, chunk_size=64)
    # Prepared once per scheduler step, shared by every layer. No device->host
    # length reads inside K1-K6 or per-layer schedule reconstruction.
    pairs = [(i, j) for i, n in enumerate(lengths) for j in range(triton.cdiv(n, 64))]
    chunks = torch.tensor(
        pairs, device=cu_seqlens.device, dtype=cu_seqlens.dtype
    ).reshape(-1, 2)
    return metadata, chunks


def prefill_supported(q, k, v, g, beta, metadata):
    return (
        backend("prefill") != "triton"
        and metadata is not None
        and ops() is not None
        and q.ndim == 4
        and q.shape == k.shape
        and q.shape[0] == 1
        and q.shape[1] > 0
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and v.shape[:2] == q.shape[:2]
        and v.shape[-2] % q.shape[-2] == 0
        and g.shape == beta.shape == v.shape[:-1]
        and all(t.device == q.device for t in (k, v, g, beta))
        and ops()[4](k, v, BT=64)
    )


def prefill(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    cu_seqlens,
    metadata,
    keep_intermediate_states=False,
):
    """Return ATOM token-major output, FP32 final KV state and optional BF16 h."""
    if not prefill_supported(q, k, v, g, beta, metadata):
        raise ValueError("Unsupported AITER FlyDSL prefill shape/metadata")
    _log_dispatch("prefill K1-K5 (ATOM K6)")
    schedule, chunks = metadata
    q, k = l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous())
    w, u, gc = ops()[1](
        k=k,
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        cu_seqlens=cu_seqlens,
        BT=64,
        use_exp2=True,
        prefill_metadata=schedule,
    )
    # Baseline accumulates and returns FP32 final state, but snapshots are BF16.
    h0 = initial_state.transpose(-1, -2).to(torch.float32).contiguous()
    h, vn, ht = ops()[2](
        k=k,
        w=w,
        u=u,
        g=gc,
        initial_state=h0,
        output_final_state=True,
        chunk_size=64,
        cu_seqlens=cu_seqlens,
        state_dtype=torch.float32,
        snapshot_dtype=torch.bfloat16,
        use_exp2=True,
        g_head_major=True,
        bf16_convert_trunc=False,
        prefill_metadata=schedule,
    )
    output = chunk_fwd_o(
        q, k, vn, h, gc, cu_seqlens=cu_seqlens, head_major_vk=True, chunk_indices=chunks
    )
    snapshots = h.transpose(-1, -2).contiguous() if keep_intermediate_states else None
    return output, ht.transpose(-1, -2), snapshots


def decode_supported(q, k, v, a, b, state, A_log, dt_bias, reads, writes):
    if backend("decode") == "triton" or ops() is None:
        return False
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] == 0:
        return False
    batch, hk, dim = q.shape[1:]
    if (
        k.shape != q.shape
        or v.shape[:2] != q.shape[:2]
        or dim != 128
        or v.shape[-1] != 128
    ):
        return False
    hv = v.shape[-2]
    return (
        q.is_cuda
        and hv % hk == 0
        and q.dtype == torch.bfloat16
        and all(t.dtype == q.dtype for t in (k, v, a, b, dt_bias))
        and a.shape == b.shape == (batch, hv)
        and state.shape[1:] == (hv, 128, 128)
        and state.stride()[1:] == (128 * 128, 1, 128)
        and state.dtype in (torch.float32, torch.bfloat16)
        and A_log.shape == dt_bias.shape == (hv,)
        and A_log.dtype in (torch.float32, torch.bfloat16)
        and all(
            t is not None and t.device == q.device
            for t in (k, v, a, b, state, A_log, dt_bias, reads, writes)
        )
        and reads.shape == writes.shape == (batch,)
        and reads.dtype == writes.dtype == torch.int32
        and reads.is_contiguous()
        and writes.is_contiguous()
        and q.stride(-1) == k.stride(-1) == 1
        and torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0]
        == "gfx942"
    )


def decode(q, k, v, a, b, state, A_log, dt_bias, reads, writes):
    """Fused gating/normalization/recurrence, zero-copy logical KV state view."""
    if not decode_supported(q, k, v, a, b, state, A_log, dt_bias, reads, writes):
        raise ValueError("Unsupported AITER FlyDSL decode inputs")
    _log_dispatch("decode (zero-copy VK state)")
    batch, hv = v.shape[1:3]
    fn = ops()[0]
    allocate = (
        torch.empty if getattr(fn, "zeroes_invalid_output", False) else torch.zeros
    )
    output = allocate((batch, 1, hv, 128), device=q.device, dtype=q.dtype)
    fn(
        query=q.transpose(0, 1),
        key=k.transpose(0, 1),
        value=v.transpose(0, 1),
        a=a.unsqueeze(1),
        b=b.unsqueeze(1),
        dt_bias=dt_bias,
        A_log=A_log,
        indices=writes,
        read_indices=reads,
        write_indices=writes,
        state=state.transpose(-1, -2),
        out=output,
        use_qk_l2norm=True,
        need_shuffle_state=False,
    )
    return output.transpose(0, 1), state
