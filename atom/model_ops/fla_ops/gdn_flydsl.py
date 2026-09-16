"""Optional AITER FlyDSL GDN for the standalone Qwen4Exp backend.

ATOM exposes logical KV state views backed by VK storage. No per-token state
packing, pool-wide transpose, or device-to-host synchronization is required.
"""

import functools
import logging
import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .chunk_o import chunk_fwd_o
from .l2norm import l2norm_fwd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GDNFlyDSLPolicy:
    """Resolved at cache binding; decode and physical VK layout are inseparable."""

    prefill: bool = False
    decode: bool = False


def decode_device_supported(device):
    return (
        torch.version.hip is not None
        and device.type == "cuda"
        and torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        == "gfx942"
    )


def select_policy(*, allowed, replayssm, lossy_decode, state, activation_dtype):
    # ReplaySSM kernels address a contiguous KV pool. Do not reinterpret it as
    # VK even when K == V makes the shape identical. Keep both stages original.
    if not allowed or replayssm:
        return GDNFlyDSLPolicy()
    prefill_enabled = backend("prefill") != "triton"
    decode_enabled = (
        backend("decode") != "triton"
        and not lossy_decode
        and state.ndim == 4
        and state.shape[-2:] == (128, 128)
        and state.dtype in (torch.bfloat16, torch.float32)
        and activation_dtype == torch.bfloat16
        and decode_device_supported(state.device)
    )
    if not (prefill_enabled or decode_enabled) or ops() is None:
        return GDNFlyDSLPolicy()
    return GDNFlyDSLPolicy(prefill_enabled, decode_enabled)


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


@triton.jit
def _prepare_prefill_state_kernel(
    pool,
    indices,
    valid,
    output,
    S0: tl.constexpr,
    SH: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    INDEXED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = seq
    live = tl.full((), True, tl.int1)
    if INDEXED:
        slot = tl.load(indices + seq).to(tl.int64)
        live = tl.load(valid + seq)
    # Output is contiguous VK; source is a logical KV view with explicit strides.
    h = (x // (V * K)).to(tl.int64)
    v = ((x // K) % V).to(tl.int64)
    k = (x % K).to(tl.int64)
    offset = slot.to(tl.int64) * S0 + h * SH + k * SK + v * SV
    value = tl.load(pool + offset, (x < H * V * K) & live, other=0)
    tl.store(
        output + seq.to(tl.int64) * H * V * K + x, value.to(tl.float32), x < H * V * K
    )


def prepare_prefill_state(state, indices=None, has_initial_state=None):
    """Gather/zero/cast a logical KV pool into FP32 VK in one GPU launch.

    Live indices must address valid pool slots, as for the original gather.
    Cold rows are masked loads, not NaN-prone multiplication by zero.
    """
    if state.ndim != 4 or state.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Expected a BF16/FP32 [slots,H,K,V] state tensor")
    if (indices is None) != (has_initial_state is None):
        raise ValueError("indices and has_initial_state must be supplied together")
    if indices is not None and (
        indices.ndim != 1
        or has_initial_state.shape != indices.shape
        or indices.dtype not in (torch.int32, torch.int64)
        or has_initial_state.dtype != torch.bool
        or not indices.is_contiguous()
        or not has_initial_state.is_contiguous()
        or indices.device != state.device
        or has_initial_state.device != state.device
    ):
        raise ValueError("Expected same-device contiguous indices and boolean flags")
    n = state.shape[0] if indices is None else indices.numel()
    _, h, k, v = state.shape
    output = torch.empty((n, h, v, k), device=state.device, dtype=torch.float32)
    if n:
        _prepare_prefill_state_kernel[(n, triton.cdiv(h * k * v, 1024))](
            state,
            indices,
            has_initial_state,
            output,
            *state.stride(),
            h,
            k,
            v,
            indices is not None,
            1024,
        )
    return output


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
    *,
    state_indices=None,
    has_initial_state=None,
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
    h0 = prepare_prefill_state(initial_state, state_indices, has_initial_state)
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
        and decode_device_supported(q.device)
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
