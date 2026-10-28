# SPDX-License-Identifier: MIT
"""Engram post-projection gate and residual update.

One program handles one token/HC branch. Reductions and gate arithmetic stay
FP32; only the final residual is cast. The precomputed q*k weights retain the
original operation order. Disabling FP fusion avoids introducing an FMA at the
residual addition. Reduction trees/rsqrt can still differ from PyTorch.
"""

import torch
import triton
import triton.language as tl

from atom.utils import envs


@triton.jit
def _engram_post_wkv_kernel(
    hidden,
    kv,
    weight,
    token_mask,
    output,
    eps,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    LOCAL_HC: tl.constexpr,
    BRANCH_START: tl.constexpr,
    GATE_ONLY: tl.constexpr,
    KV_TP: tl.constexpr = 1,
):
    row = tl.program_id(0).to(tl.int64)
    token, local_branch = row // LOCAL_HC, row % LOCAL_HC
    branch = BRANCH_START + local_branch
    col = tl.arange(0, BLOCK)
    live = col < DIM
    residual = tl.load(hidden + (token * HC + branch) * DIM + col, live, 0).to(
        tl.float32
    )
    if KV_TP > 1:
        # Gathered rows contain [K_rank, V_rank] for each rank. Read that
        # layout directly, fusing reconstruction with the original gate.
        PART_KEY: tl.constexpr = HC * DIM // KV_TP
        PART_N: tl.constexpr = (HC + 1) * DIM // KV_TP
        key_idx = (
            token * (HC + 1) * DIM
            + ((branch * DIM + col) // PART_KEY) * PART_N
            + (branch * DIM + col) % PART_KEY
        )
    else:
        key_idx = token * (LOCAL_HC + 1) * DIM + local_branch * DIM + col
    key = tl.load(kv + key_idx, live, 0).to(tl.float32)
    w = tl.load(weight + branch * DIM + col, live, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(residual * residual, 0) / DIM + eps)
    rstd *= tl.rsqrt(tl.sum(key * key, 0) / DIM + eps)
    dot = tl.sum((residual * w) * key, 0) * rstd * (DIM**-0.5)
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6))
    # copysign, including negative zero (which a dot < 0 test would lose).
    signed = (magnitude.to(tl.uint32, bitcast=True) & 0x7FFFFFFF) | (
        dot.to(tl.uint32, bitcast=True) & 0x80000000
    )
    gate = tl.sigmoid(signed.to(tl.float32, bitcast=True))
    if HAS_MASK:
        gate = tl.where(tl.load(token_mask + token), gate, 0.0)
    if GATE_ONLY:
        tl.store(output + row, gate)
    else:
        if KV_TP > 1:
            PART_V: tl.constexpr = DIM // KV_TP
            value_idx = (
                token * (HC + 1) * DIM
                + (col // PART_V) * PART_N
                + PART_KEY
                + col % PART_V
            )
        else:
            value_idx = token * (HC + 1) * DIM + HC * DIM + col
        value = tl.load(kv + value_idx, live, 0).to(tl.float32)
        tl.store(output + row * DIM + col, residual + gate * value, live)


def _gate_reference(hidden, key, weight, token_mask, eps):
    residual, key = hidden.float(), key.float()
    rstd = torch.rsqrt(residual.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (residual * weight * key).sum(-1) * rstd * hidden.shape[-1] ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return gate


def engram_post_wkv_reference(hidden, kv, weight, token_mask=None, eps=1e-20):
    """Original torch operation sequence, also used for CPU/strided inputs."""
    hc, dim = hidden.shape[-2:]
    key = kv[..., : hc * dim].reshape(*hidden.shape)
    gate = _gate_reference(hidden, key, weight, token_mask, eps)
    return (
        hidden.float() + gate.unsqueeze(-1) * kv[..., hc * dim :].float().unsqueeze(-2)
    ).to(hidden.dtype)


def _can_fuse(hidden, kv, weight, token_mask):
    return (
        envs.ATOM_ENGRAM_FUSED_GATE
        and hidden.is_cuda
        and hidden.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and kv.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and hidden.is_contiguous()
        and kv.is_contiguous()
        and weight.is_contiguous()
        and (token_mask is None or token_mask.is_contiguous())
        and not (
            torch.is_grad_enabled()
            and any(t.requires_grad for t in (hidden, kv, weight))
        )
    )


def engram_post_wkv(hidden, kv, weight, token_mask=None, eps=1e-20, *, kv_tp_size=1):
    """Fused inference path for contiguous [..., HC, DIM] residuals."""
    hc, dim = hidden.shape[-2:]
    if kv.shape != (*hidden.shape[:-2], (hc + 1) * dim):
        raise ValueError("Engram KV projection does not match residual shape")
    if weight.shape != (hc, dim):
        raise ValueError("Engram gate weight does not match residual branches")
    if token_mask is not None and token_mask.shape != hidden.shape[:-2]:
        raise ValueError("Engram token mask must match residual tokens")
    if kv_tp_size not in (1, 2, 4):
        raise ValueError("Engram KV sharding supports TP2 or TP4")
    if (hc * dim) % kv_tp_size or dim % kv_tp_size:
        raise ValueError("Engram KV TP size must divide key rows and value width")
    supported = _can_fuse(hidden, kv, weight, token_mask)
    if not supported:
        if kv_tp_size > 1:
            parts = kv.reshape(*kv.shape[:-1], kv_tp_size, -1)
            key_rows = hc * dim // kv_tp_size
            kv = torch.cat(
                (parts[..., :key_rows].flatten(-2), parts[..., key_rows:].flatten(-2)),
                dim=-1,
            )
        return engram_post_wkv_reference(hidden, kv, weight, token_mask, eps)
    output = torch.empty_like(hidden)
    if hidden.numel():
        block = triton.next_power_of_2(dim)
        _engram_post_wkv_kernel[(hidden.numel() // dim,)](
            hidden,
            kv,
            weight,
            token_mask,
            output,
            eps,
            DIM=dim,
            HC=hc,
            BLOCK=block,
            HAS_MASK=token_mask is not None,
            LOCAL_HC=hc,
            BRANCH_START=0,
            GATE_ONLY=False,
            KV_TP=kv_tp_size,
            num_warps=8 if block >= 2048 else 4,
            enable_fp_fusion=False,
        )
    return output


@triton.jit(do_not_specialize=["M"])
def _engram_apply_gates_kernel(
    hidden,
    kv,
    gates,
    output,
    # Prefill chunk lengths vary; keep the gathered-gate stride runtime.
    M,
    HC: tl.constexpr,
    DIM: tl.constexpr,
    LOCAL_HC: tl.constexpr,
    BLOCK: tl.constexpr,
    VALUE_START: tl.constexpr = 0,
):
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = off < M * HC * DIM
    token = off // (HC * DIM)
    branch = (off // DIM) % HC
    col = off % DIM
    # dim=0 all-gather retains [TP, M, LOCAL_HC], without a gate transpose.
    gate_idx = (
        (branch // LOCAL_HC) * M * LOCAL_HC + token * LOCAL_HC + branch % LOCAL_HC
    )
    gate = tl.load(gates + gate_idx, live, 0)
    h = tl.load(hidden + off, live, 0).to(tl.float32)
    # V stays at the end of each GEMM row; no packing or V all-gather.
    value_col = tl.where(col >= VALUE_START, col - VALUE_START, col + DIM - VALUE_START)
    v = tl.load(
        kv + token * (LOCAL_HC + 1) * DIM + LOCAL_HC * DIM + value_col, live, 0
    ).to(tl.float32)
    tl.store(output + off, h + gate * v, live)


def engram_post_wkv_tp(
    hidden, kv, weight, group, token_mask=None, eps=1e-20, *, value_start=0
):
    """TP2/TP4: local complete Keys + full V; gather FP32 gates."""
    hc, dim = hidden.shape[-2:]
    tp, rank = group.world_size, group.rank_in_group
    if tp not in (2, 4):
        raise ValueError("Engram WKV sharding supports TP2 or TP4")
    if hc % tp:
        raise ValueError("Engram branches must divide evenly across TP ranks")
    local_hc = hc // tp
    if kv.shape != (*hidden.shape[:-2], (local_hc + 1) * dim):
        raise ValueError("Engram local KV projection does not match residual shape")
    if weight.shape != (hc, dim):
        raise ValueError("Engram gate weight does not match residual branches")
    if token_mask is not None and token_mask.shape != hidden.shape[:-2]:
        raise ValueError("Engram token mask must match residual tokens")
    if not 0 <= value_start < dim:
        raise ValueError("Engram value start must be inside the value width")
    if not hidden.numel():
        return hidden.clone()
    m = hidden.numel() // (hc * dim)
    start = rank * local_hc
    fused = _can_fuse(hidden, kv, weight, token_mask)
    if fused:
        local_gates = torch.empty(
            (m, local_hc), device=hidden.device, dtype=torch.float32
        )
        block = triton.next_power_of_2(dim)
        _engram_post_wkv_kernel[(m * local_hc,)](
            hidden,
            kv,
            weight,
            token_mask,
            local_gates,
            eps,
            DIM=dim,
            HC=hc,
            BLOCK=block,
            HAS_MASK=token_mask is not None,
            LOCAL_HC=local_hc,
            BRANCH_START=start,
            GATE_ONLY=True,
            num_warps=8 if block >= 2048 else 4,
            enable_fp_fusion=False,
        )
    else:
        key = kv[..., : local_hc * dim].unflatten(-1, (local_hc, dim))
        local_gates = (
            _gate_reference(
                hidden[..., start : start + local_hc, :],
                key,
                weight[start : start + local_hc],
                token_mask,
                eps,
            )
            .reshape(m, local_hc)
            .contiguous()
        )
    comm = getattr(getattr(group, "device_communicator", None), "ca_comm", None)
    gates = group.all_gather(
        local_gates,
        dim=0,
        use_custom=local_gates.is_cuda and comm is not None and not comm.disabled,
    )
    if not fused:
        gates = (
            gates.reshape(tp, m, local_hc)
            .permute(1, 0, 2)
            .reshape(*hidden.shape[:-2], hc)
        )
        value = kv[..., local_hc * dim :]
        if value_start:
            value = torch.roll(value, value_start, dims=-1)
        return (hidden.float() + gates.unsqueeze(-1) * value.float().unsqueeze(-2)).to(
            hidden.dtype
        )
    output = torch.empty_like(hidden)
    _engram_apply_gates_kernel[(triton.cdiv(hidden.numel(), 1024),)](
        hidden,
        kv,
        gates,
        output,
        M=m,
        HC=hc,
        DIM=dim,
        LOCAL_HC=local_hc,
        BLOCK=1024,
        VALUE_START=value_start,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
