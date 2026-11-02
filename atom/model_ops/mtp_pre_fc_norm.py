# SPDX-License-Identifier: MIT
"""Qwen3.5 MTP draft prologue in one launch: the draft token's embedding row,
``pre_fc_norm_embedding`` on it, ``pre_fc_norm_hidden`` on the target hidden
state, each Gemma RMSNorm stored into its half of the ``fc`` input.

Two implementations of the one launch, the same bits:

* aiter's own two-input norm with its ``q_row_index`` gather (the embedding
  table is ``q``, the ids index its rows), where the installed aiter has it
  (aiter ``vgokhale/fuse-drafter``): the kernel the default path already runs,
  so bitwise by construction, and the faster of the two.
* otherwise the Triton kernel below.

The Triton kernel is bitwise the same as what ``Qwen3_5MultiTokenPredictor.forward`` computes
without it (``F.embedding``, two ``GemmaRMSNorm`` launches of aiter's
``fused_qk_rmsnorm_group_quant`` in no-quant Gemma mode, ``torch.cat``), and
as ``ATOM_MTP_DUAL_PRE_FC_NORM``'s two-input launch: the kernel reproduces
aiter's arithmetic step for step.

* aiter runs a row on 128 threads of 16 contiguous elements. Each thread sums
  its squares in order (``acc = fma(x, x, acc)``); the 128 partials then meet
  in a balanced binary tree in lane order (the wave's DPP butterfly, then the
  four waves' partials, two at a time). The kernel reads the row as 16 strided
  ``[128]`` vectors -- element ``16 t + i`` lands in lane ``t`` -- runs the
  same chain, and adds the partials pairwise with ``tl.split``. A plain
  ``tl.sum`` reduces in another order and differs in ~0.7% of rows.
* ``rstd = rsqrt(sum / H + eps)``; ``y = (x * rstd) * (w + 1)`` in fp32,
  rounded to bf16 (nearest even), as aiter's two ``v_pk_mul_f32``.

The second pass re-reads the row contiguously (from cache) for the stores:
elementwise math does not depend on the layout, so it rounds the same.
"""

import functools
import inspect

import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard

# aiter's no-quant launch for a 2048-wide row: 128 threads x 16 elements.
_THREADS = 128


@triton.jit
def _pair_sum(x, N: tl.constexpr):
    a, b = tl.split(tl.reshape(x, [N // 2, 2]))
    return a + b


@triton.jit
def _lane_tree_sum_128(acc):
    """Sum of 128 lane partials as a balanced binary tree in lane order."""
    acc = _pair_sum(acc, 128)
    acc = _pair_sum(acc, 64)
    acc = _pair_sum(acc, 32)
    acc = _pair_sum(acc, 16)
    acc = _pair_sum(acc, 8)
    acc = _pair_sum(acc, 4)
    acc = _pair_sum(acc, 2)
    return tl.reshape(acc, [])


# The ids are a view at any offset (a prefill's target ids, a pass's staged
# buffer): not specialized on alignment, so one variant serves both.
@triton.jit(do_not_specialize_on_alignment=["ids_ptr"])
def _mtp_embed_pre_fc_norms_kernel(
    ids_ptr,
    embed_ptr,
    embed_stride,
    vocab_size,
    hidden_ptr,
    hidden_stride,
    enorm_w_ptr,
    hnorm_w_ptr,
    out_ptr,
    out_stride,
    enorm_eps,
    hnorm_eps,
    H: tl.constexpr,
    THREADS: tl.constexpr,
    PER_THREAD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    # Half 0: the draft token's embedding row; half 1: the target hidden row.
    half = tl.program_id(1)
    if half == 0:
        token = tl.load(ids_ptr + row).to(tl.int64)
        # An id outside the table reads as a zero row, as the masked lookup
        # of `ReplicatedEmbedding` returns it (F.embedding would fault).
        valid = (token >= 0) & (token < vocab_size)
        src = embed_ptr + tl.where(valid, token, 0) * embed_stride
        w_ptr = enorm_w_ptr
        eps = enorm_eps
    else:
        valid = row >= 0  # a hidden row is always there
        src = hidden_ptr + row * hidden_stride
        w_ptr = hnorm_w_ptr
        eps = hnorm_eps

    lane = tl.arange(0, THREADS)
    acc = tl.zeros([THREADS], dtype=tl.float32)
    for i in tl.static_range(PER_THREAD):
        x = tl.load(src + lane * PER_THREAD + i, mask=valid, other=0.0).to(tl.float32)
        acc = tl.fma(x, x, acc)
    rstd = tl.rsqrt(_lane_tree_sum_128(acc) / H + eps)

    cols = tl.arange(0, H)
    x = tl.load(src + cols, mask=valid, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32) + 1.0
    y = (x * rstd) * w
    tl.store(
        out_ptr + row * out_stride + half * H + cols, y.to(out_ptr.dtype.element_ty)
    )


def mtp_embed_pre_fc_norms_supported(
    input_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    hidden_states: torch.Tensor,
) -> bool:
    """The shapes the kernel reproduces aiter bit for bit on: bf16 rows of
    ``128 * 16`` (aiter's launch for that width), unit-stride rows, one id per
    hidden row."""
    h = hidden_states.shape[-1]
    return (
        h == _THREADS * 16
        and hidden_states.dim() == 2
        and input_ids.dim() == 1
        and input_ids.shape[0] == hidden_states.shape[0]
        and hidden_states.dtype == torch.bfloat16
        and embed_weight.dtype == torch.bfloat16
        and embed_weight.dim() == 2
        and embed_weight.shape[1] == h
        and hidden_states.stride(-1) == 1
        and embed_weight.stride(-1) == 1
    )


def _mtp_embed_pre_fc_norms_fake(
    input_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    enorm_eps: float,
    hnorm_weight: torch.Tensor,
    hnorm_eps: float,
) -> torch.Tensor:
    del input_ids, embed_weight, enorm_weight, enorm_eps, hnorm_weight, hnorm_eps
    rows, hidden = hidden_states.shape
    return torch.empty(
        (rows, 2 * hidden), dtype=hidden_states.dtype, device=hidden_states.device
    )


@torch_compile_guard(gen_fake=_mtp_embed_pre_fc_norms_fake)
def mtp_embed_pre_fc_norms(
    input_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    enorm_eps: float,
    hnorm_weight: torch.Tensor,
    hnorm_eps: float,
) -> torch.Tensor:
    """``cat([enorm(embed[ids]), hnorm(hidden)], -1)``, one launch, bitwise.

    Graph-safe: no host reads, the grid is the row count of the tensors.
    An id outside ``[0, vocab)`` embeds as zeros (the masked lookup's rule).
    """
    out = _mtp_embed_pre_fc_norms_fake(
        input_ids,
        embed_weight,
        hidden_states,
        enorm_weight,
        enorm_eps,
        hnorm_weight,
        hnorm_eps,
    )
    rows = hidden_states.shape[0]
    if rows == 0:
        return out
    if aiter_gathers_rows() and input_ids.dtype == torch.int32:
        _launch_aiter(
            input_ids,
            embed_weight,
            hidden_states,
            enorm_weight,
            enorm_eps,
            hnorm_weight,
            hnorm_eps,
            out,
        )
    else:
        _launch_triton(
            input_ids,
            embed_weight,
            hidden_states,
            enorm_weight,
            enorm_eps,
            hnorm_weight,
            hnorm_eps,
            out,
        )
    return out


@functools.cache
def aiter_gathers_rows() -> bool:
    """Whether the installed aiter's two-input norm takes ``q_row_index``."""
    from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant

    return "q_row_index" in inspect.signature(fused_qk_rmsnorm_group_quant).parameters


def _launch_aiter(
    input_ids, embed_weight, hidden_states, enorm_w, enorm_eps, hnorm_w, hnorm_eps, out
):
    from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant

    hidden = hidden_states.shape[-1]
    fused_qk_rmsnorm_group_quant(
        q=embed_weight,
        q_weight=enorm_w,
        q_epsilon=enorm_eps,
        q_out_unquantized=out[:, :hidden],
        k=hidden_states,
        k_weight=hnorm_w,
        k_epsilon=hnorm_eps,
        k_out=out[:, hidden:],
        gemma_norm=True,
        q_row_index=input_ids.contiguous(),
    )


def _launch_triton(
    input_ids, embed_weight, hidden_states, enorm_w, enorm_eps, hnorm_w, hnorm_eps, out
):
    rows, hidden = hidden_states.shape
    _mtp_embed_pre_fc_norms_kernel[(rows, 2)](
        input_ids,
        embed_weight,
        embed_weight.stride(0),
        embed_weight.shape[0],
        hidden_states,
        hidden_states.stride(0),
        enorm_w,
        hnorm_w,
        out,
        out.stride(0),
        enorm_eps,
        hnorm_eps,
        H=hidden,
        THREADS=_THREADS,
        PER_THREAD=hidden // _THREADS,
        num_warps=_THREADS // 32,
    )
