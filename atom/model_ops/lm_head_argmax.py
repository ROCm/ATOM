import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard

# Cap the vector width; larger vocabularies stream through multiple tiles.
_MAX_BLOCK_M = 131072


@triton.jit
def _lm_head_argmax_pack_kernel(
    logits_ptr,
    packed_ptr,
    vocab_start_idx,
    M: tl.constexpr,
    stride_logits_n: tl.constexpr,
    stride_logits_m: tl.constexpr,
    stride_packed_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0)
    block_offs = tl.arange(0, BLOCK_M)
    best_val = tl.full((), -float("inf"), dtype=tl.float32)
    invalid_idx = tl.full((), M, dtype=tl.int64)
    best_idx = invalid_idx

    for block_start in tl.range(0, M, BLOCK_M):
        offs = block_start + block_offs
        mask = offs < M
        vals = tl.load(
            logits_ptr + row * stride_logits_n + offs * stride_logits_m,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        block_max = tl.max(vals, axis=0)
        idxs = offs.to(tl.int64)
        block_idx = tl.min(
            tl.where((vals == block_max) & mask, idxs, invalid_idx), axis=0
        )
        take_block = (block_max > best_val) | (
            (block_max == best_val) & (block_idx < best_idx)
        )
        best_val = tl.where(take_block, block_max, best_val)
        best_idx = tl.where(take_block, block_idx, best_idx)

    global_idx = best_idx + vocab_start_idx
    tl.store(packed_ptr + row * stride_packed_n, best_val)
    tl.store(packed_ptr + row * stride_packed_n + 1, global_idx.to(tl.float32))


def _lm_head_argmax_pack_fake(
    logits: torch.Tensor,
    vocab_start_idx: int,
) -> torch.Tensor:
    return torch.empty((logits.shape[0], 2), dtype=torch.float32, device=logits.device)


@torch_compile_guard(gen_fake=_lm_head_argmax_pack_fake)
def lm_head_argmax_pack(logits: torch.Tensor, vocab_start_idx: int) -> torch.Tensor:
    """Reduce and fp32-pack local LM-head argmax in one Triton launch."""
    if logits.dim() != 2:
        raise ValueError("lm_head_argmax_pack expects a 2-D logits tensor")

    N, M = logits.shape
    if N == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=logits.device)
    if M == 0:
        raise ValueError("lm_head_argmax_pack requires a non-empty vocabulary")

    packed = torch.empty((N, 2), dtype=torch.float32, device=logits.device)
    block_m = min(_MAX_BLOCK_M, triton.next_power_of_2(M))
    num_warps = 8 if block_m >= 2048 else 4

    _lm_head_argmax_pack_kernel[(N,)](
        logits,
        packed,
        vocab_start_idx,
        M=M,
        stride_logits_n=logits.stride(0),
        stride_logits_m=logits.stride(1),
        stride_packed_n=packed.stride(0),
        BLOCK_M=block_m,
        num_warps=num_warps,
        num_stages=2,
    )
    return packed


# ---------------------------------------------------------------------------
# All-gather of the packs as an all-reduce (ATOM_ARGMAX_HEAD_AR_GATHER).
#
# A rank writes its (max, index) pack as 8 byte-valued units -- the 4 bytes of
# the fp32 max, then the 4 of the int32 index -- into its own slot of a
# [world, rows, 8] bf16 image and zeros into every other slot. bf16 holds the
# integers 0..255 exactly and each value has one nonzero term, so a sum
# all-reduce of the image is exact under any reduction order: every rank then
# holds every rank's pack, bit for bit.
# ---------------------------------------------------------------------------


@triton.jit
def _lm_head_argmax_units_kernel(
    logits_ptr,
    units_ptr,
    vocab_start_idx,
    rank,
    N,
    M: tl.constexpr,
    stride_logits_n: tl.constexpr,
    stride_logits_m: tl.constexpr,
    WORLD: tl.constexpr,
    WORLD_P2: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # The reduction is _lm_head_argmax_pack_kernel's, step for step.
    row = tl.program_id(0)
    block_offs = tl.arange(0, BLOCK_M)
    best_val = tl.full((), -float("inf"), dtype=tl.float32)
    invalid_idx = tl.full((), M, dtype=tl.int64)
    best_idx = invalid_idx

    for block_start in tl.range(0, M, BLOCK_M):
        offs = block_start + block_offs
        mask = offs < M
        vals = tl.load(
            logits_ptr + row * stride_logits_n + offs * stride_logits_m,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        block_max = tl.max(vals, axis=0)
        idxs = offs.to(tl.int64)
        block_idx = tl.min(
            tl.where((vals == block_max) & mask, idxs, invalid_idx), axis=0
        )
        take_block = (block_max > best_val) | (
            (block_max == best_val) & (block_idx < best_idx)
        )
        best_val = tl.where(take_block, block_max, best_val)
        best_idx = tl.where(take_block, block_idx, best_idx)

    global_idx = (best_idx + vocab_start_idx).to(tl.int32)
    unit = tl.arange(0, 8)
    word = tl.where(unit < 4, best_val.to(tl.int32, bitcast=True), global_idx)
    units = ((word >> ((unit % 4) * 8)) & 255).to(tl.float32)
    slot = tl.arange(0, WORLD_P2)[:, None]
    image = tl.where(slot == rank, units[None, :], 0.0).to(tl.bfloat16)
    tl.store(
        units_ptr + (slot * N + row) * 8 + unit[None, :],
        image,
        mask=(slot < WORLD) & (unit[None, :] < 8),
    )


@triton.jit
def _lm_head_argmax_pick_kernel(
    units_ptr,
    out_ptr,
    N,
    WORLD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    unit = tl.arange(0, 8)
    mask = (rows[:, None] < N) & (unit[None, :] < 8)
    shift = (unit % 4) * 8
    best_val = tl.zeros((BLOCK_N,), dtype=tl.float32)
    best_idx = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for r in tl.static_range(WORLD):
        u = tl.load(
            units_ptr + (r * N + rows[:, None]) * 8 + unit[None, :], mask=mask, other=0
        )
        # Disjoint bytes: the sum is their OR, the sign bit included.
        u = u.to(tl.float32).to(tl.int32) << shift[None, :]
        val = tl.sum(tl.where(unit[None, :] < 4, u, 0), axis=1)
        val = val.to(tl.float32, bitcast=True)
        idx = tl.sum(tl.where(unit[None, :] < 4, 0, u), axis=1)
        if r == 0:
            best_val = val
            best_idx = idx
        else:
            # torch.argmax over the ranks: the first maximum, a NaN above all.
            nan = val != val  # noqa: PLR0124
            best_nan = best_val != best_val  # noqa: PLR0124
            take = (val > best_val) | (nan & ~best_nan)
            best_val = tl.where(take, val, best_val)
            best_idx = tl.where(take, idx, best_idx)
    tl.store(out_ptr + rows, best_idx, mask=rows < N)


def lm_head_argmax_units(
    logits: torch.Tensor, vocab_start_idx: int, rank: int, world: int
) -> torch.Tensor:
    """This rank's argmax pack as the [world * N * 8] bf16 image the packs'
    all-reduce sums (see above)."""
    N, M = logits.shape
    units = torch.empty((world * N * 8,), dtype=torch.bfloat16, device=logits.device)
    if N == 0:
        return units
    block_m = min(_MAX_BLOCK_M, triton.next_power_of_2(M))
    _lm_head_argmax_units_kernel[(N,)](
        logits,
        units,
        vocab_start_idx,
        rank,
        N,
        M=M,
        stride_logits_n=logits.stride(0),
        stride_logits_m=logits.stride(1),
        WORLD=world,
        WORLD_P2=triton.next_power_of_2(world),
        BLOCK_M=block_m,
        num_warps=8 if block_m >= 2048 else 4,
        num_stages=2,
    )
    return units


def lm_head_argmax_pick(
    units: torch.Tensor, world: int, out: torch.Tensor
) -> torch.Tensor:
    """The winning token of each row from the reduced image, into ``out``
    (int32): the same pick as the all-gather path's argmax over the ranks."""
    N = out.numel()
    if N:
        block_n = min(64, triton.next_power_of_2(N))
        _lm_head_argmax_pick_kernel[(triton.cdiv(N, block_n),)](
            units, out, N, WORLD=world, BLOCK_N=block_n, num_warps=1
        )
    return out
