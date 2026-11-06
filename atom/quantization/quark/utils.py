# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Any

import regex as re
import torch
import triton
import triton.language as tl
from aiter import QuantType

_FP8_SOURCE_DTYPES = frozenset(
    {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    }
)


def deep_compare(dict1: Any, dict2: Any) -> bool:
    if type(dict1) is not type(dict2):
        return False
    if isinstance(dict1, dict):
        if dict1.keys() != dict2.keys():
            return False
        return all(deep_compare(dict1[k], dict2[k]) for k in dict1)
    elif isinstance(dict1, list):
        return set(dict1) == set(dict2)
    else:
        return dict1 == dict2


def check_equal_or_regex_match(layer_name: str, targets: Iterable[str]) -> bool:
    """
    Checks whether a layer_name is exactly equal or a regex match for
    if target starts with 're:' to any target in list.
    """
    for target in targets:
        if _is_equal_or_regex_match(layer_name, target):
            return True
    return False


def _is_equal_or_regex_match(
    value: str, target: str, check_contains: bool = False
) -> bool:
    """
    Checks whether a value is exactly equal or a regex match for target
    if target starts with 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.
    """

    if target.startswith("re:"):
        pattern = target[3:]
        if re.match(pattern, value):
            return True
    elif check_contains:
        if target.lower() in value.lower():
            return True
    elif target == value:
        return True
    return False


@triton.jit
def _weight_dequant_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):  # type: ignore[no-untyped-def]
    """
    Triton kernel for dequantizing FP8 weights using scaling factors.

    This kernel is provided by deepseek-ai for efficient FP8 weight dequantization.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def dequant_per_block_fp8(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantize a per-block (128x128) FP8 weight using inverse scale with a
    Triton kernel.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

    _weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# Optional E8M0 dtype: only available on newer torch builds.
_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)

_NVFP4_BLOCK_SIZE = 16
# Two E2M1 values share one uint8, so a 16-value block is 8 packed bytes wide.
_NVFP4_PACKED_GROUP = _NVFP4_BLOCK_SIZE // 2
_FP4_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
# One device copy of the table per device, kept because the decode kernel
# gathers from it on every call (weight loading touches hundreds of layers).
_FP4_E2M1_LUT_BY_DEVICE: dict[torch.device, torch.Tensor] = {}


def _fp4_e2m1_lut(device: torch.device) -> torch.Tensor:
    lut = _FP4_E2M1_LUT_BY_DEVICE.get(device)
    if lut is None:
        lut = _FP4_E2M1_LUT.to(device=device)
        _FP4_E2M1_LUT_BY_DEVICE[device] = lut
    return lut


def _dequantize_nvfp4_torch(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    out_dtype: torch.dtype,
    high_nibble_first: bool,
) -> torch.Tensor:
    """Reference NVFP4 decode in pure torch.

    Kept as the CPU path (Triton needs a GPU) and as the oracle the Triton
    kernel is checked against in ``tests/test_nvfp4_dequant_triton.py``. It
    shares :func:`_check_nvfp4_inputs` and :func:`_nvfp4_global_scale_per_row`
    with the Triton path so the two accept and reject exactly the same inputs --
    a decoder that validates differently depending on the device is a trap.
    """
    rows, _packed_cols, logical_cols = _check_nvfp4_inputs(weight, weight_scale)

    low = (weight & 0xF).to(torch.int64)
    high = (weight >> 4).to(torch.int64)
    first, second = (high, low) if high_nibble_first else (low, high)
    lut = _fp4_e2m1_lut(weight.device)
    dequantized = torch.empty(
        rows, logical_cols, dtype=torch.float32, device=weight.device
    )
    dequantized[:, 0::2] = lut[first]
    dequantized[:, 1::2] = lut[second]

    global_scale, per_row = _nvfp4_global_scale_per_row(
        weight_scale_2, rows, weight.device
    )
    scale = weight_scale.to(torch.float32) * (
        global_scale.view(-1, 1) if per_row else global_scale.view(())
    )
    scale = scale.repeat_interleave(_NVFP4_BLOCK_SIZE, dim=-1)
    return (dequantized * scale).to(out_dtype)


@triton.jit
def _nvfp4_dequant_kernel(  # type: ignore[no-untyped-def]
    w_ptr,  # uint8 [rows, packed_cols], two E2M1 values per byte
    s_ptr,  # E4M3 [rows, packed_cols // PACKED_GROUP] block scales
    g_ptr,  # fp32 global scale, one value or one per row
    lut_ptr,  # fp32 [16] E2M1 decode table
    y_ptr,  # out [rows, packed_cols * 2]
    rows,
    packed_cols,
    w_row_stride,
    s_row_stride,
    y_row_stride,
    GLOBAL_SCALE_PER_ROW: tl.constexpr,
    HIGH_NIBBLE_FIRST: tl.constexpr,
    PACKED_GROUP: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_PACKED_COLS: tl.constexpr,
    BLOCK_SCALE_COLS: tl.constexpr,
):
    """Unpack E2M1 nibbles and apply the block and global scales.

    One program owns a ``BLOCK_ROWS x BLOCK_PACKED_COLS`` tile of packed bytes,
    i.e. ``2 * BLOCK_PACKED_COLS`` logical columns. ``BLOCK_PACKED_COLS`` is a
    multiple of ``PACKED_GROUP``, so a tile never splits a scale block and each
    block scale is read exactly once.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # int64 rows: `row * row_stride` is the only term that can leave int32
    # range (a gathered TP weight is already within ~4x of it), and on this
    # memory-bound kernel the wider index costs nothing.
    row_offs = (pid_m * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
    row_mask = row_offs < rows

    # Packed bytes: low/high nibble of each byte are two adjacent logical values.
    col_offs = pid_n * BLOCK_PACKED_COLS + tl.arange(0, BLOCK_PACKED_COLS)
    byte_mask = row_mask[:, None] & (col_offs[None, :] < packed_cols)
    packed = tl.load(
        w_ptr + row_offs[:, None] * w_row_stride + col_offs[None, :],
        mask=byte_mask,
        other=0,
    ).to(tl.int32)
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    if HIGH_NIBBLE_FIRST:
        first, second = high, low
    else:
        first, second = low, high
    # Indices are always in [0, 16), so the gather needs no mask.
    val_first = tl.load(lut_ptr + first)
    val_second = tl.load(lut_ptr + second)

    # Block scales: one per PACKED_GROUP bytes, broadcast back over the tile.
    scale_cols = packed_cols // PACKED_GROUP
    scale_offs = pid_n * BLOCK_SCALE_COLS + tl.arange(0, BLOCK_SCALE_COLS)
    scale = tl.load(
        s_ptr + row_offs[:, None] * s_row_stride + scale_offs[None, :],
        mask=row_mask[:, None] & (scale_offs[None, :] < scale_cols),
        other=0.0,
    ).to(tl.float32)
    # NVFP4 is a two-level format: the E4M3 block scale is always paired with
    # an FP32 global scale, so this multiply is unconditional.
    if GLOBAL_SCALE_PER_ROW:
        global_scale = tl.load(g_ptr + row_offs, mask=row_mask, other=0.0)
        scale = scale * global_scale[:, None]
    else:
        scale = scale * tl.load(g_ptr)
    scale = tl.reshape(
        tl.broadcast_to(
            scale[:, :, None], (BLOCK_ROWS, BLOCK_SCALE_COLS, PACKED_GROUP)
        ),
        (BLOCK_ROWS, BLOCK_PACKED_COLS),
    )

    # Interleave restores logical order: byte j feeds columns 2j and 2j+1.
    out = tl.interleave(val_first * scale, val_second * scale)
    out_offs = pid_n * (2 * BLOCK_PACKED_COLS) + tl.arange(0, 2 * BLOCK_PACKED_COLS)
    tl.store(
        y_ptr + row_offs[:, None] * y_row_stride + out_offs[None, :],
        out.to(y_ptr.dtype.element_ty),
        mask=row_mask[:, None] & (out_offs[None, :] < 2 * packed_cols),
    )


def _check_nvfp4_inputs(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[int, int, int]:
    """Validate the NVFP4 wire format and return ``(rows, packed_cols, K)``."""
    if weight.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError(
            "NVFP4 dequantization expects 2D weight and scale tensors, got "
            f"weight={tuple(weight.shape)}, scale={tuple(weight_scale.shape)}."
        )
    if weight.dtype != torch.uint8:
        raise TypeError(
            f"NVFP4 packed weight must use uint8 storage, got {weight.dtype}."
        )
    # Both decode paths convert the scale by value, so raw scale bytes would
    # decode consistently wrong (0x38 -> 56.0 instead of 1.0) on CPU and GPU
    # alike; only this check can catch them.
    if weight_scale.dtype != torch.float8_e4m3fn:
        raise TypeError(
            "NVFP4 block scale must be float8_e4m3fn, got "
            f"{weight_scale.dtype}; view raw scale bytes as float8_e4m3fn "
            "instead of casting them by value."
        )

    rows, packed_cols = weight.shape
    logical_cols = packed_cols * 2
    if logical_cols % _NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"NVFP4 logical K={logical_cols} must be divisible by "
            f"group_size={_NVFP4_BLOCK_SIZE}."
        )
    expected_scale_shape = (rows, logical_cols // _NVFP4_BLOCK_SIZE)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"NVFP4 scale shape {tuple(weight_scale.shape)} does not match "
            f"expected {expected_scale_shape}."
        )
    return rows, packed_cols, logical_cols


def _nvfp4_global_scale_per_row(
    weight_scale_2: torch.Tensor, rows: int, device: torch.device
) -> tuple[torch.Tensor, bool]:
    """Normalize the NVFP4 global scale to a scalar or one FP32 value per row.

    NVFP4 checkpoints carry ``weight_scale_2`` either per tensor (one scalar,
    as the MoE path passes per expert projection) or per output row (as the
    Linear path passes after expanding one scalar per merged output partition).
    Both collapse to a row-indexed lookup; anything else -- a value per scale
    *block*, say -- is a caller bug rather than a layout this decoder supports.

    :return: ``(scale, is_per_row)``; ``scale`` is contiguous FP32 on ``device``.
    """
    if weight_scale_2 is None:
        raise ValueError(
            "NVFP4 is a two-level format and always carries a global weight "
            "scale; weight_scale_2 is None. A layer that reaches this point "
            "without one failed to load its *_weight_scale_2 checkpoint tensor."
        )
    scale_2 = weight_scale_2.to(device=device, dtype=torch.float32)
    if scale_2.numel() == 1:
        return scale_2.reshape(1).contiguous(), False
    if scale_2.numel() == rows and scale_2.shape[-1] == 1:
        return scale_2.reshape(rows).contiguous(), True
    raise ValueError(
        "NVFP4 global scale must be a scalar or one value per weight row "
        f"(rows={rows}), got shape {tuple(weight_scale_2.shape)}."
    )


def dequantize_nvfp4(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    high_nibble_first: bool = False,
) -> torch.Tensor:
    """Decode Quark or NVIDIA ModelOpt NVFP4 into a floating-point 2D weight.

    ``weight`` stores two E2M1 values per uint8, ``weight_scale`` stores one
    E4M3 scale per 16 logical values, and ``weight_scale_2`` is the FP32 global
    multiplier. This mirrors SGLang's NVFP4 source decoder used by its online
    NVFP4-to-MXFP4 path.

    All three are required: NVFP4 is a two-level format by definition, so a
    layer without a global scale is not an NVFP4 layer. Both ATOM call sites
    (``LinearBase.online_quantize_weight`` and ``FusedMoE._online_quant``)
    always pass one, and a ``None`` reaching here means the checkpoint's
    ``*_weight_scale_2`` never loaded -- silently dropping it would scale the
    whole weight wrong, so it raises instead.

    ``out_dtype`` defaults to ``torch.get_default_dtype()`` -- the model dtype,
    which ``ModelRunner`` installs before any weight is loaded -- matching every
    other dequant helper here (:func:`dequant_per_tensor_fp8`,
    :func:`dequant_per_channel_fp8`, :func:`dequant_per_block_fp8`,
    :func:`dequant_mxfp8`). Decoding to FP32 instead would be pure waste: the
    only consumer is :func:`quant_weight_online`, and
    :func:`quant_mxfp4_online_even` casts anything that is not already
    FP16/BF16 down to BF16 before quantizing. Emitting BF16 here is *bit
    identical* to that -- the arithmetic still happens in FP32 and is rounded
    once, just at the store rather than in a second full-width pass -- while
    halving both the output tensor and the bytes the MXFP4 quantizer reads.

    On GPU this runs as a single Triton kernel: unpack, LUT decode, block scale
    and global scale are fused, so the only tensor materialized is the output.
    The torch path (:func:`_dequantize_nvfp4_torch`) built a full FP32 nibble
    tensor plus a full-width ``repeat_interleave``'d scale tensor first -- three
    K-wide temporaries per weight, on a path that runs for every layer at load.
    It remains the reference and the CPU fallback.
    """
    if out_dtype is None:
        out_dtype = torch.get_default_dtype()
    if not weight.is_cuda:
        return _dequantize_nvfp4_torch(
            weight, weight_scale, weight_scale_2, out_dtype, high_nibble_first
        )

    rows, packed_cols, logical_cols = _check_nvfp4_inputs(weight, weight_scale)
    # Validate before the empty-tensor shortcut, so a bad global scale is
    # reported on every shape rather than only on the ones that launch.
    global_scale, global_scale_per_row = _nvfp4_global_scale_per_row(
        weight_scale_2, rows, weight.device
    )
    out = torch.empty(rows, logical_cols, dtype=out_dtype, device=weight.device)
    if out.numel() == 0:
        return out

    # The kernel indexes columns directly, so only the row stride is free.
    if weight.stride(1) != 1:
        weight = weight.contiguous()
    if weight_scale.stride(1) != 1:
        weight_scale = weight_scale.contiguous()

    block_packed_cols = min(
        128, max(_NVFP4_PACKED_GROUP, triton.next_power_of_2(packed_cols))
    )
    block_rows = min(8, triton.next_power_of_2(rows))
    grid = (
        triton.cdiv(rows, block_rows),
        triton.cdiv(packed_cols, block_packed_cols),
    )
    _nvfp4_dequant_kernel[grid](
        weight,
        weight_scale,
        global_scale,
        _fp4_e2m1_lut(weight.device),
        out,
        rows,
        packed_cols,
        weight.stride(0),
        weight_scale.stride(0),
        out.stride(0),
        GLOBAL_SCALE_PER_ROW=global_scale_per_row,
        HIGH_NIBBLE_FIRST=high_nibble_first,
        PACKED_GROUP=_NVFP4_PACKED_GROUP,
        BLOCK_ROWS=block_rows,
        BLOCK_PACKED_COLS=block_packed_cols,
        BLOCK_SCALE_COLS=block_packed_cols // _NVFP4_PACKED_GROUP,
        num_warps=4,
    )
    return out


def _mx_block_scale_dtype():
    """The block-scale dtype mandated by the MX (microscaling) format: E8M0.

    Every MX scheme (``QuantType.per_1x32``) stores a shared power-of-two block
    scale in E8M0, regardless of whether the elements are FP4 or FP8 — this is
    fixed by the MX spec, not a per-call choice. Resolving it here gives both the
    MXFP4 and MXFP8 online-quant paths a single source of truth, so callers pass
    a value derived from the format rather than a repeated literal.
    """
    from aiter import dtypes

    return dtypes.fp8_e8m0


def dequant_mxfp8(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 32
) -> torch.Tensor:
    """Dequantize an MXFP8 weight to the default float dtype.

    MXFP8 is a standard microscaling dtype (its 1x32 block scale is part of the
    format), so the name carries no explicit granularity suffix.
    """
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, K = x.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    n_blocks = K // block_size
    assert s.shape == (M, n_blocks), f"scale shape {tuple(s.shape)} != {(M, n_blocks)}"

    if _E8M0_DTYPE is not None and s.dtype == _E8M0_DTYPE:
        # E8M0 dtype decodes straight to the 2**(e-127) multiplier.
        scale = s.to(torch.float32)
    else:
        # Raw E8M0 integer codes stored as uint8 / float.
        scale = torch.exp2(s.to(torch.float32) - 127.0)

    out_dtype = torch.get_default_dtype()
    y = x.to(torch.float32).reshape(M, n_blocks, block_size)
    y = y * scale.unsqueeze(-1)
    return y.reshape(M, K).to(out_dtype)


def dequant_per_channel_fp8(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Dequantize a per-output-channel (per_Token / PTPC) FP8 weight to the
    default float dtype.

    :param x: quantized weight ``[N, K]`` (last dim is the contracted dim).
    :param s: per-output-channel scale, ``[N]`` or ``[N, 1]``.
    """
    assert x.dim() == 2, f"expected 2D weight, got shape={tuple(x.shape)}"
    out_dtype = torch.get_default_dtype()
    scale = s.reshape(-1).to(torch.float32).view(-1, 1)
    return (x.to(torch.float32) * scale).to(out_dtype)


def dequant_per_tensor_fp8(
    x: torch.Tensor,
    s: torch.Tensor,
    output_partition_sizes: list[int] | None = None,
) -> torch.Tensor:
    """Dequantize a per-tensor  FP8 weight to the atom config float dtype.

    Merged layers (qkv / gate_up) carry one scalar scale per output partition,
    so each output row-range is scaled by its own scale. A single scale
    (``numel <= 1``) scales the whole tensor.

    :param x: quantized weight ``[N, K]``.
    :param s: per-partition scalar scale(s).
    :param output_partition_sizes: row counts of each merged output partition,
        required when there is more than one scale.
    """
    out_dtype = torch.get_default_dtype()

    w = x.to(torch.float32)
    scale = s.reshape(-1)
    if scale.numel() <= 1:
        return (w * scale.reshape(())).to(out_dtype)
    assert output_partition_sizes is not None, (
        "per_Tensor merged layer needs output_partition_sizes to map each "
        "scale to its output rows."
    )
    off = 0
    for i, sz in enumerate(output_partition_sizes):
        w[off : off + sz] = w[off : off + sz] * scale[i]
        off += sz
    return w.to(out_dtype)


def can_dequant_weight_online(
    source_quant_type: QuantType,
    source_quant_dtype: torch.dtype | None = None,
) -> bool:
    """Return whether the online path can dequantize this source."""
    if source_quant_type == QuantType.No:
        return True
    if source_quant_dtype is not None and source_quant_dtype not in _FP8_SOURCE_DTYPES:
        return False
    return source_quant_type in (
        QuantType.per_Tensor,
        QuantType.per_Token,
        QuantType.per_1x128,
        QuantType.per_1x32,
    )


def dequant_weight_online(
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    source_quant_type: QuantType,
    source_quant_dtype: torch.dtype | None = None,
    output_partition_sizes: list[int] | None = None,
) -> torch.Tensor:
    """Dequantize an online-quant SOURCE weight back to the default float dtype.

    Shared by the Linear and MoE online-quant paths for every source except
    NVFP4 (decoded by :func:`dequantize_nvfp4`), and the inverse counterpart
    of :func:`quant_weight_online`: it turns an already-quantized weight back
    into float so it can be re-quantized to a different target format.

    A source is identified by BOTH its ``quant_type`` (the block layout) and its
    element ``quant_dtype``. The layout alone is not enough: ``per_1x32`` is the
    MX layout, which the format allows to carry 4-bit elements as well, and only
    the 8-bit (MXFP8) form is accepted here. Supported sources:

    - ``No``: unquantized, returned unchanged.
    - ``per_Tensor``: per-tensor FP8, one scalar scale per output partition.
    - ``per_Token`` (ptpc_fp8): per-output-channel FP8, scale ``(N, 1)``.
    - ``per_1x128``: DeepSeek-style 128x128 block FP8.
    - ``per_1x32``: MXFP8 (1x32 E8M0 shared scale).

    :param weight: The quantized (or float, for ``No``) weight tensor.
    :param weight_scale: The source weight scale (``None`` for ``No``).
    :param source_quant_type: The source quantization scheme (block layout).
    :param source_quant_dtype: The source element dtype. Used together with
        ``source_quant_type`` to reject non-8-bit (e.g. MXFP4) sources. When
        ``None`` the dtype check is skipped (caller vouches for an 8-bit source).
    :param output_partition_sizes: row counts of each merged output partition,
        only used (and required) by ``per_Tensor`` merged layers.
    :return: The dequantized weight in the default float dtype.
    """
    if source_quant_type == QuantType.No:
        return weight

    # Reject any non-8-bit source up front. The element dtype -- not just the
    # block layout -- decides whether we can dequantize: MXFP4 (fp4x2) shares
    # the per_1x32 layout with MXFP8 but is a target-only format.
    if source_quant_dtype is not None and source_quant_dtype not in _FP8_SOURCE_DTYPES:
        raise ValueError(
            f"Unsupported online dequant source dtype={source_quant_dtype} "
            f"(quant_type={source_quant_type}); supported sources are 8-bit FP8."
        )

    if source_quant_type == QuantType.per_Tensor:
        return dequant_per_tensor_fp8(weight, weight_scale, output_partition_sizes)
    if source_quant_type == QuantType.per_Token:
        return dequant_per_channel_fp8(weight, weight_scale)
    if source_quant_type == QuantType.per_1x128:
        return dequant_per_block_fp8(weight, weight_scale)
    if source_quant_type == QuantType.per_1x32:
        # per_1x32 is only the block layout; the (8-bit) dtype check above has
        # already ruled out MXFP4, so a valid source here is always MXFP8.
        return dequant_mxfp8(weight, weight_scale)
    raise ValueError(
        f"Unsupported source quant_type for online dequant: {source_quant_type}. "
        f"Supported sources: No, per_Tensor, per_Token, per_1x128, per_1x32."
    )


def dequant_moe_weight_online(
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    source_quant_type: QuantType,
    source_quant_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Dequantize ``[E, N, K]`` by flattening row-local expert groups."""
    assert (
        weight.dim() == 3
    ), f"expected batched MoE weight [E, N, K], got shape={tuple(weight.shape)}"
    experts, rows, cols = weight.shape
    flat_weight = weight.reshape(experts * rows, cols)
    if source_quant_type == QuantType.No:
        return flat_weight

    assert (
        weight_scale is not None
    ), f"source quant_type={source_quant_type} requires a weight scale"
    if source_quant_type == QuantType.per_Token:
        flat_scale = weight_scale.reshape(experts * rows, -1)
    elif source_quant_type == QuantType.per_1x128:
        assert (
            rows % 128 == 0
        ), f"per_1x128 expert rows must be 128-aligned, got N={rows}"
        flat_scale = weight_scale.reshape(experts * (rows // 128), -1)
    elif source_quant_type == QuantType.per_1x32:
        flat_scale = weight_scale.reshape(experts * rows, -1)
    else:
        raise ValueError(
            "Batched MoE online dequant does not support "
            f"source quant_type={source_quant_type}."
        )
    return dequant_weight_online(
        flat_weight.contiguous(),
        flat_scale.contiguous(),
        source_quant_type,
        source_quant_dtype,
    )


def quant_mxfp4_online_even(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online MXFP4 weight quant via the aiter HIP kernel with ``Even`` round mode.

    Round-half-to-even on the FP4/E2M1 grid + an E8M0 block scale (note: on
    gfx942 ``Even`` falls back to round-half-away in software). Returns the
    packed weight viewed as ``dtypes.fp4x2`` and the block scale as
    ``dtypes.fp8_e8m0``.

    Shared by the Linear and MoE online-quant paths so both stay in sync.
    ``quant_mxfp4_hip`` requires a 2D contiguous fp16/bf16 input, so we
    normalise the input accordingly before calling it.
    """
    from aiter import dtypes
    from aiter.ops.quant import quant_mxfp4_hip
    from aiter.utility.mx_types import MxScaleRoundModeInt

    q_in = weight.contiguous()
    if q_in.dtype not in (torch.float16, torch.bfloat16):
        q_in = q_in.to(torch.bfloat16)
    q_weight, weight_scale = quant_mxfp4_hip(q_in, round_mode=MxScaleRoundModeInt.Even)
    return q_weight.view(dtypes.fp4x2), weight_scale.view(_mx_block_scale_dtype())


def quantize_weight_to_fp8_128x128_blockscale(weight, quant_dtype):
    """Quantize a 2D weight to FP8 with 128x128 block scales.

    Returns:
        q_weight: quantized weight with the same shape as input ``weight``.
        scale: per-block scale with shape ``(ceil(N/128), ceil(K/128))``.
    """
    assert weight.dim() == 2, f"expected 2D weight, got shape={tuple(weight.shape)}"

    w = weight.to(torch.float32).contiguous()
    n, k = w.shape
    n_blocks = (n + 127) // 128
    k_blocks = (k + 127) // 128
    n_padded = n_blocks * 128
    k_padded = k_blocks * 128

    if n_padded != n or k_padded != k:
        w = torch.nn.functional.pad(w, (0, k_padded - k, 0, n_padded - n))

    w_blocks = w.view(n_blocks, 128, k_blocks, 128).permute(0, 2, 1, 3).contiguous()

    finfo = torch.finfo(quant_dtype)
    block_amax = w_blocks.abs().amax(dim=(2, 3))
    scale = (block_amax / finfo.max).clamp_min(torch.finfo(torch.float32).tiny)

    q_blocks = torch.clamp(
        w_blocks / scale.unsqueeze(-1).unsqueeze(-1), min=finfo.min, max=finfo.max
    ).to(quant_dtype)

    q_weight = (
        q_blocks.permute(0, 2, 1, 3)
        .contiguous()
        .view(n_padded, k_padded)[:n, :k]
        .contiguous()
    )
    return q_weight, scale.contiguous()


def quant_weight_online(
    weight: torch.Tensor,
    online_quant_type: QuantType,
    online_quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch online weight quantization by target dtype / scheme.

    Single entry point shared by the Linear and MoE online-quant paths so both
    stay in sync:

    - MXFP4 (``dtypes.fp4x2``): use the aiter HIP kernel with ``Even`` round
      mode (:func:`quant_mxfp4_online_even`), matching the offline Quark kernel.
    - per_1x128 FP8: use :func:`quantize_weight_to_fp8_128x128_blockscale` to
      produce a true 128x128 block scale of shape ``(N//128, K//128)``. This is
      what the blockscale GEMM consumes; ``get_hip_quant(per_1x128)`` would
      instead produce a 1x128-along-K scale ``(N, K//128)`` that is inconsistent
      with the GEMM and collapses generation.
    - MXFP8 (``per_1x32`` + ``dtypes.fp8``): fp8 weights with a per-32 block
      scale. See the e8m0 note below for why ``scale_type`` must be forced.
    - other FP8 (incl. ptpc_fp8 per-token / per-channel): use the aiter quant
      function resolved from ``get_hip_quant(online_quant_type)``.

    :param weight: The (already dequantized) weight tensor to quantize.
    :param online_quant_type: Online quantization scheme, used to resolve the
        FP8 quant function via ``get_hip_quant``.
    :param online_quant_dtype: Target online quantization dtype.
    :return: ``(q_weight, weight_scale)``.
    """
    from aiter import dtypes, get_hip_quant

    if online_quant_dtype == dtypes.fp4x2:
        return quant_mxfp4_online_even(weight)
    if online_quant_type == QuantType.per_1x128:
        return quantize_weight_to_fp8_128x128_blockscale(weight, online_quant_dtype)
    quant_func = get_hip_quant(online_quant_type)
    # A per_1x32 scheme *is* MX (microscaling): its block scale is E8M0 by
    # definition of the format, independent of the element dtype. So the scale
    # dtype is derived from the scheme, not chosen per case — the MXFP4 branch
    # above already relies on this (aiter forces e8m0 for fp4x2), and here the
    # MXFP8 (fp8) branch needs the same E8M0 scale. We only have to pass it
    # explicitly because aiter's per_1x32 fp8 quantizer keeps scale_type=fp32 as
    # a backward-compat default, whereas the whole consuming side is E8M0:
    # Fp8MoEMethod.create_weights allocates an e8m0 (uint8) scale buffer and the
    # MXFP8 GEMM / flydsl MoE kernels only read that byte scale. Without it the
    # scale silently reverts to fp32 and weight loading / inference break.
    if online_quant_type == QuantType.per_1x32:
        mx_scale_type = _mx_block_scale_dtype()
        return quant_func(
            weight,
            quant_dtype=online_quant_dtype,
            scale_type=mx_scale_type,
        )
    return quant_func(weight, quant_dtype=online_quant_dtype)
