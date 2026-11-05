"""MXFP8 activation quant (fp8 e4m3 values, one e8m0 scale per 128-column group) for fusing into the kernel that
produces a projection GEMM's input.

Bit-identical to aiter's standalone ``per_group_quant_hip(x_bf16, quant_dtype=fp8_e4m3fn, group_size=128,
transpose_scale=True, scale_type=fp8_e8m0)`` -- the quant the per_1x128 e8m0 Linear runs on its input:

* the group amax is taken over the **bf16-rounded** values (what the unfused producer would have stored), starting
  from aiter's ``kAmaxFloor`` (1e-10, so an all-zero group gets byte 85) and capped at 448 * 2^119 (an inf group
  still gets a finite scale, as the hardware-convert path does);
* scale = RoundUp ``ceil_pow2(amax * (1/448))`` (aiter ``fp_f32_to_e8m0_scale<RoundUp, FP8_E4M3>``): the fp32
  exponent of ``amax * (1/448)``, plus one when its mantissa is non-zero; the e8m0 byte is that exponent;
* value = RNE(x * 2^-(e - 127)) to e4m3 (exact power-of-two scaling, as ``v_cvt_scalef32_pk8_fp8_bf16``);
* scale layout ``transpose_scale``: byte of (row m, group g) at ``g * M + m`` (M = rows of this call).

The caller hands the result to the Linear as ``linear(x_fp8, x_scale=scale.view(M, K // 128))``.
"""

import torch
import triton
import triton.language as tl

GROUP = 128


@triton.jit
def e8m0_group_scale(amax):
    """amax (fp32, >= 0) -> (e8m0 exponent byte as int32, 2^-(e-127) as fp32)."""
    u = (amax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
    e = ((u >> 23) & 0xFF).to(tl.int32)
    e = e + (((u & 0x7FFFFF) != 0) & (e < 0xFF)).to(tl.int32)
    # 2^(127 - e): exponent field 254 - e (e = 0, an all-zero group, gives 2^127; 0 * 2^127 = 0)
    inv = ((254 - tl.minimum(e, 253)).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    return e, inv


@triton.jit
def e8m0_quant_group(y):
    """y: [..., 128] fp32 holding bf16-representable values (one group along the last axis).
    Returns (fp8 e4m3 values [..., 128], e8m0 bytes [...] as uint8)."""
    amax = tl.minimum(tl.maximum(tl.max(tl.abs(y), axis=-1), 1e-10), 448.0 * 6.646139978924579e35)  # 2^119
    e, inv = e8m0_group_scale(amax)
    q = (y * tl.expand_dims(inv, -1)).to(tl.float8e4nv)
    return q, e.to(tl.uint8)


def alloc_mxfp8(m: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8 [m, k] values and the transposed e8m0 scale buffer, returned as the (m, k // 128) view the GEMM takes."""
    assert k % GROUP == 0, k
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=device)
    s = torch.empty((k // GROUP) * m, dtype=torch.uint8, device=device)
    return q, s


def scale_view(s: torch.Tensor, m: int, k: int) -> torch.Tensor:
    return s.view(torch.float8_e8m0fnu).view(m, k // GROUP)


def reference(y_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The unfused path: aiter per_group_quant_hip on the bf16 tensor (for tests)."""
    from aiter import dtypes
    from aiter.ops.quant import per_group_quant_hip

    return per_group_quant_hip(y_bf16.contiguous(), quant_dtype=torch.float8_e4m3fn, group_size=GROUP,
                               transpose_scale=True, scale_type=dtypes.fp8_e8m0)
