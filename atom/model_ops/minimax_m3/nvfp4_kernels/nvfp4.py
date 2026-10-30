# Derived from SGLang (https://github.com/sgl-project/sglang), Apache-2.0; modified for ATOM.
"""NVFP4 cache rows (MSA-v1 layout) and their gfx950 loads, in Triton and Gluon.

A row is one head of one token: ``D // 2`` payload bytes (two e2m1 codes per
byte, the even element in the low nibble) and ``D // 16`` E4M3 block scales,
with a unit outer scale -- the layout of the MiniMax-M3 NVFP4 sparse KV pool
(``MhaKvPool(kv_format="nvfp4")``) and of the FP4 index cache (``index_fp4``).
"""

import msgspec
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


class Nvfp4Cache(msgspec.Struct, frozen=True):
    """One layer of a packed NVFP4 cache (K, V or index K).

    ``payload`` is uint8 ``[pages, (heads,) page_size, D // 2]`` and ``scales``
    uint8 ``[pages, (heads,) page_size, D // 16]`` (E4M3 bits); the index-K
    cache has no head dimension.
    """

    payload: torch.Tensor
    scales: torch.Tensor

    @property
    def page_size(self) -> int:
        return self.payload.shape[-2]

    @property
    def device(self) -> torch.device:
        return self.payload.device

    @property
    def num_heads(self) -> int:
        return self.payload.shape[1] if self.payload.dim() == 4 else 1

    def payload_strides(self) -> tuple[int, int, int]:
        """(page, head, row) strides in bytes."""
        return _page_head_row_strides(self.payload)

    def scale_strides(self) -> tuple[int, int, int]:
        return _page_head_row_strides(self.scales)


def _page_head_row_strides(buf: torch.Tensor) -> tuple[int, int, int]:
    if buf.dim() == 4:
        return buf.stride(0), buf.stride(1), buf.stride(2)
    return buf.stride(0), 0, buf.stride(1)


@triton.jit
def _fp4x2_to_f16(byte_i32):
    """gfx950: one packed e2m1 byte -> (low, high) fp16 in one conversion.

    The instruction's f32 scale only contributes its exponent (E8M0), so the
    E4M3 block scale is applied by the caller; 1.0 leaves the codes unscaled.
    """
    return tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_f16_fp4 $0, $2, 1.0\n\tv_lshrrev_b32 $1, 16, $0",
        "=&v,=&v,v",
        [byte_i32],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )


@triton.jit
def nvfp4_load_even_odd(
    payload_ptr,
    scale_ptr,
    payload_row_off,
    scale_row_off,
    row_mask,
    D: tl.constexpr,
):
    """Load rows as their even and odd columns, two fp16 ``[N, D//2]`` tiles.

    Byte ``j`` of a row holds elements ``2j`` (low nibble) and ``2j+1``, so the
    split needs no interleave; a dot over D is the sum of the even and odd dots.
    e2m1 times an E4M3 scale has at most 6 significant bits and stays in the fp16
    normal range, so the dequantized values are exact. gfx950 only.
    """
    HALF: tl.constexpr = D // 2
    cols = tl.arange(0, HALF)
    packed = tl.load(
        payload_ptr + payload_row_off[:, None] + cols[None, :],
        mask=row_mask[:, None],
        other=0,
    )
    scale_bytes = tl.load(
        scale_ptr + scale_row_off[:, None] + (cols // 8)[None, :],
        mask=row_mask[:, None],
        other=0,
    )
    even, odd = _fp4x2_to_f16(packed.to(tl.int32))
    scale = scale_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    return even * scale, odd * scale


# Gluon reads for the gfx950 attention and fused top-k kernels.
#
# The payload is read as u16, two bytes holding four e2m1 codes, so one register feeds one
# conversion and the even (low-nibble) and odd elements come out as separate fp16 tiles,
# the split `nvfp4_load_even_odd` makes for the Triton kernels. A dot over the head dim
# is then the even-dim dot accumulated before the odd-dim one, as those kernels chain it.


def fits_buffer_offsets(cache: Nvfp4Cache) -> bool:
    """Whether 32-bit byte offsets from each buffer's base reach its last element; a
    pool page-row view spans more bytes than it holds."""
    return all(
        buf.element_size()
        * (1 + sum((n - 1) * s for n, s in zip(buf.shape, buf.stride())))
        < 2**31
        for buf in (cache.payload, cache.scales)
    )


def nvfp4_buffer_loadable(cache: Nvfp4Cache) -> bool:
    """Whether the kernels can read ``cache``: u16 payload view, 32-bit buffer offsets."""
    payload, scales = cache.payload, cache.scales
    return (
        payload.stride(-1) == 1
        and scales.stride(-1) == 1
        and all(s % 2 == 0 for s in (*payload.stride()[:-1], payload.storage_offset()))
        and fits_buffer_offsets(cache)
    )


@gluon.jit
def _fp4x4_to_f16(payload_u16, scale, PURE: gl.constexpr):
    """Two payload u16 and their fp16 block scale -> fp16 pairs (e0, e1), (e2, e3) of
    the even and (o0, o1), (o2, o3) of the odd elements of the four bytes.

    The conversion's f32 scale only applies an exponent, so it gets 1.0 and the E4M3
    scale is multiplied after; e2m1 times E4M3 is exact in fp16 (normal range, at most
    6 significant bits), so these are the values of `nvfp4_kv_dequantize`.
    """
    e01, e23, o01, o23, _ = gl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_f16_fp4 $1, $5, 1.0\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $4, $5, 1.0 op_sel:[1,0,0]\n\t"
        "v_pk_mul_f16 $1, $1, $6\n\t"
        "v_pk_mul_f16 $4, $4, $6\n\t"
        "v_pack_b32_f16 $0, $1, $4\n\t"
        "v_pack_b32_f16 $2, $1, $4 op_sel:[1,1,0]\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $1, $5, 1.0 op_sel:[0,1,0]\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $4, $5, 1.0 op_sel:[1,1,0]\n\t"
        "v_pk_mul_f16 $1, $1, $6\n\t"
        "v_pk_mul_f16 $4, $4, $6\n\t"
        "v_pack_b32_f16 $3, $1, $4 op_sel:[1,1,0]\n\t"
        "v_pack_b32_f16 $1, $1, $4",
        "=&v,=&v,=&v,=&v,=&v,v,v",
        [payload_u16, scale],
        dtype=(gl.float16, gl.float16, gl.float16, gl.float16, gl.float16),
        is_pure=PURE,
        pack=2,
    )
    return e01, e23, o01, o23


@gluon.jit
def _interleave_pairs_w(lo, hi):
    """[W, G, 4, N] pairs of a 16-element group -> [W, 8 * G, N]: u16 r of group g gives
    rows 8g + 4 * (r // 2) + r % 2 (``lo``) and those + 2 (``hi``)."""
    W: gl.constexpr = lo.shape[0]
    G: gl.constexpr = lo.shape[1]
    N: gl.constexpr = lo.shape[3]
    x = gl.join(gl.reshape(lo, [W, G, 2, 2, N]), gl.reshape(hi, [W, G, 2, 2, N]))
    return gl.reshape(gl.permute(x, [0, 1, 2, 5, 3, 4]), [W, G * 8, N])


@gluon.jit
def nvfp4_even_odd_w(payload, scales, GROUP_LAYOUT: gl.constexpr):
    """Payload [W, D // 4, N] u16 and E4M3 scales [W, D // 16, N] (per warp, dims x
    tokens) -> fp16 [W, D // 2, N] even and odd dims. ``GROUP_LAYOUT`` is the
    [W, D // 16, 4, N] view of the payload's layout, each thread's u16 of a group
    contiguous; ``scales`` is in its slice along dim 2."""
    W: gl.constexpr = payload.shape[0]
    U: gl.constexpr = payload.shape[1]
    N: gl.constexpr = payload.shape[2]
    codes = gl.convert_layout(gl.reshape(payload, [W, U // 4, 4, N]), GROUP_LAYOUT)
    scale = gl.expand_dims(scales.to(gl.float8e4nv, bitcast=True).to(gl.float16), 2)
    # Not pure: the side effects keep the next block's prefetch, issued before it in
    # program order, from being scheduled after the dequant, where it hides no latency.
    e01, e23, o01, o23 = _fp4x4_to_f16(codes, scale, False)
    return _interleave_pairs_w(e01, e23), _interleave_pairs_w(o01, o23)
