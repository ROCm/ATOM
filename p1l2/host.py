"""p1/L2-attn-decode overlay host: run the amd-kernels decode host (nvfp4_sparse_decode) with an overlay kernel
swapped in for the variant the host would pick (same arguments, grid and split schedule).
variants: d2 = dist2 tail (p1l2.sel_d2 for the row_xcd select kernel at <= 32 rows; p1l2.gen_d2 for kernel.py)."""
import functools

from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import host as H


@functools.cache
def _sel(variant, splits, quant, exact_fp16):
    assert exact_fp16
    if variant == "d2":
        from p1l2.sel_d2 import compile_nvfp4_f16mfma_gqa16_d128_p128_top16_q8_rowxcd_select as comp
        return comp(splits, quant)
    raise ValueError(variant)


@functools.cache
def _gen(variant, splits, topk, xcds, quant, exact_fp16, tail=False, fp8=False, row_xcd=False, prefetch=False,
         fixed=False):
    assert exact_fp16 and not (tail or fp8 or row_xcd or prefetch or fixed)
    if variant == "d2":
        from p1l2.gen_d2 import compile_nvfp4_sparse_decode as comp
        return comp(splits, topk, xcds, quant, exact_fp16, tail)
    raise ValueError(variant)


def sparse_decode(variant, *args, **kwargs):
    o_sel, o_gen = H._get_kernel_select, H._get_kernel
    H._get_kernel_select = lambda s, q, e: _sel(variant, s, q, e)
    H._get_kernel = lambda *a, **k: _gen(variant, *a, **k)
    try:
        H.nvfp4_sparse_decode(*args, **kwargs)
    finally:
        H._get_kernel_select, H._get_kernel = o_sel, o_gen
