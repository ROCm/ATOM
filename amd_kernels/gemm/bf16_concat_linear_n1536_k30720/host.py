# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import functools
import torch
from .kernel import compile_reduce
from .kernel_unroll import compile_concat_linear_unroll
from .kernel_segmented import compile_concat_linear_segmented
from .kernel_staged_a import compile_concat_linear_staged_a
from .kernel_staged_a_eightwave import compile_concat_linear_staged_a_eightwave

MAX_ROWS=2048  # the tiers' row bound (a caller with more rows splits them)


@functools.cache
def get_kernel(mt,nt,split):
    # Cold graph A/B: each slab keeps the original MFMA order.
    if (mt,nt,split)==(1,4,32):
        return compile_concat_linear_unroll(mt,nt,split,2)
    if (mt,nt,split)==(5,3,32):
        return compile_concat_linear_segmented(mt,nt,split,2)
    if (mt,nt,split) in ((8,6,8),(8,6,4)):
        # nt encodes block_n / 64. Eight N waves retain the 128x384 tile
        # with half as many columns per wave and the same K/slab order.
        return compile_concat_linear_staged_a_eightwave(mt,nt//2,split,2)
    assert (mt,nt,split) in ((8,3,32),(8,3,16),(8,3,8)),(mt,nt,split)
    return compile_concat_linear_staged_a(mt,nt,split,2)


@functools.cache
def get_reduce(split):return compile_reduce(split)


def run(exe,*args):
    import flydsl.compiler as flyc
    if not hasattr(exe,'_cf'):exe._cf=flyc.compile(exe,*args)
    else:exe._cf(*args)


def _partials_config(taps,weight,mt,nt,split):
    rows=taps[0].shape[0]
    partials=torch.empty((split,rows,1536),device=weight.device,dtype=torch.float32)
    run(get_kernel(mt,nt,split),*(x.data_ptr() for x in taps),weight.data_ptr(),partials.data_ptr(),rows,*(x.stride(0) for x in taps),torch.cuda.current_stream())
    return partials


def _linear_config(taps,weight,mt,nt,split,out=None):
    rows=taps[0].shape[0]
    partials=_partials_config(taps,weight,mt,nt,split)
    out=out if out is not None else torch.empty((rows,1536),device=weight.device,dtype=torch.bfloat16)
    run(get_reduce(split),partials,out,rows,torch.cuda.current_stream())
    return out


def pack_weight(weight):
    """Lossless load-time layout; replace and release the row-major weight."""
    return weight.view(1536//16,16,30720//32,4,8).permute(0,2,3,1,4).contiguous().view(1536,30720)


def config_for(rows):
    """Cold-weight decode tiers; preserve each output column's original MFMA and slab-sum order."""
    if rows<=16:return 1,4,32
    if rows<=63:return 5,3,32
    if rows<=128:return 8,3,32
    if rows<=256:return 8,3,16
    if rows<=512:return 8,3,8
    if rows<=1024:return 8,6,8
    assert rows<=MAX_ROWS,rows
    return 8,6,4


def bf16_concat_linear(taps,weight,out=None):
    """Project five bf16 taps in place through the single packed N-shard."""
    assert len(taps)==5 and weight.shape==(1536,30720) and weight.dtype==torch.bfloat16 and weight.is_contiguous()
    m=taps[0].shape[0]
    assert 0<m<=MAX_ROWS
    for x in taps:
        assert x.shape==(m,6144) and x.dtype==torch.bfloat16 and x.stride(1)==1 and x.stride(0)%8==0 and x.device==weight.device
    if out is not None:assert out.shape==(m,1536) and out.dtype==torch.bfloat16 and out.is_contiguous() and out.device==weight.device
    return _linear_config(taps,weight,*config_for(m),out)


def warmup_concat_linear(taps,weight,max_rows=2048):
    for m in (8,17,80,128,256,512,1024,2048):
        if m<=max_rows:
            xs=[x[:m] for x in taps]
            bf16_concat_linear(xs,weight)
