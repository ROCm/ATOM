# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import functools, struct
import torch
from .kernel import compile_gemma_rmsnorm_bf16_mxfp8
from .kernel_fourwave import compile_gemma_rmsnorm_bf16_mxfp8_fourwave

@functools.cache
def get_kernel(tiled,with_bf16,fourwave=False):
    compile_norm=compile_gemma_rmsnorm_bf16_mxfp8_fourwave if fourwave else compile_gemma_rmsnorm_bf16_mxfp8
    return compile_norm(tiled,with_bf16)

def gemma_rmsnorm_mxfp8(x,weight,eps,code_tile_rows=0,*,bf16_out=None):
    m=x.shape[0]
    assert x.dtype==torch.bfloat16 and x.ndim==2 and x.shape[1]==6144 and x.stride(1)==1
    assert weight.shape==(6144,) and weight.dtype==torch.bfloat16 and weight.is_contiguous() and weight.device==x.device
    assert code_tile_rows>=0 and code_tile_rows%16==0
    rows=(m+code_tile_rows-1)//code_tile_rows*code_tile_rows if code_tile_rows else m
    q=torch.empty((rows,6144),device=x.device,dtype=torch.float8_e4m3fn)
    scale=torch.empty(((rows+31)//32*32,192),device=x.device,dtype=torch.uint8)
    if bf16_out is not None:
        assert bf16_out.shape==x.shape and bf16_out.dtype==x.dtype and bf16_out.device==x.device and bf16_out.is_contiguous()
    if m:
        exe=get_kernel(code_tile_rows>0,bf16_out is not None,m>512)
        args=(x.data_ptr(),weight.data_ptr(),q.data_ptr(),scale.data_ptr(),0 if bf16_out is None else bf16_out.data_ptr(),q.numel(),scale.numel(),m,x.stride(0),struct.unpack('<i',struct.pack('<f',eps))[0],torch.cuda.current_stream())
        if getattr(exe,'_cf',None) is None:  # a compile-only call (a warm-up) leaves None: compile again
            import flydsl.compiler as flyc
            exe._cf=flyc.compile(exe,*args)
        else:exe._cf(*args)
    return q,scale

def warmup_gemma_rmsnorm_mxfp8(device='cuda'):
    for m in (16,513):
        x=torch.zeros((m,6144),device=device,dtype=torch.bfloat16);w=x[0]
        for tile in (0,16):
            for out in (None,torch.empty_like(x)):
                gemma_rmsnorm_mxfp8(x,w,1e-6,tile,bf16_out=out)
