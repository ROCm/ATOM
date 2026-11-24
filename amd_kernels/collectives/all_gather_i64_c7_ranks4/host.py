# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Reuse the sampler gather's IPC allocation and teardown, with a separate sequence."""
import functools
import torch
from amd_kernels.collectives.all_gather_sampler_i64_c8_ranks4.host import SamplerAllGather,_run
from .kernel import compile_i64_c7_gather

@functools.cache
def _exe():return compile_i64_c7_gather()

class I64C7AllGather(SamplerAllGather):
    def __call__(self,x,src,src_stride,out):
        assert not self.closed
        assert x.ndim==2 and x.shape[1]==7 and x.dtype==torch.int64 and x.stride(1)==1
        assert src.ndim==1 and src.dtype==torch.int64 and src.is_contiguous()
        assert out.shape==(src.numel(),7) and out.dtype==torch.int64 and out.is_contiguous()
        assert x.device==src.device==out.device==self.device
        assert 0<=x.shape[0]<=256 and 0<=src.numel()<=256 and 1<=src_stride<=256
        _run(_exe(),x.data_ptr(),src.data_ptr(),out.data_ptr(),*self.ptrs,*self.flags,self.counter,
             x.shape[0],src.numel(),src_stride,x.stride(0),self.rank,torch.cuda.current_stream(self.device))
        return out

    def warmup(self):
        dev=self.device
        x=torch.zeros(1,7,dtype=torch.int64,device=dev)
        src=torch.tensor([self.rank],dtype=torch.int64,device=dev)
        self(x,src,1,torch.empty_like(x));torch.cuda.synchronize(dev)
