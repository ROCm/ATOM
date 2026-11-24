# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One graph-safe four-rank communicator per stream, constructed collectively."""
import ctypes
import functools
import torch
import torch.distributed as dist
from amd_kernels.collectives.all_to_all_bf16_n50016_ranks4.host import _alloc_pool,_open_pool,_exchange,_run
from amd_kernels.common import hip_runtime as ops
from .kernel import compile_sampler_gather,SLOT_BYTES

@functools.cache
def _exe():return compile_sampler_gather()

class SamplerAllGather:
    def __init__(self,rank,group,device):
        assert dist.get_world_size(group)==4 and rank==dist.get_rank(group)
        self.rank,self.group,self.device=rank,group,torch.device(device)
        ptr,handle,self.pool=_alloc_pool(2*SLOT_BYTES,self.device)
        flag,fh=ops.allocate_shared_buffer_and_handle(256)
        hs=_exchange((handle,fh),rank,4,group)
        self.ptrs=[ptr if r==rank else _open_pool(h,self.device) for r,(h,_) in enumerate(hs)]
        self.flags=[flag if r==rank else ops.open_mem_handle(h) for r,(_,h) in enumerate(hs)]
        self.counter=torch.zeros(1,dtype=torch.int32,device=self.device)
        self.device=self.counter.device
        self.closed=False

    def __call__(self,tokens,num_sampled,num_rejected,num_nans,cu,src,src_stride,out_tokens,out_sampled,out_rejected,out_nans):
        """All ranks call, including empty senders/receivers. Map/count values
        may change between graph replays. `cu` has <=8 logit rows per request;
        `src` references only initialized local rows. NaN presence agrees on
        every rank. The caller supplies all outputs, so capture allocates none.
        """
        assert not self.closed
        n=tokens.shape[0];no=src.numel()
        assert tokens.dtype==torch.int64 and tokens.ndim==2 and 1<=tokens.shape[1]<=8 and tokens.stride(1)==1
        assert 0<=n<=256 and 0<=no<=256 and 1<=src_stride<=256
        assert src.dtype==torch.int64 and src.is_contiguous() and src.ndim==1
        assert num_sampled.shape==num_rejected.shape==(n,) and num_sampled.dtype==num_rejected.dtype==torch.int32
        assert cu.shape==(n+1,) and cu.dtype==torch.int32
        assert out_tokens.shape==(no,8) and out_tokens.dtype==torch.int64 and out_tokens.is_contiguous()
        assert out_sampled.shape==out_rejected.shape==(no,) and out_sampled.dtype==out_rejected.dtype==torch.int32
        arrays=[tokens,num_sampled,num_rejected,cu,src,out_tokens,out_sampled,out_rejected]
        assert all(a.device==self.device for a in arrays)
        assert all(a.is_contiguous() for a in arrays[1:])
        has=num_nans is not None
        if has:
            assert num_nans.dtype==out_nans.dtype==torch.int32 and num_nans.ndim==1 and out_nans.shape==(no,)
            assert num_nans.device==out_nans.device==self.device and num_nans.is_contiguous() and out_nans.is_contiguous()
        else:assert out_nans is None
        args=(tokens.data_ptr(),num_sampled.data_ptr(),num_rejected.data_ptr(),num_nans.data_ptr() if has else 0,
              cu.data_ptr(),src.data_ptr(),out_tokens.data_ptr(),out_sampled.data_ptr(),out_rejected.data_ptr(),
              out_nans.data_ptr() if has else 0,*self.ptrs,*self.flags,self.counter,n,no,src_stride,tokens.stride(0),
              tokens.shape[1],self.rank,int(has),torch.cuda.current_stream(self.device))
        _run(_exe(),*args)
        return out_tokens,out_sampled,out_rejected,out_nans

    def warmup(self):
        dev=self.device
        a=torch.zeros(1,8,dtype=torch.int64,device=dev);c=torch.ones(1,dtype=torch.int32,device=dev)
        cu=torch.arange(2,dtype=torch.int32,device=dev);src=torch.tensor([self.rank],dtype=torch.int64,device=dev)
        self(a,c,c,c,cu,src,1,torch.empty_like(a),torch.empty_like(c),torch.empty_like(c),torch.empty_like(c))
        torch.cuda.synchronize(dev)

    def close(self):
        if self.closed:return
        torch.cuda.synchronize(self.device);dist.barrier(group=self.group)
        lib=ops.load_hip_runtime();lib.hipIpcCloseMemHandle.argtypes=[ctypes.c_void_p];lib.hipFree.argtypes=[ctypes.c_void_p]
        for r in range(4):
            if r!=self.rank:
                for p in (self.ptrs[r],self.flags[r]):assert lib.hipIpcCloseMemHandle(ctypes.c_void_p(p))==0
        dist.barrier(group=self.group)
        for p in (self.ptrs[self.rank],self.flags[self.rank]):assert lib.hipFree(ctypes.c_void_p(p))==0
        self.pool=None;self.closed=True
