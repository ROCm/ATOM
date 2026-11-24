# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eager four-rank N-shard projection with a two-slot peer gather."""
import ctypes
import functools
import torch
import torch.distributed as dist
from amd_kernels.common import hip_runtime as ops
from .gather import DATA_OFF,compile_gather
from .host import bf16_concat_linear,run
from . import host
from .reduce_gather import compile_reduce_gather


class _DevicePointer:
    """A raw device allocation as a tensor (``__cuda_array_interface__``)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def exchange(obj, rank: int, group) -> list:
    """Every rank's ``obj`` (broadcasts on the CPU group: all_gather_object fails
    on gloo under inference mode)."""
    out = [[None] for _ in range(4)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


@functools.cache
def gather_kernel(rpw):return compile_gather(rpw)


@functools.cache
def reduce_gather_kernel(rpw,split):return compile_reduce_gather(rpw,split)


class ConcatLinearComm:
    def __init__(self,packed_weight,rank,group,max_rows=2048,*,fuse_reduce=True):
        assert dist.get_world_size(group)==4 and packed_weight.shape==(1536,30720) and 1<=max_rows<=host.MAX_ROWS
        self.weight,self.rank,self.group=packed_weight,rank,group
        self.fuse_reduce=fuse_reduce
        self.max_rows=max_rows
        self.slot_bytes=max_rows*1536*2
        ptr,handle=ops.allocate_shared_buffer_and_handle(DATA_OFF+2*self.slot_bytes)
        every=exchange(handle,rank,group)
        self.bufs=[ptr if r==rank else ops.open_mem_handle(every[r]) for r in range(4)]
        self.stage=torch.as_tensor(_DevicePointer(ptr+DATA_OFF,2*self.slot_bytes),device=packed_weight.device).view(torch.bfloat16).view(2,max_rows,1536)
        self.counts=torch.zeros(256,dtype=torch.int32,device=packed_weight.device)
        self.call=0
        torch.cuda.synchronize();dist.barrier(group=group)

    def __call__(self,taps,out):
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('full concat-linear communicator is eager only; Python slot rotation cannot replay')
        m=taps[0].shape[0]
        assert m<=self.max_rows and out.shape==(m,6144) and out.is_contiguous()
        slot=self.call%2;self.call+=1
        cfg=host.config_for(m)
        partials=None
        if self.fuse_reduce:
            partials=host._partials_config(taps,self.weight,*cfg)
        else:
            bf16_concat_linear(taps,self.weight,out=self.stage[slot,:m])
        rpw=1
        while (m+rpw-1)//rpw>256:rpw*=2
        args=(*self.bufs,self.counts.data_ptr(),out,m,self.rank,slot,self.slot_bytes,torch.cuda.current_stream())
        if self.fuse_reduce:
            run(reduce_gather_kernel(rpw,cfg[2]),partials,*args)
        else:
            run(gather_kernel(rpw),*args)
        return out

    def warm_up(self):
        for m in (8,17,80,128,256,512,1024,2048):
            if m<=self.max_rows:
                xs=[torch.zeros((m,6144),dtype=torch.bfloat16,device=self.weight.device) for _ in range(5)]
                out=torch.empty((m,6144),dtype=torch.bfloat16,device=self.weight.device)
                self(xs,out)

    def close(self):
        torch.cuda.synchronize();dist.barrier(group=self.group)
        hip=ops._hip()
        hip.hipFree.argtypes=[ctypes.c_void_p];hip.hipIpcCloseMemHandle.argtypes=[ctypes.c_void_p]
        for r,p in enumerate(self.bufs):
            rc=(hip.hipFree if r==self.rank else hip.hipIpcCloseMemHandle)(ctypes.c_void_p(p))
            assert rc==0,rc
        self.bufs=[]
