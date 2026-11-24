# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""A collective load-time mapping of the four vocabulary shards; no weight copy."""
import ctypes
import functools
import torch
import torch.distributed as dist
from .kernel import H,compile_vocab_gather
from .publish import GROUPS, compile_vocab_weight_publish
from .ipc import _alloc_handle,_open_handle,_run_compiled,exchange
from amd_kernels.common.hip_runtime import load_hip_runtime


@functools.cache
def _kernel(dtype):return compile_vocab_gather(32 if dtype==torch.int32 else 64)


class VocabGather:
    """Keep the checkpoint's bf16 shard on its owner and read it directly.

    Construct once on each of four TP ranks after loading weights. Shards have
    the same vocabulary size, rank order, hidden 6144, and contiguous bf16 data.
    The weights remain alive and immutable until ``close()``. No GPU copy is
    retained. Eager and graph calls launch the same stateless FlyDSL kernel.
    """
    def __init__(self,weight,rank,group):
        assert dist.get_world_size(group)==4
        assert weight.ndim==2 and weight.shape[1]==H and weight.dtype==torch.bfloat16 and weight.is_contiguous()
        if torch.cuda.current_device()!=weight.device.index or dist.get_rank(group)!=rank:
            raise ValueError('select this rank device before mapping weights')
        self.weight,self.rank,self.group=weight,rank,group
        self.shard_vocab=weight.shape[0]
        # A prior GPU load/preshuffle may leave dirty owner L2 lines. Publish
        # every XCD once before exposing these immutable weights to peers.
        done=torch.empty(GROUPS,dtype=torch.int32,device=weight.device)
        _run_compiled(compile_vocab_weight_publish(),done,torch.cuda.current_stream())
        base,handle=_alloc_handle(weight.data_ptr())
        torch.cuda.synchronize(weight.device)
        every=exchange((handle,weight.data_ptr()-base,self.shard_vocab),rank,group)
        if any(v[2]!=self.shard_vocab for v in every):raise ValueError('unequal vocabulary shards')
        self.opened={}
        pointers=[]
        for r,(raw,offset,_) in enumerate(every):
            if r==rank:pointers.append(weight.data_ptr())
            else:
                if raw not in self.opened:self.opened[raw]=_open_handle(raw)
                pointers.append(self.opened[raw]+offset)
        self.peers=torch.tensor(pointers,dtype=torch.int64,device=weight.device)
        torch.cuda.synchronize(weight.device)
        dist.barrier(group=group)

    def __call__(self,ids):
        assert ids.ndim==1 and ids.is_contiguous() and ids.dtype in (torch.int32,torch.int64) and ids.device==self.weight.device
        m=ids.numel()
        out=torch.empty((m,H),dtype=torch.bfloat16,device=ids.device)
        if m:_run_compiled(_kernel(ids.dtype),ids,self.peers,out,m,self.shard_vocab,torch.cuda.current_stream())
        return out

    def warm_up(self):
        for dtype in (torch.int32,torch.int64):
            self(torch.zeros(1,dtype=dtype,device=self.weight.device))

    def close(self):
        torch.cuda.synchronize(self.weight.device)
        dist.barrier(group=self.group)
        hip=load_hip_runtime()
        hip.hipIpcCloseMemHandle.argtypes=[ctypes.c_void_p]
        for p in self.opened.values():
            rc=hip.hipIpcCloseMemHandle(ctypes.c_void_p(p))
            assert rc==0,rc
        self.opened.clear()
