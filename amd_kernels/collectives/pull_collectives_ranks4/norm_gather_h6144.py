# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Dense Gemma add/RMSNorm/FP32 MXFP8 quant and chunk-ready all-gather.

Eight-wave arithmetic is the dense tiled norm's at every row count. Code rows
are row-major; each rank's scale block uses AITER's shuffled layout. Persistent
stagers write through L2 before taking a chunk ticket. Pullers copy a chunk as
soon as its ticket completes. Only tickets use atomic arithmetic.

Each communicator owns two staging slots and a device generation. A peer can
read slot c%2 until call c+1 publishes; call c+2 cannot start before that peer
completed c. No peer reads the producer's input, only communicator staging.
"""
import functools
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.common import (
    H, PRELOAD_HINTS, bf16x8, pk_bf16, seq_sum_sq, wave_sum_lane63, rsqrt_rn,
)
from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.add_rmsnorm_mxfp8 import (
    _block_sum, _codes, _e8m0, _scale_off, _set_fp16_ovfl,
)
from .host import WORLD, _alloc_pool, _exchange, _open_pool, _run_compiled, ops
from .kernel import SYS, _flag_add, _flag_read, _sel4

NW = 8
NT = NW * 64
PULLERS = 32
STAGERS = 192
PARTS = 8
FLAG_STRIDE = 128
MAX_PARTS = 32
END = MAX_PARTS * FLAG_STRIDE
FLAG_BYTES = END + FLAG_STRIDE


def compile_norm_gather_h6144(pullers=PULLERS, stagers=STAGERS, parts=PARTS):
    assert 1 <= parts <= MAX_PARTS
    assert pullers % 4 == 0 and stagers % 8 == 0 and pullers + stagers <= 256
    step = NT * 4 * 4
    name = f'gemma_add_rmsnorm_mxfp8_gather_ranks4_h6144_p{pullers}_n{stagers}_c{parts}'

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, NW]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        arg_x: fx.Int64, arg_r: fx.Int64, arg_w: fx.Int64, arg_res: fx.Int64,
        arg_q: fx.Int64, arg_s: fx.Int64,
        arg_p0: fx.Int64, arg_p1: fx.Int64, arg_p2: fx.Int64, arg_p3: fx.Int64,
        arg_f0: fx.Int64, arg_f1: fx.Int64, arg_f2: fx.Int64, arg_f3: fx.Int64,
        counter: fx.Tensor, i64_stride: fx.Int64, i64_scale_off: fx.Int64,
        i32_m: fx.Int32, i32_sx: fx.Int32, i32_sr: fx.Int32,
        i32_eps: fx.Int32, i32_me: fx.Int32,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        seq = fx.Int32(counter[0]) + 1
        slot = fx.Int64((seq - 1) & 1) * i64_stride
        own = _sel4(i32_me, arg_f0, arg_f1, arg_f2, arg_f3)
        if bid >= pullers:
            _set_fp16_ovfl()
            sid = bid - pullers
            lane = tx % 64
            wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
            shared = fx.SharedAllocator().allocate(Shared).peek()
            part = shared.part.ptr
            pool = _sel4(i32_me, arg_p0, arg_p1, arg_p2, arg_p3) + slot
            qr = buffer_ops.create_buffer_resource_from_addr(pool, num_records_bytes=fx.Int64(i32_m) * H)
            sr = buffer_ops.create_buffer_resource_from_addr(pool + i64_scale_off, num_records_bytes=fx.Int64(i32_m) * (H//32))
            wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=H*2)
            has1 = tx < 256
            offs = [tx*4, tx*4+2048]
            masks = [None, has1]
            wv = [bf16x8(Vec(buffer_ops.buffer_load(wr, offs[c], vec_width=4, dtype=fx.Int32, mask=masks[c]))) for c in range_constexpr(2)]
            k = fx.Int32(0)
            while k < parts:
                first = (k * (i32_m//32) // parts) * 32
                last = ((k+1) * (i32_m//32) // parts) * 32
                row = first + sid
                while row < last:
                    xr = buffer_ops.create_buffer_resource_from_addr(arg_x + fx.Int64(row)*fx.Int64(i32_sx*2), num_records_bytes=H*2)
                    rr = buffer_ops.create_buffer_resource_from_addr(arg_r + fx.Int64(row)*fx.Int64(i32_sr*2), num_records_bytes=H*2)
                    resr = buffer_ops.create_buffer_resource_from_addr(arg_res + fx.Int64(row)*(H*2), num_records_bytes=H*2)
                    xv = [bf16x8(Vec(buffer_ops.buffer_load(xr, offs[c], vec_width=4, dtype=fx.Int32, mask=masks[c]))) for c in range_constexpr(2)]
                    rv = [bf16x8(Vec(buffer_ops.buffer_load(rr, offs[c], vec_width=4, dtype=fx.Int32, mask=masks[c]))) for c in range_constexpr(2)]
                    vals = [[xv[c][i]+rv[c][i] for i in range_constexpr(8)] for c in range_constexpr(2)]
                    for c in range_constexpr(2):
                        packed = Vec.from_elements([pk_bf16(vals[c][2*i], vals[c][2*i+1]) for i in range_constexpr(4)], fx.Int32)
                        buffer_ops.buffer_store(packed,resr,offs[c],mask=masks[c])
                    acc = wave_sum_lane63(seq_sum_sq(vals[0]+vals[1]))
                    if lane == 63:
                        part[wave] = acc
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                    fx.gpu.barrier()
                    inv = rsqrt_rn(_block_sum([part[i] for i in range_constexpr(NW)]), i32_eps.bitcast(fx.Float32))
                    for c in range_constexpr(2):
                        y = [(vals[c][i]*inv)*(wv[c][i]+fx.Float32(1.0)) for i in range_constexpr(8)]
                        sb, rescale = _e8m0(y)
                        words = _codes(y,rescale)
                        buffer_ops.buffer_store(Vec.from_elements(words,fx.Int32),qr,row*(H//4)+tx*2+c*1024,mask=masks[c],cache_modifier=SYS)
                        buffer_ops.buffer_store(sb.to(fx.Int32).to(fx.Int8),sr,_scale_off(row,tx//4+c*128),mask=((tx%4)==0) & ((tx<256) if c else (tx<512)),offset_is_bytes=True,cache_modifier=SYS)
                    # All waves must finish reading the previous row's LDS sums.
                    fx.gpu.barrier()
                    row = row + stagers
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                if tx == 0:
                    flag = own + fx.Int64(k*FLAG_STRIDE)
                    if _flag_add(flag+4,1) == stagers-1:
                        _flag_add(flag+4,-stagers)
                        fr = buffer_ops.create_buffer_resource_from_addr(flag,num_records_bytes=4)
                        buffer_ops.buffer_store(seq,fr,0,cache_modifier=SYS)
                k = k+1
        else:
            peer = bid % 4
            flags = _sel4(peer,arg_f0,arg_f1,arg_f2,arg_f3)
            src = _sel4(peer,arg_p0,arg_p1,arg_p2,arg_p3)+slot
            qsrc = buffer_ops.create_buffer_resource_from_addr(src,num_records_bytes=fx.Int64(i32_m)*H)
            ssrc = buffer_ops.create_buffer_resource_from_addr(src+i64_scale_off,num_records_bytes=fx.Int64(i32_m)*(H//32))
            qdst = buffer_ops.create_buffer_resource_from_addr(arg_q+fx.Int64(peer)*fx.Int64(i32_m)*H,num_records_bytes=fx.Int64(i32_m)*H)
            sdst = buffer_ops.create_buffer_resource_from_addr(arg_s+fx.Int64(peer)*fx.Int64(i32_m)*(H//32),num_records_bytes=fx.Int64(i32_m)*(H//32))
            j = fx.Int32(0)
            while j < parts:
                if tx == 0:
                    ready = _flag_read(flags+fx.Int64(j*FLAG_STRIDE))
                    while ready < seq:
                        ready = _flag_read(flags+fx.Int64(j*FLAG_STRIDE))
                fx.gpu.barrier()
                _llvm.FenceOp(_llvm.AtomicOrdering.acquire,syncscope='')
                r0 = (j*(i32_m//32)//parts)*32
                r1 = ((j+1)*(i32_m//32)//parts)*32
                off = r0*(H//4)+(bid//4)*step+tx*4
                while off < r1*(H//4):
                    qs = [Vec(buffer_ops.buffer_load(qsrc,off+u*NT*4,vec_width=4,dtype=fx.Int32,mask=off+u*NT*4<r1*(H//4),cache_modifier=SYS)) for u in range_constexpr(4)]
                    for u in range_constexpr(4):
                        buffer_ops.buffer_store(qs[u],qdst,off+u*NT*4,mask=off+u*NT*4<r1*(H//4))
                    off = off+step*(pullers//4)
                soff = r0*(H//128)+(bid//4)*NT*4+tx*4
                while soff < r1*(H//128):
                    sv = Vec(buffer_ops.buffer_load(ssrc,soff,vec_width=4,dtype=fx.Int32,mask=soff<r1*(H//128),cache_modifier=SYS))
                    buffer_ops.buffer_store(sv,sdst,soff,mask=soff<r1*(H//128))
                    soff = soff+NT*4*(pullers//4)
                j = j+1
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        if tx == 0:
            if _flag_add(own+END,1) == pullers+stagers-1:
                _flag_add(own+END,-(pullers+stagers))
                counter[0] = seq

    @flyc.jit
    def launch_norm_gather_h6144(
        arg_x: fx.Int64, arg_r: fx.Int64, arg_w: fx.Int64, arg_res: fx.Int64,
        arg_q: fx.Int64, arg_s: fx.Int64,
        arg_p0: fx.Int64, arg_p1: fx.Int64, arg_p2: fx.Int64, arg_p3: fx.Int64,
        arg_f0: fx.Int64, arg_f1: fx.Int64, arg_f2: fx.Int64, arg_f3: fx.Int64,
        counter: fx.Tensor, i64_stride: fx.Int64, i64_scale_off: fx.Int64,
        i32_m: fx.Int32, i32_sx: fx.Int32, i32_sr: fx.Int32,
        i32_eps: fx.Int32, i32_me: fx.Int32, stream: fx.Stream,
    ):
        kernel(arg_x,arg_r,arg_w,arg_res,arg_q,arg_s,arg_p0,arg_p1,arg_p2,arg_p3,arg_f0,arg_f1,arg_f2,arg_f3,counter,i64_stride,i64_scale_off,i32_m,i32_sx,i32_sr,i32_eps,i32_me).launch(grid=(pullers+stagers,1,1),block=(NT,1,1),stream=stream)
    launch_norm_gather_h6144.kernel_name = name
    launch_norm_gather_h6144.compile_hints = PRELOAD_HINTS
    return launch_norm_gather_h6144


@functools.cache
def _kernel(pullers,stagers,parts):
    return compile_norm_gather_h6144(pullers,stagers,parts)


class NormAllGatherComm:
    """One stream's dense tail, collectively constructed; 32-aligned shard rows.

    The quant source is the FP32 norm at every M. This matches the dense tiled
    norm; SP's row-major norm has a different sum-of-squares order (gate 2).
    Use one communicator per independently ordered stream / DBO microbatch.
    """
    def __init__(self, rank, group, device, max_rows=8192, pullers=PULLERS, stagers=STAGERS, parts=PARTS):
        assert 0 < max_rows <= 8192 and max_rows % 32 == 0
        device = torch.device(device)
        assert device.type == "cuda"
        if device.index is None:
            device = torch.device("cuda",torch.cuda.current_device())
        self.rank,self.device,self.max_rows = rank,device,max_rows
        self.pullers,self.stagers,self.parts = pullers,stagers,parts
        self.scale_off = max_rows*H
        self.stride = self.scale_off+max_rows*(H//32)
        ptr,handle,self.pool = _alloc_pool(2*self.stride,device)
        flag,fh = ops.allocate_shared_buffer_and_handle(FLAG_BYTES)
        peers = _exchange((handle,fh),rank,WORLD,group)
        self.pointers = [ptr if r==rank else _open_pool(h,device) for r,(h,_) in enumerate(peers)]
        self.flags = [flag if r==rank else ops.open_mem_handle(h) for r,(_,h) in enumerate(peers)]
        self.counter = torch.zeros(1,dtype=torch.int32,device=device)
        torch.cuda.synchronize(device)

    def supports(self, rows):
        """Supported padded shard rows; a zero-row collective is a no-op."""
        return 0 <= rows <= self.max_rows and rows % 32 == 0

    def norm_all_gather(self,x,residual,weight,eps=1e-6,*,q_out=None,scale_out=None,res_out=None):
        rows = x.shape[0]
        assert self.supports(rows) and x.shape==(rows,H)
        assert x.dtype==residual.dtype==weight.dtype==torch.bfloat16
        assert residual.shape==x.shape and weight.shape==(H,) and weight.is_contiguous()
        assert x.stride(1)==residual.stride(1)==1
        assert x.device == residual.device == weight.device == self.device
        def output(t,shape,dtype):
            if t is None:
                return torch.empty(shape,dtype=dtype,device=self.device)
            assert t.shape==shape and t.dtype==dtype and t.is_contiguous()
            assert t.device == self.device
            return t
        q = output(q_out,(WORLD*rows,H),torch.float8_e4m3fn)
        s = output(scale_out,(WORLD*rows,H//32),torch.uint8)
        r = output(res_out,(rows,H),torch.bfloat16)
        bits = struct.unpack('<i',struct.pack('<f',eps))[0]
        if rows:
            _run_compiled(_kernel(self.pullers,128 if rows <= 1024 else self.stagers,self.parts),x.data_ptr(),residual.data_ptr(),weight.data_ptr(),r.data_ptr(),q.data_ptr(),s.data_ptr(),*self.pointers,*self.flags,self.counter,self.stride,self.scale_off,rows,x.stride(0),residual.stride(0),bits,self.rank,torch.cuda.current_stream(self.device))
        return q,s,r

    def warm_up(self):
        w = torch.zeros(H,dtype=torch.bfloat16,device=self.device)
        for rows in sorted({32,self.max_rows}):
            x = torch.zeros((rows,H),dtype=torch.bfloat16,device=self.device)
            self.norm_all_gather(x,x,w)
        torch.cuda.synchronize(self.device)
