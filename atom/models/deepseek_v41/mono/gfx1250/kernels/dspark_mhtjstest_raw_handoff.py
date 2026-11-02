"""Independent contiguous payload and tagged group readiness for WO inputs.

Original IRQ/WOA group32 values stay unchanged. A single flag is published
only after every producer lane's payload/scale stores retire and rendezvous.
The full K2 step fence protects all raw storage from a subsequent launch.
"""
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_wo import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_state import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import ready
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import tensor
from atom.mono.device.ops import CM_DEV
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import T


def raw_address(c,offset):return late_pointer(c,220)+fx.Int64(offset)


@traced
def put_xo_word(c,index,value):
    bo.buffer_store(value,rsrc(raw_address(c,RAW_XO_OFFSET)),index,cache_modifier=CM_DEV)


@traced
def put_xo_scale(c,index,value):
    bo.buffer_store(value,rsrc(raw_address(c,RAW_XOS_OFFSET)),index,cache_modifier=CM_DEV)


@traced
def put_xb_word(c,index,value):
    bo.buffer_store(value,rsrc(raw_address(c,RAW_XB_OFFSET)),index,cache_modifier=CM_DEV)


@traced
def put_xb_scale(c,index,value):
    bo.buffer_store(value,rsrc(raw_address(c,RAW_XBS_OFFSET)),index,cache_modifier=CM_DEV)


@traced
def finish_irq_task(c,task):
    # All four disjoint group stores must retire on every wave before one
    # lane publishes readiness for the complete producer task.
    drain()
    if c['tid']==0:
        c['put'](sreg(raw_address(c,0),RAW_XO_READY_OFFSET,'xo'),task,1)


@traced
def finish_xb(c,task,t0):
    drain()
    if c['tid']==0:
        c['put'](sreg(raw_address(c,0),RAW_XB_READY_OFFSET,'x8b'),
                 (t0//16)*c['d'].woa_tasks+task,1)
    gpu.barrier()


@traced
def make_input_job(c,kind,t0,group=0):
    d=c['d']
    if const_expr(kind=='woa'):
        inner=d.group_k;stride=d.xo_words;off=group*inner//4;offset=RAW_XO_OFFSET
    else:
        inner=d.wob_k;stride=d.yb_words;off=0;offset=RAW_XB_OFFSET
    words=inner//4
    ptr=fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type,fx.AddressSpace.Global,16),
                   raw_address(c,offset)+(fx.Int64(t0)*stride+fx.Int64(off))*4)
    # TDM cannot encode a 1024-dword padding interval. Describe K128 slabs
    # as an outer axis so each 32-dword row uses native encodable padding.
    # LDS is [K128][M16][32 payload words + 4 pad words].
    src=tensor(ptr,(inner//128,16,32),(32,stride,1))
    atom=fx.rocdl.make_tdm_atom(src,[inner//128,16,32],strides=[32,stride,None],
                              num_warps=1,pad_interval=32,pad_amount=4,
                              cache_modifier=CM_DEV)
    return atom,src


@traced
def load_input(c,kind,t0,group,job):
    d=c['d'];tid=c['tid']
    if const_expr(kind=='woa'):
        inner=d.group_k;stride=d.heads*HEAD_DIM//32;off=group*inner//32
        offset=RAW_XOS_OFFSET
        flag=sreg(raw_address(c,0),RAW_XO_READY_OFFSET,'xo')
        # At TP4 the single 16-head IRQ producer tile contains both 8-head
        # WOA groups. Every dc is needed by either group, so no new wait edge.
        assert d.head_tiles.count==1 and d.groups==2
        assert c['S']==48
        per=GROUPS//4
        indices=[t0*per+tid]
        participants=16*per
    else:
        inner=d.wob_k;stride=d.o_rows//32;off=0;offset=RAW_XBS_OFFSET
        flag=sreg(raw_address(c,0),RAW_XB_READY_OFFSET,'x8b')
        assert d.woa_tasks==64
        indices=[(t0//16)*d.woa_tasks+tid]
        participants=d.woa_tasks
    if tid<participants:c['poll']([(flag,index,1) for index in indices])
    gpu.barrier()
    if c['wave']==0:
        atom,src=job
        fx.copy(atom,src,tensor(c['xl'],(inner//128,16,32),(16*36,36,1)))
        ready(0)
    groups=inner//32
    for begin in range_constexpr(0,16*groups,THREADS):
        at=tid+begin;row=at//groups;col=at%groups
        value=bo.buffer_load(rsrc(raw_address(c,offset)),
                            (t0+row)*stride+off+col,vec_width=1,dtype=T.i32,cache_modifier=CM_DEV)
        fx.ptr_store(fx.Int32(value),c['xsl']+row*lds_row(groups)+col)
    gpu.barrier()
