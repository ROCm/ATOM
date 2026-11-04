"""Complete WOA/WOB native supply within K2, preserving gfx950 task contracts.

Weights start before tagged input waits. Eight waves compute complete K
partials in the original task geometry; native wave32 operands carry over a
three-stage wave-private TDM ring. Direct native partial reduction retains
WOA group32 requant and WOB BF16 TP1 publication.
"""
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_wo import push
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_wo import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.reference_sync import CM_NT_GFX1250
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import T
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import ready
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import retire_reads
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import tensor
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import BUFFERS
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import SLOT_WORDS
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_raw_handoff import make_input_job
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_raw_handoff import load_input
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_raw_handoff import put_xb_word
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_raw_handoff import put_xb_scale
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_raw_handoff import finish_xb


@traced
def make_job(c,weight,scale,inner,tile,start,steps):
    address=weight+fx.Int64(tile)*inner*16+fx.Int64(start)*2048
    ptr=fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type,fx.AddressSpace.Global,16),address)
    src=tensor(ptr,(1,512),(512,1))
    atom=fx.rocdl.make_tdm_atom(src,[1,512],strides=[512,None],num_warps=1,
                              cache_modifier=CM_NT_GFX1250)
    # Raw N32/K32 weight scale: one packed dword per K128, shared by N16
    # row pair. Its small complete vector is issued before the input wait.
    sw=fx.make_rmem_tensor(steps,fx.Int32)
    sw.store(fx.Vector.from_elements([fx.Int32(bo.buffer_load(rsrc(scale),
        (tile//2)*(inner//128)+start+i,vec_width=1,dtype=T.i32,
        cache_modifier=CM_NT_GFX1250)) for i in range_constexpr(steps)],fx.Int32))
    return atom,src,sw


@traced
def issue(c,job,slot,step):
    atom,src,_=job
    base=c['wo_ring']+c['wave']*(BUFFERS*SLOT_WORDS)+slot*SLOT_WORDS
    fx.copy(atom,src,tensor(base,(1,512),(512,1)),imm_offset=fx.Int64(step)*2048)


@traced
def prime(c,job):
    for i in range_constexpr(BUFFERS):issue(c,job,fx.Int32(i),fx.Int32(i))


def operand():
    return dict(w=fx.make_rmem_tensor(16,fx.Int32),x=fx.make_rmem_tensor(16,fx.Int32),
                sx=fx.make_rmem_tensor(1,fx.Int32))


@traced
def load(c,slot,st,row,inner,out):
    lane=c['lane']
    base=c['wo_ring']+c['wave']*(BUFFERS*SLOT_WORDS)+slot*SLOT_WORDS
    at=(lane//16)*64+(lane%16)*4
    v=[fx.Vector(fx.ptr_load(base+at+i*128,result_type=fx.Vector.make_type(4,fx.Int32)))
       for i in range_constexpr(4)]
    out['w'].store(fx.Vector.from_elements([v[i][j] for i in range(4) for j in range(4)],fx.Int32))
    xp=c['xl']+(st*16+row)*36+(lane//16)*4
    v=[fx.Vector(fx.ptr_load(xp+i*8,result_type=fx.Vector.make_type(4,fx.Int32)))
       for i in range_constexpr(4)]
    out['x'].store(fx.Vector.from_elements([v[i][j] for i in range(4) for j in range(4)],fx.Int32))
    sp=c['xsl']+row*lds_row(inner//32)+st*4
    codes=[fx.ptr_load(sp+i) for i in range_constexpr(4)]
    out['sx'].store(fx.Vector.from_elements([codes[0]|(codes[1]<<8)|(codes[2]<<16)|(codes[3]<<24)],fx.Int32))


@traced
def matrix(c,job,start,steps,inner):
    row=c['lane']%16
    atom=fx.make_mma_atom(fx.rocdl.WMMAScale(16,16,128,
        fx.Float8E4M3FN,fx.Float8E4M3FN,fx.Float32))
    acc=fx.make_rmem_tensor(8,fx.Float32);acc.store(fx.full(8,0.,fx.Float32))
    carry=[operand(),operand()]
    ready(BUFFERS-1)
    load(c,fx.Int32(0),start,row,inner,carry[0])
    for i in range_constexpr(steps):
        cur=carry[0]
        fx.gemm(atom,acc,cur['w'],cur['x'],acc,
                scale_a=job[2].load()[i],scale_b=cur['sx'].load()[0])
        if const_expr(i+1<steps):
            ready(min(BUFFERS-2,steps-i-2))
            load(c,fx.Int32((i+1)%BUFFERS),start+i+1,row,inner,carry[1])
        if const_expr(i+BUFFERS<steps):
            retire_reads()
            issue(c,job,fx.Int32(i%BUFFERS),fx.Int32(i+BUFFERS))
        if const_expr(i+1<steps):
            for name in ('w','x','sx'):cur[name].store(carry[1][name].load())
    ready(0)
    v=acc.load()
    for j in range_constexpr(8):
        fx.ptr_store(v[j],c['red']+(c['wave']*16+row)*16+(c['lane']//16)*8+j)
    gpu.barrier()


@traced
def woa_finish(c,task,t0):
    # Adjacent output N16 tiles each own four K partials. Keep their original
    # summation order and complete BF16/group32 quantization before X8B tags.
    for begin in range_constexpr(0,16*32,THREADS):
        q=c['tid']+begin;row=q%32;token=q//32
        tile,cc=row//16,row%16
        v=fx.Float32(0.)
        for ks in range_constexpr(WOA_SPLIT_K):
            v=v+fx.ptr_load(c['red']+((tile*WOA_SPLIT_K+ks)*16+token)*16+cc)
        y=bf16_round(v)
        amax=fx.max(butterfly(abs(y),(1,2,4,8,16),fx.max),fx.Float32(AMAX_FLOOR))
        code=code_ceil(amax*fx.Float32(1./FP8_MAX))
        value=clamp_fp8(y/pow2(code))
        lane=c['lane']
        word=fp8_pack4(value,*[lane_gather(value.bitcast(fx.Int32),
            fx.min(lane+i,31)).bitcast(fx.Float32) for i in (1,2,3)])
        if row%4==0:put_xb_word(c,(t0+token)*c['d'].yb_words+(task*32+row)//4,word)
        if row==0:put_xb_scale(c,(t0+token)*(c['d'].o_rows//32)+task,code)
    finish_xb(c,task,t0)


def context(root,stage):
    c=wo_context(root,stage)
    c.update(xl=root['wo_x'],xsl=root['wo_s'],wo_ring=root['wo_ring'],red=root['wo_red'])
    return c


@traced
def run_wo(root,key,bid):
    c=context(root,'woa');d=c['d']
    inner=d.group_k;per=inner//128//WOA_SPLIT_K
    for unit in range(first_task(bid,WOA0),d.woa_tasks*3,BLOCKS):
        task=unit%d.woa_tasks;t0=unit//d.woa_tasks*16
        tile=task*2+c['wave']//WOA_SPLIT_K
        start=(c['wave']%WOA_SPLIT_K)*per
        job=make_job(c,c['args']['woa'],c['args']['woa_s'],inner,tile,start,per)
        prime(c,job)
        group=task//(d.woa_tasks//d.groups)
        input_job=make_input_job(c,'woa',t0,group)
        load_input(c,'woa',t0,group,input_job)
        matrix(c,job,start,per,inner)
        woa_finish(c,task,t0)
    c=context(root,'wob');inner=d.wob_k;per=inner//128//2
    unit=first_task(bid,WOA0+d.woa_tasks*3)
    if unit<240:
        block,t0=unit//3,unit%3*16
        tile=block*4+c['wave']//2
        start=c['wave']%2*per
        job=make_job(c,c['args']['wob'],c['args']['wob_s'],inner,tile,start,per)
        prime(c,job)
        input_job=make_input_job(c,'wob',t0)
        load_input(c,'wob',t0,fx.Int32(0),input_job)
        matrix(c,job,start,per,inner)
        push(c,block,t0,fx.Int32(16))
