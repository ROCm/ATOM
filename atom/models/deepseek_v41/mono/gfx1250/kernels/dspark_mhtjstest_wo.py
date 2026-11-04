"""M16 N64 WO-B, native fragments, full-row TP1 publication."""
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_wo import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.reference_projection import _weight_step

@traced
def matrix(c,block,n):
    lane,wave=c['lane'],c['wave'];inner=c['d'].wob_k
    split=2;per=inner//128//split
    row=fx.min(lane%16,n-1);half=lane//16
    xl=c['xl']+row*lds_row(inner//4)
    xsl=c['xsl']+row*lds_row(inner//32)
    rg=block*4+wave//split
    rw,rs=rsrc(c['args']['wob']),rsrc(c['args']['wob_s'])
    # Complete K accumulators, two slots, native matrix fragments.
    ops=[_weight_step(c,rw,rs,inner,rg,(wave%split)*per+i) for i in range(2)]
    atom=fx.make_mma_atom(fx.rocdl.WMMAScale(16,16,128,fx.Float8E4M3FN,fx.Float8E4M3FN,fx.Float32))
    acc=fx.make_rmem_tensor(8,fx.Float32);acc.store(fx.full(8,0.,fx.Float32))
    a,b=fx.make_rmem_tensor(16,fx.Int32),fx.make_rmem_tensor(16,fx.Int32)
    for i in range_constexpr(per):
        weight,sw=ops[i%2]
        if const_expr(i+2<per):ops[i%2]=_weight_step(c,rw,rs,inner,rg,(wave%split)*per+i+2)
        st=(wave%split)*per+i
        chunks=[fx.Vector(fx.ptr_load(xl+st*32+half*4+j*8,result_type=fx.Vector.make_type(4,fx.Int32))) for j in range(4)]
        xv=fx.Vector.from_elements([(lane%16<n).select(chunks[j][q],fx.Int32(0)) for j in range(4) for q in range(4)],fx.Int32)
        codes=[fx.ptr_load(xsl+st*4+j) for j in range(4)]
        sx=codes[0]|(codes[1]<<8)|(codes[2]<<16)|(codes[3]<<24)
        sx=(lane%16<n).select(sx,fx.Int32(0x7f7f7f7f))
        a.store(weight);b.store(xv)
        fx.gemm(atom,acc,a,b,acc,scale_a=sw,scale_b=sx)
    if lane%16<n:
        v=acc.load()
        for j in range_constexpr(8):fx.ptr_store(v[j],c['red']+(wave*16+lane%16)*16+half*8+j)
    gpu.barrier()

@traced
def push(c,block,t0,n):
    tid=c['tid'];d=c['d']
    for begin in range_constexpr(0,16*64//2,THREADS):
        at=tid+begin;t=at//32;pair=at%32
        if t<n:
            group=pair//8;cc=2*(pair%8)
            vals=[]
            for j in range_constexpr(2):
                x=fx.ptr_load(c['red']+((2*group)*16+t)*16+cc+j)
                x=x+fx.ptr_load(c['red']+((2*group+1)*16+t)*16+cc+j)
                vals.append(x)
            tg=t0+t;col=block*64+pair*2
            for p in range_constexpr(4):
                dst=preg(c['peer_addr'](p),c['peer']['attn'],'attn')
                address=attn_region_pair(c['rank'],tg,c['S'],col)
                c['put_bf'](dst,2*address,vals)
    gpu.barrier()

@traced
def run_wo(root,key,bid):
    c=wo_context(root,'woa');s=key.tokens;d=c['d']
    enter_stage('k2a.wo_a')
    for unit in range(first_task(bid,WOA0),d.woa_tasks*3,BLOCKS):
        task=unit%d.woa_tasks;t0=unit//d.woa_tasks*16
        ops=woa_loads(c,task)
        woa_tile(c,task,ops,t0,fx.Int32(16))
    c=wo_context(root,'wob')
    enter_stage('k2a.wo_b_n64')
    unit=first_task(bid,WOA0+d.woa_tasks*3)
    if unit<80*3:
        block=unit//3;t0=unit%3*16
        load_wob_rows(c,0,t0,fx.Int32(16))
        matrix(c,block,fx.Int32(16))
        push(c,block,t0,fx.Int32(16))
