"""QSA paged MQA logits, FlyDSL layout-algebra version.

Run inside the FlyDSL container:
    python test_qsa_logits_kernel_2.py            # all cases
    python test_qsa_logits_kernel_2.py --case 0   # one case (handy when adding prints)
"""

import argparse

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.protocol import dsl_size_of as sizeof

BLOCK_SIZE = 256
BM, BN = 64, 64
Q_HEADS = 4
K_HEADS = 1
HDIMS = 128
KV_CACHE_BLOCK_SIZE = 64
COMPRESS_RATIO = 4
PAGE_SIZE = KV_CACHE_BLOCK_SIZE // COMPRESS_RATIO


@flyc.kernel(known_block_size=(BLOCK_SIZE, 1, 1))
def qsa_logits_fly_kernel(
    q: fx.Tensor,  # [rows, 4, 128] bf16, contiguous
    compressed_k_cache: fx.Tensor,  # [pages, 16, 1, 128] bf16, contiguous
    page_table: fx.Tensor,  # [reqs, width] int32
    token_to_request: fx.Tensor,  # [rows] int32
    query_positions: fx.Tensor,  # [rows] int32
    context_lens: fx.Tensor,  # [reqs] int32
    logits: fx.Tensor,  # [rows, num_columns] fp32, out
    visible_out: fx.Tensor,  # [rows] int32, out
    row_starts: fx.Tensor,  # [rows] int32, out
    rows: fx.Int32,
    num_columns: fx.Int32,
    num_pages: fx.Int32,
    num_requests: fx.Int32,
    width: fx.Int32,
    inv_divisor: fx.Constexpr[float],
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    wave_id = tid // 64
    lane_id = tid % 64
    block_layout = fx.make_layout(((rows + BM - 1)//BM, (num_columns + BN - 1)//BN),
                                ((num_columns + BN - 1)//BN, 1))
    bidm,bidn = fx.idx2crd(bid,block_layout).unpack()

    def get_buffer_ptr(tensor,num_elem):
        return fx.rocdl.make_buffer_ptr(
            fx.recast_iter(fx.Int8,fx.get_iter(tensor)),
            num_records_bytes=num_elem*sizeof(tensor.dtype)
        )

    def gstore(g_byte_ptr,byte_offset,vec,N,dtype):
        nbytes = N * sizeof(dtype)
        dst = fx.make_view(g_byte_ptr + byte_offset,fx.make_layout(nbytes,1))
        reg = fx.make_rmem_tensor(fx.make_layout(N,1),dtype)
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(nbytes*8),fx.Int8)
        fx.memref_store_vec(vec,reg)
        fx.copy_atom_call(copy_atom,reg,dst)

    def _make_view(memref,shape,stride):
        return fx.make_view(fx.get_iter(memref),fx.make_layout(shape,stride))

    @fx.struct
    class LDS:
        k_tile:fx.Array[fx.BFloat16,BM*HDIMS*K_HEADS,16]
        row_valid_tag:fx.Array[fx.Int32,BM,16]
        row_vis: fx.Array[fx.Int32, BM, 16]
    lds = fx.SharedAllocator().allocate(LDS).peek()
    k_tile = lds.k_tile.view(fx.make_layout((BM,HDIMS),(HDIMS,1)))
    row_valid_tag = lds.row_valid_tag.view(fx.make_layout(BM,1))
    row_vis = lds.row_vis.view(fx.make_layout(BM, 1))
    logits_ptr = get_buffer_ptr(logits, rows * num_columns)
    visible_out_ptr = get_buffer_ptr(visible_out, rows)
    row_starts_ptr = get_buffer_ptr(row_starts, rows)

    # 先算  visible out
    token_to_request = _make_view(token_to_request,rows,1)
    query_positions = _make_view(query_positions,rows,1)
    context_lens = _make_view(context_lens, num_requests, 1)
    row0 = BM * bidm
    row_thread = row0 + tid
    if tid < BM:
        row_valid = row_thread < rows # 越界的在最后不要写回
        row_thread = row_valid.select(row_thread,rows - 1)
        req = token_to_request[row_thread]
        req_valid = (req >= 0) & (req < num_requests)
        ctx_len = req_valid.select(context_lens[req_valid.select(req,0)],0)
        pos = query_positions[row_thread]
        # 如果token存在，这个表存储token对应的req
        # 如果token不存在，这个表存储的是-999
        # 如果token存在但是req无效，则数值不满足 (req >= 0) & (req < num_requests)这个条件
        row_valid_tag[tid] = row_valid.select(req,-999)
        _1 = (pos + 1) // COMPRESS_RATIO
        _2 = ctx_len // COMPRESS_RATIO
        _3 = (_1 < _2).select(_1,_2)
        vis_out = (0 > _3).select(0,_3)
        row_vis[tid] = vis_out
        gstore(visible_out_ptr,
            byte_offset = (row_valid & (bidn == 0)).select(row_thread * sizeof(fx.Int32),0x7FFFFFFF),
            vec = fx.Vector.from_elements([vis_out], dtype=fx.Int32),
            N = 1,
            dtype=fx.Int32
        )
        gstore(row_starts_ptr,
            byte_offset = (row_valid & (bidn == 0)).select(row_thread * sizeof(fx.Int32),0x7FFFFFFF),
            vec = fx.Vector.from_elements([0], dtype=fx.Int32),
            N = 1,
            dtype=fx.Int32
        )

    fx.gpu.barrier()
    # 到这里visible out已经写完，row start 写完
    # 当前block负责的64条q的request范围，以及这64行里最大的可见列数
    req_low = fx.Int32(0x7FFFFFFF)
    req_high = fx.Int32(-1)
    vis_max = fx.Int32(0)
    for tag_id in fx.range_constexpr(BM):
        req = row_valid_tag[tag_id]
        tag_ok = (req >= 0) & (req < num_requests)
        req_low = (tag_ok & (req < req_low)).select(req,req_low)
        req_high = (tag_ok & (req > req_high)).select(req,req_high)
        row_vis_i = row_vis[tag_id]
        vis_max = ((req != -999) & (row_vis_i > vis_max)).select(row_vis_i,vis_max)
    no_valid = req_high < 0
    req_low = no_valid.select(0,req_low)
    req_high = no_valid.select(0,req_high)

    # 当前block负责的column范围
    col0 = bidn * 64
    page0 = col0 // PAGE_SIZE
    # 提前退出：这64行都看不到 col0 及之后的列，top-k 只读每行 [0, visible)，
    # 所以整块既不用算也不用写。让请求循环 range(req_low, req_high+1) 变成空循环
    skip_block = col0 >= vis_max
    req_high = skip_block.select(req_low - 1,req_high)

    # 一个block负责64条q，一个wave负责16个q，每条q 4个head
    q = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(q,
                max_size=False,
                num_records_bytes=fx.Int64(rows) * fx.Int64(4*128) * fx.Int64(2))),fx.make_layout((rows,4,128),(4*128,128,1)))
    logits = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(logits,
                max_size=False,
                num_records_bytes=fx.Int64(rows) * fx.Int64(num_columns) * fx.Int64(4))),fx.make_layout((rows,num_columns),(num_columns,1)))
    page_table = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(page_table,
                max_size = False,
                num_records_bytes=fx.Int64(num_requests*width*4))),fx.make_layout((num_requests,width),(width,1)))
    kc = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(compressed_k_cache,
                max_size = False,
                num_records_bytes=fx.Int64(num_pages*16*1*128*2))),fx.make_layout((num_pages,16,1,128),(16*128*1,128*1,128,1)))

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16,16,16,fx.BFloat16))
    MFMA_K_STEP = 16 # 这里指的是做MFMA的那个K维度的步长
    MFMA_K_LOOPS = 128 // MFMA_K_STEP
    q_head0123 = [q[None,head,None] for head in fx.range_constexpr(4)]
    # tiler 要写全两维：(64,128) -> (64,128,ceil(rows/64),1)
    q_head0123_block = [fx.flat_divide(qi,(64,128))[None,None,bidm,0] for qi in q_head0123] # [64,128]
    q_head0123_wave = [fx.flat_divide(qi,(16,128))[None,None,wave_id,0] for qi in q_head0123_block] # 4 个 [16,128]
    # K 维重排：第 s 步里 lane 组 g 负责 dims 32g+4s..32g+4s+3，于是每个 lane 8 步合起来读连续的 32 个元素
    # (16,(4,4),8):(512,(1,32),4) —— 行 / (步内4个元素, lane组g) / 步 s
    q_head0123_wave = [fx.make_view(fx.get_iter(qi),fx.make_layout((16,(4,4),8),(4*128,(1,32),4))) for qi in q_head0123_wave]

    tiled_mma = fx.make_tiled_mma(mma_atom,fx.make_layout((1,1,1),(0,1,2)))
    thr_mma = tiled_mma.thr_slice(lane_id)
    q_frag = [[thr_mma.make_fragment_A(q_head0123_wave[h][None,None,0]) for s in fx.range_constexpr(MFMA_K_LOOPS)] for h in fx.range_constexpr(4)] # q_frag[head][k步]
    # global q -> q frag。重排后每个线程 8 步共读连续 32 个 bf16，编译器会合并成 4 条 buffer_load_dwordx4
    copy_global_to_frag = fx.make_copy_atom(fx.rocdl.BufferCopy32b(),fx.BFloat16)
    copy_q = fx.make_tiled_copy_A(copy_global_to_frag,tiled_mma).get_slice(lane_id)
    q_global_tile_head0123 = [copy_q.partition_S(q_head0123_wave[i]) for i in fx.range_constexpr(4)]

    logits = fx.flat_divide(logits,(64,64))[None,None,bidm,bidn]
    logits = fx.flat_divide(logits,(16,64))[None,None,wave_id,0] # [64,64]->[16,64,4,1]，选中这个wave负责的那一片logits
    # 再切成4块，每个wave都要计算4块
    logits = fx.flat_divide(logits,(16,16))[None,None,0,None] #[16,64]->[16,16,1,4]->[16,16,4]
    logits_tile0123 = [logits[None,None,i] for i in fx.range_constexpr(4)] # 每一块都是[16,16]

    # k lds 切成四块，每次加载一块
    k_tile = fx.flat_divide(k_tile,(16,128))[None,None,None,0] # [64,128]->[16,128,4,1]->[16,128,4]
    # 每页 [16,128] 做和 Q 相同的 K 维重排（LDS 行跨度 HDIMS）: (16,(4,4),8) = [N_TILE_SIZE, K_TILE_SIZE, inner_k_loops]
    k_tile_pages0123 = [fx.make_view(fx.get_iter(k_tile[None,None,pageid]),fx.make_layout((16,(4,4),8),(HDIMS,(1,32),4))) for pageid in fx.range_constexpr(4)]
    # 一次只处理一个 page：8 个 fragment 对应这个 page 的 8 个 k 步
    k_frag = [thr_mma.make_fragment_B(k_tile_pages0123[0][None,None,0]) for kk in fx.range_constexpr(MFMA_K_LOOPS)]

    copy_k_lds_frag = fx.make_copy_atom(fx.UniversalCopy32b(),fx.BFloat16)
    copy_k = fx.make_tiled_copy_B(copy_k_lds_frag,tiled_mma).get_slice(lane_id)
    k_lds_tile0123 = [copy_k.partition_S(k_tile_pages0123[i]) for i in fx.range_constexpr(4)]

    # 运行时 for 循环里出现 x.method(...) 时，循环外定义的 x 会被当成循环变量传递，
    # 而 ThrMma / ThrCopy 不能作为循环变量，所以这些调用必须放在循环外
    # 只用 1 个累加器（放在 list 里：循环里调用 acc[0].fill 不会被当成循环变量）
    acc = [thr_mma.make_fragment_C(logits_tile0123[0])]
    # Q 只和行有关，在请求循环外一次性读进寄存器
    for h in fx.range_constexpr(4):
        for s in fx.range_constexpr(MFMA_K_LOOPS):
            fx.copy(copy_global_to_frag,src = q_global_tile_head0123[h][None,None,None,s],dst = copy_q.retile(q_frag[h][s]))
    k_frag_retiled = [copy_k.retile(k_frag[kk]) for kk in fx.range_constexpr(MFMA_K_LOOPS)]

    # collective load k to lds
    for reqid in range(req_low,req_high + 1):
        reqid = fx.Int32(reqid)
        page_ok = []
        for page_offset in fx.range_constexpr(4):
            logical_page = page0 + page_offset
            logical_page_valid = (logical_page < width)
            logical_page = logical_page_valid.select(logical_page,0)
            physical_page = page_table[reqid,logical_page]
            physical_page_valid = ((physical_page >= 0) & (physical_page < num_pages))
            physical_page = physical_page_valid.select(physical_page,0)
            page_ok.append(physical_page_valid & logical_page_valid)
            k = kc[physical_page,None,0,None] # [16,128]
            # k -> lds k tile, k_tile[:,:,page_offset]=k
            copy_atom_global_lds = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(),fx.BFloat16) # 2个bf16
            # 一个线程每次复制两个bf16，256个线程每次复制512个bf16
            global_k = fx.make_view(fx.get_iter(k),fx.make_layout(16*128,1))
            global_k = fx.logical_divide(global_k,fx.make_layout((2,256),(1,2))) # [[2,256],4]

            lds_k = k_tile[None,None,page_offset] #[16,128]
            lds_k = fx.make_view(fx.get_iter(lds_k),fx.make_layout(16*128,1))
            lds_k = fx.logical_divide(lds_k,fx.make_layout((2,256),(1,2))) # [[2,256],4]
            for rest_i in fx.range_constexpr(4):
                fx.copy_atom_call(copy_atom_global_lds,global_k[None,rest_i][None,tid],lds_k[None,rest_i][None,tid])
        # 到这里，一共4个page的k已经搬运完毕
        # MFMA，每个wave一条q frag，要与对应全部四个k tile的MFMA运算
        fx.gpu.barrier()
        NEG_INF = fx.Float32(float("-inf"))
        DROP = fx.Int32(0x7FFFFFFF)
        r_grp = lane_id // 16          # C fragment：行 = 4*r_grp + jr
        c_in = lane_id % 16            #             列 = c_in
        # 本 lane 负责写的 4 行：行号、可见列数、是否有效、这一轮是否由它写
        own = []
        for jr in fx.range_constexpr(4):
            row_local = wave_id * 16 + r_grp * 4 + jr        # block 内第几行（0..63）
            tag = row_valid_tag[row_local]
            valid = (tag >= 0) & (tag < num_requests)
            invalid = (tag != -999) & ((tag < 0) | (tag >= num_requests))
            owned = (valid & (tag == reqid)) | (invalid & (reqid == req_low))
            own.append((row0 + row_local, row_vis[row_local], valid, owned))

        # page 外层、head 中层、k 步内层：同一时刻只需要 1 个累加器，每个 page 算完就写回
        for pg in fx.range_constexpr(4):
            for kk in fx.range_constexpr(MFMA_K_LOOPS):
                fx.copy(copy_k_lds_frag,src = k_lds_tile0123[pg][None,None,None,kk],dst = k_frag_retiled[kk])
            score = None
            for hd in fx.range_constexpr(4):
                acc[0].fill(0.0)
                for kk in fx.range_constexpr(MFMA_K_LOOPS):
                    fx.gemm(thr_mma, acc[0], q_frag[hd][kk], k_frag[kk], acc[0])
                v = acc[0].load()
                v = v.maximumf(fx.Vector.zeros_like(v))   # 每个 head 先 ReLU
                score = v if score is None else score + v  # 再按 head 累加
            score = score * inv_divisor

            col = col0 + pg * 16 + c_in
            for jr in fx.range_constexpr(4):
                row, vis, valid, owned = own[jr]
                keep = valid & (col < vis) & page_ok[pg]
                val = keep.select(fx.Float32(score[jr]), NEG_INF)
                off = (owned & (col < num_columns)).select((row * num_columns + col) * 4, DROP)
                gstore(logits_ptr, off, fx.Vector.from_elements([val], dtype=fx.Float32), 1, fx.Float32)
        fx.gpu.barrier()      # 所有 wave 用完这一轮的 K，下一轮才能覆盖 LDS


@flyc.jit
def qsa_logits_fly(
    q: fx.Tensor,
    compressed_k_cache: fx.Tensor,
    page_table: fx.Tensor,
    token_to_request: fx.Tensor,
    query_positions: fx.Tensor,
    context_lens: fx.Tensor,
    logits: fx.Tensor,
    visible_out: fx.Tensor,
    row_starts: fx.Tensor,
    rows: fx.Int32,
    num_columns: fx.Int32,
    num_pages: fx.Int32,
    num_requests: fx.Int32,
    width: fx.Int32,
    grid: fx.Int32,
    inv_divisor: fx.Constexpr[float],
    stream: fx.Stream,
):
    qsa_logits_fly_kernel(
        q, compressed_k_cache, page_table, token_to_request, query_positions, context_lens,
        logits, visible_out, row_starts, rows, num_columns, num_pages, num_requests, width, inv_divisor,
    ).launch(grid=(grid, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)


# ---------------------------------------------------------------- host side


def run_kernel(case):
    q, kc, pt = case["q"], case["kc"], case["page_table"]
    rows, cols = q.shape[0], case["num_columns"]
    # NaN / -7 make entries the kernel did not write easy to spot (skipped column blocks stay NaN)
    logits = torch.full((rows, cols), float("nan"), dtype=torch.float32, device=q.device)
    visible = torch.full((rows,), -7, dtype=torch.int32, device=q.device)
    row_starts = torch.full((rows,), -7, dtype=torch.int32, device=q.device)
    grid = (-(-rows // BM)) * (-(-cols // BN))
    qsa_logits_fly(
        q, kc, pt, case["token_to_request"], case["query_positions"], case["context_lens"],
        logits, visible, row_starts,
        rows, cols, kc.shape[0], pt.shape[0], pt.shape[1], grid, case["inv_divisor"],
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    return logits, visible, row_starts


def reference(case):
    q, kc, pt = case["q"].float(), case["kc"].float(), case["page_table"].long()
    t2r, pos, ctx = case["token_to_request"].long(), case["query_positions"].long(), case["context_lens"].long()
    num_requests, width = pt.shape
    num_pages = kc.shape[0]
    cols = case["num_columns"]

    req_ok = (t2r >= 0) & (t2r < num_requests)
    req = t2r.clamp(0, num_requests - 1)
    ctx_len = torch.where(req_ok, ctx[req], 0)
    visible = torch.minimum((pos + 1) // COMPRESS_RATIO, ctx_len // COMPRESS_RATIO).clamp_min(0)

    col = torch.arange(cols, device=q.device)
    logical_page, page_offset = col // PAGE_SIZE, col % PAGE_SIZE
    phys = pt[req][:, logical_page.clamp(max=width - 1)]  # [rows, cols]
    page_ok = (logical_page < width)[None, :] & (phys >= 0) & (phys < num_pages)
    k = kc[phys.clamp(0, num_pages - 1), page_offset[None, :], 0, :]  # [rows, cols, 128]
    score = torch.einsum("rhd,rcd->rhc", q, k).relu().sum(1) * case["inv_divisor"]  # ReLU per head, then sum
    valid = req_ok[:, None] & (col[None, :] < visible[:, None]) & page_ok
    return torch.where(valid, score, float("-inf")), visible.int()


def make_case(rows, num_columns, num_requests=1, *, interleaved=False, positions="ramp",
              bad_request_rows=0, hole_fraction=0.0, context_len=None, seed=0, label=""):
    g = torch.Generator().manual_seed(seed)
    width = -(-num_columns // PAGE_SIZE)
    pages = width * num_requests
    ctx = num_columns * COMPRESS_RATIO if context_len is None else context_len
    page_table = torch.randperm(pages, generator=g)[: width * num_requests].reshape(num_requests, width)
    if hole_fraction:
        page_table[torch.rand(page_table.shape, generator=g) < hole_fraction] = -1
    if interleaved:
        t2r = torch.arange(rows) % num_requests
    else:
        t2r = (torch.arange(rows) // -(-rows // num_requests)).clamp(max=num_requests - 1)
    t2r[:bad_request_rows] = -1
    if positions == "ramp":
        pos = torch.linspace(0, ctx - 1, rows).to(torch.int32)
    elif positions == "saturated":
        pos = torch.full((rows,), ctx - 1)
    else:
        pos = torch.zeros(rows)
    return dict(
        label=label,
        q=torch.randn(rows, Q_HEADS, HDIMS, generator=g).to(torch.bfloat16).cuda(),
        kc=torch.randn(pages, PAGE_SIZE, 1, HDIMS, generator=g).to(torch.bfloat16).cuda(),
        page_table=page_table.to(torch.int32).cuda(),
        token_to_request=t2r.to(torch.int32).cuda(),
        query_positions=pos.to(torch.int32).cuda(),
        context_lens=torch.full((num_requests,), ctx, dtype=torch.int32).cuda(),
        num_columns=num_columns,
        inv_divisor=1.0 / HDIMS**0.5,
    )


CASES = [
    lambda: make_case(8, 64, positions="saturated", label="tiny, all visible"),
    lambda: make_case(100, 200, num_requests=3, label="rows/cols not multiples of 64, 3 requests"),
    lambda: make_case(16, 128, num_requests=4, interleaved=True, label="4 interleaved requests"),
    lambda: make_case(16, 128, bad_request_rows=3, hole_fraction=0.25, label="invalid requests + unmapped pages"),
    lambda: make_case(16, 128, context_len=200, positions="saturated", label="context_len caps visibility"),
    lambda: make_case(8, 64, positions="zero", label="nothing visible"),
    lambda: make_case(256, 4096, num_requests=2, label="medium, 2 requests"),
    lambda: make_case(2048, 512, label="causal prefill from position 0"),
]


def check(case):
    """Top-k reads each row only over [0, visible), so that is all the kernel must get right.

    Column blocks no row of the block can see are skipped and stay NaN (the fill value).
    """
    logits, visible, row_starts = run_kernel(case)
    ref_logits, ref_visible = reference(case)
    rows, cols = logits.shape
    horizon = torch.arange(cols, device=logits.device)[None, :] < ref_visible[:, None]
    problems = []
    if not torch.equal(visible, ref_visible):
        problems.append(f"visible differs in {int((visible != ref_visible).sum())} rows")
    if not bool((row_starts == 0).all()):
        problems.append("row_starts not zero")
    if torch.isnan(logits[horizon]).any():
        problems.append(f"{int(torch.isnan(logits[horizon]).sum())} logits inside the horizon never written")
    got_inf, ref_inf = torch.isneginf(logits) & horizon, torch.isneginf(ref_logits) & horizon
    if not torch.equal(got_inf, ref_inf):
        problems.append(f"-inf mask differs in {int((got_inf != ref_inf).sum())} entries")
    finite = horizon & ~ref_inf & ~torch.isnan(logits)
    max_abs = float((logits[finite] - ref_logits[finite]).abs().max()) if finite.any() else 0.0
    if finite.any() and not torch.allclose(logits[finite], ref_logits[finite], atol=2e-3, rtol=2e-3):
        problems.append(f"values differ, max_abs={max_abs:.3e}")
    skipped = float(torch.isnan(logits).float().mean())
    status = "PASS" if not problems else "FAIL: " + "; ".join(problems)
    print(f"[{case['label']}] rows={rows} cols={cols} max_abs={max_abs:.2e} skipped={skipped:.0%}  {status}")
    return not problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=None, help=f"run only this case (0..{len(CASES) - 1})")
    args = ap.parse_args()
    picked = CASES if args.case is None else [CASES[args.case]]
    ok = sum(check(make()) for make in picked)
    print(f"\n{ok}/{len(picked)} cases passed")
