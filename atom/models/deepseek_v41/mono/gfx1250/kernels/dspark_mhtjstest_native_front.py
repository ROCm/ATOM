"""Static shared-first/router placement, with all-row normalized inputs.

The first 108 units compute shared GU. The other 148 compute router; 44
shared units then complete the remaining router tasks before any route wait.
"""
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_moe_front import *


# CT1-M3-X (from tp4-mono/ct1-m3-ugfront/ct1m3_front.py)
@traced
def load_x8_rows_batched(c, words_mb, codes_mb, xl, xsl, token_of, live):
    """MXFP8 rows ``token_of(j)``, j < ``live`` (<= TILE; traced or int), of
    scratch ``words_mb`` / ``codes_mb`` into LDS rows j of ``xl`` / ``xsl``:
    the first ``live`` lanes wait on their token's X8RDY, then plain 16-byte
    loads (half the bytes of the tagged pairs a poll would read)."""
    tid = c["tid"]
    if tid < TILE:
        c["poll"]([(c["x8rdy"], token_of(fx.min(tid, live - 1)), 1)])
    gpu.barrier()
    # CT1-M3-X: the loads of a batch are issued unconditionally (rows clamped to the last live
    # row) before any LDS store, so they are in flight together; in load_x8_rows every 16-byte
    # load sat in its own branch with its store and the L2 round trips ran back to back.
    # Same bytes, same LDS placement, same rows written.
    BATCH = 10
    for mb, width, dst in ((words_mb, X8_WORDS, xl), (codes_mb, HIDDEN // 32, xsl)):
        quads = width // 4
        n_it = -(-TILE * quads // THREADS)
        for g in range_constexpr(0, n_it, BATCH):
            got = []
            for i in range_constexpr(g, min(g + BATCH, n_it)):
                q = tid + THREADS * i
                row = q // quads
                legal = fx.min(row, live - 1)
                v = fx.Vector(
                    bo.buffer_load(
                        rsrc(c[mb].value),
                        token_of(legal) * width + q % quads * 4,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                got.append((q, row, v))
            for q, row, v in got:
                if (q < TILE * quads) & (row < live):
                    fx.ptr_store(v, dst + (row * lds_row(width) + q % quads * 4))
    gpu.barrier()


@traced
def shared_unit(c, unit, multi):
    """Shared unit ``unit`` (task, token tile): its gate weights in flight
    before x's wait, the tile's rows, ``stage_shared``."""
    s, shared_tasks, a = c["S"], c["d"].shared_tasks, c["args"]
    gops = gemv_fp8_loads(
        c, a["sgu"], a["sgu_s"], HIDDEN, plus(c["sh_gate0"], unit % shared_tasks)
    )
    if const_expr(not multi):
        load_x8_rows_batched(c, "x8", "x8s", c["x8l"], c["x8sl"], lambda j: j, s)
        stage_shared(c, unit, gops, 0, s)
    else:
        t0 = unit // shared_tasks * TILE
        n = fx.min(fx.Int32(TILE), s - t0)
        load_x8_rows_batched(c, "x8", "x8s", c["x8l"], c["x8sl"], lambda j: t0 + j, n)
        stage_shared(c, unit % shared_tasks, gops, t0, n)


@traced
def router_unit(c, task):
    wvs = router_weights(c, task)
    groups, per = c["rs"].router_groups
    per_group = c["rs"].experts // ROWS * ROUTER_PARTS
    t0 = fx.Int32(0) if groups == 1 else (task // per_group) * per
    mine = [plus(t0, t) for t in range(per)]
    await_x(c, mine)
    stage_router(c, task, wvs)


@traced
def run_moe_front(root, key, bid):
    assert key.tokens == 48 and not key.tok_par
    c = front_context(root, key, "shared")
    c["xl"], c["xsl"] = c["x8l"], c["x8sl"]
    units = c["rs"].shared_units
    router_tasks = c["rs"].router_tasks
    # CT1-M3D: placement derived from the route shape (target 384x6: 108 shared,
    # 192 router, 64 shared-first; draft 128x3: every shared unit runs first).
    assert units <= BLOCKS and router_tasks <= BLOCKS
    independent_shared = BLOCKS - router_tasks
    assert key.experts != 384 or (units == 108 and router_tasks == 192 and independent_shared == 64)
    unit = first_task(bid, c["rs"].shared0)
    enter_stage("k2b.shared_first")
    if unit < min(independent_shared, units):
        shared_unit(c, unit, True)
    cr = front_context(root, key, "router")
    enter_stage("k2b.router")
    if unit >= independent_shared:
        router_unit(cr, unit - independent_shared)
    if (unit >= independent_shared) & (unit < units):
        shared_unit(c, unit, True)
    c = front_context(root, key, "route")
    enter_stage("k2b.route")
    for t in range(first_task(bid, c["rs"].route0), c["S"], BLOCKS):
        stage_route(c, t)
