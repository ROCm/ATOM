# SPDX-License-Identifier: MIT
"""TP4 dynamic42 reference column owners, original legacy weights and two chains.

Shared down first, current-round weight prefetch before complete per-wave UGF
acquire, register MID input, retained B/SB across M16 tiles, local contributions.
"""
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_down import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.reference_sync import publish_plain as publish
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_column_down import bulk_finish

@traced
def stage_down(c, tasks):
    """Rows 16 task .. of the hidden output for each of ``tasks`` (a CTA's one or
    two, run together: a second pass would double the CTA's time): routed
    experts of U (a wave each, round robin), the shared expert, their sum ->
    pushed to every rank (token parallel: to the token's owner)."""
    assert down_parts(c["d"])==1  # TP4; EP/TP2 part branches remain out of scope
    s, wave, tab, d = c["S"], c["wave"], c["tab"], c["d"]
    n = len(tasks)
    nu = c["rs"].n_union(tab, s)
    sl = c["wsl"] + wave * down_wsl_words(d)
    sw = down_scale_words(d)
    # slots of U a round: the loads of a round stay the same
    slots = DOWN_R_MAX // n
    per_round = WAVES * slots
    # the shared expert's down projection first: its input is ready long
    # before the routed results, which a CTA mostly waits on here; after the
    # rounds it sat on the path past the last ug task
    down_shared(c, tasks)
    if const_expr(down_parts(d) > 1):
        stage_down_parts(c, tasks, nu, sl, sw, slots, per_round)
        return
    for rnd in range(0, (nu + per_round - 1) // per_round, 1):
        # the round's slots of U (wave, wave + WAVES, ...), every load of them
        # and of every task in flight before the first wait: the weights and
        # their scale blocks (no ug result: before the round's ug flags), then
        # the slots' picks' intermediate rows, each lane its pick's straight
        # from MID (``mid_operands``). A slot past U loads the last slot again
        # (L2 hits) and contributes nothing
        us = [wave + WAVES * (slots * rnd + i) for i in range(slots)]
        ucs = [fx.min(u, nu - 1) for u in us]
        firsts = [c["rs"].slot_first(tab, u) for u in ucs]
        svs, avs = down_weights(c, tasks, rnd, slots, nu)
        # this round's slots' ug tasks done, not every slot of the wave: the
        # round starts while later slots' ug run
        c["poll"]([(c["ugf"], f, 1) for f in down_ug_flags(c, ucs)])
        counts = [c["rs"].slot_first(tab, u + 1) - f for u, f in zip(ucs, firsts)]
        for q in range_constexpr(len(svs)):
            scale_block_store(c, svs[q], sl + q * sw)
        for i in range_constexpr(slots):
            # Same decision in all 32 lanes; no CTA barrier in the guarded body.
            if us[i] < nu:
                # Round B/SB and complete UGF acquire precede this slot-local MID.
                mid = mid_operands(c, firsts[i], counts[i])
                for j in range_constexpr(n):
                    down_mfma(
                        c, j, tasks[j], sl + (j * slots + i) * sw, mid, firsts[i],
                        counts[i], avs[j * slots + i], us[i] < nu,
                    )  # fmt: skip
                # a slot of more than 16 picks: its next pick tile's rows
                if const_expr(s > TILE):
                    for j0 in range_constexpr(TILE, s, TILE):
                        if j0 < counts[i]:
                            more = mid_operands(c, firsts[i] + j0, counts[i] - j0)
                            for j in range_constexpr(n):
                                down_mfma(
                                    c, j, tasks[j], sl + (j * slots + i) * sw, more,
                                    firsts[i] + j0, counts[i] - j0, avs[j * slots + i],
                                    us[i] < nu,
                                )  # fmt: skip
    gpu.barrier()
    bulk_finish(c, tasks)

@traced
def run_down(root,key,bid):
    assert key.tokens in (6,12,18,24,30,36,42) and not key.tok_par and key.ep == 1
    c=down_context(root,key)
    s=c['S']
    tid=c['tid']
    multi=len(token_tiles(s))>1
    ug_tasks=c['rs'].max_u*c['rs'].ug_parts
    assert not down_by_unit(c)
    # the route table is in LDS already past one token tile (every CTA loads it
    # for the ug queue) or when every CTA ran an ug iteration (more ug task
    # slots than CTAs), whose first loads it; else load it
    if const_expr(not multi and ug_tasks <= BLOCKS):
        load_route(c, c["tab"])
    # every CTA its tile; the first DOWN_TASKS - BLOCKS to get here an extra
    # one too: the CTAs whose ug ended first (by bid, those ending last held
    # two tiles and ended the stage)
    down_task = first_task(bid, DOWN0)
    rank = take_down_rank(c)
    if const_expr(c['down_debug']):
        if c['tid']==0:fx.ptr_store(rank.to(fx.Float32),observation(c,(c['rs'].picks+c['S'])*HIDDEN)+bid)
    has_second = rank < DOWN_TASKS - BLOCKS
    by_unit = down_by_unit(c)
    if const_expr(by_unit):
        # the down stamp at the units' end: the combine is timed in the reduce
        second = has_second.select(BLOCKS + rank, down_task)
        down_by_units(
            {**c, "second_live": has_second}, [down_task, second],
            lambda: stamp(timeline, tls, tid, tl0 + shift + 6),
        )  # fmt: skip
    elif const_expr(c["rs"].down_one_call):
        # a CTA without a second runs its own again (L2 hits), pushed once
        second = has_second.select(BLOCKS + rank, down_task)
        stage_down({**c, "second_live": has_second}, [down_task, second])
    elif has_second:
        stage_down(c, [down_task, BLOCKS + rank])
    else:
        stage_down(c, [down_task])
    flag = sreg(late_pointer(root,24),root['scratch_layout']['dnf'][0],'dnf')
    publish(c['put'],flag,bid,1,c['tid']==0)
    # token parallel: this rank's tokens' sums only (``out`` from its first
    # token's row, or from row 0: ``out_local``)
    if const_expr(c["tok_par"]):
        cr = own_view(c, () if c["out_local"] else ("out",))
    else:
        cr = c
    for task in range(first_task(fresh(bid), RED0), RED_TASKS, BLOCKS):
        stage_reduce(cr, task)
