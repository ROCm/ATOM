"""ATOM_MTP_FUSED_RS_STORE and ATOM_MTP_DRAFT_DEVICE_RNG (GPU).

The fused record store must write what `SpecRejectionSampling.store_step` and
the last pass's ``draft_ids`` copy write, bit for bit, and leave every other
output of the advance kernel unchanged. The device draw must be uniform, must
advance on every graph replay, and a graph that records it must replay without
the generator's seed / offset fills.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DEV = "cuda"
MTP_K = 3


def _rs(max_rows=64, max_num_seqs=32):
    from atom.model_ops.spec_rejection import SpecRejectionSampling

    rs = SpecRejectionSampling(
        max_rows=max_rows, mtp_k=MTP_K, max_num_seqs=max_num_seqs, device=DEV
    )
    g = torch.Generator(device=DEV).manual_seed(7)
    rs.records.copy_(
        torch.randint(-(2**31), 2**31 - 1, rs.records.shape, generator=g, device=DEV)
    )
    rs.record_tok.copy_(
        torch.randint(-1, 1 << 20, rs.record_tok.shape, generator=g, device=DEV)
    )
    rs.dbg_scratch.copy_(
        torch.randint(
            -(2**31), 2**31 - 1, rs.dbg_scratch.shape, generator=g, device=DEV
        )
    )
    rs.tok_scratch.copy_(
        torch.randint(0, 1 << 20, rs.tok_scratch.shape, generator=g, device=DEV)
    )
    perm = torch.randperm(rs.num_slots, generator=g, device=DEV)
    rs.step_slots.copy_(perm[: rs.max_rows])
    return rs


def _state(rs):
    return rs.records.clone(), rs.record_tok.clone()


@pytest.mark.parametrize("sbs,active", [(1, 1), (5, 5), (7, 4), (33, 33)])
@pytest.mark.parametrize("col", [0, 1, 2])
def test_the_last_pass_tail_writes_what_store_step_and_copy_write(sbs, active, col):
    from atom.spec_decode.eagle_fused_glue import store_last_draft_pass

    rs = _rs()
    rs.active_bs = active
    out_ids = torch.randint(0, 150000, (sbs + 3,), dtype=torch.int64, device=DEV)
    draft = torch.randint(0, 9, (sbs, MTP_K), dtype=torch.int64, device=DEV)
    before = _state(rs)

    rs.store_step(col, sbs)
    ref_draft = draft.clone()
    ref_draft[:, col].copy_(out_ids[:sbs])
    ref = _state(rs)

    rs.records.copy_(before[0])
    rs.record_tok.copy_(before[1])
    got_draft = draft.clone()
    store_last_draft_pass(
        out_ids, got_draft, col, sbs, records=rs.record_args(col, sbs)
    )
    torch.cuda.synchronize()
    assert torch.equal(got_draft, ref_draft)
    assert torch.equal(rs.records, ref[0])
    assert torch.equal(rs.record_tok, ref[1])


def test_the_last_pass_tail_without_records_is_the_copy():
    from atom.spec_decode.eagle_fused_glue import store_last_draft_pass

    out_ids = torch.randint(0, 150000, (9,), dtype=torch.int32, device=DEV)
    draft = torch.zeros(6, MTP_K, dtype=torch.int32, device=DEV)
    ref = draft.clone()
    ref[:, 2].copy_(out_ids[:6])
    store_last_draft_pass(out_ids, draft, 2, 6, records=None)
    assert torch.equal(draft, ref)


def test_record_args_skips_where_store_step_stores_nothing():
    rs = _rs()
    rs.active_bs = 0
    assert rs.record_args(0, 4) is None
    rs.active_bs = 4
    assert rs.record_args(MTP_K, 4) is None
    assert rs.record_args(0, 4, context=type("C", (), {"is_dummy_run": True})()) is None
    assert rs.record_args(1, 9)[-1] == 4


def _advance(records, sbs, rbs, first, seed=3):
    """One advance launch on scratch metadata; returns every tensor it writes."""
    from atom.spec_decode.eagle_fused_glue import _launch_advance

    g = torch.Generator(device=DEV).manual_seed(seed)
    i32 = {"dtype": torch.int32, "device": DEV}
    i64 = {"dtype": torch.int64, "device": DEV}
    H, rows, max_bs, page = 64, 4, 64, 16
    t = {
        "src_ids": torch.randint(0, 1000, (rbs,), generator=g, **i64),
        "src_hidden": torch.randn(
            rbs, H, generator=g, dtype=torch.bfloat16, device=DEV
        ),
        "dst_ids": torch.zeros(rbs, **i64),
        "dst_hidden": torch.zeros(rbs, H, dtype=torch.bfloat16, device=DEV),
        "dst_positions": torch.randint(0, 5000, (rbs,), generator=g, **i64),
        "draft_ids": torch.zeros(sbs, MTP_K, **i64),
        "context_lens": torch.randint(1, 3000, (max_bs,), generator=g, **i32),
        "kv_indptr": torch.zeros(rbs + 1, **i32),
        "block_tables": torch.randint(0, 900, (max_bs, 256), generator=g, **i32),
        "slot_mapping": torch.zeros(rbs, **i64),
        "cu_seqlens_q": torch.zeros(rbs + 1, **i32),
    }
    row_ids = torch.arange(max_bs + 1, **i32)
    extra = {}
    if first:
        extra = {
            "num_reject_tokens": torch.randint(0, 3, (sbs,), generator=g, **i32),
            "positions_src": torch.randint(0, 5000, (rbs * rows,), generator=g, **i64),
            "anchors": torch.randint(0, rbs * rows, (sbs,), generator=g, **i32),
        }
    _launch_advance(
        **t,
        num_reject_tokens=extra.get("num_reject_tokens"),
        row_ids=row_ids,
        positions_src=extra.get("positions_src"),
        anchors=extra.get("anchors"),
        draft_col=1,
        scheduled_bs=sbs,
        running_bs=rbs,
        num_slot_rows=sbs,
        kv_page=page,
        slot_page=page,
        records=records,
    )
    torch.cuda.synchronize()
    return t


@pytest.mark.parametrize("first", [True, False])
@pytest.mark.parametrize(
    "sbs,rbs,active", [(1, 1, 1), (6, 8, 6), (5, 8, 3), (33, 40, 33)]
)
def test_the_advance_stores_the_records_and_changes_nothing_else(
    first, sbs, rbs, active
):
    rs = _rs()
    rs.active_bs = active
    before = _state(rs)
    rs.store_step(1, sbs)
    ref = _state(rs)
    rs.records.copy_(before[0])
    rs.record_tok.copy_(before[1])

    plain = _advance(None, sbs, rbs, first)
    fused = _advance(rs.record_args(1, sbs), sbs, rbs, first)
    for k in plain:
        assert torch.equal(plain[k], fused[k]), k
    assert torch.equal(rs.records, ref[0])
    assert torch.equal(rs.record_tok, ref[1])


def _uniform(rs, rows):
    import triton

    from atom.model_ops.spec_rejection import _draft_uniform_kernel

    _draft_uniform_kernel[(1,)](
        rs.u_draft[:rows],
        rs._rng_state,
        rs._rng_seed,
        rows,
        BLOCK=max(16, triton.next_power_of_2(rs.max_rows)),
        num_warps=1,
    )


def test_the_device_draw_is_uniform_and_advances():
    rs = _rs(max_rows=256, max_num_seqs=128)
    draws = []
    for _ in range(64):
        _uniform(rs, 200)
        draws.append(rs.u_draft[:200].clone())
    u = torch.stack(draws)
    assert int(rs._rng_state.item()) == 64 * 200
    assert float(u.min()) >= 0.0 and float(u.max()) < 1.0
    assert abs(float(u.mean()) - 0.5) < 0.01
    assert abs(float(u.var()) - 1 / 12) < 0.005
    # no draw repeats an earlier one
    assert torch.unique(u).numel() > 0.999 * u.numel()
    # rows past `rows` are left alone
    rs.u_draft.fill_(-1.0)
    _uniform(rs, 5)
    assert bool((rs.u_draft[5:] == -1.0).all())


def _fill_kernels_in_replay(graph):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    return [n for n in names if "FillFunctor" in n], names


def test_a_recorded_device_draw_advances_every_replay_without_fills():
    rs = _rs(max_rows=64)
    _uniform(rs, 12)  # compile outside the capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        _uniform(rs, 12)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    start = int(rs._rng_state.item())
    g.replay()
    a = rs.u_draft[:12].clone()
    g.replay()
    b = rs.u_draft[:12].clone()
    torch.cuda.synchronize()
    assert int(rs._rng_state.item()) == start + 24
    assert not torch.equal(a, b)
    fills, names = _fill_kernels_in_replay(g)
    assert not fills, names


def test_a_recorded_torch_draw_replays_with_the_generator_fills():
    """The baseline this replaces: torch's uniform_ in a graph makes each replay
    launch the generator's seed / offset fills."""
    u = torch.zeros(12, dtype=torch.float32, device=DEV)
    u.uniform_()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        u.uniform_()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    fills, names = _fill_kernels_in_replay(g)
    assert len(fills) == 2, names


@pytest.mark.parametrize("replayssm", [True, False])
@pytest.mark.parametrize("nd", [1, 3, 4, 17, 63, 64])
def test_the_fused_gdn_decode_pad_writes_what_the_fills_and_copies_write(replayssm, nd):
    from atom.model_ops.attentions.gdn_attn import PAD_SLOT_ID, _gdn_spec_decode_pad

    max_bs, cols = 64, 4
    g = torch.Generator(device=DEV).manual_seed(nd)

    def bufs():
        return {
            "ns": torch.randint(
                -5, 900, (max_bs,), generator=g, dtype=torch.int32, device=DEV
            ),
            "sp": torch.randint(
                -5, 900, (max_bs, cols), generator=g, dtype=torch.int32, device=DEV
            ),
            "mask": torch.randint(0, 2, (max_bs,), generator=g, device=DEV).bool(),
            "qsl": torch.randint(
                0, 9000, (max_bs + 1,), generator=g, dtype=torch.int32, device=DEV
            ),
            "acc": torch.randint(
                0, 5, (max_bs,), generator=g, dtype=torch.int32, device=DEV
            ),
        }

    a = bufs()
    src = torch.sort(
        torch.randint(
            0, 9000, (max_bs + 1,), generator=g, dtype=torch.int32, device=DEV
        )
    ).values
    b = {k: v.clone() for k, v in a.items()}
    # the unfused sequence (`_attach_gdn_decode_metadata`, speculative branch)
    if replayssm:
        a["ns"][nd:].fill_(PAD_SLOT_ID)
    a["sp"][nd:, :].fill_(PAD_SLOT_ID)
    a["mask"][nd:].fill_(False)
    a["qsl"][: nd + 1].copy_(src[: nd + 1])
    a["qsl"][nd + 1 :].copy_(a["qsl"][nd : nd + 1])
    a["acc"][nd:].fill_(1)
    _gdn_spec_decode_pad(
        b["ns"] if replayssm else None, b["sp"], b["mask"], b["qsl"], src, b["acc"], nd
    )
    torch.cuda.synchronize()
    for k in a:
        assert torch.equal(a[k], b[k]), k
