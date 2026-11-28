# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3.1's interior checkpoints written by cutting the recurrence at them
(`kda_ckpt_split`).

The head path re-runs every interior checkpoint's head as one more sequence;
the split runs each token once, block by block on the paged state, and copies a
checkpoint out of the runtime slot once the block ending on it is done. These
tests pin the blocks the planner cuts, what the builder hands the forward, and
that on a stand-in recurrence the split leaves the outputs, final states and
checkpoints (both halves, at the same positions) the head path leaves.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def k3():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            builder=load("atom.model_ops.attentions.kimi_mla_gdn_attn"),
            model=load("atom.models.kimi_k3.model"),
            writer=load("atom.model_ops.fla_ops.state_checkpoint"),
            split=load("atom.model_ops.kimi_k3.kda_ckpt_split"),
        )


@pytest.fixture(autouse=True)
def no_tail_recurrence(monkeypatch, k3):
    """The stand-in FlashKDA here cannot run the token recurrence, and these
    tests are about the blocks, so no block takes it."""
    monkeypatch.setattr(k3.split, "TAIL_RECURRENT_MAX", 0)


def step(scheduled, cached=None, slots=None, forks=None, state_rows=None):
    n = len(scheduled)
    return SimpleNamespace(
        num_scheduled_tokens=scheduled,
        num_cached_tokens=cached or [0] * n,
        state_slots_committed=slots or list(range(10, 10 + n)),
        state_fork_srcs=forks or [-1] * n,
        state_rows=list(range(n)) if state_rows is None else state_rows,
        state_save_all=[[True]] * n,
    )


def blocks(plan):
    return [(b.start, b.end, b.bounds, b.targets) for b in plan.blocks]


def chunk_indices(cu, chunk=32):
    """FlashKDA's `prepare_chunk_indices`."""
    lens = torch.diff(torch.tensor(cu))
    indices = torch.cat([torch.arange(-(-int(n) // chunk)) for n in lens])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).tolist()


def test_a_prompt_end_anchor_cuts_one_row_in_two(k3):
    """KGAP's 1K cell: 1025 tokens after a 128K context, anchored at 1024."""
    plan = k3.split.plan_checkpoint_split(
        [(0, 7, 1024, 0, 10)], step([1025], cached=[131072])
    )
    assert blocks(plan) == [(0, 1024, ((0, 1024),), 1), (1024, 1025, ((0, 1),), 0)]
    chunks = plan.values.pop("split_chunks")
    assert chunks == [v for c in range(32) for v in (0, c)] + [0, 0]
    assert plan.values == {
        "split_slots": [10, 10],
        "split_init": [1, 1],
        "split_cu": [0, 1024, 0, 1],
        "split_ck_rows": [0],
        "split_ck_slots": [7],
        "split_ck_offs": [1024],
        "split_ck_ends": [1],
        "split_ck_runtime": [10],
    }


def test_a_short_resumed_tail_takes_the_token_recurrence(monkeypatch, k3):
    """With the default `TAIL_RECURRENT_MAX`, the 1K cell's 1-token tail block
    reads and writes its row's runtime slot through the token recurrence."""
    monkeypatch.setattr(k3.split, "TAIL_RECURRENT_MAX", 6)
    targets, batch = [(0, 7, 1024, 0, 10)], step([1025], cached=[131072])
    plan = k3.split.plan_checkpoint_split(targets, batch)
    assert [b.tail for b in plan.blocks] == [0, 1]
    assert plan.values["split_tail"] == [10, 10]
    split = bound(k3, targets, batch)
    assert split.blocks[0].tail is None
    assert split.blocks[1].tail.tolist() == [[[10]], [[10]]]
    # A tail past the limit stays on FlashKDA.
    longer = k3.split.plan_checkpoint_split(targets, step([1031], cached=[1]))
    assert [b.tail for b in longer.blocks] == [0, 0]


def test_a_fresh_row_starts_its_first_block_from_zero(k3):
    plan = k3.split.plan_checkpoint_split([(0, 7, 1024, 0, 10)], step([1025]))
    assert plan.values["split_init"] == [0, 1]


def test_rungs_cut_in_order_and_end_targets_follow_the_last_block(k3):
    targets = [(0, 8, 15360, 0, 10), (0, 9, 16384, 1, 10), (0, 7, 8192, 0, 10)]
    plan = k3.split.plan_checkpoint_split(targets, step([16384]))
    assert [(b.start, b.end, b.targets) for b in plan.blocks] == [
        (0, 8192, 1),
        (8192, 15360, 1),
        (15360, 16384, 1),
    ]
    assert plan.values["split_init"] == [0, 1, 1]
    assert plan.values["split_ck_slots"] == [7, 8, 9]
    assert plan.values["split_ck_offs"] == [8192, 15360, 16384]
    assert plan.values["split_ck_ends"] == [1, 1, 1]


def test_rows_before_a_cut_share_its_block(k3):
    targets = [(1, 9, 1024, 0, 4), (0, 8, 512, 1, 3)]
    plan = k3.split.plan_checkpoint_split(
        targets, step([512, 1025], cached=[0, 4096], slots=[3, 4])
    )
    assert blocks(plan) == [
        (0, 1536, ((0, 512), (512, 1536)), 1),
        (1536, 1537, ((0, 1),), 1),
    ]
    assert plan.values["split_cu"] == [0, 512, 1536, 0, 1]
    assert plan.values["split_slots"] == [3, 4, 4]
    assert plan.values["split_init"] == [0, 1, 1]
    assert plan.values["split_ck_rows"] == [1, 0]
    assert plan.values["split_ck_slots"] == [9, 8]
    assert plan.values["split_ck_runtime"] == [4, 3]


def test_a_scheduled_prompt_cuts_at_its_anchor(k3):
    """The block manager's prompt-end anchor, through a real `ScheduledBatch`,
    behind a row without a checkpoint."""
    from test_kda_midstep_checkpoints import CONV, admitted, anchor_only_bm
    from test_state_checkpoint import PROMPT

    from atom.model_engine.scheduler import ScheduledBatch
    from atom.model_engine.state_runtime import midstep_checkpoint_targets

    bm = anchor_only_bm()
    other = admitted(bm, range(100, 112))
    seq = admitted(bm, PROMPT)
    bm.plan_midstep(seq, 0, 44)
    [(scratch, _p, _h)] = seq.midstep_reservations
    batch = ScheduledBatch(
        seqs={0: other, 1: seq},
        num_scheduled_tokens=[12, 44],
        total_tokens_num=56,
        num_cached_tokens=[0, 0],
    )
    targets = midstep_checkpoint_targets(
        batch.state_rows,
        batch.num_cached_tokens,
        batch.num_scheduled_tokens,
        batch.state_slots_committed,
        batch.state_save_all,
        num_slots=64,
        grid=1,
        min_offset=CONV,
    )

    plan = k3.split.plan_checkpoint_split(targets, batch)

    assert blocks(plan) == [(0, 52, ((0, 12), (12, 52)), 1), (52, 56, ((0, 4),), 0)]
    assert plan.values["split_slots"] == [other.state_slot, *[seq.state_slot] * 2]
    assert plan.values["split_init"] == [0, 0, 1]
    assert plan.values["split_ck_slots"] == [scratch]
    assert plan.values["split_ck_offs"] == [40]
    assert plan.values["split_ck_runtime"] == [seq.state_slot]


def test_each_block_carries_flash_kdas_bounds_and_chunk_indices(k3):
    targets = [(1, 7, 50, 0, 11), (2, 8, 70, 0, 12)]
    plan = k3.split.plan_checkpoint_split(targets, step([100, 100, 100, 100]))
    assert blocks(plan) == [
        (0, 150, ((0, 100), (100, 150)), 1),
        (150, 270, ((0, 50), (50, 120)), 1),
        (270, 400, ((0, 30), (30, 130)), 0),
    ]
    assert plan.values["split_cu"] == [0, 100, 150, 0, 50, 120, 0, 30, 130]
    assert plan.values["split_slots"] == [10, 11, 11, 12, 12, 13]
    assert plan.values["split_init"] == [0, 0, 1, 0, 1, 0]
    got = torch.tensor(plan.values["split_chunks"]).view(-1, 2).tolist()
    want = []
    for b in plan.blocks:
        want += chunk_indices([0, *(e for _, e in b.bounds)])
    assert got == want


@pytest.mark.parametrize(
    "targets, batch",
    [
        ([(0, 7, 1024, 1, 10)], step([1024])),
        ([(0, 7, 512, 0, 10)], step([1024], forks=[5])),
        ([(0, 7, 512, 0, 10)], step([1024, 8], state_rows=[0])),
        ([(1, 7, 50, 0, 11)], step([100, 100, 100], state_rows=[0, 2, 1])),
        ([(1, 7, 50, 0, 11)], step([100, 100, 0])),
    ],
    ids=["end-only", "fork", "stateless-row", "reordered", "empty-row"],
)
def test_the_heads_stay_where_the_blocks_cannot_run_in_place(targets, batch, k3):
    assert k3.split.plan_checkpoint_split(targets, batch) is None


def flash_kda(route=lambda segmented, width, seqs, heads: False, **kw):
    """The FlashKDA names `kda_ckpt_split` reads."""
    calls = []

    def seq_bounds(cu):
        calls.append("bounds")
        return "flash_kda's bounds"

    fk = SimpleNamespace(
        FLASH_KDA_CHUNK=32,
        _SEG_MIN_CHUNKS=64,
        _k2c_gluon_usable=route,
        _seq_bounds=seq_bounds,
        prepare_chunk_indices=chunk_indices_reader(calls, "flash_kda"),
        calls=calls,
    )
    fk.__dict__.update(kw)
    return fk


def chunk_indices_reader(calls, module):
    def prepare_chunk_indices(cu, chunk_size):
        calls.append(f"{module} chunks")
        return f"{module}'s chunks"

    return prepare_chunk_indices


@pytest.fixture
def fake_aiter(monkeypatch, k3):
    """FlashKDA and `chunk_delta_attn_fwd`'s module, as `kda_ckpt_split` sees
    them, with the hook not yet installed."""
    fk = flash_kda()
    chunk_fwd = SimpleNamespace(
        prepare_chunk_indices=chunk_indices_reader(fk.calls, "chunk_fwd")
    )
    monkeypatch.setattr(k3.split, "_flash_kda", lambda: fk)
    monkeypatch.setattr(k3.split, "_chunk_fwd", lambda: chunk_fwd)
    k3.split.install_host_bounds.cache_clear()
    yield fk, chunk_fwd
    k3.split.install_host_bounds.cache_clear()


def bound(k3, targets, batch):
    plan = k3.split.plan_checkpoint_split(targets, batch)
    tensors = {n: torch.tensor(v, dtype=torch.int32) for n, v in plan.values.items()}
    return plan.bind(tensors)


@pytest.mark.parametrize(
    "targets, scheduled",
    [
        ([(0, 7, 1024, 0, 10)], [1025]),
        ([(0, 7, 1024, 0, 10)], [63 * 32 + 1]),
        ([(0, 7, 1000, 0, 10)], [1025]),
    ],
    ids=["1k", "64-chunks", "off-grid"],
)
def test_a_split_runs_on_flash_kdas_chunk(targets, scheduled, monkeypatch, k3):
    monkeypatch.setattr(k3.split, "_flash_kda", lambda: flash_kda())
    assert bound(k3, targets, step(scheduled)).runs(12, 128)


def test_no_split_runs_on_another_chunk(monkeypatch, k3):
    monkeypatch.setattr(k3.split, "_flash_kda", lambda: flash_kda(FLASH_KDA_CHUNK=64))
    split = bound(k3, [(0, 7, 1024, 0, 10)], step([1025]))
    assert not split.runs(12, 128)


def test_flash_kda_takes_the_bounds_off_a_blocks_cu_seqlens(fake_aiter, k3):
    fk, chunk_fwd = fake_aiter
    k3.split.install_host_bounds()
    k3.split.install_host_bounds()
    split = bound(k3, [(1, 7, 50, 0, 11)], step([100, 100]))
    cu = split.blocks[0].cu_seqlens
    assert fk._seq_bounds(cu) == ((0, 100), (100, 150))
    for module in (fk, chunk_fwd):
        got = module.prepare_chunk_indices(cu, 32)
        assert got.tolist() == chunk_indices([0, 100, 150])
    assert fk.calls == []
    plain = torch.tensor([0, 100, 150], dtype=torch.int32)
    assert fk._seq_bounds(plain) == "flash_kda's bounds"
    assert fk.prepare_chunk_indices(plain, 32) == "flash_kda's chunks"
    assert chunk_fwd.prepare_chunk_indices(plain, 32) == "chunk_fwd's chunks"
    assert chunk_fwd.prepare_chunk_indices(cu, 64) == "chunk_fwd's chunks"
    assert fk.calls == [
        "bounds",
        "flash_kda chunks",
        "chunk_fwd chunks",
        "chunk_fwd chunks",
    ]


def call_builder(k3, targets, batch, monkeypatch):
    b = SimpleNamespace(device="cpu", _midstep_targets=lambda batch, grid: targets)
    transfers, make_tensor = [], torch.tensor

    def record(values, **kwargs):
        transfers.append(len(values))
        return make_tensor(values, **kwargs)

    monkeypatch.setattr(torch, "tensor", record)
    got = k3.builder._KimiMLAGDNCommon._checkpoint_targets(b, batch)
    monkeypatch.setattr(torch, "tensor", make_tensor)
    return got, transfers


def heads_only(monkeypatch):
    """The builder as it plans without the split: the heads only."""
    monkeypatch.setitem(
        sys.modules,
        "atom.model_ops.kimi_k3.kda_ckpt_split",
        SimpleNamespace(plan_checkpoint_split=lambda targets, batch: None),
    )


@pytest.mark.parametrize("split_on", [False, True])
def test_the_builder_adds_the_blocks_to_its_one_transfer(split_on, monkeypatch, k3):
    if not split_on:
        heads_only(monkeypatch)
    targets = [(1, 9, 1024, 0, 4), (0, 8, 512, 1, 3)]
    batch = step([512, 1025], cached=[0, 4096], slots=[3, 4])
    got, transfers = call_builder(k3, targets, batch, monkeypatch)

    assert len(transfers) == 1
    split = got.pop("kda_split", None)
    # The heads stay planned either way, for `_paged_prefill` to fall back to.
    assert got["kda_cu"].tolist() == [0, 512, 1537, 2561]
    assert torch.equal(
        got["kda_tokens"], torch.cat([torch.arange(1537), torch.arange(512, 1536)])
    )
    assert sorted(got) == sorted(
        ["rows", "slots", "offs", "is_end", "runtime", "h_rows"]
        + ["kda_rows", "kda_cu", "kda_tokens"]
    )
    if not split_on:
        assert split is None
        return
    first, second = split.blocks
    assert (first.start, first.end, second.start, second.end) == (0, 1536, 1536, 1537)
    assert first.cu_seqlens.tolist() == [0, 512, 1536]
    assert second.cu_seqlens.tolist() == [0, 1]
    assert first.cu_seqlens.dtype == torch.int32
    assert first.cu_seqlens._atom_seq_bounds == ((0, 512), (512, 1536))
    assert second.cu_seqlens._atom_seq_bounds == ((0, 1),)
    for block in split.blocks:
        assert block.cu_seqlens._atom_chunk_indices.tolist() == chunk_indices(
            block.cu_seqlens.tolist()
        )
        assert block.cu_seqlens._atom_chunk_indices.dtype == torch.int32
    assert first.state_indices.tolist() == [3, 4]
    assert second.state_indices.tolist() == [4]
    assert first.has_initial_state.tolist() == [False, True]
    assert second.has_initial_state.tolist() == [True]
    assert second.has_initial_state.dtype == torch.bool
    assert [t.tolist() for t in first.checkpoints] == [[1], [9], [1024], [1], [4]]
    assert [t.tolist() for t in second.checkpoints] == [[0], [8], [512], [1], [3]]
    int32 = [got["rows"], first.cu_seqlens, first.state_indices]
    int32 += [*first.checkpoints, *second.checkpoints, second.state_indices]
    int32 += [first.cu_seqlens._atom_chunk_indices, second.cu_seqlens]
    assert len({t.untyped_storage().data_ptr() for t in int32}) == 1


def test_the_builder_cuts_off_flash_kdas_chunk_grid_too(monkeypatch, k3):
    got, _ = call_builder(k3, [(0, 7, 1000, 0, 10)], step([4097]), monkeypatch)
    assert [(b.start, b.end) for b in got["kda_split"].blocks] == [
        (0, 1000),
        (1000, 4097),
    ]


class Verdict:
    def __init__(self, runs):
        self.runs = lambda heads, head_dim: runs and (heads, head_dim) == (12, 128)


@pytest.mark.parametrize(
    "ckpt, paged",
    [
        (None, True),
        ({"rows": None}, True),
        ({"kda_tokens": None}, False),
        ({"kda_tokens": None, "kda_split": Verdict(True)}, True),
        ({"kda_tokens": None, "kda_split": Verdict(False)}, False),
    ],
    ids=["none", "end-only", "heads", "split", "split-turned-down"],
)
def test_a_split_step_runs_on_the_paged_state(ckpt, paged, k3):
    layer = object.__new__(k3.model.KimiKDAAttention)
    layer.num_local_heads, layer.head_dim = 12, 128
    meta = SimpleNamespace(has_state_fork=False, ssm_checkpoints=ckpt)
    # The stub's FlashKDA flags are MagicMocks, so FlashKDA reads as enabled.
    assert layer._paged_prefill(meta, torch.zeros(2, 1, 2, 2)) is paged


def recurrence(**kw):
    """A stand-in `_chunk_kda`, in either of FlashKDA's state modes: each token
    decays the state and adds its own outer product, then reads it with q."""
    q, k, v, cu = kw["q"], kw["k"], kw["v"], kw["cu_seqlens"]
    bounds = cu.tolist()
    paged = kw.get("state_cache") is not None
    out = kw["out"] if paged else torch.zeros_like(v)
    finals = []
    for i in range(len(bounds) - 1):
        if paged:
            slot = int(kw["state_indices"][i])
            state = kw["state_cache"][slot].clone()
            if not kw["has_initial_state"][i]:
                state.zero_()
        else:
            state = kw["initial_state"][i].clone()
        for t in range(bounds[i], bounds[i + 1]):
            state = 0.5 * state + v[0, t, :, :, None] * k[0, t, :, None, :]
            out[0, t] = (state * q[0, t, :, None, :]).sum(-1)
        if paged:
            kw["state_cache"][slot] = state
        else:
            finals.append(state)
    assert kw["state_v_first"] and kw["output_final_state"] == (not paged)
    return out, None if paged else torch.stack(finals)


@pytest.fixture
def cpu_writer(monkeypatch, k3):
    class Kernel:
        def __getitem__(self, grid):
            def copy(h, pool, x, conv, rows, slots, offs, ends, runtime, *a, **kw):
                _, h_rows, cu = a[:3]
                states = h.reshape(-1, *pool.shape[1:])
                for i in range(len(rows)):
                    if ends[i]:
                        src = pool[int(runtime[i])].clone()
                    else:
                        src = states[int(h_rows[i])]
                    pool[int(slots[i])].copy_(src)
                    end = int(cu[int(rows[i])] + offs[i])
                    width = kw["STATE_LEN"]
                    conv[int(slots[i]), :, :width].copy_(x[end - width : end].T)

            return copy

    monkeypatch.setattr(k3.writer, "_copy_checkpoints_kernel", Kernel())
    monkeypatch.setattr(k3.writer.triton, "cdiv", lambda n, d: (n + d - 1) // d)


CASES = {
    "anchor": ([9], [5], [2], [(0, 6, 8, 0, 2)], 9),
    "fresh-rungs-end": ([12], [0], [2], [(0, 5, 4, 0, 2), (0, 6, 8, 0, 2)], 12),
    "rows": ([5, 9], [3, 0], [2, 3], [(1, 6, 7, 0, 3), (0, 7, 5, 1, 2)], 14),
    "token-graph-pad": ([9], [5], [2], [(0, 6, 8, 0, 2)], 16),
}


@pytest.mark.parametrize("case", CASES)
def test_the_split_leaves_what_the_heads_leave(
    case, monkeypatch, k3, cpu_writer, fake_aiter
):
    scheduled, cached, slots, targets, height = CASES[case]
    if case == "fresh-rungs-end":
        targets = targets + [(0, 7, 12, 1, 2)]
    g = torch.Generator().manual_seed(0)
    H, K, D = 1, 2, 4
    q, k, v, gate = (torch.randn(1, height, H, K, generator=g) for _ in range(4))
    beta = torch.randn(1, height, H, generator=g)
    x = torch.randn(height, D, generator=g)
    pool0 = torch.randn(8, H, K, K, generator=g)
    conv0 = torch.randn(8, D, 4, generator=g)
    cu = torch.tensor([0, *torch.tensor(scheduled).cumsum(0).tolist()])
    has = torch.tensor([c > 0 for c in cached])
    batch = step(scheduled, cached=cached, slots=slots)

    layer = object.__new__(k3.model.KimiKDAAttention)
    torch.nn.Module.__init__(layer)
    layer.A_log = layer.dt_bias = None
    layer._kda_gate_lower_bound = -5.0
    layer._chunk_kda = recurrence
    layer.num_local_heads, layer.head_dim = H, K

    left = {}
    for on in ("0", "1"):
        with monkeypatch.context() as patch:
            if on == "0":
                heads_only(patch)
            ckpt, _ = call_builder(k3, targets, batch, patch)
        assert ("kda_split" in ckpt) == (on == "1")
        meta = SimpleNamespace(
            non_spec_state_indices_tensor=torch.tensor(slots, dtype=torch.int32),
            non_spec_query_start_loc=cu.to(torch.int32),
            has_initial_state=has,
            has_state_fork=False,
            ssm_checkpoints=ckpt,
            ssm_conv_state_len=3,
        )
        pool, conv = pool0.clone(), conv0.clone()
        initial = None
        if not layer._paged_prefill(meta, pool):
            initial = pool[torch.tensor(slots)].clone()
            initial[~has] = 0
        assert (initial is None) == (on == "1")
        out = layer._settle_prefill(q, k, v, gate, beta, initial, meta, x, conv, pool)
        left[on] = (out[:, : cu[-1]], pool, conv)

    for heads, split in zip(left["0"], left["1"]):
        torch.testing.assert_close(split, heads, rtol=0, atol=0)
    written = sorted({t[1] for t in targets} | set(slots))
    untouched = [s for s in range(8) if s not in written]
    torch.testing.assert_close(left["1"][1][untouched], pool0[untouched])
    torch.testing.assert_close(left["1"][2][untouched], conv0[untouched])
