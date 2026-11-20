# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3.1's midstep checkpoints: each interior one re-run as a head.

K3's chunk kernel keeps no state inside a forward, so its prefill appends each
interior checkpoint's head -- the row's tokens before the checkpoint, from the
row's initial state -- to the same varlen call as one more sequence. With that,
the prompt-end checkpoint no longer cuts a prefill in two. These tests pin the
index lists that build the call from the scheduler's reservations and what
the forward writes from them.
"""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from test_state_checkpoint import (
    BLOCK,
    PAGED_COPY_SPEC,
    PROMPT,
    ckpt_config,
    make_block_manager,
    stateful_seq,
)

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.state_runtime import (
    StateRuntime,
    StateTransfer,
    midstep_checkpoint_targets,
    midstep_rerun_rows,
)

#: K3's `conv_kernel - 1`.
CONV = 3


def runtime(readable):
    """K3's transfer with and without the re-run (`test_kda_layout_id` pins
    that the builder declares grid 1 and offset `CONV`)."""
    kw = {"readable_midstep": True, "midstep_grid": 1, "midstep_min_offset": CONV}
    return StateRuntime(
        transfer=StateTransfer.copy(
            PAGED_COPY_SPEC.layout_id, **(kw if readable else {})
        ),
        checkpoint_spec=PAGED_COPY_SPEC,
    )


def anchor_only_bm(readable=True):
    """Recipe F8's checkpoints: no ladder, so the prompt-end anchor is all."""
    config = ckpt_config(state_checkpoint_interval_tokens=-1, pool_entries={"state": 8})
    return make_block_manager(config, state_runtime=runtime(readable))


def admitted(bm, tokens):
    seq = stateful_seq(list(tokens))
    bm.allocate(seq, bm.can_allocate(seq))
    return seq


def rerun_rows(seqs, scheduled):
    batch = ScheduledBatch(
        seqs=dict(enumerate(seqs)),
        num_scheduled_tokens=scheduled,
        total_tokens_num=sum(scheduled),
        num_cached_tokens=[0] * len(seqs),
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
    return midstep_rerun_rows(targets, scheduled, CONV)


class TestAnchorInsideAPrefill:
    def test_without_the_re_run_the_anchor_cuts_the_prompt(self):
        """PROMPT's anchor is its last hashable block: 40 of 44 tokens."""
        bm = anchor_only_bm(readable=False)
        assert bm.checkpoint_cut(admitted(bm, PROMPT), 0, 44) == 10 * BLOCK

    def test_with_it_the_prompt_prefills_in_one_step(self):
        bm = anchor_only_bm()
        seq = admitted(bm, PROMPT)
        assert bm.checkpoint_cut(seq, 0, 44) == 0
        bm.plan_midstep(seq, 0, 44)
        assert [p for _s, p, _h in seq.midstep_reservations] == [10 * BLOCK]

    def test_scarce_page_scratch_keeps_the_anchor_before_the_ladder(self):
        config = ckpt_config(
            state_checkpoint_interval_tokens=32, pool_entries={"state": 2}
        )
        bm = make_block_manager(config, state_runtime=runtime(True))
        seq = admitted(bm, PROMPT)
        assert [p for p, _h in bm.midstep_positions(seq, 0, 44)] == [32, 40]
        bm.plan_midstep(seq, 0, 44)
        assert [p for _s, p, _h in seq.midstep_reservations] == [40]
        bm.cancel_midstep(seq)
        assert bm.state.num_free() == 1

    def test_the_head_rides_behind_every_row(self):
        bm = anchor_only_bm()
        other = admitted(bm, range(100, 112))
        seq = admitted(bm, PROMPT)
        bm.plan_midstep(seq, 0, 44)
        [(scratch, _p, _h)] = seq.midstep_reservations

        rows = rerun_rows([other, seq], [12, 44])

        # The rows' tokens unchanged, then row 1's first 40 again.
        assert rows["kda_tokens"] == list(range(56)) + list(range(12, 52))
        assert rows["kda_rows"] == [0, 1, 1]
        assert rows["kda_cu"] == [0, 12, 56, 96]
        assert rows["head_slots"] == rows["slots"] == [scratch]
        assert rows["conv_tokens"] == [[49, 50, 51]]
        assert "end_slots" not in rows

    def test_an_anchor_at_the_forward_s_end_copies_the_row(self):
        bm = anchor_only_bm()
        seq = admitted(bm, PROMPT)
        bm.plan_midstep(seq, 0, 40)
        [(scratch, _p, _h)] = seq.midstep_reservations

        rows = rerun_rows([seq], [40])

        assert rows == {
            "slots": [scratch],
            "conv_tokens": [[37, 38, 39]],
            "end_slots": [scratch],
            "end_runtime": [seq.state_slot],
        }


@pytest.fixture(scope="module")
def k3_modules():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            builder=load("atom.model_ops.attentions.kimi_mla_gdn_attn"),
            model=load("atom.models.kimi_k3.model"),
        )


def test_the_state_pool_lends_scratch_only_where_it_is_borrowed(k3_modules):
    """One slot per request; the scratch on top is for the checkpoints that
    two steps hold at once, and only where K3's forward writes them."""
    K3 = k3_modules.builder
    from test_kda_layout_id import builder

    def extra(**kw):
        b = object.__new__(K3.KimiAiterMLAGDNMetadataBuilder)
        inputs = builder(K3._KimiMLAGDNCommon, **kw)
        b.model_runner = inputs.model_runner
        b.num_spec, b.replayssm = inputs.num_spec, inputs.replayssm
        b._kpool_size = inputs._kpool_size
        b.num_state_layers = lambda: 1
        b._state_shape_for_runner = lambda: ((3, 8), (1, 2, 2))
        b._state_dtypes = lambda: (torch.bfloat16, torch.float32)
        b._replayssm_bytes_per_slot = lambda: 0
        b.slots_per_req = lambda: 1
        b._kpool_tail_bytes = lambda: 0
        return b.state_spec().extra_entries

    assert extra() == K3.MIDSTEP_SCRATCH_SLOTS > 0
    assert extra(pp=2) == extra(model_type="glm5_next_text") == 0

    from atom.model_ops.attentions.gdn_attn import DECODE_SNAPSHOT_SCRATCH_SLOTS

    # Midstep and decode images borrow from the same pool; keep both budgets.
    assert extra(replayssm=True, interval=32) == (
        K3.MIDSTEP_SCRATCH_SLOTS + DECODE_SNAPSHOT_SCRATCH_SLOTS
    )
    assert extra(replayssm=True, interval=32, pp=2) == DECODE_SNAPSHOT_SCRATCH_SLOTS
    assert extra(replayssm=True, interval=-1) == K3.MIDSTEP_SCRATCH_SLOTS
    assert extra(replayssm=True, interval=32, prefix_caching=False) == 0
    assert extra(interval=-1, prefix_caching=False) == 0
    assert extra(replayssm=True, interval=0) == 0
    assert extra(kpool=4) == 0


def test_the_builder_hands_the_forward_its_rows(k3_modules):
    """`kda_cu` joins the int32 query starts the KDA call is cut by."""
    K3 = k3_modules.builder
    from test_kda_layout_id import builder

    bm = anchor_only_bm()
    seq = admitted(bm, PROMPT)
    bm.plan_midstep(seq, 0, 44)
    batch = ScheduledBatch(
        seqs={0: seq},
        num_scheduled_tokens=[44],
        total_tokens_num=44,
        num_cached_tokens=[0],
    )
    b = object.__new__(K3.KimiAiterMLAGDNMetadataBuilder)
    b.model_runner = builder(K3._KimiMLAGDNCommon).model_runner
    b.model_runner.mamba_k_cache = torch.empty(1, 64)
    b.device = "cpu"

    got = b._checkpoint_targets(batch)

    assert {k: v.tolist() for k, v in got.items()} == rerun_rows([seq], [44])
    assert got["kda_cu"].dtype == torch.int32
    assert got["kda_tokens"].dtype == got["head_slots"].dtype == torch.int64


def test_finish_prefill_keeps_kda_targets_out_of_gdn_chunk_snapshots(
    monkeypatch, k3_modules
):
    K3 = k3_modules.builder
    from test_kda_layout_id import builder

    bm = anchor_only_bm()
    seq = admitted(bm, PROMPT)
    bm.plan_midstep(seq, 0, 44)
    batch = ScheduledBatch(
        seqs={0: seq},
        num_scheduled_tokens=[44],
        total_tokens_num=44,
        num_cached_tokens=[0],
    )
    b = object.__new__(K3.KimiAiterMLAGDNMetadataBuilder)
    b.model_runner = builder(K3._KimiMLAGDNCommon).model_runner
    b.model_runner.mamba_k_cache = torch.empty(1, 64)
    b.device = "cpu"
    meta = SimpleNamespace(non_spec_query_start_loc=torch.tensor([0, 44]))
    b.prepare_gdn_metadata = lambda *a, **kw: meta
    monkeypatch.setattr(K3.AiterMLAMetadataBuilder, "finish_prefill", lambda *a: None)
    attn = SimpleNamespace()

    b.finish_prefill(batch, attn, 1, 44)

    assert attn.gdn_metadata is meta
    assert {k: v.tolist() for k, v in meta.ssm_checkpoints.items()} == rerun_rows(
        [seq], [44]
    )
    assert not hasattr(meta, "ssm_chunk_offsets")


def test_the_forward_writes_heads_ends_and_conv_windows(k3_modules):
    attn = k3_modules.model.KimiKDAAttention
    ckpt = {
        "slots": torch.tensor([4, 5]),
        "conv_tokens": torch.tensor([[1, 2, 3], [5, 6, 7]]),
        "head_slots": torch.tensor([4]),
        "end_slots": torch.tensor([5]),
        "end_runtime": torch.tensor([0]),
    }
    mixed_qkv = torch.arange(8.0).repeat(2, 1).T  # token t holds t
    conv_state = torch.zeros(6, 2, 4)  # [slots, D, kernel - 1 + num_spec]
    ssm_state = torch.zeros(6, 1)
    ssm_state[0] = 9.0

    attn._write_state_checkpoints(
        ckpt, torch.tensor([[7.0]]), mixed_qkv, conv_state, ssm_state
    )

    assert ssm_state[4].item() == 7.0 and ssm_state[5].item() == 9.0
    assert conv_state[4, :, :3].tolist() == [[1.0, 2.0, 3.0]] * 2
    assert conv_state[5, :, :3].tolist() == [[5.0, 6.0, 7.0]] * 2
    assert not conv_state[:, :, 3].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashKDA needs a GPU")
def test_a_head_ends_on_the_state_a_forward_ending_there_would(k3_modules):
    """Row 1's checkpoint off FlashKDA's 32-token chunk grid, behind a row
    resuming from a state, the way a mixed step lays them out."""
    attn = k3_modules.model.KimiKDAAttention
    torch.manual_seed(0)
    heads, dim, lens, at = 4, 128, [300, 1100], 1001
    t = sum(lens)
    dev = "cuda"
    stub = SimpleNamespace(
        A_log=torch.randn(heads, device=dev),
        dt_bias=torch.randn(heads * dim, device=dev),
        _kda_gate_lower_bound=-5.0,
    )
    stub._run_kda = attn._run_kda.__get__(stub)

    def bf16(*shape):
        return torch.randn(1, t, *shape, device=dev, dtype=torch.bfloat16)

    q, k, v, g = (bf16(heads, dim) for _ in range(4))
    beta = bf16(heads)
    initial = torch.randn(2, heads, dim, dim, device=dev) * 0.1
    initial[1] = 0
    cu = torch.tensor([0, 300, t], device=dev, dtype=torch.int32)
    rows = midstep_rerun_rows([(1, 7, at, 0, 3), (1, 8, 1100, 1, 3)], lens, CONV)
    ckpt = {
        k_: torch.tensor(
            v_, device=dev, dtype=torch.int32 if k_ == "kda_cu" else torch.int64
        )
        for k_, v_ in rows.items()
    }

    out, last, head = attn._prefill_kda(stub, q, k, v, g, beta, initial, cu, ckpt)

    want_out, want_last = stub._run_kda(q, k, v, g, beta, initial, cu, True)
    span = slice(300, 300 + at)
    _, want_head = stub._run_kda(
        *(x[:, span] for x in (q, k, v, g, beta)),
        initial[1:],
        torch.tensor([0, at], device=dev, dtype=torch.int32),
        True,
    )
    torch.testing.assert_close(out, want_out, rtol=0, atol=0)
    torch.testing.assert_close(last, want_last, rtol=0, atol=0)
    torch.testing.assert_close(head, want_head, rtol=1e-5, atol=1e-5)
