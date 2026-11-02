# SPDX-License-Identifier: MIT
"""Qwen3.5 MTP mid-steps (draft steps 1..k-1) on its MRoPE target.

Qwen3.5 carries ``mrope_section``, so its positions are ``[3, N]`` with the
token axis last. ``Qwen3_5MTP`` sets ``draft_graph_supports_mrope``, so the
mid-step pass stages them as a batch-last input and the steps are warmed and
captured instead of running eager. These pin that the staged positions are
exactly what Qwen3.5's eager path (GDN builder, unfused position bump) built,
rows and all.
"""

import types
from unittest.mock import patch

import pytest
import torch

mod_eagle = pytest.importorskip(
    "atom.spec_decode.eagle_proposer",
    reason="the drafter imports aiter at module load",
    exc_type=ImportError,
)

EagleProposer = mod_eagle.EagleProposer


def _declaring_proposer(use_mrope, mtp_k=3):
    hf = types.SimpleNamespace(hidden_size=64)
    return types.SimpleNamespace(
        runner=types.SimpleNamespace(use_mrope=use_mrope),
        mtp_k=mtp_k,
        speculative_config=types.SimpleNamespace(draft_model_hf_config=hf),
        # What `Qwen3_5MTP` declares; `test_qwen35_mtp_opts_into_mrope_graphs`
        # pins the real class.
        model=types.SimpleNamespace(draft_graph_supports_mrope=True),
        dtype=torch.bfloat16,
        _step_forward=lambda *a, **kw: None,
        _step_head=lambda *a, **kw: None,
        _step_warmup_inputs=lambda *a, **kw: None,
    )


def _declare_bound(use_mrope, max_num_seqs=8):
    proposer = _declaring_proposer(use_mrope)
    (graph,) = EagleProposer._declare_draft_graphs(proposer)
    graph.bind(types.SimpleNamespace(max_num_seqs=max_num_seqs), "cpu")
    return proposer, graph


def test_qwen35_mtp_opts_into_mrope_graphs():
    mod = pytest.importorskip(
        "atom.models.qwen3_5_mtp",
        reason="the Qwen3.5 MTP model imports aiter at module load",
        exc_type=ImportError,
    )
    assert mod.Qwen3_5MTP.draft_graph_supports_mrope is True


def test_qwen35_mrope_declares_a_mid_step_graph():
    proposer, graph = _declare_bound(use_mrope=True)
    assert proposer.step is graph
    positions = graph.inputs["positions"]
    assert positions.batch_last
    assert positions.shape == (3,)
    assert positions.dtype == torch.int64
    # The rest of the pass is the same one every other model declares.
    assert not graph.inputs["input_ids"].batch_last
    assert graph.inputs["hidden_states"].shape == (64,)


def test_non_mrope_model_keeps_one_dimensional_positions():
    _, graph = _declare_bound(use_mrope=False)
    positions = graph.inputs["positions"]
    assert not positions.batch_last
    assert positions.shape == ()


def test_mtp_k_one_still_declares_nothing():
    proposer = _declaring_proposer(use_mrope=True, mtp_k=1)
    assert EagleProposer._declare_draft_graphs(proposer) == ()
    assert proposer.step is None


# ---- the positions a mid-step sees: eager vs staged ------------------------


def _enter_decode(target_positions, last_token_indices, running_bs):
    """Run the real `_enter_decode_metadata` on a non-MLA, no-flat-KV runner
    (Qwen3.5's GDN builder) and return the per-sequence positions it builds:
    what the eager mid-step used to hand the model."""
    scheduled_bs = last_token_indices.shape[0]
    cap = 8
    var = {
        name: types.SimpleNamespace(gpu=torch.zeros(cap + 1, dtype=torch.int32))
        for name in ("slot_mapping", "cu_seqlens_q", "context_lens")
    }
    var["block_tables"] = types.SimpleNamespace(
        gpu=torch.zeros(cap, 4, dtype=torch.int32)
    )
    builder = types.SimpleNamespace(
        row_ids=torch.arange(cap + 1, dtype=torch.int32),
        _publish_token_block_tables=lambda *a: None,
    )
    proposer = types.SimpleNamespace(
        runner=types.SimpleNamespace(
            forward_vars=var,
            attn_metadata_builder=builder,
            attn_family=types.SimpleNamespace(is_mla=False),
        )
    )
    fc = types.SimpleNamespace(
        attn_metadata=types.SimpleNamespace(max_seqlen_q=4),
        context=types.SimpleNamespace(is_prefill=False),
    )
    with patch("atom.spec_decode.eagle_proposer.get_forward_context", return_value=fc):
        _, positions = EagleProposer._enter_decode_metadata(
            proposer,
            scheduled_bs,
            running_bs,
            target_positions + 1,  # `propose`'s step-0 positions
            last_token_indices,
            torch.zeros(scheduled_bs, dtype=torch.int64),
        )
    return positions


def _decode_mrope_positions(context_lens, q, deltas):
    """`_build_mrope_decode_positions`: one arange per request, plus its
    `mrope_position_delta`, broadcast to all three rows."""
    cols = [
        torch.arange(ctx - q + d, ctx + d, dtype=torch.int64)
        for ctx, d in zip(context_lens, deltas)
    ]
    return torch.cat(cols).unsqueeze(0).expand(3, -1).contiguous()


@pytest.mark.parametrize("deltas", [(0, 0, 0), (0, -7, 12)])
def test_staged_positions_equal_the_eager_ones_on_text_decode(deltas):
    """Text-only decode (and a multimodal request in decode: rows equal but
    offset by its delta). The replayed graph must read, in its real columns,
    exactly the `[3, scheduled_bs]` the eager step fed the model -- delta
    included, so nothing may rebuild positions from `context_lens`."""
    q, scheduled_bs, running_bs = 4, 3, 8
    target = _decode_mrope_positions([20, 33, 41], q, deltas)
    # Anchors after a verify: a different accepted length per sequence.
    anchors = torch.tensor([1, 4 + 3, 8 + 0])
    eager = _enter_decode(target, anchors, running_bs)
    assert eager.shape == (3, scheduled_bs)
    assert (eager[0] == eager[1]).all() and (eager[0] == eager[2]).all()

    _, graph = _declare_bound(use_mrope=True)
    staged = graph.stage(running_bs, {"positions": eager})["positions"]
    assert staged.shape == (3, running_bs)
    assert staged.stride() == (running_bs, 1)
    torch.testing.assert_close(staged[:, :scheduled_bs], eager)
    torch.testing.assert_close(
        staged[:, scheduled_bs:], eager[:, -1:].expand(3, running_bs - scheduled_bs)
    )
    # Anchor columns 1, 7, 8 hold 17, 32, 37 (+ delta); step 0 adds one.
    expected = [18 + deltas[0], 33 + deltas[1], 38 + deltas[2]]
    assert staged[0, :scheduled_bs].tolist() == expected


def test_unequal_rows_from_a_vision_anchor_are_graphed_unchanged():
    """A prefill whose anchor column sits on a vision token: t/h/w differ.
    Nothing here falls back to eager or refuses -- the staged positions keep
    all three rows, so the same recording replays correctly."""
    target = torch.tensor(
        [
            [0, 1, 2, 2, 2, 5],
            [0, 1, 2, 3, 3, 5],
            [0, 1, 2, 3, 4, 5],
        ],
        dtype=torch.int64,
    )
    anchors = torch.tensor([3, 5])  # seq 0 ends on a vision token
    eager = _enter_decode(target, anchors, running_bs=4)
    assert eager[:, 0].tolist() == [3, 4, 4]
    _, graph = _declare_bound(use_mrope=True)
    staged = graph.stage(4, {"positions": eager})["positions"]
    torch.testing.assert_close(staged[:, :2], eager)
    assert staged[:, 2:].tolist() == [[6, 6], [6, 6], [6, 6]]


def test_mid_step_bump_advances_all_three_staged_rows_in_place():
    """Qwen3.5's GDN builder does not fuse the position bump, so
    `_advance_decode_metadata` does `positions += 1` itself -- on the staged
    storage the recording reads."""
    _, graph = _declare_bound(use_mrope=True)
    staged = graph.stage(4, {"positions": torch.full((3, 2), 9, dtype=torch.int64)})[
        "positions"
    ]
    calls = []
    builder = types.SimpleNamespace(
        num_attention_heads=64,
        fuse_mtp_decode_position_update=True,  # ignored for 2-D positions
        prepare_mtp_decode=lambda *a, **kw: calls.append((a, kw)) or {},
    )
    proposer = types.SimpleNamespace(
        runner=types.SimpleNamespace(attn_metadata_builder=builder)
    )
    attn_metadata = types.SimpleNamespace(
        max_seqlen_k=10, context_lens=torch.tensor([5, 6, 6, 6], dtype=torch.int32)
    )
    fc = types.SimpleNamespace(attn_metadata=attn_metadata)
    with patch("atom.spec_decode.eagle_proposer.get_forward_context", return_value=fc):
        EagleProposer._advance_decode_metadata(
            proposer,
            2,
            4,
            staged,
            4,
            num_reject_tokens=None,
            from_prefill=False,
        )
    assert graph.buffer("positions", 4).tolist() == [[10] * 4] * 3
    ((args, kwargs),) = calls
    assert args[3] is staged
    assert "positions_out" not in kwargs  # the unfused branch


def test_warmup_fills_three_equal_rows_from_context_lens():
    """The warmup batch is text-shaped: each row of `[3, bs]` is where its
    synthetic sequence is, plus the one bump `_advance_decode_metadata` makes."""
    _, graph = _declare_bound(use_mrope=True)
    staged = graph._stage_one("positions", 3, None)
    builder = types.SimpleNamespace(
        num_attention_heads=64,
        fuse_mtp_decode_position_update=False,
        prepare_mtp_decode=lambda *a, **kw: {},
    )
    proposer = types.SimpleNamespace(
        _share_mtp_indices=False,
        device=torch.device("cpu"),
        runner=types.SimpleNamespace(attn_metadata_builder=builder),
        _enter_decode_metadata=lambda *a: (4, None),
    )
    proposer._advance_decode_metadata = types.MethodType(
        EagleProposer._advance_decode_metadata, proposer
    )
    attn_metadata = types.SimpleNamespace(
        max_seqlen_q=1,
        max_seqlen_k=9,
        context_lens=torch.tensor([7, 8, 9], dtype=torch.int32),
    )
    fc = types.SimpleNamespace(attn_metadata=attn_metadata)
    with patch("atom.spec_decode.eagle_proposer.get_forward_context", return_value=fc):
        EagleProposer._step_warmup_inputs(proposer, 3, positions=staged)
    assert graph.buffer("positions", 3).tolist() == [[8, 9, 10]] * 3
