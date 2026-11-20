# SPDX-License-Identifier: MIT
"""Independent draft masks, ragged positions and the draft heads' checkpoint names.

Every import here is plain torch, so a CPU-only runner checks all of it. The
one case that needed a Triton kernel is `test_draft_attention.py`.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.deepseek_v41.draft_block import draft_step, rotate_rows
from atom.models.deepseek_v4_dspark import DSparkConfidenceHead, DSparkMarkovHead


def _rope():
    """The cached RoPE is an AITER kernel; the block mask above is not."""
    pytest.importorskip("aiter", reason="the cached RoPE is an AITER kernel")
    from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding

    return RotaryEmbedding


def _draft_model():
    """The draft model builds ATOM layers, which reach AITER."""
    pytest.importorskip("aiter", reason="the draft model builds AITER-backed layers")
    from atom.models.deepseek_v41.dspark import DeepseekV41DSpark

    return DeepseekV41DSpark


def test_block_mask_keeps_all_draft_rows_and_only_the_visible_window():
    positions = torch.tensor([[-1, 0, 1, 2, 3, 4], [125, 126, 127, 128, 129, 130]])
    step = draft_step(positions, torch.tensor([2, 129]), 5, 4)
    expected = torch.tensor(
        [
            [-1, 1, 2, 3, -1, -1, 6, 7, 8, 9, 10],
            [-1, 1, 2, 3, 4, -1, 6, 7, 8, 9, 10],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(step.indices, expected[:, None].expand(-1, 5, -1))
    assert torch.equal(
        step.positions, torch.tensor([[3, 4, 5, 6, 7], [130, 131, 132, 133, 134]])
    )


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rope_ragged_request_positions(device, inverse):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    torch.manual_seed(711)
    rope = _rope()(64, 256, base=10000).to(device)
    positions = torch.tensor(
        [[2, 3, 4, 5, 6], [126, 127, 128, 129, 130]], device=device
    )
    hidden = torch.randn(2, 5, 8, 512, dtype=torch.bfloat16, device=device)
    expected = torch.cat(
        [
            rope(hidden[i : i + 1].clone(), positions[i], inverse=inverse)
            for i in range(2)
        ]
    )
    actual = rotate_rows(rope, hidden.clone(), positions, inverse=inverse)
    assert torch.equal(actual, expected)


def test_draft_head_names_and_width_match_the_v41_checkpoint():
    """What belongs to V4.1 here is the naming, not the arithmetic.

    The Markov tables ship under this checkpoint's own names while the shared
    V4 head calls them `markov_w1` / `markov_w2`, so the rename is checked
    against the destination's real parameters rather than against a copy of
    itself. The confidence projection consumes the concatenation, which is what
    makes the checkpoint's row `[1, hidden + rank]` wide. Both heads' math is
    V4's and is covered by `tests/test_dspark.py`.
    """
    hidden, rank, vocab = 64, 32, 128
    renamed = _draft_model().weights_mapper.apply_list(
        [f"mtp.2.markov_head.{name}.weight" for name in ("embed", "head")]
    )
    assert renamed == [
        "mtp.2.markov_head.markov_w1.weight",
        "mtp.2.markov_head.markov_w2.weight",
    ]
    markov = DSparkMarkovHead(vocab, rank)
    assert {f"mtp.2.markov_head.{name}" for name in markov.state_dict()} == set(renamed)
    assert DSparkConfidenceHead(hidden, rank).proj.weight.shape == (1, hidden + rank)


def test_the_input_tap_pays_the_post_the_block_ahead_of_it_still_owes():
    """A tap is a pre-hook, so it sees the state before the block folds.

    Since the FFN post became owed rather than applied, `residual` at a block
    boundary is by definition the value before it -- one whole sublayer behind
    what the draft is meant to condition on. Nothing about the shape gives that
    away, so the feature would just be quietly wrong and cost acceptance.

    The owed post here is the identity combination over four streams of ones
    with an output of twos, which settles to threes: a tap reading raw sees
    ones, and no tolerance hides the difference. The second state is the
    control -- with nothing owed, settling must not move the value, or this is
    testing `settle` rather than the tap.
    """
    draft = _draft_model()
    from atom.model_ops.deepseek_v41.mhc import SinglePassHCState

    tokens, hc, hidden = 2, 4, 8
    pre_mix = torch.zeros(tokens, hc, dtype=torch.float32)
    pre_mix[..., 0] = 1
    owed = SinglePassHCState(
        torch.ones(tokens, hc, hidden),
        pre_mix,
        torch.full((tokens, hidden), 2.0),
        torch.ones(tokens, hc, dtype=torch.float32),
        torch.eye(hc, dtype=torch.float32).expand(tokens, hc, hc).contiguous(),
    )
    settled = SinglePassHCState(torch.ones(tokens, hc, hidden), pre_mix)

    assert torch.equal(
        draft._target_layer_input((owed,), None), torch.full((tokens, hidden), 3.0)
    )
    assert torch.equal(
        draft._target_layer_input((settled,), None), torch.ones(tokens, hidden)
    )


@pytest.mark.parametrize("runs_bf16", [True, False])
def test_project_context_projects_only_the_kv_rows_of_a_bf16_wqkv_a(runs_bf16):
    """A BF16 `wqkv_a` projects only its KV rows; an FP8 one projects both
    halves and drops the query."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    pytest.importorskip("aiter", reason="the draft attention builds AITER layers")
    from atom.models.deepseek_v41.dspark import DraftAttention

    attention = object.__new__(DraftAttention)
    torch.nn.Module.__init__(attention)
    hidden, kv = torch.randn(3, 8), torch.randn(3, 4)
    attention.wqkv_a = SimpleNamespace(runs_bf16=runs_bf16)
    attention._wkv_shard = Mock(return_value=kv)
    attention.project_qkv = Mock(return_value=(torch.randn(3, 2), kv))
    attention.kv_norm = lambda x: 2 * x
    attention.context_keys = Mock(return_value="keys")
    assert attention.project_context(hidden, "positions", "rope") == "keys"
    assert attention._wkv_shard.call_count == int(runs_bf16)
    assert attention.project_qkv.call_count == int(not runs_bf16)
    assert torch.equal(attention.context_keys.call_args.args[0], 2 * kv)


def test_a_bf16_main_proj_takes_the_joined_taps_unquantized():
    """Where `main_proj` runs in BF16 (no group32 FP8 GEMM), the taps reach it
    joined and unquantized."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    model_cls = _draft_model()
    model = object.__new__(model_cls)
    torch.nn.Module.__init__(model)
    main_proj = Mock(runs_bf16=True, return_value="projected")
    model.mtp = [SimpleNamespace(main_proj=main_proj)]
    taps = [torch.randn(3, 4, dtype=torch.bfloat16) for _ in range(3)]
    assert model._project_taps(taps) == "projected"
    (x,), kwargs = main_proj.call_args
    assert torch.equal(x, torch.cat(taps, dim=-1)[None]) and not kwargs


def test_lossless_sampling_arms_with_block_verification():
    """V4.1 samples its block through the bound `draft_sampling`, so lossless
    sampling may arm, and verifies a sampled request's drafts as one block."""
    cls = _draft_model()
    assert cls.supports_lossless_draft and cls.lossless_block_verify


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("bound", [False, True])
def test_the_block_head_takes_the_sampled_block_only_when_bound(bound, greedy, sharded):
    """The sampled block where `draft_sampling` is bound and the pass is not
    the greedy capture; otherwise the greedy block, sharded or not."""
    cls = _draft_model()
    model = SimpleNamespace(
        draft_sampling="bound" if bound else None,
        _sharded_markov_sample=lambda: sharded,
        _head_and_sample_q=lambda *a: "sampled",
        _head_and_sample_sharded=lambda *a: "greedy-sharded",
        _head_and_sample=lambda *a: "greedy",
    )
    got = cls.head_and_sample(model, ("normed", "hc"), "anchors", 5, greedy=greedy)
    if bound and not greedy:
        assert got == "sampled"
    else:
        assert got == ("greedy-sharded" if sharded else "greedy")


@pytest.mark.parametrize("sharded", [False, True])
def test_the_sampled_block_feeds_the_confidence_head_its_w1_rows(monkeypatch, sharded):
    """The sampled op hands back ids only; the confidence head still reads
    W1[x_{k-1}] of the anchor and the drafts before each position. Sharded,
    the block keeps this rank's slice of the head and names its shard."""
    cls = _draft_model()
    from atom.models.deepseek_v41 import dspark as v41_dspark

    B, T, V, r, dim = 2, 3, 16, 4, 8
    markov = DSparkMarkovHead(V, r)
    seen = {}

    def sample_block(base_logits, anchors, sampling, shard=None):
        seen["block"] = (tuple(base_logits.shape), sampling, shard)
        return torch.tensor([[1, 2, 3], [4, 5, 6]])

    def confidence(hidden, embeds):
        seen["embeds"] = embeds
        return torch.zeros(B, T)

    monkeypatch.setattr(markov, "sample_block", sample_block)
    group = SimpleNamespace(world_size=2)
    monkeypatch.setattr(v41_dspark, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        v41_dspark.tgemm, "mm", lambda x, w: torch.zeros(x.shape[0], w.shape[0])
    )
    head = SimpleNamespace(
        get_logits=lambda x: torch.zeros(x.shape[0], V),
        weight=torch.zeros(V // 2, dim),
        vocab_start_idx=V // 2,
    )
    model = SimpleNamespace(
        head=head,
        mtp=[SimpleNamespace(markov_head=markov, confidence_head=confidence)],
        draft_sampling="bound",
        _sharded_markov_sample=lambda: sharded,
    )
    ids, _ = cls._head_and_sample_q(
        model, torch.zeros(B * T, dim), torch.zeros(B, T, dim), torch.tensor([7, 8])
    )
    assert torch.equal(ids, torch.tensor([[1, 2, 3], [4, 5, 6]]))
    shape, sampling, shard = seen["block"]
    assert sampling == "bound"
    if sharded:
        assert shape == (B, T, V // 2)
        assert (shard.start, shard.group) == (V // 2, group)
    else:
        assert shape == (B, T, V) and shard is None
    prev = torch.tensor([[7, 1, 2], [8, 4, 5]])
    assert torch.equal(seen["embeds"], markov.markov_w1(prev))
