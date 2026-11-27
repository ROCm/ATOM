# SPDX-License-Identifier: MIT
"""Independent draft masks, ragged positions and the draft heads' checkpoint names.

Every import here is plain torch, so a CPU-only runner checks all of it. The
one case that needed a Triton kernel is `test_draft_attention.py`.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from atom.model_ops.deepseek_v41.draft_block import draft_step, rotate_rows
from atom.models.deepseek_v4_dspark import DSparkConfidenceHead, DSparkMarkovHead


def _rope():
    """The cached RoPE is an AITER kernel; the block mask above is not."""
    pytest.importorskip("aiter", reason="the cached RoPE is an AITER kernel")
    from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding

    return RotaryEmbedding


def _draft_module():
    """The draft model builds ATOM layers, which reach AITER."""
    pytest.importorskip("aiter", reason="the draft model builds AITER-backed layers")
    from atom.models.deepseek_v41 import dspark

    return dspark


def _draft_model():
    return _draft_module().DeepseekV41DSpark


def _draft_sharing(embed, noise_token_id=2):
    model = _draft_model()
    draft = model.__new__(model)
    torch.nn.Module.__init__(draft)
    draft.config = SimpleNamespace(dspark_noise_token_id=noise_token_id)
    draft.share_with_target(SimpleNamespace(embed=embed, head=torch.nn.Identity()))
    return draft


def _sharded_embedding(fill, dtype):
    """A rank's `VocabParallelEmbedding` (rows 2 and 3 of 4 over TP 2) whose
    module forward -- the custom all-reduce -- must never run."""
    from atom.model_ops.embed_head import VocabParallelEmbedding

    embed = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    torch.nn.Module.__init__(embed)
    embed.tp_rank, embed.tp_size = 1, 2
    embed.vocab_start_idx, embed.vocab_end_idx = 2, 4
    embed.weight = torch.nn.Parameter(
        torch.full((2, 8), fill, dtype=dtype), requires_grad=False
    )
    embed.forward = Mock(side_effect=AssertionError("custom all-reduce"))
    return embed


def test_noise_row_binds_once_and_rebinds_with_a_new_target(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    embed = torch.nn.Embedding.from_pretrained(torch.full((4, 8), 3.0))
    forward = Mock(wraps=embed.forward)
    monkeypatch.setattr(embed, "forward", forward)
    draft = _draft_sharing(embed)

    row = draft._noise_embedding()
    torch.testing.assert_close(row, torch.full((8,), 3.0))
    assert not row.requires_grad
    assert draft._noise_embedding() is row
    forward.assert_called_once()
    assert draft._embed_peer is None

    draft.share_with_target(
        SimpleNamespace(
            embed=torch.nn.Embedding.from_pretrained(torch.full((4, 8), 5.0)),
            head=torch.nn.Identity(),
        )
    )
    torch.testing.assert_close(draft._noise_embedding(), torch.full((8,), 5.0))


def test_noise_row_is_not_bound_under_graph_capture(monkeypatch):
    capturing = [True]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    draft = _draft_sharing(torch.nn.Embedding.from_pretrained(torch.ones(4, 8)))
    with pytest.raises(RuntimeError, match="not under graph capture"):
        draft._noise_embedding()
    capturing[0] = False
    torch.testing.assert_close(draft._noise_embedding(), torch.ones(8))


def test_sharded_noise_row_reads_the_owner_shard(monkeypatch):
    dspark = _draft_module()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    tp = SimpleNamespace(cpu_group=object(), device_group=object())
    monkeypatch.setattr(dspark, "get_tp_group", lambda: tp)
    embed = _sharded_embedding(4.0, torch.bfloat16)

    def read(ids, out):
        assert ids.tolist() == [3]
        return out.fill_(7.0)

    peer = Mock(side_effect=read)
    bind = Mock(return_value=peer)
    monkeypatch.setattr(dspark, "shared_peer_embedding", bind)
    draft = _draft_sharing(embed, noise_token_id=3)

    torch.testing.assert_close(
        draft._noise_embedding(), torch.full((8,), 7.0, dtype=torch.bfloat16)
    )
    bind.assert_called_once_with(embed, tp.cpu_group)
    assert draft._embed_peer is peer


@pytest.mark.parametrize("refusal", ["unmapped", "fp32"])
def test_sharded_noise_row_falls_back_to_nccl(monkeypatch, refusal):
    dspark = _draft_module()
    MonoUnsupported = dspark.MonoUnsupported

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    tp = SimpleNamespace(cpu_group=object(), device_group=object())
    monkeypatch.setattr(dspark, "get_tp_group", lambda: tp)
    monkeypatch.setattr(
        dspark,
        "shared_peer_embedding",
        Mock(side_effect=MonoUnsupported("no IPC")),
    )
    dtype = torch.float32 if refusal == "fp32" else torch.bfloat16
    embed = _sharded_embedding(4.0, dtype)

    def lookup(ids, weight, start, end):
        assert (start, end) == (2, 4)
        return weight[ids - start].clone()

    reduced = []
    monkeypatch.setattr(dspark, "masked_embedding", lookup)
    monkeypatch.setattr(
        dspark.dist, "all_reduce", lambda t, group: reduced.append(group)
    )
    draft = _draft_sharing(embed, noise_token_id=3)

    torch.testing.assert_close(
        draft._noise_embedding(), torch.full((8,), 4.0, dtype=dtype)
    )
    assert reduced == [tp.device_group]
    assert draft._embed_peer is None
    if refusal == "fp32":
        dspark.shared_peer_embedding.assert_not_called()


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
        _dp_markov_sample_pad=lambda *a: None,
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


@pytest.mark.parametrize("pad", [None, 0, 8])
def test_the_greedy_block_takes_the_dp_shard_where_a_pad_is_measured(pad):
    """TP sharding first; else the DP shard when its gate hands back a pad;
    else the replicated head. Never the DP shard for the sampled block."""
    cls = _draft_model()
    model = SimpleNamespace(
        draft_sampling=None,
        _sharded_markov_sample=lambda: False,
        _dp_markov_sample_pad=lambda *a: pad,
        _head_and_sample_dp=lambda *a: ("greedy-dp", a[-1]),
        _head_and_sample=lambda *a: "greedy",
    )
    got = cls.head_and_sample(model, ("normed", "hc"), "anchors", 5)
    assert got == ("greedy" if pad is None else ("greedy-dp", pad))
    model.draft_sampling = "bound"
    model._head_and_sample_q = lambda *a: "sampled"
    assert cls.head_and_sample(model, ("normed", "hc"), "anchors", 5) == "sampled"


class _Exchange:
    """An all-gather between threads standing in for DP ranks."""

    def __init__(self, world):
        import threading

        self.world = world
        self.slots = [None] * world
        self.barrier = threading.Barrier(world)

    def group(self, rank):
        exchange = self

        class Group:
            world_size = exchange.world
            rank_in_group = rank

            def all_gather(self, t, dim=0, use_custom=True):
                exchange.barrier.wait()
                exchange.slots[rank] = t.clone()
                exchange.barrier.wait()
                out = torch.cat(exchange.slots, dim)
                exchange.barrier.wait()
                return out

        return Group()


def _markov_argmax_shard(base, prev, w1, w2, out, start, group, **kwargs):
    """`dspark_markov_argmax_tp` in torch: this shard's (max, id), the lowest
    rank among equal maxima."""
    embed = w1[prev.long()]
    logits = (
        base.float() + embed.float() @ w2[start : start + base.shape[1]].float().t()
    )
    idx = logits.argmax(-1)
    val = logits.gather(1, idx[:, None])[:, 0]
    packed = torch.stack([val, (idx + start).float()], -1)
    pairs = group.all_gather(packed, dim=0).view(group.world_size, -1, 2)
    winner = pairs[:, :, 0].argmax(0)
    out.copy_(pairs[:, :, 1].gather(0, winner[None])[0].to(out.dtype))
    kwargs["embed_out"].copy_(embed)


@pytest.mark.parametrize("pad", [0, 8])
def test_the_dp_sharded_block_drafts_what_the_replicated_head_does(monkeypatch, pad):
    """Two DP ranks with their own rows: each one's DP-sharded block (one
    gather of rows + anchors, its vocab slice, the pick over both ranks) gives
    the ids and confidences its replicated head gives. Integer weights keep
    every sum exact, and the narrow value range makes ties common."""
    import functools
    import threading

    dspark = _draft_module()
    cls = dspark.DeepseekV41DSpark
    world, B, T, V, r, dim = 2, 3, 4, 16, 4, 8
    gen = torch.Generator().manual_seed(0)

    def ints(*shape):
        return torch.randint(-2, 3, shape, generator=gen).float()

    markov = DSparkMarkovHead(V, r)
    markov.fused_sample = False  # the replicated reference picks in torch
    markov.markov_w1.weight.data = ints(V, r)
    markov.markov_w2.weight.data = ints(V, r)
    weight = ints(V, dim)
    head = SimpleNamespace(
        num_embeddings=V,
        weight=weight,
        get_logits=lambda x: x @ weight.t(),
        _vocab_shard_logits=lambda x, rank, n: x
        @ weight[rank * V // n : (rank + 1) * V // n].t(),
    )

    def confidence(hidden, embeds):
        return hidden.sum(-1) + embeds.sum(-1)

    model = SimpleNamespace(
        head=head, mtp=[SimpleNamespace(markov_head=markov, confidence_head=confidence)]
    )
    model.forward_head = functools.partial(cls.forward_head, model)

    inputs = [
        (
            ints(B * T, dim),
            ints(B, T, dim),
            torch.randint(0, V, (B,), generator=gen).int(),
        )
        for _ in range(world)
    ]
    exchange = _Exchange(world)
    local = threading.local()
    monkeypatch.setattr(dspark, "get_dp_group", lambda: local.group)
    monkeypatch.setattr(dspark, "dspark_markov_argmax_tp", _markov_argmax_shard)
    got, errors = [None] * world, []

    def rank_main(rank):
        local.group = exchange.group(rank)
        try:
            got[rank] = cls._head_and_sample_dp(model, *inputs[rank], pad)
        except Exception as e:  # noqa: BLE001 -- surfaced below
            errors.append(e)
            exchange.barrier.abort()

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for rank in range(world):
        ref_ids, ref_conf = cls._head_and_sample(model, *inputs[rank])
        ids, conf = got[rank]
        assert torch.equal(ids, ref_ids.to(ids.dtype))
        assert torch.equal(conf, ref_conf)
