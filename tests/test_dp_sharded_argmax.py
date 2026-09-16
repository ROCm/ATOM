"""CPU-only coverage for the DP-vocab-sharded draft argmax.

Two things need no GPU and no real process group:
  - the gate `_can_use_dp_sharded_argmax` is pure predicate logic, and must be
    TOTAL (return False, never raise) outside a DP-reduced pure-DP draft;
  - the argmax exchange's offset arithmetic (`start = dp_rank * max_rows` and the
    rank-major `.view(dp_size, dp_size * max_rows, 2)` unpack) must slice back
    exactly this rank's own rows.
"""

import pytest
import torch

embed_head = pytest.importorskip(
    "atom.model_ops.embed_head",
    reason="embed_head requires the Triton/AITER runtime",
    exc_type=ImportError,
)


class _Ctx:
    def __init__(self, *, is_draft=True, unified=True, running_tokens=4):
        self.is_draft = is_draft
        self.running_tokens_are_unified = unified
        self.running_tokens = running_tokens


class _FwdCtx:
    def __init__(self, *, dp_metadata, context):
        self.dp_metadata = dp_metadata
        self.context = context


class _Group:
    def __init__(self, *, world_size, rank_in_group=0, all_gather=None):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.device_group = object()
        self._all_gather = all_gather

    def all_gather(self, x, dim=0, use_custom=False):
        return self._all_gather(x, dim)


def _head(*, tp_size=1, num_embeddings=8):
    head = embed_head.ParallelLMHead.__new__(embed_head.ParallelLMHead)
    torch.nn.Module.__init__(head)
    head.tp_size = tp_size
    head.num_embeddings = num_embeddings
    head.weight = torch.nn.Parameter(torch.empty(1), requires_grad=False)
    head.bias = None
    return head


# --------------------------------------------------------------------------- #
# Gate verdict table
# --------------------------------------------------------------------------- #
_PRESENT = object()  # a non-None dp_metadata stand-in (a DP-reduced step)


def _install_gate_env(
    monkeypatch,
    *,
    flag=True,
    plugin=False,
    world_size=2,
    dp_metadata=_PRESENT,
    context=None,
):
    monkeypatch.setattr(embed_head.envs, "ATOM_DP_DRAFT_ARGMAX", flag)
    monkeypatch.setattr(embed_head.envs, "ATOM_DP_DRAFT_ARGMAX_MAX_ROWS", 256)
    monkeypatch.setattr(embed_head, "is_plugin_mode", lambda: plugin)
    monkeypatch.setattr(
        embed_head, "get_dp_group", lambda: _Group(world_size=world_size)
    )
    monkeypatch.setattr(
        embed_head,
        "get_forward_context",
        lambda: _FwdCtx(dp_metadata=dp_metadata, context=context),
    )


@pytest.mark.parametrize(
    "kwargs, ctx_kwargs, head_kwargs, expected",
    [
        # everything aligned -> engage
        ({}, {}, {}, True),
        # env off
        ({"flag": False}, {}, {}, False),
        # plugin mode: caller decides collective counts -> never engage
        ({"plugin": True}, {}, {}, False),
        # hybrid TP present -> not pure DP
        ({}, {}, {"tp_size": 2}, False),
        # SGLang dp-attention / single-GPU: running_tokens was never reduced
        ({"dp_metadata": None}, {}, {}, False),
        # not a draft step
        ({}, {"is_draft": False}, {}, False),
        # ragged (variable-length) draft
        ({}, {"unified": False}, {}, False),
        # past the weight-read-bound crossover
        ({}, {"running_tokens": 257}, {}, False),
        # no DP (world_size == 1)
        ({"world_size": 1}, {}, {}, False),
        # vocab not evenly shardable
        ({"world_size": 3}, {}, {"num_embeddings": 8}, False),
    ],
)
def test_gate_verdict_table(monkeypatch, kwargs, ctx_kwargs, head_kwargs, expected):
    ctx = _Ctx(**ctx_kwargs)
    _install_gate_env(monkeypatch, context=ctx, **kwargs)
    head = _head(**head_kwargs)
    assert head._can_use_dp_sharded_argmax(ctx) is expected


def test_gate_is_total_when_dp_group_absent(monkeypatch):
    """dp_metadata is None must short-circuit BEFORE get_dp_group(), which raises
    when the DP group was never built (single-GPU / unit test / plugin)."""

    def _raises():
        raise AssertionError("data parallel group is not initialized")

    monkeypatch.setattr(embed_head.envs, "ATOM_DP_DRAFT_ARGMAX", True)
    monkeypatch.setattr(embed_head, "is_plugin_mode", lambda: False)
    monkeypatch.setattr(embed_head, "get_dp_group", _raises)
    monkeypatch.setattr(
        embed_head,
        "get_forward_context",
        lambda: _FwdCtx(dp_metadata=None, context=_Ctx()),
    )
    # Returns False rather than propagating the AssertionError.
    assert (
        embed_head.ParallelLMHead._can_use_dp_sharded_argmax(_head(), _Ctx()) is False
    )


# --------------------------------------------------------------------------- #
# Offset arithmetic in the argmax exchange
# --------------------------------------------------------------------------- #
def test_argmax_exchange_slices_each_rank_own_rows(monkeypatch):
    """Every rank runs the sampler over EVERY rank's rows, then must return only
    its own `local_rows` block at `dp_rank * max_rows`. Drive `_dp_sharded_logits`
    for both ranks of a dp=2 step and check each gets the right ids."""
    dp_size, max_rows, local_rows, vshard = 2, 3, 2, 4
    V = dp_size * vshard

    # Reference: a full-vocab logits per row for the Σ = dp_size * max_rows rows.
    torch.manual_seed(0)
    full_logits = torch.randn(dp_size * max_rows, V)
    ref_ids = full_logits.argmax(dim=-1)  # [Σrows] the replicated answer

    # Each shard sees the same rows but only its V/dp columns; pack (max, gidx).
    def _shard_pack(shard):
        cols = slice(shard * vshard, (shard + 1) * vshard)
        val, loc = full_logits[:, cols].max(dim=-1)
        return torch.stack([val, (loc + shard * vshard).float()], dim=-1)

    packs = [_shard_pack(s) for s in range(dp_size)]  # each [Σrows, 2]

    monkeypatch.setattr(embed_head.envs, "ATOM_USE_CUSTOM_ALL_GATHER", False)
    # tgemm.mm / lm_head_argmax_pack are the GPU steps; on CPU we feed this rank's
    # own pack straight through, so only the exchange + offset code runs for real.
    monkeypatch.setattr(embed_head.tgemm, "mm", lambda *_a, **_k: None)

    for rank in range(dp_size):
        monkeypatch.setattr(
            embed_head, "lm_head_argmax_pack", lambda *_a, _r=rank, **_k: packs[_r]
        )

        def _all_gather(x, dim, _r=rank):
            if dim == 0 and x is packs[_r]:  # the pack gather
                return torch.cat(packs, dim=0)
            return x  # the hidden gather is unused (tgemm.mm is stubbed)

        monkeypatch.setattr(
            embed_head,
            "get_dp_group",
            lambda _r=rank: _Group(
                world_size=dp_size, rank_in_group=_r, all_gather=_all_gather
            ),
        )
        monkeypatch.setattr(
            embed_head,
            "get_forward_context",
            lambda: _FwdCtx(
                dp_metadata=object(), context=_Ctx(running_tokens=max_rows)
            ),
        )
        head = _head(num_embeddings=V)
        x = torch.empty(local_rows, 4)  # [local_rows, dim]; contents unused
        got = head._dp_sharded_logits(x, "argmax")

        start = rank * max_rows
        assert got.tolist() == ref_ids[start : start + local_rows].tolist()
