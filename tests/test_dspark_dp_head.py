# SPDX-License-Identifier: MIT
"""The DSpark draft head split over the DP group: the gate, and its answer.

The sampler's answer is checked against the replicated head on CPU, four DP
ranks run as threads over an in-process group: each rank's ids and the
Markov embeds its confidence head saw must be the whole-vocab ones for its
own rows, padded or not. The Triton op is stood in for by its contract (the
per-shard ``(max, id)`` reduced across the group, lowest id on ties) -- its
own equivalence to the whole-vocab op is ``test_dspark_markov_argmax``'s.
"""

import threading
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="the module imports the AITER Markov op")

from atom import plugin_host
from atom.model_ops import dspark_dp_head
from atom.plugin_recipe import DSparkOptions


class _Exchange:
    """A rendezvous the threads gather through, rank order."""

    def __init__(self, world):
        self.world = world
        self.barrier = threading.Barrier(world)
        self.slots = [None] * world

    def gather(self, rank, item):
        self.barrier.wait()
        self.slots[rank] = item
        self.barrier.wait()
        out = list(self.slots)
        self.barrier.wait()
        return out


class _Group:
    def __init__(self, exchange, rank):
        self.world_size = exchange.world
        self.rank_in_group = rank
        self._x = exchange
        self.device_communicator = SimpleNamespace(all_gatherv=self._all_gatherv)
        self.gathers = 0

    def all_gather(self, t, dim=0, use_custom=False):
        assert dim == 0
        self.gathers += 1
        return torch.cat(self._x.gather(self.rank_in_group, t.clone()))

    def _all_gatherv(self, ts, dim=0, sizes=None):
        assert dim == 0 and sizes is None
        self.gathers += 1
        got = self._x.gather(self.rank_in_group, [t.clone() for t in ts])
        return [torch.cat([g[i] for g in got]) for i in range(len(ts))]


def _markov_tp(base, prev, w1, w2, out, vocab_start, group, use_custom, embed_out):
    """``dspark_markov_argmax_tp``'s contract in torch."""
    emb = w1[prev]
    shard = base.shape[1]
    biased = (
        base.double() + emb.double() @ w2[vocab_start : vocab_start + shard].double().T
    )
    idx = biased.argmax(dim=1)
    val = biased.gather(1, idx[:, None]).squeeze(1)
    packed = torch.stack([val, (idx + vocab_start).double()], dim=1)
    every = group.all_gather(packed, dim=0).view(group.world_size, -1, 2)
    winner = every[:, :, 0].argmax(dim=0)
    out.copy_(every[:, :, 1].gather(0, winner[None]).squeeze(0).to(out.dtype))
    embed_out.copy_(emb)
    return emb


def _mm(x, w):
    return (x.double() @ w.double().T).to(x.dtype)


def _confidence(h, e):
    # Reads both operands row by row, so a wrong embed row shows.
    return h.sum(-1) + 1000.0 * e.sum(-1)


def _reference(normed, hc_hidden, anchors, W, w1, w2):
    """The replicated head: full logits, whole-vocab Markov argmax."""
    B, T, _ = hc_hidden.shape
    base = _mm(normed, W).view(B, T, -1)
    ids = anchors.new_empty(B, T)
    embeds = []
    for k in range(T):
        prev = anchors if k == 0 else ids[:, k - 1]
        emb = w1[prev]
        ids[:, k] = (base[:, k].double() + emb.double() @ w2.double().T).argmax(dim=1)
        embeds.append(emb)
    return ids, _confidence(hc_hidden, torch.stack(embeds, 1))


@pytest.mark.parametrize("rows_per_rank,padded", [((3, 3, 3, 3), 3), ((2, 4, 1, 4), 4)])
def test_each_rank_gets_the_replicated_heads_answer_for_its_rows(
    monkeypatch, rows_per_rank, padded
):
    monkeypatch.setattr(dspark_dp_head, "dspark_markov_argmax_tp", _markov_tp)
    world, T, dim, V, r = 4, 5, 16, 64, 8
    torch.manual_seed(7)
    # Small integers: every product and sum is exact in fp32, so a mismatch
    # is a wrong row or column, never rounding.
    W = torch.randint(-3, 4, (V, dim)).float()
    w1 = torch.randint(-2, 3, (V, r)).float()
    w2 = torch.randint(-2, 3, (V, r)).float()
    # a tie across shards: the same column twice -> the lower id must win
    W[50] = W[5]
    w2[50] = w2[5]
    markov = SimpleNamespace(
        markov_w1=SimpleNamespace(weight=w1), markov_w2=SimpleNamespace(weight=w2)
    )
    inputs = []
    for B in rows_per_rank:
        hc = torch.randint(-3, 4, (B, T, dim)).float()
        inputs.append((hc.flatten(0, 1).clone(), hc, torch.randint(0, V, (B,))))
    exchange = _Exchange(world)
    results, errors = [None] * world, []

    def run(rank):
        try:
            normed, hc, anchors = inputs[rank]
            results[rank] = dspark_dp_head.dp_sharded_dspark_head_and_sample(
                normed,
                hc,
                anchors,
                W,
                markov,
                _confidence,
                _Group(exchange, rank),
                padded,
                mm=_mm,
            )
        except Exception as e:  # noqa: BLE001 -- surfaced below, not lost in the thread
            errors.append(e)
            exchange.barrier.abort()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for rank, (normed, hc, anchors) in enumerate(inputs):
        ids, conf = results[rank]
        ref_ids, ref_conf = _reference(normed, hc, anchors, W, w1, w2)
        assert torch.equal(ids, ref_ids), rank
        assert torch.equal(conf, ref_conf), rank


def _gate_inputs(**over):
    head = SimpleNamespace(tp_size=1, bias=None, num_embeddings=129280)
    markov = SimpleNamespace(
        fused_sample=True,
        markov_w1=SimpleNamespace(weight=None),
        markov_w2=SimpleNamespace(weight=None),
    )
    context = SimpleNamespace(
        is_draft=True,
        is_prefill=False,
        running_tokens_are_unified=True,
        running_tokens=120,
    )
    group = SimpleNamespace(world_size=4)
    args = {
        "head": head,
        "markov_head": markov,
        "context": context,
        "dp_metadata": object(),
        "dp_group": group,
    }
    args.update(over)
    return args


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setattr(dspark_dp_head, "markov_weights_unsupported", lambda *_: None)
    monkeypatch.setattr(
        plugin_host,
        "_plan",
        SimpleNamespace(dspark_options=DSparkOptions(dp_sharded_head=True)),
    )


def test_the_gate_opens_on_a_unified_pure_dp_draft(gate_on):
    assert dspark_dp_head.can_use_dp_sharded_dspark_head(**_gate_inputs())


def test_the_gate_is_off_by_default(monkeypatch):
    monkeypatch.setattr(dspark_dp_head, "markov_weights_unsupported", lambda *_: None)
    monkeypatch.setattr(plugin_host, "_plan", None)
    assert not DSparkOptions().dp_sharded_head
    assert not dspark_dp_head.can_use_dp_sharded_dspark_head(**_gate_inputs())


def test_the_row_bound_is_the_recipes(gate_on, monkeypatch):
    monkeypatch.setattr(
        plugin_host,
        "_plan",
        SimpleNamespace(
            dspark_options=DSparkOptions(
                dp_sharded_head=True, dp_sharded_head_max_rows=64
            )
        ),
    )
    assert not dspark_dp_head.can_use_dp_sharded_dspark_head(**_gate_inputs())


def _ctx(**over):
    c = _gate_inputs()["context"]
    for k, v in over.items():
        setattr(c, k, v)
    return c


@pytest.mark.parametrize(
    "over",
    [
        {"head": None},
        {"head": SimpleNamespace(tp_size=2, bias=None, num_embeddings=129280)},
        {"head": SimpleNamespace(tp_size=1, bias=object(), num_embeddings=129280)},
        {"head": SimpleNamespace(tp_size=1, bias=None, num_embeddings=129281)},
        {"markov_head": SimpleNamespace(fused_sample=False)},
        {"context": None},
        {"dp_metadata": None},
        {"context": _ctx(is_draft=False)},
        {"context": _ctx(is_prefill=True)},
        {"context": _ctx(running_tokens_are_unified=False)},
        {"context": _ctx(running_tokens=1025)},
        {"dp_group": SimpleNamespace(world_size=1)},
    ],
)
def test_the_gate_stays_shut_off_a_dp_agreed_pure_dp_draft(gate_on, over):
    assert not dspark_dp_head.can_use_dp_sharded_dspark_head(**_gate_inputs(**over))


def test_the_gate_shuts_on_unsupported_markov_weights(gate_on, monkeypatch):
    monkeypatch.setattr(
        dspark_dp_head, "markov_weights_unsupported", lambda *_: "not bf16"
    )
    assert not dspark_dp_head.can_use_dp_sharded_dspark_head(**_gate_inputs())
