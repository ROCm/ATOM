# SPDX-License-Identifier: MIT
"""MiniMax-M3.1 DSpark ``ATOM_DSPARK_SHARDED_MARKOV``: the block sampled off each
rank's vocab shard gives the ids and confidences of the whole-vocab path.

The TP ranks run as threads that meet in the all-gather. The fused op is
replaced by its contract in torch (this shard's ``(max, id)``, then the lowest
rank among equal maxima); its own bit-exactness against the whole-vocab op is
``test_dspark_markov_sample.py``'s. What this checks is the draft's plumbing:
the LM-head shard and its id offset, each position reading the id the previous
one wrote, and the Markov rows reaching the confidence head in block order.
Small integer weights keep every logit exact, so the many ties exercise the
lowest-id tie-break too.
"""

import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.test_minimax_m31_dspark_context import _m3_modules

HIDDEN, VOCAB, RANK, T = 64, 64, 8, 7


@pytest.fixture(scope="module")
def m3():
    with _m3_modules() as load:
        yield load("atom.models.minimax_m3.dspark")


class _Ranks:
    """TP ranks as threads: ``all_gather`` stacks every rank's tensor along ``dim``."""

    def __init__(self, tp):
        self.world_size = tp
        self._barrier = threading.Barrier(tp)
        self._slots = [None] * tp
        self.local = threading.local()

    def all_gather(self, x, dim=0, use_custom=True):
        self._slots[self.local.rank] = x.clone()
        self._barrier.wait()
        out = torch.cat(self._slots, dim=dim)
        self._barrier.wait()
        return out


def _argmax_tp(base_logits, prev_ids, w1, w2, out, vocab_start, tp_group,
               use_custom=True, embed_out=None):  # fmt: skip
    """``dspark_markov_argmax_tp``'s contract in torch."""
    emb = w1[prev_ids]
    v = base_logits.shape[1]
    logits = (
        base_logits.float() + emb.float() @ w2[vocab_start : vocab_start + v].float().T
    )
    idx = logits.argmax(-1)
    packed = torch.stack(
        [logits.gather(1, idx[:, None])[:, 0], (idx + vocab_start).float()], -1
    )
    g = tp_group.all_gather(packed, dim=0, use_custom=use_custom).view(
        tp_group.world_size, -1, 2
    )
    best = g[..., 0].argmax(0)  # the lowest rank among equal maxima
    out.copy_(g[..., 1].gather(0, best[None])[0].to(out.dtype))
    embed_out.copy_(emb)


class _Head:
    """The target's LM head on one rank: its shard for the sharded path, the
    gathered whole-vocab logits when called."""

    def __init__(self, w, r, tp):
        per = w.shape[0] // tp
        self.full, self.weight, self.bias = w, w[r * per : (r + 1) * per], None
        self.tp_size, self.vocab_start_idx = tp, r * per

    def __call__(self, x):
        return F.linear(x, self.full)


def _draft(m3, head, fused_sample, w1, w2, conf):
    hf = SimpleNamespace(
        hidden_size=HIDDEN, dspark_noise_token_id=1, dspark_target_layer_ids=[0],
        dspark_markov_rank=RANK, vocab_size=VOCAB, rms_norm_eps=1e-6,
        num_hidden_layers=0, sliding_window=0, enable_confidence_head=True,
    )  # fmt: skip
    cfg = SimpleNamespace(hf_config=hf, decode_context_parallel_size=1, max_num_seqs=4,
                          kv_cache_block_size=16)  # fmt: skip
    model = m3.MiniMaxM3DSpark(cfg)
    model.lm_head = head
    model.markov_head.fused_sample = fused_sample
    model.markov_head.markov_w1.weight.data = w1
    model.markov_head.markov_w2.weight.data = w2
    model.confidence_head.load_state_dict(conf)
    return model


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("bs", [1, 3])
def test_sharded_markov_matches_whole_vocab(m3, monkeypatch, tp, bs):
    g = torch.Generator().manual_seed(tp * 10 + bs)

    def ints(*shape, lo=-1, hi=2):
        return torch.randint(lo, hi, shape, generator=g).to(torch.bfloat16)

    w, w1, w2 = ints(VOCAB, HIDDEN), ints(VOCAB, RANK, lo=-3, hi=4), ints(VOCAB, RANK)
    hidden, anchors = ints(bs * T, HIDDEN), torch.randint(0, VOCAB, (bs,), generator=g)
    conf = {"proj.weight": torch.randn(1, HIDDEN + RANK, generator=g),
            "proj.bias": torch.randn(1, generator=g)}  # fmt: skip

    monkeypatch.delenv("ATOM_DSPARK_SHARDED_MARKOV", raising=False)
    ref = _draft(m3, _Head(w, 0, tp), False, w1, w2, conf)
    assert not ref._sharded_markov_sample()
    ids_ref, conf_ref = ref.head_and_sample(hidden, anchors, T)

    monkeypatch.setenv("ATOM_DSPARK_SHARDED_MARKOV", "1")
    group = _Ranks(tp)
    stub = ModuleType("atom.model_ops.dspark_markov_sample")
    stub.dspark_markov_argmax_tp = _argmax_tp
    monkeypatch.setitem(sys.modules, "atom.model_ops.dspark_markov_sample", stub)
    monkeypatch.setattr(m3, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        m3, "tgemm", SimpleNamespace(mm=lambda x, w_, *a, **k: F.linear(x, w_))
    )
    models = [_draft(m3, _Head(w, r, tp), True, w1, w2, conf) for r in range(tp)]
    assert all(m._sharded_markov_sample() for m in models)
    got, errors = [None] * tp, []

    def run(r):
        group.local.rank = r
        try:
            got[r] = models[r].head_and_sample(hidden, anchors, T)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            group._barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(tp)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for r, (ids, c) in enumerate(got):
        assert torch.equal(ids, ids_ref), f"rank {r}: ids differ"
        assert torch.equal(c, conf_ref), f"rank {r}: confidence differs"


def test_tp1_keeps_the_whole_vocab_path(m3, monkeypatch):
    monkeypatch.setenv("ATOM_DSPARK_SHARDED_MARKOV", "1")
    z = torch.zeros(VOCAB, RANK, dtype=torch.bfloat16)
    conf = {"proj.weight": torch.zeros(1, HIDDEN + RANK), "proj.bias": torch.zeros(1)}
    model = _draft(m3, _Head(torch.zeros(VOCAB, HIDDEN), 0, 1), True, z, z, conf)
    assert not model._sharded_markov_sample()
