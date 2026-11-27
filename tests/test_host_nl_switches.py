"""HOST's nl1..nl4 runtime switches (`atom.model_engine.host_nl`, CPU): each piece is on by default, ``ENABLED`` turns
them all off, and the off path of each switch point is the base's (rF13b's) call."""

import types

import pytest
import torch

from atom.model_engine import host_nl

PIECES = [
    "LM_HEAD_IN_GRAPH",
    "HIDDEN_FROM_GRAPH",
    "NO_DECODE_PLAN",
    "EMBED_FULL_TABLE",
    "DEFER_D2H",
    "VERIFY_NEXT_IDS",
    "PLAN_NO_REFRESH",
    "VERIFY_GREEDY_NO_NOISE",
]


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    monkeypatch.setattr(host_nl, "ENABLED", True)
    for p in PIECES:
        monkeypatch.setattr(host_nl, p, True)


def test_on_by_default_and_master_switch():
    assert all(host_nl.on(p) for p in PIECES)
    host_nl.ENABLED = False
    assert not any(host_nl.on(p) for p in PIECES)


@pytest.mark.parametrize("piece", PIECES)
def test_each_piece_alone(piece):
    setattr(host_nl, piece, False)
    assert [host_nl.on(p) for p in PIECES] == [p != piece for p in PIECES]


@pytest.mark.parametrize("on", [True, False])
def test_embedding_lookup_follows_the_switch(monkeypatch, on):
    eh = pytest.importorskip("atom.model_ops.embed_head", exc_type=ImportError)
    calls = []
    monkeypatch.setattr(eh, "replicated_embedding", lambda x, t: calls.append("full") or "full")
    monkeypatch.setattr(eh, "masked_embedding", lambda x, w, a, b: calls.append(("shard", a, b)) or "shard")
    group = types.SimpleNamespace(all_reduce=lambda y, ca_fp8_quant=False: calls.append(("ar", y)) or "reduced")
    monkeypatch.setattr(eh, "get_tp_group", lambda: group)
    host_nl.EMBED_FULL_TABLE = on
    fn = getattr(eh.vocab_lookup, "__wrapped__", None) or eh.vocab_lookup
    out = fn(torch.zeros(4, dtype=torch.int64), torch.zeros(2, 3), torch.zeros(8, 3), 2, 4)
    if on:
        assert calls == ["full"] and out == "full"
    else:  # rF13b's: the shard's rows, then the all-reduce
        assert calls == [("shard", 2, 4), ("ar", "shard")] and out == "reduced"


def test_dense_decode_metadata_is_built_with_the_switch_off():
    aa = pytest.importorskip("atom.model_ops.attentions.aiter_attention", exc_type=ImportError)
    host_nl.NO_DECODE_PLAN = False
    class Builder:  # no attributes: the switch answers before any is read
        pass

    assert aa.AiterAttentionMetadataBuilder._dense_decode_metadata_read(Builder()) is True


@pytest.mark.parametrize("next_ids, greedy_no_noise", [(True, True), (False, False), (True, False)])
def test_verify_launch_takes_the_switches(monkeypatch, next_ids, greedy_no_noise):
    svf = pytest.importorskip("atom.model_ops.spec_verify_fused", exc_type=ImportError)
    seen = {}

    def launch(grid, *args, **kw):
        seen.update(kw)
        seen["next_arg"] = args[12]

    monkeypatch.setattr(svf, "_launch", launch)
    monkeypatch.setattr(svf, "_state", lambda d: {"keys": None, "counter": None, "seed": None, "calls": 0})
    logits = torch.zeros(16, 64)
    out, verdict, nxt = svf.spec_verify(
        logits, torch.zeros(1), torch.zeros(7, dtype=torch.int32), torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32), torch.zeros(9, dtype=torch.int32), 7, greedy_eps=1e-5,
        next_ids=next_ids, greedy_no_noise=greedy_no_noise,
    )
    assert seen["NEXT_IDS"] is next_ids and seen["GREEDY_NO_NOISE"] is greedy_no_noise
    assert (nxt is not None) == next_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the verify kernel needs a GPU")
@pytest.mark.parametrize("forced", [False, True])
def test_verify_kernel_off_variant_matches_on_greedy_rows(forced):
    """rF13b's kernel variant (noise on every row, no next ids) gives the same ids and verdict as nl2 / nl4's on
    greedy rows (the noise is dropped there either way)."""
    svf = pytest.importorskip("atom.model_ops.spec_verify_fused", exc_type=ImportError)
    k, vocab, bs = 7, 4096 * 3 + 17, 3
    g = torch.Generator(device="cuda").manual_seed(7 + int(forced))
    logits = torch.randn(bs * (k + 1), vocab, device="cuda", generator=g).to(torch.bfloat16)
    best = logits.float().argmax(-1).to(torch.int32)
    input_ids = torch.zeros(bs * (k + 1) + 1, dtype=torch.int32, device="cuda")
    target_idx = torch.tensor([b * (k + 1) + p for b in range(bs) for p in range(k)], dtype=torch.int32, device="cuda")
    input_ids[target_idx.long() + 1] = best[target_idx.long()]
    bonus_idx = torch.tensor([b * (k + 1) + k for b in range(bs)], dtype=torch.int32, device="cuda")
    cu = torch.arange(1, bs + 1, dtype=torch.int32, device="cuda") * k
    temps = torch.zeros(bs, dtype=torch.float32, device="cuda")
    cond = torch.full((k,), 0.6, dtype=torch.float32, device="cuda") if forced else None
    on = svf.spec_verify(logits, temps, target_idx, bonus_idx, cu, input_ids, k, greedy_eps=1e-5, cond_rates=cond)
    st = svf._state(logits.device)
    st["calls"] -= 1  # the same seed for the off call (forced acceptance draws from it)
    off = svf.spec_verify(logits, temps, target_idx, bonus_idx, cu, input_ids, k, greedy_eps=1e-5, cond_rates=cond,
                          next_ids=False, greedy_no_noise=False)
    assert torch.equal(on[0], off[0]) and torch.equal(on[1], off[1]) and off[2] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the verify kernel needs a GPU")
def test_verify_kernel_off_variant_matches_on_sampled_rows_too():
    """nl4 is bit-exact on sampling rows as well: they draw the same Philox counters from the same seed in both
    variants; only greedy rows stop computing a noise they drop anyway."""
    svf = pytest.importorskip("atom.model_ops.spec_verify_fused", exc_type=ImportError)
    k, vocab, bs = 7, 4096 * 3 + 17, 4
    g = torch.Generator(device="cuda").manual_seed(29)
    logits = torch.randn(bs * (k + 1), vocab, device="cuda", generator=g).to(torch.bfloat16)
    best = logits.float().argmax(-1).to(torch.int32)
    input_ids = torch.zeros(bs * (k + 1) + 1, dtype=torch.int32, device="cuda")
    target_idx = torch.tensor([b * (k + 1) + p for b in range(bs) for p in range(k)], dtype=torch.int32, device="cuda")
    input_ids[target_idx.long() + 1] = best[target_idx.long()]
    bonus_idx = torch.tensor([b * (k + 1) + k for b in range(bs)], dtype=torch.int32, device="cuda")
    cu = torch.arange(1, bs + 1, dtype=torch.int32, device="cuda") * k
    temps = torch.tensor([0.0, 0.8, 1.3, 0.0], dtype=torch.float32, device="cuda")  # greedy and sampling rows
    on = svf.spec_verify(logits, temps, target_idx, bonus_idx, cu, input_ids, k, greedy_eps=1e-5)
    svf._state(logits.device)["calls"] -= 1  # the same seed
    off = svf.spec_verify(logits, temps, target_idx, bonus_idx, cu, input_ids, k, greedy_eps=1e-5,
                          next_ids=False, greedy_no_noise=False)
    assert torch.equal(on[0], off[0]) and torch.equal(on[1], off[1])
