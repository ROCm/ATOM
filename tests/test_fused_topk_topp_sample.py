# SPDX-License-Identifier: MIT
"""ATOM_FUSED_TOPK_TOPP_SAMPLE: when the sampler hands its rows to aiter's
topk_topp_sample, and (on a GPU) that the tokens and the speculative verdict
equal the eager path's with a stable top-k."""

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sampler = pytest.importorskip("atom.model_ops.sampler", exc_type=ImportError)


def _stub(monkeypatch, name):
    """The module, or a stub of it when aiter is mocked or absent; its
    attributes are set through monkeypatch either way."""
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
    return module


def _fake_aiter(monkeypatch, small_k_topp=True):
    """aiter.ops.sampling's fast-path knobs and the fused wrapper, as stubs."""
    calls = []
    samp = _stub(monkeypatch, "aiter.ops.sampling")
    monkeypatch.setattr(samp, "_SMALL_K_TOPP", small_k_topp, raising=False)
    monkeypatch.setattr(samp, "_SMALL_K_TOPP_MAX_K", 64, raising=False)
    monkeypatch.setattr(samp, "_TOPK_FIRST_FAST_MIN_VOCAB", 65536, raising=False)
    fused = _stub(monkeypatch, "aiter.ops.triton.fusions.topk_topp_sample")

    def topk_topp_sample(logits, t, k, p, **kwargs):
        calls.append((logits.shape, k, p, kwargs))
        if kwargs.get("verify_rows") is not None:
            n = kwargs["verify_rows"].numel() + kwargs["bonus_rows"].numel()
        else:
            n = logits.shape[0]
        return torch.zeros(n, dtype=torch.int32)

    monkeypatch.setattr(
        fused, "topk_topp_sample_supported", lambda logits, k: True, raising=False
    )
    monkeypatch.setattr(fused, "topk_topp_sample", topk_topp_sample, raising=False)
    monkeypatch.setattr(sampler, "AITER_TOPK_TOPP_AVAILABLE", True)
    return calls


@pytest.mark.parametrize(
    "flag,top_k,top_p,small_k,vocab,expected",
    [
        ("1", 20, 0.95, True, 248320, True),
        ("0", 20, 0.95, True, 248320, False),  # default off
        ("1", 20, None, True, 248320, False),  # top-k only: renorm + multinomial
        ("1", None, 0.95, True, 248320, False),  # top-p only: full-vocab kernel
        ("1", 80, 0.95, True, 248320, False),  # k above the small-k draw
        ("1", 20, 0.95, False, 248320, False),  # generic top-p kernel
        ("1", 20, 0.95, True, 4096, False),  # vocab below the top-k-first path
        ("1", True, 0.95, True, 248320, False),
    ],
)
def test_applicable_only_on_the_top_k_first_small_k_path(
    monkeypatch, flag, top_k, top_p, small_k, vocab, expected
):
    _fake_aiter(monkeypatch, small_k)
    monkeypatch.setenv("ATOM_FUSED_TOPK_TOPP_SAMPLE", flag)
    logits = torch.zeros(2, vocab, dtype=torch.bfloat16)
    assert sampler.fused_topk_topp_applicable(logits, top_k, top_p) is expected


def test_tensor_filters(monkeypatch):
    _fake_aiter(monkeypatch)
    monkeypatch.setenv("ATOM_FUSED_TOPK_TOPP_SAMPLE", "1")
    logits = torch.zeros(3, 248320, dtype=torch.bfloat16)
    # a per-request top_p goes in; a per-request top_k (generic kernel) does not
    assert sampler.fused_topk_topp_applicable(logits, 20, torch.rand(3))
    assert not sampler.fused_topk_topp_applicable(logits, torch.full((3,), 20), 0.9)
    # a one-element top_p is read back by the eager path: stay there
    assert not sampler.fused_topk_topp_applicable(logits, 20, torch.rand(1))


def test_sampler_dispatch(monkeypatch):
    calls = _fake_aiter(monkeypatch)
    monkeypatch.setenv("ATOM_FUSED_TOPK_TOPP_SAMPLE", "1")
    logits = torch.zeros(4, 248320, dtype=torch.bfloat16)
    out = sampler.Sampler()(logits, torch.ones(4), 20, 0.95)
    assert out.shape == (4,) and len(calls) == 1
    assert calls[0][1:3] == (20, 0.95) and calls[0][3]["eps"] == sampler.SAMPLER_EPS


def test_verify_and_bonus_rows_in_place(monkeypatch):
    calls = _fake_aiter(monkeypatch)
    monkeypatch.setenv("ATOM_FUSED_TOPK_TOPP_SAMPLE", "1")
    mr = pytest.importorskip("atom.model_engine.model_runner", exc_type=ImportError)
    monkeypatch.setattr(mr, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    logits = torch.zeros(8, 248320, dtype=torch.bfloat16)
    tgt = torch.tensor([0, 1, 2, 4, 5, 6])
    bonus = torch.tensor([3, 7])
    cu = torch.tensor([3, 6], dtype=torch.int32)
    t, b = mr.ModelRunner._sample_verify_and_bonus(
        SimpleNamespace(sampler=sampler.Sampler()),
        logits,
        tgt,
        bonus,
        cu,
        torch.ones(2),
        20,
        0.95,
    )
    assert t.shape == (6,) and b.shape == (2,)
    kw = calls[0][3]
    assert kw["verify_rows"] is tgt and kw["bonus_rows"] is bonus
    assert kw["cu_num_draft_tokens"] is cu and calls[0][0] == logits.shape


def _gpu_ready():
    if not torch.cuda.is_available():
        return False
    try:
        import aiter.ops.triton.fusions.topk_topp_sample  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.mark.skipif(not _gpu_ready(), reason="ROCm GPU and aiter required")
def test_decode_postprocess_matches_eager(monkeypatch):
    """Verify + bonus sampling, the rejection and the drafter's next tokens:
    flag on equals flag off with a stable top-k (the radix top-k's own order
    and tie set are not fixed)."""
    import aiter.ops.sampling as aiter_sampling

    from atom.model_engine import model_runner as mr
    from atom.model_ops.rejection_sampler import RejectionSampler
    from atom.utils.forward_context import SpecDecodeMetadata

    monkeypatch.setattr(aiter_sampling, "_SMALL_K_TOPP", True)
    monkeypatch.setattr(mr, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    vocab, bs, steps = 248320, 29, 3
    rows = bs * (steps + 1) + 8
    logits = (torch.randn(rows, vocab, device="cuda") * 2.5).to(torch.bfloat16)
    i = torch.arange(bs, device="cuda")
    tgt = (i[:, None] * (steps + 1) + torch.arange(steps, device="cuda")).view(-1)
    bonus = i * (steps + 1) + steps
    cu = ((i + 1) * steps).to(torch.int32)
    ids = torch.randint(0, vocab, (rows + 1,), device="cuda", dtype=torch.int32)
    md = SpecDecodeMetadata(ids, steps, np.full(bs, steps), cu, tgt, bonus)
    temps = torch.ones(bs, device="cuda")
    temps[::3] = sampler.SAMPLER_EPS
    order = torch.cat([tgt, bonus])

    def stable_topk(probs, k):
        x = logits.index_select(0, order).float()
        sel = torch.sort(x, dim=1, descending=True, stable=True).indices[:, :k]
        return probs.gather(1, sel), sel

    def post(flag):
        monkeypatch.setenv("ATOM_FUSED_TOPK_TOPP_SAMPLE", flag)
        torch.manual_seed(5)
        t, b = mr.ModelRunner._sample_verify_and_bonus(
            SimpleNamespace(sampler=sampler.Sampler()),
            logits,
            tgt,
            bonus,
            cu,
            temps,
            20,
            0.95,
        )
        out, verdict = RejectionSampler().forward(md, None, b, target_token_ids=t)
        nxt = torch.gather(out.view(bs, -1), 1, verdict[0].view(-1, 1))
        return t, b, out, verdict, nxt

    # drafts: the target's own tokens at most positions, so rows accept
    with monkeypatch.context() as patch:
        patch.setattr(aiter_sampling, "_select_topk", stable_topk)
        ids[tgt + 1] = post("0")[0]
        expected = post("0")
    actual = post("1")
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
