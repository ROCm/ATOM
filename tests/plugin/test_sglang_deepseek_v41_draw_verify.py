# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 DSpark on the SGLang plugin: a greedy draft's top-p sampled
verify accepts by the target's own draw (`dspark_draw_verify_patch`).

CPU tests: the branch predicate, delegation of every other call with its
arguments, the switch, the recipe entry and the version / source guards.
GPU tests (skipped without a GPU, SGLang 0.5.22's DSpark kernels and aiter's
sampling op): over many copies of one request the draw rule's accept-length
histogram and per-position emitted-token frequencies match SGLang's rejection
sampler given the same one-hot draft (the two rules are equal in
distribution), and the outputs keep SGLang's dtypes and shapes."""

from types import SimpleNamespace

import pytest
import torch

from atom.plugin.sglang.patches import dspark_draw_verify_patch as patch
from atom.plugin.sglang.patches.sglang_version_guard import SUPPORTED_SGLANG_VERSIONS


class _T:
    """A CUDA-looking tensor for the host-side predicate (CPU-only tests)."""

    def __init__(self, *shape, device="cuda"):
        self.shape = shape
        self.device = torch.device(device)

    def dim(self):
        return len(self.shape)

    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n


def _info(bs=4, **kw):
    base = {
        "is_all_greedy": False,
        "is_any_greedy": False,
        "need_top_k_sampling": False,
        "need_top_p_sampling": True,
        "need_min_p_sampling": False,
        "temperatures": _T(bs, 1),
        "top_ps": _T(bs),
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _args(bs=4, width=6, vocab=1000, **kw):
    a = {
        "candidates": _T(bs, width),
        "target_logits": _T(bs * width, vocab),
        "draft_block": SimpleNamespace(greedy_draft=True),
        "sampling_info": _info(bs),
        "block_verification": False,
    }
    a.update(kw)
    return a


def test_takes_the_greedy_draft_top_p_branch():
    assert patch.takes_draw_verify(**_args())


@pytest.mark.parametrize(
    "change",
    [
        {"draft_block": SimpleNamespace(greedy_draft=False)},  # sampled draft
        {"sampling_info": None},  # all-greedy accept
        {"sampling_info": _info(is_all_greedy=True)},
        {"sampling_info": _info(is_any_greedy=True)},  # a greedy row: SGLang mixes
        {"sampling_info": _info(need_top_k_sampling=True)},
        {"sampling_info": _info(need_min_p_sampling=True)},
        {"sampling_info": _info(need_top_p_sampling=False)},  # temperature only
        {"block_verification": True},
        {"target_logits": _T(24, 1000, device="cpu")},
        {"target_logits": _T(23, 1000)},  # not bs * width rows
        {"sampling_info": _info(top_ps=_T(1))},  # not one top_p per request
    ],
)
def test_other_calls_take_sglang(change):
    assert not patch.takes_draw_verify(**_args(**change))


def test_delegates_other_calls_with_their_arguments(monkeypatch):
    seen = {}

    def original(**kw):
        seen.update(kw)
        return "sglang"

    wrapped = patch.make_draw_accept(original, dspark_accept=None)
    monkeypatch.setattr(patch, "takes_draw_verify", lambda **kw: False)
    kw = {
        "candidates": "c",
        "target_logits": "l",
        "draft_block": "d",
        "sampling_info": "s",
        "draft_input": "i",
        "gamma": 5,
        "verify_num_draft_tokens": 6,
        "cutoff_layout": "cut",
        "fused_argmax": True,
        "block_verification": False,
    }
    assert wrapped(**kw) == "sglang"
    assert seen == kw


def _sglang_modules():
    try:
        from sglang.kernels.ops.speculative.dspark import dspark_accept
        from sglang.srt.speculative.dspark_components import dspark_verify
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"SGLang DSpark modules unavailable: {e}")
    return dspark_verify, dspark_accept


def test_guards_accept_the_supported_runtime():
    dspark_verify, dspark_accept = _sglang_modules()
    import sglang

    if sglang.__version__ not in SUPPORTED_SGLANG_VERSIONS:
        pytest.skip("not SGLang 0.5.22")
    if not getattr(dspark_verify.accept_draft_tokens, patch._MARK, False):
        assert patch._upstream_shape_error(dspark_verify, dspark_accept) is None


def test_guards_refuse_other_sources(monkeypatch):
    dspark_verify, dspark_accept = _sglang_modules()

    def accept_draft_tokens(**kw):  # an upstream rewrite without the one-hot branch
        return None

    monkeypatch.setattr(dspark_verify, "accept_draft_tokens", accept_draft_tokens)
    err = patch._upstream_shape_error(dspark_verify, dspark_accept)
    assert err is not None and "no longer reads" in err


# --------------------------------------------------------------------------- GPU


def _gpu_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    dspark_verify, dspark_accept = _sglang_modules()
    try:
        import aiter.ops.sampling  # noqa: F401
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"aiter sampling unavailable: {e}")
    original = dspark_verify.accept_draft_tokens
    original = getattr(original, "__wrapped__", original)
    return original, dspark_accept


def _request(vocab, g, peak):
    """Six verify rows of one request: an LM-like row each, the draft = argmax
    of the row (a greedy draft agrees with the target's mode)."""
    z = torch.randn(6, vocab, device="cuda", generator=g) * 1.5
    tops = torch.randint(0, vocab, (6, 8), device="cuda", generator=g)
    for r in range(6):
        z[r, tops[r]] += torch.linspace(peak, peak - 3.0, 8, device="cuda")
    drafts = z[:5].argmax(-1)
    return z, drafts


def _batch(z, drafts, copies, temp, top_p):
    bs = copies
    logits = z.repeat(bs, 1).contiguous()
    anchor = torch.full((bs, 1), 3, device="cuda", dtype=torch.int64)
    d = drafts.view(1, 5).repeat(bs, 1)
    cand = torch.cat([anchor, d], 1).contiguous()
    info = SimpleNamespace(
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_k_sampling=False,
        need_top_p_sampling=True,
        need_min_p_sampling=False,
        temperatures=torch.full((bs, 1), temp, device="cuda"),
        top_ps=torch.full((bs,), top_p, device="cuda"),
        top_ks=torch.full((bs,), z.shape[1], device="cuda", dtype=torch.int32),
    )
    block = SimpleNamespace(
        greedy_draft=True,
        draft_tokens=d,
        greedy_mask=torch.zeros(bs, dtype=torch.bool, device="cuda"),
        temperatures=info.temperatures,
        corrected_logits=None,
    )
    return cand, logits, info, block


def _emitted(cand, correct_len, bonus):
    """[bs, 6] ids SGLang commits (accepted drafts, then the bonus), -1 past it."""
    bs = cand.shape[0]
    pos = torch.arange(6, device=cand.device).view(1, 6)
    cl = correct_len.long().view(bs, 1)
    drafts = torch.cat([cand[:, 1:], torch.full_like(cand[:, :1], -1)], 1)
    out = torch.where(pos < cl, drafts, torch.full_like(drafts, -1))
    return torch.where(pos == cl, bonus.view(bs, 1).expand(bs, 6), out)


@pytest.mark.parametrize("temp,top_p,peak", [(0.6, 0.95, 9.0), (1.0, 0.9, 6.0)])
def test_draw_rule_matches_sglang_rejection_in_distribution(temp, top_p, peak):
    original, dspark_accept = _gpu_or_skip()
    vocab, copies, reps = 2048, 4000, 6
    g = torch.Generator(device="cuda")
    g.manual_seed(20261121)
    z, drafts = _request(vocab, g, peak)
    hist = {"draw": torch.zeros(6), "sglang": torch.zeros(6)}
    emit = {"draw": [], "sglang": []}
    for _ in range(reps):
        cand, logits, info, block = _batch(z, drafts, copies, temp, top_p)
        a = patch.draw_accept(
            candidates=cand,
            target_logits=logits,
            sampling_info=info,
            cutoff_verify_lens=None,
            dspark_accept=dspark_accept,
        )
        b = original(
            candidates=cand,
            target_logits=logits,
            draft_block=block,
            sampling_info=info,
            draft_input=SimpleNamespace(max_top_k=None, uniform_top_k_value=None),
            gamma=5,
            verify_num_draft_tokens=6,
        )
        for name, (cl, bonus, trim) in (("draw", a), ("sglang", b)):
            assert cl.shape == (copies,) and bonus.shape == (copies,)
            assert bonus.dtype == torch.int64 and trim.shape == cl.shape
            assert int(trim.abs().sum()) == 0
            hist[name] += torch.bincount(cl.long().cpu(), minlength=6)[:6].float()
            emit[name].append(_emitted(cand, cl, bonus).cpu())
        assert a[0].dtype == b[0].dtype
    n = copies * reps
    pa, pb = hist["draw"] / n, hist["sglang"] / n
    # accept-length histograms: |diff| within 5 sigma of a binomial at n samples
    sigma = torch.sqrt(torch.clamp(pb * (1 - pb), min=1e-4) * 2 / n)
    assert torch.all((pa - pb).abs() <= 5 * sigma + 1e-3), (pa, pb)
    # the emitted token at every position, binned as its 16 most frequent ids
    # under SGLang's rule plus the rest of the vocab (a wide tail would make a
    # per-id comparison all sampling noise): within 5 sigma at n samples
    ea, eb = torch.cat(emit["draw"]), torch.cat(emit["sglang"])
    for pos in range(6):
        cb_full = torch.bincount(eb[:, pos] + 1, minlength=vocab + 1)
        top = torch.topk(cb_full, 16).indices
        ca_full = torch.bincount(ea[:, pos] + 1, minlength=vocab + 1)
        fa = torch.cat([ca_full[top], (n - ca_full[top].sum()).view(1)]).float() / n
        fb = torch.cat([cb_full[top], (n - cb_full[top].sum()).view(1)]).float() / n
        sig = torch.sqrt(torch.clamp(fb * (1 - fb), min=1e-4) * 2 / n)
        assert torch.all((fa - fb).abs() <= 5 * sig + 1e-3), (pos, fa, fb)


def test_rows_out_of_the_top_p_set_are_never_drawn():
    _, _dspark_accept = _gpu_or_skip()
    vocab, bs = 1024, 512
    z = torch.full((6, vocab), -20.0, device="cuda")
    z[:, :4] = torch.tensor([3.0, 2.0, 1.0, -8.0], device="cuda")
    _cand, logits, info, _block = _batch(z, z[:5].argmax(-1), bs, 1.0, 0.9)
    ids = patch.draw_target_tokens(logits, info.temperatures, info.top_ps, 6)
    assert ids.dtype == torch.int64 and ids.shape == (bs * 6,)
    # softmax(3, 2, 1) = 0.665, 0.245, 0.090: top_p 0.9 keeps {0, 1} only
    assert set(ids.unique().tolist()) <= {0, 1}
