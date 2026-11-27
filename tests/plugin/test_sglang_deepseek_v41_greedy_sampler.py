# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 on the SGLang plugin: an all-greedy sampling step picks with
aiter's exact top-1 instead of torch.argmax (`v41_greedy_sampler_patch`)."""

from types import SimpleNamespace

import pytest
import torch

from atom.plugin.sglang.patches import v41_greedy_sampler_patch as patch


def _module(use_aiter=False, disable_aiter=False):
    calls = []
    return SimpleNamespace(
        _use_aiter=use_aiter,
        _disable_aiter_greedy_sample=disable_aiter,
        maybe_detect_nan=lambda t, msg="": calls.append(("nan_probe", msg)),
        calls=calls,
    )


class _Logits:
    """A CUDA-looking [rows, vocab] tensor for the predicate (CPU-only tests)."""

    def __init__(self, rows=2, vocab=16, dtype=torch.float32, device="cuda"):
        self.t = torch.zeros(rows, vocab, dtype=dtype)
        self.device = torch.device(device)
        self.dtype = dtype
        self.shape = self.t.shape

    def dim(self):
        return self.t.dim()

    def stride(self, d):
        return self.t.stride(d)


def _info(**kw):
    base = {
        "is_all_greedy": True,
        "sampling_mask_batch_indices": None,
        "grammars": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_takes_only_the_plain_all_greedy_branch():
    m, x, info = _module(), _Logits(), _info()
    assert patch.takes_fast_greedy(m, x, info, False, None)
    # Anything else SGLang produces in forward goes to SGLang.
    assert not patch.takes_fast_greedy(m, x, _info(is_all_greedy=False), False, None)
    assert not patch.takes_fast_greedy(m, x, info, True, None)
    assert not patch.takes_fast_greedy(m, x, info, False, object())
    capture = _info(sampling_mask_batch_indices=torch.tensor([0]))
    assert not patch.takes_fast_greedy(m, x, capture, False, None)
    # SGLANG_USE_AITER's own greedy path stays SGLang's ...
    assert not patch.takes_fast_greedy(_module(use_aiter=True), x, info, False, None)
    # ... unless SGLang itself disabled it (then it would run torch.argmax).
    on_but_disabled = _module(use_aiter=True, disable_aiter=True)
    assert patch.takes_fast_greedy(on_but_disabled, x, info, False, None)


@pytest.mark.parametrize(
    "logits",
    [
        _Logits(device="cpu"),
        _Logits(rows=0),
        _Logits(dtype=torch.int32),
    ],
)
def test_takes_only_cuda_float_rows(logits):
    assert not patch.takes_fast_greedy(_module(), logits, _info(), False, None)
    assert not patch.takes_fast_greedy(_module(), None, _info(), False, None)


def test_a_strided_vocab_goes_to_sglang():
    x = _Logits(rows=2, vocab=16)
    x.t = torch.zeros(16, 2).t()  # last stride 2
    assert not patch.takes_fast_greedy(_module(), x, _info(), False, None)


class _Sampler:
    """SGLang's Sampler surface the fast path touches."""

    def __init__(self):
        self.calls = []

    def _preprocess_logits(self, logits, sampling_info):
        self.calls.append("preprocess")
        return logits + 0  # a new tensor, as a sanitising pass may return

    def _sync_token_ids_across_tp(self, ids, sampling_info):
        self.calls.append(("sync", ids.dtype))


def _forward(module, picks):
    seen = []

    def original(self, *args, **kwargs):
        seen.append((args, kwargs))
        return "sglang"

    def pick(logits):
        picks.append(logits)
        return torch.argmax(logits, -1)

    return patch.make_fast_greedy_forward(original, module, pick=pick), seen


def test_fast_path_runs_upstreams_steps_in_order(monkeypatch):
    monkeypatch.setattr(patch, "takes_fast_greedy", lambda *a: True)
    m, picks = _module(), []
    fwd, seen = _forward(m, picks)
    s = _Sampler()
    logits = torch.tensor([[0.0, 3.0, 3.0, 1.0], [5.0, 0.0, 0.0, 5.0]])
    out = fwd(
        s, SimpleNamespace(next_token_logits=logits), _info(), False, [], [], None
    )
    assert out.tolist() == [1, 0] and out.dtype == torch.int64
    assert m.calls == [("nan_probe", "sampler: next_token_logits")]
    assert s.calls == ["preprocess", ("sync", torch.int64)]
    # The pick reads the preprocessed logits, not the raw ones.
    assert len(picks) == 1 and picks[0] is not logits and not seen


def test_other_calls_reach_sglang_unchanged(monkeypatch):
    monkeypatch.setattr(patch, "takes_fast_greedy", lambda *a: False)
    picks = []
    fwd, seen = _forward(_module(), picks)
    lo = SimpleNamespace(next_token_logits=torch.zeros(1, 4))
    marker = object()
    assert (
        fwd(_Sampler(), lo, "info", True, [3], [[1]], "pos", logprob_logits=marker)
        == "sglang"
    )
    ((args, kwargs),) = seen
    assert args == (lo, "info", True, [3], [[1]], "pos")
    assert kwargs == {"logprob_logits": marker} and not picks


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("ATOM_SGLANG_V41_FAST_GREEDY", "false")
    assert patch.install_v41_greedy_sampler_patch() is False


@pytest.fixture
def sampler_module(monkeypatch):
    sm = pytest.importorskip("sglang.srt.layers.sampler")
    monkeypatch.setattr(sm.Sampler, "forward", sm.Sampler.forward)
    return sm


def test_install_swaps_forward_once(monkeypatch, sampler_module):
    monkeypatch.delenv("ATOM_SGLANG_V41_FAST_GREEDY", raising=False)
    original = sampler_module.Sampler.forward
    if patch._upstream_shape_error(sampler_module) is not None:
        assert patch.install_v41_greedy_sampler_patch() is False
        assert sampler_module.Sampler.forward is original
        return
    assert patch.install_v41_greedy_sampler_patch() is True
    installed = sampler_module.Sampler.forward
    assert installed is not original and installed._atom_v41_fast_greedy
    assert patch.install_v41_greedy_sampler_patch() is True
    assert sampler_module.Sampler.forward is installed
    assert installed.__wrapped__ is original


def test_signature_guards(sampler_module):
    class Other:
        def forward(self, logits_output, sampling_info):
            pass

        _preprocess_logits = _sync_token_ids_across_tp = None

    fake = SimpleNamespace(
        Sampler=Other, **{a: getattr(sampler_module, a) for a in patch._MODULE_ATTRS}
    )
    assert "Sampler.forward takes" in patch._upstream_shape_error(fake)
    bare = SimpleNamespace(Sampler=sampler_module.Sampler)
    assert "lacks" in patch._upstream_shape_error(bare)


# -- on the GPU: the replacement picks exactly what torch.argmax picks --------

_V41_VOCAB = 129280


def _gpu_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    pytest.importorskip("aiter")


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gpu_pick_equals_torch_argmax(rows, dtype):
    _gpu_or_skip()
    g = torch.Generator(device="cuda").manual_seed(rows)
    base = torch.randn(rows, _V41_VOCAB, device="cuda", generator=g).mul(4)
    # SGLang's logits are bf16 values widened to fp32: many exact ties.
    x = base.to(torch.bfloat16).to(dtype)
    tie = x.clone()
    for r in range(rows):  # the max twice; the lower id must win
        top = tie[r].max() + 1
        tie[r, 100000 - r] = top
        tie[r, 5000 + r] = top
    last = x.clone()
    last[:, -1] = last.max() + 1
    first = x.clone()
    first[:, 0] = first.max() + 1
    for t in (x, tie, last, first):
        got = patch.greedy_token_ids(t)
        assert got.dtype == torch.int64 and got.shape == (rows,)
        assert torch.equal(got, torch.argmax(t, -1))


def test_gpu_pick_reads_a_row_slice():
    _gpu_or_skip()
    x = torch.randn(6, _V41_VOCAB, device="cuda").to(torch.bfloat16).float()
    view = x[2:5]  # a row-offset view, unit last stride
    assert torch.equal(patch.greedy_token_ids(view), torch.argmax(view, -1))
