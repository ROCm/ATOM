# SPDX-License-Identifier: MIT
"""Preserve the native sampled-prefix chain, including graph scratch reuse."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a gfx1250 GPU", allow_module_level=True)
arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
if arch.split(":")[0] != "gfx1250":
    pytest.skip("the fused chain is enabled only on gfx1250", allow_module_level=True)

from atom.model_ops import dspark_prob_sample as native
from atom.model_ops.dspark_prob_markov_chain import (
    can_use_prob_markov_chain,
    prob_markov_sample_chain,
)

V, R, T = 129280, 256, 5


@pytest.fixture(scope="module")
def weights():
    generator = torch.Generator(device="cuda").manual_seed(8008)
    return tuple(
        torch.randn(V, R, device="cuda", dtype=torch.bfloat16, generator=generator)
        * 0.1
        for _ in range(2)
    )


def _bits_equal(a, b):
    integer = {torch.float32: torch.int32, torch.bfloat16: torch.int16}.get(a.dtype)
    return torch.equal(
        a.view(integer) if integer else a, b.view(integer) if integer else b
    )


class _Case:
    def __init__(self, b, weights, *, strided=False, keys=None):
        self.b = b
        self.w1, self.w2 = weights
        self.guards = []

        def allocate(shape, dtype, fill):
            if not strided:
                return torch.full(shape, fill, device="cuda", dtype=dtype)
            if len(shape) == 3:
                a, c, d = shape
                storage = torch.full(
                    (a + 2, c + 2, d + 7), fill, device="cuda", dtype=dtype
                )
                slices = (slice(1, a + 1), slice(1, c + 1), slice(0, d))
            else:
                a, c = shape
                storage = torch.full(
                    (a + 2, c * 2 + 3), fill, device="cuda", dtype=dtype
                )
                slices = (slice(1, a + 1), slice(1, 2 * c + 1, 2))
            view = storage[slices]
            self.guards.append((storage, storage.clone(), slices, view))
            return view

        self.base = allocate((b, T, V), torch.bfloat16, -111)
        self.base.copy_(torch.randn_like(self.base) * 3)
        anchors = torch.randint(0, V, (2 * b + 1,), device="cuda", dtype=torch.int64)
        self.anchor = anchors[1::2] if strided else anchors[:b]
        self.temps = torch.full((b,), 0.6, device="cuda", dtype=torch.float32)
        self.temps[::3] = 0
        self.temps[1::3] = native.GREEDY_T
        self.seed = torch.tensor([1234567], device="cuda", dtype=torch.int64)
        self.out = allocate((b, T), torch.int64, -777)
        self.saved = allocate((b, T), torch.int32, -888)
        self.q = allocate((b, T, V), torch.float32, -999)
        self.emb = allocate((b, T, R), torch.bfloat16, -99)
        self.keys = (
            torch.zeros((T + 2, b + 5), device="cuda", dtype=torch.uint64)
            if keys is None
            else keys
        )

    def copy_inputs(self, other):
        for name in ("base", "anchor", "temps", "seed"):
            getattr(self, name).copy_(getattr(other, name))

    def reference(self):
        for k in range(T):
            native.prob_markov_sample(
                self.base[:, k],
                self.anchor if k == 0 else self.out[:, k - 1],
                self.w1,
                self.w2,
                self.out[:, k],
                self.q[:, k],
                self.temps,
                self.seed,
                k,
                self.emb[:, k],
            )
        self.saved.copy_(self.out)

    def fused(self):
        prob_markov_sample_chain(
            self.base,
            self.anchor,
            self.w1,
            self.w2,
            self.out,
            self.q,
            self.temps,
            self.seed,
            self.emb,
            self.keys,
            self.saved,
        )

    def check_guards(self):
        for storage, expected, slices, view in self.guards:
            expected[slices].copy_(view)
            assert _bits_equal(storage, expected), "write outside destination view"


def _compare(reference, fused):
    for name in ("out", "saved", "q", "emb"):
        assert _bits_equal(getattr(reference, name), getattr(fused, name)), name
    assert not fused.keys.view(torch.int64).ne(0).any().item()
    reference.check_guards()
    fused.check_guards()


@pytest.mark.parametrize("b,strided", [(1, False), (3, True), (24, False), (32, True)])
def test_matches_native_ids_q_bits_and_embeddings(weights, b, strided):
    reference = _Case(b, weights, strided=strided)
    fused = _Case(b, weights, strided=strided)
    assert can_use_prob_markov_chain(fused.base, *weights, fused.keys)
    for seed in (0, 2147483651, -1):
        reference.seed.fill_(seed)
        # Exercise both stochastic and greedy sampling even at B=1.
        if seed == -1:
            reference.temps.fill_(1.3)
        fused.copy_inputs(reference)
        reference.reference()
        fused.fused()
        _compare(reference, fused)


@pytest.mark.parametrize(
    "pattern",
    ["ties", "negative_inf", "positive_inf", "negative_zero", "nan_tile", "all_nan"],
)
def test_maximum_edge_cases_preserve_native_choice(pattern):
    weights = tuple(
        torch.zeros(V, R, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    )
    reference, fused = _Case(3, weights, strided=True), _Case(3, weights, strided=True)
    reference.temps.zero_()
    reference.base.zero_()
    if pattern == "ties":
        reference.base[:, :, 7] = 2
        reference.base[:, :, 128] = 2
    elif pattern == "negative_inf":
        reference.base.fill_(float("-inf"))
    elif pattern == "positive_inf":
        reference.base[:, :, 127:] = float("inf")
    elif pattern == "negative_zero":
        reference.base[:, :, :128] = -0.0
    elif pattern == "nan_tile":
        reference.base[:, :, :128] = float("nan")
    elif pattern == "all_nan":
        reference.base.fill_(float("nan"))
    fused.copy_inputs(reference)
    reference.reference()
    fused.fused()
    _compare(reference, fused)


def test_graphs_with_different_batches_share_reset_workspace(weights):
    keys = torch.zeros((T + 2, 64), device="cuda", dtype=torch.uint64)
    stream = torch.cuda.Stream()
    cases = []
    for b in (24, 32):
        reference, fused = _Case(b, weights), _Case(b, weights, keys=keys)
        fused.copy_inputs(reference)
        fused.fused()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fused.fused()
        cases.append((reference, fused, graph))
    for seed, temperature in ((17, 0.0), (444555, 0.6), (999888, 1.3)):
        for reference, fused, graph in cases:
            reference.seed.fill_(seed)
            reference.temps.fill_(temperature)
            reference.anchor.add_(7).remainder_(V)
            reference.base.add_(0.125)
            fused.copy_inputs(reference)
            reference.reference()
            graph.replay()
            _compare(reference, fused)


def test_state_opt_in_and_fallback_metadata(weights, monkeypatch):
    monkeypatch.setattr(native, "FUSED_MARKOV_CHAIN", False)
    assert native.ProbDraftState.allocate(2, T, V, "cuda").markov_keys is None
    monkeypatch.setattr(native, "FUSED_MARKOV_CHAIN", True)
    state = native.ProbDraftState.allocate(2, T, V, "cuda")
    assert state.markov_keys.shape == (T, 2)
    assert not state.markov_keys.view(torch.int64).ne(0).any().item()
    base = torch.empty((1, T, V), device="cuda", dtype=torch.bfloat16)
    keys = torch.zeros((T, 64), device="cuda", dtype=torch.uint64)
    assert can_use_prob_markov_chain(base, *weights, keys)
    assert not can_use_prob_markov_chain(base, *weights, None)
    assert not can_use_prob_markov_chain(base.expand(64, -1, -1), *weights, keys)
    assert not can_use_prob_markov_chain(base[:, :2], *weights, keys)
    assert not can_use_prob_markov_chain(base[:, :, :128], *weights, keys)
    assert not can_use_prob_markov_chain(base.float(), *weights, keys)


@pytest.mark.parametrize("b,width", [(24, 5), (64, 5), (3, 2)])
def test_model_dispatch_preserves_native_and_verifier_buffers(
    weights, monkeypatch, b, width
):
    from atom.model_ops import dspark_prob_markov_chain as fused
    from atom.models.deepseek_v41.dspark import DeepseekV41DSpark

    monkeypatch.setattr(native, "FUSED_MARKOV_CHAIN", True)
    base = torch.randn((b, width, V), device="cuda", dtype=torch.bfloat16)
    anchors = torch.randint(0, V, (b,), device="cuda", dtype=torch.int32)
    normed = torch.empty((b * width, 1), device="cuda", dtype=torch.bfloat16)
    hidden = torch.empty((b, width, 1), device="cuda", dtype=torch.bfloat16)
    markov = SimpleNamespace(
        markov_w1=SimpleNamespace(weight=weights[0]),
        markov_w2=SimpleNamespace(weight=weights[1]),
    )
    model = SimpleNamespace(
        mtp=[SimpleNamespace(markov_head=markov, confidence_head=lambda _, emb: emb)],
        head=SimpleNamespace(get_logits=lambda _: base.flatten(0, 1)),
        _sharded_markov_sample=lambda: False,
    )
    reference = native.ProbDraftState.allocate(b, width, V, "cuda")
    candidate = native.ProbDraftState.allocate(b, width, V, "cuda")
    reference.markov_keys = None
    for state in (reference, candidate):
        state.seed.fill_(1729)
        state.temps.fill_(0.6)
    expected_ids, expected_emb = DeepseekV41DSpark._head_and_sample_prob(
        model, normed, hidden, anchors, reference
    )
    calls = []
    original = fused.prob_markov_sample_chain

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(fused, "prob_markov_sample_chain", tracked)
    ids, emb = DeepseekV41DSpark._head_and_sample_prob(
        model, normed, hidden, anchors, candidate
    )
    assert len(calls) == int(b <= 32 and width == T)
    assert torch.equal(ids, expected_ids)
    assert torch.equal(candidate.ids, reference.ids)
    assert _bits_equal(candidate.q, reference.q)
    assert _bits_equal(emb, expected_emb)
    assert not candidate.markov_keys.view(torch.int64).ne(0).any().item()
