# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The dual-stream MoE may only hand its two branches back unsummed when both
are already full-rank at that point.

Two separate pieces of code have to agree on that. ``_defer_shared_add``,
computed once at init, gates whether the split custom op is used at all; the
all-reduce placement inside the forwards decides whether the branches actually
come back unsummed. If the flag says "deferrable" and a forward sums anyway,
the split op receives a ``None`` its schema cannot represent. If the flag says
otherwise and a forward defers, a TP-partial tensor escapes the module and the
model is quietly wrong in a way no shape check would catch.

The real methods are executed here against stub branches, with
``tensor_model_parallel_all_reduce`` patched to record what it saw -- so the
assertions are about behavior, not about how the source happens to be written.
No GPU and no distributed init.
"""

import contextlib

import pytest
import torch

# kimi_k3 pulls in the GPU stack at import; these tests only need the MoE
# block's Python control flow.
kimi_k3 = pytest.importorskip(
    "atom.models.kimi_k3", reason="kimi_k3 imports the aiter/GPU stack"
)
KimiSparseMoeBlock = kimi_k3.KimiSparseMoeBlock


class _FakeStream:
    """A CUDA stream's interface, minus the CUDA."""

    def wait_stream(self, other):
        pass


class _Stub:
    """Stands in for KimiSparseMoeBlock without running its __init__.

    Only the attributes the split forwards actually touch are set, so anything
    else those methods start depending on shows up as an AttributeError rather
    than silently passing.
    """

    def __init__(self, *, latent, tp_size, shared=True):
        self.use_latent_moe = latent
        self.tp_size = tp_size
        self.shared_experts = (lambda x: x * 3) if shared else None
        self.routed_expert_norm = None
        self.alt_stream = _FakeStream()
        self._use_dual_stream = True
        self.gate = lambda x: x
        self.experts = lambda x, logits: x * 2
        self.routed_expert_down_proj = lambda x: x
        self.routed_expert_up_proj = lambda x, x_scale=None: x

    # Bind the real implementations under test.
    split_moe_forward = KimiSparseMoeBlock.split_moe_forward
    routed_expert_forward = KimiSparseMoeBlock.routed_expert_forward
    single_stream_moe_forward = KimiSparseMoeBlock.single_stream_moe_forward
    single_stream_split_moe_forward = KimiSparseMoeBlock.single_stream_split_moe_forward
    dual_stream_moe_forward = KimiSparseMoeBlock.dual_stream_moe_forward
    dual_stream_split_moe_forward = KimiSparseMoeBlock.dual_stream_split_moe_forward
    _dual_stream_split = KimiSparseMoeBlock._dual_stream_split
    # staticmethod: fetching it off the class yields a plain function, which
    # would rebind as an instance method here. Re-wrap to keep the real
    # signature.
    _assert_split = staticmethod(KimiSparseMoeBlock._assert_split)


def _defer_flag(stub):
    """The init-time predicate, evaluated against a stub's config."""
    return stub.shared_experts is not None and (
        stub.use_latent_moe or stub.tp_size == 1
    )


@pytest.fixture
def reduced(monkeypatch):
    """Patch the all-reduce to mark what passed through it."""
    seen = []

    def fake_all_reduce(x):
        seen.append(x)
        # Tag the tensor so a later assertion can tell reduced from partial.
        return x + 1000.0

    monkeypatch.setattr(kimi_k3, "tensor_model_parallel_all_reduce", fake_all_reduce)
    return seen


def _fake_cuda_streams(monkeypatch):
    """Make the dual-stream path runnable on CPU."""
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _FakeStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda self, s: None, raising=False
    )


CONFIGS = [
    pytest.param(True, 8, id="latent-tp8"),
    pytest.param(True, 1, id="latent-tp1"),
    pytest.param(False, 8, id="dense-tp8"),
    pytest.param(False, 1, id="dense-tp1"),
]


@pytest.mark.parametrize("latent,tp_size", CONFIGS)
def test_split_moe_forward_defers_exactly_when_the_flag_says(latent, tp_size, reduced):
    stub = _Stub(latent=latent, tp_size=tp_size)
    x = torch.ones(4, 8)
    _routed, shared = stub.split_moe_forward(x)
    assert (shared is not None) == _defer_flag(stub), (
        f"latent={latent} tp={tp_size}: split_moe_forward "
        f"{'deferred' if shared is not None else 'summed'} but _defer_shared_add "
        f"is {_defer_flag(stub)}"
    )


@pytest.mark.parametrize("latent,tp_size", CONFIGS)
def test_dual_stream_split_defers_exactly_when_the_flag_says(
    latent, tp_size, reduced, monkeypatch
):
    _fake_cuda_streams(monkeypatch)
    stub = _Stub(latent=latent, tp_size=tp_size)
    _routed, shared = stub._dual_stream_split(torch.ones(4, 8))
    assert (shared is not None) == _defer_flag(stub)


@pytest.mark.parametrize("latent,tp_size", CONFIGS)
def test_both_paths_agree_and_sum_to_the_same_thing(
    latent, tp_size, reduced, monkeypatch
):
    """Dual-stream only changes WHERE the shared branch runs. Summed, the two
    paths must produce the identical tensor -- otherwise moving the add out
    changed the math rather than just its location."""
    _fake_cuda_streams(monkeypatch)
    x = torch.ones(4, 8)

    single = _Stub(latent=latent, tp_size=tp_size).single_stream_moe_forward(x)
    dual = _Stub(latent=latent, tp_size=tp_size).dual_stream_moe_forward(x)
    assert torch.equal(single, dual)

    # And the deferred form must sum to that same value: this is exactly what
    # the next layer's attn_res does with the two addends.
    routed, shared = _Stub(latent=latent, tp_size=tp_size).split_moe_forward(x)
    recombined = routed if shared is None else routed + shared
    assert torch.equal(recombined, single), (
        "deferring the add changed the result; the next attn_res would fold in "
        "something different from what the summing path produces"
    )


@pytest.mark.parametrize("latent,tp_size", CONFIGS)
def test_deferred_branches_are_both_past_their_allreduce(
    latent, tp_size, reduced, monkeypatch
):
    """A branch handed back unsummed must already be reduced.

    The patched all-reduce adds 1000 per call, so under TP a deferred branch
    that skipped its collective is detectable by magnitude -- a partial tensor
    escaping the module is the one failure mode here that is silent in
    production.
    """
    _fake_cuda_streams(monkeypatch)
    for method in ("split_moe_forward", "_dual_stream_split"):
        stub = _Stub(latent=latent, tp_size=tp_size)
        routed, shared = getattr(stub, method)(torch.ones(4, 8))
        if shared is None or tp_size == 1:
            continue  # nothing deferred, or no collective to have skipped
        assert routed.min() >= 1000.0, f"{method}: routed branch not all-reduced"
        assert shared.min() >= 1000.0, f"{method}: shared branch not all-reduced"


@pytest.mark.parametrize("latent,tp_size", CONFIGS)
def test_split_dispatch_targets_never_return_none(
    latent, tp_size, reduced, monkeypatch
):
    """The split op's schema returns two real tensors. Its dispatch targets must
    honor that on every config the flag lets through, and fail loudly on any
    other rather than handing a None across the op boundary."""
    _fake_cuda_streams(monkeypatch)
    for method in ("single_stream_split_moe_forward", "dual_stream_split_moe_forward"):
        stub = _Stub(latent=latent, tp_size=tp_size)
        if _defer_flag(stub):
            routed, shared = getattr(stub, method)(torch.ones(4, 8))
            assert isinstance(routed, torch.Tensor)
            assert isinstance(shared, torch.Tensor)
        else:
            with pytest.raises(AssertionError):
                getattr(stub, method)(torch.ones(4, 8))


def test_no_shared_experts_never_defers(reduced):
    """With no shared branch there is nothing to defer, and the split op must
    not be reachable -- `forward` checks the same flag."""
    stub = _Stub(latent=True, tp_size=8, shared=False)
    assert not _defer_flag(stub)
    _routed, shared = stub.split_moe_forward(torch.ones(4, 8))
    assert shared is None
