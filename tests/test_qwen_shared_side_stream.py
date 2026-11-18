# SPDX-License-Identifier: MIT
"""ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM: the Qwen3-Next / Qwen3.5 sparse MoE
block runs the router GEMM alone and the shared expert on a side stream beside
the routed experts (atom/models/qwen3_next.py, `side_stream_moe_forward`).

CPU: the flag and its row range, the fork point ("early" right after the
router, "late" through aiter's `after_routing_launched` hook), and what the
block returns (the deferred `MoEGatedSum` or the gated sum) -- with the GEMMs
and streams faked.

GPU (skipped without one): `router_logits_gemm` / `shared_expert_gemms` bitwise
the fused router + shared-expert kernel.
"""

import contextlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

mt = pytest.importorskip(
    "atom.model_ops.moe_tail_norm",
    reason="moe_tail_norm imports aiter and triton",
    exc_type=ImportError,
)

HIDDEN, EXPERTS = 2048, 256


def _qwen3_next():
    try:
        import atom.models.qwen3_next as module
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"atom.models.qwen3_next does not import here: {exc}")
    return module


class _Stream:
    """A stand-in CUDA stream that records waits."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))


def _block(qn, monkeypatch, tokens, log):
    gen = torch.Generator().manual_seed(tokens)
    logits = torch.randn(tokens, EXPERTS + 1, generator=gen).to(torch.bfloat16)
    shared = torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16)
    routed = torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16)
    block = object.__new__(qn.Qwen3NextSparseMoeBlock)
    nn.Module.__init__(block)
    block.n_routed_experts = EXPERTS
    block.is_rocm_aiter_fusion_shared_expert_enabled = False
    block.tp_size = 1
    block._use_dual_stream = False
    block.router_padded = None
    block.gate = SimpleNamespace(
        weight=torch.zeros(EXPERTS + 1, 8, dtype=torch.bfloat16), bias=None
    )
    block.shared_expert = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight="w_gate_up"),
        down_proj=SimpleNamespace(weight="w_down"),
        gated_gate_up=True,
        expert_gate=None,
    )

    def experts(hidden_states, router_logits):
        log.append(("experts", router_logits.shape[1]))
        hook = getattr(
            sys.modules.get("aiter.ops.flydsl.grouped_moe_gfx1250"), "_hook", None
        )
        if hook is not None:
            hook()  # the grouped MoE reaches its expert GEMMs
        return routed

    block.experts = experts
    main, side = _Stream(log, "main"), _Stream(log, "side")
    monkeypatch.setattr(qn, "_shared_expert_stream", lambda: side)
    monkeypatch.setattr(qn.torch.cuda, "current_stream", lambda: main)

    @contextlib.contextmanager
    def on_stream(s):
        log.append(("on", s.name))
        yield

    monkeypatch.setattr(qn.torch.cuda, "stream", on_stream)

    def router(x, w, n):
        log.append(("router", n))
        return logits

    def shared_gemms(x, wgu, wd):
        log.append(("shared", wgu, wd))
        return shared

    monkeypatch.setattr(qn, "router_logits_gemm", router)
    monkeypatch.setattr(qn, "shared_expert_gemms", shared_gemms)
    monkeypatch.setattr(qn, "fused_router_shared_available", lambda: True)
    return block, SimpleNamespace(logits=logits, shared=shared, routed=routed)


@pytest.fixture
def hook_module(monkeypatch):
    """aiter's grouped MoE module with `after_routing_launched` faked."""
    fake = types.ModuleType("aiter.ops.flydsl.grouped_moe_gfx1250")
    fake._hook = None

    class after_routing_launched:
        def __init__(self, fn):
            self.fn, self.called = fn, False

        def __enter__(self):
            def run():
                self.called = True
                fake._hook = None
                self.fn()

            fake._hook = run
            return self

        def __exit__(self, *exc):
            fake._hook = None

    fake.after_routing_launched = after_routing_launched
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.grouped_moe_gfx1250", fake)
    return fake


def test_flags_default_off(monkeypatch):
    from atom.utils import envs

    for name in (
        "ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM",
        "ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM_MIN_TOKENS",
        "ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM_MAX_TOKENS",
        "ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM_LATE_MIN_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert envs.ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM is False
    assert envs.ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM_MIN_TOKENS == 129
    assert envs.ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM_LATE_MIN_TOKENS == 3072


@pytest.mark.parametrize(
    "tokens, expected",
    [
        (64, None),
        (129, "early"),
        (455, "early"),
        (3071, "early"),
        (3072, "late"),
        (70000, None),
    ],
)
def test_mode_by_row_count(monkeypatch, hook_module, tokens, expected):
    qn = _qwen3_next()
    monkeypatch.setenv("ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM", "1")
    block, _ = _block(qn, monkeypatch, 4, [])
    x = torch.zeros(tokens, 8, dtype=torch.bfloat16)
    assert block._shared_side_stream_mode(x) == expected


def test_mode_off_without_flag_or_on_tp(monkeypatch, hook_module):
    qn = _qwen3_next()
    block, _ = _block(qn, monkeypatch, 4, [])
    x = torch.zeros(455, 8, dtype=torch.bfloat16)
    monkeypatch.delenv("ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM", raising=False)
    assert block._shared_side_stream_mode(x) is None
    monkeypatch.setenv("ATOM_QWEN_SHARED_EXPERT_SIDE_STREAM", "1")
    block.tp_size = 4
    assert block._shared_side_stream_mode(x) is None


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("defer", [False, True])
def test_side_stream_forward(monkeypatch, hook_module, late, defer):
    qn = _qwen3_next()
    log = []
    tokens = 5
    block, t = _block(qn, monkeypatch, tokens, log)
    block.defer_gated_sum = defer
    block.defer_gather = False
    monkeypatch.setattr(qn, "ATOM_FUSED_SIGMOID_GATE", False)
    out = block.side_stream_moe_forward(
        torch.zeros(tokens, 8, dtype=torch.bfloat16), late
    )
    fork = [("wait", "side", "main"), ("on", "side"), ("shared", "w_gate_up", "w_down")]
    if late:
        # forked inside the routed call, once the routing is launched
        assert log == [("router", EXPERTS + 1), ("experts", EXPERTS)] + fork + [
            ("wait", "main", "side")
        ]
    else:
        assert log == [("router", EXPERTS + 1)] + fork + [
            ("experts", EXPERTS),
            ("wait", "main", "side"),
        ]
    if defer:
        assert isinstance(out, mt.MoEGatedSum)
        assert out.routed is t.routed and out.shared is t.shared
        assert out.gate_logits.data_ptr() == t.logits[:, EXPERTS:].data_ptr()
    else:
        assert torch.equal(out, F.sigmoid(t.logits[:, EXPERTS:]) * t.shared + t.routed)


def test_late_fork_not_reached_runs_the_shared_expert_inline(monkeypatch, hook_module):
    qn = _qwen3_next()
    log = []
    block, t = _block(qn, monkeypatch, 3, log)
    block.defer_gated_sum = True
    block.defer_gather = False
    block.experts = lambda hidden_states, router_logits: t.routed  # never forks
    out = block.side_stream_moe_forward(torch.zeros(3, 8, dtype=torch.bfloat16), True)
    assert ("wait", "side", "main") not in log and ("wait", "main", "side") not in log
    assert ("shared", "w_gate_up", "w_down") in log
    assert out.shared is t.shared


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("m", [4, 64, 256, 512, 1024, 2048])
def test_gemms_bitwise_the_fused_kernel(m):
    if not mt.fused_router_shared_available():
        pytest.skip("needs aiter's gfx1250 router + shared expert kernel")
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
    wr = mt.pad_router_weight(
        (torch.randn(EXPERTS + 1, HIDDEN, generator=gen, device="cuda") / 45).to(
            torch.bfloat16
        )
    )
    wgu = mt.interleave_gate_up(
        (torch.randn(1024, HIDDEN, generator=gen, device="cuda") / 45).to(
            torch.bfloat16
        )
    )
    wd = (torch.randn(HIDDEN, 512, generator=gen, device="cuda") / 22).to(
        torch.bfloat16
    )
    logits, shared = mt.router_shared_expert(
        x, wr, EXPERTS + 1, wgu, wd, prefill=m > 512
    )
    assert torch.equal(mt.router_logits_gemm(x, wr, EXPERTS + 1), logits)
    assert torch.equal(mt.shared_expert_gemms(x, wgu, wd), shared)
