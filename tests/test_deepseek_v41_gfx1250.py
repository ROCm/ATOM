# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 DeepSeek-V4.1 single-launch decode step on the CPU (no GPU):
- which deployments and which steps it takes (needs aiter, skipped without it);
- its kernel's per-step arithmetic (kernels/index_plan.py, moe_plan.py) and counter / LDS layout (layout.py), run on
  Python ints through a plain-integer stand-in for FlyDSL's traced Int32 (no FlyDSL, no aiter needed).
"""

import importlib.util
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import pytest
import torch
from torch import nn

from atom.models.deepseek_v41.mono import gfx1250
from atom.models.deepseek_v41.mono.gfx1250 import config, dispatch
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.mono.runtime import deployment

try:  # the gfx950 mono dispatch (its draft runner) needs aiter
    from atom.models.deepseek_v41.mono import dispatch as mono_dispatch
except ImportError:
    mono_dispatch = None

needs_aiter = pytest.mark.skipif(
    mono_dispatch is None, reason="atom.models.deepseek_v41.mono.dispatch needs aiter"
)


def atom_config(**overrides):
    fields = {
        "tensor_parallel_size": 4,
        "enable_expert_parallel": False,
        "enable_dp_attention": False,
        "parallel_config": NS(data_parallel_size=1),
        "pipeline_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_tbo": False,
        "enable_tbo_decode": False,
        "compilation_config": NS(level=0),
        "kv_cache_dtype": "bf16",
        "index_cache_dtype": "fp8",
        "speculative_config": NS(method="dspark", num_speculative_tokens=5),
        "dspark": NS(confidence_schedule=False, ragged=False),
    }
    fields.update(overrides)
    return NS(**fields)


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(deployment, "_gfx", lambda: "gfx1250")
    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)
    monkeypatch.setattr(deployment, "_takes_weight_updates", lambda c: False)
    monkeypatch.setattr(deployment, "_uses_tr_hybrid", lambda c: False)
    monkeypatch.setattr(deployment, "pcp_group_size_of", lambda c: 1)
    monkeypatch.setenv("ATOM_MONO_CHECK", "0")


def test_the_supported_deployment_is_not_refused():
    assert config.config_refusal(atom_config(), Dims) is None


@pytest.mark.parametrize(
    "change, why",
    [
        ({"tensor_parallel_size": 8}, "TP 8"),
        ({"kv_cache_dtype": "fp8"}, "kv cache fp8"),
        ({"index_cache_dtype": "fp4"}, "index cache fp4"),
        (
            {"speculative_config": NS(method="mtp", num_speculative_tokens=5)},
            "speculative method mtp",
        ),
        (
            {"speculative_config": NS(method="dspark", num_speculative_tokens=3)},
            "DSpark block size",
        ),
        ({"speculative_config": None}, "speculative method None"),
        ({"enable_expert_parallel": True}, "expert parallel"),
        ({"pipeline_parallel_size": 2}, "PP > 1"),
        (
            {"dspark": NS(confidence_schedule=True, ragged=True)},
            "DSpark confidence schedule",
        ),
        (
            {"dspark": NS(confidence_schedule=False, ragged=True)},
            "DSpark ragged verify",
        ),
    ],
)
def test_other_deployments_are_refused(change, why):
    assert why in config.config_refusal(atom_config(**change), Dims)


def test_the_arch_refusal_of_the_gfx950_kernels_does_not_apply():
    # the shared refusals reject every arch but gfx950; the single launch is gfx1250 code
    assert config.config_refusal(atom_config(), Dims) is None


class _Lazy:
    """LazyRunner stand-in: records the widths it was asked to make ready."""

    def __init__(self, create, name):
        self.asked = []
        self.runner = object()

    def ready(self, rows):
        self.asked.append(rows)
        return True


class _Inner(nn.Module):
    def forward(self, input_ids, positions, inputs_embeds=None):
        return ("inner", input_ids, inputs_embeds)

    def compute_logits(self, hidden_states):
        return hidden_states


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(dispatch, "LazyRunner", _Lazy)

    return dispatch.SingleStepModel(_Inner(), drafter=None, atom_config=atom_config())


@pytest.mark.parametrize(
    "rows",
    range(config.ROWS_PER_REQ, config.MAX_ROWS + 1, config.ROWS_PER_REQ),
)
def test_every_whole_request_width_up_to_eight_requests_is_served(model, rows):
    assert model.step_ready(rows)
    assert model._single.asked == [rows]


@pytest.mark.parametrize("rows", [0, 1, 5, 7, 13, 47, 54, 96])
def test_other_widths_take_the_production_path(model, rows):
    assert not model.step_ready(rows)
    assert model._single.asked == []


def test_the_forward_is_the_production_one(model):
    ids = torch.arange(6)
    assert model(ids, ids)[0] == "inner"
    assert model.compute_logits(3) == 3


def test_the_wrapper_forwards_the_model_attributes(model):
    model.model.extra = 7
    assert model.extra == 7


def test_off_when_mono_is_switched_off(monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    assert not gfx1250.enabled()


def test_a_refused_deployment_keeps_the_model(monkeypatch):
    inner = nn.Linear(1, 1)
    out = dispatch.install_single_step(
        inner, atom_config(kv_cache_dtype="fp8"), drafter=None
    )
    assert out is inner


@needs_aiter
def test_mono_dispatch_routes_to_the_single_launch_when_enabled(monkeypatch):
    monkeypatch.setattr(gfx1250, "enabled", lambda: True)
    seen = {}

    def install(model, cfg, drafter):
        seen["args"] = (model, cfg, drafter)
        return "single"

    monkeypatch.setattr(dispatch, "install_single_step", install)
    cfg = atom_config()
    assert mono_dispatch.install_mono_decode("m", cfg, "d") == "single"
    assert seen["args"] == ("m", cfg, "d")


class _Runner:
    def __init__(self, ok=True):
        self.ok = ok

    def precheck(self, batch, forward_context):
        return self.ok


def _batch(**kw):
    return NS(
        **{
            "structured_requests": None,
            "any_logprobs": False,
            "required_channel": None,
            "total_seqs_num": 8,
            "req_ids": [f"r{i}" for i in range(8)],
            **kw,
        }
    )


@pytest.fixture
def verify_step(monkeypatch):
    fc = NS(spec_decode_metadata=object())
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: fc)
    return fc


def test_a_decode_verify_step_is_served(model, verify_step):
    model._single.runner = _Runner()
    assert model.single_step_runner(_batch(), 48) is model._single.runner


@pytest.mark.parametrize(
    "batch",
    [
        _batch(structured_requests=[1]),
        _batch(any_logprobs=True),
        _batch(required_channel=0),
    ],
)
def test_steps_needing_logits_take_the_production_path(model, verify_step, batch):
    model._single.runner = _Runner()
    assert model.single_step_runner(batch, 48) is None


def test_a_step_without_drafts_takes_the_production_path(model, verify_step):
    verify_step.spec_decode_metadata = None
    model._single.runner = _Runner()
    assert model.single_step_runner(_batch(), 48) is None


def test_a_request_shape_the_kernel_refuses_takes_the_production_path(
    model, verify_step
):
    model._single.runner = _Runner(ok=False)
    assert model.single_step_runner(_batch(), 48) is None


def test_check_mode_never_serves(model, verify_step, monkeypatch):
    monkeypatch.setenv("ATOM_MONO_CHECK", "1")
    model._single.runner = _Runner()
    assert model.single_step_runner(_batch(), 48) is None


class _ServingRunner(_Runner):
    def __init__(self):
        super().__init__()
        self.steps, self.forgot = 0, []

    def run_step(self, input_ids, running_bs, forward_context, batch):
        self.steps += 1

    def draft_ids(self, bs):
        return ("kernel", bs)

    def forget_drafts(self, req_ids):
        self.forgot += list(req_ids)


class _Drafter:
    def __init__(self):
        self.kv = 0

    def compute_draft_kv(self, *args, **kwargs):
        self.kv += 1

    def propose(self, *args, next_token_ids, **kwargs):
        return "production"


@pytest.fixture
def served(verify_step, monkeypatch):
    monkeypatch.setattr(dispatch, "LazyRunner", _Lazy)
    drafter = _Drafter()
    m = dispatch.SingleStepModel(_Inner(), drafter, atom_config())
    m._single.runner = _ServingRunner()
    return m, drafter


def test_a_served_step_drafts_from_the_kernel(served):
    m, drafter = served
    runner = m._single.runner
    assert m.step_override(torch.zeros(48), 8, _batch()) is runner
    assert runner.steps == 1
    drafter.compute_draft_kv()
    assert drafter.kv == 0
    assert drafter.propose(next_token_ids=torch.zeros(8)) == ("kernel", 8)
    assert runner.forgot == []


def test_a_step_with_fewer_rows_than_6_a_request_is_not_served(served):
    m, _ = served
    # 8 requests in 42 rows (a ragged or confidence-scheduled verify): the launch reads 6 rows a request
    assert m.step_override(torch.zeros(42), 8, _batch()) is None
    assert m._single.runner.steps == 0


def test_a_padded_step_is_not_served(served):
    m, _ = served
    # input_ids not 6 rows a request (a padded or ragged step): the launch would misread them
    assert m.step_override(torch.zeros(48), 7, _batch(total_seqs_num=7)) is None
    assert m.step_override(torch.zeros(42), 8, _batch()) is None
    assert m._single.runner.steps == 0


def test_a_production_step_drafts_from_the_drafter(served):
    m, drafter = served
    runner = m._single.runner
    batch = _batch(any_logprobs=True, total_seqs_num=1, req_ids=["a", "b"])
    assert m.step_override(torch.zeros(6), 1, batch) is None
    assert runner.steps == 0
    drafter.compute_draft_kv()
    assert drafter.kv == 1
    assert drafter.propose(next_token_ids=torch.zeros(1)) == "production"
    assert runner.forgot == [
        "a"
    ]  # this batch's requests: their kept q no longer match their drafts


def test_a_constrained_replay_declines_previous_served_drafts(served):
    m, drafter = served
    runner = m._single.runner
    assert m.step_override(torch.zeros(48), 8, _batch()) is runner
    batch = _batch(
        structured_requests=[1], total_seqs_num=1, req_ids=["current", "other"]
    )
    assert m.step_override(torch.zeros(48), 8, batch) is None
    drafter.compute_draft_kv()
    assert drafter.kv == 1
    assert drafter.propose(next_token_ids=torch.zeros(1)) == "production"
    assert runner.steps == 1
    assert runner.forgot == ["current"]


def test_a_prefill_keeps_the_running_requests_kernel_q(served):
    m, drafter = served
    runner = m._single.runner
    m.step_override(torch.zeros(48), 8, _batch())  # a served decode step
    m(
        torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)
    )  # then a prefill (production forward)
    assert drafter.propose(next_token_ids=torch.zeros(1)) == "production"
    assert (
        runner.forgot == []
    )  # the new request has no kept q; the running ones keep theirs


# --- the kernel's per-step arithmetic on Python ints ----------------------------------------------------------------
KERNELS = (
    Path(__file__).resolve().parents[1]
    / "atom/models/deepseek_v41/mono/gfx1250/kernels"
)


class B:
    """A traced comparison's stand-in: ``.select`` and the bitwise and / or the kernel code uses."""

    def __init__(self, v):
        self.v = bool(v)

    def select(self, a, b):
        return a if self.v else b

    def __and__(self, o):
        return B(self.v and bool(o))

    def __or__(self, o):
        return B(self.v or bool(o))

    def __bool__(self):
        return self.v


class Z(int):
    """A traced Int32's stand-in: arithmetic stays Z, comparisons give B; ``//`` and ``%`` only on non-negative
    operands (where Python's floor and the device's truncation agree)."""

    def _w(f):
        return lambda a, b: Z(f(int(a), int(b)))

    __add__, __radd__ = _w(int.__add__), _w(int.__radd__)
    __sub__, __rsub__ = _w(int.__sub__), _w(int.__rsub__)
    __mul__, __rmul__ = _w(int.__mul__), _w(int.__rmul__)

    def __floordiv__(a, b):
        assert int(a) >= 0 and int(b) > 0, (a, b)
        return Z(int(a) // int(b))

    def __mod__(a, b):
        assert int(a) >= 0 and int(b) > 0, (a, b)
        return Z(int(a) % int(b))

    __lt__ = lambda a, b: B(int(a) < int(b))
    __le__ = lambda a, b: B(int(a) <= int(b))
    __gt__ = lambda a, b: B(int(a) > int(b))
    __ge__ = lambda a, b: B(int(a) >= int(b))
    __eq__ = lambda a, b: B(int(a) == int(b))
    __hash__ = int.__hash__


def load(name):
    """kernels/<name>.py with its FlyDSL names (``fx``, ``I32``) bound to the integer stand-in."""
    stub = {}  # without FlyDSL (CI): a stand-in, only while the file loads
    if importlib.util.find_spec("flydsl") is None:
        fl = types.ModuleType("flydsl")
        fl.expr = types.ModuleType("flydsl.expr")
        fl.expr.Int32 = Z
        stub = {"flydsl": fl, "flydsl.expr": fl.expr}
    spec = importlib.util.spec_from_file_location(
        f"gfx1250_{name}", KERNELS / f"{name}.py"
    )
    m = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stub):
        spec.loader.exec_module(m)
    if hasattr(m, "fx"):
        m.fx = NS(
            max=lambda a, b: Z(max(int(a), int(b))),
            min=lambda a, b: Z(min(int(a), int(b))),
            Int32=Z,
        )
        m.I32 = Z
    return m


IP, MP, LY = load("index_plan"), load("moe_plan"), load("layout")


# --- index_plan ---------------------------------------------------------------------------------------------------
KINDS = {"sel": (IP.KSEL, IP.SEG // IP.KSEL), "blk": (IP.KBLK, IP.SEG // IP.KBLK)}
EDGES = [0, 1, 7, 8, 9, IP.TOPK - 1, IP.TOPK, IP.TOPK + 1, IP.SEG - 1, IP.SEG, IP.SEG + 1, 2 * IP.SEG,
         16384, 16385, 26624, 65535, 65536, 131072, 1 << 20]  # fmt: skip


def bounds():
    rng = random.Random(0)
    return EDGES + [rng.randrange(0, 1 << 20) for _ in range(200)]


def outputs(n, lev, j, k, fan):
    """Reference: the values list j at level lev holds (its inputs, kept to k)."""
    if lev == 0:
        return min(max(n - j * IP.SEG, 0), IP.SEG)
    kids = range(j * fan, min((j + 1) * fan, int(IP.lists(Z(n), lev - 1, fan))))
    return sum(min(outputs(n, lev - 1, c, k, fan), k) for c in kids)


@pytest.mark.parametrize("kind", KINDS)
def test_tree_inputs(kind):
    k, fan = KINDS[kind]
    for n in bounds():
        for lev in range(IP.NLMAX + 1):
            nl = int(IP.lists_rt(Z(n), Z(lev), Z(fan)))
            assert nl == int(IP.lists(Z(n), lev, fan))
            for j in {0, max(nl - 1, 0)}:
                if j < nl:
                    assert int(
                        IP.list_inputs_rt(Z(n), Z(lev), Z(j), Z(k), Z(fan))
                    ) == outputs(n, lev, j, k, fan), (n, lev, j)


@pytest.mark.parametrize("kind", KINDS)
def test_task_counts(kind):
    _, fan = KINDS[kind]
    for n in bounds():
        counts = [
            int(IP.list_count_rt(Z(n), Z(lev), Z(fan))) for lev in range(IP.NLMAX + 1)
        ]
        assert counts[0] == max(int(IP.lists(Z(n), 0, fan)), 1)
        for lev in range(1, IP.NLMAX + 1):
            below = int(IP.lists(Z(n), lev - 1, fan))
            assert counts[lev] == (int(IP.lists(Z(n), lev, fan)) if below > 1 else 0)
        if n <= (1 << 20) // (
            8 if kind == "blk" else 1
        ):  # NLMAX levels reach one list (blk: blocks of a 1M bound)
            assert int(IP.lists(Z(n), IP.NLMAX, fan)) <= 1


def rx(bd, cand):
    kept = IP.rx_kept(Z(bd))
    last = Z(cand[max(int(kept) - 1, 0)]) if int(kept) else Z(0)
    return int(kept), int(IP.rx_context(kept, last, Z(bd)))


def test_reindex_key_rows_past_16384():
    """A REINDEX token keeps <= 2048 candidate blocks (<= 16384 compacted columns) but their real rows run to its bound:
    the key row is clamped to the bound, not to the compacted count (the bug fixed 2026-11-09).
    """
    rng = random.Random(1)
    for bd in [100, 16383, 16384, 16385, 18053, 26624, 65536]:
        nblk = -(-bd // 8)
        cand = sorted(rng.sample(range(nblk), min(nblk, IP.NCAND)))
        kept, cx = rx(bd, cand)
        assert kept == min(nblk, IP.NCAND) and cx <= IP.NCAND * 8
        for col in range(0, cx, 7):
            krow = cand[col // 8] * 8 + col % 8
            row = int(IP.key_row(Z(krow), Z(bd)))
            assert row == min(krow, bd - 1) and row < bd
            if krow < bd:
                assert row == krow
        if bd > 16384:  # the old clamp (to cx - 1) would move some real rows
            assert any(cand[c // 8] * 8 + c % 8 > cx - 1 for c in range(cx))


def test_reindex_context():
    """The compacted columns: every kept block's 8 rows but the last block's rows past the bound."""
    for bd in [0, 1, 8, 9, 100, 16384, 20000]:
        nblk = -(-bd // 8)
        cand = list(range(nblk))[-IP.NCAND :]
        kept, cx = rx(bd, cand)
        exp = (
            0 if not kept else (kept - 1) * 8 + min(max(bd - cand[kept - 1] * 8, 0), 8)
        )
        assert cx == exp


@pytest.mark.parametrize("TR", [6, 12, 18, 24, 30, 36, 42, 48])
def test_score_split(TR):
    sub = IP.score_sub(TR, 256)
    assert 1 <= sub <= IP.SUB and (TR * -(-16 // sub) <= 256 or sub == IP.SUB)
    tcols = sub * IP.CHUNK
    for cx in bounds():
        tasks = int(IP.score_tasks(Z(cx), Z(tcols)))
        if cx <= IP.TOPK:
            assert tasks == 0
        else:
            assert tasks * tcols >= cx > (tasks - 1) * tcols


# --- moe_plan -----------------------------------------------------------------------------------------------------
NUG, CTAS, WAVES = 18, 256, 8


def assigned(nch, U):
    ndc = MP.dedicated_ctas(nch, NUG)
    CR = CTAS - ndc
    stripe = MP.striped(nch)
    seen = []
    for b in range(CR):
        u0, u1 = MP.block_lo(b, U, CR), MP.block_lo(b + 1, U, CR)
        for w in range(WAVES):
            wu = u0 + w
            for it in range(max(MP.unit_iters(stripe, U, CR, b, w, u1, wu), 0)):
                seen.append(MP.unit_of(stripe, it, CR, b, w, wu))
    return seen


@pytest.mark.parametrize("nch", [1, 2, 3])
@pytest.mark.parametrize(
    "U", [0, 1, 17, 18 * 6, 2034, 2048, 2049, 2600, 18 * (288 + 16)]
)
def test_units_once(nch, U):
    seen = assigned(nch, U)
    assert sorted(seen) == list(range(U))


def test_schedules():
    assert [MP.striped(n) for n in (1, 2, 3)] == [False, False, True]
    assert [MP.sh_early(n) for n in (1, 2, 3)] == [False, False, MP.SH_EARLY]
    assert MP.dedicated_ctas(3, NUG) == 0  # wide: the shared UG runs early in the front
    assert MP.dedicated_ctas(1, NUG) == NUG  # narrow: one shared unit a dedicated CTA
    assert MP.dedicated_ctas(2, NUG) == 2 * NUG


# --- layout -------------------------------------------------------------------------------------------------------
def test_layer_slots_disjoint():
    used = {}
    for name, (off, nb) in LY.layer_slots().items():
        assert off % 4 == 0 and off + nb <= LY.LAYER_BYTES, name
        for b in range(off, off + nb, 4):
            assert b not in used, (name, used.get(b))
            used[b] = name


def test_tail_slots_ordered():
    seq = [
        LY.HEAD,
        LY.HEAD_DONE,
        LY.SAMPLE,
        LY.VERIFY_DONE,
        LY.DRAFT_HEAD_DONE,
        LY.MARKOV,
        LY.DRAFT_STATE,
        LY.DRAFT_FRONT,
        LY.TARGET_STATE,
        LY.TARGET_FRONT,
    ]
    assert seq == sorted(seq) and len(set(seq)) == len(seq)


def test_lds_regions():
    assert LY.TP_PAYLOAD < LY.FRONT_LDS < LY.RKEEP
