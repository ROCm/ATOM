# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Kimi-K3 step tail's contract with the engine on the CPU (no GPU, no
FlyDSL; needs aiter for the mono dispatch it wraps): which deployments get
the step override, which steps it serves, what a served step hands
``ModelRunner`` and what every other step keeps."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from torch import nn

tail = pytest.importorskip(
    "atom.models.kimi_k3.step_tail.dispatch",
    reason="the Kimi-K3 mono dispatch needs aiter",
)

from atom.models.kimi_k3.mono.dispatch import MonoDecodeModel
from atom.models.kimi_k3.step_tail import layout as L
from atom.models.kimi_k3.step_tail.runner import step_outputs
from atom.mono.runtime.step_override import StepOverrideModel

KEY = tail.request_bucket(8, 8)


def atom_config(**overrides):
    fields = {
        "tensor_parallel_size": 8,
        "prefill_context_parallel_size": 1,
        "speculative_config": NS(
            method="dspark", num_speculative_tokens=7, synthetic_acceptance_rates=None
        ),
    }
    fields.update(overrides)
    return NS(**fields)


class _Graph:
    def __init__(self):
        self.keys, self.replayed = {KEY}, []

    def has(self, key):
        return key in self.keys

    def replay(self, key):
        self.replayed.append(key)


class _Drafter:
    lossless = None
    verify_scheduler = None

    def __init__(self):
        self.runner = NS(
            req_graph=_Graph(),
            tokenID_processor=NS(is_deferred_out=True),
            forward_vars={"outputs": torch.zeros(64, L.HIDDEN, dtype=torch.bfloat16)},
        )
        self.kv = []

    def compute_draft_kv(self, positions, hidden_states, next_token_ids):
        self.kv.append(hidden_states)


class _Runner:
    output_stream = "the verify stream"
    outputs_deferred = None

    def __init__(self):
        self.launched, self.checks = [], []

    def check_waits(self, wait):
        self.checks.append(wait)

    def launch_verify(self, rows, qlen, hidden, input_ids):
        self.launched.append((rows, qlen, hidden, input_ids))

    def verify_outputs(self, bs):
        return ("ids", bs), ("verdict", bs)

    draft_ok = True

    def prepare_draft(self, requests, drafts):
        self.launched.append(("prepare", requests, drafts))
        return self.draft_ok

    def launch_draft(self, requests, drafts, hidden, anchors):
        self.launched.append(("draft", requests, drafts))
        return "the kernel's drafts"


class _Lazy:
    """LazyRunner stand-in: records the widths it was asked to make ready."""

    def __init__(self, create, name):
        self.asked, self.ok, self.runner = [], True, _Runner()

    def ready(self, rows, *shape):
        self.asked.append((rows, *shape))
        return self.ok


class _Inner(nn.Module):
    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        return "inner"

    def compute_logits(self, hidden_states):
        return hidden_states


def _batch(**kw):
    fields = {
        "structured_requests": None,
        "any_logprobs": False,
        "total_seqs_num": 8,
        "temperatures": np.zeros(8, dtype=np.float32),
    }
    fields.update(kw)
    return NS(**fields)


def _spec(**kw):
    fields = {
        "num_spec_steps": 7,
        "num_draft_tokens_np": np.full(8, 7),
        "input_ids": torch.zeros(64, dtype=torch.int32),
    }
    fields.update(kw)
    return NS(**fields)


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(tail, "_gfx", lambda: "gfx1250")
    monkeypatch.setattr(tail, "_relaxed_acceptance", lambda: False)
    monkeypatch.setattr(
        tail, "_has_constraints", lambda batch: batch.structured_requests is not None
    )
    monkeypatch.setattr(tail, "LazyRunner", _Lazy)
    monkeypatch.setenv("ATOM_KIMI_MONO_STEP_TAIL", "1")


@pytest.fixture
def step(monkeypatch):
    fc = NS(spec_decode_metadata=_spec(), attn_metadata=NS(max_seqlen_q=8))
    monkeypatch.setattr(tail, "get_forward_context", lambda: fc)
    return fc


@pytest.fixture
def served(step):
    drafter = _Drafter()
    return tail.StepTailDecodeModel(_Inner(), drafter, None), drafter


def _mono_installer(model, atom_config, drafter, metadata_builder=None):
    return MonoDecodeModel(model, drafter, metadata_builder)


# --- which deployments -------------------------------------------------------


def test_off_is_the_mono_installer_unchanged(monkeypatch):
    monkeypatch.setenv("ATOM_KIMI_MONO_STEP_TAIL", "0")
    seen = {}

    def install(model, cfg, drafter, metadata_builder=None):
        seen["args"] = (model, cfg, drafter, metadata_builder)
        return "what mono installed"

    monkeypatch.setattr(tail.mono_dispatch, "install_mono_decode", install)
    cfg = atom_config()
    out = tail.install_mono_decode("m", cfg, "d", metadata_builder="b")
    assert out == "what mono installed"
    assert seen["args"] == ("m", cfg, "d", "b")


def test_the_default_is_off(monkeypatch):
    monkeypatch.delenv("ATOM_KIMI_MONO_STEP_TAIL")
    monkeypatch.setattr(tail.mono_dispatch, "install_mono_decode", _mono_installer)
    out = tail.install_mono_decode(_Inner(), atom_config(), _Drafter())
    assert type(out) is MonoDecodeModel
    assert not isinstance(out, StepOverrideModel)


def test_the_covered_deployment_gets_the_step_override(monkeypatch):
    monkeypatch.setattr(tail.mono_dispatch, "install_mono_decode", _mono_installer)
    out = tail.install_mono_decode(_Inner(), atom_config(), _Drafter())
    assert isinstance(out, tail.StepTailDecodeModel)
    assert isinstance(out, StepOverrideModel) and isinstance(out, MonoDecodeModel)


@pytest.mark.parametrize(
    "change, why",
    [
        ({"tensor_parallel_size": 4}, "TP 4"),
        (
            {
                "speculative_config": NS(
                    method="mtp",
                    num_speculative_tokens=7,
                    synthetic_acceptance_rates=None,
                )
            },
            "not DSpark",
        ),
        ({"speculative_config": None}, "not DSpark"),
        (
            {
                "speculative_config": NS(
                    method="dspark",
                    num_speculative_tokens=8,
                    synthetic_acceptance_rates=None,
                )
            },
            "draft width",
        ),
        (
            {
                "speculative_config": NS(
                    method="dspark",
                    num_speculative_tokens=7,
                    synthetic_acceptance_rates=(1.0, 0.5),
                )
            },
            "synthetic acceptance",
        ),
        ({"prefill_context_parallel_size": 2}, "prefill context parallelism"),
    ],
)
def test_other_deployments_keep_the_mono_model(monkeypatch, change, why):
    monkeypatch.setattr(tail.mono_dispatch, "install_mono_decode", _mono_installer)
    cfg = atom_config(**change)
    assert why in tail.tail_refusal(cfg, _Drafter())
    out = tail.install_mono_decode(_Inner(), cfg, _Drafter())
    assert type(out) is MonoDecodeModel


@pytest.mark.parametrize(
    "attr, why",
    [("lossless", "lossless"), ("verify_scheduler", "confidence-scheduled")],
)
def test_a_drafter_the_tail_does_not_verify_for_is_refused(attr, why):
    drafter = _Drafter()
    setattr(drafter, attr, object())
    assert why in tail.tail_refusal(atom_config(), drafter)
    assert "not DSpark" in tail.tail_refusal(atom_config(), None)


def test_other_archs_and_relaxed_acceptance_are_refused(monkeypatch):
    monkeypatch.setattr(tail, "_gfx", lambda: "gfx950")
    assert tail.tail_refusal(atom_config(), _Drafter()) == "not gfx1250"
    monkeypatch.setattr(tail, "_gfx", lambda: "gfx1250")
    monkeypatch.setattr(tail, "_relaxed_acceptance", lambda: True)
    assert tail.tail_refusal(atom_config(), _Drafter()) == "relaxed acceptance"


def test_a_deployment_mono_refused_stays_as_installed(monkeypatch):
    inner = _Inner()
    monkeypatch.setattr(
        tail.mono_dispatch,
        "install_mono_decode",
        lambda m, c, d, metadata_builder=None: m,
    )
    assert tail.install_mono_decode(inner, atom_config(), _Drafter()) is inner


# --- which steps -------------------------------------------------------------


def test_a_greedy_verify_step_is_served(served, step):
    m, drafter = served
    ids = torch.zeros(64, dtype=torch.int32)
    runner = m.step_override(ids, 8, _batch())
    assert runner is m._tail.runner
    # the engine's recorded forward, then the head + verify launch on its rows
    assert drafter.runner.req_graph.replayed == [KEY]
    ((rows, qlen, hidden, drafts),) = runner.launched
    assert (rows, qlen) == (64, 8)
    assert hidden.shape == (64, L.HIDDEN) and hidden.dtype == torch.bfloat16
    assert drafts is step.spec_decode_metadata.input_ids
    # what ModelRunner.postprocess reads
    assert runner.verify_outputs(8) == (("ids", 8), ("verdict", 8))
    assert runner.output_stream == "the verify stream"


def test_a_padded_batch_is_served_at_its_running_requests(served):
    m, _ = served
    ids = torch.zeros(64, dtype=torch.int32)
    assert m.step_override(ids, 8, _batch(total_seqs_num=5)) is m._tail.runner
    assert m._tail.asked == [(64, 8)]


@pytest.mark.parametrize(
    "batch",
    [
        None,
        _batch(any_logprobs=True),
        _batch(structured_requests=[1]),
        _batch(temperatures=np.array([0, 0, 0.7, 0, 0, 0, 0, 0], dtype=np.float32)),
    ],
)
def test_steps_needing_logits_or_sampling_take_the_production_path(served, batch):
    m, drafter = served
    assert m.step_override(torch.zeros(64, dtype=torch.int32), 8, batch) is None
    assert drafter.runner.req_graph.replayed == []
    assert m._tail.runner.launched == [] and m._tail.asked == []


def test_a_sampled_pad_row_does_not_refuse_the_step(served):
    m, _ = served
    hot_pad = np.array([0, 0, 0, 0, 0, 0.9, 0.9, 0.9], dtype=np.float32)
    batch = _batch(total_seqs_num=5, temperatures=hot_pad)
    assert m.step_override(torch.zeros(64, dtype=torch.int32), 8, batch) is not None


@pytest.mark.parametrize(
    "spec",
    [
        None,
        _spec(num_draft_tokens_np=np.array([7, 7, 6, 7, 7, 7, 7, 7])),
        _spec(num_spec_steps=4),  # 64 rows are not 8 requests of 5
        _spec(input_ids=torch.zeros(64, dtype=torch.int64)),
        _spec(input_ids=torch.zeros(32, dtype=torch.int32)),
    ],
)
def test_a_step_that_is_not_a_uniform_verify_takes_the_production_path(
    served, step, spec
):
    m, drafter = served
    step.spec_decode_metadata = spec
    assert m.step_override(torch.zeros(64, dtype=torch.int32), 8, _batch()) is None
    assert drafter.runner.req_graph.replayed == [] and m._tail.runner.launched == []


def test_rows_that_are_not_whole_requests_take_the_production_path(served):
    m, drafter = served
    assert m.step_override(torch.zeros(56, dtype=torch.int32), 8, _batch()) is None
    assert m.step_override(torch.zeros(64, dtype=torch.int32), 7, _batch()) is None
    assert drafter.runner.req_graph.replayed == []


def test_a_step_without_a_recorded_forward_takes_the_production_path(served):
    m, drafter = served
    drafter.runner.req_graph.keys = set()
    assert m.step_override(torch.zeros(64, dtype=torch.int32), 8, _batch()) is None
    assert m._tail.runner.launched == []


def test_hidden_rows_the_kernel_cannot_read_take_the_production_path(served):
    m, drafter = served
    drafter.runner.forward_vars["outputs"] = torch.zeros(64, L.HIDDEN)  # fp32
    assert m.step_override(torch.zeros(64, dtype=torch.int32), 8, _batch()) is None
    assert drafter.runner.req_graph.replayed == []


def test_a_width_without_a_build_takes_the_production_path(served):
    m, drafter = served
    m._tail.ok = False
    assert m.step_override(torch.zeros(64, dtype=torch.int32), 8, _batch()) is None
    assert m._tail.asked == [(64, 8)]
    assert drafter.runner.req_graph.replayed == [] and m._tail.runner.launched == []


# --- the drafter and the forward --------------------------------------------


def test_a_served_step_feeds_the_context_kv_the_forwards_rows(served):
    m, drafter = served
    m.step_override(torch.zeros(64, dtype=torch.int32), 8, _batch())
    drafter.compute_draft_kv("positions", None, None)
    (hidden,) = drafter.kv
    assert hidden.shape == (64, L.HIDDEN)


def test_a_production_step_leaves_the_drafter_alone(served):
    m, drafter = served
    assert (
        m.step_override(
            torch.zeros(64, dtype=torch.int32), 8, _batch(any_logprobs=True)
        )
        is None
    )
    drafter.compute_draft_kv("positions", "the engine's rows", None)
    drafter.compute_draft_kv("positions", None, None)
    assert drafter.kv == ["the engine's rows", None]


def test_a_declined_step_after_a_served_one_does_not_reuse_its_rows(served):
    m, drafter = served
    m.step_override(torch.zeros(64, dtype=torch.int32), 8, _batch())
    assert (
        m.step_override(
            torch.zeros(64, dtype=torch.int32), 8, _batch(any_logprobs=True)
        )
        is None
    )
    drafter.compute_draft_kv("positions", None, None)
    assert drafter.kv == [None]


def test_a_forward_of_the_engines_own_ends_the_served_step(served, monkeypatch):
    m, drafter = served
    m.step_override(torch.zeros(64, dtype=torch.int32), 8, _batch())
    monkeypatch.setattr(
        tail.mono_dispatch, "get_forward_context", lambda: NS(context=None)
    )
    ids = torch.zeros(3, dtype=torch.long)
    assert m(ids, ids) == "inner"  # a prefill: the production forward
    drafter.compute_draft_kv("positions", None, None)
    assert drafter.kv == [None]
    assert m.compute_logits(3) == 3


# --- a wait that gave up -----------------------------------------------------


def test_a_served_step_tells_the_runner_when_the_engine_reads_its_ids(served):
    m, drafter = served
    ids = torch.zeros(64, dtype=torch.int32)
    assert m.step_override(ids, 8, _batch()).outputs_deferred is True
    drafter.runner.tokenID_processor = NS(is_deferred_out=False)
    assert m.step_override(ids, 8, _batch()).outputs_deferred is False
    # an engine that does not say reads them at once, as far as the tail knows
    del drafter.runner.tokenID_processor
    assert m.step_override(ids, 8, _batch()).outputs_deferred is False
    # a served step looks at what has landed before it replays or launches;
    # the runner settles the rest where it hands the outputs over
    assert m._tail.runner.checks == [False, False, False]
    assert len(drafter.runner.req_graph.replayed) == 3


def test_a_step_left_to_the_engine_settles_the_served_steps_waits_first(
    served, monkeypatch
):
    m, _ = served
    assert m.step_override(torch.zeros(64, dtype=torch.int32), 8, None) is None
    assert m._tail.runner.checks == [True]
    monkeypatch.setattr(
        tail.mono_dispatch, "get_forward_context", lambda: NS(context=None)
    )
    ids = torch.zeros(3, dtype=torch.long)
    assert m(ids, ids) == "inner"
    assert m._tail.runner.checks == [True, True]


def test_a_wait_that_gave_up_fails_the_step(served):
    m, drafter = served

    def gave_up(wait):
        raise RuntimeError(
            "Kimi-K3 step tail: 1 cross-rank waits gave up so far (rank 5, last site 2)"
        )

    m._tail.runner.check_waits = gave_up
    ids = torch.zeros(64, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="gave up"):  # a served step
        m.step_override(ids, 8, _batch())
    with pytest.raises(RuntimeError, match="gave up"):  # a step left to the engine
        m.step_override(ids, 8, None)
    with pytest.raises(RuntimeError, match="gave up"):  # a forward of the engine's
        m(ids, ids)
    assert drafter.runner.req_graph.replayed == []
    assert m._tail.runner.launched == []


def test_before_the_runner_exists_there_is_nothing_to_settle(served):
    m, _ = served
    m._tail.runner, m._tail.ok = None, False
    assert m.step_override(torch.zeros(64, dtype=torch.int32), 8, None) is None


def test_the_tail_is_built_with_the_widths_mono_forward(served, monkeypatch):
    m, _ = served
    monkeypatch.setattr(
        MonoDecodeModel, "_mono_forward", lambda self, i, p, q: "hidden"
    )
    assert m._mono_forward(torch.zeros(16), None, 8) == "hidden"
    assert (
        m._mono_forward(torch.zeros(4), None, 1) == "hidden"
    )  # a plain decode: no drafts
    assert m._tail.asked == [(16, 8)]


def test_the_wrapper_forwards_the_model_attributes(served):
    m, _ = served
    m.model.extra = 7
    assert m.extra == 7


# --- the outputs' layout -----------------------------------------------------


def test_the_outputs_are_the_rejection_samplers_shapes():
    requests, qlen, bs = 8, 8, 5
    outtok = torch.arange(L.MAX_REQ * L.MAX_QLEN, dtype=torch.int32)
    verdict = torch.arange(3 * L.MAX_REQ, dtype=torch.int32)
    ids, v = step_outputs(outtok, verdict, requests, qlen, bs)
    assert ids.shape == (bs, qlen) and v.shape == (3, bs)
    # request r's ids are its qlen words; the verdict's rows are requests wide
    assert ids[3].tolist() == list(range(3 * qlen, 4 * qlen))
    assert v[1].tolist() == list(range(requests, requests + bs))
    assert v[2].tolist() == list(range(2 * requests, 2 * requests + bs))


# --- the draft block's epilogue ------------------------------------------------


class _DraftModel:
    draft_sampling = None
    confidence_head = None

    def head_and_sample(self, out, anchor_ids, num_draft, *, greedy=False):
        return ("production's drafts", greedy), None


@pytest.fixture
def drafting():
    drafter = _Drafter()
    drafter.model = _DraftModel()
    m = tail.StepTailDecodeModel(_Inner(), drafter, None)
    out = torch.zeros(56, L.HIDDEN, dtype=torch.bfloat16)
    anchors = torch.zeros(8, dtype=torch.int32)
    return m, drafter.model, out, anchors


def test_a_draft_block_with_a_build_is_sampled_by_the_kernel(drafting):
    m, model, out, anchors = drafting
    assert model.head_and_sample(out, anchors, 7) == ("the kernel's drafts", None)
    assert m._tail.runner.launched == [("prepare", 8, 7), ("draft", 8, 7)]
    # the greedy variant of an armed drafter is the same launch
    model.draft_sampling = object()
    assert (
        model.head_and_sample(out, anchors, 7, greedy=True)[0] == "the kernel's drafts"
    )


def test_a_block_without_a_build_keeps_the_models_head(drafting):
    m, model, out, anchors = drafting
    m._tail.runner.draft_ok = False
    assert model.head_and_sample(out, anchors, 7) == (
        ("production's drafts", False),
        None,
    )
    assert m._tail.runner.launched == [("prepare", 8, 7)]


def test_before_the_runner_exists_the_model_drafts(drafting):
    m, model, out, anchors = drafting
    m._tail.runner = None
    got = model.head_and_sample(out, anchors, 7, greedy=True)
    assert got == (("production's drafts", True), None)


@pytest.mark.parametrize(
    "change",
    [
        {"draft_sampling": object()},  # a sampled block publishes a q
        {"confidence_head": object()},  # it reads the rows the launch does not keep
    ],
)
def test_a_draft_the_launch_does_not_cover_keeps_the_models_head(drafting, change):
    m, model, out, anchors = drafting
    for name, value in change.items():
        setattr(model, name, value)
    assert model.head_and_sample(out, anchors, 7)[0] == ("production's drafts", False)
    assert m._tail.runner.launched == []


def test_rows_or_anchors_the_kernel_cannot_read_keep_the_models_head(drafting):
    m, model, out, anchors = drafting
    for bad_out, bad_anchors in (
        (out.float(), anchors),
        (out[:49], anchors),
        (out, anchors.long()),
        ((out, out), anchors),
    ):
        got = model.head_and_sample(bad_out, bad_anchors, 7)
        assert got[0] == ("production's drafts", False)
    assert m._tail.runner.launched == []


def test_a_drafter_without_a_draft_model_is_left_alone(served):
    _, drafter = served
    assert not hasattr(drafter, "model")
