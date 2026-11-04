# SPDX-License-Identifier: MIT
"""A decode step planned for the opus MLA decode skips aiter's MLA work
metadata, through the real ``AiterMLAMetadataBuilder.prepare_decode`` (the
H2D test harness). The opus decoder is a recorder: no aiter kernel runs."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU (the H2D harness)", allow_module_level=True)

from test_h2d_attention_publication import (
    decode_batch,
    finish,
    make_runner,
    start_decode,
)

from atom.model_ops import opus_mla


class _Decoder:
    def __init__(self, max_batch):
        self.plans = []

    def plan(self, kv_indptr, q_len, num_heads, cp_world_size=1):
        self.plans.append((q_len, num_heads, cp_world_size))

    def clear(self):
        pass


@pytest.fixture
def opus(monkeypatch):
    monkeypatch.setenv("ATOM_OPUS_MLA", "1")
    monkeypatch.setattr(opus_mla, "OpusMlaFlatDecode", _Decoder)
    monkeypatch.setattr(opus_mla, "_decoder", None)


def _runner(monkeypatch, *, q=6, dcp=1):
    runner, builder = make_runner(monkeypatch, q=q, dcp=dcp, tbo=False)
    builds = []

    def build(*args, **kw):
        builder._mla_meta_pending = None  # as the real one does
        builds.append((args, kw))
        return {}

    builder.set_mla_persistent_worker_buffers = build
    return runner, builder, builds


def _step(runner, builder, q, count=4):
    batch = decode_batch(count, q=q)
    start_decode(runner, builder, batch, q)
    md, _ = builder.prepare_decode(batch, count, count * q, q)
    finish(runner)
    return md


def test_opus_step_skips_the_metadata(monkeypatch, opus):
    runner, builder, builds = _runner(monkeypatch)
    md = _step(runner, builder, 6)
    assert md.opus_plan is not None and md.opus_plan.heads == 16
    assert builds == []
    assert md.mla_meta_hook == builder._mla_meta_for_call
    assert builder._mla_meta_pending[0] is md
    for name in ("work_indptr", "work_meta_data", "reduce_partial_map"):
        assert getattr(md, name) is runner.forward_vars[name]


def test_a_call_that_keeps_mla_decode_fwd_builds_it(monkeypatch, opus):
    runner, builder, builds = _runner(monkeypatch)
    md = _step(runner, builder, 6)
    md.mla_meta_hook(md)
    assert len(builds) == 1
    (bs, q_len), kw = builds[0]
    assert (bs, q_len, kw) == (4, 6, {"is_cp_round_robin": False})
    md.mla_meta_hook(md)  # once a step
    assert len(builds) == 1


def test_steps_off_opus_build_as_before(monkeypatch, opus):
    runner, builder, builds = _runner(monkeypatch, q=1)  # no MTP: no opus
    md = _step(runner, builder, 1)
    assert md.opus_plan is None and len(builds) == 1
    assert builder._mla_meta_pending is None


def test_switch_off_builds_as_before(monkeypatch):
    monkeypatch.delenv("ATOM_OPUS_MLA", raising=False)
    runner, builder, builds = _runner(monkeypatch)
    md = _step(runner, builder, 6)
    assert md.opus_plan is None and len(builds) == 1


def test_an_mtp_drafter_without_dcp_keeps_the_build(monkeypatch, opus):
    from atom.spec_decode.eagle_proposer import EagleProposer

    runner, builder, builds = _runner(monkeypatch)
    runner.drafter = EagleProposer.__new__(EagleProposer)
    md = _step(runner, builder, 6)
    assert md.opus_plan is not None and len(builds) == 1
    assert builder._mla_meta_pending is None
