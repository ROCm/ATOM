# SPDX-License-Identifier: MIT

"""PAGE-backed state checkpoints under pipeline parallelism.

Only the head stage schedules, so every stage receives the same unit ids for
one image. The stages agree on the geometry once, at sizing, and every op the
head builds must pass each stage's own validation.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="model_runner and gdn_attn import aiter")

from atom.model_engine.model_runner import ModelRunner
from atom.model_engine.page_unit_checkpoint import (
    CheckpointStoreOp,
    PagedStateCheckpointSpec,
)
from atom.model_ops.attentions.gdn_attn import GDNStateMixin

# Kimi-K3 PP8 x TP1: (page unit bytes, KDA slot bytes) per stage.
K3_PP8 = [
    (147456, 30302208),
    (221184, 26935296),
    *[(221184, 30302208)] * 5,
    (294912, 23568384),
]


def _stage_spec(unit_bytes: int, slot_bytes: int, stage: int):
    return PagedStateCheckpointSpec(
        page_unit_bytes=unit_bytes,
        slot_bytes=slot_bytes,
        image_bytes=slot_bytes,
        layout_id=f"kda-stage-{stage}",
    )


def _agree(monkeypatch, stage: int, geometry=K3_PP8, tp: int = 1):
    specs = [_stage_spec(u, s, i) for i, (u, s) in enumerate(geometry)]
    per_rank = [
        (
            spec.layout_id,
            spec.image_bytes,
            spec.page_unit_bytes,
            spec.units_per_checkpoint,
        )
        for spec in specs
        for _ in range(tp)
    ]

    def all_gather_object(out, _mine):
        out[:] = per_rank

    dist = pytest.importorskip("torch").distributed
    monkeypatch.setattr(dist, "get_world_size", lambda: len(per_rank))
    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    runner = SimpleNamespace(
        config=SimpleNamespace(pipeline_parallel_size=len(geometry))
    )
    return specs[stage], ModelRunner._agree_pipeline_checkpoint_spec(
        runner, specs[stage]
    )


@pytest.mark.parametrize("tp", [1, 2])
def test_stages_take_the_largest_unit_count(monkeypatch, tp):
    own, agreed = _agree(monkeypatch, stage=7, tp=tp)

    assert own.units_per_checkpoint == 80
    assert agreed.units_per_checkpoint == 206
    assert agreed.image_bytes == own.image_bytes
    assert agreed.page_unit_bytes == own.page_unit_bytes
    assert agreed.op_bytes == sum(slot for _, slot in K3_PP8)


def test_every_stage_reaches_the_same_identity(monkeypatch):
    agreed = [_agree(monkeypatch, stage=i)[1] for i in range(len(K3_PP8))]

    assert len({spec.layout_id for spec in agreed}) == 1
    assert len({spec.units_per_checkpoint for spec in agreed}) == 1
    assert len({spec.op_bytes for spec in agreed}) == 1
    assert agreed[0].layout_id.startswith("pp8-joint:")


def test_the_identity_moves_with_any_stage_geometry(monkeypatch):
    _, base = _agree(monkeypatch, stage=0)
    moved = list(K3_PP8)
    moved[5] = (221184, 26935296)
    _, other = _agree(monkeypatch, stage=0, geometry=moved)

    assert base.layout_id != other.layout_id


def _validating_stage(spec: PagedStateCheckpointSpec, num_blocks: int = 1000):
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            state_runtime=SimpleNamespace(checkpoint_spec=spec),
            config=SimpleNamespace(num_kvcache_blocks=num_blocks),
        )
    )


def test_a_head_op_passes_every_stage_validation(monkeypatch):
    agreed = [_agree(monkeypatch, stage=i)[1] for i in range(len(K3_PP8))]
    head = agreed[0]
    op = CheckpointStoreOp(
        src_slot=0,
        unit_ids=tuple(range(head.units_per_checkpoint)),
        total_bytes=head.op_bytes,
        layout_id=head.layout_id,
    )

    for spec in agreed:
        GDNStateMixin._validate_paged_state_op(_validating_stage(spec), op)


def test_a_stage_own_op_is_refused_under_the_joint_geometry(monkeypatch):
    own, agreed = _agree(monkeypatch, stage=3)
    op = CheckpointStoreOp(
        src_slot=0,
        unit_ids=tuple(range(own.units_per_checkpoint)),
        total_bytes=own.image_bytes,
        layout_id=own.layout_id,
    )

    with pytest.raises(RuntimeError, match="layout mismatch"):
        GDNStateMixin._validate_paged_state_op(_validating_stage(agreed), op)
