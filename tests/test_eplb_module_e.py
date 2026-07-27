# SPDX-License-Identifier: MIT
# Tests for atom/model_ops/eplb.py (Module-E commit / visibility)

import pytest

torch = pytest.importorskip("torch")

# Keep config import order consistent; skip if the full atom import env
# (aiter/triton) is unavailable.
try:
    import atom.config  # noqa: F401
    import atom.model_ops.eplb as eplb
except Exception as _e:  # aiter/triton absent under bare non-GPU pytest
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)


def test_move_from_buffer_applies_plan_inplace():
    temp = [torch.tensor([[10.0], [20.0]], dtype=torch.float32)]
    weight = [torch.tensor([[1.0], [2.0]], dtype=torch.float32)]
    weight_obj = weight[0]

    # Copy temp[1] -> weight[0], temp[0] -> weight[1]
    plan = [(1, 0), (0, 1)]
    eplb.move_from_buffer(plan, temp, weight)

    assert weight[0] is weight_obj
    assert weight[0][0, 0].item() == pytest.approx(20.0)
    assert weight[0][1, 0].item() == pytest.approx(10.0)


def test_move_from_buffer_supports_free_rider_plan():
    temp = [torch.tensor([[7.0], [99.0]], dtype=torch.float32)]
    weight = [torch.zeros((2, 1), dtype=torch.float32)]
    # free-rider: two dst slots reuse one source slot
    plan = [(0, 0), (0, 1)]
    eplb.move_from_buffer(plan, temp, weight)
    assert weight[0][0, 0].item() == pytest.approx(7.0)
    assert weight[0][1, 0].item() == pytest.approx(7.0)


def test_commit_layer_calls_meta_update_after_copy():
    temp = [torch.tensor([[3.0], [4.0]], dtype=torch.float32)]
    weight = [torch.tensor([[0.0], [0.0]], dtype=torch.float32)]
    plan = [(1, 0), (0, 1)]

    class _LiveMeta:
        def __init__(self):
            self.calls = []

        def update(self, new_meta, layer_ids):
            # Verify copy is visible before map switch.
            assert weight[0][0, 0].item() == pytest.approx(4.0)
            assert weight[0][1, 0].item() == pytest.approx(3.0)
            self.calls.append((new_meta, list(layer_ids)))

    live_meta = _LiveMeta()
    new_meta = object()
    eplb.commit_layer(
        plan=plan,
        temp_buffers=temp,
        expert_weights=weight,
        live_meta=live_meta,
        new_meta=new_meta,
        layer_id=7,
    )

    assert live_meta.calls == [(new_meta, [7])]


def test_commit_experts_chunk_uses_upper_plans_explicitly():
    temp = [torch.tensor([[1.0], [2.0]], dtype=torch.float32)]
    w0 = [torch.tensor([[0.0], [0.0]], dtype=torch.float32)]
    w1 = [torch.tensor([[0.0], [0.0]], dtype=torch.float32)]
    plans = {0: [(1, 0)], 1: [(0, 1)]}
    calls = []

    class _LiveMeta:
        def update(self, new_meta, layer_ids):
            calls.append((new_meta, list(layer_ids)))

    live = _LiveMeta()
    new = object()
    eplb.commit_experts_chunk(
        layer_ids=[0, 1],
        plans=plans,
        temp_buffers=temp,
        expert_weights_of_layer={0: w0, 1: w1},
        live_meta=live,
        new_meta=new,
    )

    assert w0[0][0, 0].item() == pytest.approx(2.0)
    assert w1[0][1, 0].item() == pytest.approx(1.0)
    assert calls == [(new, [0]), (new, [1])]


def test_migrate_and_commit_chunk_passes_same_stream(monkeypatch):
    class _Meta:
        def __init__(self, p2l):
            self.physical_to_logical_map = p2l

    old = _Meta(torch.tensor([[0, 1]], dtype=torch.int32))
    new = _Meta(torch.tensor([[1, 0]], dtype=torch.int32))
    live = type("LiveMeta", (), {"update": lambda self, nm, ls: None})()
    temp = [torch.zeros((2, 1), dtype=torch.float32)]
    weights = {0: [torch.zeros((2, 1), dtype=torch.float32)]}
    stream_obj = object()
    seen = {}

    def _fake_migrate(**kwargs):
        seen["migrate_stream"] = kwargs["cuda_stream"]
        return {0: [(0, 0)]}

    def _fake_commit(**kwargs):
        seen["commit_stream"] = kwargs["cuda_stream"]
        assert kwargs["plans"] == {0: [(0, 0)]}

    monkeypatch.setattr(eplb, "migrate_experts_chunk", _fake_migrate)
    monkeypatch.setattr(eplb, "commit_experts_chunk", _fake_commit)
    eplb.migrate_and_commit_chunk(
        layer_ids=[0],
        old_meta=old,
        new_meta=new,
        expert_weights_of_layer=weights,
        temp_buffers=temp,
        ep_group=None,
        nnodes=1,
        rank=0,
        live_meta=live,
        cuda_stream=stream_obj,
    )
    assert seen["migrate_stream"] is stream_obj
    assert seen["commit_stream"] is stream_obj
