# SPDX-License-Identifier: MIT
"""Opt-in FlyDSL paged-decode plan shape for head_dim 256 (Qwen3.5).

ATOM_FLYDSL_PA_LOWBS_CAP: plan ceiling per batch size (bs1 256, bs2 128, else
64) for capture-owned and runtime plans. Wiring only; numerics and timing are
GPU work.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace

import pytest

pytest.importorskip("triton", reason="base_attention defines @triton.jit kernels")
pytest.importorskip("aiter", reason="base_attention imports the AITER runtime")

try:
    _HAS_FLYDSL_PA = importlib.util.find_spec("aiter.ops.flydsl.pa_decode") is not None
except ModuleNotFoundError:
    _HAS_FLYDSL_PA = False

pytestmark = pytest.mark.skipif(
    not _HAS_FLYDSL_PA, reason="needs aiter FlyDSL pa_decode"
)

_CAP64 = {"max_partitions": 64, "workgroup_budget": 256}


def _fake_ctx(n):
    import torch

    return SimpleNamespace(
        shape=(n,),
        device=SimpleNamespace(index=0),
        dtype=torch.int32,
        is_cuda=True,
        ndim=1,
        is_contiguous=lambda: True,
    )


class _Ctx:
    """_fake_ctx that slices, as the op does with context_lens[:n]."""

    def __init__(self, n):
        self.__dict__.update(vars(_fake_ctx(n)))

    def __getitem__(self, s):
        return _Ctx(len(range(*s.indices(self.shape[0]))))


class _Plan:
    def __init__(self, n, max_partitions):
        import torch

        self.reduce_info = torch.empty((n, 1), device="meta")
        self.num_kv_heads = 1
        self.max_partitions = max_partitions
        self.capacity = 256


class TestPlanShape:
    @pytest.mark.parametrize(
        "lowbs, num_seqs, want",
        [
            ("0", 1, _CAP64),
            ("0", 2, _CAP64),
            ("1", 1, {"max_partitions": 256, "workgroup_budget": 1024}),
            ("1", 2, {"max_partitions": 128, "workgroup_budget": 512}),
            ("1", 3, _CAP64),
            ("1", 4, _CAP64),
            ("1", 8, _CAP64),
            # No batch known: the fixed cap, as before.
            ("1", None, _CAP64),
        ],
    )
    def test_head_256(self, monkeypatch, lowbs, num_seqs, want):
        from atom.model_ops.base_attention import flydsl_plan_shape

        monkeypatch.setenv("ATOM_FLYDSL_PA_LOWBS_CAP", lowbs)
        assert flydsl_plan_shape(256, num_seqs) == want

    def test_default_is_off(self, monkeypatch):
        from atom.model_ops.base_attention import flydsl_plan_shape

        monkeypatch.delenv("ATOM_FLYDSL_PA_LOWBS_CAP", raising=False)
        assert flydsl_plan_shape(256, 1) == _CAP64

    @pytest.mark.parametrize("head_dim", [128, 64, None])
    def test_other_heads_keep_aiter_defaults(self, monkeypatch, head_dim):
        from atom.model_ops.base_attention import flydsl_plan_shape

        monkeypatch.setenv("ATOM_FLYDSL_PA_LOWBS_CAP", "1")
        assert flydsl_plan_shape(head_dim, 1) == {}

    def test_returned_shape_is_a_copy(self, monkeypatch):
        from atom.model_ops import base_attention as ba

        monkeypatch.setenv("ATOM_FLYDSL_PA_LOWBS_CAP", "1")
        ba.flydsl_plan_shape(256, 1)["max_partitions"] = 7
        assert ba._FLYDSL_D256_LOWBS_SHAPE[1]["max_partitions"] == 256


class TestCapturePlans:
    """Each captured batch size owns a plan built for it (key holds the batch)."""

    def _builder(self, monkeypatch, head_dim=256):
        from atom.model_ops.attentions import aiter_attention as aa
        from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry

        seen = []

        def fake_plan(context_lens, num_kv_heads, **kwargs):
            seen.append((int(context_lens.shape[0]), kwargs))
            return SimpleNamespace(
                max_partitions=kwargs.get("max_partitions", 256), capacity=256
            )

        monkeypatch.setattr(
            importlib.import_module("aiter.ops.flydsl.pa_decode"),
            "plan_pa_decode",
            fake_plan,
        )
        builder = aa.AiterAttentionMetadataBuilder.__new__(
            aa.AiterAttentionMetadataBuilder
        )
        builder._flydsl_kv_heads = 1
        builder._flydsl_plans = {}
        builder._flydsl_plan_unplanned = False
        builder.kv_pools = {KvGeometry(1, head_dim): None}
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL", True)
        return builder, seen

    def test_ladder_with_switch_on(self, monkeypatch):
        builder, seen = self._builder(monkeypatch)
        monkeypatch.setenv("ATOM_FLYDSL_PA_LOWBS_CAP", "1")
        for n in (1, 2, 4, 8):
            assert builder.refresh_flydsl_plan(_fake_ctx(n), create=True)
        assert seen == [
            (1, {"query_length": 1, "max_partitions": 256, "workgroup_budget": 1024}),
            (2, {"query_length": 1, "max_partitions": 128, "workgroup_budget": 512}),
            (4, {"query_length": 1, **_CAP64}),
            (8, {"query_length": 1, **_CAP64}),
        ]
        # Replay refreshes the rung's own plan in place; no new shape.
        plan1 = builder._flydsl_plans[(1, 1, 0)]
        assert builder.refresh_flydsl_plan(_fake_ctx(1)) is plan1
        assert seen[-1] == (1, {"query_length": 1, "plan": plan1})

    def test_ladder_with_switch_off(self, monkeypatch):
        builder, seen = self._builder(monkeypatch)
        monkeypatch.delenv("ATOM_FLYDSL_PA_LOWBS_CAP", raising=False)
        for n in (1, 2, 4):
            builder.refresh_flydsl_plan(_fake_ctx(n), create=True)
        assert [kw for _, kw in seen] == [{"query_length": 1, **_CAP64}] * 3


class TestRuntimePlans:
    """A batch with no capture-owned plan gets the same per-batch shape."""

    def _setup(self, monkeypatch):
        import torch

        from atom.model_ops import base_attention as ba

        built = []

        def fake_plan(context_lens, num_kv_heads, **kw):
            if kw.get("plan") is not None:
                return kw["plan"]
            built.append((int(context_lens.shape[0]), kw))
            return _Plan(int(context_lens.shape[0]), kw.get("max_partitions"))

        monkeypatch.setattr(
            importlib.import_module("aiter.ops.flydsl.pa_decode"),
            "plan_pa_decode",
            fake_plan,
        )
        monkeypatch.setattr(ba, "_FLYDSL_RUNTIME_PLANS", {})
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda device=None: SimpleNamespace(cuda_stream=7),
        )
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        return ba, built

    def _call(self, ba, n, existing_plan=None, cap=None, head_dim=256):
        import torch

        return ba._flydsl_prepare_explicit_plan(
            q=torch.empty((n * 6, 8, head_dim), dtype=torch.bfloat16, device="meta"),
            k_cache=torch.empty((64, 1, 32, 16, 8), device="meta"),
            context_lens=_Ctx(n),
            num_seqs=n,
            query_length=6,
            plan_partition_cap=cap,
            context_bound=2048,
            existing_plan=existing_plan,
        )

    def test_per_batch_shape(self, monkeypatch):
        ba, built = self._setup(monkeypatch)
        monkeypatch.setenv("ATOM_FLYDSL_PA_LOWBS_CAP", "1")
        for n in (1, 2, 5):
            self._call(ba, n)
        assert [
            (n, kw["max_partitions"], kw["workgroup_budget"]) for n, kw in built
        ] == [
            (1, 256, 1024),
            (2, 128, 512),
            (5, 64, 256),
        ]

    def test_caller_cap_is_kept(self, monkeypatch):
        """Static-cap callers (M3 sparse, vLLM bridge) are not touched."""
        ba, built = self._setup(monkeypatch)
        monkeypatch.setenv("ATOM_FLYDSL_PA_LOWBS_CAP", "1")
        self._call(ba, 1, cap=8)
        assert built[0][1]["max_partitions"] == 8
        assert "workgroup_budget" not in built[0][1]

    def test_matching_plan_is_returned_before_any_shape(self, monkeypatch):
        ba, built = self._setup(monkeypatch)
        monkeypatch.setenv("ATOM_FLYDSL_PA_LOWBS_CAP", "1")
        calls = []
        real = ba.flydsl_plan_shape
        monkeypatch.setattr(
            ba, "flydsl_plan_shape", lambda *a: calls.append(a) or real(*a)
        )
        captured = _Plan(1, 256)
        assert self._call(ba, 1, existing_plan=captured) is captured
        assert built == [] and calls == []
