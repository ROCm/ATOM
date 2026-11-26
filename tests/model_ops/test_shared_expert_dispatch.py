from types import SimpleNamespace

import pytest
import torch
from conftest import atom_config_double

pytest.importorskip("aiter", reason="full dispatch tests require aiter")

from aiter import QuantType, dtypes

import atom.model_ops.eplb as eplb_module
import atom.model_ops.moe as moe_module
import atom.model_ops.topK as topK_module
from atom.model_ops.fused_moe import shared_expert_dispatch as dispatch_module
from atom.model_ops.fused_moe.expert_layout import MoEExpertLayout
from atom.model_ops.moe import FusedMoE, FusedMoEParallelConfig, Mxfp4MoEMethod
from atom.model_ops.topK import (
    is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config,
)
from atom.quant_spec import LayerQuantConfig, fuses_ptpc_shared_into_online_ptpc


class _FakeMoELayer(SimpleNamespace):
    shared_dispatch_base = FusedMoE.shared_dispatch_base
    shared_expert_weight = FusedMoE.shared_expert_weight


def _dispatch_layer(*, ep_rank: int = 1) -> _FakeMoELayer:
    layout = MoEExpertLayout.make(
        num_routed=8,
        num_fused_shared_experts=1,
        num_configured_redundant=0,
        ep_size=2,
        use_all2all=True,
        eplb_enabled=False,
    )
    ns = _FakeMoELayer(
        expert_layout=layout,
        num_fused_shared_experts=1,
        global_num_experts=10,
        num_redundant_experts=0,
        routed_scaling_factor=2.0,
        ep_size=2,
        ep_rank=ep_rank,
        local_num_experts=5,
        layer_id=None,
        use_ep=True,
        expert_map=torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3, 4, -1]),
        expert_mask=torch.empty(0, dtype=torch.int32),
        # routing parameters `select_experts` reads off the layer
        top_k=2,
        renormalize=True,
        use_grouped_topk=False,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        shared_expert_scoring_func=None,
    )
    return ns


def test_to_dispatch_space_is_backend_neutral(monkeypatch):
    layer = _dispatch_layer(ep_rank=1)
    monkeypatch.setattr(dispatch_module, "_HAS_TRITON", False)
    monkeypatch.setattr(
        moe_module, "is_rocm_aiter_fuse_routed_scaling_factor", lambda: False
    )
    # No GPU here; pin to the torch reference.
    # The last row is a pad row: routed to -1 throughout, so no shared expert.
    routed_ids = torch.tensor([[0, 4], [3, -1], [-1, -1]], dtype=torch.int32)
    routed_weights = torch.tensor([[0.7, 0.3], [0.6, 0.0], [0.0, 0.0]])

    weights, ids = FusedMoE.to_dispatch_space(layer, routed_weights, routed_ids)

    assert ids.tolist() == [[0, 5, 9], [3, -1, 9], [-1, -1, -1]]
    assert torch.equal(weights[:, :2], routed_weights)
    assert weights[:, 2].tolist() == [0.5, 0.5, 0.0]


def test_select_experts_keeps_shared_out_of_router_and_eplb(monkeypatch):
    layer = _dispatch_layer(ep_rank=1)
    monkeypatch.setattr(dispatch_module, "_HAS_TRITON", False)
    captured = {}
    routed_weights = torch.tensor([[0.75, 0.25]])
    routed_ids = torch.tensor([[0, 4]], dtype=torch.int32)

    def fake_select_experts(**kwargs):
        captured["num_fused_shared_experts"] = kwargs["num_fused_shared_experts"]
        return routed_weights, routed_ids

    def fake_map_and_record(received_layer, received_ids, gathered):
        # EPLB must see routed ids only.
        assert received_layer is layer
        assert received_ids.shape[1] == 2
        captured["eplb_mapped"] = gathered
        return received_ids

    monkeypatch.setattr(FusedMoE, "select_experts", fake_select_experts)
    # moe.py binds the name at import: patch it where it is called.
    monkeypatch.setattr(moe_module, "eplb_map_and_record_fused", fake_map_and_record)
    monkeypatch.setattr(
        moe_module, "is_rocm_aiter_fuse_routed_scaling_factor", lambda: True
    )
    # Pins the ordering router -> EPLB -> shared, not the kernel arithmetic.
    layer.to_dispatch_space = lambda weights, ids: FusedMoE.to_dispatch_space(
        layer, weights, ids
    )

    method = object.__new__(Mxfp4MoEMethod)
    method.use_flydsl_herd = False
    weights, ids = Mxfp4MoEMethod.select_experts(
        method, layer, torch.empty(1, 4), torch.empty(1, 8)
    )
    weights, ids = Mxfp4MoEMethod.place_experts(method, layer, weights, ids)

    assert captured["num_fused_shared_experts"] == 0
    # Rows no gather moved: the replica spread mixes in this rank.
    assert captured["eplb_mapped"] is False
    assert ids.tolist() == [[0, 5, 9]]
    assert weights.tolist() == [[0.75, 0.25, 1.0]]


def test_eplb_appends_unscaled_shared_weight(monkeypatch):
    layer = _dispatch_layer()
    captured = {}
    layer.expert_layout = MoEExpertLayout.make(
        num_routed=8,
        num_fused_shared_experts=1,
        num_configured_redundant=0,
        ep_size=2,
        use_all2all=True,
        eplb_enabled=True,
    )

    def fake_select_experts(**kwargs):
        captured["scale"] = kwargs["routed_scaling_factor"]
        # The second row is a pad row: routed to -1 throughout.
        return (
            torch.tensor([[1.5, 0.5], [0.0, 0.0]]),
            torch.tensor([[0, 4], [-1, -1]], dtype=torch.int32),
        )

    monkeypatch.setattr(FusedMoE, "select_experts", fake_select_experts)
    # The CPU twin of the Triton kernel; no EPLB metadata, so no remap.
    monkeypatch.setattr(eplb_module, "_EPLB_HAS_TRITON", False)
    monkeypatch.setattr(
        moe_module, "is_rocm_aiter_fuse_routed_scaling_factor", lambda: True
    )

    method = object.__new__(Mxfp4MoEMethod)
    method.use_flydsl_herd = False
    weights, ids = Mxfp4MoEMethod.select_experts(
        method, layer, torch.empty(2, 4), torch.empty(2, 8)
    )
    weights, ids = Mxfp4MoEMethod.place_experts(method, layer, weights, ids)

    assert weights.tolist() == [[1.5, 0.5, 1.0], [0.0, 0.0, 0.0]]
    assert ids.tolist() == [[0, 4, 8], [-1, -1, -1]]
    assert captured["scale"] == 2.0


def _parallel_config(
    *, dp_size: int, use_ep: bool, backend: str = "auto"
) -> FusedMoEParallelConfig:
    return FusedMoEParallelConfig(
        tp_size=8,
        dp_size=dp_size,
        ep_size=8 if use_ep else 1,
        tp_rank=0,
        dp_rank=0,
        ep_rank=0,
        use_ep=use_ep,
        local_ep_size=8,
        requested_all2all_backend=backend,
    )


@pytest.mark.parametrize(
    "dp_size, use_ep, has_mori, disable_mori, backend, expected_backend",
    [
        # EP without DP still runs the legacy AITER fusion: no all2all backend,
        # so there is no dispatch space to fold the shared expert into.
        (1, True, True, False, "auto", None),
        (8, True, True, False, "auto", "mori"),
        (8, False, True, False, "auto", None),
        (8, True, False, False, "auto", None),
        (8, True, True, True, "auto", None),
        # The native backend is explicit and does not depend on MoRI import or
        # ATOM_DISABLE_MORI_EP, whose scope remains MoRI only.
        (8, True, False, True, "rccl", "rccl"),
        (8, True, True, False, "none", None),
    ],
)
def test_ep_alone_does_not_imply_all2all(
    monkeypatch,
    dp_size,
    use_ep,
    has_mori,
    disable_mori,
    backend,
    expected_backend,
):
    monkeypatch.setattr(moe_module, "_has_module", lambda name: has_mori)
    monkeypatch.setattr(moe_module.envs, "ATOM_DISABLE_MORI_EP", disable_mori)

    config = _parallel_config(dp_size=dp_size, use_ep=use_ep, backend=backend)

    assert config.selected_all2all_backend == expected_backend
    assert config.use_all2all_kernels is (expected_backend is not None)


class _QuantConfig:
    """Checkpoint specs by layer, and the routed experts' online target."""

    online_quant = True

    def __init__(self, shared, routed, online):
        self.shared, self.routed, self.online = shared, routed, online

    def get_layer_quant_config(self, name, use_online_quant=False, **_):
        if "shared" in name:
            return self.shared
        return self.online if use_online_quant else self.routed


def _spec(quant_type, quant_dtype):
    return LayerQuantConfig(quant_type=quant_type, quant_dtype=quant_dtype)


_PTPC = _spec(QuantType.per_Token, dtypes.fp8)
_BLOCK = _spec(QuantType.per_1x128, dtypes.fp8)
_MXFP8 = _spec(QuantType.per_1x32, dtypes.fp8)
_MXFP4 = _spec(QuantType.per_1x32, dtypes.fp4x2)
_BF16 = _spec(QuantType.No, torch.bfloat16)


@pytest.mark.parametrize(
    "shared, routed, online, staged",
    [
        (_PTPC, _MXFP4, _PTPC, True),
        # Already per-channel: same-format fusion, nothing to stage.
        (_PTPC, _PTPC, _PTPC, False),
        # The online target has to be the shared expert's format.
        (_PTPC, _MXFP4, _MXFP4, False),
        (_PTPC, _BLOCK, _MXFP4, False),
        # No weight-scale params to stage the scales through.
        (_PTPC, _BF16, _PTPC, False),
    ],
)
def test_ptpc_shared_expert_staging(monkeypatch, shared, routed, online, staged):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "0")
    quant_config = _QuantConfig(shared, routed, online)
    prefix = "model.layers.3.mlp.experts"
    assert (
        fuses_ptpc_shared_into_online_ptpc(quant_config, prefix, shared, routed)
        is staged
    )


def test_ptpc_shared_expert_is_not_staged_into_streamed_experts(monkeypatch):
    """The staging refuses streamed routed buffers, so the decision leaves the
    shared expert standalone. MXFP4 sources are not streamed."""
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "1")
    prefix = "model.layers.3.mlp.experts"
    for routed, staged in ((_BLOCK, False), (_MXFP4, True)):
        quant_config = _QuantConfig(_PTPC, routed, _PTPC)
        assert (
            fuses_ptpc_shared_into_online_ptpc(quant_config, prefix, _PTPC, routed)
            is staged
        )


@pytest.mark.parametrize(
    "shared, routed, online, fused",
    [
        # FP4PTPC: the MXFP4 routed experts become per-channel FP8 online.
        (_PTPC, _MXFP4, _PTPC, True),
        # One checkpoint format: the slot is re-quantized with the routed experts.
        (_BLOCK, _BLOCK, _MXFP4, True),
        # Only a per-channel FP8 shared expert is staged into an online slot.
        (_MXFP8, _MXFP4, _MXFP8, False),
        # A native shared expert (original V4.1, DSpark draft) stays its own MLP.
        (_MXFP8, _MXFP4, _PTPC, False),
    ],
)
def test_online_quant_shared_expert_fusion(monkeypatch, shared, routed, online, fused):
    monkeypatch.setenv("ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4", "0")
    monkeypatch.setattr(
        topK_module,
        "get_current_atom_config",
        lambda: atom_config_double(
            parallel_config=SimpleNamespace(data_parallel_size=1)
        ),
    )
    enabled = is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
        _QuantConfig(shared, routed, online),
        shared_expert_prefix="model.layers.3.mlp.shared_experts",
        routed_expert_prefix="model.layers.3.mlp.experts",
    )
    assert bool(enabled) is fused


def test_ptpc_shared_expert_stays_unfused_under_dp_expert_parallel(monkeypatch):
    monkeypatch.setattr(
        topK_module,
        "get_current_atom_config",
        lambda: atom_config_double(
            parallel_config=SimpleNamespace(data_parallel_size=8),
            enable_expert_parallel=True,
        ),
    )
    assert not is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
        _QuantConfig(_PTPC, _MXFP4, _PTPC),
        shared_expert_prefix="model.layers.3.mlp.shared_experts",
        routed_expert_prefix="model.layers.3.mlp.experts",
    )
