# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Fused-MoE weight sync: routing math and the silent-drop guard.

CPU-only, no vLLM and no model needed. The fake FusedMoE mirrors the parts of
``vllm.model_executor.layers.fused_moe.layer.FusedMoE.weight_loader`` this code
relies on: a 3D ``loaded_weight`` selects the full-load branch, ``w1``/``w3``
address the first/second half of ``w13_weight`` along the intermediate dim, and
a non-local expert id makes the loader return False instead of writing.

Run: python -m pytest atomrl/tests/atomrl/test_moe_weight_sync.py
"""

import os

import torch
from atomrl.engine.inference.vllm_moe_weight_sync import (
    FusedMoEWeightRouter,
    assert_weight_sync_coverage,
    unstack_expert_weights,
)
from torch import nn

E, I, H = 4, 6, 8
N_LAYERS = 2


class _ParallelConfig:
    def __init__(self, ep_size: int, tp_size: int):
        self.ep_size = ep_size
        self.tp_size = tp_size


class _MoEConfig:
    def __init__(self, ep_size: int, is_act_and_mul: bool, tp_size: int, tp_rank: int):
        self.moe_parallel_config = _ParallelConfig(ep_size, tp_size)
        self.is_act_and_mul = is_act_and_mul
        # vLLM reads the rank off moe_config and the sizes off the parallel
        # config; keep that split so _tp_rank is exercised the way it is used.
        self.tp_rank = tp_rank


class FakeFusedMoE(nn.Module):
    """Stand-in for vLLM's FusedMoE with its real weight_loader semantics.

    Includes the detail this module has to work around: ``_load_w13`` /
    ``_load_w2`` narrow ``loaded_weight`` to this TP rank only when
    ``load_full`` is false, so a 3D tensor is taken to be already sharded.
    """

    def __init__(
        self,
        ep_size: int = 1,
        is_act_and_mul: bool = True,
        local_experts=None,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        inter = I // tp_size
        out13 = 2 * inter if is_act_and_mul else inter
        self.w13_weight = nn.Parameter(torch.zeros(E, out13, H), requires_grad=False)
        self.w2_weight = nn.Parameter(torch.zeros(E, H, inter), requires_grad=False)
        self.moe_config = _MoEConfig(ep_size, is_act_and_mul, tp_size, tp_rank)
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.local_experts = local_experts
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        self.calls.append((shard_id, tuple(loaded_weight.shape)))
        if self.local_experts is not None and expert_id not in self.local_experts:
            return False if return_success else None

        full_load = loaded_weight.ndim == 3
        expert_data = param.data if full_load else param.data[expert_id]
        shard_dim = (1 if shard_id == "w2" else 0) + (1 if full_load else 0)

        if not full_load and self.tp_size > 1:
            per_rank = loaded_weight.shape[shard_dim] // self.tp_size
            loaded_weight = loaded_weight.narrow(
                shard_dim, per_rank * self.tp_rank, per_rank
            )

        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        else:
            size = expert_data.shape[shard_dim] // (
                2 if self.moe_config.is_act_and_mul else 1
            )
            start = 0 if shard_id == "w1" else size
            expert_data.narrow(shard_dim, start, size).copy_(loaded_weight)
        return True if return_success else None


class FakeLayer(nn.Module):
    def __init__(self, **moe_kwargs):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(H, E, bias=False)
        self.mlp.experts = FakeFusedMoE(**moe_kwargs)
        self.mlp.add_module("gate", self.mlp.gate)
        self.mlp.add_module("experts", self.mlp.experts)


class FakeModel(nn.Module):
    """Mimics vLLM's load_weights: unknown names are dropped without raising."""

    def __init__(self, **moe_kwargs):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [FakeLayer(**moe_kwargs) for _ in range(N_LAYERS)]
        )
        self.model.add_module("layers", self.model.layers)
        self.lm_head = nn.Linear(H, 16, bias=False)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name not in params:
                continue
            params[name].data.copy_(tensor)
            loaded.add(name)
        return loaded


def _fused_payload(model, scale=1.0):
    """What a transformers-5.x state_dict() sends for this model."""
    weights = []
    for idx in range(N_LAYERS):
        p = f"model.layers.{idx}.mlp"
        weights.append((f"{p}.experts.gate_up_proj", torch.randn(E, 2 * I, H) * scale))
        weights.append((f"{p}.experts.down_proj", torch.randn(E, H, I) * scale))
        weights.append((f"{p}.gate.weight", torch.randn(E, H) * scale))
    weights.append(("lm_head.weight", torch.randn(16, H) * scale))
    return weights


def test_gate_up_splits_into_w13_halves():
    torch.manual_seed(0)
    model = FakeModel()
    gate_up = torch.randn(E, 2 * I, H)
    router = FusedMoEWeightRouter(model)
    passthrough, loaded = router.route(
        [("model.layers.0.mlp.experts.gate_up_proj", gate_up)]
    )

    assert passthrough == []
    assert loaded == {"model.layers.0.mlp.experts.w13_weight"}
    w13 = model.model.layers[0].mlp.experts.w13_weight.data
    assert torch.equal(w13, gate_up), "fused gate_up must land verbatim in w13"
    assert torch.equal(w13[:, :I], gate_up[:, :I])  # w1 = gate
    assert torch.equal(w13[:, I:], gate_up[:, I:])  # w3 = up
    assert [c[0] for c in model.model.layers[0].mlp.experts.calls] == ["w1", "w3"]


def test_down_proj_lands_in_w2_whole():
    model = FakeModel()
    down = torch.randn(E, H, I)
    _, loaded = FusedMoEWeightRouter(model).route(
        [("model.layers.1.mlp.experts.down_proj", down)]
    )
    assert loaded == {"model.layers.1.mlp.experts.w2_weight"}
    assert torch.equal(model.model.layers[1].mlp.experts.w2_weight.data, down)


def test_full_round_trip_touches_every_parameter():
    """The regression this module exists for: nothing may be silently dropped."""
    torch.manual_seed(1)
    model = FakeModel()
    router = FusedMoEWeightRouter(model)
    passthrough, loaded = router.route(_fused_payload(model))
    loaded |= model.load_weights(passthrough)
    assert_weight_sync_coverage(model, loaded, context="test")

    # Without the router the fused names match nothing and 2/3 of the model
    # keeps its old values -- exactly the bug that hid for 54 steps.
    bare = FakeModel()
    bare_loaded = bare.load_weights(_fused_payload(bare))
    try:
        assert_weight_sync_coverage(bare, bare_loaded, context="test")
    except RuntimeError as exc:
        assert "w13_weight" in str(exc) and "untouched" in str(exc)
    else:
        raise AssertionError("coverage check missed the dropped expert tensors")


def test_non_expert_weights_pass_through_untouched():
    model = FakeModel()
    payload = [
        ("model.layers.0.mlp.gate.weight", torch.randn(E, H)),
        ("lm_head.weight", torch.randn(16, H)),
    ]
    passthrough, loaded = FusedMoEWeightRouter(model).route(payload)
    assert loaded == set()
    assert [n for n, _ in passthrough] == [n for n, _ in payload]


def test_router_is_inactive_without_fused_moe():
    dense = nn.Sequential(nn.Linear(H, H))
    router = FusedMoEWeightRouter(dense)
    assert not router.active
    payload = [("0.weight", torch.randn(H, H))]
    passthrough, loaded = router.route(payload)
    assert passthrough == payload and loaded == set()


def test_non_3d_fused_tensor_is_loud():
    model = FakeModel()
    router = FusedMoEWeightRouter(model)
    try:
        router.route(
            [("model.layers.0.mlp.experts.gate_up_proj", torch.randn(2 * I, H))]
        )
    except RuntimeError as exc:
        assert "must be 3D" in str(exc)
    else:
        raise AssertionError("a 2D fused tensor must not be loaded blindly")


def test_expert_parallel_falls_back_to_per_expert():
    model = FakeModel(ep_size=2, local_experts={0, 1})
    gate_up = torch.randn(E, 2 * I, H)
    _, loaded = FusedMoEWeightRouter(model).route(
        [("model.layers.0.mlp.experts.gate_up_proj", gate_up)]
    )
    assert loaded == {"model.layers.0.mlp.experts.w13_weight"}
    experts = model.model.layers[0].mlp.experts
    assert len(experts.calls) == 2 * E, "one call per (expert, shard) under EP"
    assert all(len(shape) == 2 for _, shape in experts.calls)
    assert torch.equal(experts.w13_weight.data[:2], gate_up[:2])
    assert torch.equal(experts.w13_weight.data[2:], torch.zeros(E - 2, 2 * I, H))


def test_tensor_parallel_takes_the_per_expert_path_and_the_right_slice():
    """TP>1 must not use the 3D full-load branch: it skips the tp_rank narrowing.

    With tp_size=2 / tp_rank=1 and I=6, this rank owns intermediate rows 3..5 of
    each logical matrix, so w13's two halves come from gate_up[:, 3:6] and
    gate_up[:, 9:12], and w2 from down[:, :, 3:6].
    """
    torch.manual_seed(3)
    tp_size, tp_rank = 2, 1
    inter = I // tp_size
    model = FakeModel(tp_size=tp_size, tp_rank=tp_rank)
    gate_up = torch.randn(E, 2 * I, H)
    down = torch.randn(E, H, I)

    _, loaded = FusedMoEWeightRouter(model).route(
        [
            ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
            ("model.layers.0.mlp.experts.down_proj", down),
        ]
    )
    assert loaded == {
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w2_weight",
    }

    experts = model.model.layers[0].mlp.experts
    assert all(len(shape) == 2 for _, shape in experts.calls), (
        "a 3D tensor would be copied without narrowing and corrupt every rank "
        "but tp_rank 0"
    )
    assert len(experts.calls) == 3 * E, "w1 + w3 + w2, once per expert"

    w13 = experts.w13_weight.data
    assert torch.equal(w13[:, :inter], gate_up[:, inter : 2 * inter])
    assert torch.equal(w13[:, inter:], gate_up[:, I + inter : I + 2 * inter])
    assert torch.equal(experts.w2_weight.data, down[:, :, inter : 2 * inter])


def test_tensor_parallel_verify_catches_the_wrong_slice():
    """Verification must follow the same slice, or TP loads go unchecked."""
    torch.manual_seed(4)
    os.environ["ATOMRL_WEIGHT_SYNC_VERIFY"] = "1"
    try:
        payload = lambda: [
            ("model.layers.0.mlp.experts.gate_up_proj", torch.randn(E, 2 * I, H)),
            ("model.layers.0.mlp.experts.down_proj", torch.randn(E, H, I)),
        ]
        # A correct TP=2 rank-1 load verifies clean.
        FusedMoEWeightRouter(FakeModel(tp_size=2, tp_rank=1)).route(payload())

        # A loader that writes rank 0's slice while claiming to be rank 1 is
        # exactly the bug the full-load branch would introduce.
        class WrongRankMoE(FakeFusedMoE):
            def weight_loader(
                self,
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
                return_success=False,
            ):
                saved, self.tp_rank = self.tp_rank, 0
                try:
                    return super().weight_loader(
                        param,
                        loaded_weight,
                        weight_name,
                        shard_id,
                        expert_id,
                        return_success,
                    )
                finally:
                    self.tp_rank = saved

        model = FakeModel(tp_size=2, tp_rank=1)
        model.model.layers[0].mlp.experts.__class__ = WrongRankMoE
        try:
            FusedMoEWeightRouter(model).route(payload())
        except RuntimeError as exc:
            assert "verify failed" in str(exc)
        else:
            raise AssertionError("verify accepted another rank's slice")
    finally:
        os.environ.pop("ATOMRL_WEIGHT_SYNC_VERIFY", None)


def test_non_gated_experts_load_as_one_shard():
    model = FakeModel(is_act_and_mul=False)
    gate_up = torch.randn(E, I, H)
    FusedMoEWeightRouter(model).route(
        [("model.layers.0.mlp.experts.gate_up_proj", gate_up)]
    )
    assert torch.equal(model.model.layers[0].mlp.experts.w13_weight.data, gate_up)


def test_coverage_ignores_quantization_artifacts():
    model = FakeModel()
    model.register_parameter("lm_head_weight_scale", nn.Parameter(torch.ones(1)))
    router = FusedMoEWeightRouter(model)
    passthrough, loaded = router.route(_fused_payload(model))
    loaded |= model.load_weights(passthrough)
    assert_weight_sync_coverage(model, loaded, context="test")


def test_verify_accepts_a_correct_load_and_rejects_a_corrupt_one():
    torch.manual_seed(2)
    os.environ["ATOMRL_WEIGHT_SYNC_VERIFY"] = "1"
    try:
        payload = lambda: [
            ("model.layers.0.mlp.experts.gate_up_proj", torch.randn(E, 2 * I, H)),
            ("model.layers.0.mlp.experts.down_proj", torch.randn(E, H, I)),
        ]
        FusedMoEWeightRouter(FakeModel()).route(payload())

        class DroppingMoE(FakeFusedMoE):
            def weight_loader(
                self,
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
                return_success=False,
            ):
                if shard_id == "w3":  # simulate a shard that never lands
                    return True if return_success else None
                return super().weight_loader(
                    param,
                    loaded_weight,
                    weight_name,
                    shard_id,
                    expert_id,
                    return_success,
                )

        model = FakeModel()
        model.model.layers[0].mlp.experts.__class__ = DroppingMoE
        try:
            FusedMoEWeightRouter(model).route(payload())
        except RuntimeError as exc:
            assert "verify failed" in str(exc) and "w3" in str(exc)
        else:
            raise AssertionError("verify missed a shard that was never written")
    finally:
        os.environ.pop("ATOMRL_WEIGHT_SYNC_VERIFY", None)


def test_coverage_credits_the_shortened_container_name():
    """A fused expert param reported without the ``routed_experts`` segment counts.

    RoutedExperts.load_weights yields names relative to the FusedMoE layer, and
    AutoWeightsLoader prefixes them with the FusedMoE's path, so vLLM reports
    ``...experts.w13_weight`` for a parameter that named_parameters() calls
    ``...experts.routed_experts.w13_weight``.
    """
    inner = FakeModel()
    reported = {n for n, _ in inner.named_parameters()}  # what vLLM hands back
    assert any(n.endswith("w13_weight") for n in reported)

    class NestedExpertNames(nn.Module):
        """Exposes the deeper paths real vLLM registers the buffers under."""

        def named_parameters(self, *args, **kwargs):
            for name, param in inner.named_parameters(*args, **kwargs):
                yield name.replace(".experts.w", ".experts.routed_experts.w"), param

    model = NestedExpertNames()
    assert_weight_sync_coverage(model, reported, context="test")

    # Aliasing must not become a blanket amnesty: drop the fused params from the
    # reported set and they have to come back as missing.
    withheld = {n for n in reported if n.endswith("w13_weight")}
    try:
        assert_weight_sync_coverage(model, reported - withheld, context="test")
    except RuntimeError as exc:
        assert "w13_weight" in str(exc)
    else:
        raise AssertionError("aliasing must not hide a parameter nobody loaded")


def test_coverage_modes_are_configurable():
    model = FakeModel()
    previous = os.environ.get("ATOMRL_WEIGHT_SYNC_CHECK")
    try:
        for mode in ("warn", "off", "OFF"):
            os.environ["ATOMRL_WEIGHT_SYNC_CHECK"] = mode
            assert_weight_sync_coverage(model, set(), context="test")
    finally:
        if previous is None:
            os.environ.pop("ATOMRL_WEIGHT_SYNC_CHECK", None)
        else:
            os.environ["ATOMRL_WEIGHT_SYNC_CHECK"] = previous


def _stacked_payload(ep: int, scale=1.0):
    """What the Megatron-native trainer sends with stack_experts at EP=ep:
    per layer, stacks of experts ``e, e + E/ep, ...`` for each local index e."""
    torch.manual_seed(7)
    full_gate_up = [torch.randn(E, 2 * I, H) * scale for _ in range(N_LAYERS)]
    full_down = [torch.randn(E, H, I) * scale for _ in range(N_LAYERS)]
    step = E // ep
    weights = []
    for idx in range(N_LAYERS):
        p = f"model.layers.{idx}.mlp.experts"
        for e in range(step):
            weights.append(
                (f"{p}.{e}::{step}.gate_up_proj", full_gate_up[idx][e::step].clone())
            )
            weights.append(
                (f"{p}.{e}::{step}.down_proj", full_down[idx][e::step].clone())
            )
    return weights, full_gate_up, full_down


def test_stacked_experts_land_in_strided_slots_with_one_copy():
    model = FakeModel()
    weights, gate_up, down = _stacked_payload(ep=2)
    passthrough, loaded = FusedMoEWeightRouter(model).route(weights)
    assert passthrough == []
    for idx in range(N_LAYERS):
        experts = model.model.layers[idx].mlp.experts
        assert torch.equal(experts.w13_weight.data, gate_up[idx])
        assert torch.equal(experts.w2_weight.data, down[idx])
        assert experts.calls == [], "unsharded stacks must bypass the per-expert loader"
    assert loaded == {
        f"model.layers.{i}.mlp.experts.{w}"
        for i in range(N_LAYERS)
        for w in ("w13_weight", "w2_weight")
    }


def test_stacked_experts_cover_the_model():
    model = FakeModel()
    weights, _, _ = _stacked_payload(ep=4)
    weights += [
        (f"model.layers.{i}.mlp.gate.weight", torch.randn(E, H))
        for i in range(N_LAYERS)
    ]
    weights.append(("lm_head.weight", torch.randn(16, H)))
    router = FusedMoEWeightRouter(model)
    passthrough, loaded = router.route(weights)
    loaded |= model.load_weights(passthrough)
    assert_weight_sync_coverage(model, loaded, context="test")


def test_stacked_experts_under_ep_use_global_ids():
    """A rollout EP rank keeps only its experts; the stack's ids are global."""
    model = FakeModel(ep_size=2, local_experts={1, 3})
    weights, gate_up, down = _stacked_payload(ep=2)
    FusedMoEWeightRouter(model).route(weights[:2])  # layer 0, experts 0 and 2
    FusedMoEWeightRouter(model).route(weights[2:4])  # layer 0, experts 1 and 3
    experts = model.model.layers[0].mlp.experts
    assert all(len(shape) == 2 for _, shape in experts.calls)
    for e in range(E):
        want13 = gate_up[0][e] if e in {1, 3} else torch.zeros(2 * I, H)
        want2 = down[0][e] if e in {1, 3} else torch.zeros(H, I)
        assert torch.equal(experts.w13_weight.data[e], want13), e
        assert torch.equal(experts.w2_weight.data[e], want2), e


def test_stacked_experts_under_tp_take_the_right_slice():
    tp_size, tp_rank = 2, 1
    inter = I // tp_size
    model = FakeModel(tp_size=tp_size, tp_rank=tp_rank)
    weights, gate_up, down = _stacked_payload(ep=2)
    FusedMoEWeightRouter(model).route(weights)
    experts = model.model.layers[1].mlp.experts
    assert len(experts.calls) == 3 * E, "w1 + w3 + w2, once per expert"
    w13 = experts.w13_weight.data
    assert torch.equal(w13[:, :inter], gate_up[1][:, inter : 2 * inter])
    assert torch.equal(w13[:, inter:], gate_up[1][:, I + inter : I + 2 * inter])
    assert torch.equal(experts.w2_weight.data, down[1][:, :, inter : 2 * inter])


def test_stacked_experts_under_tp_and_ep():
    """vLLM TP and EP together: only owned experts land, each with this TP rank's slice."""
    tp_size, tp_rank, owned = 2, 1, {1, 3}
    inter = I // tp_size
    model = FakeModel(tp_size=tp_size, tp_rank=tp_rank, ep_size=2, local_experts=owned)
    weights, gate_up, down = _stacked_payload(ep=2)
    passthrough, _ = FusedMoEWeightRouter(model).route(weights)
    assert passthrough == []
    rows = slice(tp_rank * inter, (tp_rank + 1) * inter)
    for idx in range(N_LAYERS):
        experts = model.model.layers[idx].mlp.experts
        assert all(len(shape) == 2 for _, shape in experts.calls)
        for e in range(E):
            w13, w2 = experts.w13_weight.data[e], experts.w2_weight.data[e]
            if e not in owned:
                assert not w13.any() and not w2.any(), (idx, e)
                continue
            assert torch.equal(w13[:inter], gate_up[idx][e][:I][rows]), (idx, e)
            assert torch.equal(w13[inter:], gate_up[idx][e][I:][rows]), (idx, e)
            assert torch.equal(w2, down[idx][e][:, rows]), (idx, e)


def test_stack_for_an_unknown_layer_is_loud():
    model = FakeModel()
    try:
        FusedMoEWeightRouter(model).route(
            [("model.layers.9.mlp.experts.0::2.gate_up_proj", torch.randn(2, 2 * I, H))]
        )
    except RuntimeError as exc:
        assert "names no FusedMoE layer" in str(exc)
    else:
        raise AssertionError("a stack nobody can load must not pass silently")


def test_unstack_gives_per_expert_checkpoint_names():
    weights, gate_up, down = _stacked_payload(ep=2)
    flat = dict(unstack_expert_weights(weights))
    assert len(flat) == N_LAYERS * E * 3
    p = "model.layers.1.mlp.experts.3."
    assert torch.equal(flat[p + "gate_proj.weight"], gate_up[1][3][:I])
    assert torch.equal(flat[p + "up_proj.weight"], gate_up[1][3][I:])
    assert torch.equal(flat[p + "down_proj.weight"], down[1][3])
    passthrough = [("lm_head.weight", torch.randn(16, H))]
    assert unstack_expert_weights(passthrough) == passthrough


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("all fused-MoE weight sync tests passed")
