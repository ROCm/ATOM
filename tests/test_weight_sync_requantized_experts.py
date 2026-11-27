# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Full-precision experts into an MoE that holds them in FP8.

A trainer sends BF16. An online-quantized MoE keeps its routed experts in FP8
with a scale per block or per channel, and `weight_loader` would cast the
BF16 bytes into that buffer under the scale of the old weight. So a layer that
says it can (`requantizes_synced_experts`) is written through its
`requantize_expert_shard` on every route an expert takes, and one that cannot
is refused before the sync writes anything.

The layer here is a stand-in that records what reaches it; what the real
`FusedMoE.requantize_expert_shard` writes is covered on a GPU in
`test_rollout_expert_weight_sync.py`.
"""

import pytest
import torch
from conftest import atom_config_double
from torch import nn

from atom.model_engine.collective_rpc import WeightReloadRefused
from atom.rollout.weight_updater import WeightUpdaterMixin
from atom.utils.selector import Family

HIDDEN = 8
INTERMEDIATE = 4
EXPERTS = 2
FP8 = torch.float8_e4m3fn


class _Experts(nn.Module):
    """A FusedMoE's expert buffers and the two writers a sync can call."""

    def __init__(self, *, dtype=FP8, requantizes=True):
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.zeros(EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=dtype),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(EXPERTS, HIDDEN, INTERMEDIATE, dtype=dtype),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            torch.ones(EXPERTS, 2 * INTERMEDIATE), requires_grad=False
        )
        self.w2_weight_scale = nn.Parameter(
            torch.ones(EXPERTS, HIDDEN), requires_grad=False
        )
        self.intermediate_size_per_partition = INTERMEDIATE
        self.expert_map = None
        self.num_redundant_experts = 0
        self._requantizes = requantizes
        self.loaded = []
        self.requantized = []

    def requantizes_synced_experts(self):
        return self._requantizes

    def _buffer_name(self, param):
        return "w13_weight" if param is self.w13_weight else "w2_weight"

    def weight_loader(self, param, tensor, weight_name="", shard_id="", expert_id=0):
        self.loaded.append((self._buffer_name(param), shard_id, expert_id))

    def requantize_expert_shard(
        self, param, tensor, shard_id, expert_id, *, rank_local=False
    ):
        self.requantized.append(
            (
                self._buffer_name(param),
                shard_id,
                expert_id,
                rank_local,
                tuple(tensor.shape),
                float(tensor.float().mean()),
            )
        )


def _model(experts):
    """One MoE layer, with the expert mapping `make_expert_params_mapping` builds."""
    mlp = nn.Module()
    mlp.experts = experts
    layer = nn.Module()
    layer.mlp = mlp
    model = nn.Module()
    model.layers = nn.ModuleList([layer])
    model.get_expert_mapping = lambda: [
        (
            "experts.w13_" if proj in ("gate_proj", "up_proj") else "experts.w2_",
            f"experts.{expert_id}.{proj}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(EXPERTS)
        for shard_id, proj in (
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj"),
        )
    ]
    return model


def _updater(model):
    class _Updater(WeightUpdaterMixin):
        device = torch.device("cpu")
        label = "test"
        rank = 0
        world_size = 1
        config = atom_config_double()
        attn_family = Family.MHA

        def clear_kv_cache(self):
            pass

    updater = _Updater()
    updater._init_weight_updater()
    updater.model = model
    return updater


@pytest.fixture
def relaid(monkeypatch):
    """The slices each sync handed to the relayout, in place of the aiter
    shuffle this runner cannot import."""
    seen = []

    def finalize(self):
        for (_module, param_name), arrived in self._expert_relayout_pending.items():
            seen.append(
                (param_name, {e: sorted(s) for e, s in sorted(arrived.items())})
            )
        self._expert_relayout_pending.clear()

    monkeypatch.setattr(WeightUpdaterMixin, "_finalize_expert_weight_sync", finalize)
    return seen


_PROJ_VALUE = {"gate_proj": 1.0, "up_proj": 2.0, "down_proj": 3.0}
_PROJ_SHARD = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


def _trainer_tensors():
    """One BF16 tensor per (expert, projection), each holding its own value."""
    return [
        (
            f"layers.0.mlp.experts.{expert_id}.{proj}.weight",
            torch.full(
                (
                    (HIDDEN, INTERMEDIATE)
                    if proj == "down_proj"
                    else (INTERMEDIATE, HIDDEN)
                ),
                10.0 * expert_id + value,
                dtype=torch.bfloat16,
            ),
        )
        for expert_id in range(EXPERTS)
        for proj, value in _PROJ_VALUE.items()
    ]


def _every_expert(*shards):
    return {e: sorted(shards) for e in range(EXPERTS)}


# ── which layers a sync may write ─────────────────────────────────────────


def test_an_fp8_moe_that_requantizes_is_not_refused():
    assert _updater(_model(_Experts()))._reload_refusal() is None


def test_an_fp8_moe_that_cannot_is_refused_before_anything_is_written():
    experts = _Experts(requantizes=False)

    with pytest.raises(WeightReloadRefused, match="full-precision write cannot"):
        _updater(_model(experts)).update_weights(_trainer_tensors())

    assert experts.loaded == [] and experts.requantized == []


# ── every route an expert takes ───────────────────────────────────────────


def test_per_expert_tensors_are_requantized_not_cast(relaid):
    """The route a trainer that unfuses its experts takes, whole tensors."""
    experts = _Experts()

    updated = _updater(_model(experts)).update_weights(_trainer_tensors())

    assert updated == 3 * EXPERTS
    assert experts.loaded == [], "a BF16 tensor was cast into an FP8 buffer"
    assert sorted(experts.requantized) == sorted(
        (
            "w2_weight" if proj == "down_proj" else "w13_weight",
            _PROJ_SHARD[proj],
            expert_id,
            False,
            (HIDDEN, INTERMEDIATE) if proj == "down_proj" else (INTERMEDIATE, HIDDEN),
            10.0 * expert_id + value,
        )
        for expert_id in range(EXPERTS)
        for proj, value in _PROJ_VALUE.items()
    )
    assert relaid == [
        ("w13_weight", _every_expert("w1", "w3")),
        ("w2_weight", _every_expert("w2")),
    ]


def test_fused_trainer_tensors_are_requantized_per_expert(relaid):
    """(E, 2I, H) gate_up_proj and (E, H, I) down_proj hold this rank's slice
    already, and w13's first half is the gate."""
    experts = _Experts()
    gate_up = torch.cat(
        [
            torch.full((EXPERTS, INTERMEDIATE, HIDDEN), 1.0),
            torch.full((EXPERTS, INTERMEDIATE, HIDDEN), 2.0),
        ],
        dim=1,
    ).to(torch.bfloat16)
    down = torch.full((EXPERTS, HIDDEN, INTERMEDIATE), 3.0, dtype=torch.bfloat16)

    _updater(_model(experts)).update_weights(
        [
            ("layers.0.mlp.experts.gate_up_proj", gate_up),
            ("layers.0.mlp.experts.down_proj", down),
        ]
    )

    assert experts.loaded == []
    assert sorted(experts.requantized) == sorted(
        [
            ("w13_weight", "w1", e, True, (INTERMEDIATE, HIDDEN), 1.0)
            for e in range(EXPERTS)
        ]
        + [
            ("w13_weight", "w3", e, True, (INTERMEDIATE, HIDDEN), 2.0)
            for e in range(EXPERTS)
        ]
        + [
            ("w2_weight", "w2", e, True, (HIDDEN, INTERMEDIATE), 3.0)
            for e in range(EXPERTS)
        ]
    )
    assert relaid == [
        ("w13_weight", _every_expert("w1", "w3")),
        ("w2_weight", _every_expert("w2")),
    ]


def test_an_atom_named_buffer_is_requantized_per_expert(relaid):
    """A trainer mirroring ATOM's state dict sends the fused buffer under its
    own name, which used to be cast into it whole."""
    experts = _Experts()
    w13 = torch.cat(
        [
            torch.full((EXPERTS, INTERMEDIATE, HIDDEN), 1.0),
            torch.full((EXPERTS, INTERMEDIATE, HIDDEN), 2.0),
        ],
        dim=1,
    ).to(torch.bfloat16)

    _updater(_model(experts)).update_weights([("layers.0.mlp.experts.w13_weight", w13)])

    assert sorted(experts.requantized) == sorted(
        [
            ("w13_weight", "w1", e, True, (INTERMEDIATE, HIDDEN), 1.0)
            for e in range(EXPERTS)
        ]
        + [
            ("w13_weight", "w3", e, True, (INTERMEDIATE, HIDDEN), 2.0)
            for e in range(EXPERTS)
        ]
    )
    assert experts.w13_weight.float().eq(0).all(), "the BF16 bytes were copied in"
    assert relaid == [("w13_weight", _every_expert("w1", "w3"))]


@pytest.mark.parametrize(
    "leaf, shape",
    [
        ("gate_up_proj", (EXPERTS, 2 * 2 * INTERMEDIATE, HIDDEN)),
        ("down_proj", (EXPERTS, HIDDEN, 2 * INTERMEDIATE)),
    ],
)
def test_a_global_fused_tensor_is_refused_not_requantized_as_rank_0s(
    leaf, shape, relaid
):
    """This rank holds INTERMEDIATE of the 2 * INTERMEDIATE a TP2 layer has.
    `requantize_expert_shard` takes each expert's leading rows, so the global
    tensor was written as rank 0's shard on every rank, and nothing said so."""
    experts = _Experts()
    tensor = torch.ones(shape, dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="this rank's"):
        _updater(_model(experts)).update_weights(
            [(f"layers.0.mlp.experts.{leaf}", tensor)]
        )

    assert experts.requantized == [] and experts.loaded == []
    assert relaid == []


@pytest.mark.parametrize("dtype", [FP8, torch.float8_e5m2])
@pytest.mark.parametrize(
    "name, shape",
    [
        ("layers.0.mlp.experts.0.gate_proj.weight", (INTERMEDIATE, HIDDEN)),
        ("layers.0.mlp.experts.gate_up_proj", (EXPERTS, 2 * INTERMEDIATE, HIDDEN)),
        ("layers.0.mlp.experts.w13_weight", (EXPERTS, 2 * INTERMEDIATE, HIDDEN)),
    ],
    ids=["per_expert", "fused", "atom_named"],
)
def test_an_fp8_expert_is_refused_not_requantized(name, shape, dtype, relaid):
    """Requantizing FP8 values reads them as if they were the weight and drops
    the scale they were stored under, which no sync sends for experts."""
    experts = _Experts()
    tensor = torch.full(shape, 0.5).to(dtype)

    with pytest.raises(NotImplementedError, match="requantizes from full precision"):
        _updater(_model(experts)).update_weights([(name, tensor)])

    assert experts.requantized == [] and experts.loaded == []
    assert relaid == []


def test_a_bf16_moe_keeps_its_loader(relaid):
    """Requantizing is for FP8 buffers; a full-precision one takes what is sent."""
    experts = _Experts(dtype=torch.bfloat16)

    _updater(_model(experts)).update_weights(_trainer_tensors())

    assert experts.requantized == []
    assert len(experts.loaded) == 3 * EXPERTS


# ── what a sync counts as rewritten ───────────────────────────────────────


def test_a_full_sync_counts_the_scales_it_rewrote(relaid):
    """The load records an online-quantized MoE's scales as loaded, so a sync
    that has to rewrite every parameter needs them covered too. No trainer
    sends them; requantizing an expert rewrites its slice of both."""
    model = _model(_Experts())
    updater = _updater(model)
    updater._checkpoint_param_names = frozenset(n for n, _ in model.named_parameters())

    updater.update_weights(_trainer_tensors(), require_full=True)


def test_a_full_sync_still_misses_an_expert_nobody_sent(relaid):
    model = _model(_Experts())
    updater = _updater(model)
    updater._checkpoint_param_names = frozenset(n for n, _ in model.named_parameters())
    tensors = [(n, t) for n, t in _trainer_tensors() if ".experts.1." not in n]

    with pytest.raises(RuntimeError, match="w13_weight_scale"):
        updater.update_weights(tensors, require_full=True)
