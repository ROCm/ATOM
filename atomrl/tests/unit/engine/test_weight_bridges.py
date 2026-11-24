# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Streaming weight bridges: GPT (Qwen3-shaped) round-trips, laziness and rule hygiene.

Round-trips go HF -> Megatron through the (dict-based) load direction, then back
through the streaming export, and require every HF tensor to come back
bit-identical. That pins the QKV interleave, the gate/up splits and all three
routed-expert layouts at once.
"""

import logging

import pytest
import torch
from atomrl.engine.training.bridges import dsv3, gpt
from atomrl.engine.training.bridges.core import pp_layer_range, stacked_expert_index

H, NH, KV, HD, FFN, V, E, MF, SF = 8, 4, 2, 2, 6, 10, 4, 3, 5

DENSE = gpt.GPTDims(
    num_layers=4,
    hidden=H,
    num_heads=NH,
    num_kv_groups=KV,
    head_dim=HD,
    ffn=FFN,
    vocab=V,
)
LEGACY_MOE = gpt.GPTDims(
    num_layers=2,
    hidden=H,
    num_heads=NH,
    num_kv_groups=KV,
    head_dim=HD,
    ffn=FFN,
    vocab=V,
    num_experts=E,
    moe_ffn=MF,
    shared_expert_ffn=SF,
    shared_expert_gate=True,
)
MOE = gpt.GPTMoEDims(
    num_layers=2,
    hidden=H,
    num_heads=NH,
    num_kv_groups=KV,
    head_dim=HD,
    ffn=0,
    vocab=V,
    num_experts=E,
    moe_ffn=MF,
    shared_ffn=SF,
)


def _hf(num_layers: int, moe: bool) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(num_layers + moe)

    def r(*shape):
        return torch.randn(*shape, generator=g)

    hf = {
        "model.embed_tokens.weight": r(V, H),
        "model.norm.weight": r(H),
        "lm_head.weight": r(V, H),
    }
    for i in range(num_layers):
        p = f"model.layers.{i}."
        hf.update(
            {
                p + "input_layernorm.weight": r(H),
                p + "self_attn.q_proj.weight": r(NH * HD, H),
                p + "self_attn.k_proj.weight": r(KV * HD, H),
                p + "self_attn.v_proj.weight": r(KV * HD, H),
                p + "self_attn.q_norm.weight": r(HD),
                p + "self_attn.k_norm.weight": r(HD),
                p + "self_attn.o_proj.weight": r(H, NH * HD),
                p + "post_attention_layernorm.weight": r(H),
            }
        )
        if not moe:
            hf.update(
                {
                    p + "mlp.gate_proj.weight": r(FFN, H),
                    p + "mlp.up_proj.weight": r(FFN, H),
                    p + "mlp.down_proj.weight": r(H, FFN),
                }
            )
            continue
        hf[p + "mlp.gate.weight"] = r(E, H)
        for e in range(E):
            hf.update(
                {
                    p + f"mlp.experts.{e}.gate_proj.weight": r(MF, H),
                    p + f"mlp.experts.{e}.up_proj.weight": r(MF, H),
                    p + f"mlp.experts.{e}.down_proj.weight": r(H, MF),
                }
            )
        hf.update(
            {
                p + "mlp.shared_expert.gate_proj.weight": r(SF, H),
                p + "mlp.shared_expert.up_proj.weight": r(SF, H),
                p + "mlp.shared_expert.down_proj.weight": r(H, SF),
                p + "mlp.shared_expert_gate.weight": r(1, H),
            }
        )
    return hf


def _assert_same(expected: dict, got: list) -> None:
    names = [n for n, _ in got]
    assert len(names) == len(set(names)), "an HF name was exported twice"
    assert set(names) == set(expected)
    for n, t in got:
        assert torch.equal(t, expected[n]), n


def _native_moe_named(hf: dict, sequential: bool) -> list:
    """Global names in the engine's order: non-experts, then each layer's experts."""
    named = list(gpt.non_expert_hf_to_megatron(hf, MOE).items())
    for L in range(MOE.num_layers):
        for e in range(E):
            pre = f"module.decoder.layers.{L}.mlp.experts."
            fc1 = (
                f"local_experts.{e}.linear_fc1.weight"
                if sequential
                else f"linear_fc1.weight{e}"
            )
            fc2 = (
                f"local_experts.{e}.linear_fc2.weight"
                if sequential
                else f"linear_fc2.weight{e}"
            )
            named.append((pre + fc1, gpt.hf_expert_fc1(hf, MOE, L, e)))
            named.append((pre + fc2, gpt.hf_expert_fc2(hf, MOE, L, e)))
    return named


def _native_moe_stacked_named(hf: dict, sequential: bool, ep: int = 2) -> list:
    """The engine's ``stack_experts`` output: one ``[ep, out, in]`` stack per local expert."""
    named = list(gpt.non_expert_hf_to_megatron(hf, MOE).items())
    num_local = E // ep
    for L in range(MOE.num_layers):
        for local_e in range(num_local):
            idx = stacked_expert_index(local_e, num_local)
            pre = f"module.decoder.layers.{L}.mlp.experts."
            fc1 = (
                f"local_experts.{idx}.linear_fc1.weight"
                if sequential
                else f"linear_fc1.weight{idx}"
            )
            fc2 = (
                f"local_experts.{idx}.linear_fc2.weight"
                if sequential
                else f"linear_fc2.weight{idx}"
            )
            experts = [j * num_local + local_e for j in range(ep)]
            named.append(
                (
                    pre + fc1,
                    torch.stack([gpt.hf_expert_fc1(hf, MOE, L, e) for e in experts]),
                )
            )
            named.append(
                (
                    pre + fc2,
                    torch.stack([gpt.hf_expert_fc2(hf, MOE, L, e) for e in experts]),
                )
            )
    return named


@pytest.mark.parametrize("sequential", [False, True])
def test_gpt_moe_stacked_experts_round_trip(sequential):
    """Stacks keep their stride in the HF name, and the receiver's expansion
    recovers every per-expert checkpoint tensor bit for bit."""
    from atomrl.engine.inference.vllm_moe_weight_sync import unstack_expert_weights

    hf = _hf(2, moe=True)
    exported = list(gpt.megatron_to_hf(_native_moe_stacked_named(hf, sequential), MOE))
    stacked = [n for n, _ in exported if "::" in n]
    assert "model.layers.1.mlp.experts.1::2.gate_up_proj" in stacked
    assert "model.layers.0.mlp.experts.0::2.down_proj" in stacked
    assert len(stacked) == MOE.num_layers * E  # (E / ep) stacks x 2 leaves x ep=2
    _assert_same(hf, unstack_expert_weights(exported))


@pytest.mark.parametrize("te", [True, False])
def test_gpt_dense_round_trip(te):
    hf = _hf(4, moe=False)
    meg = gpt.hf_to_megatron(hf, DENSE, te=te)
    _assert_same(hf, list(gpt.megatron_to_hf(meg.items(), DENSE)))


def test_gpt_dense_pp_stages_use_layer_offset():
    hf = _hf(4, moe=False)
    got = []
    for rank in range(2):
        meg = gpt.hf_to_megatron(hf, DENSE, pp_rank=rank, pp_size=2)
        offset, _ = pp_layer_range(DENSE, rank, 2, None)
        got += list(gpt.megatron_to_hf(meg.items(), DENSE, layer_offset=offset))
    _assert_same(hf, got)


@pytest.mark.parametrize("sequential", [False, True])
def test_gpt_moe_native_round_trip(sequential):
    hf = _hf(2, moe=True)
    got = list(gpt.megatron_to_hf(_native_moe_named(hf, sequential), MOE))
    _assert_same(hf, got)


@pytest.mark.parametrize("grouped", [True, False])
def test_gpt_legacy_moe_round_trip(grouped):
    hf = _hf(2, moe=True)
    meg = gpt.hf_to_megatron(hf, LEGACY_MOE, use_grouped_mlp=grouped)
    _assert_same(hf, list(gpt.megatron_to_hf(meg.items(), LEGACY_MOE)))


@pytest.mark.parametrize("grouped", [True, False])
def test_gpt_legacy_moe_ep_ranks_use_expert_offset(grouped):
    hf = _hf(2, moe=True)
    got = {}
    for ep_rank in range(2):
        meg = gpt.hf_to_megatron(
            hf, LEGACY_MOE, ep_rank=ep_rank, ep_size=2, use_grouped_mlp=grouped
        )
        got.update(
            gpt.megatron_to_hf(meg.items(), LEGACY_MOE, expert_offset=ep_rank * E // 2)
        )
    _assert_same(hf, list(got.items()))


def test_export_streams_one_tensor_at_a_time():
    """The bridge must hand a tensor on before it reads the next one."""
    meg = list(gpt.hf_to_megatron(_hf(4, moe=False), DENSE, te=True).items())
    pulled = 0

    def lazy():
        nonlocal pulled
        for item in meg:
            pulled += 1
            yield item

    out = gpt.megatron_to_hf(lazy(), DENSE)
    next(out)
    assert pulled == 1


def test_unmatched_names_are_reported(caplog):
    named = [
        ("decoder.layers.0.mlp.unknown.weight", torch.zeros(1)),
        ("output_layer.weight", torch.zeros(1)),
    ]
    with caplog.at_level(logging.WARNING):
        got = list(gpt.megatron_to_hf(named, DENSE))
    assert [n for n, _ in got] == ["lm_head.weight"]
    assert "decoder.layers.0.mlp.unknown.weight" in caplog.text


@pytest.mark.parametrize(
    "bridge,names",
    [
        (
            gpt.GPT,
            lambda: [
                *gpt.hf_to_megatron(_hf(2, moe=True), LEGACY_MOE, use_grouped_mlp=True),
                *gpt.hf_to_megatron(
                    _hf(2, moe=True), LEGACY_MOE, use_grouped_mlp=False
                ),
                *gpt.hf_to_megatron(_hf(4, moe=False), DENSE, te=True),
                *gpt.hf_to_megatron(_hf(4, moe=False), DENSE, te=False),
                *(n for n, _ in _native_moe_named(_hf(2, moe=True), sequential=False)),
                *(
                    n
                    for n, _ in _native_moe_stacked_named(
                        _hf(2, moe=True), sequential=False
                    )
                ),
                *(
                    n
                    for n, _ in _native_moe_stacked_named(
                        _hf(2, moe=True), sequential=True
                    )
                ),
            ],
        ),
        (
            dsv3.DSV3,
            lambda: [
                "decoder.layers.1.mlp.router.expert_bias",
                "decoder.layers.1.mlp.experts.linear_fc1.weight3",
                "decoder.layers.1.mlp.experts.local_experts.3.linear_fc2.weight",
                "decoder.layers.0.mlp.linear_fc1.layer_norm_weight",
                "decoder.layers.1.pre_mlp_layernorm.weight",
            ],
        ),
    ],
)
def test_every_name_matches_exactly_one_rule(bridge, names):
    for raw in names():
        name = raw.removeprefix("module.")
        hits = [r.megatron for r in bridge.rules if r.pattern.match(name)]
        assert len(hits) == 1, (name, hits)
