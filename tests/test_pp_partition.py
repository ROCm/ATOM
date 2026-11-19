# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_PP_LAYER_PARTITION=kv_balanced: the split that maximises the pages
every PP stage can hold (atom/models/pp_partition.py), how it is enabled, and
that every process and rank uses the one split the config resolved."""

import copy
import itertools
import json
import logging
import pathlib
import pickle
import struct
import subprocess
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest

from atom.config import Config
from atom.models import pp_partition as P
from atom.models import utils as model_utils
from atom.models.pp_partition import LayerCost, PartitionProblem
from atom.models.utils import (
    get_pp_indices,
    install_pp_layer_partition,
    pp_layer_partition,
)

GiB = 1 << 30
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Kimi-K3: 93 layers, MLA at 1-based 4, 8, ..., 92 and 93, KDA elsewhere.
K3_LAYERS = 93
K3_FULL_ATTN = [*range(4, 93, 4), 93]
K3_MLA = {i - 1 for i in K3_FULL_ATTN}
K3_KDA = [i for i in range(1, 94) if i not in K3_FULL_ATTN]
# Kimi-K3 at PP8 x TP1: checkpoint bytes per layer (MXFP4 experts; layer 0 has
# a dense MLP), the 576 B fp8 latent row, the 8380416 B KDA slot its run
# logged (DSpark 4 with ReplaySSM, 64 sequences), the DSpark drafter's weights
# less its embedding.
MLA_WEIGHT, KDA_WEIGHT, DENSE_WEIGHT = 16_570_000_000, 16_990_000_000, 2_340_000_000
KDA_SLOT = 8_380_416
EMBED = HEAD = 2_350_000_000
DRAFT_WEIGHT = 4_780_000_000
CONTEXT_PROJ = 514_000_000
EVEN_8 = (11, 11, 12, 12, 12, 12, 12, 11)
HAND_PICKED_8 = (13, 13, 13, 13, 13, 12, 12, 4)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No split installed by another test, no override from the shell."""
    monkeypatch.setattr(model_utils, "_installed_pp_partitions", {})
    for name in (
        "VLLM_PP_LAYER_PARTITION",
        "ATOM_PP_LAYER_PARTITION",
        "ATOM_PP_PARTITION_MAX_EXTRA_LAYERS",
        "ATOM_ENABLE_REPLAYSSM",
        "ATOM_REPLAYSSM_CACHE_LEN",
        "ATOM_GDN_SSM_DTYPE",
    ):
        monkeypatch.delenv(name, raising=False)


def k3_layers() -> tuple[LayerCost, ...]:
    def cost(i):
        if i in K3_MLA:
            return LayerCost(MLA_WEIGHT, kv_bytes_per_token=576)
        weight = DENSE_WEIGHT if i == 0 else KDA_WEIGHT
        return LayerCost(weight, state_bytes_per_request=KDA_SLOT)

    return tuple(cost(i) for i in range(K3_LAYERS))


def k3_problem(pp=8, extra=1, drafter=True, **kw) -> PartitionProblem:
    fields = {
        "layers": k3_layers(),
        "pp_size": pp,
        # 0.9 utilization less the 10% runtime reserve of a 432 GiB GPU
        "stage_bytes": int(432 * GiB * 0.8),
        "block_size": 128,
        "max_num_seqs": 64,
        "first_stage_weight_bytes": EMBED,
        "last_stage_weight_bytes": HEAD + (DRAFT_WEIGHT if drafter else 0),
        "last_stage_kv_bytes_per_token": 5 * 576 if drafter else 0,
        "max_extra_layers": extra,
    }
    fields.update(kw)
    return PartitionProblem(**fields)


def stage_ranges(partition):
    lo = 0
    for n in partition:
        yield range(lo, lo + n)
        lo += n


# --------------------------------------------------------------------------
# The split it picks
# --------------------------------------------------------------------------


def test_k3_layout_beats_the_even_split_within_one_extra_layer():
    problem = k3_problem()
    plan = P.plan_partition(problem)

    assert plan.even == EVEN_8
    assert sum(plan.partition) == K3_LAYERS and len(plan.partition) == 8
    # Within the compute limit: one layer above the even split's 12.
    assert max(plan.partition) <= 13
    # The pool grows by ~90%: the drafter's KV no longer caps it.
    assert plan.pool >= 1.8 * plan.even_pool
    # As large as the hand-picked 13,13,13,13,13,12,12,4 within tolerance,
    # with fewer 13-layer stages.
    hand = P.predicted_pages(problem, HAND_PICKED_8)
    assert plan.pool >= min(hand) * (1 - P.POOL_TOLERANCE)
    assert plan.partition.count(13) < HAND_PICKED_8.count(13)
    # The split it settles on (pinned so a change to it is deliberate).
    assert plan.partition == (12, 12, 12, 13, 13, 13, 13, 5)


def test_k3_layout_reports_the_pages_it_predicts():
    plan = P.plan_partition(k3_problem())
    assert plan.pages == P.predicted_pages(k3_problem(), plan.partition)
    assert plan.pool == min(plan.pages)
    assert plan.even_pages == P.predicted_pages(k3_problem(), EVEN_8)
    # The even split's last stage (4 target MLA + 5 draft rows) is its cap.
    assert plan.even_pool == plan.even_pages[-1]
    text = plan.describe()
    assert "[12,12,12,13,13,13,13,5]" in text and "[11,11,12,12,12,12,12,11]" in text


def test_predicted_pages_are_free_memory_over_page_bytes():
    problem = k3_problem()
    pages = P.predicted_pages(problem, EVEN_8)
    # Stage 0 holds layers 0-10: the dense layer 0, MLA 3 and 7, eight KDA
    # layers, and the embedding.
    free = (
        problem.stage_bytes
        - (DENSE_WEIGHT + 2 * MLA_WEIGHT + 8 * KDA_WEIGHT + EMBED)
        - 9 * KDA_SLOT * 64
    )
    assert pages[0] == free // (2 * 576 * 128)
    # The last holds 82-92: MLA 83, 87, 91, 92, seven KDA layers, lm_head and
    # the drafter, whose 5 MLA rows share its page.
    free = (
        problem.stage_bytes
        - (4 * MLA_WEIGHT + 7 * KDA_WEIGHT + HEAD + DRAFT_WEIGHT)
        - 7 * KDA_SLOT * 64
    )
    assert pages[-1] == free // ((4 + 5) * 576 * 128)


def test_per_request_state_moves_the_split_at_high_concurrency():
    # 512 sequences of KDA state (~4.3 GB a layer) make the KDA-heavy stages
    # the tight ones, so the split shifts.
    at_64 = P.plan_partition(k3_problem()).partition
    at_512 = P.plan_partition(k3_problem(max_num_seqs=512)).partition
    assert at_512 != at_64
    assert max(at_512) <= 13


def test_no_extra_layers_keeps_the_largest_stage_and_still_gains():
    plan = P.plan_partition(k3_problem(extra=0))
    assert max(plan.partition) == max(EVEN_8) == 12
    assert plan.pool >= 1.3 * plan.even_pool


def test_more_extra_layers_never_shrink_the_pool():
    # Each limit's choice is within the tolerance of its own best pool, and a
    # looser limit's best is at least a tighter one's.
    pools = [P.plan_partition(k3_problem(extra=e)).pool for e in (0, 1, 2)]
    assert pools[0] <= pools[1] / (1 - P.POOL_TOLERANCE)
    assert pools[1] <= pools[2] / (1 - P.POOL_TOLERANCE)


def test_every_stage_keeps_a_linear_attention_and_a_kv_layer():
    # A drafter so large that the best last stage would be its last MLA
    # layers alone (91, 92): the runner refuses a stage without recurrent
    # state, so the split must reach back to KDA layer 90.
    plan = P.plan_partition(k3_problem(last_stage_kv_bytes_per_token=60 * 576))
    for stage in stage_ranges(plan.partition):
        assert any(i in K3_MLA for i in stage), plan.partition
        assert any(i not in K3_MLA for i in stage), plan.partition
    assert plan.partition[-1] >= 3


def test_k3_at_other_pp_sizes_stays_within_the_limit():
    for pp in (2, 4, 16):
        plan = P.plan_partition(k3_problem(pp=pp))
        assert sum(plan.partition) == K3_LAYERS and len(plan.partition) == pp
        assert max(plan.partition) <= max(plan.even) + 1
        assert plan.pool >= plan.even_pool


def test_uniform_model_keeps_the_even_split():
    # No per-token-KV imbalance: every layer alike, no drafter.
    for embeddings in (0, 2 * GiB):
        problem = PartitionProblem(
            layers=(LayerCost(10 * GiB, kv_bytes_per_token=576),) * 61,
            pp_size=4,
            stage_bytes=300 * GiB,
            block_size=64,
            max_num_seqs=256,
            first_stage_weight_bytes=embeddings,
            last_stage_weight_bytes=embeddings,
        )
        plan = P.plan_partition(problem)
        assert plan.partition == plan.even == tuple(P.even_partition(61, 4))


def test_hybrid_with_evenly_spread_kv_keeps_the_even_split():
    kv = LayerCost(10 * GiB, kv_bytes_per_token=1024)
    linear = LayerCost(10 * GiB, state_bytes_per_request=4 << 20)
    problem = PartitionProblem(
        layers=(linear, kv) * 16,
        pp_size=4,
        stage_bytes=200 * GiB,
        block_size=64,
        max_num_seqs=64,
    )
    assert P.plan_partition(problem).partition == (8, 8, 8, 8)


def test_a_model_without_per_token_kv_keeps_the_even_split():
    problem = PartitionProblem(
        layers=(LayerCost(GiB, state_bytes_per_request=1 << 20),) * 10,
        pp_size=3,
        stage_bytes=100 * GiB,
        block_size=16,
        max_num_seqs=8,
    )
    assert P.plan_partition(problem).partition == (3, 4, 3)


def test_no_feasible_split_keeps_the_even_split():
    # Every KV layer on the first half: no split gives each stage one.
    kv = LayerCost(GiB, kv_bytes_per_token=512)
    linear = LayerCost(GiB, state_bytes_per_request=1 << 20)
    problem = PartitionProblem(
        layers=(kv,) * 8 + (linear,) * 8,
        pp_size=4,
        stage_bytes=100 * GiB,
        block_size=16,
        max_num_seqs=8,
    )
    assert P.plan_partition(problem).partition == (4, 4, 4, 4)


def test_plan_is_deterministic():
    assert P.plan_partition(k3_problem()) == P.plan_partition(k3_problem())


def test_predicted_pages_refuses_a_split_of_something_else():
    with pytest.raises(ValueError):
        P.predicted_pages(k3_problem(), (12,) * 8)


# --------------------------------------------------------------------------
# What a stage holds, from the config
# --------------------------------------------------------------------------


def k3_text_config(**kw):
    fields = {
        "model_type": "kimi_linear",
        "architectures": ["KimiK3ForConditionalGeneration"],
        "num_hidden_layers": K3_LAYERS,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "tie_word_embeddings": False,
        "linear_attn_config": {
            "full_attn_layers": K3_FULL_ATTN,
            "kda_layers": K3_KDA,
            "num_heads": 96,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        },
    }
    fields.update(kw)
    return SimpleNamespace(**fields)


def dspark_spec(model, tokens=4, dtype="fp8"):
    return SimpleNamespace(
        method="dspark",
        model=model,
        num_speculative_tokens=tokens,
        draft_kv_dtype=dtype,
        draft_model_hf_config=SimpleNamespace(
            model_type="k3_dspark",
            architectures=["K3DSparkModel"],
            num_hidden_layers=5,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        ),
        use_dspark=lambda: True,
        use_dspark_with_draft=lambda: True,
    )


def test_kimi_k3_layer_kinds_follow_linear_attn_config():
    kinds = P._layer_kinds(k3_text_config(), K3_LAYERS)
    assert {i for i, kv in enumerate(kinds) if kv} == K3_MLA


def test_kimi_k3_layer_kinds_refuse_an_incomplete_assignment():
    text = k3_text_config()
    text.linear_attn_config = dict(text.linear_attn_config, kda_layers=K3_KDA[:-1])
    with pytest.raises(P.PartitionUnavailable):
        P._layer_kinds(text, K3_LAYERS)


def test_unknown_layer_types_are_not_estimated():
    text = SimpleNamespace(
        model_type="gpt_oss",
        num_attention_heads=64,
        layer_types=["sliding_attention", "full_attention"] * 2,
    )
    with pytest.raises(P.PartitionUnavailable):
        P._layer_kinds(text, 4)


def test_kv_bytes_per_token():
    k3 = k3_text_config()
    assert P._kv_bytes_per_token(k3, "fp8", tp=1) == 576  # measured: 221184/3/128
    assert P._kv_bytes_per_token(k3, "fp8", tp=8) == 576  # the latent is replicated
    assert P._kv_bytes_per_token(k3, "bf16", tp=1) == 1152
    gqa = SimpleNamespace(num_key_value_heads=8, num_attention_heads=64, head_dim=128)
    assert P._kv_bytes_per_token(gqa, "bf16", tp=2) == 2 * 4 * 128 * 2
    assert P._kv_bytes_per_token(gqa, "bf16", tp=16) == 2 * 1 * 128 * 2


def test_kda_state_matches_the_measured_slot(monkeypatch):
    config = SimpleNamespace(speculative_config=dspark_spec("/draft"))
    text = k3_text_config()
    # DSpark 4 with ReplaySSM, record depth 16: the 8380416 B per layer the
    # PP8 run's state pool logged (75423744 B for 9 KDA layers).
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "1")
    assert P._linear_state_bytes_per_request(text, config, tp=1) == KDA_SLOT
    # Without it, 1 + 4 slots of conv + fp32 temporal state, 6807552 B each
    # (the image size the run logged).
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "0")
    assert P._linear_state_bytes_per_request(text, config, tp=1) == 5 * 6_807_552


def write_checkpoint(directory: pathlib.Path, tensors: dict[str, int], shards=1):
    """safetensors shard headers declaring ``tensors`` (name -> bytes); the
    planner reads nothing past the header."""
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    weight_map = {}
    for shard in range(shards):
        header, offset = {}, 0
        file = f"model-{shard + 1:05d}-of-{shards:05d}.safetensors"
        for name in names[shard::shards]:
            header[name] = {
                "dtype": "U8",
                "shape": [tensors[name]],
                "data_offsets": [offset, offset + tensors[name]],
            }
            offset += tensors[name]
            weight_map[name] = file
        blob = json.dumps(header).encode()
        (directory / file).write_bytes(struct.pack("<Q", len(blob)) + blob)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def k3_checkpoint(directory: pathlib.Path):
    tensors = {
        "language_model.model.embed_tokens.weight": EMBED,
        "language_model.lm_head.weight": HEAD,
        "language_model.model.norm.weight": 14_336,
        "vision_tower.encoder.layers.0.wqkv.weight": 9_000_000,
        "mm_projector.proj.0.weight": 34_000_000,
    }
    for i in range(K3_LAYERS):
        weight = MLA_WEIGHT if i in K3_MLA else DENSE_WEIGHT if i == 0 else KDA_WEIGHT
        prefix = f"language_model.model.layers.{i}"
        tensors[f"{prefix}.self_attn.weight"] = 1_000_000
        tensors[f"{prefix}.mlp.weight"] = weight - 1_000_000
    write_checkpoint(directory, tensors, shards=4)


def draft_checkpoint(directory: pathlib.Path):
    tensors = {"embed_tokens.weight": EMBED, "context_proj.weight": CONTEXT_PROJ}
    for i in range(5):
        tensors[f"layers.{i}.weight"] = (DRAFT_WEIGHT - CONTEXT_PROJ) // 5
    write_checkpoint(directory, tensors)


def test_checkpoint_bytes_by_where_they_live(tmp_path):
    write_checkpoint(
        tmp_path,
        {
            "model.embed_tokens.weight": 100,
            "model.layers.0.mlp.weight": 10,
            "model.layers.1.mlp.weight": 11,
            "model.layers.2.mlp.weight": 50,  # an MTP layer past the 2 targets
            "mtp.layers.0.mlp.weight": 60,
            "visual.blocks.layers.0.weight": 7,
            "lm_head.weight": 200,
            "model.norm.weight": 1,
        },
        shards=3,
    )
    got = P._checkpoint_bytes(str(tmp_path), 2)
    assert got.layers == (10, 11)
    assert (got.beyond, got.embed, got.head, got.other) == (110, 100, 200, 8)


def k3_config(tmp_path, **kw):
    k3_checkpoint(tmp_path / "k3")
    draft_checkpoint(tmp_path / "draft")
    fields = {
        "model": str(tmp_path / "k3"),
        "hf_config": k3_text_config(),
        "pipeline_parallel_size": 8,
        "tensor_parallel_size": 1,
        "kv_cache_dtype": "fp8",
        "kv_cache_block_size": 128,
        "max_num_seqs": 64,
        "gpu_memory_utilization": 0.9,
        "speculative_config": dspark_spec(str(tmp_path / "draft")),
        "kv_transfer_config": {"kv_connector": "mooncake", "kv_role": "kv_producer"},
    }
    fields.update(kw)
    return SimpleNamespace(**fields)


def test_k3_problem_from_its_checkpoint_and_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "1")
    monkeypatch.setattr(P, "device_memory_bytes", lambda: 432 * GiB)
    problem = P.partition_problem(k3_config(tmp_path))

    assert problem.pp_size == 8 and problem.block_size == 128
    assert problem.stage_bytes == int(432 * GiB * 0.8)
    assert [layer.kv_bytes_per_token > 0 for layer in problem.layers] == [
        i in K3_MLA for i in range(K3_LAYERS)
    ]
    assert {layer.state_bytes_per_request for layer in problem.layers} == {
        0,
        KDA_SLOT,
    }
    assert problem.layers[0].weight_bytes == DENSE_WEIGHT
    # The drafter's 5 MLA rows, and its weights less the embedding it shares;
    # a P/D producer's last stage keeps no embedding copy.
    assert problem.last_stage_kv_bytes_per_token == 5 * 576
    assert problem.last_stage_weight_bytes == HEAD + DRAFT_WEIGHT
    assert problem.first_stage_weight_bytes == EMBED + 14_336 + 9_000_000 + 34_000_000
    assert P.plan_partition(problem).partition == (12, 12, 12, 13, 13, 13, 13, 5)


def test_a_drafting_last_stage_also_holds_an_embedding_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "device_memory_bytes", lambda: 432 * GiB)
    problem = P.partition_problem(k3_config(tmp_path, kv_transfer_config=None))
    assert problem.last_stage_weight_bytes == HEAD + DRAFT_WEIGHT + EMBED


def test_tp_shares_the_weights_and_state_not_the_latent(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "device_memory_bytes", lambda: 432 * GiB)
    tp1 = P.partition_problem(k3_config(tmp_path))
    tp2 = P.partition_problem(k3_config(tmp_path, tensor_parallel_size=2))
    assert tp2.layers[1].weight_bytes == tp1.layers[1].weight_bytes // 2
    assert tp2.layers[3].kv_bytes_per_token == tp1.layers[3].kv_bytes_per_token
    assert tp2.layers[1].state_bytes_per_request < tp1.layers[1].state_bytes_per_request


def test_missing_checkpoint_or_memory_is_unavailable(tmp_path, monkeypatch):
    config = k3_config(tmp_path)
    monkeypatch.setattr(P, "device_memory_bytes", lambda: None)
    with pytest.raises(P.PartitionUnavailable, match="memory"):
        P.partition_problem(config)
    monkeypatch.setattr(P, "device_memory_bytes", lambda: 432 * GiB)
    config.model = str(tmp_path / "nowhere")
    with pytest.raises(P.PartitionUnavailable, match="shard headers"):
        P.partition_problem(config)


def test_device_memory_from_sysfs(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "amdsmi", None)  # import fails
    cards = []
    for i, size in enumerate((432 * GiB, 432 * GiB, 1 * GiB)):
        path = tmp_path / f"card{i}"
        path.write_text(f"{size}\n")
        cards.append(str(path))
    monkeypatch.setattr(P.glob, "glob", lambda pattern: cards)
    assert P.device_memory_bytes() == 432 * GiB


# --------------------------------------------------------------------------
# How it is enabled
# --------------------------------------------------------------------------


def k3_balanced(monkeypatch):
    monkeypatch.setenv("ATOM_PP_LAYER_PARTITION", "kv_balanced")
    monkeypatch.setattr(P, "partition_problem", lambda config: k3_problem())
    return SimpleNamespace(pipeline_parallel_size=8)


def test_the_default_is_the_even_split(monkeypatch):
    monkeypatch.setattr(P, "partition_problem", mock.Mock(side_effect=AssertionError))
    assert (
        P.resolve_pp_layer_partition(SimpleNamespace(pipeline_parallel_size=8)) is None
    )
    monkeypatch.setenv("ATOM_PP_LAYER_PARTITION", "even")
    assert (
        P.resolve_pp_layer_partition(SimpleNamespace(pipeline_parallel_size=8)) is None
    )


def test_kv_balanced_resolves_and_logs_the_split(monkeypatch, caplog):
    config = k3_balanced(monkeypatch)
    with caplog.at_level(logging.INFO, logger="atom"):
        partition = P.resolve_pp_layer_partition(config)
    assert partition == (12, 12, 12, 13, 13, 13, 13, 5)
    assert "VLLM_PP_LAYER_PARTITION=12,12,12,13,13,13,13,5 pins it" in caplog.text
    assert "predicted pages per stage" in caplog.text


def test_vllm_pp_layer_partition_keeps_winning(monkeypatch, caplog):
    config = k3_balanced(monkeypatch)
    monkeypatch.setattr(P, "partition_problem", mock.Mock(side_effect=AssertionError))
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", ",".join(map(str, HAND_PICKED_8)))
    with caplog.at_level(logging.INFO, logger="atom"):
        assert P.resolve_pp_layer_partition(config) is None
    assert "overrides ATOM_PP_LAYER_PARTITION" in caplog.text
    # ...and over a split installed before it was set.
    install_pp_layer_partition(K3_LAYERS, (12, 12, 12, 13, 13, 13, 13, 5))
    assert pp_layer_partition(K3_LAYERS, 8) == list(HAND_PICKED_8)
    assert get_pp_indices(K3_LAYERS, 7, 8) == (89, 93)


def test_pp1_and_plugin_mode_resolve_nothing(monkeypatch):
    k3_balanced(monkeypatch)
    assert (
        P.resolve_pp_layer_partition(SimpleNamespace(pipeline_parallel_size=1)) is None
    )
    with mock.patch("atom.plugin.is_plugin_mode", return_value=True):
        assert (
            P.resolve_pp_layer_partition(SimpleNamespace(pipeline_parallel_size=8))
            is None
        )


def test_an_unestimable_model_keeps_the_even_split(monkeypatch, caplog):
    monkeypatch.setenv("ATOM_PP_LAYER_PARTITION", "kv_balanced")
    monkeypatch.setattr(
        P,
        "partition_problem",
        mock.Mock(side_effect=P.PartitionUnavailable("no estimate for v4")),
    )
    with caplog.at_level(logging.WARNING, logger="atom"):
        assert (
            P.resolve_pp_layer_partition(SimpleNamespace(pipeline_parallel_size=4))
            is None
        )
    assert "no estimate for v4; keeping the even split" in caplog.text


def test_a_balanced_model_resolves_to_nothing(monkeypatch):
    monkeypatch.setenv("ATOM_PP_LAYER_PARTITION", "kv_balanced")
    uniform = PartitionProblem(
        layers=(LayerCost(10 * GiB, kv_bytes_per_token=576),) * 64,
        pp_size=8,
        stage_bytes=300 * GiB,
        block_size=64,
        max_num_seqs=64,
    )
    monkeypatch.setattr(P, "partition_problem", lambda config: uniform)
    assert (
        P.resolve_pp_layer_partition(SimpleNamespace(pipeline_parallel_size=8)) is None
    )


def test_mode_and_limit_env_vars(monkeypatch):
    from atom.utils import envs

    assert envs.ATOM_PP_LAYER_PARTITION == "even"
    assert envs.ATOM_PP_PARTITION_MAX_EXTRA_LAYERS == 1
    monkeypatch.setenv("ATOM_PP_LAYER_PARTITION", " KV_Balanced ")
    monkeypatch.setenv("ATOM_PP_PARTITION_MAX_EXTRA_LAYERS", "0")
    assert envs.ATOM_PP_LAYER_PARTITION == "kv_balanced"
    assert envs.ATOM_PP_PARTITION_MAX_EXTRA_LAYERS == 0
    monkeypatch.setenv("ATOM_PP_LAYER_PARTITION", "auto")
    with pytest.raises(ValueError, match="ATOM_PP_LAYER_PARTITION"):
        _ = envs.ATOM_PP_LAYER_PARTITION
    monkeypatch.setenv("ATOM_PP_PARTITION_MAX_EXTRA_LAYERS", "-1")
    with pytest.raises(ValueError, match="ATOM_PP_PARTITION_MAX_EXTRA_LAYERS"):
        _ = envs.ATOM_PP_PARTITION_MAX_EXTRA_LAYERS


# --------------------------------------------------------------------------
# One split everywhere
# --------------------------------------------------------------------------

BALANCED = (12, 12, 12, 13, 13, 13, 13, 5)


def test_every_rank_gets_a_contiguous_slice_of_the_installed_split():
    install_pp_layer_partition(K3_LAYERS, BALANCED)
    spans = [get_pp_indices(K3_LAYERS, rank, 8) for rank in range(8)]
    assert spans[0][0] == 0 and spans[-1][1] == K3_LAYERS
    assert all(a[1] == b[0] for a, b in itertools.pairwise(spans))
    assert tuple(hi - lo for lo, hi in spans) == BALANCED
    # Another model or PP size in the same process keeps its own split.
    assert pp_layer_partition(K3_LAYERS, 4) == P.even_partition(K3_LAYERS, 4)
    assert pp_layer_partition(61, 8) == P.even_partition(61, 8)


def test_install_refuses_a_split_of_something_else():
    with pytest.raises(ValueError):
        install_pp_layer_partition(K3_LAYERS, (12,) * 8)
    with pytest.raises(ValueError):
        install_pp_layer_partition(K3_LAYERS, (0, 93))


def test_offload_keys_follow_the_installed_split():
    from atom.kv_transfer.offload.kv_store.keys import pp_stage_layer_spans

    config = SimpleNamespace(
        hf_config=SimpleNamespace(num_hidden_layers=K3_LAYERS),
        pipeline_parallel_size=8,
    )
    even = pp_stage_layer_spans(config)
    install_pp_layer_partition(K3_LAYERS, BALANCED)
    spans = pp_stage_layer_spans(config)
    assert spans == [get_pp_indices(K3_LAYERS, r, 8) for r in range(8)]
    assert spans != even


def resolved_config(partition=BALANCED):
    """A `Config` as `__post_init__` leaves it, without a model directory."""
    config = Config.__new__(Config)
    config.__dict__.update(
        hf_config=SimpleNamespace(num_hidden_layers=K3_LAYERS),
        pp_layer_partition=partition,
    )
    return config


def test_unpickling_a_config_installs_its_split():
    blob = pickle.dumps(resolved_config())
    assert pp_layer_partition(K3_LAYERS, 8) == list(EVEN_8)
    pickle.loads(blob)
    assert tuple(pp_layer_partition(K3_LAYERS, 8)) == BALANCED


def test_a_copied_config_installs_its_split():
    copy.deepcopy(resolved_config())
    assert tuple(pp_layer_partition(K3_LAYERS, 8)) == BALANCED


def test_a_config_without_a_resolved_split_installs_nothing():
    pickle.loads(pickle.dumps(resolved_config(partition=None)))
    assert pp_layer_partition(K3_LAYERS, 8) == list(EVEN_8)


def test_a_spawned_process_splits_as_the_resolving_one():
    """What a per-stage EngineCore or a worker does: unpickle its config in a
    fresh interpreter, then ask `get_pp_indices`."""
    child = (
        "import pickle, sys\n"
        "from atom.models.utils import get_pp_indices\n"
        "pickle.loads(sys.stdin.buffer.read())\n"
        "print([get_pp_indices(93, r, 8) for r in range(8)])\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", child],
        input=pickle.dumps(resolved_config()),
        capture_output=True,
        cwd=REPO_ROOT,
        check=True,
        timeout=300,
    ).stdout.decode()
    install_pp_layer_partition(K3_LAYERS, BALANCED)
    expected = [get_pp_indices(K3_LAYERS, r, 8) for r in range(8)]
    assert out.strip().splitlines()[-1] == str(expected)


def fake_pp_group(monkeypatch, world_size):
    group = SimpleNamespace(world_size=world_size, cpu_group=object())
    ps = types.ModuleType("aiter.dist.parallel_state")
    ps.get_pp_group = lambda: group
    for name, module in (
        ("aiter", types.ModuleType("aiter")),
        ("aiter.dist", types.ModuleType("aiter.dist")),
        ("aiter.dist.parallel_state", ps),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return group


def test_stages_that_agree_build(monkeypatch):
    fake_pp_group(monkeypatch, 8)
    install_pp_layer_partition(K3_LAYERS, BALANCED)

    def gather(out, mine, group):
        out[:] = [list(mine)] * len(out)

    with mock.patch("torch.distributed.all_gather_object", side_effect=gather):
        model_utils.agree_pp_layer_partition(K3_LAYERS)


def test_stages_that_disagree_refuse_together(monkeypatch):
    fake_pp_group(monkeypatch, 8)
    install_pp_layer_partition(K3_LAYERS, BALANCED)

    def gather(out, mine, group):
        # stage 5 runs on a node that resolved the even split
        out[:] = [list(mine)] * len(out)
        out[5] = list(EVEN_8)

    with (
        mock.patch("torch.distributed.all_gather_object", side_effect=gather),
        pytest.raises(RuntimeError, match="split the 93 layers differently"),
    ):
        model_utils.agree_pp_layer_partition(K3_LAYERS)


def test_compile_cache_key_follows_the_resolved_split():
    def key(partition):
        config = SimpleNamespace(
            quant_config=None,
            compilation_config=None,
            parallel_config=None,
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            sequence_parallel_size=1,
            dcp_config=None,
            enable_dp_attention=False,
            index_cache_dtype=None,
            index_cache_layout=None,
            hf_config=SimpleNamespace(),
            max_model_len=4096,
            max_num_seqs=64,
            max_num_batched_tokens=16384,
            kv_cache_dtype="fp8",
            speculative_config=None,
            pp_layer_partition=partition,
        )
        return Config.compute_hash(config)

    assert key(BALANCED) != key(None)
    assert key(BALANCED) != key(EVEN_8)
    assert key(None) == key(None)
