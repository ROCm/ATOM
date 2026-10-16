# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen MonoKernel host/runtime contracts; no model weights or GPU needed."""

from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.qwen4_exp_mono import (
    check,
    config,
    contract,
    dims,
    dispatch,
    peer_layout,
    plan,
    state,
    step_meta,
    weights,
)
from atom.models.qwen4_exp_mono.kernels import (
    collective,
    finish,
    finish_shape,
    g1,
    gdn_layer,
    gdn_tp8,
    qsa_post,
    qsa_pre,
    qsa_tp8,
)
from atom.mono.runtime import deployment
from atom.mono.runtime.consensus import MonoUnsupported
from atom.quant_spec import QuantType
from atom.utils.debug_helper import dump as debug_dump


@pytest.mark.parametrize("paired_hc,shared_only", [(False, False), (True, False)])
def test_tp8_shared_g2_pair_requires_shared_g1(paired_hc, shared_only):
    key = plan.MonoBuild(2, plan.LayerKind.GDN, tp=8,
                         expert_layout=plan.ExpertLayout.EXPERT_ID)
    with pytest.raises(ValueError, match="paired shared G2 requires"):
        finish._build_finish_tp8(key, paired_hc=paired_hc, shared_only=shared_only,
                                 paired_shared_g2=True)


class _Mode:
    def requires_piecewise_compilation(self):
        return False


def _atom_config():
    layer_types = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(48)
    ]
    return SimpleNamespace(
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_tbo=False,
        enable_tbo_decode=False,
        compilation_config=SimpleNamespace(cudagraph_mode=_Mode()),
        torch_dtype=torch.bfloat16,
        kv_cache_dtype="bf16",
        speculative_config=None,
        # The released checkpoint contains a vision tower. Mono only takes an
        # eligible decode step; image prefills and image-bearing forwards stay
        # on the original model path.
        multimodal_config=SimpleNamespace(vision_config=object()),
        hf_config=SimpleNamespace(
            num_hidden_layers=48,
            hidden_size=2560,
            hc_count=4,
            hc_lowrank=320,
            num_experts=512,
            num_experts_per_tok=10,
            moe_intermediate_size=640,
            shared_expert_intermediate_size=640,
            layer_types=layer_types,
            ple_layer_ids=[2],
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            num_attention_heads=24,
            num_key_value_heads=2,
            head_dim=256,
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
            ple_conv_kernel_size=4,
            ngram_size=3,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            mamba_ssm_dtype="float32",
            norm_topk_prob=True,
            output_gate_type="sigmoid",
        ),
    )


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    monkeypatch.setenv("ATOM_QWEN_MONO_ENABLE", "1")
    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)


def _layers():
    cfg = _atom_config().hf_config
    return [
        SimpleNamespace(
            layer_type=kind,
            ple=object() if i + 1 in cfg.ple_layer_ids else None,
        )
        for i, kind in enumerate(cfg.layer_types)
    ]


def test_plan_covers_every_layer_and_declares_width_gates():
    task_plan = plan.make_plan(_layers())
    assert len(task_plan.layers) == 48
    assert sum(task.is_gdn for task in task_plan.layers) == 36
    assert sum(task.ple for task in task_plan.layers) == 1
    assert sum(task.kind is plan.LayerKind.QSA for task in task_plan.layers) == 12
    assert config.SUPPORTED_WIDTHS == (1, 2, 4, 8)
    assert config.ENABLED_WIDTHS == (1,)
    assert config.SUPPORTED_TP == (1, 8)
    assert config.ENABLED_TP == (1,)
    for width in config.SUPPORTED_WIDTHS:
        for kind in plan.LayerKind:
            assert plan.MonoBuild(width, kind).tokens == width
    with pytest.raises(ValueError, match="unsupported Qwen mono width 3"):
        plan.MonoBuild(3, plan.LayerKind.GDN)
    tp8 = plan.MonoBuild(
        1,
        plan.LayerKind.GDN,
        tp=8,
        expert_layout=plan.ExpertLayout.EXPERT_ID,
    )
    assert tp8.dims.local_experts == 64
    with pytest.raises(ValueError, match="requires expert_id"):
        plan.MonoBuild(1, plan.LayerKind.GDN, tp=8)


def test_qsa_pre_allocates_hc_up_group_mailbox():
    layout = qsa_pre.scratch_layout()
    offset, size = layout["up_group_ready"]
    assert size == 2 * 8  # two tagged (payload, generation) pairs
    assert offset % config.ALIGN == 0
    assert offset + size <= layout["qsa_qkv_ready"][0]


def test_tp8_dims_match_native_rank_local_shapes():
    d = dims.Dims(8)
    assert (d.gdn_key_heads, d.gdn_value_heads) == (2, 6)
    assert (d.gdn_key_dim, d.gdn_value_dim) == (256, 768)
    assert (d.gdn_conv_dim, d.gdn_proj_dim) == (1280, 2060)
    assert (d.qsa_heads, d.qsa_q_size) == (3, 768)
    assert (d.qsa_kv_heads, d.qsa_kv_size, d.qsa_kv_replicas) == (1, 256, 4)
    assert d.qsa_qkv_size == 2048
    assert d.qsa_index_size == 640
    assert d.shared_intermediate == 80
    assert d.routed_intermediate == 640
    assert d.local_experts == 64
    assert d.expert_layout == "expert_id"


def test_tp8_compact_pull_peer_layout_has_no_source_axis():
    layout = peer_layout.peer_layout(1, 8)
    assert tuple(layout) == ("attention", "moe")
    assert layout["attention"] == (0, 20_480)
    assert layout["moe"] == (20_480, 20_480)
    assert peer_layout.peer_bytes(1, 8) == 40_960
    assert peer_layout.peer_bytes(8, 8) == 327_680
    assert state.peer_buffer_bytes(1) == state.PEER_BYTES == 16
    assert state.peer_buffer_bytes(8) == 40_960
    assert peer_layout.pair_index(0, 1, 0, 0) == 0
    assert peer_layout.pair_index(1, 1, 0, 0) == config.HIDDEN // 2
    assert peer_layout.pair_index(2, 1, 0, 0) == 0
    assert collective.PART_COLS == 32


@pytest.mark.parametrize("tokens", (1, 2, 4, 8))
def test_tp8_pull_widths_cover_disjoint_peer_rows(tokens):
    collective.build_pull(tokens, 8)
    pairs = config.HIDDEN // 2
    for layer in (0, 1, 2, 47):
        slot = layer % peer_layout.SLOTS
        for token in range(tokens):
            first = peer_layout.pair_index(layer, tokens, token, 0)
            last = peer_layout.pair_index(layer, tokens, token, config.HIDDEN - 2)
            assert first == (slot * tokens + token) * pairs
            assert last == first + pairs - 1


@pytest.mark.parametrize("tokens,tp", ((0, 8), (3, 8), (16, 8), (2, 1)))
def test_tp8_pull_rejects_unimplemented_widths(tokens, tp):
    with pytest.raises(ValueError, match="supports S=1/S=2/S=4/S=8 TP8"):
        collective.build_pull(tokens, tp)


def test_unknown_layer_and_ple_qsa_are_refused():
    with pytest.raises(MonoUnsupported, match="unknown attention"):
        plan.classify_layer(SimpleNamespace(layer_type="dense", ple=None), 3)
    with pytest.raises(MonoUnsupported, match="PLE on QSA"):
        plan.classify_layer(SimpleNamespace(layer_type="full_attention", ple=1), 3)


def test_s1_finish_abi_and_scratch_are_fixed_and_compact():
    assert finish.ABI.names[:3] == (
        "streams",
        "attn_out",
        "attn_raw",
    )
    assert finish.ABI.names[3:6] == (
        "hc_norm",
        "hc_down_inject",
        "hc_up",
    )
    assert finish.ABI.names[-5:] == (
        "pending_out",
        "pending_raw",
        "moe_out",
        "scratch",
        "layer",
    )
    layout = finish.hc_scratch_layout()
    assert tuple(layout) == (
        "hc_norm_ready",
        "hc_down_ready",
        "hc_gate_ready",
        "hc_up_ready",
        "hc_normed",
        "hc_part",
        "hc_gate",
    )
    assert finish_shape.HC_DOWN_TASKS == 84
    assert finish_shape.HC_OUT_TASKS == 160
    assert finish.hc_scratch_bytes() == 28_304
    finish_layout = finish.scratch_layout()
    assert tuple(finish_layout)[-18:] == (
        "up_group_ready",
        "xq_ready",
        "router_tile_ready",
        "route_ready",
        "g1_ready",
        "midq_ready",
        "shared_mid_ready",
        "xq",
        "xq_scale",
        "router_logits",
        "route_ids",
        "route_weights",
        "shared_gate",
        "g1_queue",
        "routed_mid",
        "shared_mid",
        "midq",
        "midq_scale",
    )
    assert finish_shape.XQ_GROUPS == 20
    assert finish_shape.XQ_WORDS == 640
    assert finish_shape.ROUTER_TASKS == 33
    assert finish_shape.G1_TASKS == 440
    assert finish_shape.G2_TASKS == 160
    assert finish.scratch_bytes() == 57_680
    finish_spans = sorted(
        (off, off + size, name) for name, (off, size) in finish_layout.items()
    )
    assert all(left[1] <= right[0] for left, right in pairwise(finish_spans))
    assert finish.FinishSmem.__dsl_size_of__() == 21_776
    spans = sorted((off, off + size, name) for name, (off, size) in layout.items())
    assert all(left[1] <= right[0] for left, right in pairwise(spans))


def test_finish_builds_s1_s2_and_pair_composed_s4():
    for width in (1, 2, 4):
        finish.build_finish(plan.MonoBuild(width, plan.LayerKind.GDN))
        gdn_layer.build_gdn_layer(plan.MonoBuild(width, plan.LayerKind.GDN))
    with pytest.raises(ValueError, match="S=1/S=2/S=4"):
        finish.build_finish(plan.MonoBuild(8, plan.LayerKind.GDN))
    with pytest.raises(ValueError, match="S=1/S=2/S=4"):
        gdn_layer.build_gdn_layer(plan.MonoBuild(8, plan.LayerKind.GDN))


@pytest.mark.parametrize("tokens", (1, 2, 4))
def test_tp8_finish_builds_expert_owned_width_specializations(tokens):
    key = plan.MonoBuild(
        tokens,
        plan.LayerKind.GDN,
        tp=8,
        expert_layout=plan.ExpertLayout.EXPERT_ID,
    )
    finish.build_finish(key)
    assert "expert_map" in finish.TP8_ABI.names
    assert "peer_sym" in finish.TP8_ABI.names
    with pytest.raises(ValueError, match="supports only S=1/S=2/S=4"):
        finish.build_finish(
            plan.MonoBuild(
                8,
                plan.LayerKind.GDN,
                tp=8,
                expert_layout=plan.ExpertLayout.EXPERT_ID,
            )
        )


def test_tp8_gdn_k1_k2_have_fixed_shape_abis_and_fit_the_workspace():
    key = plan.MonoBuild(
        1,
        plan.LayerKind.GDN,
        tp=8,
        expert_layout=plan.ExpertLayout.EXPERT_ID,
    )
    gdn_tp8.build_gdn_k1(key)
    gdn_tp8.build_gdn_k2(key)
    assert (gdn_tp8.KEY_HEADS, gdn_tp8.VALUE_HEADS) == (2, 6)
    assert (gdn_tp8.CONV_DIM, gdn_tp8.PROJ_DIM) == (1280, 2060)
    assert (gdn_tp8.PROJECTION_TASKS, gdn_tp8.RECURRENCE_TASKS) == (129, 12)
    assert gdn_tp8.PROJ_DIM % gdn_tp8.ROWS == 12
    assert gdn_tp8.SCRATCH_BYTES <= state.LAYER_SCRATCH_BYTES
    assert gdn_tp8.K1_ABI.names[-3:] == (
        "peer_sym",
        "scratch",
        "layer",
    )
    assert gdn_tp8.K2_ABI.names[:3] == ("peers", "streams", "attn_out")
    assert gdn_tp8.K2_ABI.names[-3:] == ("moe_out", "scratch", "layer")


def test_workspace_is_aligned_and_matches_the_layer_contract():
    assert finish.scratch_bytes() == 57_680
    assert state.LAYER_SCRATCH_BYTES == 144_896
    assert state.LAYER_SCRATCH_BYTES % 16 == 0
    assert state.QSA_PRE_SCRATCH_BYTES == qsa_pre.scratch_bytes()
    assert state.QSA_PRE_SCRATCH_BYTES % 16 == 0
    assert state.QSA_SCRATCH_BYTES % 16 == 0
    assert state.QSA_POST_SCRATCH_BYTES == qsa_post.scratch_bytes()
    assert state.QSA_POST_SCRATCH_BYTES % 16 == 0
    assert state.QSA_POST_SCRATCH_OFFSET >= state.QSA_PRE_SCRATCH_BYTES
    assert state.QSA_POST_SCRATCH_OFFSET % 16 == 0
    assert state.QSA_SCRATCH_BYTES == (
        state.QSA_POST_SCRATCH_OFFSET + state.QSA_POST_SCRATCH_BYTES
    )
    assert state.PEER_BYTES == 16
    for width in config.SUPPORTED_WIDTHS:
        assert state.layer_scratch_bytes(width) == width * state.LAYER_SCRATCH_BYTES
        assert state.qsa_post_scratch_offset(width) % config.ALIGN == 0
        assert state.qsa_post_scratch_offset(width) >= (
            width * state.QSA_PRE_SCRATCH_BYTES
        )
        assert state.qsa_width_scratch_bytes(width) == (
            state.qsa_post_scratch_offset(width) + width * state.QSA_POST_SCRATCH_BYTES
        )
    assert gdn_layer.LayerSmem.__dsl_size_of__() == finish.FinishSmem.__dsl_size_of__()


def test_width_scratch_partitions_have_token_stride_without_aliasing():
    width = 4
    arena = torch.zeros(state.layer_scratch_bytes(width), dtype=torch.uint8)
    out_off, _ = state.LAYER_SCRATCH["gdn_out"]
    rows = state.MonoState._partition_view(
        arena,
        width,
        state.LAYER_SCRATCH_BYTES,
        out_off,
        torch.bfloat16,
        (config.HIDDEN,),
    )
    assert rows.shape == (width, config.HIDDEN)
    assert rows.stride(0) == state.LAYER_SCRATCH_BYTES // 2
    rows[:, 0] = torch.arange(width, dtype=torch.bfloat16)
    for token in range(width):
        raw = arena.narrow(0, token * state.LAYER_SCRATCH_BYTES + out_off, 2).view(
            torch.bfloat16
        )
        assert raw.item() == token


def test_qsa_k1_abi_and_seam_are_explicit():
    assert qsa_pre.ABI.names[:6] == (
        "streams",
        "pending_out",
        "pending_raw",
        "hc_norm",
        "hc_down_inject",
        "hc_up",
    )
    assert qsa_pre.ABI.names[-7:] == (
        "streams_out",
        "attn_raw_out",
        "query_out",
        "gate_out",
        "index_query_out",
        "scratch",
        "layer",
    )
    layout = qsa_pre.scratch_layout()
    assert tuple(layout)[-5:] == (
        "qsa_qkv_ready",
        "qsa_index_ready",
        "qsa_mixed",
        "qsa_qkv",
        "qsa_index",
    )
    assert qsa_pre.QKV_GROUPS == 832
    assert qsa_pre.INDEX_GROUPS == 40
    assert qsa_pre.scratch_bytes() == state.QSA_PRE_SCRATCH_BYTES
    assert finish.FinishS2Smem.__dsl_size_of__() == 27_152
    for width in (1, 2, 4):
        qsa_pre.build_qsa_pre(plan.MonoBuild(width, plan.LayerKind.QSA))
    with pytest.raises(ValueError, match="S=1/S=2/S=4"):
        qsa_pre.build_qsa_pre(plan.MonoBuild(8, plan.LayerKind.QSA))


def test_qsa_k1_s4_pair_partition_covers_projection_tasks_once():
    blocks = qsa_pre.S4_PAIR_BLOCKS
    projection = [
        bid + turn * blocks
        for bid in range(blocks)
        for turn in range(qsa_pre.S4_PROJECTION_TURNS)
        if bid + turn * blocks < qsa_pre.TOTAL_PROJECTION_GROUPS
    ]
    assert sorted(projection) == list(range(qsa_pre.TOTAL_PROJECTION_GROUPS))


def test_tp8_qsa_k1_has_rank_local_seam_and_fits_existing_arena():
    key = plan.MonoBuild(
        1,
        plan.LayerKind.QSA,
        tp=8,
        expert_layout=plan.ExpertLayout.EXPERT_ID,
    )
    qsa_tp8.build_qsa_k1(key)
    assert (qsa_tp8.Q_HEADS, qsa_tp8.KV_HEADS) == (3, 1)
    assert (qsa_tp8.Q_SIZE, qsa_tp8.KV_SIZE, qsa_tp8.QKV_ROWS) == (
        768,
        256,
        2048,
    )
    assert qsa_tp8.TOTAL_PROJECTION_TASKS == 168
    assert qsa_tp8.EPILOGUE_TASKS == 10
    assert qsa_tp8.scratch_bytes() == 5_184
    assert qsa_tp8.scratch_bytes() <= state.QSA_PRE_SCRATCH_BYTES
    assert qsa_tp8.ABI.names[:5] == (
        "mixed",
        "qkv_weight",
        "q_norm",
        "k_norm",
        "index_weight",
    )
    assert qsa_tp8.ABI.names[-5:] == (
        "query_out",
        "gate_out",
        "index_query_out",
        "scratch",
        "layer",
    )

    mono = object.__new__(state.MonoState)
    mono.dims = dims.Dims(8)
    view = mono._allocate_view(1, torch.device("cpu"))
    assert view.qsa.query.shape == (1, 3, 256)
    assert view.qsa.gate.shape == (1, 768)
    assert view.qsa.index_query.shape == (1, 4, 128)


def test_s4_g1_static_pair_partition_covers_each_routed_task_once():
    stride = g1.G1_S4_PAIR_FIXED // 2
    routed = {0: [], 1: []}
    shared_owners = {0: [], 1: []}
    for bid in range(g1.G1_S4_PAIR_FIXED):
        token = bid % 2
        task = bid // 2
        while task < finish_shape.G1_ROUTED_TASKS:
            routed[token].append(task)
            task += stride

        shared = (
            bid - finish_shape.ROUTER_TASKS + g1.G1_S4_PAIR_FIXED
        ) % g1.G1_S4_PAIR_FIXED
        if shared < finish_shape.G1_SHARED_TASKS:
            shared_owners[token].append(shared)

    expected = list(range(finish_shape.G1_ROUTED_TASKS))
    assert sorted(routed[0]) == expected
    assert sorted(routed[1]) == expected
    assert len(routed[0]) == len(set(routed[0]))
    assert len(routed[1]) == len(set(routed[1]))
    assert sorted(shared_owners[0] + shared_owners[1]) == list(
        range(finish_shape.G1_SHARED_TASKS)
    )
    assert len(shared_owners[0]) == len(shared_owners[1]) == 20


def test_qsa_k2_abi_and_task_graph_are_explicit():
    assert qsa_post.ABI.names[:5] == (
        "streams",
        "attn_raw",
        "query",
        "gate",
        "index_query",
    )
    assert qsa_post.ABI.names[-7:] == (
        "routed_w2_scale",
        "streams_out",
        "pending_out",
        "pending_raw",
        "moe_out",
        "scratch",
        "layer",
    )
    layout = qsa_post.scratch_layout()
    assert qsa_post.INDEX_SCORE_TASKS == 128
    assert qsa_post.ATTN_KEYS_PER_SPLIT == 16
    assert qsa_post.ATTN_SPLITS == 129
    assert qsa_post.ATTN_TASKS == 258
    assert qsa_post.DIRECT_ATTN_SPLITS == 16
    assert qsa_post.DIRECT_ATTN_SPLITS * qsa_post.ATTN_KEYS_PER_SPLIT == 256
    assert qsa_post.OPROJ_TASKS == 160
    assert tuple(layout)[-8:] == (
        "qsa_scores",
        "qsa_selected_groups",
        "qsa_selected_tokens",
        "qsa_attn_max",
        "qsa_attn_sum",
        "qsa_attn_partials",
        "qsa_gated",
        "qsa_attn_out",
    )
    assert qsa_post.QsaPostSmem.__dsl_size_of__() == finish.FinishSmem.__dsl_size_of__()
    for width in (1, 2, 4):
        qsa_post.build_qsa_post(
            plan.MonoBuild(width, plan.LayerKind.QSA),
            direct=True,
            direct_splits=qsa_post.DIRECT_ATTN_SPLITS,
        )
    with pytest.raises(ValueError, match="S=1/S=2/S=4"):
        qsa_post.build_qsa_post(plan.MonoBuild(8, plan.LayerKind.QSA))


def test_gdn_step_view_binds_graph_stable_state_without_copying():
    conv = torch.empty(8, 3, 10240, dtype=torch.bfloat16)
    recurrent = torch.empty(8, 48, 128, 128, dtype=torch.float32).transpose(-1, -2)
    tokens = 4
    read = torch.tensor([3, 2, 1, 0], dtype=torch.int32)
    write = torch.tensor([5, 4, 3, 2], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=tokens,
        num_actual_tokens=tokens,
        num_spec_decodes=0,
        replayssm=False,
        non_spec_state_indices_in_tensor=read,
        non_spec_state_indices_tensor=write,
    )
    fwd = SimpleNamespace(
        attn_metadata=SimpleNamespace(gdn_metadata=metadata),
        kv_cache_data={"layer_7": SimpleNamespace(k_cache=conv, v_cache=recurrent)},
    )
    view = step_meta.gdn_step_view(fwd, 7, tokens)
    assert view.conv_state is conv
    assert view.recurrent_state is recurrent
    assert view.state_read is read
    assert view.state_write is write

    fwd.kv_cache_data["layer_7"].v_cache = recurrent.to(torch.bfloat16)
    with pytest.raises(MonoUnsupported, match="recurrent_state is torch.bfloat16"):
        step_meta.gdn_step_view(fwd, 7, tokens)

    malformed = torch.empty(8, 128, 48, 128, dtype=torch.float32).transpose(1, 2)
    fwd.kv_cache_data["layer_7"].v_cache = malformed
    with pytest.raises(
        MonoUnsupported, match="recurrent_state is torch.float32 stride="
    ):
        step_meta.gdn_step_view(fwd, 7, tokens)


def test_gdn_step_view_rejects_speculative_or_malformed_state():
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=1,
        num_actual_tokens=1,
        num_spec_decodes=1,
        replayssm=False,
        non_spec_state_indices_in_tensor=torch.zeros(1, dtype=torch.int32),
        non_spec_state_indices_tensor=torch.zeros(1, dtype=torch.int32),
    )
    fwd = SimpleNamespace(
        attn_metadata=SimpleNamespace(gdn_metadata=metadata), kv_cache_data={}
    )
    with pytest.raises(MonoUnsupported, match="non-speculative"):
        step_meta.gdn_step_view(fwd, 0, 1)


def test_qsa_step_view_binds_metadata_and_cache_planes_without_copying():
    tokens = 4
    qsa = SimpleNamespace(
        block_tables=torch.zeros(tokens, 8, dtype=torch.int32),
        slot_mapping=torch.arange(5, 5 + tokens, dtype=torch.int64),
        compressed_slot_mapping=torch.full((tokens,), -1, dtype=torch.int64),
        token_to_req=torch.arange(tokens, dtype=torch.int32),
        logical_positions=torch.arange(5, 5 + tokens, dtype=torch.int64),
        seq_lens=torch.full((tokens,), 6, dtype=torch.int32),
        max_seq_len=6,
    )
    qsa.for_tokens = lambda rows: qsa if rows == tokens else None
    attention = SimpleNamespace(
        k_cache=torch.empty(4, 16, 2, 256, dtype=torch.bfloat16),
        v_cache=torch.empty(4, 16, 2, 256, dtype=torch.bfloat16),
        raw_key_cache=torch.empty(4, 16, 1, 128, dtype=torch.bfloat16),
        compressed_key_cache=torch.empty(4, 4, 1, 128, dtype=torch.bfloat16),
        rope_position_cache=torch.empty(4, 16, 1, 3, dtype=torch.int64),
    )
    fwd = SimpleNamespace(attn_metadata=SimpleNamespace(qsa_metadata=qsa))
    view = step_meta.qsa_step_view(fwd, attention, tokens)
    assert view.metadata is qsa
    assert view.k_cache is attention.k_cache
    assert view.compressed_key_cache is attention.compressed_key_cache
    assert view.max_seq_len == 6

    attention.raw_key_cache = torch.empty(4, 16, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(MonoUnsupported, match="raw cache"):
        step_meta.qsa_step_view(fwd, attention, tokens)


def test_qsa_step_view_uses_rank_local_tp8_kv_cache_shape():
    tokens = 1
    qsa = SimpleNamespace(
        block_tables=torch.zeros(tokens, 8, dtype=torch.int32),
        slot_mapping=torch.zeros(tokens, dtype=torch.int64),
        compressed_slot_mapping=torch.full((tokens,), -1, dtype=torch.int64),
        token_to_req=torch.zeros(tokens, dtype=torch.int32),
        logical_positions=torch.zeros(tokens, dtype=torch.int64),
        seq_lens=torch.ones(tokens, dtype=torch.int32),
        max_seq_len=1,
    )
    qsa.for_tokens = lambda rows: qsa if rows == tokens else None
    attention = SimpleNamespace(
        k_cache=torch.empty(4, 16, 1, 256, dtype=torch.bfloat16),
        v_cache=torch.empty(4, 16, 1, 256, dtype=torch.bfloat16),
        raw_key_cache=torch.empty(4, 16, 1, 128, dtype=torch.bfloat16),
        compressed_key_cache=torch.empty(4, 4, 1, 128, dtype=torch.bfloat16),
        rope_position_cache=None,
    )
    fwd = SimpleNamespace(attn_metadata=SimpleNamespace(qsa_metadata=qsa))

    view = step_meta.qsa_step_view(fwd, attention, tokens, tp=8)
    assert view.k_cache.shape[-2:] == (1, 256)

    with pytest.raises(MonoUnsupported, match="TP1 QSA K/V cache tail"):
        step_meta.qsa_step_view(fwd, attention, tokens)


def test_finish_resource_contract_caps_scalar_spills(monkeypatch):
    accepted = {
        "name": "finish",
        "scratch": 0,
        "lds": 21_764,
        "vgpr_spill": 0,
        "sgpr_spill": 75,
    }
    monkeypatch.setattr(contract, "launcher_resources", lambda _launcher: [accepted])
    assert contract.check_launcher(object()) == [accepted]
    rejected = {**accepted, "sgpr_spill": 76}
    monkeypatch.setattr(contract, "launcher_resources", lambda _launcher: [rejected])
    with pytest.raises(RuntimeError, match="sgpr_spill=76"):
        contract.check_launcher(object())
    assert contract.sgpr_spill_limit(4, "gdn") == 532
    assert contract.sgpr_spill_limit(4, "qsa_direct") == 599
    assert contract.sgpr_spill_limit(4, "qsa_full") == 924
    with pytest.raises(ValueError, match="no resource contract"):
        contract.sgpr_spill_limit(8, "gdn")


def test_check_finish_compares_repeatable_reference(monkeypatch):
    calls = []
    values = iter(
        [
            ("ref_streams", "ref_raw", "ref_moe"),
            ("again_streams", "again_raw", "again_moe"),
        ]
    )
    monkeypatch.setattr(check, "_reference", lambda *_args: next(values))
    monkeypatch.setattr(
        check, "compare", lambda got, ref: calls.append((got, ref)) or "ok"
    )
    check.check_finish(
        0,
        7,
        object(),
        "streams",
        "attn_out",
        "attn_raw",
        "mono_streams",
        "mono_raw",
        "mono_moe",
    )
    assert calls == [
        ("again_streams", "ref_streams"),
        ("again_raw", "ref_raw"),
        ("again_moe", "ref_moe"),
        ("mono_streams", "ref_streams"),
        ("mono_raw", "ref_raw"),
        ("mono_moe", "ref_moe"),
    ]


class _HC:
    def __init__(self):
        self.hc_norm = SimpleNamespace(
            weight=torch.empty(4 * 2560, dtype=torch.bfloat16, device="meta")
        )
        self.fused_w_cat = torch.empty(
            324, 4 * 2560, dtype=torch.bfloat16, device="meta"
        )
        self.input_mix_weight_up = SimpleNamespace(
            weight=torch.empty(4 * 2560, 320, dtype=torch.bfloat16, device="meta")
        )


class _Quant:
    block_quant = True
    quant_type = QuantType.per_1x128


class _Experts:
    def __init__(self):
        self.num_fused_shared_experts = 0
        self.intermediate_size_per_partition = 640
        self.quant_method = _Quant()
        self.w13_input_scale = None
        self.w2_input_scale = None
        self.w13_weight = torch.empty(
            512, 1280, 2560, dtype=torch.float8_e4m3fn, device="meta"
        )
        self.w13_weight.is_shuffled = True
        self.w13_weight_scale = torch.empty(
            512, 10, 20, dtype=torch.float32, device="meta"
        )
        self.w2_weight = torch.empty(
            512, 2560, 640, dtype=torch.float8_e4m3fn, device="meta"
        )
        self.w2_weight.is_shuffled = True
        self.w2_weight_scale = torch.empty(
            512, 20, 5, dtype=torch.float32, device="meta"
        )


class _Shared:
    def __init__(self):
        self.gate_up_proj = SimpleNamespace(
            weight=torch.empty(1280, 2560, dtype=torch.bfloat16, device="meta")
        )
        self.down_proj = SimpleNamespace(
            weight=torch.empty(2560, 640, dtype=torch.bfloat16, device="meta")
        )


class _Moe:
    def __init__(self):
        self.fuse_shared = False
        self.gate = SimpleNamespace(
            weight=torch.empty(513, 2560, dtype=torch.bfloat16, device="meta")
        )
        self.shared_expert = _Shared()
        self.experts = _Experts()


class _Gdn:
    quantized_inputs = False
    num_k_heads = 16
    num_v_heads = 48
    head_k_dim = 128
    head_v_dim = 128
    conv_kernel_size = 4
    conv_dim = 10240
    value_dim = 6144

    def __init__(self):
        self.in_proj_qkvzba = SimpleNamespace(
            weight=torch.empty(16480, 2560, dtype=torch.bfloat16, device="meta")
        )
        self.conv1d = SimpleNamespace(
            weight=torch.empty(10240, 1, 4, dtype=torch.bfloat16, device="meta")
        )
        self.A_log = torch.empty(48, dtype=torch.bfloat16, device="meta")
        self.dt_bias = torch.empty(48, dtype=torch.bfloat16, device="meta")
        self.norm = SimpleNamespace(
            weight=torch.empty(128, dtype=torch.bfloat16, device="meta")
        )
        self.out_proj = SimpleNamespace(
            weight=torch.empty(2560, 6144, dtype=torch.bfloat16, device="meta")
        )


class _Projection:
    def __init__(self, rows):
        self.weight = torch.empty(rows, 2560, dtype=torch.bfloat16, device="meta")


class _Rope:
    def __init__(self):
        self.cos_cache = torch.empty(
            8192, 1, 1, 32, dtype=torch.bfloat16, device="meta"
        )
        self.sin_cache = torch.empty(
            8192, 1, 1, 32, dtype=torch.bfloat16, device="meta"
        )


class _Qsa:
    def __init__(self):
        self.qkv_proj = _Projection(13312)
        self.o_proj = SimpleNamespace(
            weight=torch.empty(2560, 6144, dtype=torch.bfloat16, device="meta")
        )
        self.q_norm = SimpleNamespace(
            weight=torch.empty(256, dtype=torch.bfloat16, device="meta")
        )
        self.k_norm = SimpleNamespace(
            weight=torch.empty(256, dtype=torch.bfloat16, device="meta")
        )
        self.rotary_emb = _Rope()
        self.indexer = SimpleNamespace(
            index_qk_proj=_Projection(640),
            q_layernorm=SimpleNamespace(
                weight=torch.empty(128, dtype=torch.bfloat16, device="meta")
            ),
            k_layernorm=SimpleNamespace(
                weight=torch.empty(128, dtype=torch.bfloat16, device="meta")
            ),
            rotary_emb=_Rope(),
        )


def _weighted_layer(kind="linear_attention"):
    layer = SimpleNamespace(
        layer_type=kind,
        ple=None,
        attn_hyper_connection=_HC(),
        mlp_hyper_connection=_HC(),
        mlp=_Moe(),
    )
    if kind == "linear_attention":
        layer.linear_attn = _Gdn()
    else:
        layer.self_attn = _Qsa()
    return layer


def test_weight_binder_checks_shape_dtype_and_contiguity():
    task = plan.classify_layer(_weighted_layer(), 0)
    bound = weights.bind_layer(_weighted_layer(), task)
    assert bound.task is task
    assert tuple(bound.mlp_hc.up.shape) == (10240, 320)
    assert tuple(bound.moe.w13.shape) == (512, 1280, 2560)
    assert bound.moe.expert_map is None
    assert bound.gdn is not None
    assert tuple(bound.gdn.qkvzba.shape) == (16480, 2560)
    assert tuple(bound.gdn.out.shape) == (2560, 6144)
    bad = _weighted_layer()
    bad.mlp_hyper_connection.hc_norm.weight = torch.empty(
        10240, dtype=torch.float32, device="meta"
    )
    with pytest.raises(MonoUnsupported, match="expected torch.bfloat16"):
        weights.bind_layer(bad, task)
    bad = _weighted_layer()
    bad.mlp.experts.w13_weight.is_shuffled = False
    with pytest.raises(MonoUnsupported, match="shuffle_weight"):
        weights.bind_layer(bad, task)
    bad = _weighted_layer()
    bad.mlp.experts.w2_weight_scale = torch.empty(
        512, 20, 4, dtype=torch.float32, device="meta"
    )
    with pytest.raises(MonoUnsupported, match="w2_weight_scale"):
        weights.bind_layer(bad, task)
    bad = _weighted_layer()
    bad.linear_attn.in_proj_qkvzba.weight = torch.empty(
        16479, 2560, dtype=torch.bfloat16, device="meta"
    )
    with pytest.raises(MonoUnsupported, match="in_proj_qkvzba"):
        weights.bind_layer(bad, task)


def test_qsa_weight_binder_has_no_gdn_tensors():
    layer = _weighted_layer("full_attention")
    task = plan.classify_layer(layer, 3)
    bound = weights.bind_layer(layer, task)
    assert bound.gdn is None
    assert bound.qsa is not None
    assert tuple(bound.qsa.qkv.shape) == (13312, 2560)
    assert tuple(bound.qsa.o_proj.shape) == (2560, 6144)
    assert tuple(bound.qsa.index_qk.shape) == (640, 2560)
    assert tuple(bound.attn_hc.norm.shape) == (10240,)
    assert tuple(bound.mlp_hc.norm.shape) == (10240,)


def _weighted_layer_tp8(kind="linear_attention"):
    layer = _weighted_layer(kind)
    experts = layer.mlp.experts
    experts.use_ep = True
    experts.ep_size = 8
    experts.tp_size = 1
    experts.local_num_experts = 64
    # Native AITER appends one non-local-token sentinel after the routed map.
    experts.expert_map = torch.empty(513, dtype=torch.int32, device="meta")
    experts.w13_weight = torch.empty(
        64, 1280, 2560, dtype=torch.float8_e4m3fn, device="meta"
    )
    experts.w13_weight.is_shuffled = True
    experts.w13_weight_scale = torch.empty(
        64, 10, 20, dtype=torch.float32, device="meta"
    )
    experts.w2_weight = torch.empty(
        64, 2560, 640, dtype=torch.float8_e4m3fn, device="meta"
    )
    experts.w2_weight.is_shuffled = True
    experts.w2_weight_scale = torch.empty(64, 20, 5, dtype=torch.float32, device="meta")
    layer.mlp.shared_expert.gate_up_proj.weight = torch.empty(
        160, 2560, dtype=torch.bfloat16, device="meta"
    )
    layer.mlp.shared_expert.down_proj.weight = torch.empty(
        2560, 80, dtype=torch.bfloat16, device="meta"
    )
    if kind == "linear_attention":
        layer.linear_attn.in_proj_qkvzba.weight = torch.empty(
            2060, 2560, dtype=torch.bfloat16, device="meta"
        )
        layer.linear_attn.conv1d.weight = torch.empty(
            1280, 1, 4, dtype=torch.bfloat16, device="meta"
        )
        layer.linear_attn.A_log = torch.empty(6, dtype=torch.bfloat16, device="meta")
        layer.linear_attn.dt_bias = torch.empty(6, dtype=torch.bfloat16, device="meta")
        layer.linear_attn.out_proj.weight = torch.empty(
            2560, 768, dtype=torch.bfloat16, device="meta"
        )
    else:
        layer.self_attn.qkv_proj.weight = torch.empty(
            2048, 2560, dtype=torch.bfloat16, device="meta"
        )
        layer.self_attn.o_proj.weight = torch.empty(
            2560, 768, dtype=torch.bfloat16, device="meta"
        )
    return layer


@pytest.mark.parametrize("kind", ["linear_attention", "full_attention"])
def test_tp8_weight_binder_accepts_dense_shards_and_expert_ownership(kind):
    layer = _weighted_layer_tp8(kind)
    task = plan.classify_layer(layer, 0 if kind == "linear_attention" else 3)
    bound = weights.bind_layer(layer, task, tp=8)
    assert tuple(bound.moe.w13.shape) == (64, 1280, 2560)
    assert tuple(bound.moe.shared_w13.shape) == (160, 2560)
    assert bound.moe.expert_map is not None
    assert tuple(bound.moe.expert_map.shape) == (512,)
    if kind == "linear_attention":
        assert bound.gdn is not None
        assert tuple(bound.gdn.qkvzba.shape) == (2060, 2560)
        assert tuple(bound.gdn.out.shape) == (2560, 768)
    else:
        assert bound.qsa is not None
        assert tuple(bound.qsa.qkv.shape) == (2048, 2560)
        assert tuple(bound.qsa.o_proj.shape) == (2560, 768)


def test_supported_config_is_explicitly_opted_in(native):
    assert config.config_refusal(_atom_config()) is None


def test_tp8_requires_expert_ownership_and_stays_behind_production_gate(
    native, monkeypatch
):
    candidate = _atom_config()
    candidate.tensor_parallel_size = 8
    candidate.enable_expert_parallel = True
    assert config.config_refusal(candidate) == "TP 8 production gate"

    monkeypatch.setattr(config, "ENABLED_TP", (1, 8))
    assert config.config_refusal(candidate) is None
    candidate.enable_expert_parallel = False
    assert config.config_refusal(candidate) == "TP8 requires MoE expert ownership"

    candidate = _atom_config()
    candidate.enable_expert_parallel = True
    assert (
        config.config_refusal(candidate)
        == "MoE expert ownership is only supported at TP8"
    )


def test_off_switch_and_wrong_shape_keep_original_path(native, monkeypatch):
    monkeypatch.setenv("ATOM_QWEN_MONO_ENABLE", "0")
    assert config.config_refusal(_atom_config()) == "ATOM_QWEN_MONO_ENABLE is off"
    monkeypatch.setenv("ATOM_QWEN_MONO_ENABLE", "1")
    bad = _atom_config()
    bad.hf_config.hidden_size = 4096
    assert config.config_refusal(bad) == "hidden size"


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("tensor_parallel_size", 2, "TP 2"),
        ("kv_cache_dtype", "fp8", "KV cache is not BF16"),
        ("speculative_config", object(), "speculative decode"),
    ],
)
def test_unimplemented_deployment_modes_are_refused(native, field, value, reason):
    candidate = _atom_config()
    setattr(candidate, field, value)
    assert reason in config.config_refusal(candidate)


def test_numerical_and_layer_contracts_are_exact(native):
    candidate = _atom_config()
    candidate.hf_config.layer_types = [
        "qwen_sparse_attention" if kind == "full_attention" else kind
        for kind in candidate.hf_config.layer_types
    ]
    assert config.config_refusal(candidate) is None

    candidate = _atom_config()
    candidate.hf_config.rms_norm_eps = 1e-5
    assert config.config_refusal(candidate) == "RMS epsilon"

    candidate = _atom_config()
    candidate.hf_config.layer_types[0] = "full_attention"
    assert config.config_refusal(candidate) == "layer type table"


def _context(
    monkeypatch,
    *,
    prefill=False,
    ubatch=None,
    spec_decodes=0,
    replayssm=False,
    image_mask=None,
):
    gdn = SimpleNamespace(
        num_prefills=0,
        num_decodes=1,
        num_actual_tokens=1,
        num_spec_decodes=spec_decodes,
        replayssm=replayssm,
    )
    fwd = SimpleNamespace(
        context=SimpleNamespace(is_prefill=prefill),
        ubatch_slices=ubatch,
        attn_metadata=SimpleNamespace(gdn_metadata=gdn, image_mask=image_mask),
    )
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: fwd)
    return fwd


def test_single_token_decode_is_routed(monkeypatch):
    _context(monkeypatch)
    ids = torch.zeros(1, dtype=torch.int32)
    pos = torch.zeros(1, dtype=torch.int64)
    assert dispatch.step_supported(ids, pos, None, None)


def test_mrope_decode_positions_are_routed(monkeypatch):
    _context(monkeypatch)
    ids = torch.zeros(1, dtype=torch.int32)
    pos = torch.zeros(3, 1, dtype=torch.int64)
    assert dispatch.step_supported(ids, pos, None, None)


def test_multimodal_prefill_inputs_stay_on_the_original_path(monkeypatch):
    _context(monkeypatch, image_mask=torch.ones(1, dtype=torch.bool))
    ids = torch.zeros(1, dtype=torch.int32)
    pos = torch.zeros(3, 1, dtype=torch.int64)
    assert not dispatch.step_supported(ids, pos, None, None)

    _context(monkeypatch)
    embeds = torch.zeros(1, config.HIDDEN, dtype=torch.bfloat16)
    assert not dispatch.step_supported(ids, pos, None, embeds)


@pytest.mark.parametrize("shape", [(2, 4), (4, 4)])
def test_non_mrope_position_planes_fall_back(monkeypatch, shape):
    _context(monkeypatch)
    ids = torch.zeros(4, dtype=torch.int32)
    pos = torch.zeros(shape, dtype=torch.int64)
    assert not dispatch.step_supported(ids, pos, None, None)


def test_non_row_input_ids_fall_back(monkeypatch):
    _context(monkeypatch)
    ids = torch.zeros(2, 2, dtype=torch.int32)
    pos = torch.zeros(4, dtype=torch.int64)
    assert not dispatch.step_supported(ids, pos, None, None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spec_decodes": 1},
        {"replayssm": True},
        {"image_mask": object()},
    ],
)
def test_unimplemented_runtime_metadata_falls_back(monkeypatch, kwargs):
    _context(monkeypatch, **kwargs)
    ids = torch.zeros(1, dtype=torch.int32)
    pos = torch.zeros(1, dtype=torch.int64)
    assert not dispatch.step_supported(ids, pos, None, None)


@pytest.mark.parametrize(
    "tokens,prefill,ubatch",
    [
        (2, False, None),
        (4, False, None),
        (8, False, None),
        (3, False, None),
        (1, True, None),
        (1, False, object()),
    ],
)
def test_other_steps_fall_back(monkeypatch, tokens, prefill, ubatch):
    _context(monkeypatch, prefill=prefill, ubatch=ubatch)
    ids = torch.zeros(tokens, dtype=torch.int32)
    pos = torch.zeros(tokens, dtype=torch.int64)
    assert not dispatch.step_supported(ids, pos, None, None)


def test_non_atom_token_dtype_falls_back(monkeypatch):
    _context(monkeypatch)
    ids = torch.zeros(1, dtype=torch.int64)
    pos = torch.zeros(1, dtype=torch.int64)
    assert not dispatch.step_supported(ids, pos, None, None)


def test_refused_install_returns_the_original_model(native):
    model = torch.nn.Linear(2, 2)
    bad = _atom_config()
    bad.tensor_parallel_size = 2
    assert dispatch.install_mono_decode(model, bad, None, None) is model
    metadata_builder = SimpleNamespace()
    wrapped = dispatch.install_mono_decode(
        model, _atom_config(), None, metadata_builder
    )
    assert isinstance(wrapped, dispatch.MonoDecodeModel)
    assert wrapped.in_features == 2
    assert metadata_builder.cudagraph_context_buckets == (256, 8192)
    assert dispatch.install_mono_decode(model, _atom_config(), object(), None) is model


def test_permanent_build_refusal_closes_runtime_before_fallback(monkeypatch):
    class Original(torch.nn.Module):
        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            del positions, intermediate_tensors, inputs_embeds
            return input_ids + 1

    class Refused:
        enabled = False

        def __init__(self):
            self.runner = SimpleNamespace(closed=False)
            self.runner.close = lambda: setattr(self.runner, "closed", True)

        def ready(self, _tokens):
            return False

    wrapped = dispatch.MonoDecodeModel(Original())
    refused = Refused()
    runner = refused.runner
    wrapped._mono = refused
    monkeypatch.setattr(dispatch, "step_supported", lambda *_args: True)
    ids = torch.zeros(4, dtype=torch.int64)
    out = wrapped(ids, torch.zeros(4, dtype=torch.int64))
    assert torch.equal(out, torch.ones(4, dtype=torch.int64))
    assert runner.closed
    assert refused.runner is None


def test_forward_output_dump_selects_first_real_decode(monkeypatch, tmp_path):
    path = tmp_path / "forward.pt"
    monkeypatch.setenv("ATOM_FWD_OUTPUT_DUMP_PATH", str(path))
    debug_dump._forward_output_dumped_paths.clear()
    hidden = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]])

    debug_dump.maybe_dump_forward_outputs(
        hidden, logits, is_prefill=True, is_dummy=False
    )
    debug_dump.maybe_dump_forward_outputs(
        hidden, logits, is_prefill=False, is_dummy=True
    )
    assert not path.exists()

    debug_dump.maybe_dump_forward_outputs(
        hidden, logits, is_prefill=False, is_dummy=False
    )
    packet = torch.load(path, weights_only=True)
    assert torch.equal(packet["hidden"], hidden)
    assert torch.equal(packet["logits"], logits)
    assert torch.equal(packet["argmax"], torch.tensor([1, 2]))
    assert packet["hidden_shape"] == (2, 4)
    assert packet["logits_shape"] == (2, 3)

    debug_dump.maybe_dump_forward_outputs(
        hidden + 1, logits + 1, is_prefill=False, is_dummy=False
    )
    assert torch.equal(torch.load(path, weights_only=True)["hidden"], hidden)
