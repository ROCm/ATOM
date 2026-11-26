# SPDX-License-Identifier: MIT
"""Construct native KDA layers and reach the real fused API on CPU tensors."""

from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield NS(
            kimi=load("atom.models.kimi_k3.model"),
            glm=load("atom.models.glm5_next"),
            linear=load("atom.model_ops.linear"),
            replay=load("atom.model_ops.fla_ops.replayssm"),
            gdn=load("atom.model_ops.attentions.gdn_attn"),
            ops=load("atom.model_ops.kimi_k3"),
        )


def layer_case(modules, monkeypatch, verify=False, glm=False, hd=128):
    kimi, replay = modules.kimi, modules.replay
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        modules.linear, "get_tp_group", lambda: NS(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(kimi, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(kimi, "get_cu_num", lambda: 6)
    config = NS(
        hf_config=NS(
            hidden_size=16,
            linear_num_key_heads=2,
            linear_key_head_dim=hd,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6,
            linear_attn_config={"gate_lower_bound": -5.0},
        ),
        compilation_config=NS(static_forward_context={}),
        load_dummy=True,
    )
    monkeypatch.setattr(kimi, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(modules.glm, "get_current_atom_config", lambda: config)
    cls = modules.glm.Glm5NextKDAAttention if glm else kimi.KimiKDAAttention
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        attn = cls(config, None, prefix="layers.0.self_attn")
    finally:
        torch.set_default_dtype(dtype)
    torch.manual_seed(694)
    for weight in attn.parameters():
        weight.data.normal_(std=0.125)
    tail = torch.cat([attn.b_proj.weight, attn.f_a_proj.weight])
    attn.process_weights_after_loading()
    torch.testing.assert_close(attn.in_proj.weight[8 * hd :], tail)
    assert attn.in_proj.weight.shape == (9 * hd + 2, 16)
    assert attn.conv_weight.dtype == torch.float32
    assert config.compilation_config.static_forward_context[attn.layer_name] is attn
    # Build the production graph metadata: three padded requests, two live.
    rows = 8 if verify else 1
    cu = torch.tensor([0, rows, 2 * rows, 2 * rows], dtype=torch.int32)
    slots = torch.tensor([5, 1, -1], dtype=torch.int32)
    conv = torch.tensor([[0, 3], [2, 7], [-1, -1]], dtype=torch.int32)
    read = torch.tensor([[4, 7], [6, 3], [-1, -1]], dtype=torch.int32)
    builder = NS(
        use_spec_decode=verify,
        num_spec=rows - 1,
        replayssm=True,
        replayssm_cache_len=16,
        replayssm_route="auto",
        replayssm_max_query_len=rows,
        non_spec_state_indices_tensor=NS(gpu=slots),
        non_spec_state_indices_in_tensor=NS(gpu=read[:, 0]),
        spec_state_indices_tensor=NS(gpu=conv),
        spec_query_start_loc=cu,
        non_spec_query_start_loc=cu,
        spec_sequence_masks=torch.ones(3, dtype=torch.bool),
        spec_token_indx=torch.arange(3 * rows),
        non_spec_token_indx=torch.arange(3 * rows),
        num_accepted_tokens=torch.tensor([1, rows, 1], dtype=torch.int32),
        model_runner=NS(
            replayssm_write_pos=torch.tensor(
                [-1, -1, 0, 0, 0, 9, 0, 0], dtype=torch.int32
            )
        ),
    )
    meta = modules.gdn.GDNAttentionMetadataBuilder._build_gdn_capture_metadata(
        builder, 3
    )
    meta.non_spec_state_indices_tensor = conv[:, 0]
    meta.spec_state_indices_in_tensor = read
    meta.replayssm_fork_slots = torch.tensor([[5], [7]])
    cache = NS(
        k_cache=torch.zeros(8, rows + 2, 6 * hd, dtype=torch.bfloat16),
        v_cache=torch.zeros(8, 2, hd, hd),
        replay_buf_k=torch.zeros(8, 2, 16, hd, dtype=torch.bfloat16),
        replay_buf_u=torch.zeros(8, 2, 16, hd, dtype=torch.bfloat16),
        replay_buf_g=torch.zeros(8, 2, 16, hd),
        replay_flush_arrivals=torch.zeros(8, 2, dtype=torch.int32),
        replay_flush_wait="s_waitcnt vmcnt(0)",
    )
    ctx = NS(attn_metadata=NS(kda_metadata=meta), kv_cache_data={"layer_0": cache})
    monkeypatch.setattr(kimi, "get_forward_context", lambda: ctx)
    calls = NS(gemms=[], fused=[], conv=[], split=[], forks=[], outputs=[])

    def mm(x, w, *args, **kwargs):
        out = F.linear(x.to(w.dtype), w)
        calls.gemms.append((w, out))
        return out

    class Kernel:
        def __getitem__(self, grid):
            def launch(**kw):
                calls.fused.append((grid, kw))
                kw["o"].fill_(2)

            return launch

    def conv_update(x, *args, **kwargs):
        calls.conv.append(kwargs)
        return x.split(2 * hd, -1)

    def recurrence(*args, o, **kwargs):
        calls.split.append(kwargs)
        return o.fill_(3)

    def norm(out, gate, is_prefill):
        calls.outputs.append(out)
        return out

    monkeypatch.setattr(modules.linear.tgemm, "mm", mm, raising=False)
    monkeypatch.setattr(replay, "_replayssm_kda_fused_kernel", Kernel())
    monkeypatch.setattr(
        replay.triton,
        "next_power_of_2",
        lambda n: 1 << (n - 1).bit_length(),
        raising=False,
    )
    monkeypatch.setattr(
        replay, "seed_forked_slots", lambda s, f: calls.forks.append((s, f))
    )
    monkeypatch.setattr(kimi, "causal_conv1d_update", conv_update)
    monkeypatch.setattr(kimi, "causal_conv1d_fn", lambda x, *a, **k: conv_update(x.T))
    monkeypatch.setattr(kimi, "replayssm_sigmoid_gating_delta_rule", recurrence)
    monkeypatch.setattr(kimi, "fused_sigmoid_gating_delta_rule_update", recurrence)
    monkeypatch.setattr(attn.o_norm, "forward", norm)
    monkeypatch.setattr(attn.o_proj, "forward", lambda x: x)
    return NS(
        attn=attn,
        meta=meta,
        cache=cache,
        calls=calls,
        hidden=torch.ones(3 * rows, 16, dtype=torch.bfloat16),
        read=read,
        conv=conv,
    )


def test_conv_weight_rank_is_rejected_before_kernel_launch(modules, monkeypatch):
    c = layer_case(modules, monkeypatch)
    c.attn.conv_weight = c.attn.conv_weight.unsqueeze(1)
    assert c.attn.conv_weight.shape == (768, 1, 4)
    with pytest.raises(AssertionError, match="conv weight must be two-dimensional"):
        c.attn._forward_impl(c.hidden)
    assert not c.calls.fused and not c.calls.conv and not c.calls.split
    assert not c.calls.forks


@pytest.mark.parametrize("verify", [False, True], ids=["decode", "verify"])
def test_fallback_passes_pool_owned_flush_resources(modules, monkeypatch, verify):
    c = layer_case(modules, monkeypatch, verify=verify)
    monkeypatch.setattr(modules.kimi, "get_cu_num", lambda: 5)
    for _ in range(2):
        c.attn._forward_impl(c.hidden)
    assert len(c.calls.split) == 2 and not c.calls.fused
    for call in c.calls.split:
        assert call["flush_arrivals"] is c.cache.replay_flush_arrivals
        assert call["flush_wait"] == c.cache.replay_flush_wait


@pytest.mark.parametrize("verify", [False, True], ids=["decode", "verify"])
@pytest.mark.parametrize("glm", [False, True], ids=["kimi", "glm"])
def test_constructed_layer_routes_views_and_slots_to_real_api(
    modules, monkeypatch, verify, glm
):
    c = layer_case(modules, monkeypatch, verify, glm)
    cursor = c.meta.write_pos.clone()
    out = c.attn._forward_impl(c.hidden)
    assert len(c.calls.fused) == 1
    ((grid, kw),) = c.calls.fused
    assert grid == (6,) and not c.calls.conv and not c.calls.split
    assert len(c.calls.gemms) == 1
    projected = c.calls.gemms[0][1]
    for name, columns in (
        ("x", slice(0, 768)),
        ("b", slice(1024, 1026)),
        ("f_a", slice(1026, None)),
    ):
        view = kw[name].squeeze(0) if name == "b" else kw[name]
        assert (
            view.untyped_storage().data_ptr() == projected.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(view, projected[:, columns])
    for name, tensor in (
        ("ckpt", c.cache.v_cache),
        ("buf_k", c.cache.replay_buf_k),
        ("buf_u", c.cache.replay_buf_u),
        ("buf_g", c.cache.replay_buf_g),
        ("write_pos", c.meta.write_pos),
        ("f_b_weight", c.attn.f_b_proj.weight),
        ("conv_weight", c.attn.conv_weight),
    ):
        assert kw[name] is tensor
    torch.testing.assert_close(kw["slot_idx"], c.meta.slot_idx)
    cu = c.meta.spec_query_start_loc if verify else c.meta.non_spec_query_start_loc
    torch.testing.assert_close(kw["cu_seqlens"], cu)
    torch.testing.assert_close(kw["conv_idx"], c.conv[:, 0])
    torch.testing.assert_close(kw["conv_idx_in"], c.read[:, 0])
    assert kw["conv_idx"].stride(0) == 2
    assert kw["num_accepted"] is c.meta.num_accepted_tokens
    assert kw["LOWER_BOUND"] == -5 and kw["T_MAX"] == (8 if verify else 1)
    assert c.calls.forks[0][0] is c.cache.v_cache
    assert c.calls.forks[0][1] is c.meta.replayssm_fork_slots
    assert kw["o"] is c.calls.outputs[0]
    torch.testing.assert_close(out, torch.full_like(out, 2))
    c.attn._forward_impl(c.hidden)
    torch.testing.assert_close(c.meta.write_pos, cursor)


@pytest.mark.parametrize(
    "reason",
    [
        "wave",
        "arch",
        "classic",
        "plugin",
        "mixed",
        "head",
        "conv",
        "window",
        "state_len",
        "records",
        "layout",
        "input",
        "conv_dtype",
        "weight_dtype",
        "quant",
        "weight_stride",
        "bias",
        "activation",
    ],
)
def test_unsupported_steps_execute_existing_path(modules, monkeypatch, reason):
    c = layer_case(
        modules, monkeypatch, verify=True, hd=64 if reason == "head" else 128
    )
    if reason == "wave":
        monkeypatch.setattr(modules.kimi, "get_cu_num", lambda: 5)
    elif reason == "arch":
        monkeypatch.setattr(modules.kimi, "get_gfx_runtime", lambda: "gfx1250")
    elif reason in ("classic", "plugin"):
        c.meta.replayssm = False
        for name in ("replay_buf_k", "replay_buf_u", "replay_buf_g"):
            delattr(c.cache, name)
    elif reason == "mixed":
        c.meta.num_decodes = 1
        c.meta.non_spec_query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        c.meta.non_spec_state_indices_in_tensor = c.read[:, 0]
    elif reason == "conv":
        c.attn.conv_kernel_size = 3
    elif reason == "window":
        c.meta.replayssm_max_query_len = 9
    elif reason == "state_len":
        c.cache.k_cache = c.cache.k_cache[:, :3]
    elif reason == "records":
        c.cache.replay_buf_k = c.cache.replay_buf_k[:, :, :8]
    elif reason == "layout":
        c.cache.v_cache = c.cache.v_cache.transpose(-1, -2)
    elif reason == "input":
        c.attn.in_proj.weight.data = c.attn.in_proj.weight.data.float()
    elif reason == "conv_dtype":
        c.cache.k_cache = c.cache.k_cache.float()
    elif reason == "weight_dtype":
        c.attn.f_b_proj.weight.data = c.attn.f_b_proj.weight.data.float()
    elif reason == "quant":
        c.attn.f_b_proj.quant_type = modules.kimi.QuantType.per_Token
        monkeypatch.setattr(
            c.attn.f_b_proj, "forward", lambda x: F.linear(x, c.attn.f_b_proj.weight)
        )
    elif reason == "weight_stride":
        c.attn.f_b_proj.weight.data = c.attn.f_b_proj.weight.data.T.contiguous().T
    elif reason == "bias":
        c.attn.f_b_proj.bias = torch.nn.Parameter(torch.zeros(256))
    elif reason == "activation":
        c.attn.activation = "relu"
    out = c.attn._forward_impl(c.hidden)
    assert not c.calls.fused and len(c.calls.conv) == len(c.calls.split) == 1
    torch.testing.assert_close(out, torch.full_like(out, 3))


@pytest.mark.parametrize("rank", [0, 1])
def test_wave_fallback_logs_geometry_once(modules, monkeypatch, caplog, rank):
    c = layer_case(modules, monkeypatch, verify=True)
    c.attn.tp_size, c.attn.tp_rank = 2, rank
    monkeypatch.setattr(modules.kimi, "get_cu_num", lambda: 5)
    report = modules.kimi._log_kda_wave_fallback
    report.cache_clear()
    try:
        with caplog.at_level("INFO", logger="atom"):
            for _ in range(2):
                out = c.attn._forward_impl(c.hidden)
        assert not c.calls.fused and len(c.calls.split) == 2
        torch.testing.assert_close(out, torch.full_like(out, 3))
        messages = [
            r.getMessage() for r in caplog.records if "one-wave guard" in r.getMessage()
        ]
        if rank == 0:
            assert messages == [
                (
                    "Fused KDA uses split fallback: one-wave guard "
                    "(tp=2, padded_requests=3, local_heads=2, programs=6, CUs=5)."
                )
            ]
        else:
            assert not messages
    finally:
        report.cache_clear()


@pytest.mark.parametrize("prefill", [False, True])
def test_prefill_and_empty_steps_keep_existing_behavior(modules, monkeypatch, prefill):
    c = layer_case(modules, monkeypatch)
    c.meta.num_prefills, c.meta.num_decodes = int(prefill), 0
    monkeypatch.setattr(
        modules.ops, "gather_kda_initial_state", lambda s, *args: s[:3].clone()
    )
    monkeypatch.setattr(
        c.attn, "_run_kda", lambda q, k, v, g, b, s, *args: (torch.full_like(q, 3), s)
    )
    out = c.attn._forward_impl(c.hidden)
    assert not c.calls.fused and len(c.calls.conv) == int(prefill)
    torch.testing.assert_close(out, torch.full_like(out, 3 if prefill else 0))
