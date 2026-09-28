"""Real MI308 numerical tests against ATOM's original GDN pipeline."""

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("ROCm GPU required", allow_module_level=True)

from atom.model_ops.attention_gdn import fused_gdn_gating
from atom.model_ops.fla_ops import gdn_flydsl as fly
from atom.model_ops.fla_ops.chunk import (
    chunk_gated_delta_rule,
    pop_last_intermediate_states,
)
from atom.model_ops.fla_ops.fused_recurrent import fused_recurrent_gated_delta_rule

requires_gfx942 = pytest.mark.skipif(
    torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx942",
    reason="FlyDSL GDN decode is supported only on gfx942",
)


def inputs(tokens, hk=8, hv=24, seed=123):
    torch.manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    q, k, v = (
        rand(1, tokens, hk, 128),
        rand(1, tokens, hk, 128),
        rand(1, tokens, hv, 128),
    )
    # Production projection gates are non-contiguous views with batch > 1.
    ba = rand(tokens, 2 * hv)
    b, a = ba.split(hv, -1)
    log = torch.full((hv,), -2.0, device="cuda", dtype=torch.float32)
    bias = rand(hv)
    return q, k, v, a, b, log, bias


def close(actual, expected, name):
    assert torch.isfinite(actual).all(), name
    diff = (actual.float() - expected.float()).abs()
    rel = (
        diff.square().mean().sqrt()
        / expected.float().square().mean().sqrt().clamp_min(1e-8)
    )
    print(f"{name}: max_abs={diff.max().item():.8g}, relative_rms={rel.item():.8g}")
    assert rel < 0.015, (name, rel.item())
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.03, atol=0.015)


@pytest.mark.parametrize(
    "lengths", [(64,), (129,), (1024,), (8192,), (3808, 4352), (7648, 512)]
)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_prefill(lengths, state_dtype):
    q, k, v, a, b, log, bias = inputs(sum(lengths))
    g, beta = fused_gdn_gating(log, a, b, bias)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    state = (
        torch.randn(len(lengths), 24, 128, 128, device="cuda", dtype=state_dtype) * 0.1
    )
    metadata = fly.build_prefill_metadata(lengths, cu)
    assert fly.prefill_supported(q, k, v, g, beta, metadata)
    baseline, ht = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
        keep_intermediate_states=True,
    )
    h = pop_last_intermediate_states()
    result, result_ht, result_h = fly.prefill(
        q, k, v, g, beta, state, cu, metadata, True
    )
    close(result, baseline, f"prefill {lengths} {state_dtype} output")
    close(result_ht, ht, "final_state")
    close(result_h, h, "snapshots")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vk", [False, True])
@pytest.mark.parametrize("flags", [[False, False], [True, True], [True, False]])
def test_fused_prefill_state(dtype, vk, flags):
    pool = torch.randn(4, 24, 128, 128, device="cuda", dtype=dtype)
    if vk:
        pool = pool.transpose(-1, -2).contiguous().transpose(-1, -2)
    indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    live = torch.tensor(flags, device="cuda", dtype=torch.bool)
    for i, flag in zip([3, 1], flags):
        if not flag:
            pool[i].fill_(float("nan"))
    before = pool.clone()
    dense = pool[indices].contiguous()
    dense[~live] = 0
    expected = dense.transpose(-1, -2).float().contiguous()
    actual = fly.prepare_prefill_state(pool, indices, live)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pool, before, rtol=0, atol=0, equal_nan=True)
    assert actual.is_contiguous()
    q, k, v, a, b, log, bias = inputs(128)
    g, beta = fused_gdn_gating(log, a, b, bias)
    cu = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    meta = fly.build_prefill_metadata([64, 64], cu)
    ref, ref_ht, _ = fly.prefill(q, k, v, g, beta, dense, cu, meta)
    out, ht, _ = fly.prefill(
        q,
        k,
        v,
        g,
        beta,
        pool,
        cu,
        meta,
        state_indices=indices,
        has_initial_state=live,
    )
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(ht, ref_ht, rtol=0, atol=0)


def baseline_decode(q, k, v, a, b, state, log, bias, reads, writes):
    g, beta = fused_gdn_gating(log, a, b, bias)
    cu = torch.arange(q.shape[1] + 1, device=q.device, dtype=torch.int32)
    return fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=writes,
        ssm_state_indices_in=reads,
        use_qk_l2norm_in_kernel=True,
    )[0]


@pytest.mark.parametrize("batch", [1, 3, 4, 16, 64, 128])
@requires_gfx942
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("physical_vk", [True])
def test_decode(batch, state_dtype, physical_vk):
    q, k, v, a, b, log, bias = inputs(batch)
    state = torch.randn(batch * 2, 24, 128, 128, device="cuda", dtype=state_dtype) * 0.1
    ref = state.clone()
    if physical_vk:
        state = state.transpose(-1, -2).contiguous().transpose(-1, -2)
    reads = torch.arange(batch, device="cuda", dtype=torch.int32)
    writes = reads + batch
    assert fly.decode_supported(q, k, v, a, b, state, log, bias, reads, writes)
    expected = baseline_decode(q, k, v, a, b, ref, log, bias, reads, writes)
    out, _ = fly.decode(q, k, v, a, b, state, log, bias, reads, writes)
    close(out, expected, f"decode B={batch} {state_dtype}")
    close(state[batch:], ref[batch:], "state writeback")
    torch.testing.assert_close(state[:batch], ref[:batch], rtol=0, atol=0)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("physical_vk", [True])
@requires_gfx942
def test_decode_graph_mixed_padding(state_dtype, physical_vk):
    q, k, v, a, b, log, bias = inputs(4)
    original = torch.randn(8, 24, 128, 128, device="cuda", dtype=state_dtype) * 0.1
    state = original.clone()
    if physical_vk:
        state = state.transpose(-1, -2).contiguous().transpose(-1, -2)
    reads = torch.tensor([0, 1, 2, -1], device="cuda", dtype=torch.int32)
    writes = torch.tensor([4, 5, 6, -1], device="cuda", dtype=torch.int32)
    for _ in range(3):
        fly.decode(q, k, v, a, b, state, log, bias, reads, writes)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result, _ = fly.decode(q, k, v, a, b, state, log, bias, reads, writes)
    for live in (3, 2, 1, 3):
        ri = list(range(live)) + [-1] * (4 - live)
        wi = list(range(4, 4 + live)) + [-1] * (4 - live)
        reads.copy_(torch.tensor(ri, device="cuda", dtype=torch.int32))
        writes.copy_(torch.tensor(wi, device="cuda", dtype=torch.int32))
        state.copy_(original)
        graph.replay()
        reference = original.clone()
        expected = baseline_decode(
            q[:, :live].contiguous(),
            k[:, :live].contiguous(),
            v[:, :live].contiguous(),
            a[:live],
            b[:live],
            reference,
            log,
            bias,
            reads[:live],
            writes[:live],
        )
        close(result[:, :live], expected, f"graph live={live}")
        assert torch.count_nonzero(result[:, live:]) == 0
        close(state, reference, "graph state")


def test_vk_checkpoint_layout():
    from atom.model_ops.fla_ops.state_checkpoint import write_state_checkpoints

    torch.manual_seed(42)

    def index(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    pool = torch.randn(8, 24, 128, 128, device="cuda", dtype=torch.float32)
    vk = pool.transpose(-1, -2).contiguous().transpose(-1, -2)
    h = torch.randn(1, 4, 24, 128, 128, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(256, 4, device="cuda", dtype=torch.bfloat16)
    conv = torch.zeros(8, 4, 3, device="cuda", dtype=torch.bfloat16)
    conv_vk = conv.clone()
    args = [
        index(x)
        for x in ([0, 1], [4, 5], [64, 128], [0, 1], [0, 1], [0, 2, 4], [0, 128, 256])
    ]
    write_state_checkpoints(h, pool, x, conv, *args, 64)
    write_state_checkpoints(h, vk, x, conv_vk, *args, 64)
    torch.testing.assert_close(vk, pool, rtol=0, atol=0)
    torch.testing.assert_close(conv_vk, conv, rtol=0, atol=0)


def test_triton_decode_vk_fallback():
    q, k, v, a, b, log, bias = inputs(3)
    state = torch.randn(6, 24, 128, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    vk = state.transpose(-1, -2).contiguous().transpose(-1, -2)
    reads = torch.arange(3, device="cuda", dtype=torch.int32)
    writes = reads + 3
    expected = baseline_decode(q, k, v, a, b, state, log, bias, reads, writes)
    actual = baseline_decode(q, k, v, a, b, vk, log, bias, reads, writes)
    # Different memory layouts compile independently, so near-zero values may
    # differ by a BF16 rounding unit. That goes for the output as well as the
    # state: on gfx950 two of 9216 output elements land 1.5e-5 apart, at a
    # magnitude of 2e-3. A fallback reading the WRONG values is not what this
    # admits -- it would be wrong by its own magnitude, not by an ulp.
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=2e-6)
    torch.testing.assert_close(vk, state, rtol=0.008, atol=2e-6)


@pytest.mark.parametrize("replayssm", [False, True])
@pytest.mark.parametrize("supported_arch", [False, True])
@pytest.mark.parametrize("decode_backend", ["auto", "triton", "flydsl"])
@pytest.mark.parametrize("allowed", [False, True])
def test_qwen_backend_binds_zero_copy_vk_state(
    monkeypatch, replayssm, supported_arch, decode_backend, allowed
):
    from types import SimpleNamespace

    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder

    raw = torch.zeros(4, 24, 128, 128, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(
        GDNAttentionMetadataBuilder,
        "build_kv_cache_tensor",
        lambda self, module: SimpleNamespace(v_cache=raw),
    )
    builder = object.__new__(Qwen4ExpMetadataBuilder)
    builder.replayssm = replayssm
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_ENABLE_GDN_DECODE_LOSSY_FAST", False)
    monkeypatch.setattr(fly, "backend", lambda stage: decode_backend)
    monkeypatch.setattr(fly, "ops", lambda: object())
    monkeypatch.setattr(fly, "decode_device_supported", lambda device: supported_arch)
    attention = SimpleNamespace(allow_aiter_flydsl=allowed, dt_bias=raw)
    layer = SimpleNamespace(base_linear_attention=None, impl=attention)
    result = builder.build_kv_cache_tensor(layer)
    assert result.v_cache.data_ptr() == raw.data_ptr()
    expected = (
        allowed and not replayssm and supported_arch and decode_backend != "triton"
    )
    assert attention.gdn_flydsl_policy.decode == expected
    assert result.v_cache.stride()[-2:] == ((1, 128) if expected else (128, 1))
    if replayssm:
        assert not attention.gdn_flydsl_policy.prefill
        assert not builder._flydsl_prefill_enabled
        assert result.v_cache is raw


def test_unsupported_device_decode_fallback(monkeypatch):
    from types import SimpleNamespace

    q, k, v, a, b, log, bias = inputs(3)
    state = torch.randn(6, 24, 128, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    vk = state.transpose(-1, -2).contiguous().transpose(-1, -2)
    reads = torch.arange(3, device="cuda", dtype=torch.int32)
    writes = reads + 3
    monkeypatch.setattr(fly, "backend", lambda stage: "auto")
    monkeypatch.setattr(fly, "ops", lambda: object())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx950"),
    )
    assert not fly.decode_supported(q, k, v, a, b, vk, log, bias, reads, writes)
    with pytest.raises(ValueError, match="Unsupported"):
        fly.decode(q, k, v, a, b, vk, log, bias, reads, writes)
    # The fallback consumes the same logical KV values; no FlyDSL launch occurs.
    monkeypatch.undo()
    expected = baseline_decode(q, k, v, a, b, state, log, bias, reads, writes)
    actual = baseline_decode(q, k, v, a, b, vk, log, bias, reads, writes)
    # Same BF16 rounding-unit allowance as `test_triton_decode_vk_fallback`,
    # for the same reason: the two layouts compile independently.
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=2e-6)
    torch.testing.assert_close(vk, state, rtol=0.008, atol=2e-6)


@pytest.mark.parametrize("missing_ops", [False, True])
@pytest.mark.parametrize("lossy_decode", [False, True])
def test_policy_dependencies_and_lossy(monkeypatch, missing_ops, lossy_decode):
    monkeypatch.setattr(fly, "backend", lambda stage: "auto")
    monkeypatch.setattr(fly, "ops", lambda: None if missing_ops else object())
    monkeypatch.setattr(fly, "decode_device_supported", lambda device: True)
    policy = fly.select_policy(
        allowed=True,
        replayssm=False,
        lossy_decode=lossy_decode,
        state=torch.empty(1, 24, 128, 128, device="cuda", dtype=torch.bfloat16),
        activation_dtype=torch.bfloat16,
    )
    assert policy.prefill == (not missing_ops)
    assert policy.decode == (not missing_ops and not lossy_decode)


@pytest.mark.parametrize("mode", ["replayssm", "unsupported", "disabled", "flydsl"])
def test_cache_policy_forward_dispatch(monkeypatch, mode):
    """Exercise the cache-builder -> real GDN forward contract, not just gates."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from atom.model_ops import attention_gdn as gdn
    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder
    from atom.model_ops.fla_ops.replayssm import replayssm_buffer_shapes
    from atom.utils import envs

    if mode == "flydsl" and not fly.decode_device_supported(torch.device("cuda")):
        pytest.skip("FlyDSL decode numerical coverage requires gfx942")
    q, k, v, a, b, log, bias = inputs(3)
    raw = torch.randn(3, 24, 128, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    reference = raw.clone()
    idx = torch.arange(3, device="cuda", dtype=torch.int32)
    cu = torch.arange(4, device="cuda", dtype=torch.int32)
    sk, su, sg = replayssm_buffer_shapes(4, 24, 128, 128, False)
    cache = SimpleNamespace(
        v_cache=raw,
        k_cache=torch.zeros(3, 1, 1, device="cuda", dtype=torch.bfloat16),
        replay_buf_k=torch.zeros((3, *sk), device="cuda", dtype=torch.bfloat16),
        replay_buf_u=torch.zeros((3, *su), device="cuda", dtype=torch.bfloat16),
        replay_buf_g=torch.zeros((3, *sg), device="cuda", dtype=torch.float32),
    )
    attention = gdn.GatedDeltaNet.__new__(gdn.GatedDeltaNet)
    torch.nn.Module.__init__(attention)
    for name, value in {
        "layer_num": 0,
        "tp_size": 1,
        "num_k_heads": 8,
        "num_v_heads": 24,
        "head_k_dim": 128,
        "head_v_dim": 128,
        "A_log": log,
        "dt_bias": bias,
        "allow_aiter_flydsl": True,
        "activation": "silu",
        "conv1d": SimpleNamespace(weight=torch.zeros(1, 1, 1), bias=None),
    }.items():
        setattr(attention, name, value)
    builder = object.__new__(Qwen4ExpMetadataBuilder)
    builder.replayssm = mode == "replayssm"
    monkeypatch.setattr(envs, "ATOM_ENABLE_GDN_DECODE_LOSSY_FAST", False)
    monkeypatch.setattr(
        fly, "backend", lambda stage: "triton" if mode == "disabled" else "auto"
    )
    if mode == "unsupported":
        monkeypatch.setattr(fly, "decode_device_supported", lambda device: False)
    monkeypatch.setattr(
        GDNAttentionMetadataBuilder, "build_kv_cache_tensor", lambda self, module: cache
    )
    cache = builder.build_kv_cache_tensor(
        SimpleNamespace(base_linear_attention=None, impl=attention)
    )
    # Production binds a zero-filled pool before prefill populates logical KV
    # states. Seed logical values after binding, not physical pre-bind storage.
    cache.v_cache.copy_(reference)
    metadata = SimpleNamespace(
        replayssm=builder.replayssm,
        has_initial_state=None,
        spec_query_start_loc=None,
        non_spec_query_start_loc=cu,
        spec_sequence_masks=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=idx,
        non_spec_state_indices_in_tensor=idx,
        num_actual_tokens=3,
        num_accepted_tokens=None,
        num_prefills=0,
        num_decodes=3,
        write_pos=torch.zeros(3, device="cuda", dtype=torch.int32),
        slot_idx=idx,
        replayssm_max_query_len=1,
        replayssm_route="serial",
    )
    context = SimpleNamespace(
        attn_metadata=SimpleNamespace(gdn_metadata=metadata),
        kv_cache_data={"layer_0": cache},
    )
    monkeypatch.setattr(gdn, "get_forward_context", lambda: context)
    # Isolate GDN from the unrelated convolution, preserving real q/k/v values.
    monkeypatch.setattr(
        gdn, "causal_conv1d_update", lambda *args, **kwargs: (q[0], k[0], v[0])
    )
    recurrent = Mock(wraps=gdn.fused_recurrent_gated_delta_rule)
    replay = Mock(wraps=gdn.replayssm_gated_delta_rule)
    decode = Mock(wraps=fly.decode)
    monkeypatch.setattr(gdn, "fused_recurrent_gated_delta_rule", recurrent)
    monkeypatch.setattr(gdn, "replayssm_gated_delta_rule", replay)
    monkeypatch.setattr(fly, "decode", decode)
    actual = attention(
        torch.empty(3, 1, device="cuda"), b, a, torch.empty_like(v[0]), "layer_0"
    )
    expected = baseline_decode(q, k, v, a, b, reference, log, bias, idx, idx)
    close(actual, expected[0], mode)
    assert decode.call_count == int(mode == "flydsl")
    assert replay.call_count == int(mode == "replayssm")
    assert recurrent.call_count == int(mode in ("unsupported", "disabled"))
    if mode == "replayssm":
        assert cache.v_cache.is_contiguous()
    else:
        close(cache.v_cache, reference, "forward state")
