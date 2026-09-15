"""Real MI308 numerical tests against ATOM's original GDN pipeline."""

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("ROCm GPU required", allow_module_level=True)

from atom.model_ops.attention_gdn import fused_gdn_gating
from atom.model_ops.fla_ops import aiter_flydsl as fly
from atom.model_ops.fla_ops.chunk import (
    chunk_gated_delta_rule,
    pop_last_intermediate_states,
)
from atom.model_ops.fla_ops.fused_recurrent import fused_recurrent_gated_delta_rule


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
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Different memory layouts compile independently; near-zero state values
    # may differ by a BF16 rounding unit even when output tokens are identical.
    torch.testing.assert_close(vk, state, rtol=0.008, atol=2e-6)


def test_qwen_backend_binds_zero_copy_vk_state(monkeypatch):
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
    layer = SimpleNamespace(base_linear_attention=None)
    result = builder.build_kv_cache_tensor(layer)
    assert result.v_cache.data_ptr() == raw.data_ptr()
    assert result.v_cache.stride()[-2:] == (1, 128)
    monkeypatch.setattr(fly, "backend", lambda stage: "triton")
    assert builder.build_kv_cache_tensor(layer).v_cache is raw
