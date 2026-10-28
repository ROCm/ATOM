"""AITER fused GDN verify step (gfx1250, ReplaySSM) against float64 and against
the state that ATOM's own verify chain leaves."""

import types

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("ROCm GPU required", allow_module_level=True)
if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx1250":
    pytest.skip("AITER fused GDN verify is gfx1250-only", allow_module_level=True)
verify_op = pytest.importorskip(
    "aiter.ops.triton.gated_delta_net.fused_gdn_verify_qkvz"
)

from atom.model_ops.attention_gdn import fused_gdn_gating
from atom.model_ops.fla_ops.replayssm import (
    replayssm_commit,
    replayssm_gated_delta_rule,
)
from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_update

K, V, T, CAP = 128, 128, 4, 16  # MTP-3: a window of 4 tokens
WIDTH = T + 2  # the speculative conv window: 3 inputs of history + T - 1
SLOTS = 80
EPS = 1e-6
# (key heads, value heads) of Qwen3.5-35B-A3B at TP4, and at TP1 / DP attention.
SHAPES = [(4, 8), (16, 32)]


def make_layer(kh, vh, seed):
    torch.manual_seed(seed)
    bf16 = {"dtype": torch.bfloat16, "device": "cuda"}
    qkv = 2 * kh * K + vh * V
    return types.SimpleNamespace(
        kh=kh,
        vh=vh,
        qkv=qkv,
        conv_weight=torch.randn(qkv, 4, **bf16) * 0.3,
        A_log=torch.randn(vh, device="cuda"),
        dt_bias=torch.randn(vh, **bf16),
        norm=types.SimpleNamespace(weight=torch.randn(V, **bf16), eps=EPS),
    )


def make_pools(layer, seed):
    """ReplaySSM pools as the cache builder binds them: the conv window stored
    [slot, WIDTH, channels] and passed transposed; K-major checkpoints."""
    torch.manual_seed(seed)
    bf16 = {"dtype": torch.bfloat16, "device": "cuda"}
    vh = layer.vh
    record_k = torch.nn.functional.normalize(
        torch.randn(SLOTS, vh, CAP, K, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    return {
        "conv": torch.randn(SLOTS, WIDTH, layer.qkv, **bf16),
        "ckpt": torch.randn(SLOTS, vh, K, V, **bf16) * 0.1,
        "rk": record_k,
        "ru": torch.randn(SLOTS, vh, CAP, V, **bf16) * 0.1,
        "rg": -torch.rand(SLOTS, vh, CAP, device="cuda") * 0.3,
        "wp": torch.randint(0, CAP - T + 1, (SLOTS,), dtype=torch.int32, device="cuda"),
    }


def make_step(layer, seqs, seed, slots=None):
    torch.manual_seed(seed)
    bf16 = {"dtype": torch.bfloat16, "device": "cuda"}
    n, vh = seqs * T, layer.vh
    # One projection output [q|k|v|z|b|a] with padded rows, as Qwen3.5 produces it.
    qkvzba = torch.randn(n, layer.qkv + vh * V + 2 * vh + 8, **bf16)
    mixed_qkv, z, b, a = qkvzba[:, : layer.qkv + vh * V + 2 * vh].split(
        [layer.qkv, vh * V, vh, vh], -1
    )
    if slots is None:
        slots = torch.randperm(SLOTS, device="cuda")[:seqs].to(torch.int32)
    return types.SimpleNamespace(
        n=n,
        mixed_qkv=mixed_qkv,
        z=z.view(n, vh, V),
        b=b,
        a=a,
        slots=slots,
        accepted=torch.randint(1, T + 1, (seqs,), dtype=torch.int32, device="cuda"),
        cu=torch.arange(0, n + 1, T, dtype=torch.int32, device="cuda"),
    )


def gated_rmsnorm_reference(core, z, norm):
    c, zf = core.float(), z.float()
    rms = torch.rsqrt(c.square().mean(-1, keepdim=True) + norm.eps)
    return c * rms * norm.weight.float() * zf * torch.sigmoid(zf)


def unfused_verify(layer, step, pools):
    """causal_conv1d_update -> fused_gdn_gating -> replayssm -> gated RMSNorm."""
    q, k, v = causal_conv1d_update(
        step.mixed_qkv,
        pools["conv"].transpose(-1, -2),
        layer.conv_weight,
        layer.kh * K,
        layer.vh * V,
        None,
        "silu",
        conv_state_indices=step.slots,
        num_accepted_tokens=step.accepted,
        query_start_loc=step.cu,
        max_query_len=T,
        validate_data=False,
    )
    g, beta = fused_gdn_gating(layer.A_log, step.a, step.b, layer.dt_bias)
    n = step.n
    core = replayssm_gated_delta_rule(
        q=q.view(1, n, -1, K),
        k=k.view(1, n, -1, K),
        v=v.view(1, n, -1, V),
        g=g,
        beta=beta,
        ckpt=pools["ckpt"],
        buf_k=pools["rk"],
        buf_u=pools["ru"],
        buf_g=pools["rg"],
        write_pos=pools["wp"],
        slot_idx=step.slots,
        cu_seqlens=step.cu,
        max_query_len=T,
        use_qk_l2norm_in_kernel=True,
        route="auto",
    )
    return gated_rmsnorm_reference(core.squeeze(0), step.z, layer.norm)


def fused_args(layer, step, pools, out):
    n, vh = step.n, layer.vh
    row = step.mixed_qkv.stride(0)
    return {
        "projected_qkvz": step.mixed_qkv.as_strided((n, layer.qkv + vh * V), (row, 1)),
        "projected_ba": step.b.as_strided((n, 2 * vh), (step.b.stride(0), 1)),
        "conv_state": pools["conv"].transpose(-1, -2),
        "conv_state_indices": step.slots,
        "num_accepted_tokens": step.accepted,
        "query_start_loc": step.cu,
        "checkpoint": pools["ckpt"],
        "record_k": pools["rk"],
        "record_u": pools["ru"],
        "record_g": pools["rg"],
        "write_pos": pools["wp"],
        "slot_idx": step.slots,
        "conv_weight": layer.conv_weight,
        "conv_bias": None,
        "A_log": layer.A_log,
        "dt_bias": layer.dt_bias,
        "norm_weight": layer.norm.weight,
        "out": out,
    }


def fused_verify(layer, step, pools):
    out = torch.empty(step.n, layer.vh, V, dtype=torch.bfloat16, device="cuda")
    args = fused_args(layer, step, pools, out)
    verify_op.fused_gdn_verify_qkvz(
        max_query_len=T, scale=K**-0.5, norm_eps=EPS, **args
    )
    return out.float()


def commit(pools, step):
    # Once per forward, before any layer, as GDNAttentionMetadataBuilder does.
    replayssm_commit(pools["wp"], step.slots, step.accepted, T, CAP)


def reference64(layer, step, pools):
    """The verify step in float64, from the pools as they are (after the commit),
    with the bf16 roundings of the production chain: conv outputs, beta and the
    readout before the norm. Returns the output and the pools the step must leave;
    changes nothing."""
    kh, vh, n = layer.kh, layer.vh, step.n
    seqs = step.slots.numel()
    slots = step.slots.long()
    expected = {name: x.clone() for name, x in pools.items()}
    # Conv: the three inputs before the window start at accepted - 1, and the new
    # window is [h1, h2, x_0 .. x_{T-1}].
    position = (step.accepted.long() - 1)[:, None] + torch.arange(3, device="cuda")
    history = pools["conv"][slots[:, None], position]  # [seqs, 3, C]
    inputs = step.mixed_qkv.reshape(seqs, T, -1)
    expected["conv"][slots] = torch.cat([history[:, 1:], inputs], 1)
    window = torch.cat([history, inputs], 1).double()
    taps = layer.conv_weight.double().t()  # [4, C]
    conv = torch.stack([(window[:, i : i + 4] * taps).sum(1) for i in range(T)], 1)
    conv = torch.nn.functional.silu(conv).to(torch.bfloat16).double().view(n, -1)
    q, k, v = conv.split([kh * K, kh * K, vh * V], -1)
    ratio = vh // kh
    q = q.view(n, kh, K).repeat_interleave(ratio, 1)
    k = k.view(n, kh, K).repeat_interleave(ratio, 1)
    v = v.view(n, vh, V)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * K**-0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    g = -torch.exp(layer.A_log.double()) * torch.nn.functional.softplus(
        step.a.double() + layer.dt_bias.double()
    )
    beta = torch.sigmoid(step.b.double()).to(torch.bfloat16).double()
    # Rebuild from the checkpoint and the committed records.
    state = pools["ckpt"][slots].double()  # [seqs, vh, K, V]
    committed = pools["wp"][slots].long().clamp_min(0)
    rk, ru, rg = (pools[name][slots].double() for name in ("rk", "ru", "rg"))
    for j in range(CAP):
        keep = (j < committed).view(seqs, 1, 1, 1)
        replayed = (
            torch.exp(rg[:, :, j, None, None]) * state
            + rk[:, :, j, :, None] * ru[:, :, j, None, :]
        )
        state = torch.where(keep, replayed, state)
    flush = committed + 2 * T > CAP
    expected["ckpt"][slots[flush]] = state[flush].to(pools["ckpt"].dtype)
    first = torch.where(flush, 0, committed)
    out = torch.empty(seqs, T, vh, V, dtype=torch.float64, device="cuda")
    q, k, v = (x.view(seqs, T, vh, -1) for x in (q, k, v))
    g, beta = g.view(seqs, T, vh), beta.view(seqs, T, vh)
    for i in range(T):
        state = torch.exp(g[:, i, :, None, None]) * state
        predicted = (state * k[:, i, :, :, None]).sum(2)
        u = beta[:, i, :, None] * (v[:, i] - predicted)
        state = state + k[:, i, :, :, None] * u[:, :, None, :]
        out[:, i] = (state * q[:, i, :, :, None]).sum(2)
        record = (slots, slice(None), first + i)
        expected["rk"][record] = k[:, i].to(pools["rk"].dtype)
        expected["ru"][record] = u.to(pools["ru"].dtype)
        expected["rg"][record] = g[:, i].to(pools["rg"].dtype)
    core = out.view(n, vh, V).to(torch.bfloat16).double()
    z = step.z.double()
    rms = torch.rsqrt((core * core).mean(-1, keepdim=True) + layer.norm.eps)
    return core * rms * layer.norm.weight.double() * z * torch.sigmoid(z), expected


def check_output(actual, truth):
    """Against float64: the unfused chain itself is off by up to 0.25 (0.13 of the
    value) on these inputs, from its bf16 record replay; the fused kernel stays near
    1/3 of that mean error."""
    err = (actual.double() - truth).abs()
    relative = (err / truth.abs().clamp_min(1.0)).max().item()
    assert relative < 2e-2, f"max error {relative:.4f} of max(|x|, 1)"
    assert err.mean().item() < 1e-3, f"mean error {err.mean().item():.5f}"


def check_pools(got, expected, unfused=None):
    """The pools against the float64 step: conv windows and cursors exact, the rest
    within one bf16 rounding step. The unfused chain is only compared where it is
    deterministic: replayssm splits a head's columns over two programs, and on a
    flush step the first one writes k and g records from position 0 while the
    second can still be reading the old ones, so its checkpoint and u records are
    occasionally wrong."""
    assert torch.equal(got["conv"], expected["conv"]), "conv windows differ"
    assert torch.equal(got["wp"], expected["wp"])
    for name, tol in (("ckpt", 4e-3), ("rk", 8e-3), ("ru", 3.2e-2), ("rg", 1e-6)):
        torch.testing.assert_close(
            got[name].float(), expected[name].float(), atol=tol, rtol=tol
        )
    if unfused is not None:
        assert torch.equal(unfused["conv"], got["conv"])
        for name, tol in (("rk", 8e-3), ("rg", 1e-6)):
            torch.testing.assert_close(
                got[name].float(), unfused[name].float(), atol=tol, rtol=tol
            )


@pytest.mark.parametrize("kh,vh", SHAPES)
@pytest.mark.parametrize("seqs", [1, 3, 16, 64])
def test_one_step(kh, vh, seqs):
    layer = make_layer(kh, vh, seed=seqs)
    ref = make_pools(layer, seed=100 + seqs)
    got = {name: t.clone() for name, t in ref.items()}
    step = make_step(layer, seqs, seed=200 + seqs)
    commit(ref, step)
    commit(got, step)
    truth, expected = reference64(layer, step, got)
    unfused_verify(layer, step, ref)
    check_output(fused_verify(layer, step, got), truth)
    check_pools(got, expected, unfused=ref)


@pytest.mark.parametrize("kh,vh", SHAPES)
def test_flush_boundaries(kh, vh):
    """Cursors on both sides of the flush rule (h + 2T > CAP), and every count of
    accepted tokens, in one batch."""
    layer = make_layer(kh, vh, seed=7)
    ref = make_pools(layer, seed=8)
    seqs = 4 * (CAP - T + 1)
    step = make_step(layer, seqs, seed=9)
    cursor = torch.arange(CAP - T + 1, device="cuda").repeat_interleave(4)
    ref["wp"][step.slots.long()] = cursor.to(torch.int32)
    step.accepted = (
        torch.arange(1, T + 1, device="cuda").repeat(CAP - T + 1).to(torch.int32)
    )
    got = {name: t.clone() for name, t in ref.items()}
    commit(ref, step)
    commit(got, step)
    truth, expected = reference64(layer, step, got)
    unfused_verify(layer, step, ref)
    check_output(fused_verify(layer, step, got), truth)
    check_pools(got, expected, unfused=ref)


@pytest.mark.parametrize("kh,vh", SHAPES)
def test_many_steps(kh, vh):
    """Twelve verify steps on the same sequences with random acceptance: the cursor
    walks through several flushes. Each step is checked against float64 from the
    fused path's own pools, so error cannot pile up unseen."""
    layer = make_layer(kh, vh, seed=11)
    pools = make_pools(layer, seed=12)
    slots = torch.randperm(SLOTS, device="cuda")[:8].to(torch.int32)
    flushes = 0
    for i in range(12):
        step = make_step(layer, 8, seed=300 + i, slots=slots)
        commit(pools, step)
        flushes += (pools["wp"][slots.long()] + 2 * T > CAP).sum().item()
        truth, expected = reference64(layer, step, pools)
        check_output(fused_verify(layer, step, pools), truth)
        check_pools(pools, expected)
    assert flushes > 8, "the steps must cross the flush rule"


def test_padded_sequences_are_skipped():
    layer = make_layer(4, 8, seed=21)
    pools = make_pools(layer, seed=22)
    step = make_step(layer, 3, seed=23)
    step.slots[1] = -1
    step.cu = torch.tensor([0, 4, 4, 8], dtype=torch.int32, device="cuda")
    before = {name: t.clone() for name, t in pools.items()}
    out = torch.full((step.n, layer.vh, V), 7.0, dtype=torch.bfloat16, device="cuda")
    args = fused_args(layer, step, pools, out)
    verify_op.fused_gdn_verify_qkvz(
        max_query_len=T, scale=K**-0.5, norm_eps=EPS, **args
    )
    # Rows 4..7 belong to the next sequence; the padded one has no rows.
    assert torch.all(out[8:] == 7.0)
    untouched = [i for i in range(SLOTS) if i not in step.slots.tolist()]
    for name in ("conv", "ckpt", "rk", "ru", "rg"):
        assert torch.equal(pools[name][untouched], before[name][untouched]), name
