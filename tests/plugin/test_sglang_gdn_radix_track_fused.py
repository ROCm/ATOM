"""One-launch radix-track extend snapshots (`write_radix_track_snapshots`).

The fused kernel replaces the plugin's conv-window `index_put` and SGLang's
`MambaAttnBackendBase._track_mamba_state_extend` on device. These tests run
both on the same inputs and require bit-identical pools, across the pool views
the plugin actually passes (transposed conv pool, VK SSM view, a column slice
of the fused projection), and check that it is a single kernel launch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.fla_ops import chunk as chunk_mod
from atom.plugin.sglang.attention_backend.attention_gdn import (
    SGLangGatedDeltaNet,
    SGLangGDNForwardContext,
)
from atom.plugin.sglang.attention_backend.gdn_radix_track import (
    fused_track_supported,
    write_radix_track_snapshots,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

H, K, V = 2, 128, 128
D = 1100  # not a multiple of the 1024 block
STATE_LEN = 3
NUM_SLOTS = 12
NT = 9
T = 40


def _sglang_track_extend():
    mod = pytest.importorskip("sglang.srt.layers.attention.hybrid_linear_attn_backend")
    return mod.MambaAttnBackendBase._track_mamba_state_extend


def _pools(conv_transposed: bool, ssm_vk: bool, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(0)
    if conv_transposed:
        conv = torch.randn(
            NUM_SLOTS, STATE_LEN, D, device=dev, generator=g, dtype=torch.bfloat16
        )
        conv = conv.transpose(1, 2)
    else:
        conv = torch.randn(
            NUM_SLOTS, D, STATE_LEN, device=dev, generator=g, dtype=torch.bfloat16
        )
    if ssm_vk:
        ssm = torch.randn(NUM_SLOTS, H, V, K, device=dev, generator=g).transpose(-1, -2)
    else:
        ssm = torch.randn(NUM_SLOTS, H, K, V, device=dev, generator=g)
    return conv, ssm


def _inputs(dev="cuda", h_vk=False):
    g = torch.Generator(device=dev).manual_seed(1)
    # Column slice of a wider fused projection: row stride != D.
    fused = torch.randn(T, D + 300, device=dev, generator=g, dtype=torch.bfloat16)
    x = fused[:, 100 : 100 + D]
    if h_vk:
        # FlyDSL's chunk states: the logical [K, V] view of VK storage.
        h = torch.randn(1, NT, H, V, K, device=dev, generator=g, dtype=torch.bfloat16)
        return x, h.transpose(-1, -2)
    h = torch.randn(1, NT, H, K, V, device=dev, generator=g, dtype=torch.bfloat16)
    return x, h


def _metadata(n_conv: int, n_h: int, n_f: int, dev="cuda"):
    # Window starts include a clamped one (all zeros), as SGLang's
    # `_init_track_conv_indices` produces for a short aligned length.
    starts = torch.tensor([5, 20, -2, 33][:n_conv], device=dev)
    conv_idx = (starts[:, None] + torch.arange(STATE_LEN, device=dev)).clamp(0, T - 1)
    md = SimpleNamespace(
        has_mamba_track_mask=True,
        track_conv_indices=conv_idx,
        conv_states_mask_indices=torch.tensor([7, 9, 10, 11][:n_conv], device=dev),
        track_ssm_h_src=torch.tensor([2, 0, 8][:n_h], device=dev, dtype=torch.int64),
        track_ssm_h_dst=torch.tensor([7, 10, 11][:n_h], device=dev, dtype=torch.int64),
        track_ssm_final_src=torch.tensor([1, 3][:n_f], device=dev, dtype=torch.int64),
        track_ssm_final_dst=torch.tensor([9, 8][:n_f], device=dev, dtype=torch.int64),
        track_ssm_recompute_dst=None,
    )
    return md


def _reference(x, h, conv, ssm, md):
    if md.track_conv_indices.numel() > 0:
        conv[md.conv_states_mask_indices] = x[md.track_conv_indices].transpose(1, 2)
    _sglang_track_extend()(None, None, h, ssm, md)


def _fused(x, h, conv, ssm, md):
    conv_idx = md.track_conv_indices if md.track_conv_indices.numel() > 0 else None
    write_radix_track_snapshots(
        x,
        conv_idx,
        md.conv_states_mask_indices if conv_idx is not None else None,
        conv,
        h,
        md.track_ssm_h_src,
        md.track_ssm_h_dst,
        md.track_ssm_final_src,
        md.track_ssm_final_dst,
        ssm,
    )


@pytest.mark.parametrize("conv_transposed", [False, True])
@pytest.mark.parametrize("ssm_vk", [False, True])
@pytest.mark.parametrize("h_vk", [False, True])
@pytest.mark.parametrize(
    "counts", [(4, 3, 2), (4, 0, 2), (4, 3, 0), (0, 3, 2), (2, 1, 0), (0, 0, 1)]
)
def test_fused_matches_eager_bit_for_bit(conv_transposed, ssm_vk, h_vk, counts):
    x, h = _inputs(h_vk=h_vk)
    md = _metadata(*counts)
    conv_ref, ssm_ref = _pools(conv_transposed, ssm_vk)
    conv_new, ssm_new = _pools(conv_transposed, ssm_vk)
    _reference(x, h, conv_ref, ssm_ref, md)
    _fused(x, h, conv_new, ssm_new, md)
    torch.cuda.synchronize()
    # Compare whole pools so stray writes to untouched slots are caught too.
    assert torch.equal(conv_new, conv_ref)
    assert torch.equal(ssm_new, ssm_ref)


@pytest.mark.parametrize("ssm_vk", [False, True])
@pytest.mark.parametrize("h_vk", [False, True])
def test_fused_keeps_k_and_v_apart(ssm_vk, h_vk):
    """Unequal K and V catch an interchanged axis on either side."""
    k, v = 64, 128
    g = torch.Generator(device="cuda").manual_seed(2)
    shape = (1, NT, H, v, k) if h_vk else (1, NT, H, k, v)
    h = torch.randn(*shape, device="cuda", generator=g, dtype=torch.bfloat16)
    if h_vk:
        h = h.transpose(-1, -2)
    if ssm_vk:
        ssm = torch.randn(NUM_SLOTS, H, v, k, device="cuda", generator=g)
        ssm = ssm.transpose(-1, -2)
    else:
        ssm = torch.randn(NUM_SLOTS, H, k, v, device="cuda", generator=g)
    ssm_ref = ssm.clone()
    x, _ = _inputs()
    conv, _ = _pools(False, False)
    md = _metadata(0, 3, 2)
    _reference(x, h, conv, ssm_ref, md)
    _fused(x, h, conv, ssm, md)
    assert torch.equal(ssm, ssm_ref)


def test_nothing_to_track_launches_nothing():
    x, h = _inputs()
    md = _metadata(0, 0, 0)
    conv, ssm = _pools(False, False)
    conv0, ssm0 = conv.clone(), ssm.clone()
    _fused(x, h, conv, ssm, md)
    assert torch.equal(conv, conv0) and torch.equal(ssm, ssm0)


def _kernel_count(fn) -> int:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(
        1
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and "Memcpy" not in e.name
        and "Memset" not in e.name
    )


@pytest.mark.parametrize("h_vk", [False, True])
def test_one_launch_replaces_the_eager_chain(h_vk):
    x, h = _inputs(h_vk=h_vk)
    md = _metadata(4, 3, 2)
    conv, ssm = _pools(True, True)
    _fused(x, h, conv, ssm, md)  # compile outside the profiled window
    _reference(x, h, conv, ssm, md)
    fused = _kernel_count(lambda: _fused(x, h, conv, ssm, md))
    eager = _kernel_count(lambda: _reference(x, h, conv, ssm, md))
    assert fused == 1
    assert eager >= 5


def test_fused_track_supported_gates_non_gdn_paths():
    _, h = _inputs()
    md = _metadata(1, 1, 1)
    assert fused_track_supported(md, h)
    md.track_ssm_recompute_dst = torch.tensor([3], device="cuda")
    assert not fused_track_supported(md, h)
    md = _metadata(1, 1, 1)
    assert not fused_track_supported(md, None)
    assert not fused_track_supported(md, h[0, 0])
    md = _metadata(1, 0, 1)
    assert fused_track_supported(md, None)


class _Mode:
    def is_decode_or_idle(self):
        return False

    def is_extend(self):
        return True


@pytest.mark.parametrize("flydsl_decode", [False, True])
@pytest.mark.parametrize("h_vk", [False, True])
def test_track_mamba_states_uses_one_launch_on_device(monkeypatch, flydsl_decode, h_vk):
    """End to end through the layer: same pools as the eager path, and the
    SGLang helper is not called."""
    x, h = _inputs(h_vk=h_vk)
    md = _metadata(4, 3, 2)
    # The plugin's FlyDSL-decode pool is stored VK and handed over transposed.
    conv_ref, ssm_ref = _pools(True, flydsl_decode)
    conv_new, ssm_new = _pools(True, flydsl_decode)
    temporal_new = ssm_new.transpose(-1, -2) if flydsl_decode else ssm_new
    calls = []
    lb = SimpleNamespace(
        forward_metadata=md,
        req_to_token_pool=SimpleNamespace(
            mamba2_layer_cache=lambda layer: SimpleNamespace(
                conv=[conv_new], temporal=temporal_new
            )
        ),
        _track_mamba_state_extend=lambda *a: calls.append(a),
    )
    monkeypatch.setattr(
        SGLangGDNForwardContext,
        "_resolve_attn_backend",
        staticmethod(lambda fb: SimpleNamespace(linear_attn_backend=lb)),
    )
    monkeypatch.setattr(chunk_mod.ChunkGatedDeltaRuleFunction, "last_h", h)
    impl = SGLangGatedDeltaNet.__new__(SGLangGatedDeltaNet)
    torch.nn.Module.__init__(impl)
    impl.layer_num = 3
    impl.gdn_flydsl_policy = SimpleNamespace(decode=True) if flydsl_decode else None
    fb = SimpleNamespace(
        mamba_track_mask=torch.ones(4, dtype=torch.bool, device="cuda"),
        forward_mode=_Mode(),
    )

    impl._track_mamba_states(fb, x)
    _reference(x, h, conv_ref, ssm_ref, md)
    torch.cuda.synchronize()

    assert calls == []
    assert torch.equal(conv_new, conv_ref)
    assert torch.equal(ssm_new, ssm_ref)
    assert chunk_mod.ChunkGatedDeltaRuleFunction.last_h is None


def test_track_mamba_states_falls_back_for_recompute_rows(monkeypatch):
    x, h = _inputs()
    md = _metadata(1, 1, 0)
    md.track_ssm_recompute_dst = torch.tensor([4], device="cuda")
    conv, ssm = _pools(False, False)
    calls = []
    lb = SimpleNamespace(
        forward_metadata=md,
        req_to_token_pool=SimpleNamespace(
            mamba2_layer_cache=lambda layer: SimpleNamespace(conv=[conv], temporal=ssm)
        ),
        _track_mamba_state_extend=lambda fb, hh, s, m: calls.append((hh, s, m)),
    )
    monkeypatch.setattr(
        SGLangGDNForwardContext,
        "_resolve_attn_backend",
        staticmethod(lambda fb: SimpleNamespace(linear_attn_backend=lb)),
    )
    monkeypatch.setattr(chunk_mod.ChunkGatedDeltaRuleFunction, "last_h", h)
    impl = SGLangGatedDeltaNet.__new__(SGLangGatedDeltaNet)
    torch.nn.Module.__init__(impl)
    impl.layer_num = 3
    impl.gdn_flydsl_policy = None
    fb = SimpleNamespace(
        mamba_track_mask=torch.ones(1, dtype=torch.bool, device="cuda"),
        forward_mode=_Mode(),
    )

    impl._track_mamba_states(fb, x)

    ((hh, s, m),) = calls
    assert hh is h and s is ssm and m is md
    torch.testing.assert_close(conv[7], x[md.track_conv_indices[0]].T, rtol=0, atol=0)
