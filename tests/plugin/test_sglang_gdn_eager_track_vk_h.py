# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""SGLang radix track reads FlyDSL's chunk states as a view, not a copy.

FlyDSL's prefill hands out its per-chunk states `h` as a logical KV view over
VK storage (`h.transpose(-1, -2)`). The plugin's extend track
(`SGLangGatedDeltaNet._track_mamba_states`) passes it to SGLang's eager
`_track_mamba_state_extend`, which gathers `h[src]`. These tests run that real
helper on the view and on its contiguous copy and require the same pool bytes,
on a KV pool and on the VK view of the pool FlyDSL decode uses.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

hlab = pytest.importorskip("sglang.srt.layers.attention.hybrid_linear_attn_backend")

from atom.model_ops.attention_gdn import GatedDeltaNet
from atom.model_ops.fla_ops import chunk as chunk_mod
from atom.plugin.sglang.attention_backend.attention_gdn import (
    SGLangGatedDeltaNet,
    SGLangGDNForwardContext,
)
from atom.plugin.sglang.runtime import bind_current_forward_batch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, K, V = 2, 8, 16  # K != V: an interchanged axis cannot go unnoticed
NT, SLOTS, D, W = 6, 12, 40, 3


class _Mode:
    def is_decode_or_idle(self):
        return False

    def is_extend(self):
        return True

    def is_target_verify(self):
        return False


class _Pool:
    def __init__(self, conv, temporal):
        self.mamba_map = {0: 0}
        self.mamba_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(conv=[conv[None]], temporal=temporal[None])
        )

    def mamba2_layer_index(self, layer_id):
        return self.mamba_map[layer_id]

    def mamba2_layer_cache(self, layer_id):
        state = self.mamba_pool.mamba_cache
        index = self.mamba2_layer_index(layer_id)
        return SimpleNamespace(
            conv=[state.conv[0][index]], temporal=state.temporal[index]
        )


def _h(vk_view: bool) -> torch.Tensor:
    g = torch.Generator().manual_seed(3432)
    h = torch.randn(1, NT, H, V, K, generator=g).to(DEV, torch.bfloat16)
    h = h.transpose(-1, -2)  # FlyDSL's "kv" snapshot: KV view of VK storage
    return h if vk_view else h.contiguous()


def _metadata(n_h: int, n_f: int):
    def t(xs):
        return torch.tensor(xs, device=DEV, dtype=torch.int64)

    rows = n_h + n_f
    return SimpleNamespace(
        has_mamba_track_mask=True,
        track_conv_indices=(t(range(rows))[:, None] * 2 + torch.arange(W, device=DEV)),
        conv_states_mask_indices=t(range(6, 6 + rows)),
        track_ssm_h_src=t([4, 1, 5][:n_h]),
        track_ssm_h_dst=t(range(6, 6 + n_h)),
        track_ssm_final_src=t([0, 2][:n_f]),
        track_ssm_final_dst=t(range(6 + n_h, 6 + rows)),
        track_ssm_recompute_dst=None,
    )


def _run(monkeypatch, h, md, pool_vk: bool):
    g = torch.Generator().manual_seed(7)
    conv = torch.zeros(SLOTS, D, W, device=DEV, dtype=torch.bfloat16)
    if pool_vk:
        temporal = torch.randn(SLOTS, H, V, K, generator=g).to(DEV)
    else:
        temporal = torch.randn(SLOTS, H, K, V, generator=g).to(DEV)
    x = torch.randn(16, D, generator=g).to(DEV, torch.bfloat16)
    backend = SimpleNamespace(
        forward_metadata=md,
        req_to_token_pool=_Pool(conv, temporal),
        _track_mamba_state_extend=lambda fb, hh, s, m: (
            hlab.MambaAttnBackendBase._track_mamba_state_extend(None, fb, hh, s, m)
        ),
    )
    monkeypatch.setattr(
        SGLangGDNForwardContext,
        "_resolve_attn_backend",
        staticmethod(lambda fb: SimpleNamespace(linear_attn_backend=backend)),
    )
    monkeypatch.setattr(chunk_mod.ChunkGatedDeltaRuleFunction, "last_h", h)
    impl = SGLangGatedDeltaNet.__new__(SGLangGatedDeltaNet)
    torch.nn.Module.__init__(impl)
    impl.layer_num = 0
    impl.dt_bias = torch.nn.Parameter(torch.zeros(1, device=DEV, dtype=x.dtype))
    impl.gdn_flydsl_policy = SimpleNamespace(decode=True) if pool_vk else None
    rows = md.track_conv_indices.shape[0]
    fb = SimpleNamespace(
        mamba_track_mask=torch.ones(rows, dtype=torch.bool, device=DEV),
        forward_mode=_Mode(),
    )
    # Supply the snapshot independently of GDN compute, then exercise the
    # frontend's real forward/track dispatch and SGLang's eager writer.
    monkeypatch.setattr(
        GatedDeltaNet, "forward", lambda self, x, b, a, out, layer_name: out
    )
    gates = torch.zeros(1, device=DEV, dtype=x.dtype)
    with bind_current_forward_batch(fb):
        impl.forward(x, gates, gates, torch.empty_like(x), "layer_0")
    assert chunk_mod.ChunkGatedDeltaRuleFunction.last_h is None
    return conv, temporal


@pytest.mark.parametrize("pool_vk", [False, True])
@pytest.mark.parametrize("counts", [(3, 2), (1, 0), (0, 2), (2, 1)])
def test_eager_track_reads_vk_view_bit_for_bit(monkeypatch, pool_vk, counts):
    md = _metadata(*counts)
    conv_ref, ssm_ref = _run(monkeypatch, _h(False), md, pool_vk)
    view = _h(True)
    assert view.stride(-2) == 1 and not view.is_contiguous()
    conv_new, ssm_new = _run(monkeypatch, view, md, pool_vk)
    assert torch.equal(conv_new, conv_ref)
    assert torch.equal(ssm_new, ssm_ref)
    # Non-vacuous: the h rows hold h[src] in the logical KV order.
    ssm_kv = ssm_new.transpose(-1, -2) if pool_vk else ssm_new
    for src, dst in zip(md.track_ssm_h_src.tolist(), md.track_ssm_h_dst.tolist()):
        assert torch.equal(ssm_kv[dst], view[0, src].float())
