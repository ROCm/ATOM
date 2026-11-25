# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 FP4PTPC: `SharedExpertStaging(ptpc=True)` copies the per-channel
FP8 shared expert into routed experts that online quant made per-channel FP8, on
the MiniMax-M3.1 test's FusedMoE stand-in."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.test_minimax_m31_shared_fusion import (
    FP8,
    I_LOCAL,
    ROUTED,
    SHARED,
    H,
    _fused_mxfp4_layer,
    _load_shared,
    _param,
    staging_mod,
)


@pytest.mark.parametrize("fnuz", [False, True])
@pytest.mark.parametrize(("tp_size", "tp_rank"), [(1, 0), (2, 1)])
def test_per_channel_fp8_shared_expert_is_copied_into_its_slot(fnuz, tp_size, tp_rank):
    """FP4PTPC: once online quant made the routed experts per-channel FP8, this
    rank's shard of the shared expert goes into its slot as it is (FNUZ bytes
    and doubled scales where the routed experts are FNUZ); a w2 row keeps its
    whole scale."""
    layer = _fused_mxfp4_layer(ptpc=True, tp_size=tp_size, tp_rank=tp_rank)
    torch.manual_seed(0)
    inter = I_LOCAL * tp_size
    w = {k: (torch.randn(*s) * 4).to(FP8) for k, s in
         (("w1", (inter, H)), ("w3", (inter, H)), ("w2", (H, inter)))}  # fmt: skip
    s = {k: torch.rand(v.shape[0], 1) + 0.5 for k, v in w.items()}
    _load_shared(layer, w, s)
    # What `_online_quant` leaves: unpadded per-channel FP8 buffers.
    e = ROUTED + SHARED
    f8 = torch.float8_e4m3fnuz if fnuz else FP8
    layer.w13_weight = nn.Parameter(_param(e, 2 * I_LOCAL, H).fill_(9).to(f8), False)
    layer.w13_weight_scale = _param(e, 2 * I_LOCAL).fill_(9)
    layer.w2_weight = nn.Parameter(_param(e, H, I_LOCAL).fill_(9).to(f8), False)
    layer.w2_weight_scale = _param(e, H).fill_(9)
    layer.shared_expert_staging.fuse(layer)

    k = 2.0 if fnuz else 1.0
    rows = slice(tp_rank * I_LOCAL, (tp_rank + 1) * I_LOCAL)
    w13, w13_s = layer.w13_weight.data[ROUTED], layer.w13_weight_scale.data[ROUTED]
    for shard, off in (("w1", 0), ("w3", I_LOCAL)):
        got = w13[off : off + I_LOCAL].float() * w13_s[off : off + I_LOCAL, None]
        assert torch.equal(got, w[shard][rows].float() * s[shard][rows])
        assert torch.equal(w13_s[off : off + I_LOCAL], k * s[shard][rows].flatten())
    w2, w2_s = layer.w2_weight.data[ROUTED], layer.w2_weight_scale.data[ROUTED]
    assert torch.equal(w2.float() * w2_s[:, None], w["w2"][:, rows].float() * s["w2"])
    assert torch.equal(w2_s, k * s["w2"].flatten())
    assert (layer.w13_weight.data[:ROUTED].float() == 9).all()  # routed untouched


def test_per_channel_fp8_shared_expert_refuses_all2all():
    """The FP8 online method has no all2all shared-expert dispatch."""
    layer = _fused_mxfp4_layer()
    layer.expert_layout.uses_all2all_fusion = True
    with pytest.raises(NotImplementedError, match="all2all"):
        staging_mod.SharedExpertStaging(layer, None, H, I_LOCAL, ptpc=True)


def test_dummy_load_leaves_the_slot_alone(monkeypatch):
    layer = _fused_mxfp4_layer(ptpc=True)
    monkeypatch.setattr(
        staging_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(load_dummy="zero"),
    )
    layer.shared_expert_staging.fuse(layer)
    assert layer.shared_expert_staging.staged is None
