# SPDX-License-Identifier: MIT
# Gating of the fused MLA decode kernel (ATOM_MLA_FUSED_KUP_QPE): default off,
# and every unsupported case falls back to the existing two-kernel path.

from types import SimpleNamespace

import pytest
import torch
from conftest import gpu_stack_import

from atom.model_ops.mla_fused import kup_qpe_fusion_ok
from atom.utils import envs

FP8 = torch.float8_e4m3fn
H, L, P, R, BS, M = 128, 512, 128, 64, 64, 96


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("ATOM_MLA_FUSED_KUP_QPE", raising=False)
    assert envs.ATOM_MLA_FUSED_KUP_QPE is False
    monkeypatch.setenv("ATOM_MLA_FUSED_KUP_QPE", "1")
    assert envs.ATOM_MLA_FUSED_KUP_QPE is True
    monkeypatch.setenv("ATOM_MLA_FUSED_KUP_QPE", "0")
    assert envs.ATOM_MLA_FUSED_KUP_QPE is False


def _kup_args(**over):
    args = {
        "enabled": True,
        "arch": "gfx1250",
        "is_prefill": False,
        "use_qrep": False,
        "dcp_world_size": 1,
        "q_out": torch.empty(M, H, L + R, dtype=FP8),
        "num_tokens": M,
        "slot_mapping": torch.zeros(M, dtype=torch.int64),
        "w_k": torch.empty(H, L, P, dtype=torch.bfloat16),
        "cos_cache": torch.empty(4096, 1, 1, R // 2, dtype=torch.bfloat16),
        "rope_dim": R,
        "kv_cache": torch.empty(16, 1, BS, L + R, dtype=FP8),
        "k_scale": torch.ones(1),
    }
    args.update(over)
    return args


def test_kup_qpe_eligible():
    assert kup_qpe_fusion_ok(**_kup_args())
    assert kup_qpe_fusion_ok(**_kup_args(k_scale=None))
    assert kup_qpe_fusion_ok(**_kup_args(arch="gfx1250-strict"))


@pytest.mark.parametrize(
    "over",
    [
        {"enabled": False},
        {"arch": "gfx950"},
        {"arch": ""},
        {"is_prefill": True},
        {"use_qrep": True},
        {"dcp_world_size": 2},
        {"q_out": None},
        {"kv_cache": None},
        {"slot_mapping": torch.zeros(M + 3, dtype=torch.int64)},
        {"q_out": torch.empty(M, H, L + R, dtype=torch.bfloat16)},
        {"kv_cache": torch.empty(16, 1, BS, L + R, dtype=torch.bfloat16)},
        {"kv_cache": torch.empty(16, 1, 8, L + R, dtype=FP8)},
        {"w_k": torch.empty(H, L, P, dtype=torch.float8_e4m3fn)},
        {"cos_cache": torch.empty(4096, R, dtype=torch.bfloat16)},
        {"k_scale": 1.0},
        {"k_scale": torch.ones(2)},
        {"num_tokens": 0, "q_out": torch.empty(0, H, L + R, dtype=FP8),
         "slot_mapping": torch.zeros(0, dtype=torch.int64)},
    ],
)  # fmt: skip
def test_kup_qpe_falls_back(over):
    assert not kup_qpe_fusion_ok(**_kup_args(**over))


def test_q_proj_heads_matches_old_split():
    mla = gpu_stack_import("atom.model_ops.attention_mla")
    q = torch.randn(M, H * (P + R))
    fake = SimpleNamespace(
        qrep_enabled=False,
        num_heads=H,
        qrep_num_heads=H,
        qk_head_dim=P + R,
        q_proj=lambda x, s: x,
    )
    out = mla.MLAAttention._q_proj_heads(fake, q, None, group=False)
    assert out.shape == (M, H, P + R) and out.data_ptr() == q.data_ptr()
