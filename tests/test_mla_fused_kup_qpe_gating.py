# SPDX-License-Identifier: MIT
# Gating of the fused MLA decode kernel (ATOM_MLA_FUSED_KUP_QPE): default off,
# and every unsupported case falls back to the existing two-kernel path.

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


@pytest.mark.parametrize("value", ["true", "TRUE", "on", "yes"])
def test_flag_uses_standard_bool_parser(monkeypatch, value):
    monkeypatch.setenv("ATOM_MLA_FUSED_KUP_QPE", value)
    assert envs.ATOM_MLA_FUSED_KUP_QPE is True


def _kup_args(**over):
    args = {
        "enabled": True,
        "is_prefill": False,
        "use_qrep": False,
        "dcp_world_size": 1,
        "q_out": torch.empty(M, H, L + R, dtype=FP8),
        "num_tokens": M,
        "slot_mapping": torch.zeros(M, dtype=torch.int64),
        "w_k": torch.empty(H, L, P, dtype=torch.bfloat16),
        "cos_cache": torch.empty(4096, 1, 1, R // 2, dtype=torch.bfloat16),
        "sin_cache": torch.empty(4096, 1, 1, R // 2, dtype=torch.bfloat16),
        "positions": torch.zeros(M, dtype=torch.int64),
        "rope_dim": R,
        "kv_cache": torch.empty(16, 1, BS, L + R, dtype=FP8),
        "k_scale": torch.ones(1),
    }
    args.update(over)
    if "cos_cache" in over and "sin_cache" not in over:
        args["sin_cache"] = args["cos_cache"]
    return args


def test_kup_qpe_eligible():
    assert kup_qpe_fusion_ok(**_kup_args())
    assert kup_qpe_fusion_ok(**_kup_args(k_scale=None))


@pytest.mark.parametrize(
    "over",
    [
        {"enabled": False},
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
        {"sin_cache": torch.empty(4096, R // 2, dtype=torch.float32)},
        {"sin_cache": torch.empty(4096, R, dtype=torch.bfloat16)},
        {"sin_cache": torch.empty_strided((4096, R // 2), (R, 1), dtype=torch.bfloat16)},
        {"positions": torch.zeros(3, M, dtype=torch.int64)},
        {"num_tokens": 0, "q_out": torch.empty(0, H, L + R, dtype=FP8),
         "slot_mapping": torch.zeros(0, dtype=torch.int64)},
    ],
)  # fmt: skip
def test_kup_qpe_falls_back(over):
    assert not kup_qpe_fusion_ok(**_kup_args(**over))


def _cpu_mla_method(name):
    # Execute the actual dependency-free production method without importing
    # attention_mla's GPU kernels. No stand-in copy of its branching logic.
    path = Path(__file__).parents[1] / "atom/model_ops/attention_mla.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MLAAttention"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace = {}
    exec(  # noqa: S102 - executes only the trusted production method above
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize(
    "qrep,group", [(False, False), (False, True), (True, False), (True, True)]
)
def test_q_proj_heads_matches_independent_projection(qrep, group):
    # Distinct weights for every head expose a wrong split or the wrong qrep
    # projection. Local heads are the second rank's columns of the group weight.
    m, h, group_h, d, k = 3, 2, 4, 5, 7
    x = torch.arange(m * k, dtype=torch.float32).reshape(m, k) / 10
    weight = (
        torch.arange(k * group_h * d, dtype=torch.float32).reshape(k, group_h * d) / 50
    )
    local_weight = weight[:, h * d :]
    scale = torch.tensor(0.37)
    calls = []

    def project(w, label):
        def run(a, s):
            calls.append(label)
            assert s is scale
            return (a * s) @ w

        return run

    fake = SimpleNamespace(
        qrep_enabled=qrep,
        num_heads=h,
        qrep_num_heads=group_h,
        qk_head_dim=d,
        q_proj=project(weight if qrep else local_weight, "group"),
        _local_q_proj=lambda: project(local_weight, "local"),
    )
    out = _cpu_mla_method("_q_proj_heads")(fake, x, scale, group=group)
    full = ((x * scale) @ weight).reshape(m, group_h, d)
    expected = full if qrep and group else full[:, h:, :]
    torch.testing.assert_close(out, expected)
    assert calls == (["local"] if qrep and not group else ["group"])


@pytest.mark.parametrize("rank", [0, 768, 1280, 1536, 2048])
def test_non_supported_rank_falls_back(rank):
    assert not kup_qpe_fusion_ok(
        **_kup_args(
            w_k=torch.empty(H, rank, P, dtype=torch.bfloat16),
            q_out=torch.empty(M, H, rank + R, dtype=FP8),
            kv_cache=torch.empty(16, 1, BS, rank + R, dtype=FP8),
        )
    )


@pytest.mark.parametrize("nope", [0, 16, 96, 160])
def test_non_supported_nope_falls_back(nope):
    assert not kup_qpe_fusion_ok(
        **_kup_args(w_k=torch.empty(H, L, nope, dtype=torch.bfloat16))
    )


def test_lds_over_budget_falls_back():
    assert not kup_qpe_fusion_ok(
        **_kup_args(
            w_k=torch.empty(H, 256, 128, dtype=torch.bfloat16),
            q_out=torch.empty(M, H, 512, dtype=FP8),
            rope_dim=256,
            cos_cache=torch.empty(4096, 128, dtype=torch.bfloat16),
            kv_cache=torch.empty(16, 1, BS, 512, dtype=FP8),
        )
    )


@pytest.mark.parametrize(
    "rank,nope,rope", [(256, 128, 128), (512, 128, 256), (1024, 32, 32)]
)
def test_supported_partition_bounds(rank, nope, rope):
    assert kup_qpe_fusion_ok(
        **_kup_args(
            w_k=torch.empty(H, rank, nope, dtype=torch.bfloat16),
            q_out=torch.empty(M, H, rank + rope, dtype=FP8),
            rope_dim=rope,
            cos_cache=torch.empty(4096, rope // 2, dtype=torch.bfloat16),
            kv_cache=torch.empty(16, 1, BS, rank + rope, dtype=FP8),
        )
    )


@pytest.mark.parametrize(
    "cos",
    [
        torch.empty(()),
        torch.empty(0, R // 2),
        torch.empty(16, 2, R // 2),
        torch.empty(16, R // 2),
    ],
)
def test_invalid_freq_cache_falls_back(cos):
    assert not kup_qpe_fusion_ok(**_kup_args(cos_cache=cos))


def test_q_out_int32_limit_falls_back():
    # Meta tensors exercise the byte-offset bound without a multi-GB allocation.
    args = _kup_args()
    args.update(
        {k: v.to("meta") for k, v in args.items() if isinstance(v, torch.Tensor)}
    )
    args["q_out"] = torch.empty_strided(
        (M, H, L + R), (2**31 // M + 1, L + R, 1), dtype=FP8, device="meta"
    )
    assert not kup_qpe_fusion_ok(**args)


@pytest.mark.parametrize("m", [64, 65, 128, 512])
def test_empty_partition_descriptor_bounds(m):
    path = Path(__file__).parents[1] / "atom/model_ops/mla_fused/fused_kup_qpe.py"
    kernel = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_kup_qpe_persistent_kernel"
    )
    names = {
        "num_m_tiles",
        "chunk",
        "t0",
        "t1",
        "n_local",
        "r0",
        "r_end",
        "r_base",
        "desc_rows",
    }
    code = [
        n
        for n in kernel.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in names
    ]
    compiled = compile(ast.Module(body=code, type_ignores=[]), str(path), "exec")
    for ms in (0, 1):
        values = {
            "M": m,
            "BLOCK_M": 64,
            "M_SPLIT": 2,
            "ms": ms,
            "gl": SimpleNamespace(
                cdiv=lambda a, b: (a + b - 1) // b, minimum=min, maximum=max
            ),
        }
        exec(compiled, values)  # noqa: S102 - trusted production descriptor arithmetic
        assert 0 <= values["r_base"] < m
        assert values["desc_rows"] >= 1
        if m == 64 and ms == 1:
            assert values["n_local"] == 0 and values["desc_rows"] == 1


def test_disabled_forward_gate_does_not_read_runtime_inputs():
    path = Path(__file__).parents[1] / "atom/model_ops/attention_mla.py"
    gate = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Attribute)
        and n.test.attr == "_kup_qpe_enabled"
    )
    # No q, metadata, cache, arch or env objects exist in this namespace. Any
    # accidental read on the default path therefore raises instead of passing.
    values = {"self": SimpleNamespace(_kup_qpe_enabled=False), "fused_kup_qpe": False}
    exec(  # noqa: S102 - trusted production gate
        compile(ast.Module(body=[gate], type_ignores=[]), str(path), "exec"), values
    )
    assert values["fused_kup_qpe"] is False
