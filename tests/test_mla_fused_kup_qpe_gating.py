# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
# Gating of the fused MLA decode kernel (ATOM_MLA_FUSED_KUP_QPE): default off,
# and every unsupported case falls back to the existing two-kernel path.

import ast
import builtins
import functools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.mla_fused import kup_qpe_fusion_reason
from atom.utils import envs

FP8 = torch.float8_e4m3fn
H, L, P, R, BS, M = 128, 512, 128, 64, 64, 96


def _eligible(**kwargs):
    return kup_qpe_fusion_reason(**kwargs) is None


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
        "use_qrep": False,
        "q_out": torch.empty(M, H, L + R, dtype=FP8),
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
    assert _eligible(**_kup_args())
    assert _eligible(**_kup_args(k_scale=None))


@pytest.mark.parametrize(
    "over",
    [
        {"use_qrep": True},
        {"q_out": None},
        {"kv_cache": None},
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
        {"q_out": torch.empty(0, H, L + R, dtype=FP8),
         "positions": torch.zeros(0, dtype=torch.int64)},
    ],
)  # fmt: skip
def test_kup_qpe_falls_back(over):
    assert not _eligible(**_kup_args(**over))


def _cpu_mla_method(name, **globals_):
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
    method.decorator_list = []
    namespace = globals_
    exec(  # noqa: S102 - executes only the trusted production method above
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize(
    "qrep,group", [(False, False), (False, True), (True, False), (True, True)]
)
def test_inline_q_proj_matches_independent_projection(qrep, group):
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
        qk_nope_head_dim=3,
        qk_rope_head_dim=2,
        W_K=torch.arange(h * 8 * 3, dtype=torch.float32).view(h, 8, 3) / 20,
        W_K_scale=None,
        W_K_qrep=torch.arange(group_h * 8 * 3, dtype=torch.float32).view(group_h, 8, 3)
        / 30,
        W_K_qrep_scale=None,
        q_proj=project(weight if qrep else local_weight, "group"),
        _local_q_proj=lambda: project(local_weight, "local"),
    )

    def bmm(a, w, *, YQ):
        YQ.copy_(torch.bmm(a, w.transpose(1, 2)).to(torch.bfloat16))

    nope, pe = _cpu_mla_method(
        "_q_proj_and_k_up_proj",
        torch=torch,
        is_rocm_aiter_fp4bmm_enabled=lambda: False,
        _aiter_triton_bf16_bmm=bmm,
    )(fake, x, scale, group=group)
    full = ((x * scale) @ weight).reshape(m, group_h, d)
    expected = full if qrep and group else full[:, h:, :]
    weights = fake.W_K_qrep if qrep and group else fake.W_K
    expected_nope = torch.einsum("mhp,hlp->mhl", expected[..., :3], weights).to(
        torch.bfloat16
    )
    torch.testing.assert_close(nope, expected_nope.to(nope.dtype))
    torch.testing.assert_close(pe, expected[..., 3:])
    assert calls == (["local"] if qrep and not group else ["group"])


@pytest.mark.parametrize("rank", [0, 768, 1280, 1536, 2048])
def test_non_supported_rank_falls_back(rank):
    assert not _eligible(
        **_kup_args(
            w_k=torch.empty(H, rank, P, dtype=torch.bfloat16),
            q_out=torch.empty(M, H, rank + R, dtype=FP8),
            kv_cache=torch.empty(16, 1, BS, rank + R, dtype=FP8),
        )
    )


@pytest.mark.parametrize("nope", [0, 16, 96, 160])
def test_non_supported_nope_falls_back(nope):
    assert not _eligible(
        **_kup_args(w_k=torch.empty(H, L, nope, dtype=torch.bfloat16))
    )


def test_lds_over_budget_falls_back():
    assert not _eligible(
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
    assert _eligible(
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
    assert not _eligible(**_kup_args(cos_cache=cos))


def test_q_out_int32_limit_falls_back():
    # Meta tensors exercise the byte-offset bound without a multi-GB allocation.
    args = _kup_args()
    args.update(
        {k: v.to("meta") for k, v in args.items() if isinstance(v, torch.Tensor)}
    )
    args["q_out"] = torch.empty_strided(
        (M, H, L + R), (2**31 // M + 1, L + R, 1), dtype=FP8, device="meta"
    )
    assert not _eligible(**args)


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
    driver = ast.FunctionDef(
        name="driver",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[gate],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[driver], type_ignores=[]))
    values = {}
    exec(  # noqa: S102 - trusted production gate
        compile(module, str(path), "exec"), values
    )
    assert values["driver"](SimpleNamespace(_kup_qpe_enabled=False)) is None


def test_fallback_reasons_distinguish_failures():
    assert "matching FP8" in kup_qpe_fusion_reason(
        **_kup_args(q_out=torch.empty(M, H, L + R, dtype=torch.bfloat16))
    )
    assert "sin/cos" in kup_qpe_fusion_reason(
        **_kup_args(sin_cache=torch.empty(4096, R, dtype=torch.bfloat16))
    )


def test_fusion_declines_prefill_after_parent_static_gate():
    path = Path(__file__).parents[1] / "atom/model_ops/attention_mla.py"
    branch = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Attribute)
        and n.test.attr == "_kup_qpe_enabled"
    )
    driver = ast.FunctionDef(
        name="driver",
        decorator_list=[],
        body=[branch],
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
    )
    module = ast.fix_missing_locations(ast.Module(body=[driver], type_ignores=[]))
    reasons = []
    fake = SimpleNamespace(
        _kup_qpe_enabled=True,
        _log_kup_qpe=lambda reason, count: reasons.append(reason),
    )
    namespace = {
        "context": SimpleNamespace(is_prefill=True),
        "q": torch.empty(64, 1),
    }
    exec(  # noqa: S102 - trusted production fusion guards
        compile(module, str(path), "exec"), namespace
    )
    assert namespace["driver"](fake) is None
    assert reasons == ["prefill"]


def test_diagnostics_log_each_reason_once():
    rows = []
    layer_type = SimpleNamespace(_kup_qpe_logged=set())
    logger = SimpleNamespace(info=lambda *args: rows.append(args))
    log = _cpu_mla_method("_log_kup_qpe", MLAAttention=layer_type, logger=logger)
    fake = SimpleNamespace()
    for reason in (
        "query/cache dtype must be matching FP8",
        "sin/cos cache layout or dtype mismatch",
        "eligible",
    ):
        log(fake, reason, 64)
        log(fake, reason, 128)
    assert len(rows) == 3
    assert [row[-1] for row in rows] == [
        "query/cache dtype must be matching FP8",
        "sin/cos cache layout or dtype mismatch",
        "eligible",
    ]


@pytest.mark.parametrize(
    "cause", ["absorb", "shuffled", "qout", "tdm", "aiter", "waiting", "off"]
)
def test_initial_diagnostics_name_prerequisites(monkeypatch, cause):
    for name in (
        "ATOM_MLA_FUSED_KUP_QPE",
        "ATOM_MLA_ABSORB_BF16",
        "ATOM_MLA_ABSORB_BF16_FUSE_QOUT",
    ):
        monkeypatch.setenv(name, "1")
    if cause == "absorb":
        monkeypatch.setenv("ATOM_MLA_ABSORB_BF16", "0")
    if cause == "qout":
        monkeypatch.setenv("ATOM_MLA_ABSORB_BF16_FUSE_QOUT", "0")
    if cause == "off":
        monkeypatch.delenv("ATOM_MLA_FUSED_KUP_QPE")
    path = Path(__file__).parents[1] / "atom/model_ops/attention_mla.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MLAAttention"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    nodes = [
        n
        for n in init.body
        if (
            isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Attribute)
            and n.targets[0].attr in ("_kup_qpe_requested", "_kup_qpe_enabled")
        )
        or (
            isinstance(n, ast.If)
            and isinstance(n.test, ast.Attribute)
            and n.test.attr == "_kup_qpe_requested"
        )
    ]
    helpers = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_mla_absorb_bf16_enabled", "_mla_absorb_bf16_fuse_qout_enabled")
    ]
    reasons = []
    fake = SimpleNamespace(
        kv_shuffled=cause != "shuffled",
        _q_nope_prestored_supported=cause != "aiter",
        _log_kup_qpe=reasons.append,
    )

    def tdm():
        assert cause != "off", "default initialization queried the architecture"
        return cause != "tdm"

    namespace = {
        "self": fake,
        "envs": envs,
        "arch_info": SimpleNamespace(is_tdm_avail=tdm, get_arch=lambda: "gfx950"),
    }
    exec(  # noqa: S102 - trusted initialization and registered flag helpers
        compile(ast.Module(body=helpers + nodes, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    if cause == "off":
        assert not reasons
    else:
        expected = {
            "absorb": "ATOM_MLA_ABSORB_BF16=0",
            "shuffled": "ATOM_MLA_KV_SHUFFLED=0",
            "qout": "ATOM_MLA_ABSORB_BF16_FUSE_QOUT=0",
            "tdm": "no TDM on gfx950",
            "aiter": "q_nope_prestored is unsupported",
            "waiting": "M>=64",
        }
        assert expected[cause] in reasons[0]


def test_layout_builder_caches_each_shape():
    path = Path(__file__).parents[1] / "atom/model_ops/mla_fused/fused_kup_qpe.py"
    method = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_layouts"
    )
    calls = []

    def wmma(warps):
        calls.append(("wmma", warps))
        return object(), object(), object()

    def shared(*args):
        calls.append(("shared", args))
        return object(), object()

    def imports(name, *args, **kwargs):
        if name == "aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16":
            return SimpleNamespace(
                create_wmma_layouts=wmma, create_shared_layouts=shared
            )
        return builtins.__import__(name, *args, **kwargs)

    namespace = {
        "functools": functools,
        "__builtins__": {**vars(builtins), "__import__": imports},
    }
    exec(  # noqa: S102 - trusted layout builder with local import stand-ins
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    layout = namespace["_layouts"]
    first = layout(64, 256, 128, 4)
    assert layout(64, 256, 128, 4) is first
    assert layout(64, 256, 32, 4) is not first
    assert len(calls) == 4
