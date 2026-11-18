# SPDX-License-Identifier: Apache-2.0
"""Ragged prefill must not discover a new tuned launcher during serving."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def test_warmup_includes_intermediate_variant_and_resolves_active_rows(atom_modules):
    warmup = importlib.import_module("atom.model_ops.gemm_warmup")
    table = [
        {"M": m, "N": 5120, "K": 2304, "w_scale_block": "32x32"}
        for m in (0, 4096, 8192, 14336, 16384, 32768)
    ]
    table += [
        {"M": 128, "N": 64, "K": 2304, "w_scale_block": "32x32"},
        {"M": 256, "N": 5120, "K": 2304, "w_scale_block": "1x32"},
    ]
    queried = []

    def lookup(m, n, k, block):
        queried.append((m, n, k, block))
        name = "small" if m < 8192 else "4x4" if m == 14336 else "3x4"
        return {"kernelId": "mxfp8", "kernelName": name}

    rows = list(warmup._variant_rows(table, 5120, 2304, "32x32", 16384, lookup))
    assert rows == [
        (1, ("mxfp8", "small")),
        (8192, ("mxfp8", "3x4")),
        (14336, ("mxfp8", "4x4")),
    ]
    assert [row[0] for row in queried] == [1, 4096, 8192, 14336, 16384]


def test_warmup_resolves_padded_row_above_capacity_and_skips_untuned(atom_modules):
    warmup = importlib.import_module("atom.model_ops.gemm_warmup")
    table = [{"M": 16384, "N": 5120, "K": 2304, "w_scale_block": "32x32"}]

    def lookup(m, *_):
        if m == 15000:
            return {"kernelId": "mxfp8", "kernelName": "padded16384"}

    assert list(warmup._variant_rows(table, 5120, 2304, "32x32", 15000, lookup)) == [
        (15000, ("mxfp8", "padded16384"))
    ]


def test_model_without_native_preshuffled_fp8_needs_no_aiter_table(atom_modules):
    from types import SimpleNamespace
    from unittest.mock import Mock

    warmup = importlib.import_module("atom.model_ops.gemm_warmup")
    model = SimpleNamespace(modules=lambda: [SimpleNamespace()], forward=Mock())
    assert warmup.warmup_native_fp8_gemms(model, 16384) == 0
    model.forward.assert_not_called()


def test_single_row_and_multi_row_share_name_but_need_two_launchers(atom_modules):
    warmup = importlib.import_module("atom.model_ops.gemm_warmup")
    table = [
        {"M": m, "N": 5120, "K": 2304, "w_scale_block": "32x32"} for m in (1, 2, 4, 8)
    ]
    name = "flydsl_mxpsh_16x32x256_F8_F8_B16_w0_x0_sk1_pf_nt"
    rows = list(
        warmup._variant_rows(
            table,
            5120,
            2304,
            "32x32",
            8,
            lambda *_: {"kernelId": "mxfp8", "kernelName": name},
        )
    )
    assert rows == [(1, ("mxfp8", name)), (2, ("mxfp8", name))]


def test_a16_warmup_uses_bias_dtype_layout_and_keeps_splitk_variants(atom_modules):
    from types import SimpleNamespace

    import torch

    warmup = importlib.import_module("atom.model_ops.gemm_warmup")
    weight = SimpleNamespace(shape=(128, 5120), dtype=torch.bfloat16, is_shuffled=True)
    table = [{"M": m, "N": 128, "K": 5120} for m in (16, 128, 512, 4096, 32768)]
    table += [{"M": 256, "N": 64, "K": 5120}]
    queries = []

    def lookup(*args):
        queries.append(args)
        return {
            "libtype": "flydsl",
            "solidx": -1,
            "kernelName": "same",
            "splitK": 1 if args[0] < 512 else 2,
        }

    assert list(warmup._a16_variant_rows(table, weight, object(), 2048, lookup)) == [
        1,
        512,
    ]
    assert [q[0] for q in queries] == [1, 16, 128, 512, 2048]
    assert all(
        q[1:] == (128, 5120, True, "torch.bfloat16", "torch.bfloat16", False, True)
        for q in queries
    )


def test_a16_fp16_weights_use_linear_default_bf16_output(atom_modules):
    from types import SimpleNamespace

    import torch

    warmup = importlib.import_module("atom.model_ops.gemm_warmup")
    weight = SimpleNamespace(shape=(128, 5120), dtype=torch.float16)
    queries = []

    def lookup(*args):
        queries.append(args)
        return {"libtype": "hipblaslt", "solidx": 0}

    assert list(warmup._a16_variant_rows([], weight, None, 2048, lookup)) == [1]
    assert all(q[4:6] == ("torch.float16", "torch.bfloat16") for q in queries)


def test_bmm_warmup_covers_non_power_of_two_split_and_active_tune(atom_modules):
    warmup = importlib.import_module("atom.model_ops.gemm_warmup")
    queries = []

    def lookup(b, m, n, k, **kwargs):
        queries.append((b, m, n, k, kwargs))
        if m >= 7:
            return {
                "libtype": "opus",
                "kernelName": "tuned",
                "kernelId": 2,
                "splitK": 1 if m < 9 else 2,
            }

    rows = list(
        warmup._bmm_variant_rows(
            8,
            1024,
            512,
            12,
            lookup,
            lambda b, m, n, k: "split" if m <= 3 else "unsplit",
        )
    )
    assert rows == [1, 4, 7, 9]
    assert [q[1] for q in queries] == list(range(1, 13))
    assert all(q[4] == {"w_scale_block": "32x32", "bpreshuffle": True} for q in queries)


def test_bmm_warmup_skips_models_without_grouped_fp8_projection(atom_modules):
    from types import SimpleNamespace

    warmup = importlib.import_module("atom.model_ops.gemm_warmup")
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(groups=8, o_rank=1024)])
    assert warmup.warmup_v41_bmms(model, 16384) == 0


@pytest.fixture
def a16_runtime(atom_modules, monkeypatch, tmp_path):
    warmup = importlib.import_module("atom.model_ops.gemm_warmup")
    linear = importlib.import_module("atom.model_ops.linear")
    compressor = importlib.import_module("atom.model_ops.deepseek_v41.compressor")
    # Same weight geometry, different serving output dtypes: dedup must keep both.
    model = torch.nn.ModuleList(
        [
            linear.ReplicatedLinear(64, 64),
            compressor.Compressor(64, 32, 2, 1e-6),
            compressor.Compressor(64, 64, 1, 1e-6),
        ]
    )
    for layer in model.modules():
        if isinstance(layer, linear.LinearBase):
            layer.weight = torch.nn.Parameter(torch.zeros(64, 64, dtype=torch.bfloat16))
    table = tmp_path / "a16.csv"
    table.write_text("M,N,K\n1,64,64\n2,64,64\n4,64,64\n8,64,64\n")

    def config(m, n, k, bias, dtype, output_dtype, scale, shuffled):
        return {
            "libtype": "flydsl",
            "solidx": -1,
            "kernelName": "same",
            "splitK": 2 if output_dtype == str(torch.float32) and m >= 4 else 1,
        }

    lookup = Mock(side_effect=config)
    mm = Mock()
    core = SimpleNamespace(
        AITER_CONFIGS=SimpleNamespace(AITER_CONFIG_GEMM_BF16_FILE=str(table))
    )
    monkeypatch.setattr(warmup, "get_GEMM_A16W16_config", lookup)
    monkeypatch.setattr(warmup, "tgemm", SimpleNamespace(mm=mm))
    monkeypatch.setattr(warmup, "AITER_CONFIGS", core.AITER_CONFIGS)
    synchronize = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    return SimpleNamespace(
        warmup=warmup,
        model=model,
        lookup=lookup,
        mm=mm,
        core=core,
        synchronize=synchronize,
    )


def test_a16_warmup_keeps_fp32_compressor_and_bf16_projection_variants(a16_runtime):
    state = a16_runtime
    assert state.warmup.warmup_a16_gemms(state.model, 8) == 3
    assert [
        (call.args[0].shape[0], call.kwargs["otype"])
        for call in state.mm.call_args_list
    ] == [(1, torch.bfloat16), (1, torch.float32), (4, torch.float32)]
    assert {call.args[5] for call in state.lookup.call_args_list} == {
        str(torch.bfloat16),
        str(torch.float32),
    }
    assert all(call.args[0].dtype == torch.bfloat16 for call in state.mm.call_args_list)
    assert state.synchronize.call_count == 2


@pytest.mark.parametrize("all_untuned", [False, True])
def test_a16_warmup_skips_missing_configs_without_empty_allocation(
    a16_runtime, monkeypatch, all_untuned
):
    state = a16_runtime
    state.lookup.side_effect = lambda m, *_: (
        None if all_untuned or m == 1 else {"libtype": "hipblaslt", "solidx": 0}
    )
    allocate = Mock(wraps=torch.zeros)
    monkeypatch.setattr(torch, "zeros", allocate)
    count = state.warmup.warmup_a16_gemms(state.model, 8)
    if all_untuned:
        assert count == 0
        state.mm.assert_not_called()
        state.synchronize.assert_not_called()
        allocate.assert_not_called()
    else:
        assert count == 2
        assert all(call.args[0].shape[0] == 2 for call in state.mm.call_args_list)


def test_a16_warmup_warns_and_skips_missing_table(
    a16_runtime, monkeypatch, caplog, tmp_path
):
    state = a16_runtime
    monkeypatch.setattr(
        state.core.AITER_CONFIGS,
        "AITER_CONFIG_GEMM_BF16_FILE",
        str(tmp_path / "missing.csv"),
    )
    with caplog.at_level("WARNING", logger="atom"):
        assert state.warmup.warmup_a16_gemms(state.model, 8) == 0
    assert "tune table" in caplog.text and "missing.csv" in caplog.text
    state.mm.assert_not_called()


def test_a16_warmup_does_not_hide_kernel_launch_failure(a16_runtime):
    state = a16_runtime
    state.mm.side_effect = RuntimeError("kernel launch failed")
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        state.warmup.warmup_a16_gemms(state.model, 8)


@pytest.mark.parametrize("kind", ["native", "bmm"])
def test_fp8_warmup_still_launches_available_variants(
    a16_runtime, monkeypatch, tmp_path, kind
):
    state = a16_runtime
    launcher = Mock()
    monkeypatch.setattr(
        state.warmup,
        "get_mxscale_bpreshuffle_config",
        lambda m, *_: {
            "kernelId": "fp8",
            "kernelName": "single" if m == 1 else "multi",
        },
    )
    monkeypatch.setattr(state.warmup, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(state.warmup, "batched_gemm_a8w8_mxscale_bpreshuffle", launcher)
    monkeypatch.setattr(
        state.warmup, "lookup_mxscale_bmm_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        state.warmup,
        "pick_bmm_kernel_name",
        lambda b, m, n, k: "small" if m <= 2 else "large",
    )
    table = tmp_path / "fp8.csv"
    table.write_text("M,N,K,w_scale_block\n1,64,64,32x32\n2,64,64,32x32\n")
    state.core.AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE_FILE = str(
        table
    )
    blockscale = importlib.import_module("atom.model_ops.blockscale")
    monkeypatch.setattr(blockscale, "native_quant_linear", launcher)
    weight = torch.empty((64, 64), dtype=torch.float8_e4m3fn)
    weight.is_shuffled = True
    scale = torch.empty((2, 2), dtype=torch.float8_e8m0fnu)
    layer = SimpleNamespace(
        weight=weight,
        weight_scale=scale,
        native_a8_group_rows=32,
        groups=2,
        o_rank=32,
        wo_a=SimpleNamespace(weight=weight, weight_scale=scale),
    )
    model = SimpleNamespace(modules=lambda: [layer])
    if kind == "native":
        assert state.warmup.warmup_native_fp8_gemms(model, 8) == 2
        assert [c.args[0].shape for c in launcher.call_args_list] == [(1, 64), (2, 64)]
        assert [c.kwargs["x_scale"].shape for c in launcher.call_args_list] == [
            (1, 2),
            (2, 2),
        ]
    else:
        assert state.warmup.warmup_v41_bmms(model, 8) == 2
        assert [c.args[0].shape for c in launcher.call_args_list] == [
            (1, 2, 64),
            (3, 2, 64),
        ]
        assert [c.args[2].shape for c in launcher.call_args_list] == [
            (1, 2, 2),
            (3, 2, 2),
        ]
        assert all(c.kwargs["dtype"] == torch.bfloat16 for c in launcher.call_args_list)
    assert all(c.args[0].dtype == torch.float8_e4m3fn for c in launcher.call_args_list)
    state.synchronize.assert_called_once()
    scales = [
        c.kwargs["x_scale"] if kind == "native" else c.args[2]
        for c in launcher.call_args_list
    ]
    assert all(scale.dtype == torch.float8_e8m0fnu for scale in scales)
    assert all((scale.view(torch.uint8) == 127).all() for scale in scales)
