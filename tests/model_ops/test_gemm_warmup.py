# SPDX-License-Identifier: Apache-2.0
"""Ragged prefill must not discover a new tuned launcher during serving."""

import importlib


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
