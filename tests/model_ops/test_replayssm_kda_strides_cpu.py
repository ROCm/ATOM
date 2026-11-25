# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exercise the production KDA wrapper and pointer expressions without a GPU."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from aiter_stub import atom_under_stubs

SOURCE = Path(__file__).resolve().parents[2] / "atom/model_ops/fla_ops/replayssm.py"
HV, D = 12, 128
FUSED_N = 4 * HV * D + HV + D


class TestKdaStrides(unittest.TestCase):
    def test_wrapper_copies_only_non_dense_rows(self):
        with atom_under_stubs() as load:
            replay = load("atom.model_ops.fla_ops.replayssm")
            for tokens in (32, 128):
                for layout in (
                    "dense_rows",
                    "gate_heads",
                    "gate_channels",
                    "beta_heads",
                ):
                    with self.subTest(tokens=tokens, layout=layout):
                        self._check_wrapper_layout(replay, tokens, layout)

    def _check_wrapper_layout(self, replay, tokens, layout):
        captured = {}

        class Kernel:
            def __getitem__(self, grid, target=captured):
                return target.update

        triton = SimpleNamespace(
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
            cdiv=lambda a, b: -(-a // b),
        )
        fused = torch.arange(tokens * FUSED_N, dtype=torch.bfloat16).view(
            tokens, FUSED_N
        )
        q, k, v = (
            fused[:, start * HV * D : (start + 1) * HV * D].view(1, tokens, HV, D)
            for start in range(3)
        )
        a = fused.as_strided(
            (1, tokens, HV, D),
            (
                tokens * FUSED_N,
                FUSED_N,
                2 * D if layout == "gate_heads" else D,
                2 if layout == "gate_channels" else 1,
            ),
            storage_offset=(2 if layout == "gate_heads" else 3) * HV * D,
        )
        b = fused.as_strided(
            (1, tokens, HV),
            (tokens * FUSED_N, FUSED_N, 2 if layout == "beta_heads" else 1),
            storage_offset=4 * HV * D,
        )
        with (
            patch.object(replay, "triton", triton),
            patch.object(replay, "_replayssm_kda_fwd_kernel", Kernel()),
        ):
            out = replay.replayssm_sigmoid_gating_delta_rule(
                q,
                k,
                v,
                a,
                b,
                torch.empty(HV),
                torch.empty(HV * D),
                torch.empty(16, HV, D, D),
                *(torch.empty(16, HV, 24, D, dtype=torch.bfloat16) for _ in range(3)),
                torch.zeros(16, dtype=torch.int32),
                torch.arange(tokens // 8, dtype=torch.int32),
                torch.arange(0, tokens + 1, 8, dtype=torch.int32),
                8,
            )
        for name, original in (("q", q), ("k", k), ("v", v), ("a", a), ("b", b)):
            self.assertTrue(torch.equal(captured[name], original), name)
        for name in ("q", "k", "v"):
            self.assertTrue(captured[name].is_contiguous(), name)
        self.assertEqual(
            captured["a"] is a,
            layout not in ("gate_heads", "gate_channels"),
        )
        self.assertEqual(captured["b"] is b, layout != "beta_heads")
        if layout in ("gate_heads", "gate_channels"):
            self.assertTrue(captured["a"].is_contiguous())
        if layout == "beta_heads":
            self.assertTrue(captured["b"].is_contiguous())
        self.assertEqual(captured["stride_a_token"], captured["a"].stride(-3))
        self.assertEqual(captured["stride_b_token"], captured["b"].stride(-2))
        self.assertEqual(captured["N"], tokens // 8)
        self.assertEqual(out.shape, (1, tokens, HV, D))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_sequence_starts_address_the_original_views(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        kernel = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_replayssm_kda_fwd_kernel"
        )
        for tokens in (32, 128):
            fused = np.arange(tokens * FUSED_N).reshape(tokens, FUSED_N)
            gate = fused[:, 3 * HV * D : 4 * HV * D].reshape(1, tokens, HV, D)
            beta = fused[:, 4 * HV * D : 4 * HV * D + HV][None]
            for name in ("p_a", "p_b"):
                with self.subTest(tokens=tokens, pointer=name):
                    assignment = next(
                        n
                        for n in ast.walk(kernel)
                        if isinstance(n, ast.Assign)
                        and any(
                            isinstance(t, ast.Name) and t.id == name for t in n.targets
                        )
                    )
                    program = compile(
                        ast.Module(body=[assignment], type_ignores=[]),
                        str(SOURCE),
                        "exec",
                    )
                    for bos in range(0, tokens, 8):
                        for head in (0, 5, 11):
                            for channel in ((0, 127) if name == "p_a" else (0,)):
                                values = {
                                    "a": 3 * HV * D,
                                    "b": 4 * HV * D,
                                    "bos": bos,
                                    "HV": HV,
                                    "K": D,
                                    "i_hv": head,
                                    "o_k": channel,
                                    "stride_a_token": gate.strides[1] // gate.itemsize,
                                    "stride_b_token": beta.strides[1] // beta.itemsize,
                                }
                                exec(program, values)  # noqa: S102
                                expected = (
                                    gate[0, bos, head, channel]
                                    if name == "p_a"
                                    else beta[0, bos, head]
                                )
                                self.assertEqual(fused.ravel()[values[name]], expected)


if __name__ == "__main__":
    unittest.main()
