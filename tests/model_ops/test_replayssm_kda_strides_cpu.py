# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exercise the production KDA wrapper and pointer expressions without a GPU."""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "atom/model_ops/fla_ops/replayssm.py"
HV, D = 12, 128
FUSED_N = 4 * HV * D + HV + D


class Tensor:
    def __init__(self, shape, strides=None):
        self.shape = shape
        self.dense = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
        self.strides = self.dense if strides is None else strides
        self.dtype = "bf16"
        self.contiguous_calls = 0

    def stride(self, axis):
        return self.strides[axis]

    def contiguous(self):
        self.contiguous_calls += 1
        return self if self.strides == self.dense else Tensor(self.shape)

    def new_empty(self, *shape):
        return Tensor(shape)

    def numel(self):
        return math.prod(self.shape)


def production_function(name, namespace):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    definition = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    definition.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, definition], type_ignores=[])
    )
    # Execute only the checked-in production definition.
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102
    return namespace[name]


class TestKdaStrides(unittest.TestCase):
    def test_arrivals_are_owned_by_the_checkpoint_pool(self):
        capturing = False
        torch = SimpleNamespace(
            int32=np.int32,
            zeros=lambda size, **kw: np.zeros(size, dtype=kw["dtype"]),
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing),
        )
        run = production_function(
            "_flush_arrivals", {"torch": torch, "_FLUSH_ARRIVALS": {}}
        )

        def pool(ptr):
            return SimpleNamespace(
                device="cuda:0", shape=(8, 12, 128, 128), data_ptr=lambda: ptr
            )

        first, second = run(pool(100)), run(pool(200))
        self.assertIsNot(first, second)
        self.assertEqual([x.shape for x in (first, second)], [(96,), (96,)])
        self.assertFalse(first.any() or second.any())
        first[0] = 1
        self.assertEqual(second[0], 0)
        capturing = True
        self.assertIs(run(pool(100)), first)
        with self.assertRaisesRegex(RuntimeError, "eagerly"):
            run(pool(300))

    def test_flush_wait_for_supported_targets(self):
        for arch, expected in (
            ("gfx942", "s_waitcnt vmcnt(0)"),
            ("gfx950", "s_waitcnt vmcnt(0)"),
            ("gfx1200", "s_wait_loadcnt 0"),
            ("gfx1250", "s_wait_loadcnt 0"),
        ):
            with self.subTest(arch=arch):
                run = production_function(
                    "_flush_wait", {"get_gfx_runtime": lambda arch=arch: arch}
                )
                self.assertEqual(run(), expected)

    def test_wrapper_copies_only_non_dense_rows(self):
        for tokens in (32, 128):
            for layout in ("dense_rows", "gate_heads", "gate_channels", "beta_heads"):
                with self.subTest(tokens=tokens, layout=layout):
                    captured = {}

                    class Kernel:
                        def __getitem__(self, grid, target=captured):
                            return target.update

                    namespace = {
                        "triton": SimpleNamespace(
                            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
                            cdiv=lambda a, b: -(-a // b),
                        ),
                        "seed_forked_slots": lambda *args: None,
                        "flush_threshold_ok": lambda cap, length: cap >= 2 * length,
                        "_replay_dot_mode": lambda dtype: "ieee",
                        "_flush_arrivals": lambda ckpt: object(),
                        "_flush_wait": lambda: "s_waitcnt vmcnt(0)",
                        "_replayssm_kda_fwd_kernel": Kernel(),
                    }
                    run = production_function(
                        "replayssm_sigmoid_gating_delta_rule", namespace
                    )
                    q, k, v = (Tensor((1, tokens, HV, D)) for _ in range(3))
                    a = Tensor(
                        (1, tokens, HV, D),
                        (
                            tokens * FUSED_N,
                            FUSED_N,
                            2 * D if layout == "gate_heads" else D,
                            2 if layout == "gate_channels" else 1,
                        ),
                    )
                    b = Tensor(
                        (1, tokens, HV),
                        (tokens * FUSED_N, FUSED_N, 2 if layout == "beta_heads" else 1),
                    )
                    run(
                        q,
                        k,
                        v,
                        a,
                        b,
                        Tensor((HV,)),
                        Tensor((HV * D,)),
                        Tensor((16, HV, D, D)),
                        *(Tensor((16, HV, 24, D)) for _ in range(3)),
                        Tensor((16,)),
                        Tensor((tokens // 8,)),
                        Tensor((tokens // 8 + 1,)),
                        8,
                    )
                    self.assertEqual([x.contiguous_calls for x in (q, k, v)], [1, 1, 1])
                    self.assertEqual(
                        captured["a"] is a,
                        layout not in ("gate_heads", "gate_channels"),
                    )
                    self.assertEqual(captured["b"] is b, layout != "beta_heads")
                    self.assertEqual(
                        captured["stride_a_token"], captured["a"].stride(-3)
                    )
                    self.assertEqual(
                        captured["stride_b_token"], captured["b"].stride(-2)
                    )
                    self.assertEqual(captured["N"], tokens // 8)
                    self.assertEqual(captured["WAIT"], "s_waitcnt vmcnt(0)")

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
