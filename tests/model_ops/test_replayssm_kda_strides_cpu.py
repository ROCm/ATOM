# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exercise ReplaySSM pool ownership, wrapper wiring and pointers without a GPU."""

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
    def test_sglang_ring_owns_scratch_before_capture(self):
        with atom_under_stubs() as load:
            plugin = load("atom.plugin.sglang.attention_backend.gdn_replayssm")
            temporal = torch.empty(7, HV, D, D)
            args = {
                "num_v_heads": HV,
                "head_k_dim": D,
                "head_v_dim": D,
                "max_query_len": 4,
                "activation": temporal,
            }
            with (
                patch.object(plugin, "install_sglang_replayssm_commit"),
                patch.object(
                    plugin, "replayssm_flush_wait", return_value="wait"
                ) as wait,
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                ) as capture,
            ):
                first = plugin.prepare_layer(0, temporal, **args)
                second = plugin.prepare_layer(1, temporal, **args)
                self.assertEqual(first[3].shape, (7, HV))
                self.assertEqual(first[3].dtype, torch.int32)
                self.assertFalse(first[3].any() or second[3].any())
                first[3][0, 0] = 1
                self.assertEqual(second[3][0, 0], 0)
                capture.return_value = True
                with patch.object(
                    torch,
                    "zeros",
                    side_effect=AssertionError("capture allocated scratch"),
                ):
                    again = plugin.prepare_layer(0, temporal, **args)
                self.assertIs(first[3], again[3])
                self.assertEqual(again[-1], "wait")
                self.assertEqual(wait.call_count, 1)
                plugin.reset_replay_runtime_for_tests()
                self.assertFalse(plugin.replay_runtime().bufs)

    def test_arrivals_are_owned_by_the_checkpoint_pool(self):
        with atom_under_stubs() as load:
            gdn = load("atom.model_ops.attentions.gdn_attn")
            kda = load("atom.model_ops.attentions.kimi_mla_gdn_attn")
            for cls in (
                gdn.GDNAttentionMetadataBuilder,
                kda.KimiAiterMLAGDNMetadataBuilder,
            ):
                for enabled in (False, True):
                    with self.subTest(builder=cls.__name__, enabled=enabled):
                        builder = object.__new__(cls)
                        builder.replayssm = enabled
                        builder.model_runner = SimpleNamespace(
                            config=SimpleNamespace(
                                torch_dtype=torch.bfloat16,
                                hf_config=SimpleNamespace(index_kpool=1),
                            )
                        )
                        builder._state_shape_for_runner = lambda: ((3, 8), (HV, 4, 4))
                        builder._state_dtypes = lambda: (torch.bfloat16, torch.float32)
                        builder.num_state_layers = lambda: 2
                        builder._replayssm_buffer_shapes = lambda enabled=enabled: (
                            ((HV, 8, 4), (HV, 8, 4), (HV, 8)) if enabled else None
                        )
                        builder._assert_checkpoint_geometry_still_holds = lambda: None
                        zeros = torch.zeros

                        def cpu_zeros(*args, zeros=zeros, **kwargs):
                            self.assertEqual(kwargs.pop("device"), "cuda")
                            return zeros(*args, **kwargs)

                        with (
                            patch.object(torch, "zeros", cpu_zeros),
                            patch.object(
                                gdn, "replayssm_flush_wait", return_value="wait"
                            ) as wait,
                        ):
                            pools = [
                                builder.allocate_per_req_cache(
                                    {gdn.STATE_SLOT_CLASS: 3}
                                )
                                for _ in range(2)
                            ]
                        self.assertEqual(wait.call_count, 2 if enabled else 0)
                        if enabled:
                            first, second = [
                                p["replayssm_flush_arrivals"] for p in pools
                            ]
                            self.assertEqual(first.shape, (2, 3, HV))
                            self.assertEqual(first.dtype, torch.int32)
                            self.assertFalse(first.any() or second.any())
                            first[0, 0, 0] = 1
                            self.assertEqual(second[0, 0, 0], 0)
                            actual = sum(
                                pools[0][name].numel() * pools[0][name].element_size()
                                for name in (
                                    "replayssm_buf_k",
                                    "replayssm_buf_u",
                                    "replayssm_buf_g",
                                    "replayssm_flush_arrivals",
                                )
                            )
                            self.assertEqual(
                                builder._replayssm_bytes_per_slot() * 3, actual
                            )
                        else:
                            self.assertNotIn("replayssm_flush_arrivals", pools[0])
                        for name, tensor in pools[0].items():
                            setattr(builder.model_runner, name, tensor)
                        module = SimpleNamespace(
                            layer_num=5, base_linear_attention=True
                        )
                        # Pool rows key modules by identity, as production does.
                        module = type("Layer", (), vars(module))()
                        builder._pool_rows = {gdn.LINEAR_STATE_ROWS: {module: 1}}
                        cache = builder.build_kv_cache_tensor(module)
                        if enabled:
                            self.assertEqual(
                                cache.replay_flush_arrivals.data_ptr(),
                                pools[0]["replayssm_flush_arrivals"][1].data_ptr(),
                            )
                            self.assertEqual(cache.replay_flush_wait, "wait")
                        else:
                            self.assertIsNone(cache.replay_flush_arrivals)
                            self.assertIsNone(cache.replay_flush_wait)

    def test_flush_wait_for_supported_targets(self):
        with atom_under_stubs() as load:
            replay = load("atom.model_ops.fla_ops.replayssm")
            for arch, expected in (
                ("gfx942", "s_waitcnt vmcnt(0)"),
                ("gfx950", "s_waitcnt vmcnt(0)"),
                ("gfx1200", "s_wait_loadcnt 0"),
                ("gfx1250", "s_wait_loadcnt 0"),
            ):
                with (
                    self.subTest(arch=arch),
                    patch.object(replay, "get_gfx_runtime", return_value=arch),
                ):
                    self.assertEqual(replay.replayssm_flush_wait(), expected)

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
        arrivals = torch.zeros(16, HV, dtype=torch.int32)
        with (
            patch.object(replay, "triton", triton),
            patch.object(replay, "_replayssm_kda_fwd_kernel", Kernel()),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(
                replay,
                "replayssm_flush_wait",
                side_effect=AssertionError("launch queried ISA"),
            ),
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
                flush_arrivals=arrivals,
                flush_wait="s_waitcnt vmcnt(0)",
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
        self.assertIs(captured["flush_arrivals"], arrivals)
        self.assertEqual(captured["WAIT"], "s_waitcnt vmcnt(0)")
        self.assertEqual(out.shape, (1, tokens, HV, D))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_gdn_wrapper_passes_pool_scratch(self):
        with atom_under_stubs() as load:
            replay = load("atom.model_ops.fla_ops.replayssm")
            for window in (1, 2, 4):
                with self.subTest(window=window):
                    captured = {}

                    class Kernel:
                        def __getitem__(self, grid, target=captured):
                            return target.update

                    n, heads, dim = 3, 16, 8
                    q, k = (torch.empty(1, n * window, 4, dim) for _ in range(2))
                    v = torch.empty(1, n * window, heads, dim)
                    g, beta = (torch.empty(1, n * window, heads) for _ in range(2))
                    arrivals = torch.zeros(7, heads, dtype=torch.int32)
                    out = torch.empty(n * window, heads, dim)
                    triton = SimpleNamespace(
                        next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
                        cdiv=lambda a, b: -(-a // b),
                    )
                    with (
                        patch.object(replay, "triton", triton),
                        patch.object(
                            replay, "_serial_launch_config", return_value=(4, 1, 3)
                        ),
                        patch.object(replay, "_replayssm_fwd_kernel", Kernel()),
                        patch.object(
                            replay,
                            "replayssm_flush_wait",
                            side_effect=AssertionError("launch queried ISA"),
                        ),
                        patch.object(
                            torch,
                            "zeros",
                            side_effect=AssertionError("launch allocated scratch"),
                        ),
                    ):
                        result = replay.replayssm_gated_delta_rule(
                            q,
                            k,
                            v,
                            g,
                            beta,
                            torch.empty(7, heads, dim, dim),
                            torch.empty(7, heads, 16, dim),
                            torch.empty(7, heads, 16, dim),
                            torch.empty(7, heads, 16),
                            torch.empty(7, dtype=torch.int32),
                            torch.arange(n, dtype=torch.int32),
                            torch.arange(
                                0, (n + 1) * window, window, dtype=torch.int32
                            ),
                            window,
                            o=out,
                            flush_arrivals=arrivals,
                            flush_wait="s_waitcnt vmcnt(0)",
                        )
                    self.assertIs(captured["flush_arrivals"], arrivals)
                    self.assertEqual(captured["WAIT"], "s_waitcnt vmcnt(0)")
                    self.assertEqual(captured["T_MAX"], window)
                    self.assertEqual(captured["N"], n)
                    self.assertEqual(result.data_ptr(), out.data_ptr())
                    for name, tensor in (
                        ("q", q),
                        ("k", k),
                        ("v", v),
                        ("g", g),
                        ("beta", beta),
                    ):
                        self.assertIs(captured[name], tensor)

    def test_fallback_record_stores_have_an_owns_kg_guard(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        for name in ("_replayssm_fwd_kernel", "_replayssm_kda_fwd_kernel"):
            kernel = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            aliases = {"buf_k", "buf_g"}
            assignments = [n for n in ast.walk(kernel) if isinstance(n, ast.Assign)]
            while True:
                extra = {
                    t.id
                    for n in assignments
                    for t in n.targets
                    if isinstance(t, ast.Name)
                    and any(
                        isinstance(v, ast.Name) and v.id in aliases
                        for v in ast.walk(n.value)
                    )
                }
                if extra <= aliases:
                    break
                aliases |= extra
            checked = []

            def visit(node, owned=False, aliases=aliases, name=name, checked=checked):
                if isinstance(node, ast.If):
                    visit(node.test, owned)
                    guarded = (
                        isinstance(node.test, ast.Name) and node.test.id == "owns_kg"
                    )
                    for child in node.body:
                        visit(child, owned or guarded)
                    for child in node.orelse:
                        visit(child, owned)
                    return
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "tl"
                    and node.func.attr == "store"
                    and any(
                        isinstance(v, ast.Name) and v.id in aliases
                        for v in ast.walk(node.args[0])
                    )
                ):
                    self.assertTrue(
                        owned, f"{name}:{node.lineno} record store lacks owns_kg"
                    )
                    checked.append(node)
                for child in ast.iter_child_nodes(node):
                    visit(child, owned)

            visit(kernel)
            self.assertTrue(checked, name)

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
