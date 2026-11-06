"""CPU contracts for owner inputs crossing the upstream route/expert boundary."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class Method:
    def __init__(self):
        self.use_triton = self.use_triton_ep = False
        self.selections = 0

    def select_experts(self, layer, hidden, logits):
        self.selections += 1
        return "native_weights", "native_ids"


def tensor(shape, dtype):
    return SimpleNamespace(
        shape=shape, dtype=dtype, device="cpu", is_contiguous=lambda: True
    )


def owner():
    source = Path(__file__).resolve().parents[3] / "atom/model_ops/moe.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "FusedMoE"
    )
    selected = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"forward_impl", "_validate_preselected_routes"}
    ]
    namespace = {
        "torch": SimpleNamespace(Tensor=object, float32="float32", int32="int32"),
        "dtypes": SimpleNamespace(fp8="fp8", fp8_e8m0="e8m0"),
        "Mxfp4MoEMethod": Method,
        "get_dp_group": lambda: SimpleNamespace(world_size=1),
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)  # noqa: S102 - local AST
    binding = type("Owner", (), {node.name: namespace[node.name] for node in selected})
    obj = binding()
    obj.quant_method = Method()
    obj.supports_prequantized_input = lambda: True
    obj.tp_size, obj.ep_size, obj.dp_size = 1, 4, 1
    obj.moe_parallel_config = SimpleNamespace(
        dp_logical_ratio=1, use_all2all_kernels=False, pcp_merge=False
    )
    obj.num_fused_shared_experts, obj.top_k, obj.renormalize = 0, 6, True
    obj.reduce_results = False
    obj.calls = []

    def consume(hidden, weights, ids, **kwargs):
        obj.calls.append((hidden, weights, ids, kwargs))
        return "output"

    obj._run_experts = consume
    return obj


class OwnerMoERebaseTests(unittest.TestCase):
    def test_native_routing_still_uses_upstream_expert_entry(self):
        layer = owner()
        hidden = tensor((64, 5120), "bf16")
        self.assertEqual(layer.forward_impl(hidden, "logits"), "output")
        self.assertEqual(layer.quant_method.selections, 1)
        self.assertEqual(layer.calls[0][1:3], ("native_weights", "native_ids"))
        self.assertEqual(layer.calls[0][3], {"x_scale": None, "gathered": False})

    def test_packet_and_scale_reach_same_entry_without_second_routing(self):
        layer = owner()
        hidden = tensor((7835, 5120), "fp8")
        scale = tensor((7835, 160), "e8m0")
        routes = tensor((7835, 6), "float32"), tensor((7835, 6), "int32")
        self.assertEqual(
            layer.forward_impl(hidden, None, x_scale=scale, preselected_routes=routes),
            "output",
        )
        self.assertEqual(layer.quant_method.selections, 0)
        self.assertEqual(len(layer.calls), 1)
        self.assertIs(layer.calls[0][1], routes[0])
        self.assertIs(layer.calls[0][2], routes[1])
        self.assertIs(layer.calls[0][3]["x_scale"], scale)

    def test_wrong_scale_is_rejected_before_expert_launch(self):
        layer = owner()
        for scale in (tensor((64, 159), "e8m0"), tensor((64, 160), "float32")):
            with self.assertRaises(ValueError):
                layer.forward_impl(tensor((64, 5120), "fp8"), "logits", x_scale=scale)
        self.assertEqual(layer.calls, [])

    def test_packet_cannot_bypass_parallel_or_route_shape_contract(self):
        hidden = tensor((64, 5120), "fp8")
        for pcp, rows in ((True, 64), (False, 63)):
            layer = owner()
            layer.moe_parallel_config.pcp_merge = pcp
            routes = tensor((rows, 6), "float32"), tensor((rows, 6), "int32")
            with self.assertRaises(ValueError):
                layer.forward_impl(hidden, None, preselected_routes=routes)
            self.assertEqual(layer.calls, [])


if __name__ == "__main__":
    unittest.main()
