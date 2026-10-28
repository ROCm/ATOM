"""Exercise the native DSpark writer against the owner's BF16 cache adapter."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def extracted(path, class_name, method=None):
    source = ROOT / path
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    node = (
        cls
        if method is None
        else next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method
        )
    )
    namespace = {"torch": torch}
    exec(  # noqa: S102 - execute the actual local consumer without GPU imports
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[node.name], namespace


def packet_class():
    return extracted(
        "atom/models/deepseek_v41/owner_service/owner_serving.py", "ContextPacket"
    )[0]


@pytest.mark.parametrize("fusion", ["none", "projection", "tail"])
@pytest.mark.parametrize("replay", [False, True])
def test_native_writer_preserves_stage_rows_and_bf16_packet(fusion, replay):
    writer, namespace = extracted(
        "atom/models/deepseek_v41/dspark.py", "DeepseekV41DSpark", "write_context_kv"
    )
    positions = torch.arange(5)
    aux = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
    tail = SimpleNamespace(width=2, replay=None, requests=(1,))
    step = SimpleNamespace(width=4, replay=None)
    if replay:
        step.replay = SimpleNamespace(tail=tail, tail_rows=torch.tensor([1, 3]))
    selected_step = tail if replay else step
    packet = packet_class()(selected_step)
    forward = SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=False),
        attn_metadata=SimpleNamespace(cache=packet, step=step),
    )
    selected = aux[[1, 3]] if replay else aux[:4]
    expected_positions = positions[[1, 3]] if replay else positions[:4]
    stages = torch.stack([selected + i * 32 for i in range(3)])
    calls = []

    def project(hidden, pos, rope, *, packed, stage):
        assert packed is False
        torch.testing.assert_close(hidden[0], selected)
        torch.testing.assert_close(pos[0], expected_positions)
        calls.append("project")
        return stages[stage : stage + 1]

    layers = [
        SimpleNamespace(
            spec=SimpleNamespace(layer_id=40 + i),
            kv_norm=SimpleNamespace(weight=None, eps=1e-6),
            project_context=lambda *a, stage=i, **kw: project(*a, stage=stage, **kw),
        )
        for i in range(3)
    ]

    class Rope:
        cos_cache = sin_cache = None

        def __call__(self, kv, pos):
            torch.testing.assert_close(pos, expected_positions)
            return kv

    def quantize(kv, *, dequantize):
        assert dequantize is True
        calls.append("quantize")
        return kv

    def fused_tail(kv, norm, pos, cos, sin, eps, *, packed):
        assert packed is False
        torch.testing.assert_close(pos, expected_positions)
        calls.append("tail")
        return kv[0].permute(1, 0, 2).contiguous()

    namespace.update(
        native_quant_linear=lambda *a, **kw: stages.permute(1, 0, 2).reshape(
            1, selected_step.width, 12
        ),
        rmsnorm2d_fwd_=lambda kv, *args: kv,
        quantize_fp8=quantize,
        fused_draft_kv_tail=fused_tail,
    )
    model = SimpleNamespace(
        project_context=lambda value: value,
        context_layers=layers,
        rope=Rope(),
        _context_kv_fusion=lambda: (
            None if fusion == "none" else (None, None, 1, 4, False)
        ),
        _draft_kv_tail_fusion=lambda: (None, 1e-6) if fusion == "tail" else None,
    )
    context_module = SimpleNamespace(get_forward_context=lambda: forward)
    with patch.dict("sys.modules", {"atom.utils.forward_context": context_module}):
        writer(model, aux, positions)
    assert (
        calls
        == {
            "none": ["project"] * 3,
            "projection": ["quantize"],
            "tail": ["tail"],
        }[fusion]
    )
    assert tuple(packet.keys) == (40, 41, 42)
    for i, value in enumerate(packet.keys.values()):
        assert isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16
        torch.testing.assert_close(value, stages[i : i + 1])


def test_packet_refuses_another_step_before_state_write():
    packet = packet_class()(object())
    with pytest.raises(ValueError, match="another owner step"):
        packet.write_window(40, torch.zeros(1), object())
    assert packet.keys == {}
