# SPDX-License-Identifier: MIT
"""Shared AF contracts with model-independent inputs and capture state."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.utils.attn_ffn_piecewise import piecewise_core
from atom.utils.piecewise_af import PiecewiseAFRunner, SharedGraphInputs


def test_shared_inputs_refresh_per_call_and_per_forward():
    inputs, memo = SharedGraphInputs(), {}
    values = {"activation": torch.ones(4), "offsets": torch.arange(4), "scale": 2}

    def stage():
        return inputs.prepare(values, memo=memo, copy_each_call=("activation",))

    first = stage()
    values["activation"].fill_(7)
    second = stage()
    assert first["activation"].data_ptr() == second["activation"].data_ptr()
    torch.testing.assert_close(second["activation"], torch.full((4,), 7.0))
    # Reusing a semantic input name for another source must invalidate its memo.
    original = values["offsets"]
    values["offsets"] = original + 10
    torch.testing.assert_close(stage()["offsets"], original + 10)
    values["offsets"] = original
    torch.testing.assert_close(stage()["offsets"], original)
    original.add_(20)
    memo.clear()
    third = stage()
    torch.testing.assert_close(third["offsets"], original)
    assert third["scale"] == 2
    assert third["offsets"].data_ptr() == first["offsets"].data_ptr()
    inputs.release_graphs()
    assert not inputs._slots


class _State:
    def __init__(self, inputs):
        self.inputs = inputs

    @contextmanager
    def bind(self):
        yield {"offsets": self.inputs["offsets"]}


def test_failed_or_skipped_capture_does_not_keep_state():
    runner = PiecewiseAFRunner()
    sources = {"x": torch.ones(4), "offsets": torch.ones(4)}
    with (
        pytest.raises(RuntimeError, match="failed"),
        runner.bind_capture((), sources, _State),
    ):
        raise RuntimeError("failed")
    with runner.bind_capture((), sources, _State):
        pass  # The decorated core can decline capture, too.
    assert not runner._captures and not runner._graphs
    runner.max_entries = 0
    runner.inputs.release_graphs()
    with runner.bind_capture((), sources, _State) as binding:
        assert binding is None
    assert not runner.inputs._slots


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")
@pytest.mark.parametrize("direct", [False, True])
def test_generic_core_replays_changed_inputs_and_bounds_new_shapes(direct):
    @piecewise_core(capture_prefill=True)
    def core(layer, *, x: torch.Tensor, offsets: object):
        layer.count.add_(1)
        return x + offsets

    runner = PiecewiseAFRunner()
    runner.max_entries = 2
    layer = SimpleNamespace(count=torch.zeros(1, device="cuda"))
    context = SimpleNamespace(
        in_hipgraph=False, context=SimpleNamespace(is_dummy_run=False)
    )
    x = torch.zeros(4, device="cuda")
    memo = {}
    for i in range(1, 5):
        x.fill_(i)
        offsets = torch.full_like(x, 10 * i)
        sources = {"x": x, "offsets": offsets}
        memo.clear()
        with runner.bind_capture(
            (),
            sources,
            _State,
            memo=memo,
            copy_each_call=("x",),
            zero_copy=("x",) if direct else (),
        ) as binding:
            assert binding.first_call == (i == 1)
            result = core(
                layer,
                x=binding.inputs["x"],
                piecewise=True,
                forward_context=context,
                **binding.arguments,
            )
        torch.testing.assert_close(result, x + offsets)
        assert layer.count.item() == i
    assert len(runner._graphs) == 1
    # Changed producer address requires a new recording only on the direct path.
    sources["x"] = x.clone()
    with runner.bind_capture(
        (),
        sources,
        _State,
        zero_copy=("x",) if direct else (),
    ) as binding:
        assert binding.first_call == direct
        core(
            layer,
            x=binding.inputs["x"],
            piecewise=True,
            forward_context=context,
            **binding.arguments,
        )
    assert layer.count.item() == 5
    assert len(runner._graphs) == (2 if direct else 1)
    runner.max_entries = len(runner._graphs)
    with runner.bind_capture(("new branch",), sources, _State) as binding:
        assert binding is None
    assert runner.release_graphs() == (2 if direct else 1)
    assert not runner._captures
