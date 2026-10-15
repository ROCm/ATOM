# SPDX-License-Identifier: MIT
"""Exercise the decorator's metadata contract without a V4.1 dependency."""

from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.utils.attn_ffn_piecewise import piecewise_core
from atom.utils.piecewise_af import (
    PiecewiseAFRunner,
    PiecewiseMetadata,
    SharedGraphInputs,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")


def test_shared_inputs_refresh_and_isolate_stores_and_releases():
    inputs, memo = SharedGraphInputs(), {}
    values = {"activation": torch.ones(4), "offsets": torch.arange(4), "scale": 2}

    def stage():
        return inputs.prepare(values, memo=memo, copy_each_call=("activation",))

    first = stage()
    values["activation"].fill_(7)
    second = stage()
    assert first["activation"] is second["activation"]
    torch.testing.assert_close(second["activation"], torch.full((4,), 7.0))
    original = values["offsets"]
    values["offsets"] = original + 10
    torch.testing.assert_close(stage()["offsets"], original + 10)
    values["offsets"] = original
    torch.testing.assert_close(stage()["offsets"], original)
    original.add_(20)
    memo.clear()
    torch.testing.assert_close(stage()["offsets"], original)
    other = SharedGraphInputs().prepare(values, memo=memo)
    assert other["offsets"] is not first["offsets"]
    inputs.release_graphs()
    assert not inputs._slots
    original.add_(20)
    after_release = stage()
    assert after_release["offsets"] is not first["offsets"]
    torch.testing.assert_close(after_release["offsets"], original)


@dataclass
class Metadata:
    offsets: torch.Tensor
    scale: int = 1
    enabled: bool = True
    derived: dict = field(default_factory=dict)
    memo: dict = field(default_factory=dict)

    def piecewise_metadata(self):
        if not self.enabled:
            return None

        def rebuild(values):
            assert len(values["offsets"]) == 1
            offsets = values["offsets"].pop()
            return {
                "md": replace(self, offsets=offsets, scale=values["scale"], derived={})
            }

        def publish(arguments):
            self.derived.update(arguments["md"].derived)

        return PiecewiseMetadata(
            key=(),
            inputs={
                "offsets": [self.offsets],
                "scale": self.scale,
                "optional": None,
                "empty": (),
            },
            rebuild=rebuild,
            publish=publish,
            memo=self.memo,
            copy_each_call=(("offsets", 0),),
        )


class Layer:
    def __init__(self):
        self.count = torch.zeros(1, device="cuda")
        self.executions = 0

    @piecewise_core(
        capture_prefill=True,
        metadata=lambda layer, args: args["md"].piecewise_metadata(),
    )
    def core(self, x: torch.Tensor, md: Metadata, void: bool = False):
        self.executions += 1
        self.count.add_(1)
        if "output" not in md.derived:
            md.derived["output"] = (x + md.offsets) * md.scale
        return None if void else md.derived["output"] + 1


def context(**kwargs):
    return SimpleNamespace(
        in_hipgraph=False,
        context=SimpleNamespace(is_dummy_run=False),
        attn_metadata=SimpleNamespace(dummy=False),
        **kwargs,
    )


@cuda
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("void", [False, True])
def test_metadata_core_refreshes_replays_and_publishes_current_request(direct, void):
    layer, runner = Layer(), PiecewiseAFRunner()
    runner.max_entries = 2
    x = torch.zeros(4, device="cuda")
    previous = None
    for i in range(1, 5):
        x.fill_(i)
        md = Metadata(torch.full_like(x, 10 * i))
        actual = layer.core(
            x,
            md,
            void,
            piecewise=True,
            capture=True,
            runner=runner,
            forward_context=context(),
            zero_copy_inputs=("x",) if direct else (),
        )
        expected = x + 10 * i
        torch.testing.assert_close(md.derived["output"], expected)
        if void:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected + 1)
        assert layer.count.item() == i
        assert layer.executions == 2  # No Python math on replay.
        if previous is not None:
            assert previous.derived is not md.derived
        previous = md
    assert len(runner._graphs) == 1
    # Direct inputs with new producer addresses need new graphs; copied inputs do not.
    x = x.clone()
    layer.core(
        x,
        Metadata(x),
        void,
        piecewise=True,
        capture=True,
        runner=runner,
        zero_copy_inputs=("x",) if direct else (),
        forward_context=context(),
    )
    assert len(runner._graphs) == (2 if direct else 1)
    assert layer.count.item() == 5
    runner.max_entries = len(runner._graphs)
    slots = len(runner.inputs._slots)
    md = Metadata(x, scale=7)
    layer.core(
        x,
        md,
        void,
        piecewise=True,
        capture=True,
        runner=runner,
        zero_copy_inputs=(),
        forward_context=context(),
    )
    torch.testing.assert_close(md.derived["output"], 14 * x)
    assert layer.count.item() == 6
    assert len(runner.inputs._slots) == slots  # Cache-full fallback stages nothing.
    count = runner.release_graphs()
    assert count == (2 if direct else 1)
    assert not runner._captures and not runner.inputs._slots


@cuda
def test_host_values_layouts_and_per_call_metadata_values_are_keyed():
    layer, runner = Layer(), PiecewiseAFRunner()
    x = torch.ones(4, device="cuda")
    md = Metadata(torch.zeros(4, device="cuda"))
    for i, (scale, strided) in enumerate(
        [(1, False), (2, False), (1, False), (1, True)], 1
    ):
        md.scale = scale
        if strided:
            md.offsets = torch.zeros(8, device="cuda")[::2]
        md.offsets.fill_(i)
        md.derived.clear()
        actual = layer.core(
            x,
            md,
            piecewise=True,
            capture=True,
            runner=runner,
            forward_context=context(),
            zero_copy_inputs=(),
        )
        torch.testing.assert_close(actual, (x + i) * scale + 1)
        assert layer.count.item() == i
    assert len(runner._graphs) == 3
    runner.release_graphs()


@cuda
def test_interface_and_decorator_need_no_explicit_runner():
    layer = Layer()
    x = torch.ones(4, device="cuda")
    for i in range(3):
        actual = layer.core(
            x, Metadata(x + i), piecewise=True, capture=True, forward_context=context()
        )
        torch.testing.assert_close(actual, x * 2 + i + 1)
    assert layer.count.item() == 3 and layer.executions == 2
    stores = layer._piecewise_core_runners
    assert len(stores) == 1
    assert next(iter(stores.values())).release_graphs() == 1


@cuda
def test_shared_runner_isolates_methods_and_keys_scalar_config():
    @piecewise_core(
        capture_prefill=True,
        metadata=lambda layer, args: args["md"].piecewise_metadata(),
    )
    def other(layer, x: torch.Tensor, md: Metadata):
        return x * 10

    layer, runner = Layer(), PiecewiseAFRunner()
    x = torch.ones(4, device="cuda")
    kwargs = {
        "piecewise": True,
        "capture": True,
        "runner": runner,
        "forward_context": context(),
    }
    first = layer.core(x, Metadata(x), **kwargs)
    second = other(layer, x, Metadata(x), **kwargs)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, x * 3)
    torch.testing.assert_close(second, x * 10)
    assert layer.core(x, Metadata(x), void=True, **kwargs) is None
    assert layer.count.item() == 2
    assert runner.release_graphs() == 3
    actual = layer.core(x, Metadata(x * 2), **kwargs)
    torch.testing.assert_close(actual, x * 4)
    assert layer.count.item() == 3
    assert runner.release_graphs() == 1


@pytest.mark.parametrize(
    "gate",
    [
        "full",
        "plain",
        "nested",
        "dummy",
        "dummy_metadata",
        "runtime_full",
        "cap",
        "declined",
    ],
)
def test_ineligible_calls_do_not_prepare_or_publish(gate):
    seen = []

    def metadata(layer, arguments):
        seen.append("metadata")

    @piecewise_core(capture_prefill=True, metadata=metadata, max_tokens=4)
    def core(layer, x: torch.Tensor):
        seen.append("core")
        return x * 2

    fc = context()
    fc.in_hipgraph = gate == "nested"
    fc.context.is_dummy_run = gate == "dummy"
    fc.attn_metadata.dummy = gate == "dummy_metadata"
    if gate == "runtime_full":
        fc.cudagraph_runtime_mode = SimpleNamespace(name="FULL")
    runner = PiecewiseAFRunner()
    x = torch.ones(8 if gate == "cap" else 4)
    actual = core(
        SimpleNamespace(),
        x,
        piecewise=gate != "full",
        capture=gate != "plain",
        runner=runner,
        forward_context=fc,
    )
    torch.testing.assert_close(actual, x * 2)
    assert seen == (["metadata", "core"] if gate == "declined" else ["core"])
    assert not runner._graphs and not runner._captures and not runner.inputs._slots


@cuda
@pytest.mark.parametrize("failure", ["body", "rebuild", "tree"])
def test_failed_capture_never_installs_state_or_publishes(failure):
    publications = []

    def metadata(layer, arguments):
        return PiecewiseMetadata(
            key=(),
            inputs={"bad": object()} if failure == "tree" else {},
            rebuild=lambda values: (
                {"x": arguments["x"]} if failure == "rebuild" else {}
            ),
            publish=publications.append,
        )

    @piecewise_core(capture_prefill=True, metadata=metadata)
    def core(layer, x: torch.Tensor):
        raise RuntimeError("body failure")

    runner = PiecewiseAFRunner()
    error = {"body": RuntimeError, "rebuild": ValueError, "tree": TypeError}[failure]
    with pytest.raises(error):
        core(
            SimpleNamespace(),
            torch.ones(4, device="cuda"),
            piecewise=True,
            capture=True,
            runner=runner,
            forward_context=context(),
        )
    assert not runner._graphs and not runner._captures and not publications
