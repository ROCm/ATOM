# SPDX-License-Identifier: MIT
"""Schema behavior without importing any model or its metadata types."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.utils.attn_ffn_piecewise import piecewise_core
from atom.utils.piecewise_af import _flatten, _unflatten
from atom.utils.piecewise_metadata import PiecewiseMetadataSchema


@dataclass(frozen=True)
class Rows:
    offsets: torch.Tensor
    derived: dict = field(default_factory=dict)
    selected: dict = field(default_factory=dict)
    plans: dict = field(default_factory=dict)


def declaration():
    return {
        "bindings": {
            "group": "layer.group",
            "plan": lambda s: s["md"].plans.get(s["group"]),
        },
        "key": ("layer.enabled",),
        "when": "layer.enabled",
        "objects": {
            "plan": {"inputs": ("tensor",)},
            "md": {
                "inputs": ("offsets",),
                "select": {"selected": {"keys": ("group",)}},
                "reset": {"derived": {"nested": []}},
                "bind": {"plans": {"object": "plan", "key": "group"}},
            },
            "subset": {"source": "query", "template": "md", "inputs": ("offsets",)},
            "query": {"inputs": (), "bind": {"view": "subset"}},
        },
        "arguments": ("md", "query"),
        "publish": {"md.selected": {"keys": ("layer.output",)}, "md.derived": "update"},
    }


def fresh_inputs(inputs):
    leaves = {}
    structure = _flatten(inputs, leaves, ("metadata",))
    leaves = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in leaves.items()
    }
    return _unflatten(structure, leaves)


@pytest.mark.parametrize("has_plan", [False, True])
@pytest.mark.parametrize("has_query", [False, True])
def test_fresh_views_explicit_aliases_and_scoped_publication(has_plan, has_query):
    layer = SimpleNamespace(group=3, output=7, enabled=True)
    upstream, unrelated = torch.tensor([10]), torch.tensor([20])
    md = Rows(
        torch.arange(4),
        derived={"old": 1},
        selected={3: upstream, 11: unrelated},
        plans={3: {"tensor": torch.ones(4)}} if has_plan else {},
    )
    query = {"offsets": torch.arange(2)} if has_query else None
    schema = PiecewiseMetadataSchema(**declaration())
    metadata = schema(layer, {"md": md, "query": query})
    warm = metadata.rebuild(fresh_inputs(metadata.inputs))
    warm["md"].derived["nested"].append(1)
    warm["md"].selected[7] = torch.tensor([30])
    recorded = metadata.rebuild(fresh_inputs(metadata.inputs))
    assert recorded["md"].derived == {"nested": []}
    assert 7 not in recorded["md"].selected
    assert recorded["md"].offsets.data_ptr() != md.offsets.data_ptr()
    assert recorded["md"].plans.keys() == md.plans.keys()
    if has_plan:
        assert (
            recorded["md"].plans[3]["tensor"].data_ptr()
            != md.plans[3]["tensor"].data_ptr()
        )
    if has_query:
        view = recorded["query"]["view"]
        assert view.selected is recorded["md"].selected
        assert view.derived is recorded["md"].derived
        assert view.offsets.shape == (2,)
    else:
        assert recorded["query"] is None
    metadata.publish(warm)
    assert md.selected[3] is upstream and md.selected[11] is unrelated
    assert md.selected[7] is warm["md"].selected[7]
    # Rebind publication to a different live request, with no capture-time closure.
    current = Rows(torch.arange(4), selected={3: upstream, 11: unrelated})
    schema(layer, {"md": current, "query": None}).publish(recorded)
    assert 7 not in current.selected
    assert current.derived == {"nested": []}
    assert md.derived["nested"] == [1]
    layer.output = 8
    assert schema(layer, {"md": md, "query": query}).key != metadata.key
    layer.enabled = False
    assert schema(layer, {"md": md, "query": query}) is None


def test_dictionary_filters_and_conditional_field_writes():
    schema = PiecewiseMetadataSchema(
        objects={
            "md": {
                "select": {"buffers": {"suffix": "layer.suffix"}},
                "reset": {"result": None, "count": 0},
            }
        },
        arguments=("md",),
        publish={"md.result": "if_not_none", "md.count": "replace"},
    )
    md = {
        "buffers": {"z_2": torch.ones(1), "unused_1": object(), "a_2": 2},
        "result": "keep",
        "count": 7,
    }
    metadata = schema(SimpleNamespace(suffix="_2"), {"md": md})
    assert list(metadata.inputs["md"]["buffers"]) == ["a_2", "z_2"]
    captured = metadata.rebuild(fresh_inputs(metadata.inputs))
    metadata.publish(captured)
    assert md["result"] == "keep" and md["count"] == 0
    captured["md"]["result"] = "new"
    metadata.publish(captured)
    assert md["result"] == "new"


@pytest.mark.parametrize(
    "objects,error",
    [
        ({"md": {"input": ("offsets",)}}, ValueError),
        ({"md": {"inputs": "offsets"}}, TypeError),
        ({"md": {"inputs": ("offsets",), "reset": {"offsets": None}}}, ValueError),
        ({"md": {"bind": {"view": "later"}}, "later": {}}, ValueError),
        ({"md": {"template": "md"}}, ValueError),
        (
            {"md": {"select": {"selected": {"keys": (), "suffix": "layer.suffix"}}}},
            ValueError,
        ),
        ({"md": {"reset": {"tensor": torch.ones(1)}}}, TypeError),
    ],
)
def test_invalid_declarations_fail_before_capture(objects, error):
    with pytest.raises(error):
        PiecewiseMetadataSchema(objects=objects, arguments=("md",))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")
def test_decorator_schema_replays_mutable_inputs_without_model_callbacks():
    class Layer:
        def __init__(self):
            self.calls = 0
            self.counter = torch.zeros(1, device="cuda")

        @piecewise_core(
            capture_prefill=True,
            metadata={
                "objects": {"md": {"inputs": ("offsets",), "reset": {"derived": {}}}},
                "arguments": ("md",),
                "publish": {"md.derived": "update"},
                "memo": "md.memo",
                "copy_each_call": (("md", "offsets"),),
            },
        )
        def core(self, x: torch.Tensor, md: object):
            self.calls += 1
            self.counter.add_(1)
            if "result" not in md.derived:
                md.derived["result"] = x + md.offsets
            return md.derived["result"] * 2

    layer = Layer()
    x = torch.ones(4, device="cuda")
    md = SimpleNamespace(offsets=x.clone(), derived={}, memo={})
    for i in range(4):
        md.offsets.fill_(i)
        md.derived.clear()
        result = layer.core(x, md, piecewise=True, capture=True, zero_copy_inputs=())
        torch.testing.assert_close(result, (x + i) * 2)
        torch.testing.assert_close(md.derived["result"], x + i)
        assert layer.counter.item() == i + 1
    assert layer.calls == 2
    assert next(iter(layer._piecewise_core_runners.values())).release_graphs() == 1
