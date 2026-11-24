# SPDX-License-Identifier: MIT
"""Graph-contract fixtures for every public plugin seam.

Real HIP graph/ReqGraph and real router; math is a small stand-in, not model weights.
Separate plans avoid region/child ownership ambiguity in the coverage matrix.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from atom import plugin_host as host
from atom import plugin_recipe as recipes
from atom.utils import forward_context
from atom.utils.region_graph import ReqGraph


def _v41_seams():
    """The V4.1 plugin model's seams (``atom.plugin_seams_v41``), not the host's own."""
    return {n: s for n, s in host.SEAMS.items() if n.startswith("v41.")}


DRAFT = "v41.draft.ctx"
CASES = (
    "v41.block.ffn",
    "v41.attn.out",
    "v41.attn.qk_norm",
    "v41.moe.shared_act",
    "v41.moe.topk",
    "v41.op.quant_fp8_g32",
    "v41.attn.index_topk",
    "v41.attn.attend",
    DRAFT,
)


# Called once at boot, before any capture (ATOM captures the installed model afterwards):
# there is no graph contract to exercise.
BOOT_ONLY = "v41.mono.install"
# The spans the plugin model adds for plugin_poc's fused providers; their call
# sites and nesting are CPU-tested in test_plugin_modules.py, their kernels'
# capture by the providers' own pad-M gates.
EXTENDED = {
    "v41.moe.shared_expert",
    "v41.attn.project_out",
    "v41.attn.out_full",
    "v41.attn.attend_out",
    "v41.draft.markov",
    "v41.spec.verify_sample",
}


def test_capture_contract_catalog_is_complete():
    captured = {n: s for n, s in _v41_seams().items() if n != BOOT_ONLY}
    assert set(captured) == set(CASES) | EXTENDED
    assert host.SEAMS[BOOT_ONLY].graph == "eager"
    assert all(s.graph in {"capture_safe", "opaque_op"} for s in captured.values())


def _args(seam, x, scratch_i, scratch_f):
    cases = {
        "v41.block.ffn": ((x,), {}),
        "v41.attn.out": ((x, None, x, None), {"dtype": x.dtype}),
        "v41.attn.qk_norm": ((x, x, x[0], x[0], 1e-5), {"eps_k": 1e-5}),
        "v41.moe.shared_act": ((x, None), {}),
        "v41.moe.topk": (
            (scratch_f, scratch_i, x, None, True, 1.0),
            {"score_func": "softmax"},
        ),
        "v41.op.quant_fp8_g32": ((x,), {"dequantize": False}),
        "v41.attn.index_topk": (
            (x, scratch_i[:, 0], scratch_i[:, 0], scratch_i, 4, 8, 1),
            {"k": 8},
        ),
        "v41.attn.attend": (
            (x, x, scratch_i, scratch_i, x, scratch_i, scratch_i, x[0], 1.0),
            {"out": scratch_f},
        ),
        DRAFT: (([x, x], None, 4), {}),
    }
    return cases[seam]


def _math(seam, candidate, *args, **kw):
    x = args[2] if seam == "v41.moe.topk" else args[0]
    if seam == DRAFT:
        x = x[0]
    y = torch.add(x, x) if candidate else x * 2
    if seam == "v41.moe.topk":
        args[0].copy_(y)
        args[1].copy_(y.to(torch.int32))
        return None
    if seam == "v41.attn.index_topk":
        args[3].copy_(y.to(torch.int32))
        return None
    if seam == "v41.attn.attend":
        return kw["out"].copy_(y)
    count = len(host.SEAMS[seam].returns)
    return y if count == 1 else (y, *[y + i for i in range(1, count)])


def _install(seam, monkeypatch, mark):
    owner = "graph-fixture"
    p = recipes.resolve(
        [
            {
                "schema": recipes.RECIPE_SCHEMA,
                "strategy": "capture-fixture",
                "model_arch": "DeepseekV41ForCausalLM",
                "graph": {"mode": "capture", "buckets": [4]},
                "plugins": [{"name": owner, "serves": [seam]}],
                "plan": {"seams": {seam: [owner]}},
                "defaults": {"on_decline": "error"},
            }
        ]
    )

    def register(h):
        def candidate(*args, capability, valid_rows, **kwargs):
            assert capability.captured and capability.bucket == 4
            mark.fill_(7)  # Device-side proof that replay contains candidate work.
            return _math(seam, True, *args, **kwargs)

        h.register(
            seam=seam,
            owner=owner,
            contract=host.SEAMS[seam].contract,
            graph="capture_safe",
            impl=candidate,
        )

    monkeypatch.setattr(
        forward_context, "get_forward_context", lambda: SimpleNamespace(context=None)
    )
    monkeypatch.setattr(host, "_configure_aiter", lambda providers: None)
    monkeypatch.setattr(host, "_valid_rows", lambda rows: None)
    monkeypatch.setattr(
        host, "_entry_points", lambda: {owner: SimpleNamespace(load=lambda: register)}
    )
    host._reset_for_testing()
    host.configure(p, kill_switch=False)

    def native(*args, **kwargs):
        return _math(seam, False, *args, **kwargs)

    if host.SEAMS[seam].form == "bind":
        if seam == "v41.block.ffn":
            return host.bind(seam, native, layer="graph-ffn", fake=torch.empty_like)
        bound = host.bind(seam, native)
        if seam == "v41.moe.shared_act":
            return host.span_op(
                seam,
                "graph-shared",
                lambda x, w: bound(x, w)[0] + 3,
                lambda x, w: torch.empty_like(x),
            )
        return bound
    return host.site(seam)(native)


GPU_GRAPH = pytest.mark.skipif(
    os.environ.get("ATOM_TOGETHER_GPU_TEST") != "1" or not torch.cuda.is_available(),
    reason="opt-in ROCm/CUDA capture",
)


@GPU_GRAPH
@pytest.mark.parametrize("seam", CASES)
def test_real_graph_all_seams(seam, monkeypatch):
    mark = torch.zeros(1, dtype=torch.int32, device="cuda")
    fn = _install(seam, monkeypatch, mark)
    try:
        # Use the engine's ReqGraph, including fixed-address replay inputs.
        graphs = {}
        x = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8)
        s_i = torch.zeros_like(x, dtype=torch.int32)
        s_f = torch.zeros_like(x)

        def forward(inp):
            args, kw = _args(seam, inp, s_i, s_f)
            out = fn(*args, **kw)
            if seam == "v41.moe.topk":
                return s_f, s_i
            if seam == "v41.attn.index_topk":
                return s_i
            return out

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), host.capture_bucket(4):
            for arm in ("native", "plugin"):
                host.set_arm(arm)
                forward(x)
                g = ReqGraph()
                g.record(forward, (x,), key=4)
                graphs[arm] = g
        torch.cuda.current_stream().wait_stream(stream)
        assert host.route_proof(seam, "graph-fixture", 2) == 2
        frozen_counts = host.decision_counts()
        host.set_arm("native")
        for delta in (1, 7):
            live = x + delta
            outs = []
            for arm in ("native", "plugin"):
                mark.zero_()
                out = graphs[arm].replay((live,), key=4)
                if not isinstance(out, tuple):
                    out = (out,)
                outs.append(tuple(v.clone() for v in out))
                assert mark.item() == (7 if arm == "plugin" else 0)
            assert len(outs[0]) == len(outs[1])
            assert all(torch.equal(a, b) for a, b in zip(*outs))
            delta_y = 3 if seam == "v41.moe.shared_act" else 0
            assert torch.equal(outs[0][0], (live * 2 + delta_y).to(outs[0][0].dtype))
        assert host.decision_counts() == frozen_counts
        print(
            "PLUGIN_GRAPH_EVIDENCE "
            + json.dumps(
                {
                    "seam": seam,
                    "captured_hits": 1,
                    "replays": 4,
                    "device_candidate_marker": True,
                    "redispatch": 0,
                    "scope": "seam-fixture",
                }
            ),
            flush=True,
        )
        torch.cuda.synchronize()
    finally:
        host._reset_for_testing()
