# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The plugin host (`atom/plugin_host.py`) and its recipes (`atom/plugin_recipe.py`).

CPU only: plugins are stand-in entry points, tensors are CPU tensors, and the
graph state a CUDA-graph capture would report is patched in.
"""

import json
import sys
import types

import pytest
import torch

from atom import plugin_host as host
from atom import plugin_recipe as recipes

OUT = "v41.attn.out"
QUANT = "v41.op.quant_fp8_g32"
FFN = "v41.block.ffn"
ACT = "v41.moe.shared_act"
CONFIGURE_AITER = host._configure_aiter  # before the fixture stands it in
GRAPH_STATE = host._graph_state


class _EntryPoint:
    def __init__(self, register):
        self._register = register

    def load(self):
        return self._register


@pytest.fixture(autouse=True)
def fresh_host(monkeypatch):
    """Default deny, no entry points, AITER left alone, eager graph state."""
    aiter = []
    monkeypatch.setattr(host, "_entry_points", dict)
    monkeypatch.setattr(host, "_configure_aiter", lambda names: aiter.append(names))
    monkeypatch.setattr(host, "_graph_state", lambda: (False, None))
    monkeypatch.setattr(host, "_valid_rows", lambda rows: None)
    host._reset_for_testing()
    yield aiter
    host._reset_for_testing()


def _pair(*args, **kwargs):
    return "y", "scale"


def _plugins(monkeypatch, **registers):
    eps = {name: _EntryPoint(fn) for name, fn in registers.items()}
    monkeypatch.setattr(host, "_entry_points", lambda: eps)


def _offer(seam, owner, impl=_pair, **kwargs):
    """An entry point offering ``owner`` for ``seam``."""

    def register(h):
        h.register(
            seam=seam,
            contract=h.SEAMS[seam].contract,
            impl=impl,
            owner=owner,
            **kwargs,
        )

    return register


def _recipe(seams, *, plugins=None, mode="any", buckets=None, **extra):
    owners = {o for os_ in seams.values() for o in os_}
    serves = {o: [s for s, os_ in seams.items() if o in os_] for o in owners}
    graph = {"mode": mode}
    if buckets is not None:
        graph["buckets"] = buckets
    return {
        "schema": recipes.RECIPE_SCHEMA,
        "model_arch": "DeepseekV41ForCausalLM",
        "strategy": "tp4-c8",
        "graph": graph,
        "plugins": plugins
        or [{"name": o, "kind": "layer", "serves": serves[o]} for o in sorted(owners)],
        "plan": {"seams": seams},
        **extra,
    }


x = torch.zeros(8, 2, 64)


# -- registration ---------------------------------------------------------


def test_register_accepts_a_planned_owner(monkeypatch):
    _plugins(monkeypatch, bmm=_offer(OUT, "bmm"))
    host.configure(["bmm"], kill_switch=False)
    assert host.owners(OUT) == ("bmm",)
    assert host.status()["seams"] == {OUT: {"tables": {"any": ("bmm",)}}}


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"seam": "v41.attn.nope"}, "unknown-seam"),
        ({"contract": 9}, "contract-mismatch"),
        ({"requires_api": ">=3"}, "api-mismatch"),
        ({"requires_api": "nonsense"}, "api-mismatch"),
        ({"graph": "opaque_op"}, "graph-unsupported"),
        ({"select": "yes"}, "not-callable"),
        ({"owner": "other"}, "owner-mismatch"),
    ],
)
def test_register_refuses_with_a_reason(monkeypatch, kwargs, reason):
    caught = []

    def register(h):
        args = {
            "seam": OUT,
            "contract": 1,
            "impl": _pair,
            "owner": "bmm",
            **kwargs,
        }
        try:
            h.register(**args)
        except h.RegistrationRefused as refused:
            caught.append(refused.reason)

    _plugins(monkeypatch, bmm=register)
    host.configure(["bmm"], kill_switch=False)
    assert caught == [reason]
    assert host.status()["refusals"][-1].reason == reason
    assert host.status()["seams"] == {}


def test_an_owner_registers_a_seam_once(monkeypatch):
    caught = []

    def twice(h):
        _offer(OUT, "bmm")(h)
        try:
            _offer(OUT, "bmm")(h)
        except h.RegistrationRefused as refused:
            caught.append(refused.reason)

    _plugins(monkeypatch, bmm=twice)
    host.configure(["bmm"], kill_switch=False)
    assert caught == ["duplicate"] and host.owners(OUT) == ("bmm",)


def test_default_deny_and_unplanned_seams_register_nothing(monkeypatch):
    done = host.register(seam=OUT, contract=1, impl=_pair, owner="bmm")
    assert (done.accepted, done.reason) == (False, "not-enabled")

    def both(h):
        _offer(OUT, "bmm")(h)
        _offer(QUANT, "bmm", impl=lambda x, **k: (x, x))(h)

    _plugins(monkeypatch, bmm=both)
    host.configure(
        recipes.resolve([_recipe({OUT: ["bmm"]}, plugins=[_serves("bmm", OUT)])]),
        kill_switch=False,
    )
    assert host.status()["refusals"][-1].reason == "not-planned"


def _serves(name, *seams, kind="layer"):
    return {"name": name, "kind": kind, "serves": list(seams)}


def test_kill_switch_wins_and_disables_aiter_too(monkeypatch, fresh_host):
    monkeypatch.setenv("ATOM_PLUGIN_HOST", "0")
    _plugins(monkeypatch, bmm=_offer(OUT, "bmm"))
    host.configure(["bmm"])
    assert host.status()["kill_switch"] is True and host.status()["seams"] == {}
    assert fresh_host[-1] == ()


def test_no_environment_variable_enables_a_plugin(monkeypatch):
    monkeypatch.setenv("ATOM_PLUGIN_HOST", "1")
    monkeypatch.setenv("ATOM_PLUGIN_HOST_STRICT", "1")
    _plugins(monkeypatch, bmm=_offer(OUT, "bmm"))
    host.configure(None)
    assert host.status()["seams"] == {} and host.status()["strict"] is True


def test_shorthand_isolates_a_broken_plugin(monkeypatch):
    def broken(h):
        raise ImportError("no kernel")

    _plugins(monkeypatch, x=broken)
    host.configure(["x"], kill_switch=False)
    assert host.status()["refusals"][-1] == host.Registration(
        "*", "x", False, "load-error"
    )


def test_a_recipe_refuses_to_start_half_loaded(monkeypatch):
    def broken(h):
        raise ImportError("no kernel")

    plan = recipes.resolve([_recipe({OUT: ["x"]})])
    _plugins(monkeypatch, x=broken)
    with pytest.raises(recipes.RecipeRefused) as refused:
        host.configure(plan, kill_switch=False)
    assert refused.value.reason == "recipe-plugin-drift"
    _plugins(monkeypatch)
    with pytest.raises(recipes.RecipeRefused, match="not installed"):
        host.configure(plan, kill_switch=False)
    _plugins(monkeypatch, x=lambda h: None)
    with pytest.raises(recipes.RecipeRefused) as refused:
        host.configure(plan, kill_switch=False)
    assert refused.value.reason == "recipe-coverage-miss"


def test_configure_after_build_is_refused(monkeypatch):
    host.configure(None)
    host.bind(OUT, _pair)
    with pytest.raises(RuntimeError, match="set_arm"):
        host.configure(["bmm"])
    host.configure(None)  # restating the built plan is fine


def test_a_second_runner_may_restate_the_built_plan(monkeypatch):
    _plugins(monkeypatch, bmm=_offer(OUT, "bmm"))
    host.configure(["bmm"], kill_switch=False)
    slot = host.bind(OUT, _native)
    host.configure(["bmm"], arm="native", kill_switch=False)
    assert slot(x, x, x, x) == ("native", 8)
    host.set_arm("plugin")
    assert host.owners(OUT, "any") == ("bmm",)


def test_the_aiter_half_gets_the_plans_providers(monkeypatch, fresh_host):
    plan = recipes.resolve(
        [
            _recipe(
                {OUT: ["bmm"]},
                plugins=[_serves("bmm", OUT), _serves("dense", kind="aiter")],
            )
        ]
    )
    _plugins(monkeypatch, bmm=_offer(OUT, "bmm"))
    host.configure(plan, kill_switch=False)
    assert fresh_host[-1] == ("dense",)
    host._reset_for_testing()
    host.configure(["bmm", "dense"], kill_switch=False)
    assert fresh_host[-1] == ("bmm", "dense")


def test_aiter_configure_is_driven_when_present(monkeypatch):
    seen = {}
    plugin = types.ModuleType("aiter.plugin")
    plugin.configure = lambda *, source, enable: seen.update(src=source, on=enable)
    aiter = types.ModuleType("aiter")
    aiter.plugin = plugin
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.plugin", plugin)
    assert CONFIGURE_AITER(("silu8k",)) is True
    assert seen == {"src": host.AITER_SOURCE, "on": ["silu8k"]}
    monkeypatch.setitem(sys.modules, "aiter.plugin", None)
    monkeypatch.setitem(sys.modules, "aiter", types.ModuleType("aiter"))
    assert CONFIGURE_AITER(("silu8k",)) is False


# -- bind / call / decisions ----------------------------------------------


def _native(x, *rest, **kw):
    return "native", x.shape[0]


def test_bind_is_the_native_object_without_an_owner():
    host.configure(None)
    native = _native
    assert host.bind(OUT, native) is native
    assert host.call(QUANT, _native, x) == ("native", 8)
    assert host.decision_counts() == {} and host.last_decision() is None
    assert OUT in host.status()["bound"]


def test_bind_and_site_check_the_seam_form():
    with pytest.raises(KeyError):
        host.bind(QUANT, _native)
    with pytest.raises(KeyError):
        host.site(OUT)


def _serve(monkeypatch, *owners, plan=None, **select_by_owner):
    """Owners of v41.attn.out whose select accepts the rows they list."""
    regs = {}
    for owner in owners:
        rows = select_by_owner.get(owner)
        regs[owner] = _offer(
            OUT,
            owner,
            impl=lambda *a, owner=owner, capability, valid_rows, **kw: (
                owner,
                capability,
            ),
            select=(
                None
                if rows is None
                else (lambda cap, *a, rows=rows, **kw: cap.rows in rows)
            ),
        )
    _plugins(monkeypatch, **regs)
    host.configure(plan or list(owners), kill_switch=False)
    return host.bind(
        OUT,
        _native,
        capability={"quant": host.MX, "flags": {"native_fused_epilogue": False}},
    )


def test_a_selected_owner_gets_the_capability_and_is_recorded(monkeypatch):
    slot = _serve(monkeypatch, "bmm")
    owner, cap = slot(x, x, x, x, dtype=torch.bfloat16)
    assert owner == "bmm"
    assert cap == host.Capability(
        OUT, quant=host.MX, rows=8, flags={"native_fused_epilogue": False}
    )
    assert host.last_decision() == host.Decision(OUT, "plugin", "selected", "bmm")
    assert host.route_proof(OUT, "bmm", 1) == 1
    with pytest.raises(AssertionError):
        host.route_proof(OUT, "bmm", 2)


def test_owners_on_disjoint_cells_coexist(monkeypatch):
    slot = _serve(monkeypatch, "bmm8k", "bmm_quant", bmm8k=(8,), bmm_quant=(6,))
    assert slot(x, x, x, x)[0] == "bmm8k"
    assert slot(torch.zeros(6, 1), x, x, x)[0] == "bmm_quant"
    assert host.last_decision().declined == 1
    assert slot(torch.zeros(5, 1), x, x, x) == ("native", 5)
    assert host.last_decision() == host.Decision(OUT, "native", "declined", declined=2)
    assert host.decision_counts() == {
        (OUT, "plugin", "selected", "bmm8k"): 1,
        (OUT, "plugin", "selected", "bmm_quant"): 1,
        (OUT, "native", "declined", None): 1,
    }


def test_select_is_asked_once_per_key(monkeypatch):
    asked = []

    def select(cap, *a, **kw):
        asked.append(cap.rows)
        return True

    _plugins(monkeypatch, bmm=_offer(OUT, "bmm", select=select))
    host.configure(["bmm"], kill_switch=False)
    slot = host.bind(OUT, _native)
    for rows in (8, 8, 8, 4, 8):
        slot(torch.zeros(rows, 1), x, x, x, dtype=torch.bfloat16)
    slot(torch.zeros(8, 1), x, x, x, dtype=torch.float16)  # a kwarg is in the key
    assert asked == [8, 4, 8]
    assert host.decision_counts()[(OUT, "plugin", "selected", "bmm")] == 6


def test_draft_ctx_keys_running_tokens(monkeypatch):
    """An aux select reads running_tokens: another value must ask again."""
    asked = []

    def select(cap, taps, rows, running_tokens):
        asked.append(running_tokens)
        return running_tokens == 6

    ctx = "v41.draft.ctx"
    _plugins(monkeypatch, aux=_offer(ctx, "aux", select=select))
    host.configure(["aux"], kill_switch=False)
    slot = host.bind(ctx, lambda taps, rows, n: ("native", n))
    taps = [torch.zeros(8, 4)] * 3
    assert slot(taps, None, 6) == ("y", "scale")
    assert slot(taps, None, 7) == ("native", 7)
    assert slot(taps, None, 6) == ("y", "scale")
    assert asked == [6, 7]


def test_moe_topk_keys_its_output_width(monkeypatch):
    """The in-place outputs carry top-k: a router plugin serves one K."""
    topk = "v41.moe.topk"
    _plugins(
        monkeypatch,
        r=_offer(
            topk,
            "r",
            impl=lambda *a, **k: None,
            select=lambda cap, weights, ids, *a, **k: ids.shape[1] == 6,
        ),
    )
    host.configure(["r"], kill_switch=False)
    native = []
    logits, bias = torch.zeros(4, 256), torch.zeros(256)
    for k in (6, 8):
        out_w, out_ids = torch.zeros(4, k), torch.zeros(4, k, dtype=torch.int32)
        host.call(
            topk,
            lambda *a, **kw: native.append(a[1].shape[1]),
            out_w,
            out_ids,
            logits,
            bias,
            True,
            1.5,
            score_func="sqrtsoftplus",
            num_valid_rows=None,
        )
    assert native == [8]
    assert host.decision_counts()[(topk, "plugin", "selected", "r")] == 1


def test_select_reads_only_what_the_key_holds():
    """Every input a select could tell calls apart by is keyed, or is a
    per-call-site constant (weights) or a per-step index (CSR) by contract."""
    unkeyed = {
        name: tuple(i for i in spec.inputs if i not in spec.key)
        for name, spec in host.SEAMS.items()
    }
    assert unkeyed == {
        FFN: (),
        OUT: ("wo_a_w", "wo_a_ws"),
        "v41.attn.qk_norm": ("q_weight", "k_weight"),
        ACT: (),
        "v41.moe.topk": (),
        QUANT: (),
        "v41.attn.index_topk": (),
        "v41.attn.attend": (
            "kv_indices_prefix",
            "kv_indptr_prefix",
            "kv_indices_extend",
            "kv_indptr_extend",
        ),
        "v41.draft.ctx": (),
    }


def test_the_route_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(host, "_CACHE_LIMIT", 3)
    _plugins(monkeypatch, bmm=_offer(OUT, "bmm"))
    host.configure(["bmm"], kill_switch=False)
    slot = host.bind(OUT, _native)
    for rows in range(1, 8):
        slot(torch.zeros(rows, 1), x, x, x)
    assert len(host._cache) <= 3


def test_same_shape_with_other_strides_or_alignment_asks_again(monkeypatch):
    """A select may check layout; a cached answer must not outlive it."""
    asked = []

    def select(cap, x_fp8, *a, **kw):
        asked.append((x_fp8.stride(), x_fp8.data_ptr() % 16))
        return x_fp8.is_contiguous() and x_fp8.data_ptr() % 16 == 0

    _plugins(monkeypatch, bmm=_offer(OUT, "bmm", select=select))
    host.configure(["bmm"], kill_switch=False)
    slot = host.bind(OUT, _native)
    base = torch.zeros(8, 32)
    dense = base[:, :16].contiguous()  # [8,16] stride (16, 1)
    strided = base[:, :16]  # [8,16] stride (32, 1): same shape, other layout
    shifted = torch.zeros(8 * 16 + 1)[1:].view(8, 16)  # 4-byte offset
    assert slot(dense, x, x, x)[0] == "y"
    assert slot(strided, x, x, x) == ("native", 8)
    assert slot(shifted, x, x, x) == ("native", 8)
    assert slot(dense.clone(), x, x, x)[0] == "y"  # cached: same layout
    assert asked == [((16, 1), 0), ((32, 1), 0), ((16, 1), 4)]


def test_strict_mode_refuses_a_call_two_owners_accept(monkeypatch):
    slot = _serve(monkeypatch, "a", "b", a=(8,), b=(8, 4))
    host._strict = True
    with pytest.raises(host.PluginConflict, match="a and b"):
        slot(x, x, x, x)
    assert slot(torch.zeros(4, 1), x, x, x)[0] == "b"


def test_a_raising_select_declines_and_a_wrong_result_raises(monkeypatch):
    _plugins(
        monkeypatch,
        a=_offer(OUT, "a", select=lambda *a, **k: 1 / 0),
        b=_offer(OUT, "b", impl=lambda *a, **k: "only-y"),
    )
    host.configure(["a", "b"], kill_switch=False)
    slot = host.bind(OUT, _native)
    with pytest.raises(host.PluginContractError):
        slot(x, x, x, x)
    assert host.last_decision().declined == 1


def test_on_decline_error_catches_a_silent_fallback(monkeypatch):
    plan = recipes.resolve([_recipe({OUT: ["bmm"]}, defaults={"on_decline": "error"})])
    _plugins(monkeypatch, bmm=_offer(OUT, "bmm", select=lambda *a, **k: False))
    host.configure(plan, kill_switch=False)
    with pytest.raises(host.PluginDeclined):
        host.bind(OUT, _native)(x, x, x, x)


def test_arms_switch_routes_without_rebuilding(monkeypatch):
    slot = _serve(monkeypatch, "bmm")
    host.set_arm("native")
    assert slot(x, x, x, x) == ("native", 8)
    assert host.last_decision().reason == "arm-native"
    host.set_arm("identity")
    assert slot(x, x, x, x) == ("native", 8)
    assert host.last_decision() == host.Decision(OUT, "plugin", "identity", "bmm")
    host.set_arm("plugin")
    assert slot(x, x, x, x)[0] == "bmm"
    with pytest.raises(ValueError):
        host.set_arm("nope")


def test_a_recipe_arm_narrows_the_plan(monkeypatch):
    plan = recipes.resolve(
        [
            _recipe(
                {OUT: ["a", "b"]},
                defaults={"allow_partition": True},
                arms={"only_b": ["b"]},
            )
        ]
    )
    _plugins(monkeypatch, a=_offer(OUT, "a"), b=_offer(OUT, "b"))
    host.configure(plan, arm="only_b", kill_switch=False)
    assert host.owners(OUT, "any") == ("b",)
    assert host.bind(OUT, _native)(x, x, x, x)[0] == "y"
    host.set_arm("plugin")
    assert host.owners(OUT, "any") == ("a", "b")


# -- graph capture: per-bucket plans, pad-M, graph claims ------------------


def _captured(monkeypatch, bucket):
    monkeypatch.setattr(host, "_graph_state", lambda: (True, bucket))


def test_capture_buckets_freeze_their_own_plan(monkeypatch):
    eager = _recipe({OUT: ["prefill"]}, mode="eager")
    small = _recipe({OUT: ["decode"]}, mode="capture", buckets=[8, 16])
    plan = recipes.resolve([eager, small])
    _plugins(
        monkeypatch,
        prefill=_offer(OUT, "prefill", impl=lambda *a, **k: ("prefill", None)),
        decode=_offer(OUT, "decode", impl=lambda *a, **k: ("decode", None)),
    )
    host.configure(plan, kill_switch=False)
    slot = host.bind(OUT, _native)
    assert slot(x, x, x, x)[0] == "prefill"
    _captured(monkeypatch, 8)
    assert slot(x, x, x, x)[0] == "decode"
    assert host.last_decision().bucket == 8 and host.last_decision().captured
    _captured(monkeypatch, 32)  # a bucket no recipe names: native
    assert slot(x, x, x, x) == ("native", 8)
    assert host.owners(OUT, 16) == ("decode",) and host.owners(OUT, 32) == ()


def test_pad_m_reaches_the_impl(monkeypatch):
    seen = {}
    valid = torch.tensor(5, dtype=torch.int32)
    monkeypatch.setattr(host, "_valid_rows", lambda rows: valid if rows == 8 else None)
    _captured(monkeypatch, 8)

    def impl(*a, capability, valid_rows, **kw):
        seen.update(cap=capability, valid=valid_rows)
        return "y", None

    _plugins(monkeypatch, bmm=_offer(OUT, "bmm", impl=impl))
    host.configure(["bmm"], kill_switch=False)
    host.bind(OUT, _native)(x, x, x, x)
    assert seen["valid"] is valid
    assert (seen["cap"].rows, seen["cap"].captured, seen["cap"].bucket) == (8, True, 8)


def _forward(monkeypatch, *, in_hipgraph, rows, capturing=False):
    from atom.utils import forward_context

    forward = types.SimpleNamespace(
        in_hipgraph=in_hipgraph,
        context=types.SimpleNamespace(running_tokens=rows),
    )
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: forward)
    monkeypatch.setattr(host, "_stream_capturing", lambda: capturing)
    monkeypatch.setattr(host, "_graph_state", GRAPH_STATE)


def test_a_captured_pass_routes_by_its_padded_rows(monkeypatch):
    """A DSpark k=5 verify of capture batch 8 runs 48 rows: bucket 48."""
    _forward(monkeypatch, in_hipgraph=True, rows=8 * (1 + 5))
    assert host._graph_state() == (True, 48)
    _forward(monkeypatch, in_hipgraph=False, rows=8 * 5)
    assert host._graph_state() == (False, None)  # an eager draft pass


def test_a_draft_graph_pass_is_captured_by_its_own_rows(monkeypatch):
    """A DraftGraph records outside `in_hipgraph`: the stream says it captures."""
    _forward(monkeypatch, in_hipgraph=False, rows=8 * 5, capturing=True)
    assert host._graph_state() == (True, 40)


def test_a_table_miss_is_recorded_never_silent(monkeypatch, caplog):
    plan = recipes.resolve(
        [
            _recipe({OUT: ["prefill"]}, mode="eager"),
            _recipe({OUT: ["decode"]}, mode="capture", buckets=[48, 40]),
        ]
    )
    _plugins(monkeypatch, prefill=_offer(OUT, "prefill"), decode=_offer(OUT, "decode"))
    host.configure(plan, kill_switch=False, strict=True)
    assert host.status()["capture_rows"] == [40, 48]
    slot = host.bind(OUT, _native)
    _captured(monkeypatch, 8)  # a bucket in batch size, not rows: no table
    with caplog.at_level("WARNING", logger="atom"):
        assert slot(x, x, x, x) == ("native", 8)
        slot(x, x, x, x)
    assert host.last_decision() == host.Decision(
        OUT, "native", "no-table", captured=True, bucket=8
    )
    assert host.decision_counts()[(OUT, "native", "no-table", None)] == 2
    assert sum("no table for captured rows 8" in r.message for r in caplog.records) == 1
    _captured(monkeypatch, 48)
    assert slot(x, x, x, x)[0] == "y"
    assert host.last_decision().owner == "decode"


def test_an_eager_only_plugin_is_never_captured(monkeypatch):
    _plugins(monkeypatch, bmm=_offer(OUT, "bmm", graph="eager"))
    host.configure(["bmm"], kill_switch=False)
    slot = host.bind(OUT, _native)
    _captured(monkeypatch, 8)
    assert slot(x, x, x, x) == ("native", 8)
    assert host.last_decision().reason == "graph-unsafe"


@pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(), reason="Dynamo unsupported here"
)
def test_traced_calls_stay_native_without_a_graph_break(monkeypatch):
    torch._dynamo.reset()
    _plugins(monkeypatch, q=_offer(QUANT, "q", impl=lambda x, **k: (x * 100, x)))
    host.configure(["q"], kill_switch=False)

    @torch.compile(backend="eager", fullgraph=True)
    def forward(t):
        return host.call(QUANT, lambda v, **k: (v + 1, v), t)[0]

    assert torch.equal(forward(torch.ones(4)), torch.full((4,), 2.0))
    assert host.decision_counts() == {(QUANT, "native", "traced", None): 1}
    torch._dynamo.reset()


# -- opaque op ---------------------------------------------------------------


def _ffn_plugin(monkeypatch, select=None):
    served = []

    def impl(t, *, capability, valid_rows):
        served.append(t.shape[0])
        return t * 10

    _plugins(monkeypatch, ffn=_offer(FFN, "ffn", impl=impl, select=select))
    host.configure(["ffn"], kill_switch=False)
    return served


def test_an_opaque_seam_without_owner_stays_inline():
    host.configure(None)
    native = _native
    assert host.bind(FFN, native, layer="L0", fake=torch.empty_like) is native
    assert host.span_op(ACT, "L0", native, torch.empty_like) is native


def test_an_opaque_seam_with_an_owner_is_one_op(monkeypatch):
    served = _ffn_plugin(monkeypatch)
    slot = host.bind(FFN, lambda t: t + 1, layer="L1", fake=torch.empty_like)
    assert torch.equal(slot(torch.ones(3)), torch.full((3,), 10.0))
    assert served == [3] and host.status()["span_ops"] == ("L1",)
    meta = slot(torch.empty(5, device="meta"))
    assert (meta.shape, meta.device.type) == ((5,), "meta")
    with pytest.raises(ValueError):
        host.bind(FFN, lambda t: t, layer="L1", fake=torch.empty_like)


@pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(), reason="Dynamo unsupported here"
)
def test_a_compiled_opaque_seam_decides_per_execution(monkeypatch):
    torch._dynamo.reset()
    served = _ffn_plugin(monkeypatch, select=lambda cap, t: cap.rows > 2)
    slot = host.bind(FFN, lambda t: t + 1, layer="L2", fake=torch.empty_like)
    compiled = torch.compile(lambda t: slot(t), backend="eager", fullgraph=True)
    compiled(torch.ones(4))
    out = compiled(torch.ones(2))
    compiled(torch.ones(8))
    assert served == [4, 8] and torch.equal(out, torch.full((2,), 2.0))
    torch._dynamo.reset()


def test_overhead_probe_reports_the_three_costs(monkeypatch):
    _plugins(monkeypatch, q=_offer(QUANT, "q", impl=lambda x, **k: (x, x)))
    host.configure(["q"], kill_switch=False)
    probe = host.overhead_probe(QUANT, lambda t: (t, t), (x,), n=10)
    assert set(probe) == {
        "native_direct_us",
        "router_native_arm_us",
        "router_plugin_arm_us",
    }
    assert host.status()["arm"] == "plugin"


# -- recipes -------------------------------------------------------------------


def test_a_recipe_resolves_to_a_deterministic_plan():
    doc = _recipe({OUT: ["bmm"]}, arms={"x": ["bmm"]})
    first, again = recipes.resolve([doc]), recipes.resolve(
        [json.loads(json.dumps(doc))]
    )
    assert first.plan_id == again.plan_id and first.plan_id.startswith("sha256:")
    assert dict(first.arms) == {
        "x": ("bmm",),
        "native": (),
        "plugin": ("bmm",),
        "identity": ("bmm",),
    }
    assert first.dump()["tables"] == {"any": {OUT: ["bmm"]}}
    assert recipes.resolve([_recipe({OUT: ["other"]})]).plan_id != first.plan_id


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda d: d.update(schema="x"), "recipe-schema"),
        (lambda d: d.update(model_arch="Llama"), "recipe-model-mismatch"),
        (lambda d: d.update(requires={"atom_hook_api": ">=3"}), "recipe-pin-mismatch"),
        (lambda d: d.update(requires={"seams": {OUT: 2}}), "recipe-contract-mismatch"),
        (lambda d: d["plan"]["seams"].update({"v41.nope": []}), "recipe-unknown-seam"),
        (
            lambda d: d["plan"]["seams"].update({OUT: ["ghost"]}),
            "recipe-unknown-provider",
        ),
        (lambda d: d["plugins"][0].update(serves=[]), "recipe-plugin-drift"),
        (lambda d: d.update(arms={"a": ["ghost"]}), "recipe-unknown-provider"),
        (lambda d: d.update(graph={"mode": "capture"}), "recipe-schema"),
        (lambda d: d.update(defaults={"on_decline": "maybe"}), "recipe-schema"),
    ],
)
def test_recipe_refusals(mutate, reason):
    doc = _recipe({OUT: ["bmm"]})
    mutate(doc)
    with pytest.raises(recipes.RecipeRefused) as refused:
        recipes.resolve([doc], model_arch="DeepseekV41ForCausalLM")
    assert refused.value.reason == reason


def test_two_owners_need_a_declared_partition():
    with pytest.raises(recipes.RecipeRefused) as refused:
        recipes.resolve([_recipe({OUT: ["a", "b"]})])
    assert refused.value.reason == "recipe-conflict"
    plan = recipes.resolve(
        [_recipe({OUT: ["a", "b"]}, defaults={"allow_partition": True})]
    )
    assert plan.tables["any"][OUT] == ("a", "b")


def test_a_region_owner_shadows_its_children_only_when_declared():
    seams = {FFN: ["whole"], ACT: ["silu"], "v41.moe.topk": ["whole"]}
    with pytest.raises(recipes.RecipeRefused) as refused:
        recipes.resolve([_recipe(seams)])
    assert refused.value.reason == "recipe-shadow"
    plan = recipes.resolve([_recipe(seams, defaults={"shadow_policy": "shadow"})])
    assert plan.shadowed == ((ACT, "silu", "whole"),)


def test_a_recipe_set_must_agree_and_not_overlap():
    eager = _recipe({OUT: ["a"]}, mode="eager")
    with pytest.raises(recipes.RecipeRefused) as refused:
        recipes.resolve([eager, _recipe({OUT: ["a"]}, mode="any")])
    assert refused.value.reason == "recipe-set-conflict"
    capture = _recipe({OUT: ["a"]}, mode="capture", buckets=[8])
    with pytest.raises(recipes.RecipeRefused, match="8"):
        recipes.resolve([eager, capture, dict(capture)])
    other = dict(capture, strategy="tp8")
    with pytest.raises(recipes.RecipeRefused, match="strategy"):
        recipes.resolve([eager, other])
    plan = recipes.resolve([eager, capture])
    assert set(plan.tables) == {"eager", 8}


def test_load_reads_a_strategy_directory(tmp_path):
    (tmp_path / "eager.json").write_text(
        json.dumps(_recipe({OUT: ["a"]}, mode="eager"))
    )
    (tmp_path / "b8.json").write_text(
        json.dumps(_recipe({OUT: ["a"]}, mode="capture", buckets=[8]))
    )
    plan = recipes.load(tmp_path)
    assert plan.strategy == "tp4-c8" and set(plan.tables) == {"eager", 8}
    assert recipes.load(tmp_path / "eager.json").tables.keys() == {"eager"}


def test_cli_may_only_narrow_a_recipe(tmp_path):
    path = tmp_path / "r.json"
    doc = _recipe({OUT: ["a", "b"]}, defaults={"allow_partition": True})
    path.write_text(json.dumps(doc))
    narrowed = recipes.from_cli(str(path), "b").resolve()
    assert narrowed.arms["plugin"] == ("b",)
    assert narrowed.plan_id != recipes.from_cli(str(path), None).plan_id
    with pytest.raises(recipes.RecipeRefused):
        recipes.from_cli(str(path), "c")
    with pytest.raises(recipes.RecipeRefused) as refused:
        recipes.from_cli(str(path), None, arm="ghost")
    assert refused.value.reason == "recipe-unknown-arm"
    assert recipes.from_cli(None, None) is None
    source = recipes.from_cli(None, "a, b")
    assert source.enable == ("a", "b") and source.documents == ()
    assert source.resolve().providers == ("a", "b")


def test_runtime_expect_checks_the_server():
    plan = recipes.resolve(
        [_recipe({OUT: ["a"]}, runtime={"expect": {"tensor_parallel_size": 4}})]
    )
    good = types.SimpleNamespace(tensor_parallel_size=4)
    recipes.check_runtime(plan, good)
    with pytest.raises(recipes.RecipeRefused) as refused:
        recipes.check_runtime(plan, types.SimpleNamespace(tensor_parallel_size=8))
    assert refused.value.reason == "recipe-runtime-mismatch"


def test_recipe_check_command(tmp_path, capsys):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_recipe({OUT: ["a"]})))
    assert recipes.main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["tables"] == {"any": {OUT: ["a"]}}
    assert recipes.main([str(path), "--installed", "a=v41.attn.qk_norm"]) == 2
    assert "recipe-plugin-drift" in capsys.readouterr().out


# -- wiring: CLI, Config, compile cache, ModelRunner -----------------------------

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "argv, providers, arm",
    [
        ([], None, "plugin"),
        (["--plugin-enable", " bmm8k,,silu8k "], ("bmm8k", "silu8k"), "plugin"),
        (["--plugin-enable", "a", "--plugin-arm", "native"], ("a",), "native"),
    ],
)
def test_plugin_args_reach_the_config(argv, providers, arm):
    from atom.config import Config
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    kwargs = EngineArgs.from_cli_args(parser.parse_args(argv))._get_engine_kwargs()
    source = kwargs["plugin_source"]
    assert (None if source is None else source.resolve().providers) == providers
    assert kwargs["plugin_arm"] == arm
    assert {"plugin_source", "plugin_arm"} <= set(Config.__dataclass_fields__)


def _config_with(source):
    """A Config carrying ``source``, as a rank's copy would (no model load)."""
    from atom.config import Config

    config = object.__new__(Config)
    config.__dict__.update(plugin_source=source, plugin_arm="plugin")
    return config


def _a_recipe_source(tmp_path):
    seams = {OUT: ["a", "b"], ACT: ["c"]}
    capture = _recipe(seams, mode="capture", buckets=[8, 16])
    capture["defaults"] = {"allow_partition": True}
    eager = _recipe({OUT: ["a"]}, mode="eager")
    (tmp_path / "b8.json").write_text(json.dumps(capture))
    (tmp_path / "eager.json").write_text(json.dumps(eager))
    return recipes.from_cli(str(tmp_path), "a,c", "plugin")


def test_a_config_with_a_plan_copies_and_pickles(tmp_path):
    """engine_core_mgr deep-copies the Config per rank and ranks unpickle it."""
    import copy
    import pickle

    source = _a_recipe_source(tmp_path)
    assert not source.empty
    config = _config_with(source)
    for clone in (copy.deepcopy(config), pickle.loads(pickle.dumps(config))):
        assert clone.plugin_source == source
        assert clone.plugin_source.resolve().plan_id == source.plan_id
    plan = source.resolve()
    assert pickle.loads(pickle.dumps(plan)) == plan == copy.deepcopy(plan)
    assert set(plan.tables) == {"eager", 8, 16}
    assert plan.arms["plugin"] == ("a", "c")


def _plan_id_in_a_child(config):
    return config.plugin_source.resolve().plan_id


@pytest.mark.skipif(sys.platform == "win32", reason="ATOM workers are Linux")
def test_a_config_with_a_plan_crosses_a_spawned_process(tmp_path):
    import multiprocessing

    source = _a_recipe_source(tmp_path)
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        assert pool.apply(_plan_id_in_a_child, (_config_with(source),)) == (
            source.plan_id
        )


def test_a_source_that_resolves_otherwise_refuses():
    source = recipes.from_cli(None, "a")
    forged = recipes.RecipeSource(source.documents, ("b",), source.plan_id, False)
    with pytest.raises(recipes.RecipeRefused):
        forged.resolve()


def test_status_lists_the_capture_rows_of_a_recipe_set(tmp_path, monkeypatch):
    """A strategy's directory, one file per capture batch size, keyed by rows
    (DSpark k=5: verify bs*6, draft bs*5), as a worker gets it: a pickled
    RecipeSource resolved in configure."""
    import pickle

    for bs in (1, 2, 8):
        doc = _recipe(
            {OUT: ["bmm"], "v41.draft.ctx": ["aux"]},
            mode="capture",
            buckets=[bs * 6, bs * 5],
        )
        doc["defaults"] = {"strict_overlap": True}
        (tmp_path / f"{bs}.json").write_text(json.dumps(doc))
    eager = _recipe({QUANT: ["q16k"]}, mode="eager")
    eager["defaults"] = {"strict_overlap": True}
    (tmp_path / "eager.json").write_text(json.dumps(eager))
    source = pickle.loads(pickle.dumps(recipes.from_cli(str(tmp_path), None)))

    def both(h):
        _offer(OUT, "bmm")(h)

    _plugins(
        monkeypatch,
        bmm=both,
        aux=_offer("v41.draft.ctx", "aux"),
        q16k=_offer(QUANT, "q16k", impl=lambda x, **k: (x, x)),
    )
    host.configure(source, kill_switch=False)
    rows = [5, 6, 10, 12, 40, 48]
    assert host.status()["capture_rows"] == rows
    host.bind(OUT, _native)
    for arm in ("native", "identity", "plugin"):
        host.set_arm(arm)
        assert host.status()["capture_rows"] == rows
    assert host.owners(OUT, 48) == ("bmm",) and host.owners(QUANT) == ("q16k",)


def test_a_bad_recipe_refuses_at_argument_time(tmp_path):
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    bad = tmp_path / "r.json"
    bad.write_text(json.dumps({"schema": "nope"}))
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = EngineArgs.from_cli_args(parser.parse_args(["--plugin-recipe", str(bad)]))
    with pytest.raises(recipes.RecipeRefused):
        args._get_engine_kwargs()


def test_the_plan_keys_the_compile_cache_only_when_set():
    from atom.config import Config

    base = {
        "quant_config": None,
        "compilation_config": None,
        "parallel_config": None,
        "tensor_parallel_size": 4,
        "prefill_context_parallel_size": 1,
        "sequence_parallel_size": 1,
        "dcp_config": None,
        "enable_dp_attention": False,
        "index_cache_dtype": None,
        "index_cache_layout": None,
        "hf_config": types.SimpleNamespace(),
        "max_model_len": 4096,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 16384,
        "kv_cache_dtype": "fp8",
        "speculative_config": None,
    }

    def key(**kw):
        return Config.compute_hash(types.SimpleNamespace(**base, **kw))

    a, b = recipes.from_cli(None, "a"), recipes.from_cli(None, "b")
    assert key() == key(plugin_source=None)
    assert key(plugin_source=a) not in (key(), key(plugin_source=b))


def test_model_runner_configures_the_host_before_building():
    source = (REPO_ROOT / "atom/model_engine/model_runner.py").read_text(
        encoding="utf-8"
    )
    configure = source.index("plugin_host.configure(")
    assert source.index("config.plugin_source, arm=config.plugin_arm", configure) < (
        source.index("self._build_and_load_model(model_class)")
    )


# -- V4.1 seams --------------------------------------------------------------------


def _seam_uses(path):
    """(form, seam) of each plugin_host.bind/call/site/span_op naming a seam."""
    import ast

    uses = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "plugin_host"
            and node.func.attr in ("bind", "call", "site", "span_op")
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            uses.append((node.func.attr, node.args[0].value))
    return uses


def test_every_seam_is_reached_exactly_once_from_its_target():
    found = {}
    for path in (REPO_ROOT / "atom").rglob("*.py"):
        for form, seam in _seam_uses(path):
            if form != "span_op":
                where = path.relative_to(REPO_ROOT).as_posix()
                found.setdefault(seam, []).append((form, where))
    expected = {
        name: [
            ("site" if s.form == "call" and s.name in _DECORATED else s.form, s.target)
        ]
        for name, s in host.SEAMS.items()
    }
    assert found == expected


_DECORATED = {"v41.op.quant_fp8_g32", "v41.attn.attend"}


def test_only_the_shared_act_seam_builds_its_own_span():
    spans = [
        (seam, path.relative_to(REPO_ROOT).as_posix())
        for path in (REPO_ROOT / "atom").rglob("*.py")
        for form, seam in _seam_uses(path)
        if form == "span_op"
    ]
    assert spans == [(ACT, "atom/models/deepseek_v4/__init__.py")]


def test_seam_specs_are_self_consistent():
    for name, spec in host.SEAMS.items():
        assert spec.name == name and spec.contract >= 1
        assert spec.kind in ("op", "region") and spec.form in ("bind", "call")
        assert spec.graph in ("eager", "capture_safe", "opaque_op")
        assert set(spec.key) <= set(spec.inputs)
        assert spec.parent is None or host.SEAMS[spec.parent].kind == "region"
        assert (REPO_ROOT / spec.target).is_file()


def _attention_module():
    from aiter_stub import atom_under_stubs

    with atom_under_stubs() as load:
        return load("atom.models.deepseek_v41.attention")


def _project_out(module, fused, groups=2, rank=64, k=128, rows=3, heads=4):
    """`Attention._project_out` on a stand-in layer; wo_b's (y, x_scale)."""
    from functools import partial

    x_fp8 = torch.randn(rows, groups, k)
    x_scale = torch.randn(rows, groups, k // 32)
    module.inverse_rope_group_quant = lambda *a, **kw: (x_fp8, x_scale)
    module.batched_gemm_a8w8_mxscale_bpreshuffle = lambda x, w, xs, ws, dtype: (
        torch.full((rows, groups, rank), 1.0)
    )
    module.batched_gemm_a8w8_mxscale_bpreshuffle_fused_quant = lambda x, w, xs, ws: (
        torch.full((rows, groups, rank), 2.0),
        torch.full((rows, groups, rank // 32), 3.0),
    )
    wo_b = []
    layer = types.SimpleNamespace(
        heads=heads,
        head_dim=groups * k // heads,
        groups=groups,
        o_rank=rank,
        _fuse_woa_quant=fused,
        wo_a=types.SimpleNamespace(
            weight=torch.zeros(groups * rank, k),
            weight_scale=torch.zeros(groups * rank // 32, k // 32),
        ),
        wo_b=lambda y, x_scale=None: wo_b.append((y, x_scale)) or "out",
    )
    native = partial(module.Attention._wo_a, layer)
    capability = partial(module.Attention._wo_a_capability, layer)
    layer._wo_a_seam = host.bind(OUT, native, capability=capability)
    output = torch.zeros(1, rows, heads, groups * k // heads)
    rope = types.SimpleNamespace(cos_cache=None, sin_cache=None)
    module.Attention._project_out(layer, output, rope, torch.zeros(1, rows))
    return layer, native, wo_b[0]


@pytest.mark.parametrize("fused", [True, False])
def test_attn_out_native_feeds_wo_b_as_before(fused):
    host.configure(None)
    layer, native, (y, y_scale) = _project_out(_attention_module(), fused)
    assert layer._wo_a_seam is native
    if fused:
        assert torch.equal(y, torch.full((1, 3, 128), 2.0))
        assert torch.equal(y_scale, torch.full((1, 3, 4), 3.0))
    else:
        assert y_scale is None and torch.equal(y, torch.ones(1, 3, 128))


def test_attn_out_plugin_pair_reaches_wo_b(monkeypatch):
    got = {}

    def bmm(x_fp8, x_scale, w, ws, *, dtype, capability, valid_rows):
        got.update(shapes=(w.shape, ws.shape), cap=capability, dtype=dtype)
        return torch.full((3, 2, 64), 7.0), torch.full((3, 2, 2), 8.0)

    _plugins(monkeypatch, bmm=_offer(OUT, "bmm", impl=bmm))
    host.configure(["bmm"], kill_switch=False)
    _, _, (y, y_scale) = _project_out(_attention_module(), fused=False)
    assert got["shapes"] == ((2, 64, 128), (2, 2, 4))
    assert got["cap"].quant == host.MX and got["cap"].rows == 3
    assert dict(got["cap"].flags) == {
        "native_fused_epilogue": False,
        "groups": 2,
        "o_rank": 64,
        "wo_a_preshuffled": True,
    }
    assert torch.equal(y, torch.full((1, 3, 128), 7.0))
    assert torch.equal(y_scale, torch.full((1, 3, 4), 8.0))


DIM = 8


def _shared_expert(name, fused=False):
    """deepseek_v4 `Expert`, built (bind/span_op) on a stand-in layer after
    `configure`: (forward, w2's calls)."""
    from functools import partial

    from aiter_stub import atom_under_stubs

    with atom_under_stubs() as load:
        module = load("atom.models.deepseek_v4")
    module.aiter_silu_and_mul = lambda out, combined, limit: out.copy_(
        combined[:, : out.shape[1]].clamp(max=limit)
    )
    w2 = []

    class W2:
        native_a8_group_rows = 32
        output_size = DIM

        def __call__(self, x, x_scale=None):
            w2.append((x, x_scale))
            return x.new_ones(x.shape[0], DIM, dtype=torch.bfloat16)

    layer = types.SimpleNamespace(
        gate_up_proj=lambda x: torch.cat([x, x], dim=-1),
        use_fused_clamp_act_mul=fused,
        swiglu_limit=10.0,
        w2=W2(),
    )
    for method in ("_act", "_act_down", "_act_down_fake", "_act_capability"):
        setattr(layer, method, partial(getattr(module.Expert, method), layer))
    layer._act_seam = host.bind(ACT, layer._act, capability=layer._act_capability)
    layer.act_down = host.span_op(ACT, name, layer._act_down, layer._act_down_fake)
    return layer, partial(module.Expert.forward, layer), w2


def test_shared_act_without_a_plugin_stays_inline():
    host.configure(None)
    layer, forward, w2 = _shared_expert("layers.0.ffn.shared_experts")
    assert layer.act_down == layer._act_down and layer._act_seam == layer._act
    x = torch.arange(6.0).reshape(2, 3) * 5
    assert forward(x).shape == (2, DIM)
    act, scale = w2[0]
    assert scale is None and torch.equal(act, x.clamp(max=10.0))


def test_shared_act_plugin_runs_inside_one_opaque_op(monkeypatch):
    seen = []

    def silu(combined, weights, *, capability, valid_rows):
        seen.append((combined.shape[0], capability.quant, dict(capability.flags)))
        return torch.zeros(combined.shape[0], 3), torch.ones(combined.shape[0], 1)

    _plugins(
        monkeypatch,
        silu=_offer(ACT, "silu", impl=silu, select=lambda cap, c, w: c.shape[0] > 2),
    )
    host.configure(["silu"], kill_switch=False)
    layer, forward, w2 = _shared_expert("layers.1.ffn.shared_experts")
    assert layer.act_down != layer._act_down
    forward(torch.ones(4, 3))
    forward(torch.ones(2, 3))
    assert seen == [(4, host.MX, {"swiglu_limit": 10.0, "native_fused_act": False})]
    assert torch.equal(w2[0][1], torch.ones(4, 1)) and w2[1][1] is None
    meta = layer.act_down(torch.empty(5, 6, device="meta"), None)
    assert (meta.shape, meta.dtype) == ((5, DIM), torch.bfloat16)


def test_draft_ctx_native_is_the_old_cat_and_slice():
    import ast

    tree = ast.parse(
        (REPO_ROOT / "atom/models/deepseek_v41/dspark.py").read_text(encoding="utf-8")
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41DSpark"
    )
    node = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_concat_taps"
    )
    node.decorator_list = []
    namespace = {"torch": torch}
    code = compile(ast.Module([node], []), "dspark.py", "exec")
    exec(code, namespace)  # noqa: S102 -- ATOM's own source, one function
    concat = namespace["_concat_taps"]
    taps = [torch.randn(8, 4) for _ in range(3)]
    rows = torch.tensor([1, 3, 5, 7])
    old = torch.cat(taps, dim=-1)
    assert torch.equal(concat(taps, None, 6)[0], old[:6])
    assert torch.equal(concat(taps, rows, 3)[0], old[rows][:3])
    assert torch.equal(concat(old, rows, 3)[0], old[rows][:3])
    assert concat(taps, None, 6)[1] is None


def test_draft_ctx_and_ffn_call_sites():
    dspark = (REPO_ROOT / "atom/models/deepseek_v41/dspark.py").read_text(
        encoding="utf-8"
    )
    assert dspark.count("main_proj(aux_concat, x_scale=aux_scale)") == 2
    assert "self._ctx_seam(aux_concat, rows, step.running_tokens)" in dspark
    proposer = (REPO_ROOT / "atom/spec_decode/dspark_proposer.py").read_text(
        encoding="utf-8"
    )
    assert 'getattr(self.model, "context_takes_taps", False)' in proposer
    model = (REPO_ROOT / "atom/models/deepseek_v41/model.py").read_text(
        encoding="utf-8"
    )
    assert "self.finish_ffn(self._ffn(normed), residual, pre, post, comb)" in model
    assert "self.ffn.__call__," in model


@pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(), reason="Dynamo unsupported here"
)
def test_a_bound_module_call_traces_like_the_module():
    """No owner: Block keeps `self.ffn.__call__`, which must trace as the module."""
    torch._dynamo.reset()
    host.configure(None)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = torch.nn.Linear(4, 4)
            self._ffn = host.bind(
                FFN, self.ffn.__call__, layer="toy", fake=torch.empty_like
            )

        def forward(self, x):
            return self._ffn(x)

    toy = Toy()
    assert toy._ffn == toy.ffn.__call__
    assert [n for n, _ in toy.named_modules()] == ["", "ffn"]
    x = torch.randn(2, 4)
    compiled = torch.compile(toy, backend="eager", fullgraph=True)
    assert torch.allclose(compiled(x), toy.ffn(x))
    torch._dynamo.reset()
