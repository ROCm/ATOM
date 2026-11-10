# SPDX-License-Identifier: MIT
"""CPU checks for the provider-backed MiniMax-M3 SP QKV prologue.

No AITER/triton/GPU is needed: the provider interface, the neutral op and the
attention tail are faked.  Pins the required behaviour:
* configure() is called exactly when --aiter-provider-enable is set;
* the ``aiter.plugin.explain`` gate selects provider vs #3084's non-fused path;
* the flat 15-argument + keyword payload matches the neutral contract;
* a provider decline keeps the non-fused path (no fused call, no exception);
* no removed ``aiter.gemm_fusion`` / ``aiter.ops.gemm_fusion`` import remains.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import atom.aiter_provider as provider
import atom.model_ops.gemm_fusion.site as site_mod
from atom.model_ops.gemm_fusion import GemmFusionSites


@pytest.fixture(autouse=True)
def _reset_process_gate(monkeypatch):
    monkeypatch.setattr(site_mod, "_ENSURE_DONE", False)
    monkeypatch.setattr(site_mod, "_PLUGIN", None)


def _fwd_ctx():
    return SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=False),
        attn_metadata=SimpleNamespace(slot_mapping=torch.arange(6, dtype=torch.int64)),
    )


def _state():
    return {
        "k_cache": torch.zeros((2, 1, 8, 128, 16), dtype=torch.float8_e4m3fn),
        "v_cache": torch.zeros((2, 1, 8, 128, 16), dtype=torch.float8_e4m3fn),
        "k_scale": torch.zeros((2, 1, 128)),
        "v_scale": torch.zeros((2, 1, 128)),
        "slot_mapping": torch.arange(6, dtype=torch.int64),
    }


def _impl(mode="dense"):
    return SimpleNamespace(
        head_dim=128,
        num_heads=16,
        num_kv_heads=1,
        num_idx_heads=4,
        index_head_dim=128,
        rotary_emb=SimpleNamespace(
            rotary_dim=64,
            is_neox_style=True,
            cos_cache=torch.zeros(8, 32, dtype=torch.bfloat16),
            sin_cache=torch.zeros(8, 32, dtype=torch.bfloat16),
        ),
        q_norm=SimpleNamespace(weight=torch.zeros(128), variance_epsilon=1e-6),
        k_norm=SimpleNamespace(weight=torch.zeros(128)),
        index_q_norm=SimpleNamespace(weight=torch.zeros(128)),
        index_k_norm=SimpleNamespace(weight=torch.zeros(128)),
        index_cache=torch.zeros((2, 128, 128), dtype=torch.float8_e4m3fn),
        gemm_fusion_prologue_state=lambda _fwd: _state(),
        forward_after_prologue=lambda *a, **k: torch.zeros((6, 2048)),
    )


class _Linear:
    def __init__(self):
        self.weight = torch.zeros((9216, 6144), dtype=torch.uint8)
        self.weight.is_shuffled = True
        self.weight_scale = torch.zeros(9216)
        self.input_size = 6144
        self.output_size = 9216
        self.is_output_padded = True

    def __call__(self, x, x_scale=None):
        return torch.zeros((x.shape[0], self.output_size), dtype=torch.bfloat16)


def _site(mode="dense", provider_enabled=True, kv_ready=True):
    attn = SimpleNamespace(
        impl=_impl(mode),
        sp_head_exchange=None,
        layer_name="model.layers.0.self_attn",
    )
    config = SimpleNamespace(
        sequence_parallel_size=4,
        max_num_batched_tokens=8,
        aiter_provider_enable=("sp-qkv-provider",) if provider_enabled else (),
    )
    site = site_mod.QkvSite("layer0.qkv", attn, _Linear(), mode, config, out_width=2048)
    site.kv_ready = kv_ready
    return site


def _x():
    return torch.zeros((2, 6144), dtype=torch.float8_e4m3fn)


def _positions():
    return torch.arange(6, dtype=torch.int64)


def _install_attention_tail(monkeypatch):
    monkeypatch.setattr(site_mod, "get_forward_context", _fwd_ctx)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    module = types.ModuleType("atom.model_ops.sp_attention")
    module.ulysses_attention = lambda attn, q, k, v, pos, q_scale, qkv: (
        torch.zeros((6, 2048)),
        torch.arange(6),
    )
    module.gather_heads = lambda out, rows, ca, scale: "OUT"
    monkeypatch.setitem(sys.modules, "atom.model_ops.sp_attention", module)


def _install_gate(monkeypatch, route="provider"):
    calls = []

    def explain(op, *args, **kwargs):
        assert op == "sp_qkv_fused"
        selected = route(args[1].shape[0]) if callable(route) else route
        return SimpleNamespace(decision=SimpleNamespace(route=selected), declines={})

    def op(*args, **kwargs):
        calls.append((args, kwargs))

    module = types.ModuleType("aiter")
    module.plugin = SimpleNamespace(
        API_VERSION=1,
        configure=lambda **kwargs: None,
        status=lambda: {"routed_ops": {"sp_qkv_fused": {"version": 1}}},
        explain=explain,
    )
    module.sp_qkv_fused = op
    monkeypatch.setitem(sys.modules, "aiter", module)
    monkeypatch.setattr(site_mod, "get_forward_context", _fwd_ctx)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(site_mod, "_sp_group", lambda: None)
    _install_attention_tail(monkeypatch)
    return calls


# ---------------------------------------------------------------- enablement
def test_configure_not_called_when_flag_off(monkeypatch):
    configured = []
    module = types.ModuleType("aiter")
    module.plugin = SimpleNamespace(
        API_VERSION=1,
        configure=lambda **kwargs: configured.append(kwargs),
        status=lambda: {"routed_ops": {}},
    )
    monkeypatch.setitem(sys.modules, "aiter", module)
    config = SimpleNamespace(aiter_provider_enable=())
    assert provider.configure_aiter_providers(config) is False
    assert configured == []


def test_configure_called_once_with_source_when_enabled(monkeypatch):
    configured = []
    module = types.ModuleType("aiter")
    module.plugin = SimpleNamespace(
        API_VERSION=1,
        configure=lambda **kwargs: configured.append(kwargs),
        status=lambda: {"routed_ops": {"sp_qkv_fused": {}}, "enabled": ("p",)},
    )
    monkeypatch.setitem(sys.modules, "aiter", module)
    config = SimpleNamespace(aiter_provider_enable=("sp-qkv-provider",))
    assert provider.configure_aiter_providers(config) is True
    assert configured and configured[0]["enable"] == ("sp-qkv-provider",)
    assert configured[0]["source"].startswith("atom ")


def test_configure_resolves_lazy_routed_ops_before_status(monkeypatch):
    """A lazy aiter export registers its route on attribute access."""
    events = []
    resolved = [False]
    module = types.ModuleType("aiter")

    def lazy(name):
        if name == "sp_qkv_fused":
            events.append("getattr")
            resolved[0] = True
            return object()
        raise AttributeError(name)

    module.__getattr__ = lazy
    module.plugin = SimpleNamespace(
        API_VERSION=1,
        configure=lambda **kwargs: events.append("configure"),
        status=lambda: (
            events.append("status"),
            {"routed_ops": {"sp_qkv_fused": {}} if resolved[0] else {}},
        )[1],
    )
    monkeypatch.setitem(sys.modules, "aiter", module)
    config = SimpleNamespace(aiter_provider_enable=("sp-qkv-provider",))
    assert provider.configure_aiter_providers(config) is True
    assert resolved[0] is True
    assert events.index("getattr") < events.index("status")


def test_site_reasserts_enablement_once_per_process(monkeypatch):
    events = []
    configured = []

    def status():
        events.append("status")
        return {
            "enabled": ("sp-qkv-provider",) if configured else (),
            "routed_ops": {"sp_qkv_fused": {}},
        }

    def configure(**kwargs):
        events.append("configure")
        configured.append(kwargs)

    def explain(op, *args, **kwargs):
        events.append("explain")
        return SimpleNamespace(decision=SimpleNamespace(route="provider"), declines={})

    module = types.ModuleType("aiter")
    module.plugin = SimpleNamespace(
        API_VERSION=1, status=status, configure=configure, explain=explain
    )
    module.sp_qkv_fused = object()
    monkeypatch.setitem(sys.modules, "aiter", module)

    site = _site()
    assert site._serves(2, [], {}) is True
    assert [c["enable"] for c in configured] == [("sp-qkv-provider",)]
    assert events.index("configure") < events.index("explain")

    # A second site in the same process must not configure again.
    site2 = _site()
    assert site2._serves(2, [], {}) is True
    assert len(configured) == 1
    assert site_mod._ENSURE_DONE is True


def test_provider_warmup_is_framework_owned_and_fail_open(monkeypatch):
    warmed = []

    class _Provider:
        def warmup(self, payload, *, group=None, device=None):
            warmed.append((payload, group, device))
            return True

    monkeypatch.setattr(provider, "_load_provider", lambda name: _Provider())
    config = SimpleNamespace(aiter_provider_enable=("sp-qkv-provider",))
    assert provider.warmup_aiter_providers(
        config, {"mode": "dense"}, group="g", device=3
    )
    assert warmed == [({"mode": "dense"}, "g", 3)]


# ---------------------------------------------------------------- explain gate
def test_explain_gate_selects_provider_and_assembles_flat_payload(monkeypatch):
    calls = _install_gate(monkeypatch, route="provider")
    site = _site()
    assert site.run(_x(), torch.zeros((2, 1)), _positions(), None) == "OUT"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) == 15
    assert args[3] is site.linear.weight and args[4] is site.linear.weight_scale
    assert args[5].dtype == torch.float32 and args[6].dtype == torch.float32
    assert args[7].shape == args[8].shape == (8, 32)
    assert args[9] is not None and args[10] is not None
    assert kwargs["mode"] == "dense"
    assert kwargs["k"] is not None and kwargs["v"] is not None
    assert kwargs["index_q"] is None
    assert kwargs["eps"] == 1e-6
    assert kwargs["group"] is None and kwargs["stream"] is None


def test_explain_gate_is_cached_per_local_m(monkeypatch):
    calls = _install_gate(monkeypatch, route="provider")
    site = _site()
    for _ in range(2):
        site.run(_x(), torch.zeros((2, 1)), _positions(), None)
    assert len(calls) == 2  # provider op runs per call
    assert site._routes == {("dense", 2): True}  # explain asked once for m=2


def test_indexer_payload_carries_index_tensors_only(monkeypatch):
    calls = _install_gate(monkeypatch, route="provider")
    site = _site(mode="indexer")
    site.run(_x(), torch.zeros((2, 1)), _positions(), None)
    _, kwargs = calls[0]
    assert kwargs["mode"] == "indexer"
    assert kwargs["k"] is None and kwargs["v"] is None
    assert kwargs["index_q"] is not None
    assert kwargs["index_q_norm_w"] is not None
    assert kwargs["index_k_norm_w"] is not None
    assert kwargs["index_cache"] is not None


def test_provider_decline_keeps_3084_non_fused_path(monkeypatch):
    calls = _install_gate(monkeypatch, route="native")
    site = _site()
    assert site.run(_x(), torch.zeros((2, 1)), _positions(), None) == "OUT"
    assert calls == []
    assert site._routes == {("dense", 2): False}


class _FakeGroup:
    def __init__(self, verdicts):
        self.verdicts = verdicts


def _fake_rank_gather(monkeypatch, group):
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, obj, group=None: output.__setitem__(slice(None), group.verdicts),
    )


def test_rank_disagreement_falls_back_on_every_rank(monkeypatch):
    _install_gate(monkeypatch, route="provider")
    site = _site()
    group = _FakeGroup([(True, 2), (False, 2), (True, 2), (True, 2)])
    site._group = group
    _fake_rank_gather(monkeypatch, group)
    assert site._serves(2, [], {}) is False
    assert site._routes == {("dense", 2): False}


def test_rank_agreement_serves_as_one_sequence(monkeypatch):
    _install_gate(monkeypatch, route="provider")
    site = _site()
    group = _FakeGroup([(True, 2), (True, 2), (True, 2), (True, 2)])
    site._group = group
    _fake_rank_gather(monkeypatch, group)
    assert site._serves(2, [], {}) is True
    assert site._routes == {("dense", 2): True}


def test_rank_m_mismatch_falls_back_on_every_rank(monkeypatch):
    _install_gate(monkeypatch, route="provider")
    site = _site()
    group = _FakeGroup([(True, 2), (True, 2), (True, 3), (True, 2)])
    site._group = group
    _fake_rank_gather(monkeypatch, group)
    assert site._serves(2, [], {}) is False
    assert site._routes == {("dense", 2): False}


def test_no_provider_enabled_never_touches_the_interface(monkeypatch):
    _install_gate(monkeypatch, route="native")
    site = _site(provider_enabled=False)
    assert site.run(_x(), torch.zeros((2, 1)), _positions(), None) == "OUT"


def test_provider_deferred_until_kv_pool_ready(monkeypatch):
    calls = _install_gate(monkeypatch, route="provider")
    site = _site(kv_ready=False)
    assert site.run(_x(), torch.zeros((2, 1)), _positions(), None) == "OUT"
    assert calls == []
    assert site._routes == {}
    site.mark_kv_ready()
    assert site.run(_x(), torch.zeros((2, 1)), _positions(), None) == "OUT"
    assert len(calls) == 1


def test_prepare_resets_provider_and_marks_sites_kv_ready(monkeypatch):
    import atom.model_ops.gemm_fusion as gf

    events = []
    site = SimpleNamespace(
        _routes={2: True},
        mark_kv_ready=lambda: (
            events.append("kv-ready"),
            site._routes.clear(),
        ),
    )
    monkeypatch.setattr(
        provider,
        "shutdown_aiter_providers",
        lambda config, group=None: events.append("shutdown"),
    )
    monkeypatch.setattr(gf, "_warmup", lambda config, sites: events.append("warmup"))
    monkeypatch.setattr(site_mod, "_sp_group", lambda: None)
    registry = gf.GemmFusionSites(
        [site], SimpleNamespace(aiter_provider_enable=("sp-qkv-provider",))
    )
    registry.prepare(None)
    assert events == ["shutdown", "kv-ready", "warmup"]
    assert site._routes == {}


def test_provider_flag_on_without_interface_falls_back(monkeypatch):
    _install_attention_tail(monkeypatch)
    monkeypatch.delitem(sys.modules, "aiter", raising=False)
    monkeypatch.setattr(
        site_mod, "_sp_group", lambda: (_ for _ in ()).throw(RuntimeError)
    )
    site = _site(provider_enabled=True)
    assert site.run(_x(), torch.zeros((2, 1)), _positions(), None) == "OUT"


def test_warmup_payload_has_every_contract_key(monkeypatch):
    monkeypatch.setattr(site_mod, "_sp_group", lambda: None)
    payload = _site().warmup_payload()
    expected = {
        "q",
        "x",
        "x_scale",
        "w",
        "w_scale",
        "q_norm_w",
        "k_norm_w",
        "cos_cache",
        "sin_cache",
        "positions",
        "slot_mapping",
        "k_cache",
        "v_cache",
        "k_scale",
        "v_scale",
        "mode",
        "k",
        "v",
        "index_q",
        "index_q_norm_w",
        "index_k_norm_w",
        "index_cache",
        "eps",
        "group",
        "stream",
    }
    assert set(payload) == expected
    assert payload["mode"] == "dense"


def test_large_band_asks_explain_then_keeps_the_fallback(monkeypatch):
    calls = _install_gate(
        monkeypatch, route=lambda m: "provider" if m <= 256 else "native"
    )
    site = _site()
    x = torch.zeros((512, 6144), dtype=torch.float8_e4m3fn)
    x_scale = torch.zeros((512, 1))
    assert site.run(x, x_scale, _positions(), None) == "OUT"
    assert calls == []  # no provider call for the large cell
    assert site._routes == {("dense", 512): False}


def test_initialize_is_none_without_the_provider_flag():
    from atom.model_ops.gemm_fusion import initialize_gemm_fusion

    config = SimpleNamespace(aiter_provider_enable=())
    assert initialize_gemm_fusion(None, config) is None


def test_capture_gate_requires_preprepare(monkeypatch):
    _install_gate(monkeypatch, route="provider")
    site = _site()
    monkeypatch.setattr(site_mod, "_in_hipgraph", lambda: True)
    assert site._serves(2, [], {}) is False  # no pre-bound capture cell
    monkeypatch.setattr(site_mod, "_in_hipgraph", lambda: False)
    assert site._serves(2, [], {}) is True  # eager path unchanged
    site._capture_prepared.add(("dense", 2))
    monkeypatch.setattr(site_mod, "_in_hipgraph", lambda: True)
    assert site._serves(2, [], {}) is True


def test_provider_prepare_api_is_called_and_fail_open(monkeypatch):
    calls = []

    class _Prepared:
        def prepare(self, payload, *, group=None, device=None):
            calls.append((payload, group, device))
            return True

    monkeypatch.setattr(provider, "_load_provider", lambda name: _Prepared())
    config = SimpleNamespace(aiter_provider_enable=("sp-qkv-provider",))
    assert provider.prepare_aiter_providers(config, {"payload": 1}, group="g", device=2)
    assert calls == [({"payload": 1}, "g", 2)]

    class _NoAPI:
        pass

    monkeypatch.setattr(provider, "_load_provider", lambda name: _NoAPI())
    assert provider.prepare_aiter_providers(config, {"payload": 1}) is False


class _CaptureSite:
    mode = "dense"
    provider_enabled = True
    kv_ready = True
    name = "capture-site"

    def __init__(self):
        self._capture_prepared = set()
        self._capture_declined = set()
        self._serves_value = True

    def capture_gate_args(self, local_m, fwd_ctx):
        return [], {}

    def _serves(self, local_m, args, kwargs, *, capture_gate=False):
        assert capture_gate is True
        return self._serves_value

    def capture_prepare_payload(self, local_m, fwd_ctx):
        return {"sentinel": int(local_m)}

    def _group_value(self):
        return "group"


def test_prepare_capture_payload_once_per_cell(monkeypatch):
    site = _CaptureSite()
    calls = []
    monkeypatch.setattr(
        provider,
        "prepare_aiter_providers",
        lambda config, payload, group=None, device=None: (
            calls.append((payload, group, device)),
            True,
        )[1],
    )
    registry = GemmFusionSites(
        [site], SimpleNamespace(aiter_provider_enable=("sp-qkv-provider",))
    )
    registry.prepare_capture(2)
    registry.prepare_capture(2)
    assert calls == [({"sentinel": 2}, "group", None)]
    assert site._capture_prepared == {("dense", 2)}


def test_prepare_capture_false_falls_back_and_caches(monkeypatch):
    site = _CaptureSite()
    calls = []
    monkeypatch.setattr(
        provider,
        "prepare_aiter_providers",
        lambda config, payload, group=None, device=None: (
            calls.append(1),
            False,
        )[1],
    )
    registry = GemmFusionSites(
        [site], SimpleNamespace(aiter_provider_enable=("sp-qkv-provider",))
    )
    registry.prepare_capture(2)
    registry.prepare_capture(2)
    assert len(calls) == 1
    assert site._capture_declined == {("dense", 2)}
    assert site._capture_prepared == set()


def test_mark_kv_ready_clears_capture_prepare_cache(monkeypatch):
    site = _site()
    site._capture_prepared.add(("dense", 2))
    site._capture_declined.add(("dense", 3))
    site.mark_kv_ready()
    assert site._capture_prepared == set()
    assert site._capture_declined == set()


# ---------------------------------------------------------------- import proof
def test_no_removed_aiter_gemm_fusion_import_remains():
    root = Path(site_mod.__file__).resolve().parents[2]
    patterns = (
        "import aiter.gemm_fusion",
        "from aiter.gemm_fusion",
        "import aiter.ops.gemm_fusion",
        "from aiter.ops.gemm_fusion",
        "aiter.ops.gemm_fusion.",
        "aiter.gemm_fusion.",
    )
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern in text for pattern in patterns):
            offenders.append(str(path.relative_to(root)))
    assert offenders == []
