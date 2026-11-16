# SPDX-License-Identifier: MIT
"""CPU checks for the ATOM-side ``a2a_gemm_fused`` provider rail and fused seam.

No AITER, Triton or GPU is needed: the provider object, the routed op, the SP
group and the attention/quantizer tail are faked. Pins the required behaviour:

* default-off touches nothing (no aiter import, no configure, no op);
* the static deployment/weight gate matrix declines unsupported cases;
* the provider is warmed up once, after weight load, before the first routed
  call, and the class table is agreed with exactly ONE rank collective;
* the fused seam runs at the **pre-head-exchange** point: the routed call gets
  ``x`` = fp8 ``[4*M, 2048]`` send buffer, ``x_scale`` = contiguous fp32 ``[M]``,
  ``total_tokens == 4*M == x.shape[0]``, ``bucket_m == M``, the 16x16-preshuffled
  weight, a caller-owned bf16 ``[M, 6144]`` out, and ``stream=None``;
* in-band classes write the quantizer into the provider's registered producer
  view (``registered=True``); staged classes pass a plain buffer
  (``registered=False``);
* unsupported classes / capture / torch.compile tracing / provider declines
  keep ATOM's existing head-exchange + o_proj composition;
* a routed ``run`` error propagates -- there is no Python fallback after it;
* the ModelRunner hook order is device setup < configure < model build <
  install < warm-up/capture.
"""

import ast
import hashlib
import logging
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import atom.aiter_provider as provider_rail
import atom.model_ops.a2a_gemm as a2a

# The module registers its custom op for the CUDA dispatch key (production is
# CUDA-only); tests drive the same Python body on CPU by adding that impl.
try:
    torch.library.impl(
        "aiter::a2a_gemm_attend_project", "CPU", a2a.a2a_gemm_attend_project
    )
except Exception:  # noqa: BLE001, S110 - pragma: no cover; already registered
    pass

# The real helpers, captured before the `env` fixture replaces them: the
# regression tests that drive a REAL ForwardContext must not use the stubbed
# module attributes (that stub is exactly how the field mismatch escaped).
_REAL_IN_HIPGRAPH = a2a._in_hipgraph
_REAL_IS_DUMMY_RUN = a2a._is_dummy_run
_REAL_EAGER_FUSED = a2a._eager_fused_enabled

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The frozen neutral-op contract (aiter owns it; the pin is agreed across repos).
_CONTRACT_SIGNATURE = (
    "x:positional, x_scale:positional, w:positional, w_scale:positional, out:positional, "
    "total_tokens:keyword-only, bucket_m:keyword-only=None, registered:keyword-only=False, "
    "group:keyword-only=None, stream:keyword-only=None, -> None"
)
_CONTRACT_FINGERPRINT = "388d9816496c260e"

_PROVIDER = "a2a-gemm-provider"
_M = 2048  # registered band, direct dispatch class
_M_STAGED = 17  # staged class, direct dispatch class
_M_UNCOVERED = 3000  # inside the band but not a frozen class


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class _FakeProvider:
    """The provider object surface the rail/call site uses."""

    def __init__(
        self,
        coverage=None,
        registered_input=None,
        covers=None,
        policy_version=None,
    ):
        self.name = _PROVIDER
        self.op_versions = {a2a._OP_NAME: {1}}
        self.coverage_rows = (
            coverage
            if coverage is not None
            else (
                (1, 1, "staged"),
                (_M_STAGED, _M_STAGED, "staged"),
                (_M, _M, "registered"),
                (_M + 1, 8192, "registered"),  # a Pad seam: ATOM never serves it
            )
        )
        self.registered_input = registered_input or self._default_registered
        self.warmups = []
        # Newer providers serve arbitrary M through a covers(M) predicate; the
        # old 82+7 table remains the fallback when this is absent.
        if covers is not None:
            self.covers = covers
        if policy_version is not None:
            self.policy_version = policy_version

    def _default_registered(self, m):
        return torch.empty((a2a._ROWS_PER_M * m, a2a._SHARD), dtype=torch.float8_e4m3fn)

    def coverage(self):
        return self.coverage_rows

    def warmup(self, payload=None, *, group=None, device=None):
        self.warmups.append((payload, group, device))
        return True


class _FakeLinear:
    def __init__(self, *, shape=(6144, 8192), dtype=None, shuffled=True):
        dtype = torch.float8_e4m3fn if dtype is None else dtype
        # fp8 has no CPU fill op; the bytes of a uint8 buffer reinterpreted.
        self.weight = torch.empty(shape, dtype=torch.uint8).view(dtype)
        if shuffled:
            self.weight.is_shuffled = True
        # Production per-token FP8 o_proj layout: per-channel scale is (N, 1);
        # the frozen op contract wants a contiguous (N,), normalized at the call.
        self.weight_scale = torch.empty(shape[0], dtype=torch.float32).reshape(-1, 1)
        self.input_size = shape[1]
        self.output_size = shape[0]
        self.tp_size = 1
        self.quant_type = SimpleNamespace(value=2)  # QuantType.per_Token
        self.params_dtype = torch.float8_e4m3fn
        self.calls = 0
        self.last = None

    def __call__(self, x, x_scale=None):
        self.calls += 1
        self.last = torch.zeros((int(x.shape[0]), 6144), dtype=torch.bfloat16)
        return self.last


class _FakeAttention:
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.sp_head_exchange = None


class _FakeAttentionModule:
    def __init__(self, name, linear):
        self.attn = _FakeAttention(name)
        self.o_proj = linear


class _FakeModel:
    def __init__(self, modules):
        self._modules = modules

    def modules(self):
        return [self, *self._modules]


class _FakePlugin:
    API_VERSION = 1

    def __init__(self):
        self.configured = []

    def configure(self, *, source, enable):
        self.configured.append((source, tuple(enable)))

    def status(self):
        return {
            "enabled": (_PROVIDER,),
            "routed_ops": {a2a._OP_NAME: {"version": 1}},
            "skipped_providers": (),
            "api_versions_accepted": {1},
        }

    def explain(self, op, *args, **kwargs):
        return SimpleNamespace(decision=SimpleNamespace(route="provider"), declines={})


class _FakePcpGroup:
    """The SP group (AITER PCP rank dimension) as the fused path reads it."""

    world_size = a2a._SP_WORLD
    rank_in_group = 0
    device_group = "SP-GROUP"

    def __init__(self):
        self.device_communicator = SimpleNamespace(all_gatherv=lambda tensor: tensor)


class _FakeSpAttention:
    """Stand-in for atom.model_ops.sp_attention (needs Triton/AITER to import)."""

    def __init__(self):
        self.ulysses_calls = []
        self.gather_calls = []
        self.x = None

    def ulysses_attention(
        self, attn, q, k, v, positions, q_scale, qkv, output_transform=None
    ):
        self.ulysses_calls.append((attn, q, k, v, positions, q_scale, qkv))
        rows = a2a._ROWS_PER_M * int(q.shape[0])
        self.x = torch.zeros((rows, a2a._SHARD), dtype=torch.bfloat16)
        assert output_transform is not None
        return output_transform(self.x, rows)

    def gather_heads(self, x, rows, ca=None, scale=None):
        self.gather_calls.append((x, rows, ca, scale))
        return torch.empty(
            (int(rows) // a2a._SP_WORLD, a2a._K_IN), dtype=torch.float8_e4m3fn
        )


class _FakeTransport:
    """Stand-in for the pre-exchange FP8 transport (amax gather + quantizer)."""

    def __init__(self):
        self.calls = []
        self.override = None

    def gather_amax_and_quantize(self, x, padded_tokens, *, out=None, scale=None):
        self.calls.append((x, int(padded_tokens), out, scale))
        if self.override is not None:
            return self.override(x, padded_tokens, out, scale)
        padded_tokens = int(padded_tokens)
        q = (
            out
            if out is not None
            else torch.empty((padded_tokens, a2a._SHARD), dtype=torch.float8_e4m3fn)
        )
        if scale is None:
            scale = torch.empty(
                (padded_tokens // a2a._SP_WORLD, 1), dtype=torch.float32
            )
        return q, scale


def _fake_aiter(plugin, op):
    module = types.ModuleType("aiter")
    module.plugin = plugin
    module.a2a_gemm_fused = op
    return module


def _config(**over):
    base = {
        "enable_a2a_gemm": True,
        "aiter_provider_enable": (_PROVIDER,),
        "sequence_parallel_size": 4,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
    }
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def env(monkeypatch):
    """A process with a fake aiter/provider/SP group, no GPU, no real AITER."""
    state = SimpleNamespace()
    state.op_calls = []
    state.provider = _FakeProvider()
    state.plugin = _FakePlugin()
    state.pcp = _FakePcpGroup()
    state.sp = _FakeSpAttention()
    state.transport = _FakeTransport()

    def record(*args, **kwargs):
        state.op_calls.append({"args": args, "kwargs": kwargs})

    state.aiter_module = _fake_aiter(state.plugin, record)
    monkeypatch.setitem(sys.modules, "aiter", state.aiter_module)
    monkeypatch.setattr(provider_rail, "_load_provider", lambda name: state.provider)
    monkeypatch.setattr(a2a, "_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(a2a, "_device_index", lambda: 0)
    monkeypatch.setattr(a2a, "_pcp_group", lambda: state.pcp)
    # Single-rank install path by default; tests that exercise the collective
    # pass an explicit group to initialize_a2a_gemm.
    monkeypatch.setattr(a2a, "_group_or_none", lambda: None)
    monkeypatch.setattr(a2a, "_sp_attention", lambda: state.sp)
    monkeypatch.setattr(a2a, "_transport", lambda: state.transport)
    monkeypatch.setattr(a2a, "_on_device", lambda tensor: True)
    monkeypatch.setattr(a2a, "_in_hipgraph", lambda: False)
    monkeypatch.setattr(a2a, "_is_dummy_run", lambda: False)
    monkeypatch.setattr(a2a, "_is_compiling", lambda: False)
    # The fused-eager suite exercises the opt-in path; the default-off regression
    # restores the real helper and drives the env flag explicitly.
    monkeypatch.setattr(a2a, "_eager_fused_enabled", lambda: True)
    monkeypatch.setattr(a2a, "_PLUGIN", None)
    monkeypatch.setattr(a2a, "_ENSURE_DONE", False)
    monkeypatch.setattr(a2a, "_SITES", {})
    monkeypatch.setattr(a2a, "_OUTPUTS", {})
    monkeypatch.setattr(a2a, "_EAGER_SEND", {})
    monkeypatch.setattr(a2a, "_EAGER_SCALE", {})
    monkeypatch.setattr(a2a, "_PRODUCER_PTRS", set())

    def fake_attention_modules(model):
        yield from model._modules

    monkeypatch.setattr(a2a, "_minimax_m3_attention_modules", fake_attention_modules)
    return state


def _install(env, config=None, *, modules=None, group=None):
    linear = _FakeLinear()
    modules = (
        modules
        if modules is not None
        else [_FakeAttentionModule("model.layers.0.self_attn", linear)]
    )
    model = _FakeModel(modules)
    sites = a2a.initialize_a2a_gemm(
        model, config if config is not None else _config(), group=group, device=0
    )
    return model, sites, linear


def _fake_q(m=_M_STAGED):
    return torch.empty((m, a2a._K_IN), dtype=torch.bfloat16)


def _pre_exchange_x(m=_M_STAGED):
    return torch.zeros((a2a._ROWS_PER_M * m, a2a._SHARD), dtype=torch.bfloat16)


def _install_full(env, **kwargs):
    """Install and return ``(sites, site, attn, linear)``."""
    _, sites, linear = _install(env, **kwargs)
    assert sites is not None
    site = sites.sites[0]
    site._group = "SP-GROUP"
    return sites, site, site.attn, linear


def _pose_site(env):
    """Install a site and return ``(site, attn, linear)``."""
    _, site, attn, linear = _install_full(env)
    return site, attn, linear


# ---------------------------------------------------------------------------
# the frozen contract pin
# ---------------------------------------------------------------------------
def test_contract_pin_matches_the_frozen_neutral_signature():
    assert a2a._OP_NAME == "a2a_gemm_fused"
    assert a2a._OP_VERSION == 1
    assert a2a._FINGERPRINT == _CONTRACT_FINGERPRINT
    derived = hashlib.sha256(_CONTRACT_SIGNATURE.encode("utf-8")).hexdigest()[:16]
    assert derived == _CONTRACT_FINGERPRINT
    assert provider_rail._EXPECTED_ROUTED_OPS == ("a2a_gemm_fused",)
    # Pre-exchange x: four 2048-wide head shards per real token.
    assert a2a._ROWS_PER_M == 4 and a2a._SHARD == 2048 and a2a._K_IN == 8192


def test_provider_rail_has_no_module_scope_aiter_or_torch_import():
    tree = ast.parse(
        (_REPO_ROOT / "atom" / "aiter_provider.py").read_text(encoding="utf-8")
    )
    module_level = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    roots = set()
    for node in module_level:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif node.module:
            roots.add(node.module.split(".")[0])
    assert "aiter" not in roots
    assert "torch" not in roots


def test_a2a_gemm_has_no_module_scope_triton_or_aiter_import():
    tree = ast.parse((_REPO_ROOT / "atom" / "model_ops" / "a2a_gemm.py").read_text())
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert "aiter" not in roots
    assert "triton" not in roots


# ---------------------------------------------------------------------------
# enablement: default off
# ---------------------------------------------------------------------------
def test_disabled_switch_touches_nothing(monkeypatch):
    monkeypatch.delitem(sys.modules, "aiter", raising=False)
    assert (
        a2a.initialize_a2a_gemm(
            object(), _config(enable_a2a_gemm=False), group=None, device=0
        )
        is None
    )
    assert a2a._SITES == {}


def test_enabled_without_a_provider_name_declines(monkeypatch):
    monkeypatch.delitem(sys.modules, "aiter", raising=False)
    assert (
        a2a.initialize_a2a_gemm(
            object(), _config(aiter_provider_enable=()), group=None, device=0
        )
        is None
    )


def test_normalize_provider_names():
    assert provider_rail.normalize_provider_names(None) == ()
    assert provider_rail.normalize_provider_names("") == ()
    assert provider_rail.normalize_provider_names(" a, ,b ,a ") == ("a", "b")
    assert provider_rail.normalize_provider_names(["a", None, " b "]) == ("a", "b")
    with pytest.raises(TypeError):
        provider_rail.normalize_provider_names(["a", 3])


# ---------------------------------------------------------------------------
# the static gate matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"sequence_parallel_size": 2}, "sp-world"),
        ({"tensor_parallel_size": 2}, "tp-world"),
        ({"pipeline_parallel_size": 2}, "pipeline-parallel"),
        ({"data_parallel_size": 2}, "data-parallel"),
    ],
)
def test_deployment_gate_matrix(env, overrides, reason):
    assert a2a.deployment_reason(_config(**overrides)) == reason


def test_deployment_gate_requires_gfx950(env, monkeypatch):
    monkeypatch.setattr(a2a, "_gfx_runtime", lambda: "gfx942")
    assert a2a.deployment_reason(_config()) == "gfx"
    monkeypatch.setattr(a2a, "_gfx_runtime", lambda: None)
    assert a2a.deployment_reason(_config()) == "gfx-unknown"
    monkeypatch.setattr(a2a, "_gfx_runtime", lambda: "gfx950")
    assert a2a.deployment_reason(_config()) is None


def test_unsupported_deployment_installs_nothing(env):
    assert (
        a2a.initialize_a2a_gemm(
            object(), _config(sequence_parallel_size=8), group=None, device=0
        )
        is None
    )


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"shape": (6144, 4096)}, "weight-shape"),
        ({"dtype": torch.uint8}, "weight-dtype"),
        ({"shuffled": False}, "weight-layout"),
    ],
)
def test_static_weight_gate_matrix(env, kwargs, reason):
    assert a2a.static_weight_reason(_FakeLinear(**kwargs)) == reason


def test_weight_scale_and_input_gate(env):
    linear = _FakeLinear()
    linear.weight_scale = torch.empty(1024, dtype=torch.float32)
    assert a2a.static_weight_reason(linear) == "weight-scale"
    linear.weight_scale = torch.empty(6144, dtype=torch.bfloat16)
    assert a2a.static_weight_reason(linear) == "weight-scale"
    linear = _FakeLinear()
    linear.input_size = 4096
    assert a2a.static_weight_reason(linear) == "input-size"


def test_ineligible_layer_is_not_installed_but_eligible_one_is(env):
    good = _FakeLinear()
    bad = _FakeLinear(shape=(6144, 4096))
    modules = [
        _FakeAttentionModule("model.layers.1.self_attn", good),
        _FakeAttentionModule("model.layers.2.self_attn", bad),
    ]
    _, sites, _ = _install(env, modules=modules)
    assert sites is not None
    assert [s.name for s in sites.sites] == ["model.layers.1.self_attn"]
    assert "model.layers.1.self_attn" in a2a._SITES
    assert "model.layers.2.self_attn" not in a2a._SITES


def test_non_per_token_fp8_o_proj_is_not_installed(env):
    linear = _FakeLinear()
    linear.quant_type = SimpleNamespace(value=1)
    _, sites, _ = _install(
        env, modules=[_FakeAttentionModule("model.layers.0.self_attn", linear)]
    )
    assert sites is None


# ---------------------------------------------------------------------------
# warm-up and the one rank-uniform collective
# ---------------------------------------------------------------------------
def test_install_warms_provider_before_agreeing_classes(env, monkeypatch):
    events = []
    orig_warm = env.provider.warmup

    def warm(payload=None, *, group=None, device=None):
        events.append("warmup")
        return orig_warm(payload, group=group, device=device)

    env.provider.warmup = warm
    orig_agree = a2a.agree_class_table

    def agree(payload, group):
        events.append("agree")
        assert payload["deployment"] is True
        return orig_agree(payload, group)

    monkeypatch.setattr(a2a, "agree_class_table", agree)
    _, sites, _ = _install(env)
    assert sites is not None
    assert events == ["warmup", "agree"]
    assert env.provider.warmups and env.provider.warmups[0][0] is None


def _fake_gather(monkeypatch, verdicts):
    calls = []

    def all_gather_object(output, obj, group=None):
        calls.append(obj)
        output[:] = verdicts

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: len(verdicts)
    )
    return calls


def test_rank_agreement_accepts_identical_payloads(monkeypatch):
    payload = {"deployment": True, "sites": ("a",), "direct": (1, 2)}
    _fake_gather(monkeypatch, [payload, payload, payload, payload])
    assert a2a.agree_class_table(payload, object()) == payload


def test_rank_disagreement_declines_on_every_rank(monkeypatch):
    payload = {"deployment": True, "sites": ("a",), "direct": (1,)}
    other = {"deployment": True, "sites": ("a",), "direct": (2,)}
    _fake_gather(monkeypatch, [payload, payload, other, payload])
    assert a2a.agree_class_table(payload, object()) is None


def test_rank_veto_declines_on_every_rank(monkeypatch):
    veto = {"deployment": False, "sites": (), "direct": ()}
    payload = {"deployment": True, "sites": ("a",), "direct": (1,)}
    _fake_gather(monkeypatch, [payload, veto, payload, payload])
    assert a2a.agree_class_table(payload, object()) is None


def test_install_does_exactly_one_rank_collective_and_caches_classes(env, monkeypatch):
    payload = {
        "deployment": True,
        "sites": ("model.layers.0.self_attn",),
        "direct": (1, _M_STAGED, _M),
        "registered": (_M,),
    }
    calls = _fake_gather(monkeypatch, [payload] * 4)
    _, sites, _ = _install(env, group=object())
    assert sites is not None
    assert len(calls) == 1
    site = sites.sites[0]
    assert site.direct == {1: 1, _M_STAGED: _M_STAGED, _M: _M}
    assert site.registered == frozenset({_M})
    # The class table is cached: two eager gate checks never collect again.
    attn, linear = site.attn, site.linear
    assert a2a.attends_via_routed_op(attn, linear, _fake_q(_M_STAGED))
    assert a2a.attends_via_routed_op(attn, linear, _fake_q(_M_STAGED))
    assert len(calls) == 1


def test_rank_disagreement_installs_nothing(env, monkeypatch):
    payload = {
        "deployment": True,
        "sites": ("model.layers.0.self_attn",),
        "direct": (1, _M_STAGED, _M),
        "registered": (_M,),
    }
    differing = dict(payload, direct=(1,))
    _fake_gather(monkeypatch, [payload, differing, payload, payload])
    _, sites, _ = _install(env, group=object())
    assert sites is None
    assert a2a._SITES == {}


def test_provider_without_coverage_declines_everything(env):
    env.provider.coverage_rows = ()
    _, sites, _ = _install(env)
    assert sites is None
    assert a2a._SITES == {}


def test_provider_warmup_failure_declines_everything(env):
    env.provider.warmup = lambda payload=None, *, group=None, device=None: False
    _, sites, _ = _install(env)
    assert sites is None


def test_provider_without_routed_op_declines(env):
    env.plugin.status = lambda: {
        "enabled": (_PROVIDER,),
        "routed_ops": {},
        "skipped_providers": (),
    }
    _, sites, _ = _install(env)
    assert sites is None


# ---------------------------------------------------------------------------
# provider coverage: generalized covers(M) policy + old static table fallback
# ---------------------------------------------------------------------------
def _use_provider(env, monkeypatch, provider):
    env.provider = provider
    monkeypatch.setattr(provider_rail, "_load_provider", lambda name: provider)
    return provider


def test_install_accepts_a_covers_predicate_without_a_static_table(env, monkeypatch):
    provider = _use_provider(
        env,
        monkeypatch,
        _FakeProvider(
            coverage=(),
            covers=lambda m: 1 <= int(m) <= 16384,
            policy_version="frozen-rule-v2",
        ),
    )
    # coverage() is empty, so the static table is not the decision.
    assert a2a._normalize_coverage(provider) == ({}, set())
    _, sites, _ = _install(env)
    assert sites is not None
    site = sites.sites[0]
    assert site.covers is not None
    assert site.covered(3000) is True and site.can_serve(3000) is True
    assert site.covered(9999) is True
    assert site.can_serve(_M) is True  # registered band + provider producer view


def test_covers_predicate_arbitrary_m_reaches_the_routed_call(env, monkeypatch):
    _use_provider(
        env,
        monkeypatch,
        _FakeProvider(coverage=(), covers=lambda m: 1 <= int(m) <= 16384),
    )
    _site, attn, linear = _pose_site(env)
    # Out-of-band real M: staged ingress, arbitrary bucket.
    m = 9999
    assert a2a.attends_via_routed_op(attn, linear, _fake_q(m)) is True
    a2a.fuse_attention_o_proj(
        attn, linear, _fake_q(m), _fake_q(), _fake_q(), None, _fake_q()
    )
    call = env.op_calls[0]
    assert call["kwargs"]["bucket_m"] == m
    assert call["kwargs"]["total_tokens"] == a2a._ROWS_PER_M * m
    assert call["kwargs"]["registered"] is False
    assert tuple(call["args"][0].shape) == (a2a._ROWS_PER_M * m, a2a._SHARD)
    assert tuple(call["args"][4].shape) == (m, a2a._N_OUT)


def test_covers_predicate_in_band_arbitrary_m_uses_the_registered_view(
    env, monkeypatch
):
    _use_provider(
        env,
        monkeypatch,
        _FakeProvider(coverage=(), covers=lambda m: 1 <= int(m) <= 16384),
    )
    site, attn, linear = _pose_site(env)
    m = 3000  # in the producer-registered band, arbitrary (unverified) M
    assert site.covered(m) is True and site.can_serve(m) is True
    a2a.fuse_attention_o_proj(
        attn, linear, _fake_q(m), _fake_q(), _fake_q(), None, _fake_q()
    )
    call = env.op_calls[0]
    assert call["kwargs"]["registered"] is True
    assert call["args"][0] is site._producer_view(m)
    assert call["kwargs"]["bucket_m"] == m
    assert call["kwargs"]["total_tokens"] == a2a._ROWS_PER_M * m


def test_covers_predicate_failure_falls_back_to_the_static_table(env, monkeypatch):
    def broken(m):
        raise RuntimeError("predicate unavailable")

    _use_provider(
        env,
        monkeypatch,
        _FakeProvider(coverage=((_M_STAGED, _M_STAGED, "staged"),), covers=broken),
    )
    site, _attn, _linear = _pose_site(env)
    # The static table is the fallback: its class serves, anything else declines.
    assert site.covered(_M_STAGED) is True and site.can_serve(_M_STAGED) is True
    assert site.covered(_M_UNCOVERED) is False
    assert site.can_serve(_M_UNCOVERED) is False
    site._output_transform(
        _pre_exchange_x(_M_UNCOVERED), a2a._ROWS_PER_M * _M_UNCOVERED
    )
    assert env.op_calls == []


def test_table_only_provider_keeps_the_old_82_class_behaviour(env):
    # No covers() on this provider: direct table hits serve, Pad seams and
    # non-covered M stay native. The model-side gate is metadata-only now; the
    # runtime coverage decision keeps the uncovered classes native.
    site, attn, linear = _pose_site(env)
    assert site.covered(_M_STAGED) is True and site.covered(_M) is True
    assert site.covered(_M_UNCOVERED) is False and site.covered(_M + 1) is False
    assert a2a.attends_via_routed_op(attn, linear, _fake_q(_M_STAGED)) is True
    site._output_transform(
        _pre_exchange_x(_M_UNCOVERED), a2a._ROWS_PER_M * _M_UNCOVERED
    )
    assert env.op_calls == []
    assert env.sp.gather_calls


def test_provider_with_neither_coverage_nor_covers_declines(env, monkeypatch):
    provider = SimpleNamespace(
        op_versions={a2a._OP_NAME: {1}},
        registered_input=lambda m: torch.empty(
            (a2a._ROWS_PER_M * m, a2a._SHARD), dtype=torch.float8_e4m3fn
        ),
        warmup=lambda payload=None, *, group=None, device=None: True,
    )
    _use_provider(env, monkeypatch, provider)
    _, sites, _ = _install(env)
    assert sites is None


def test_rank_vote_payload_carries_policy_and_covers_mode(env, monkeypatch):
    seen = {}

    def agree(payload, group):
        seen.update(payload)
        return payload

    monkeypatch.setattr(a2a, "agree_class_table", agree)
    _use_provider(
        env,
        monkeypatch,
        _FakeProvider(covers=lambda m: True, policy_version="frozen-rule-v2"),
    )
    _, sites, _ = _install(env)
    assert sites is not None
    assert seen["covers"] is True
    assert seen["policy"] == ("policy_version", "frozen-rule-v2")


def test_old_table_provider_payload_has_no_covers(env, monkeypatch):
    seen = {}

    def agree(payload, group):
        seen.update(payload)
        return payload

    monkeypatch.setattr(a2a, "agree_class_table", agree)
    _install(env)
    assert seen["covers"] is False
    assert seen["policy"] is None
    assert seen["direct"] == (1, _M_STAGED, _M)


def test_policy_disagreement_declines_everywhere(monkeypatch):
    payload = {
        "deployment": True,
        "sites": ("a",),
        "policy": ("policy_version", "v1"),
        "covers": True,
        "direct": (),
        "registered": (),
    }
    other = dict(payload, policy=("policy_version", "v2"))
    _fake_gather(monkeypatch, [payload, other, payload, payload])
    assert a2a.agree_class_table(payload, object()) is None


def test_policy_version_callable_is_resolved_and_rank_comparable():
    class _PolicyProvider:
        def policy_version(self):
            return "r9_frozen_select_config.v1+rv1+38c6e1224d20f085"

    one = a2a._provider_policy(_PolicyProvider())
    two = a2a._provider_policy(_PolicyProvider())
    assert one == (
        "policy_version",
        "r9_frozen_select_config.v1+rv1+38c6e1224d20f085",
    )
    # A bound method's str carries its address; the resolved value must not.
    assert "0x" not in one[1]
    assert one == two


def test_policy_version_failure_is_not_voted():
    class _Broken:
        def policy_version(self):
            raise RuntimeError("identity unavailable")

    assert a2a._provider_policy(_Broken()) is None


# ---------------------------------------------------------------------------
# the model-side gate
# ---------------------------------------------------------------------------
def test_gate_requires_an_installed_site(env):
    attn = _FakeAttention("model.layers.0.self_attn")
    linear = _FakeLinear()
    assert a2a.attends_via_routed_op(attn, linear, _fake_q(_M_STAGED)) is False


def test_gate_true_for_covered_eager_class(env):
    _site, attn, linear = _pose_site(env)
    assert a2a.attends_via_routed_op(attn, linear, _fake_q(_M_STAGED)) is True
    assert a2a.attends_via_routed_op(attn, linear, _fake_q(1)) is True
    assert a2a.attends_via_routed_op(attn, linear, _fake_q(_M)) is True


def test_gate_is_static_so_uncovered_classes_still_enter_the_op(env):
    # The model-side branch is metadata-only now: the opaque op makes the
    # dynamic (coverage/capture/dummy) decision at runtime, so tracing is
    # stable. Uncovered classes simply fall back to native inside the op.
    _site, attn, linear = _pose_site(env)
    assert a2a.attends_via_routed_op(attn, linear, _fake_q(_M_UNCOVERED)) is True
    assert a2a.attends_via_routed_op(attn, linear, _fake_q(_M + 1)) is True
    # The runtime decision still keeps them native.
    assert _site.covered(_M_UNCOVERED) is False
    assert _site.can_serve(_M + 1) is False


@pytest.mark.parametrize(
    "override, value",
    [
        ("_is_compiling", lambda: True),
        ("_in_hipgraph", lambda: True),
        ("_is_dummy_run", lambda: True),
    ],
)
def test_gate_is_static_under_tracing_capture_and_dummy_runs(
    env, monkeypatch, override, value
):
    # torch.compile tracing must serve the fused path (custom op), and capture
    # eligibility is decided in the eager capture-prep phase, not by the gate.
    _site, attn, linear = _pose_site(env)
    monkeypatch.setattr(a2a, override, value)
    assert a2a.attends_via_routed_op(attn, linear, _fake_q(_M_STAGED)) is True


def test_gate_false_for_wrong_q_or_o_proj_shape(env):
    _site, attn, linear = _pose_site(env)
    assert a2a.attends_via_routed_op(attn, linear, _fake_q().to(torch.float16)) is False
    bad_q = torch.empty((_M_STAGED, 2048), dtype=torch.bfloat16)
    assert a2a.attends_via_routed_op(attn, linear, bad_q) is False
    linear.input_size = 4096
    assert a2a.attends_via_routed_op(attn, linear, _fake_q(_M_STAGED)) is False


def test_registered_class_without_producer_view_is_runtime_native(env):
    def no_view(m):
        raise RuntimeError("not warmed up")

    env.provider.registered_input = no_view
    site, attn, linear = _pose_site(env)
    # Static gate still admits the call; the runtime decision declines it.
    assert a2a.attends_via_routed_op(attn, linear, _fake_q(_M)) is True
    assert site.can_serve(_M) is False
    assert (
        site._output_transform(_pre_exchange_x(_M), a2a._ROWS_PER_M * _M)[0] is not None
    )
    assert env.op_calls == []
    assert linear.calls == 1


# ---------------------------------------------------------------------------
# the routed call (pre-exchange seam)
# ---------------------------------------------------------------------------
def test_fused_call_uses_the_pre_exchange_send_buffer(env):
    _site, attn, linear = _pose_site(env)
    q, k, v = _fake_q(_M_STAGED), _fake_q(_M_STAGED), _fake_q(_M_STAGED)
    out = a2a.fuse_attention_o_proj(
        attn, linear, q, k, v, torch.empty(4 * _M_STAGED, dtype=torch.int64), q
    )
    assert len(env.sp.ulysses_calls) == 1
    # x handed to the transform is the [4*M, 2048] pre-exchange BF16 send buffer.
    assert env.sp.x.shape == (a2a._ROWS_PER_M * _M_STAGED, a2a._SHARD)
    assert len(env.op_calls) == 1
    call = env.op_calls[0]
    x_call, x_scale, w, w_scale, out_arg = call["args"]
    assert x_call.dtype == torch.float8_e4m3fn
    assert tuple(x_call.shape) == (a2a._ROWS_PER_M * _M_STAGED, a2a._SHARD)
    assert tuple(x_scale.shape) == (_M_STAGED,)
    assert x_scale.dtype == torch.float32 and x_scale.is_contiguous()
    assert w is linear.weight
    # The (6144,1) o_proj scale reaches the op as the contract's contiguous (6144,).
    assert tuple(w_scale.shape) == (a2a._N_OUT,)
    assert w_scale.dtype == torch.float32 and w_scale.is_contiguous()
    assert w_scale.data_ptr() == linear.weight_scale.data_ptr()
    assert out_arg.dtype == torch.bfloat16
    assert tuple(out_arg.shape) == (_M_STAGED, a2a._N_OUT)
    assert call["kwargs"] == {
        "total_tokens": a2a._ROWS_PER_M * _M_STAGED,
        "bucket_m": _M_STAGED,
        "registered": False,
        "group": "SP-GROUP",
        "stream": None,
    }
    assert call["kwargs"]["total_tokens"] == x_call.shape[0]
    assert out is out_arg


def test_weight_scale_2d_is_flattened_to_the_contract_shape(env):
    _site, attn, linear = _pose_site(env)
    # Production per-token FP8 o_proj layout: (6144, 1).
    values = torch.arange(a2a._N_OUT, dtype=torch.float32)
    linear.weight_scale = values.reshape(a2a._N_OUT, 1)
    assert tuple(linear.weight_scale.shape) == (a2a._N_OUT, 1)

    a2a.fuse_attention_o_proj(
        attn, linear, _fake_q(_M_STAGED), _fake_q(), _fake_q(), None, _fake_q()
    )
    call = env.op_calls[0]
    w_scale = call["args"][3]
    assert tuple(w_scale.shape) == (a2a._N_OUT,)
    assert w_scale.dtype == torch.float32
    assert w_scale.is_contiguous()
    # Flattening the contiguous (N,1) production layout is a zero-copy view.
    assert w_scale.data_ptr() == linear.weight_scale.data_ptr()
    assert torch.equal(w_scale, values)
    assert call["args"][2] is linear.weight
    assert call["kwargs"]["bucket_m"] == _M_STAGED


def test_weight_scale_already_1d_is_unchanged(env):
    _site, attn, linear = _pose_site(env)
    values = torch.arange(a2a._N_OUT, dtype=torch.float32)
    linear.weight_scale = values
    a2a.fuse_attention_o_proj(
        attn, linear, _fake_q(_M_STAGED), _fake_q(), _fake_q(), None, _fake_q()
    )
    w_scale = env.op_calls[0]["args"][3]
    assert tuple(w_scale.shape) == (a2a._N_OUT,)
    assert w_scale.is_contiguous()
    assert w_scale.data_ptr() == linear.weight_scale.data_ptr()
    assert torch.equal(w_scale, values)


def test_weight_scale_wrong_numel_is_native_before_the_call(env):
    site, _attn, linear = _pose_site(env)
    linear.weight_scale = torch.empty(4096, dtype=torch.float32)
    site._output_transform(
        torch.zeros((a2a._ROWS_PER_M * _M_STAGED, a2a._SHARD), dtype=torch.bfloat16),
        a2a._ROWS_PER_M * _M_STAGED,
    )
    assert env.op_calls == []
    assert linear.calls == 1  # native head exchange + o_proj


def test_weight_scale_wrong_dtype_is_native_before_the_call(env):
    site, _attn, linear = _pose_site(env)
    linear.weight_scale = torch.empty(a2a._N_OUT, dtype=torch.bfloat16)
    site._output_transform(
        torch.zeros((a2a._ROWS_PER_M * _M_STAGED, a2a._SHARD), dtype=torch.bfloat16),
        a2a._ROWS_PER_M * _M_STAGED,
    )
    assert env.op_calls == []
    assert linear.calls == 1


def test_weight_scale_non_contiguous_is_normalized_before_the_call(env):
    _site, attn, linear = _pose_site(env)
    base = torch.arange(2 * a2a._N_OUT, dtype=torch.float32).reshape(a2a._N_OUT, 2)
    # (N,1) slice with stride (2,1): not contiguous, flattened to a contiguous
    # contract-shaped copy rather than reaching the op as a bad layout.
    linear.weight_scale = base[:, :1]
    assert tuple(linear.weight_scale.shape) == (a2a._N_OUT, 1)
    assert not linear.weight_scale.is_contiguous()
    a2a.fuse_attention_o_proj(
        attn, linear, _fake_q(_M_STAGED), _fake_q(), _fake_q(), None, _fake_q()
    )
    w_scale = env.op_calls[0]["args"][3]
    assert tuple(w_scale.shape) == (a2a._N_OUT,)
    assert w_scale.is_contiguous()
    assert torch.equal(w_scale, linear.weight_scale.reshape(-1))


def test_contract_weight_scale_helper_contract(env):
    good = torch.arange(a2a._N_OUT, dtype=torch.float32).reshape(a2a._N_OUT, 1)
    flat = a2a._contract_weight_scale(good)
    assert tuple(flat.shape) == (a2a._N_OUT,) and flat.data_ptr() == good.data_ptr()
    assert a2a._contract_weight_scale(None) is None
    assert a2a._contract_weight_scale(torch.empty(4096)) is None
    assert (
        a2a._contract_weight_scale(torch.empty(a2a._N_OUT, dtype=torch.bfloat16))
        is None
    )


def test_registered_class_writes_the_quantizer_into_the_producer_view(env):
    site, attn, linear = _pose_site(env)
    out = a2a.fuse_attention_o_proj(
        attn, linear, _fake_q(_M), _fake_q(_M), _fake_q(_M), None, _fake_q(_M)
    )
    view = site._producer_view(_M)
    assert view is not None
    assert env.transport.calls and env.transport.calls[0][2] is view
    call = env.op_calls[0]
    assert call["args"][0] is view
    assert a2a.owns_producer_pointer(view)
    assert call["kwargs"]["registered"] is True
    assert call["kwargs"]["bucket_m"] == _M
    assert out is call["args"][4]


def _enable_real_eager_flag(monkeypatch, value):
    """Restore the real eager opt-in helper and set ATOM_A2A_GEMM_EAGER_FUSED."""
    from atom.utils import envs

    monkeypatch.setattr(a2a, "_eager_fused_enabled", _REAL_EAGER_FUSED)
    monkeypatch.setattr(envs, "ATOM_A2A_GEMM_EAGER_FUSED", bool(value))


def test_noncapture_eager_tail_is_native_by_default(env, monkeypatch):
    # Deployment default: fused mode is the captured graph; a non-captured
    # prefill would pay a per-new-M-class provider bind, so it stays native.
    _enable_real_eager_flag(monkeypatch, False)
    site, _attn, linear = _pose_site(env)
    rows = a2a._ROWS_PER_M * _M_STAGED
    out = site._output_transform(_pre_exchange_x(_M_STAGED), rows)
    assert out[0] is not None
    # No routed op -> no provider bind inside the eager forward.
    assert env.op_calls == []
    # ATOM's native composition ran, and no eager scratch was allocated.
    assert env.sp.gather_calls and linear.calls == 1
    assert a2a._EAGER_SEND == {} and a2a._EAGER_SCALE == {}


def test_noncapture_eager_flag_opts_into_the_fused_pointer_stable_path(
    env, monkeypatch
):
    _enable_real_eager_flag(monkeypatch, True)
    site, _attn, _linear = _pose_site(env)
    rows = a2a._ROWS_PER_M * _M_STAGED
    site._output_transform(_pre_exchange_x(_M_STAGED), rows)
    site._output_transform(_pre_exchange_x(_M_STAGED), rows)
    assert len(env.op_calls) == 2
    first, second = env.op_calls[0]["args"], env.op_calls[1]["args"]
    assert first[0] is second[0] and first[4] is second[4]
    assert first[1].data_ptr() == second[1].data_ptr()
    assert (
        len(
            {
                (a[0].data_ptr(), a[1].data_ptr(), a[4].data_ptr())
                for a in (first, second)
            }
        )
        == 1
    )


def test_capture_serves_even_when_eager_fusion_is_off(env, monkeypatch):
    # The design boundary: captured graphs serve the fused tail regardless of
    # the non-captured opt-in.
    _enable_real_eager_flag(monkeypatch, False)
    sites, site, _attn, _linear = _install_full(env)
    sites.begin_capture()
    assert sites.prepare_capture(_M_STAGED) == 1
    cell = site._capture_cells[_M_STAGED]
    monkeypatch.setattr(a2a, "_in_hipgraph", lambda: True)
    out = site._output_transform(
        _pre_exchange_x(_M_STAGED), a2a._ROWS_PER_M * _M_STAGED
    )
    assert out[0] is cell.out
    assert len(env.op_calls) == 2
    assert env.op_calls[-1]["args"][4] is cell.out
    assert site.capture_served == 1


def test_phaseF_noncapture_eager_buffers_are_pointer_stable(env):
    """Phase F prefill regression: eager send/scale/out must not move.

    The provider keys its prepared call on (x, x_scale, w, w_scale, out), so a
    fresh per-call send/scale tensor makes it re-run its collective SP4 bind on
    every layer/prefill chunk. Cached eager buffers keep the key identical
    (content overwritten in place), so the second call is a binding hit.
    """
    site, _attn, _linear = _pose_site(env)
    x = _pre_exchange_x(_M_STAGED)
    rows = a2a._ROWS_PER_M * _M_STAGED
    site._output_transform(x, rows)
    first = env.op_calls[0]["args"]
    first_transport = env.transport.calls[-1]
    site._output_transform(x, rows)
    second = env.op_calls[1]["args"]
    second_transport = env.transport.calls[-1]

    # Identical send/scale/out buffers on both eager calls.
    assert first[0] is second[0]
    assert first[1].data_ptr() == second[1].data_ptr()
    assert first[4] is second[4]
    # The quantizer wrote into those same cached buffers, not fresh ones.
    assert first_transport[2] is first[0] and second_transport[2] is second[0]
    # The routed call's x_scale is the cached cell reshaped to (M,): same
    # storage, view object only.
    assert first_transport[3].data_ptr() == first[1].data_ptr()
    assert second_transport[3] is first_transport[3]
    assert first[1].data_ptr() == second[1].data_ptr()
    # One unique provider binding key (the provider binds once per class).
    keys = {
        (c["args"][0].data_ptr(), c["args"][1].data_ptr(), c["args"][4].data_ptr())
        for c in env.op_calls
    }
    assert len(keys) == 1
    # Send and scale are distinct cells, and scaled content changes per call
    # without the pointers moving.
    assert first[0].data_ptr() != first[1].data_ptr()

    # A different M class gets its own (also stable) cell.
    x2 = _pre_exchange_x(_M)
    site._output_transform(x2, a2a._ROWS_PER_M * _M)
    other = env.op_calls[-1]["args"]
    assert other[0] is not first[0] and other[4] is not first[4]
    site._output_transform(x2, a2a._ROWS_PER_M * _M)
    assert env.op_calls[-1]["args"][0] is other[0]
    assert env.op_calls[-1]["args"][4] is other[4]


def test_registered_class_without_producer_view_stays_native(env):
    # The gate already declines this call; the tail must also stay native if the
    # view disappears between the gate and the transform.
    site, _attn, linear = _pose_site(env)
    site._producer_view = lambda m: None  # type: ignore[method-assign]
    assert site.can_serve(_M) is False
    site._output_transform(
        torch.zeros((a2a._ROWS_PER_M * _M, a2a._SHARD), dtype=torch.bfloat16),
        a2a._ROWS_PER_M * _M,
    )
    assert env.op_calls == []
    assert env.sp.gather_calls  # native head exchange + o_proj
    assert linear.calls == 1


def test_uncovered_class_in_transform_stays_native(env):
    site, _attn, linear = _pose_site(env)
    site._output_transform(
        torch.zeros((a2a._ROWS_PER_M * _M_UNCOVERED, a2a._SHARD), dtype=torch.bfloat16),
        a2a._ROWS_PER_M * _M_UNCOVERED,
    )
    assert env.op_calls == []
    assert env.sp.gather_calls
    assert linear.calls == 1


def test_bad_quantizer_result_stays_native(env):
    site, _attn, linear = _pose_site(env)

    def bad(x, padded, out, scale):
        return torch.empty((1, 1), dtype=torch.float8_e4m3fn), None

    env.transport.override = bad
    site._output_transform(
        torch.zeros((a2a._ROWS_PER_M * _M_STAGED, a2a._SHARD), dtype=torch.bfloat16),
        a2a._ROWS_PER_M * _M_STAGED,
    )
    assert env.op_calls == []
    assert linear.calls == 1


def test_bad_pre_exchange_shape_stays_native(env):
    site, _attn, linear = _pose_site(env)
    site._output_transform(
        torch.zeros((4, 4096), dtype=torch.bfloat16), a2a._ROWS_PER_M * _M_STAGED
    )
    assert env.op_calls == []
    assert linear.calls == 1


def test_routed_run_error_propagates_without_native_retry(env):
    def boom(*args, **kwargs):
        raise RuntimeError("kernel exploded")

    env.aiter_module.a2a_gemm_fused = boom
    _site, attn, linear = _pose_site(env)
    with pytest.raises(RuntimeError, match="kernel exploded"):
        a2a.fuse_attention_o_proj(
            attn, linear, _fake_q(_M_STAGED), _fake_q(), _fake_q(), None, _fake_q()
        )
    assert linear.calls == 0
    assert env.sp.gather_calls == []


def test_shared_output_buffer_is_one_per_class(env):
    _site, attn, linear = _pose_site(env)
    a = a2a.fuse_attention_o_proj(
        attn, linear, _fake_q(_M_STAGED), _fake_q(), _fake_q(), None, _fake_q()
    )
    b = a2a.fuse_attention_o_proj(
        attn, linear, _fake_q(_M_STAGED), _fake_q(), _fake_q(), None, _fake_q()
    )
    assert a is b


# ---------------------------------------------------------------------------
# model / attention seam wiring
# ---------------------------------------------------------------------------
def _model_source():
    return (_REPO_ROOT / "atom" / "models" / "minimax_m3" / "model.py").read_text(
        encoding="utf-8"
    )


def test_model_uses_the_shared_attend_and_project_helper():
    tree = ast.parse(_model_source())
    helper = None
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_minimax_m3_attend_and_project"
        ):
            helper = node
    assert helper is not None, "_minimax_m3_attend_and_project missing"
    # The helper branches to the static gate then the opaque custom op.
    branches = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("attends_via_routed_op", "a2a_gemm_attend_project")
    ]
    assert len(branches) == 2
    # Both attention classes hand their tail to the helper.
    callers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_minimax_m3_attend_and_project"
    ]
    assert len(callers) == 2


def test_attend_helper_calls_the_opaque_custom_op():
    source = _model_source()
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_minimax_m3_attend_and_project"
    )
    ops_calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "a2a_gemm_attend_project"
        # torch.ops.aiter.a2a_gemm_attend_project(...)
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "aiter"
    ]
    assert len(ops_calls) == 1


def _model_module_imports():
    tree = ast.parse(_model_source())
    return [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def _attend_helper_ast():
    for node in ast.parse(_model_source()).body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_minimax_m3_attend_and_project"
        ):
            return node
    raise AssertionError("_minimax_m3_attend_and_project missing")


def test_model_imports_a2a_gemm_at_module_scope():
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "atom.model_ops"
        and any(alias.name == "a2a_gemm" for alias in node.names)
        for node in _model_module_imports()
    )


def test_attend_helper_contains_no_import_statements():
    # torch.compile level 3 traces every attention; an import statement in the
    # traced helper is a graph break even with the fusion disabled.
    imports = [
        node
        for node in ast.walk(_attend_helper_ast())
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert imports == []


def _compiled_attend_helper():
    """Compile the real helper body against a stub `a2a_gemm` namespace."""
    source = _model_source()
    helper = _attend_helper_ast()
    namespace = {}
    exec(  # noqa: S102 - deliberate: compile the repo's real helper body
        textwrap.dedent(ast.get_source_segment(source, helper)), namespace
    )
    return namespace["_minimax_m3_attend_and_project"]


class _GateStub:
    def __init__(self, route):
        self.route = route
        self.calls = []

    def attends_via_routed_op(self, attn, o_proj, q):
        self.calls.append("gate")
        return self.route


class _OpsStub:
    def __init__(self):
        self.calls = []

    def a2a_gemm_attend_project(self, *args):
        self.calls.append(args)
        return "FUSED"


def _bind_helper_globals(helper, gate):
    """Give the compiled helper its stub globals (a2a_gemm + torch.ops)."""
    helper.__globals__["a2a_gemm"] = gate
    helper.__globals__["torch"] = SimpleNamespace(ops=SimpleNamespace(aiter=_OpsStub()))
    return helper.__globals__["torch"].ops.aiter


def test_attend_helper_native_path_calls_attn_then_o_proj_exactly_once():
    helper = _compiled_attend_helper()
    gate = _GateStub(route=False)
    _bind_helper_globals(helper, gate)
    calls = {"attn": 0, "proj": 0}

    def attn(q, k, v, positions=None, qkv=None):
        calls["attn"] += 1
        return ("attention-output",)

    def o_proj(x):
        calls["proj"] += 1
        return ("projected", x)

    out = helper(attn, o_proj, None, None, None, None, None)
    assert out == ("projected", "attention-output")
    assert calls == {"attn": 1, "proj": 1}
    assert gate.calls == ["gate"]


def test_attend_helper_routed_path_calls_the_opaque_op_only():
    helper = _compiled_attend_helper()
    gate = _GateStub(route=True)
    ops = _bind_helper_globals(helper, gate)
    calls = {"attn": 0, "proj": 0}

    def attn_call(**kwargs):
        calls["attn"] += 1
        return ("attention-output",)

    attn = SimpleNamespace(layer_name="model.layers.0.self_attn", __call__=attn_call)

    def o_proj(x):
        calls["proj"] += 1
        return x

    out = helper(attn, o_proj, "q", "k", "v", "positions", "qkv")
    assert out == "FUSED"
    assert calls == {"attn": 0, "proj": 0}
    assert gate.calls == ["gate"]
    assert len(ops.calls) == 1
    # (q, k, v, positions, layer_name, qkv)
    assert ops.calls[0][4] == "model.layers.0.self_attn"


def test_attend_helper_uses_the_layer_name_of_the_attention():
    helper = _compiled_attend_helper()
    gate = _GateStub(route=True)
    ops = _bind_helper_globals(helper, gate)
    attn = SimpleNamespace(layer_name="model.layers.7.self_attn")

    helper(attn, lambda x: x, "q", "k", "v", "pos", "qkv")
    assert ops.calls[0][4] == "model.layers.7.self_attn"


def test_sp_attention_ulysses_has_the_output_transform_hook():
    tree = ast.parse(
        (_REPO_ROOT / "atom" / "model_ops" / "sp_attention.py").read_text(
            encoding="utf-8"
        )
    )
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "ulysses_attention":
            names = [arg.arg for arg in node.args.args]
            assert "output_transform" in names
            assert "query" in names and "qkv" in names
            calls = [
                child
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "output_transform"
            ]
            assert len(calls) == 1
            return
    raise AssertionError("ulysses_attention missing")


def test_attention_fp8_transport_reuses_atoms_quantizer():
    source = (
        _REPO_ROOT / "atom" / "model_ops" / "minimax_m3" / "attention_fp8.py"
    ).read_text(encoding="utf-8")
    assert "head_amax" in source
    assert "quantize_with_gathered_amax" in source
    assert "all_gatherv" in source
    tree = ast.parse(source)
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert "triton" not in roots and "aiter" not in roots


# ---------------------------------------------------------------------------
# opaque custom op: the torch.compile / CUDA-graph boundary
# ---------------------------------------------------------------------------
def test_custom_op_is_registered_with_a_fake_impl():
    op = torch.ops.aiter.a2a_gemm_attend_project
    assert callable(op)
    q = torch.empty((4, a2a._K_IN), dtype=torch.bfloat16)
    fake = a2a._a2a_gemm_attend_project_fake(
        q, q, q, torch.empty(4, dtype=torch.int64), "layer", q
    )
    assert tuple(fake.shape) == (4, a2a._N_OUT)
    assert fake.dtype == torch.bfloat16


def test_custom_op_dispatches_to_the_installed_site(env):
    _site, attn, _linear = _pose_site(env)
    q = _fake_q(_M_STAGED)
    out = torch.ops.aiter.a2a_gemm_attend_project(
        q,
        q,
        q,
        torch.empty(a2a._ROWS_PER_M * _M_STAGED, dtype=torch.int64),
        attn.layer_name,
        q,
    )
    assert tuple(out.shape) == (_M_STAGED, a2a._N_OUT)
    assert len(env.op_calls) == 1


def test_custom_op_raises_without_an_installed_site():
    q = _fake_q()
    with pytest.raises(RuntimeError, match="no installed site"):
        torch.ops.aiter.a2a_gemm_attend_project(
            q, q, q, torch.empty(0, dtype=torch.int64), "missing.layer", q
        )


# ---------------------------------------------------------------------------
# CUDA-graph capture: eager pre-bind, capture-served, replay counters
# ---------------------------------------------------------------------------
@pytest.fixture
def real_forward_context():
    """Drive the real ``atom.utils.forward_context`` (no monkeypatched helper)."""
    from atom.utils import forward_context as fc

    def _set(in_hipgraph=False, dummy=False):
        import numpy as np

        fc.set_forward_context(
            attn_metadata=fc.AttentionMetaData(),
            atom_config=SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_size=1),
                compilation_config=SimpleNamespace(static_forward_context=None),
            ),
            context=fc.Context(positions=torch.zeros(1), is_dummy_run=bool(dummy)),
            running_tokens_per_dp_rank=np.zeros(1, dtype=np.int32),
            in_hipgraph=bool(in_hipgraph),
        )
        return fc.get_forward_context()

    try:
        yield _set
    finally:
        fc.reset_forward_context()


def test_in_hipgraph_reads_the_forward_context_field(real_forward_context, monkeypatch):
    from atom.utils.forward_context import Context, ForwardContext

    monkeypatch.setattr(a2a, "_in_hipgraph", _REAL_IN_HIPGRAPH)
    fctx = real_forward_context(in_hipgraph=True)
    assert isinstance(fctx, ForwardContext)
    # The field lives on ForwardContext; Context only carries is_dummy_run.
    assert "in_hipgraph" in ForwardContext.__dataclass_fields__
    assert "in_hipgraph" not in Context.__annotations__
    assert fctx.in_hipgraph is True
    assert a2a._in_hipgraph() is True

    real_forward_context(in_hipgraph=False)
    assert a2a._in_hipgraph() is False


def test_is_dummy_run_reads_the_context_field(real_forward_context, monkeypatch):
    monkeypatch.setattr(a2a, "_in_hipgraph", _REAL_IN_HIPGRAPH)
    monkeypatch.setattr(a2a, "_is_dummy_run", _REAL_IS_DUMMY_RUN)
    real_forward_context(in_hipgraph=True, dummy=True)
    assert a2a._is_dummy_run() is True
    assert a2a._in_hipgraph() is True


def test_capture_path_uses_the_prebound_cell_under_a_real_forward_context(
    env, real_forward_context, caplog, monkeypatch
):
    monkeypatch.setattr(a2a, "_in_hipgraph", _REAL_IN_HIPGRAPH)
    sites, site, _attn, _linear = _install_full(env)
    sites.begin_capture()
    assert sites.prepare_capture(_M_STAGED) == 1
    cell = site._capture_cells[_M_STAGED]
    assert len(env.op_calls) == 1  # the eager pre-bind
    real_forward_context(in_hipgraph=True)
    with caplog.at_level(logging.INFO, logger="atom"):
        out = site._output_transform(
            _pre_exchange_x(_M_STAGED), a2a._ROWS_PER_M * _M_STAGED
        )
    # The real flag (not a monkeypatched helper) routes through the capture cell:
    # same pre-bound pointers, no re-allocation, no extra provider binding work.
    assert out[0] is cell.out
    assert len(env.op_calls) == 2
    call = env.op_calls[-1]
    assert call["args"][0] is cell.q
    assert call["args"][1].data_ptr() == cell.scale.data_ptr()
    assert call["args"][4] is cell.out
    assert site.capture_served == 1
    assert any(
        f"a2a_gemm {site.name}: capture-served M={_M_STAGED}" in r.message
        for r in caplog.records
    )


def test_prepare_capture_is_deduped_per_m_class(env):
    # The capture ladder maps many batch sizes onto one local M, so a class is
    # pre-bound (vote + eager routed call) exactly once per epoch.
    sites, site, _attn, _linear = _install_full(env)
    sites.begin_capture()
    assert sites.prepare_capture(_M_STAGED) == 1
    calls = len(env.op_calls)
    assert calls == 1
    # Same local M again: no vote, no routed call, no re-bind.
    assert sites.prepare_capture(_M_STAGED) == 0
    assert len(env.op_calls) == calls
    assert sites.capture_prepared_classes == {_M_STAGED}
    assert site.capture_classes == frozenset({_M_STAGED})
    # A different class still pre-binds.
    assert sites.prepare_capture(_M) == 1
    assert len(env.op_calls) == calls + 1
    # The next capture epoch can prepare the class again.
    sites.begin_capture()
    assert sites.capture_prepared_classes == set()
    assert sites.prepare_capture(_M_STAGED) == 1


def test_begin_capture_bumps_epoch_and_drops_old_prebinds(env):
    sites, site, _attn, _linear = _install_full(env)
    first = sites.begin_capture()
    assert site._capture_epoch == first
    assert site.capture_prebind(_M_STAGED) is True
    assert site.capture_classes == frozenset({_M_STAGED})
    second = sites.begin_capture()
    assert second == first + 1
    assert sites.capture_epoch == second and site._capture_epoch == second
    assert site.capture_classes == frozenset()
    assert site.capture_prebound == 0


def test_prepare_capture_prebinds_and_capture_reuses_the_same_pointers(
    env, monkeypatch
):
    sites, site, _attn, _linear = _install_full(env)
    sites.begin_capture()
    assert sites.prepare_capture(_M_STAGED) == 1
    assert site.capture_classes == frozenset({_M_STAGED})
    assert len(env.op_calls) == 1  # one eager pre-bind call
    cell = site._capture_cells[_M_STAGED]
    assert cell.q.dtype == torch.float8_e4m3fn
    assert tuple(cell.scale.shape) == (_M_STAGED, 1)
    monkeypatch.setattr(a2a, "_in_hipgraph", lambda: True)
    out = site._output_transform(
        _pre_exchange_x(_M_STAGED), a2a._ROWS_PER_M * _M_STAGED
    )
    assert out[0] is cell.out
    call = env.op_calls[-1]
    assert call["args"][0] is cell.q
    assert call["args"][1].data_ptr() == cell.scale.data_ptr()
    assert call["args"][4] is cell.out
    assert call["kwargs"]["registered"] is False
    assert site.capture_served == 1


def test_registered_capture_class_uses_the_producer_view(env, monkeypatch):
    sites, site, _attn, _linear = _install_full(env)
    sites.begin_capture()
    assert sites.prepare_capture(_M) == 1
    cell = site._capture_cells[_M]
    assert cell.registered is True
    assert cell.q is site._producer_view(_M)
    assert a2a.owns_producer_pointer(cell.q)
    monkeypatch.setattr(a2a, "_in_hipgraph", lambda: True)
    site._output_transform(_pre_exchange_x(_M), a2a._ROWS_PER_M * _M)
    assert env.op_calls[-1]["kwargs"]["registered"] is True


def test_capture_class_agreement_requires_every_rank(env, monkeypatch):
    sites, site, _attn, _linear = _install_full(env)
    sites.group = object()
    _fake_gather(
        monkeypatch,
        [
            {site.name: True},
            {site.name: False},
            {site.name: True},
            {site.name: True},
        ],
    )
    sites.begin_capture()
    assert sites.prepare_capture(_M_STAGED) == 0
    assert site.capture_classes == frozenset()
    assert _M_STAGED in site._capture_native
    assert env.op_calls == []


def test_unprebound_capture_class_is_native_inside_capture(env, monkeypatch):
    sites, site, _attn, linear = _install_full(env)
    sites.begin_capture()
    monkeypatch.setattr(a2a, "_in_hipgraph", lambda: True)
    out = site._output_transform(
        _pre_exchange_x(_M_UNCOVERED), a2a._ROWS_PER_M * _M_UNCOVERED
    )
    assert out[0] is not None and env.op_calls == []
    assert env.sp.gather_calls and linear.calls == 1


def test_capture_prebind_refused_inside_actual_graph_capture(env, monkeypatch):
    sites, site, _attn, _linear = _install_full(env)
    sites.begin_capture()
    monkeypatch.setattr(a2a, "_is_cuda_capturing", lambda: True)
    assert sites.prepare_capture(_M_STAGED) == 0
    assert site.capture_prebind(_M_STAGED) is False
    assert site.capture_classes == frozenset()
    assert env.op_calls == []


def test_capture_prebind_failure_marks_the_class_native(env):
    def boom(*args, **kwargs):
        raise RuntimeError("bind failed")

    env.aiter_module.a2a_gemm_fused = boom
    sites, site, _attn, _linear = _install_full(env)
    sites.begin_capture()
    assert sites.prepare_capture(_M_STAGED) == 0
    assert site.capture_classes == frozenset()
    assert _M_STAGED in site._capture_native


def test_capture_prebind_requires_the_provider_not_aiter_native(env, monkeypatch):
    class _NativeDecisionPlugin(_FakePlugin):
        def last_decision(self):
            return SimpleNamespace(route="native")

    sites, site, _attn, _linear = _install_full(env)
    plugin = _NativeDecisionPlugin()
    env.plugin = plugin
    env.aiter_module.plugin = plugin
    monkeypatch.setattr(a2a, "_PLUGIN", plugin)
    sites.begin_capture()
    assert sites.prepare_capture(_M_STAGED) == 0
    assert site.capture_classes == frozenset()
    assert _M_STAGED in site._capture_native


def test_capture_provider_decision_accepts_a_provider_route(env, monkeypatch):
    class _ProviderDecisionPlugin(_FakePlugin):
        def last_decision(self):
            return SimpleNamespace(route="provider")

    sites, site, _attn, _linear = _install_full(env)
    plugin = _ProviderDecisionPlugin()
    env.plugin = plugin
    env.aiter_module.plugin = plugin
    monkeypatch.setattr(a2a, "_PLUGIN", plugin)
    sites.begin_capture()
    assert sites.prepare_capture(_M_STAGED) == 1
    assert site.capture_classes == frozenset({_M_STAGED})


def test_replay_counter_and_log(env, caplog):
    sites, _site, _attn, _linear = _install_full(env)
    sites.begin_capture()
    with caplog.at_level(logging.INFO, logger="atom"):
        sites.note_graph_replay(16)
        sites.note_graph_replay(16)
        sites.note_graph_replay(32)
    replay_lines = [
        record.message
        for record in caplog.records
        if "a2a_gemm graph replay" in record.message
    ]
    assert len(replay_lines) == 2  # once per distinct bs
    assert "bs=16" in replay_lines[0] and "bs=32" in replay_lines[1]


def test_capture_log_lines_for_phase_f(env, monkeypatch, caplog):
    sites, site, _attn, _linear = _install_full(env)
    sites.begin_capture()
    with caplog.at_level(logging.INFO, logger="atom"):
        sites.prepare_capture(_M_STAGED)
        monkeypatch.setattr(a2a, "_in_hipgraph", lambda: True)
        site._output_transform(_pre_exchange_x(_M_STAGED), a2a._ROWS_PER_M * _M_STAGED)
        sites.note_graph_replay(8)
    messages = [record.message for record in caplog.records]
    assert any(
        f"a2a_gemm capture prebind layer={site.name} M={_M_STAGED}" in m
        for m in messages
    )
    assert any(
        f"provider capture {site.name} mode=o_proj m={_M_STAGED}: fused (prepare)" in m
        for m in messages
    )
    assert any(
        f"a2a_gemm {site.name}: capture-served M={_M_STAGED}" in m for m in messages
    )
    assert any("a2a_gemm graph replay bs=8" in m for m in messages)


# ---------------------------------------------------------------------------
# ModelRunner hooks
# ---------------------------------------------------------------------------
def _model_runner_init():
    tree = ast.parse(
        (_REPO_ROOT / "atom" / "model_engine" / "model_runner.py").read_text(
            encoding="utf-8"
        )
    )
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return item
    raise AssertionError("ModelRunner.__init__ not found")


def _call_order(function_def):
    found = []

    def name_of(node):
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
        return None

    for node in ast.walk(function_def):
        if isinstance(node, ast.Call) and node.lineno >= function_def.lineno:
            name = name_of(node.func)
            if name is not None:
                found.append((node.lineno, name))
    return [name for _, name in sorted(found)]


def test_runner_hook_order_is_setup_configure_build_install_warmup():
    calls = _call_order(_model_runner_init())

    def index(name):
        assert name in calls, f"{name} missing from ModelRunner.__init__"
        return calls.index(name)

    assert index("_setup_device_and_distributed") < index("configure_aiter_providers")
    assert index("configure_aiter_providers") < index("_build_and_load_model")
    assert index("_build_and_load_model") < index("initialize_a2a_gemm")
    assert index("initialize_a2a_gemm") < index("_maybe_warmup")


def _runner_function(name):
    tree = ast.parse(
        (_REPO_ROOT / "atom" / "model_engine" / "model_runner.py").read_text(
            encoding="utf-8"
        )
    )
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"ModelRunner.{name} not found")


def test_runner_capture_hooks_are_eager_and_replay_is_logged():
    capture = _runner_function("capture_cudagraph")
    # begin_capture runs before the graph-capture context (eager prep) and the
    # per-M pre-bind runs after each bucket's forward context is set.
    begin_lines = [
        node.lineno
        for node in ast.walk(capture)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "begin_capture"
    ]
    prepare_lines = [
        node.lineno
        for node in ast.walk(capture)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "prepare_capture"
    ]
    graph_capture_lines = [
        node.lineno
        for node in ast.walk(capture)
        if isinstance(node, ast.Name) and node.id == "graph_capture"
    ]
    assert begin_lines and prepare_lines and graph_capture_lines
    assert begin_lines[0] < min(graph_capture_lines)
    assert prepare_lines[0] > begin_lines[0]
    # The replay path logs that the captured graph re-entered the fused tail.
    run_model = _runner_function("run_model")
    replay_logs = [
        node
        for node in ast.walk(run_model)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "note_graph_replay"
    ]
    assert len(replay_logs) == 1
    # A captured prefill bucket pre-binds too.
    prefill = _runner_function("_run_capture_prefill")
    prefill_prepare = [
        node
        for node in ast.walk(prefill)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "prepare_capture"
    ]
    assert len(prefill_prepare) == 1


# ---------------------------------------------------------------------------
# config / CLI rail
# ---------------------------------------------------------------------------
def test_engine_args_map_the_provider_flags():
    from atom.model_engine.arg_utils import EngineArgs

    args = EngineArgs(aiter_provider_enable="p1, p2", enable_a2a_gemm=True)
    kwargs = args._get_engine_kwargs()
    assert kwargs["aiter_provider_enable"] == ("p1", "p2")
    assert kwargs["enable_a2a_gemm"] is True

    defaults = EngineArgs()._get_engine_kwargs()
    assert defaults["aiter_provider_enable"] == ()
    assert defaults["enable_a2a_gemm"] is False


def _config_annotation(name):
    tree = ast.parse((_REPO_ROOT / "atom" / "config.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            for item in node.body:
                if (
                    isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                    and item.target.id == name
                ):
                    return ast.literal_eval(item.value)
    raise AssertionError(f"Config.{name} not found")


def test_config_flags_default_off_and_are_normalized():
    assert _config_annotation("aiter_provider_enable") == ()
    assert _config_annotation("enable_a2a_gemm") is False
    source = (_REPO_ROOT / "atom" / "config.py").read_text(encoding="utf-8")
    assert "normalize_provider_names(raw_providers)" in source
    assert '"a2a_gemm_fused_v1"' in source
