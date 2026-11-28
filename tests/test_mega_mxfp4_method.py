# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing

try:
    # aiter/triton absent under bare non-GPU pytest
    import atom.model_ops.moe as moe_mod
except ImportError as exc:
    skip_if_dependency_missing(exc, "requires full atom import env")


def test_mega_method_is_mxfp4_specialization():
    assert issubclass(moe_mod.MegaMxfp4MoEMethod, moe_mod.Mxfp4MoEMethod)


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("backend_name", ["standard", "mega"])
def test_mxfp4_method_selection(monkeypatch, backend_name, sp):
    standard_method = object()
    mega_method = object()
    config = SimpleNamespace(moe_backend=backend_name)
    monkeypatch.setattr(moe_mod, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(moe_mod, "sp_is_enabled", lambda: sp)
    monkeypatch.setattr(moe_mod, "Mxfp4MoEMethod", lambda *_args: standard_method)
    monkeypatch.setattr(moe_mod, "MegaMxfp4MoEMethod", lambda *_args: mega_method)

    selected = moe_mod._make_mxfp4_moe_method(object(), object())

    # Under SP, `mega` is SP MegaMoE over the standard method's weights.
    use_mega = backend_name == "mega" and not sp
    assert selected is (mega_method if use_mega else standard_method)
    if sp:
        # Nor does it hold the layer to the EP backend's requirements.
        moe_mod.FusedMoE._validate_moe_backend(SimpleNamespace(quant_method=selected))


def _mxfp4_method(monkeypatch, gfx, fp8_activations):
    """An Mxfp4MoEMethod on `gfx` with ATOM_MOE_GU_ITLV off."""
    monkeypatch.setattr(
        moe_mod, "get_current_atom_config", lambda: SimpleNamespace(eplb_enable=False)
    )
    monkeypatch.setattr(moe_mod, "get_gfx", lambda: gfx)
    monkeypatch.setattr(moe_mod.envs, "is_set", lambda _name: True)
    monkeypatch.setattr(moe_mod.envs, "ATOM_USE_TRITON_MOE", False)
    monkeypatch.setattr(moe_mod.envs, "ATOM_MOE_GU_ITLV", False)
    quant_config = SimpleNamespace(
        quant_type=object(), quant_dtype=object(), quant_method=None, is_dynamic=True
    )
    moe_config = SimpleNamespace(
        a_quant_dtype=None,
        mxfp4_fp8_activations=fp8_activations,
        use_ep=False,
        in_dtype=torch.bfloat16,
        hidden_dim=256,
    )
    return moe_mod.Mxfp4MoEMethod(quant_config, moe_config)


@pytest.mark.parametrize("fp8_activations", [False, True])
def test_fp8_activations_interleave_without_the_env_flag(monkeypatch, fp8_activations):
    # aiter's A8W4 kernels read interleaved gate/up rows only, so a layer that
    # asks for MXFP8 activations must get that layout whatever the env says.
    method = _mxfp4_method(monkeypatch, "gfx950", fp8_activations)

    assert method.fp8_activations is fp8_activations
    assert method.is_guinterleave is fp8_activations
    # Routed rows that move between ranks travel in the MXFP8 the experts take.
    wire = method.routed_rows_wire
    assert (wire is not None and wire.dtype == moe_mod.dtypes.fp8) is fp8_activations


@pytest.mark.parametrize("fp8_activations", [False, True])
def test_gfx1250_interleaves_without_the_env_flag(monkeypatch, fp8_activations):
    # aiter's gfx1250 fused_moe rejects the separated gate/up layout, so every
    # MXFP4 layer there gets interleaved rows, MXFP8 activations or not.
    method = _mxfp4_method(monkeypatch, "gfx1250", fp8_activations)

    assert method.is_gfx1250 is True
    assert method.is_guinterleave is True


@pytest.mark.parametrize(
    ("eplb_enabled", "expected_triton"),
    [(False, True), (True, False)],
)
def test_eplb_controls_effective_triton_backend(
    monkeypatch, eplb_enabled, expected_triton
):
    config = SimpleNamespace(eplb_enable=eplb_enabled)
    monkeypatch.setattr(moe_mod, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(moe_mod, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(moe_mod.envs, "is_set", lambda _name: True)
    monkeypatch.setattr(moe_mod.envs, "ATOM_USE_TRITON_MOE", True)
    monkeypatch.setattr(moe_mod.envs, "ATOM_USE_TRITON_MOE_DECODE", True)
    monkeypatch.setattr(moe_mod.envs, "ATOM_MOE_GU_ITLV", False)

    quant_config = SimpleNamespace(
        quant_type=object(),
        quant_dtype=object(),
        quant_method=None,
        is_dynamic=True,
    )
    # TP, not EP: `use_triton` is `use_triton_moe and not use_ep`, so the
    # expectation below is about the TP half of that pair.
    moe_config = SimpleNamespace(
        a_quant_dtype=None, mxfp4_fp8_activations=False, use_ep=False
    )

    method = moe_mod.Mxfp4MoEMethod(quant_config, moe_config)

    assert method.use_triton is expected_triton
    assert method.use_triton_decode is expected_triton


@pytest.mark.parametrize("fp8_activations", [False, True])
def test_standard_post_routing_arguments_are_preserved(monkeypatch, fp8_activations):
    calls = []

    def fake_fused_moe(*args, **kwargs):
        calls.append((args, kwargs))
        return "output"

    monkeypatch.setattr(moe_mod, "fused_moe", fake_fused_moe)
    method = object.__new__(moe_mod.Mxfp4MoEMethod)
    method.fused_experts = None
    method.use_flydsl_herd = False
    method.quant_type = "mxfp4"
    method.is_guinterleave = True
    method.fp8_activations = fp8_activations
    method.hidden_pad = 0
    method.intermediate_pad = 0
    # Skip the Triton block so apply() reaches the FlyDSL tail.
    # use_triton_decode=False also short-circuits get_forward_context().
    # `use_triton_ep` is the EP half of the same pair and is read further down;
    # False keeps this on the TP path the FlyDSL tail belongs to.
    method.use_triton = False
    method.use_triton_decode = False
    method.use_triton_ep = False
    layer = SimpleNamespace(
        w13_weight="w1",
        w2_weight="w2",
        w13_weight_scale="w1_scale",
        w2_weight_scale="w2_scale",
        w13_input_scale="a1_scale",
        w2_input_scale="a2_scale",
        w13_bias="b1",
        w2_bias="b2",
        expert_mask="mask",
        swiglu_limit=7.0,
    )

    result = method.apply(
        layer=layer,
        x="x",
        topk_weights="topk_weights",
        topk_ids="topk_ids",
        global_num_experts=16,
        expert_map="expert_map",
        activation="silu",
        apply_router_weight_on_input=False,
    )

    assert result == "output"
    args, kwargs = calls.pop()
    assert args == ("x", "w1", "w2", "topk_weights", "topk_ids")
    # An A8W4 layer names its activation dtype rather than leaving it to
    # aiter's per-arch default.
    assert kwargs.pop("quant_dtype_a", None) == (
        moe_mod.dtypes.fp8 if fp8_activations else None
    )
    assert kwargs == {
        "expert_mask": "mask",
        "activation": "silu",
        "quant_type": "mxfp4",
        "w1_scale": "w1_scale",
        "w2_scale": "w2_scale",
        "a1_scale": "a1_scale",
        "a2_scale": "a2_scale",
        "doweight_stage1": False,
        "bias1": "b1",
        "bias2": "b2",
        # No count: these rows were not marked as this rank's own.
        "num_local_tokens": None,
        "gate_mode": moe_mod.GateMode.INTERLEAVE.value,
        "swiglu_limit": 7.0,
    }


def test_mega_eplb_views_are_expert_major_aliases():
    layer = SimpleNamespace(
        _mega_w1=torch.arange(24),
        _mega_w1_scale=torch.arange(12),
        _mega_w2=torch.arange(36),
        _mega_w2_scale=torch.arange(8),
        local_num_experts=4,
    )
    method = object.__new__(moe_mod.MegaMxfp4MoEMethod)

    views = method.get_eplb_weight_views(layer)

    assert [tuple(view.shape) for view in views] == [(4, 6), (4, 3), (4, 9), (4, 2)]
    views[0][2, 1] = -1
    assert layer._mega_w1.view(4, -1)[2, 1].item() == -1


def test_mega_eplb_views_reject_non_expert_major_storage():
    layer = SimpleNamespace(
        _mega_w1=torch.arange(10),
        _mega_w1_scale=torch.arange(12),
        _mega_w2=torch.arange(36),
        _mega_w2_scale=torch.arange(8),
        local_num_experts=4,
    )
    method = object.__new__(moe_mod.MegaMxfp4MoEMethod)

    with pytest.raises(RuntimeError, match="evenly divisible"):
        method.get_eplb_weight_views(layer)


def test_mega_eplb_views_include_shared_expert():
    layer = SimpleNamespace(
        _mega_w1=torch.arange(30),
        _mega_w1_scale=torch.arange(15),
        _mega_w2=torch.arange(45),
        _mega_w2_scale=torch.arange(10),
        local_num_experts=5,
    )
    method = object.__new__(moe_mod.MegaMxfp4MoEMethod)

    views = method.get_eplb_weight_views(layer)

    assert [tuple(view.shape) for view in views] == [(5, 6), (5, 3), (5, 9), (5, 2)]
    assert torch.equal(views[0], layer._mega_w1.view(5, -1))


def test_mega_rejects_the_legacy_shared_expert_fusion(monkeypatch):
    monkeypatch.setattr(
        moe_mod, "get_current_atom_config", lambda: SimpleNamespace(moe_backend="mega")
    )
    layer = SimpleNamespace(
        quant_method=object.__new__(moe_mod.MegaMxfp4MoEMethod),
        expert_layout=SimpleNamespace(mode=moe_mod.SharedExpertMode.LEGACY_AITER),
    )

    with pytest.raises(NotImplementedError, match="dispatch space"):
        moe_mod.FusedMoE._validate_moe_backend(layer)

    layer.expert_layout.mode = moe_mod.SharedExpertMode.LOCAL_REPLICA
    moe_mod.FusedMoE._validate_moe_backend(layer)
    layer.expert_layout.mode = moe_mod.SharedExpertMode.EPLB_ROUTED
    moe_mod.FusedMoE._validate_moe_backend(layer)


def test_mega_backend_uses_global_dispatch_width(monkeypatch):
    pytest.importorskip(
        "triton", reason="MegaMoE imports its Triton row-zeroing kernel"
    )
    from atom.model_ops.fused_moe import flydsl_mega_experts as mega_module

    captured = {}

    def fake_run(_layer, _hidden, _weights, _ids, **kwargs):
        captured.update(kwargs)
        return "output"

    monkeypatch.setattr(mega_module, "run_mega_moe", fake_run)
    layer = SimpleNamespace(
        _mega_w1=torch.empty(33, 1),
        local_num_experts=33,
        swiglu_limit=0.0,
    )
    backend = mega_module.MegaFusedExperts(
        layer,
        model_dim=16,
        inter_dim=8,
        mtpr=256,
    )

    output = backend(
        hidden_states="hidden",
        topk_weights="weights",
        topk_ids=SimpleNamespace(shape=(1, 9)),
        global_num_experts=264,
    )

    assert output == "output"
    assert captured["experts"] == 264
    assert captured["topk"] == 9


def _m31_experts(monkeypatch):
    """The experts layer MiniMax-M3.1's MoE block builds, its FusedMoE recorded."""
    m3 = pytest.importorskip("atom.models.minimax_m3.model")

    class Experts(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.activation = kwargs["activation"]
            self.quant_method = None

    monkeypatch.setattr(m3, "FusedMoE", Experts)
    monkeypatch.setattr(m3, "ReplicatedLinear", lambda *_a, **_k: torch.nn.Identity())
    monkeypatch.setattr(m3, "get_tensor_model_parallel_world_size", lambda: 1)
    config = SimpleNamespace(
        num_local_experts=128,
        num_experts_per_tok=4,
        hidden_size=16,
        intermediate_size=8,
        swiglu_limit=7.0,
    )
    return m3.MiniMaxM3MoE(config, layer_id=3).experts


class _PrepareFinalize:
    """A MoRI-shaped prepare/finalize that hands the rows straight through."""

    def topk_indices_dtype(self):
        return None

    def supports_async(self):
        return False

    def needs_dispatch_output_trim(self):
        return True

    def prepare(self, hidden, weights, ids, *_args, x_scale=None, **_kwargs):
        return hidden, None, SimpleNamespace(expert_num_tokens="counts"), None, None

    def finalize(self, output, fused_out, *_args, **_kwargs):
        return fused_out


def _mega_layer(**attrs):
    return SimpleNamespace(
        # Emptied: Mega's weights replace it.
        w13_weight=torch.empty(0, dtype=torch.uint8),
        w2_weight=torch.empty(0, dtype=torch.uint8),
        _mega_w1=torch.empty(4, 2, dtype=torch.uint8),
        _mega_w1_scale=torch.empty(4, 3),
        _mega_w2=torch.empty(4, 5, dtype=torch.uint8),
        _mega_w2_scale=torch.empty(4, 6),
        w13_bias=None,
        w2_bias=None,
        expert_mask="mask",
        local_num_experts=4,
        **attrs,
    )


def _mega_method(
    monkeypatch, layer, a_quant_dtype=None, prepare_finalize=_PrepareFinalize
):
    """A MegaMxfp4MoEMethod after the real init_prepare_finalize on `layer`,
    over a pass-through MoRI transport, recording the inherited apply (Mega's
    step), fused_moe (the fallback's) and aiter's resolver."""
    pytest.importorskip(
        "triton", reason="MegaMoE imports its Triton row-zeroing kernel"
    )
    from importlib import import_module

    from atom.model_ops.fused_moe import flydsl_mega_experts as mega_module
    from atom.model_ops.fused_moe import modular_kernel

    state = SimpleNamespace(
        across_dp=None, fused_moe=[], mega=[], resolved=[], prepared=0, contexts=0
    )

    def resolve(quant_type, w_dtype, *, activation, gate_mode, M):
        state.resolved.append((quant_type, w_dtype, activation, gate_mode, M))
        return moe_mod.dtypes.fp4x2 if M >= 256 else torch.bfloat16

    monkeypatch.setattr(
        import_module("aiter.fused_moe"),
        "resolve_activation_dtype",
        resolve,
        raising=False,
    )

    def forward_context():
        state.contexts += 1
        return SimpleNamespace(
            context=SimpleNamespace(
                running_tokens_across_dp=state.across_dp, is_prefill=False
            )
        )

    monkeypatch.setattr(moe_mod, "get_forward_context", forward_context)
    monkeypatch.setattr(modular_kernel, "get_forward_context", forward_context)
    monkeypatch.setattr(
        modular_kernel,
        "fused_moe",
        lambda *args, **kwargs: state.fused_moe.append((args, kwargs)) or args[0],
    )

    def inherited_apply(method, layer, x, *_args, **kwargs):
        assert isinstance(method.fused_experts, mega_module.MegaFusedExperts)
        state.mega.append(kwargs)
        return x

    monkeypatch.setattr(moe_mod.Mxfp4MoEMethod, "apply", inherited_apply)

    method = object.__new__(moe_mod.MegaMxfp4MoEMethod)
    method.moe = SimpleNamespace(max_num_tokens=512, a_quant_dtype=a_quant_dtype)
    method.fused_experts = None
    method.topk_indices_dtype = None
    method.quant_type = moe_mod.QuantType.per_1x32
    method.hidden_size = 16
    method.intermediate_size = 8
    method.hidden_pad = 0
    method.intermediate_pad = 0
    # The MXFP4 dispatch wire ATOM_MORI_FP4_DISPATCH would set: Mega takes bf16.
    method.routed_rows_wire = "fp4 wire"

    def make_prepare_finalize():
        state.prepared += 1
        state.wire = method.routed_rows_wire
        return prepare_finalize and prepare_finalize()

    method.maybe_make_prepare_finalize = make_prepare_finalize
    method.get_fused_moe_quant_config = lambda _layer: "quant config"
    method.init_prepare_finalize(layer)
    state.method, state.layer = method, layer
    state.mega_module, state.modular_kernel = mega_module, modular_kernel
    return state


@pytest.fixture
def m31_method(monkeypatch):
    experts = _m31_experts(monkeypatch)
    layer = _mega_layer(
        activation=experts.activation,
        swiglu_limit=experts.swiglu_limit,
        mega_combine_wire=experts.mega_combine_wire,
        mega_quantized_steps_only=experts.mega_quantized_steps_only,
    )
    # The Quark checkpoint declares MXFP4 inputs for the routed experts.
    return _mega_method(monkeypatch, layer, a_quant_dtype="fp4")


def _apply(state, rows):
    x = torch.zeros(rows, 16, dtype=torch.bfloat16)
    return state.method.apply(
        state.layer,
        x,
        torch.ones(rows, 4),
        torch.zeros(rows, 4, dtype=torch.int32),
        global_num_experts=16,
        activation=state.layer.activation,
    )


def test_m31_mega_combines_over_a_bf16_wire_with_swiglu_oai(monkeypatch):
    pytest.importorskip(
        "triton", reason="MegaMoE imports its Triton row-zeroing kernel"
    )
    from atom.model_ops.fused_moe import flydsl_mega_experts as mega_module

    experts = _m31_experts(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        mega_module,
        "run_mega_moe",
        lambda _layer, hidden, *_args, **kwargs: captured.update(kwargs) or hidden,
    )
    experts._mega_w1 = torch.empty(4, 1)
    experts.local_num_experts = 4
    backend = mega_module.MegaFusedExperts(experts, model_dim=16, inter_dim=8, mtpr=256)
    backend(
        hidden_states="hidden",
        topk_weights="weights",
        topk_ids=SimpleNamespace(shape=(1, 4)),
        global_num_experts=128,
        activation=experts.activation,
    )

    assert captured["combine_wire"] == "bf16"
    assert captured["act"] == "swigluoai"
    assert captured["swiglu_limit"] == 7.0


def test_m31_mega_builds_the_mori_kernel_next_to_its_own(m31_method):
    method = m31_method.method
    assert method.fused_experts._quant == "a4w4"
    assert isinstance(method.fused_experts, m31_method.mega_module.MegaFusedExperts)
    assert isinstance(
        method.mori_experts, m31_method.modular_kernel.FusedMoEModularKernel
    )
    # MoRI carries bf16 rows, as Mega's own dispatch takes them.
    assert m31_method.wire is None
    assert method.routed_rows_wire is None


@pytest.mark.parametrize(
    ("across_dp", "mega"),
    [((64, 64, 64, 63), False), ((64, 64, 64, 64), True), ((256, 0, 0, 0), True)],
)
def test_m31_mega_takes_a_step_iff_the_standard_method_quantizes(
    m31_method, across_dp, mega
):
    """The choice asks aiter about the group's rows in the standard method's
    separated layout, so every rank of the group makes it alike."""
    m31_method.across_dp = across_dp
    for local_rows in across_dp:
        _apply(m31_method, max(local_rows, 1))

    assert len(m31_method.mega) == (4 if mega else 0)
    assert len(m31_method.fused_moe) == (0 if mega else 4)
    assert set(m31_method.resolved) == {
        (
            moe_mod.QuantType.per_1x32,
            torch.uint8,
            moe_mod.ActivationType.Swiglu,
            moe_mod.GateMode.SEPARATED,
            sum(across_dp),
        )
    }


def test_m31_fallback_runs_fused_moe_on_megas_interleaved_weights(m31_method):
    layer = m31_method.layer
    m31_method.across_dp = (2, 2, 2, 2)
    _apply(m31_method, 2)

    ((args, kwargs),) = m31_method.fused_moe
    assert args[1] is layer._mega_w1
    assert args[2] is layer._mega_w2
    assert kwargs["w1_scale"] is layer._mega_w1_scale
    assert kwargs["w2_scale"] is layer._mega_w2_scale
    assert kwargs["gate_mode"] == moe_mod.GateMode.INTERLEAVE.value
    assert kwargs["swiglu_limit"] == 7.0
    assert args[6] == moe_mod.ActivationType.Swiglu
    assert not m31_method.mega


def test_without_the_group_table_mega_takes_the_step(m31_method):
    _apply(m31_method, 8)

    assert len(m31_method.mega) == 1
    assert not m31_method.fused_moe and not m31_method.resolved


def test_without_the_switch_mega_takes_every_step_as_before(monkeypatch):
    state = _mega_layer_without_switch(monkeypatch)
    state.across_dp = (1, 1, 1, 1)
    _apply(state, 1)

    assert state.method.mori_experts is None and state.prepared == 0
    assert len(state.mega) == 1
    # No forward context read and no resolver call: the inherited apply alone.
    assert state.contexts == 0 and not state.resolved and not state.fused_moe


def _mega_layer_without_switch(monkeypatch, a_quant_dtype=None):
    return _mega_method(
        monkeypatch,
        _mega_layer(activation=moe_mod.ActivationType.Silu),
        a_quant_dtype=a_quant_dtype,
    )


@pytest.mark.parametrize(
    ("a_quant_dtype", "quant"),
    # None / "": no declared activation quantization keeps a8w4 (DeepSeek-V4).
    [
        ("fp4", "a4w4"),
        ("uint8", "a4w4"),
        ("fp8_e4m3", "a8w4"),
        (None, "a8w4"),
        ("", "a8w4"),
    ],
)
def test_mega_quant_follows_the_declared_activation_quant(
    monkeypatch, a_quant_dtype, quant
):
    state = _mega_layer_without_switch(monkeypatch, a_quant_dtype)

    assert state.method.fused_experts._quant == quant
    assert moe_mod.MegaMxfp4MoEMethod.mega_quant(state.method.moe) == quant


def test_the_fallback_runs_bf16_at_every_row_count_it_serves(monkeypatch):
    """aiter's own resolver, on Mega's own weights: every group row count the
    choice leaves to the fallback is one fused_moe runs in bf16, as the
    standard method does there (gfx950, the arch Mega runs on)."""
    fused_moe = pytest.importorskip("aiter.fused_moe")
    if not hasattr(fused_moe, "resolve_activation_dtype"):
        pytest.skip("AITER without resolve_activation_dtype")
    from aiter import ActivationType, QuantType, dtypes
    from aiter.ops.flydsl.moe_common import GateMode

    from atom.model_ops.fused_moe.flydsl_mega_experts import build_mega_weights

    monkeypatch.setattr(fused_moe, "get_gfx_runtime", lambda: "gfx950")
    experts, inter, hidden = 2, 128, 256
    layer = SimpleNamespace(
        w13_weight=torch.zeros(experts, 2 * inter, hidden // 2, dtype=torch.uint8).view(
            dtypes.fp4x2
        ),
        w13_weight_scale=torch.zeros(
            experts, 2 * inter, hidden // 32, dtype=torch.uint8
        ),
        w2_weight=torch.zeros(experts, hidden, inter // 2, dtype=torch.uint8).view(
            dtypes.fp4x2
        ),
        w2_weight_scale=torch.zeros(experts, hidden, inter // 32, dtype=torch.uint8),
        local_num_experts=experts,
        activation=ActivationType.Swiglu,
    )
    build_mega_weights(layer)
    method = object.__new__(moe_mod.MegaMxfp4MoEMethod)
    method.quant_type = QuantType.per_1x32

    served = [m for m in range(1, 4097) if not method.mega_runs(layer, m)]
    assert served == list(range(1, len(served) + 1)) and served
    w1 = layer._mega_w1
    for m in served:
        # fused_moe's own call, as the fallback makes it.
        assert (
            fused_moe.resolve_activation_dtype(
                QuantType.per_1x32,
                w1.dtype,
                activation=ActivationType.Swiglu,
                gate_mode=GateMode.INTERLEAVE,
                M=m,
                hidden_dtype=torch.bfloat16,
                has_a1_scale=False,
                gate_up_interleave=getattr(w1, "gate_up_interleave", None),
            )
            == dtypes.bf16
        ), m


def _mori_v2_prepare_finalize():
    """A MoRI v2 transport (ATOM_MORI_V2), which runs its own MegaMoE."""
    from atom.model_ops.fused_moe import mori_v2_prepare_finalize as v2

    class V2(v2.MoriV2PrepareAndFinalize):
        def __init__(self):
            self.mega = None

        def topk_indices_dtype(self):
            return None

        def bind_mega_transport(self, _layer, _method):
            self.mega = object()

    return V2()


@pytest.mark.parametrize(
    "prepare_finalize",
    [None, _mori_v2_prepare_finalize],
    ids=["no-all2all-backend", "mori-v2"],
)
def test_quantized_steps_only_without_mori_v1_is_refused_at_build(
    monkeypatch, prepare_finalize
):
    """Without MoRI v1's kernel no step could fall back, so Mega would take
    them all: the build refuses rather than drop the switch silently."""
    experts = _m31_experts(monkeypatch)
    layer = _mega_layer(
        activation=experts.activation,
        swiglu_limit=experts.swiglu_limit,
        mega_quantized_steps_only=True,
    )
    with pytest.raises(NotImplementedError, match="quantized-steps-only"):
        _mega_method(monkeypatch, layer, "fp4", prepare_finalize=prepare_finalize)
