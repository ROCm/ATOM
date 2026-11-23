# SPDX-License-Identifier: MIT
"""CPU checks for the scoped MI308X MoE layout and MTP fusion policy."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def impl():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            qwen=load("atom.models.qwen4_exp"),
            mtp=load("atom.models.qwen4_exp_mtp"),
            topk=load("atom.model_ops.topK"),
            config=load("atom.config"),
        )


@pytest.fixture
def setup(monkeypatch, impl):
    hf = SimpleNamespace(
        hidden_size=2560,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        hidden_act="silu",
        num_hidden_layers=48,
        hc_count=4,
        hc_lowrank=320,
        vocab_size=32,
        rms_norm_eps=1e-6,
    )
    state = SimpleNamespace(tp=4, gfx="gfx942", cu=80)
    config = SimpleNamespace(
        hf_config=hf,
        quant_config=impl.config.QuantizationConfig(),
        model="unused-checkpoint",
        enable_expert_parallel=False,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        moe_ep_flatten_tp_across_dp=False,
        moe_all2all_backend="none",
    )
    monkeypatch.setattr(impl.qwen, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(impl.topk, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(impl.qwen, "get_gfx_runtime", lambda: state.gfx)
    monkeypatch.setattr(impl.qwen, "get_cu_num", lambda: state.cu)
    for module in (impl.qwen, impl.mtp):
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: state.tp
        )

    class Gate(torch.nn.Module):
        def __init__(self, hidden, outputs, **kwargs):
            super().__init__()
            self.outputs = sum(outputs)

        def forward(self, hidden):
            return hidden.new_zeros((hidden.shape[0], self.outputs))

    class Experts(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.args = kwargs
            self.use_ep = config.enable_expert_parallel

        def forward(self, hidden_states, router_logits):
            return hidden_states + 1

    monkeypatch.setattr(impl.qwen, "MergedReplicatedLinear", Gate)
    monkeypatch.setattr(impl.qwen, "FusedMoE", Experts)
    monkeypatch.setattr(impl.qwen, "install_stacked_expert_loaders", lambda _: None)
    monkeypatch.setattr(impl.qwen, "Qwen3NextMLP", lambda *a, **k: torch.nn.Identity())
    return state, config


def _policy(impl, config, mode):
    q = impl.qwen
    spec = q.LayerQuantConfig(
        quant_dtype=q.dtypes.fp8 if mode == "fp8" else torch.bfloat16,
        quant_type=q.QuantType.per_Token if mode == "fp8" else q.QuantType.No,
        quant_method="compressed-tensors" if mode == "fp8" else None,
    )
    config.quant_config.global_spec = spec
    config.quant_config.quant_method = spec.quant_method
    return spec


def _block(impl, config, *, fused=True):
    return impl.qwen.Qwen4ExpSparseMoeBlock(
        config.hf_config,
        config.quant_config,
        prefix="model.layers.0.mlp",
        fuse_shared_expert=fused,
    )


@pytest.mark.parametrize("mode", ["fp8", "bf16"])
@pytest.mark.parametrize("tp,physical", [(2, 320), (4, 192)])
def test_customer_layout_and_single_token_dispatch(
    impl, setup, monkeypatch, mode, tp, physical
):
    state, config = setup
    state.tp = tp
    _policy(impl, config, mode)
    block = _block(impl, config)
    args = block.experts.args
    assert block.fuse_shared and block.use_gfx942_tuned_moe
    assert args["num_experts"] + args["config"].n_shared_experts == 513
    assert args["top_k"] + args["config"].n_shared_experts == 11
    assert args["pad_align"] == 64
    logical = config.hf_config.moe_intermediate_size // tp
    assert (logical + args["pad_align"] - 1) // args["pad_align"] * args[
        "pad_align"
    ] == physical

    def legacy(*args):
        pytest.fail("Scoped single-token forward must reach FusedMoE")

    monkeypatch.setattr(block, "_single_token_supported", legacy)
    hidden = torch.zeros((1, 2560), dtype=torch.bfloat16)
    torch.testing.assert_close(block(hidden, reduce_results=False), hidden + 1)


OUTSIDE_SCOPE = [
    pytest.param("gfx", "gfx950", None, id="other-arch"),
    pytest.param("cu", 304, None, id="other-gfx942-cu-count"),
    pytest.param("tp", 1, 128, id="tp1"),
    pytest.param("tp", 8, None, id="tp8"),
    pytest.param("enable_expert_parallel", True, None, id="ep"),
    pytest.param("data_parallel_size", 2, None, id="dp"),
    pytest.param("hidden_size", 2048, None, id="other-hidden"),
    pytest.param("moe_intermediate_size", 512, 128, id="other-intermediate"),
    pytest.param("num_experts", 256, None, id="other-experts"),
    pytest.param("num_experts_per_tok", 8, None, id="other-topk"),
    pytest.param(
        "shared_expert_intermediate_size", 1280, None, id="other-shared-width"
    ),
    pytest.param("hidden_act", "gelu", None, id="other-activation"),
]


def _change(state, config, name, value):
    owner = (
        state
        if name in ("gfx", "cu", "tp")
        else (
            config
            if name == "enable_expert_parallel"
            else (
                config.parallel_config
                if name == "data_parallel_size"
                else config.hf_config
            )
        )
    )
    setattr(owner, name, value)


@pytest.mark.parametrize("mode", ["fp8", "bf16"])
@pytest.mark.parametrize("name,value,alignment", OUTSIDE_SCOPE)
def test_other_configurations_keep_previous_layout(
    impl, setup, mode, name, value, alignment
):
    state, config = setup
    _change(state, config, name, value)
    _policy(impl, config, mode)
    block = _block(impl, config)
    assert block.experts.args["pad_align"] == alignment
    assert not block.use_gfx942_tuned_moe


@pytest.mark.parametrize("use_ep", [False, True])
def test_dispatch_uses_actual_expert_parallel_layout(impl, setup, monkeypatch, use_ep):
    """Isolate the final EP guard from the earlier padding policy check.

    A recommended alignment alone must not override the constructed MoE's
    parallel layout. Keeping that recommendation fixed makes dropping the
    final `not self.experts.use_ep` check observable here.
    """
    _, config = setup
    config.enable_expert_parallel = use_ep
    _policy(impl, config, "fp8")
    monkeypatch.setattr(impl.qwen, "moe_pad_align", lambda *a, **k: 64)
    block = _block(impl, config)
    assert block.experts.use_ep is use_ep
    assert block.use_gfx942_tuned_moe is (not use_ep)
    hidden = torch.zeros((1, 2560), dtype=torch.bfloat16)

    # EP must check whether the legacy path can address the local experts;
    # pure TP must bypass that check and dispatch directly to FusedMoE.
    checks = []

    def legacy_supported():
        checks.append(True)
        return False

    monkeypatch.setattr(block, "_single_token_supported", legacy_supported)
    torch.testing.assert_close(block(hidden, reduce_results=False), hidden + 1)
    assert checks == ([True] if use_ep else [])


@pytest.mark.parametrize("mode", ["fp8", "bf16"])
def test_unfused_shared_expert_does_not_select_513_layout(impl, setup, mode):
    _, config = setup
    _policy(impl, config, mode)
    block = _block(impl, config, fused=False)
    assert not block.fuse_shared
    assert block.experts.args["pad_align"] is None
    assert not block.use_gfx942_tuned_moe


@pytest.mark.parametrize(
    "variant", ["block", "static", "native-fp8", "float16", "shared-bf16"]
)
def test_other_quantization_layouts_do_not_opt_in(impl, setup, variant):
    from dataclasses import replace

    _, config = setup
    q = impl.qwen
    spec = _policy(impl, config, "fp8")
    overrides = {
        "block": {"quant_type": q.QuantType.per_1x128, "weight_block_size": (128, 128)},
        "static": {"is_dynamic": False},
        "native-fp8": {"quant_method": "fp8"},
        "float16": {"quant_type": q.QuantType.No, "quant_dtype": torch.float16},
        "shared-bf16": {},
    }
    config.quant_config.global_spec = replace(spec, **overrides[variant])
    if variant == "shared-bf16":
        config.quant_config.exclude_layers = ["model.layers.0.mlp.shared_expert"]
    block = _block(impl, config)
    assert block.experts.args["pad_align"] is None
    assert not block.use_gfx942_tuned_moe
    if variant == "shared-bf16":
        assert not block.fuse_shared


@pytest.mark.parametrize("source", ["bf16", "fp8"])
@pytest.mark.parametrize("excluded", [False, True])
def test_online_sources_and_exclusions_use_runtime_policy(
    impl, setup, source, excluded
):
    _, config = setup
    q = impl.qwen
    _policy(impl, config, source)
    config.quant_config.online_quant = True
    config.quant_config.online_global_spec = q.LayerQuantConfig(
        quant_dtype=q.dtypes.fp8, quant_type=q.QuantType.per_Tensor, quant_method="fp8"
    )
    config.quant_config.online_exclude_layers = ["model.*"] if excluded else []
    block = _block(impl, config)
    assert block.experts.args["pad_align"] == (64 if excluded else None)
    assert block.use_gfx942_tuned_moe is excluded


@pytest.mark.parametrize(
    "name,value,alignment",
    [pytest.param("tp", 2, 64, id="tp2"), pytest.param("tp", 4, 64, id="tp4")]
    + OUTSIDE_SCOPE,
)
def test_mtp_fusion_obeys_same_scope(impl, setup, monkeypatch, name, value, alignment):
    state, config = setup
    _change(state, config, name, value)
    _policy(impl, config, "fp8")
    # Exercise real predictor construction and the block's BF16 policy. Only
    # embeddings, attention, projections and expert storage are CPU stand-ins.
    monkeypatch.setattr(impl.mtp, "_mtp_weights_quantized", lambda _: False)
    for cls in (
        "VocabParallelEmbedding",
        "ReplicatedLinear",
        "Qwen4ExpGroupedRMSNorm",
        "Qwen4ExpHyperConnection",
    ):
        monkeypatch.setattr(impl.mtp, cls, lambda *a, **k: torch.nn.Identity())

    def decoder(
        draft, layer_type, *, prefix, layer_num, quant_config, fuse_shared_expert
    ):
        layer = torch.nn.Module()
        layer.mlp = impl.qwen.Qwen4ExpSparseMoeBlock(
            draft.hf_config,
            quant_config,
            prefix=prefix + ".mlp",
            fuse_shared_expert=fuse_shared_expert,
        )
        return layer

    monkeypatch.setattr(impl.mtp, "Qwen4ExpDecoderLayer", decoder)
    block = impl.mtp.Qwen4ExpMultiTokenPredictor(config).layers[0].mlp
    assert block.fuse_shared is (alignment == 64)
    assert block.use_gfx942_tuned_moe is (alignment == 64)
    assert block.experts.args["pad_align"] == alignment
