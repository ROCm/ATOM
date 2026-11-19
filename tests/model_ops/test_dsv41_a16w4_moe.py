"""A16W4 MoE regression: factory dispatch and weight layout.

CPU coverage
------------
* Factory selects ``A16W4MoEMethod`` when ``ATOM_DSV41_A16W4_MOE=1``,
  gfx942, no EP/SP/mega/EPLB/GU-interleave, **and**
  ``quant_config.online_quant is False``.
* Factory raises with a clear message when ``online_quant=True``.
* ``create_weights`` produces the expected 128-byte-aligned physical shapes
  and zero-fills allocation.
* ``_process_weight_layout_after_loading`` writes ``127`` into the pad-scale
  regions and nowhere else (verified with identity-shuffle stubs).

GPU coverage (gfx942 only) is exercised by the flydsl_a16w4_moe op tests
in aiter; this module stays CPU-only so it can run in CI without a GPU build.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

TEST_ROOT = str(Path(__file__).resolve().parents[1])
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from aiter_stub import ActivationType, QuantType, atom_under_stubs  # noqa: E402
from conftest import atom_config_double  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def atom_modules():
    """Import ATOM model-ops under aiter/triton stubs (no GPU build needed)."""
    with atom_under_stubs() as load:
        yield SimpleNamespace(
            moe=load("atom.model_ops.moe"),
            config=load("atom.config"),
            envs=load("atom.utils.envs"),
        )


def _atom_cfg(*, online_quant=False, model_type="deepseek_v41", **kw):
    """Minimal Config double that satisfies the A16W4 factory gates."""
    return atom_config_double(
        quant_config=SimpleNamespace(
            online_quant=online_quant,
            exclude_layers=[],
            quant_dtype=None,
        ),
        moe_backend="auto",
        eplb_enable=False,
        enable_expert_parallel=False,
        enable_dp_attention=False,
        prefill_context_parallel_size=1,
        speculative_config=None,
        torch_dtype=torch.bfloat16,
        hf_config=SimpleNamespace(model_type=model_type),
        **kw,
    )


def _fake_moe(layer_hidden=128, layer_inter=64):
    """Fake ``FusedMoEConfig`` with only the fields the method path reads."""
    return SimpleNamespace(
        use_ep=False,
        use_mori_kernels=False,
        use_rccl_kernels=False,
        hidden_dim=layer_hidden,
        in_dtype=torch.bfloat16,
        a_quant_dtype=None,
        mxfp4_fp8_activations=False,
        max_num_tokens=64,
        has_bias=False,
        is_lora_enabled=False,
        expert_layout=SimpleNamespace(
            uses_all2all_fusion=False,
            uses_dispatch_remap=False,
            num_fused_shared_experts=0,
            routed_physical_per_rank=0,
        ),
        moe_parallel_config=SimpleNamespace(low_latency=False, tp_size=4, dp_size=1),
    )


def _patch_shuffle_identity(monkeypatch):
    """Replace aiter shuffle helpers with identity so CPU tests can reach
    the pad-fill logic without crashing on a missing AITER build."""
    import aiter.ops.shuffle as _s

    monkeypatch.setattr(_s, "shuffle_weight_a16w4", lambda w, *a, **k: w, raising=False)
    monkeypatch.setattr(_s, "shuffle_scale_a16w4", lambda s, *a, **k: s, raising=False)


# ---------------------------------------------------------------------------
# 1. Factory dispatch (CPU)
# ---------------------------------------------------------------------------


class TestA16W4FactoryCpu:
    """Factory gates under mocked gfx942 / no online_quant."""

    @pytest.fixture(autouse=True)
    def _a16_env(self, monkeypatch):
        monkeypatch.setenv("ATOM_DSV41_A16W4_MOE", "1")

    def test_selects_a16w4_when_no_online_quant(self, monkeypatch, atom_modules):
        monkeypatch.setattr(atom_modules.moe, "get_gfx", lambda: "gfx942")
        monkeypatch.setattr(atom_modules.moe, "sp_is_enabled", lambda: False)
        monkeypatch.setattr(
            atom_modules.config, "_current_atom_config", _atom_cfg(online_quant=False)
        )

        from atom.quant_spec import LayerQuantConfig

        method = atom_modules.moe._make_mxfp4_moe_method(
            LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=torch.uint8),
            _fake_moe(),
        )
        assert isinstance(method, atom_modules.moe.A16W4MoEMethod)

    def test_rejects_online_quant_with_clear_error(self, monkeypatch, atom_modules):
        monkeypatch.setattr(atom_modules.moe, "get_gfx", lambda: "gfx942")
        monkeypatch.setattr(atom_modules.moe, "sp_is_enabled", lambda: False)
        monkeypatch.setattr(
            atom_modules.config, "_current_atom_config", _atom_cfg(online_quant=True)
        )

        from atom.quant_spec import LayerQuantConfig

        with pytest.raises(RuntimeError, match="native MXFP4 experts, not online"):
            atom_modules.moe._make_mxfp4_moe_method(
                LayerQuantConfig(
                    quant_type=QuantType.per_1x32, quant_dtype=torch.uint8
                ),
                _fake_moe(),
            )

    def test_rejects_wrong_model_type(self, monkeypatch, atom_modules):
        monkeypatch.setattr(atom_modules.moe, "get_gfx", lambda: "gfx942")
        monkeypatch.setattr(atom_modules.moe, "sp_is_enabled", lambda: False)
        bad_cfg = _atom_cfg(online_quant=False)
        bad_cfg.hf_config = SimpleNamespace(model_type="llama")
        monkeypatch.setattr(atom_modules.config, "_current_atom_config", bad_cfg)

        from atom.quant_spec import LayerQuantConfig

        with pytest.raises(RuntimeError, match="not supported for model_type"):
            atom_modules.moe._make_mxfp4_moe_method(
                LayerQuantConfig(
                    quant_type=QuantType.per_1x32, quant_dtype=torch.uint8
                ),
                _fake_moe(),
            )

    def test_rejects_non_tp4(self, monkeypatch, atom_modules):
        monkeypatch.setattr(atom_modules.moe, "get_gfx", lambda: "gfx942")
        monkeypatch.setattr(atom_modules.moe, "sp_is_enabled", lambda: False)
        monkeypatch.setattr(
            atom_modules.config, "_current_atom_config", _atom_cfg(online_quant=False)
        )

        from atom.quant_spec import LayerQuantConfig

        moe = _fake_moe()
        moe.moe_parallel_config = SimpleNamespace(low_latency=False, tp_size=2)
        with pytest.raises(RuntimeError, match="requires TP4, got TP2"):
            atom_modules.moe._make_mxfp4_moe_method(
                LayerQuantConfig(
                    quant_type=QuantType.per_1x32, quant_dtype=torch.uint8
                ),
                moe,
            )

    def test_disabled_env_keeps_mxfp4(self, monkeypatch, atom_modules):
        monkeypatch.setenv("ATOM_DSV41_A16W4_MOE", "0")
        monkeypatch.setattr(atom_modules.moe, "get_gfx", lambda: "gfx942")
        monkeypatch.setattr(atom_modules.moe, "sp_is_enabled", lambda: False)
        monkeypatch.setattr(
            atom_modules.config, "_current_atom_config", _atom_cfg(online_quant=False)
        )

        from atom.quant_spec import LayerQuantConfig

        method = atom_modules.moe._make_mxfp4_moe_method(
            LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=torch.uint8),
            _fake_moe(),
        )
        assert type(method) is atom_modules.moe.Mxfp4MoEMethod


# ---------------------------------------------------------------------------
# 2. Weight layout (CPU)
# ---------------------------------------------------------------------------


class TestA16W4WeightLayoutCpu:
    """create_weights sizing and _process_weight_layout_after_loading pad fill."""

    @pytest.fixture(autouse=True)
    def _a16_env(self, monkeypatch):
        monkeypatch.setenv("ATOM_DSV41_A16W4_MOE", "1")

    def test_create_weights_128_padding_and_shape(self, monkeypatch, atom_modules):
        monkeypatch.setattr(atom_modules.moe, "get_gfx", lambda: "gfx942")
        monkeypatch.setattr(atom_modules.moe, "sp_is_enabled", lambda: False)
        monkeypatch.setattr(
            atom_modules.config, "_current_atom_config", _atom_cfg(online_quant=False)
        )

        from atom.quant_spec import LayerQuantConfig

        method = atom_modules.moe.A16W4MoEMethod(
            LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=torch.uint8),
            _fake_moe(layer_hidden=128, layer_inter=64),
        )

        layer = torch.nn.Module()
        layer.hidden_size = 128
        layer.intermediate_size_per_partition = 64
        layer.has_bias = False
        layer.num_fused_shared_experts = 0

        method.create_weights(
            layer,
            num_experts=4,
            hidden_size=128,
            intermediate_size_per_partition=64,
            params_dtype=torch.uint8,
        )

        # pad_align=128  =>  logical 64 -> physical 128
        assert method.intermediate_size == 128
        assert method.hidden_size == 128
        assert method.intermediate_pad == 64

        # w13  [E, 2*physical, hidden/2]   = [4, 256, 64]
        # w2   [E, hidden, physical/2]     = [4, 128, 64]
        assert layer.w13_weight.shape == (4, 256, 64)
        assert layer.w2_weight.shape == (4, 128, 64)

        # scales
        assert layer.w13_weight_scale.shape == (4, 256, 4)  # hidden//32 = 4
        assert layer.w2_weight_scale.shape == (4, 128, 4)  # physical//32 = 4

        # zero-filled at allocation (padding bytes are zero)
        assert layer.w13_weight.data.view(torch.uint8).sum() == 0
        assert layer.w2_weight.data.view(torch.uint8).sum() == 0

    def test_process_weights_fills_pad_scales_127(self, monkeypatch, atom_modules):
        monkeypatch.setattr(atom_modules.moe, "get_gfx", lambda: "gfx942")
        monkeypatch.setattr(atom_modules.moe, "sp_is_enabled", lambda: False)
        monkeypatch.setattr(
            atom_modules.config, "_current_atom_config", _atom_cfg(online_quant=False)
        )

        from atom.quant_spec import LayerQuantConfig

        method = atom_modules.moe.A16W4MoEMethod(
            LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=torch.uint8),
            _fake_moe(layer_hidden=128, layer_inter=64),
        )

        layer = torch.nn.Module()
        layer.hidden_size = 128
        layer.intermediate_size_per_partition = 64
        layer.has_bias = False
        layer.num_fused_shared_experts = 0
        layer.activation = ActivationType.Silu
        layer.swiglu_limit = 10.0

        method.create_weights(
            layer,
            num_experts=4,
            hidden_size=128,
            intermediate_size_per_partition=64,
            params_dtype=torch.uint8,
        )

        # populate scales with non-pad values so we can detect the overwrite
        layer.w13_weight_scale.data.fill_(123)
        layer.w2_weight_scale.data.fill_(124)
        layer.w13_weight.data[:, :64].fill_(0x12)
        layer.w13_weight.data[:, 128:192].fill_(0x34)
        layer.w2_weight.data[:, :, :32].fill_(0x56)
        before_w13 = layer.w13_weight.data.clone()
        before_w2 = layer.w2_weight.data.clone()

        _patch_shuffle_identity(monkeypatch)
        method._process_weight_layout_after_loading(layer)
        assert torch.equal(layer.w13_weight.data, before_w13)
        assert torch.equal(layer.w2_weight.data, before_w2)

        E, logical, physical = 4, 64, 128

        # w13 scales flattened -> (E*2*physical, hidden//32) = (1024, 4)
        assert layer.w13_weight_scale.shape == (E * 2 * physical, 128 // 32)
        for e in range(E):
            base = e * 2 * physical
            # gate half padding  [logical : physical]
            assert (
                layer.w13_weight_scale.data[base + logical : base + physical] == 127
            ).all()
            # up half padding   [physical+logical : 2*physical]
            assert (
                layer.w13_weight_scale.data[
                    base + physical + logical : base + 2 * physical
                ]
                == 127
            ).all()
            # real gate half NOT all 127
            assert not (layer.w13_weight_scale.data[base : base + logical] == 127).all()
            # real up half NOT all 127
            assert not (
                layer.w13_weight_scale.data[base + physical : base + physical + logical]
                == 127
            ).all()

        # w2 scales flattened -> (E*hidden, physical//32) = (512, 4)
        assert layer.w2_weight_scale.shape == (E * 128, 4)
        # logical//32 = 2 real columns; remaining 2 are padding
        assert (layer.w2_weight_scale.data[:, logical // 32 :] == 127).all()
        assert not (layer.w2_weight_scale.data[:, : logical // 32] == 127).all()
