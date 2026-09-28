# SPDX-License-Identifier: MIT
# Tests for LayerQuantConfig, QuantizationConfig, and the
# parser registry (atom/config.py + atom/quant_spec.py).
#
# Covers: per-layer quant config dispatch, quark config parsing,
# layer name matching (exact / regex / fnmatch), packed-module remapping,
# typed LayerQuantConfig API, and backward compatibility.
#
# atom.config depends on torch, aiter, and transformers.  We load the source
# files under temporary sys.modules mocks so the tests run in any environment.

import contextlib
import enum
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ATOM_ROOT = str(Path(__file__).resolve().parent.parent)

# -------------------------------------------------------------------------
# Mock primitives
# -------------------------------------------------------------------------


class QuantType(enum.IntEnum):
    No = 0
    per_Token = 1
    per_Tensor = 2
    per_1x32 = 3
    per_1x128 = 4


BF16 = "torch.bfloat16"
FP8 = "mock_fp8"
FP4X2 = "mock_fp4x2"
INT8 = "mock_int8"

D_DTYPES = {
    "fp8": FP8,
    "fp4x2": FP4X2,
    "int8": INT8,
    "int4x2": "mock_int4x2",
    "i8": INT8,
    "i4x2": "mock_int4x2",
}


class FakeHFConfig:
    """Lightweight stand-in for transformers.PretrainedConfig."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def get_config_dict(model):
        return {}, {}


# -------------------------------------------------------------------------
# Module loader — patch sys.modules only while exec-ing config.py
# -------------------------------------------------------------------------


@contextlib.contextmanager
def _temporary_mocks():
    mock_torch = MagicMock()
    mock_torch.bfloat16 = BF16

    mock_aiter = types.ModuleType("aiter")
    mock_aiter.QuantType = QuantType

    mock_aiter_dtypes = types.ModuleType("aiter.utility.dtypes")
    mock_aiter_dtypes.d_dtypes = D_DTYPES

    mock_transformers = types.ModuleType("transformers")
    mock_transformers.PretrainedConfig = FakeHFConfig
    mock_transformers.AutoConfig = MagicMock()
    mock_transformers.GenerationConfig = MagicMock()

    mock_atom_utils = types.ModuleType("atom.utils")
    mock_atom_utils.envs = MagicMock()
    mock_atom_utils.get_open_port = MagicMock(return_value=8000)

    mock_dist_utils = types.ModuleType("atom.utils.distributed.utils")
    mock_dist_utils.stateless_init_torch_distributed_process_group = MagicMock()

    mock_aiter.__path__ = []

    mock_plugin = types.ModuleType("atom.plugin")
    mock_plugin.is_plugin_mode = MagicMock(return_value=False)
    mock_plugin.is_vllm = MagicMock(return_value=False)
    mock_plugin_config = types.ModuleType("atom.plugin.config")
    mock_plugin_config.PluginConfig = MagicMock()

    patches = {
        "torch": mock_torch,
        "torch.distributed": MagicMock(),
        "aiter": mock_aiter,
        "aiter.utility": types.ModuleType("aiter.utility"),
        "aiter.utility.dtypes": mock_aiter_dtypes,
        "transformers": mock_transformers,
        "atom.utils": mock_atom_utils,
        "atom.utils.distributed": types.ModuleType("atom.utils.distributed"),
        "atom.utils.distributed.utils": mock_dist_utils,
        "atom.plugin": mock_plugin,
        "atom.plugin.config": mock_plugin_config,
    }

    saved = {}
    for name, mock in patches.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock
    try:
        yield
    finally:
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


def _load_module(filename: str, module_name: str):
    path = os.path.join(ATOM_ROOT, "atom", filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass etc. can resolve the module
    sys.modules[module_name] = mod
    with _temporary_mocks():
        spec.loader.exec_module(mod)
        # `quant_spec` resolves its AITER handles on first *use* rather than at
        # import, so executing the module body is no longer enough to bind
        # them. Touch them while the stand-ins above are still installed --
        # afterwards there is no aiter to resolve against on a CPU-only runner.
        #
        # Deliberately not left in `sys.modules` instead: a lingering fake
        # `aiter` would satisfy `pytest.importorskip("aiter")` in the other
        # test modules, and whether it did would depend on collection order.
        if hasattr(mod, "QuantType"):
            _ = mod.QuantType.No
            _ = mod.d_dtypes.get("fp8")
    return mod


# Load quant_spec first, then inject it so config.py can import it -- and put
# the real one back, which the previous version of this never did.
#
# Leaving this copy installed replaced `atom.quant_spec` for the whole session.
# Modules imported before this file kept the original `LayerQuantConfig` and
# `QuantType`; modules imported after got these. Two enums that print the same
# and compare unequal, so `quant_type != QuantType.No` was true for a config
# that said `No` -- `test_qwen4_exp_quantization` then built a quantized layer
# and died on a `weight_scale` that branch does not create. `tests/conftest.py`
# now fails whichever test leaves such a duplicate behind.
_real_quant_spec = sys.modules.get("atom.quant_spec")
_qs = _load_module("quant_spec.py", "atom.quant_spec")
try:
    _m = _load_module("config.py", "_atom_config_test")
finally:
    if _real_quant_spec is None:
        sys.modules.pop("atom.quant_spec", None)
    else:
        sys.modules["atom.quant_spec"] = _real_quant_spec

QuantizationConfig = _m.QuantizationConfig
LayerQuantConfig = _qs.LayerQuantConfig
NVFP4_DTYPE = _qs.NVFP4_DTYPE
QuarkParser = _qs.QuarkParser
QuarkOnlineParser = _qs.QuarkOnlineParser
ModelOptParser = _qs.ModelOptParser
GenericParser = _qs.GenericParser
get_quant_parser = _qs.get_quant_parser
will_online_requant = _qs.will_online_requant
validate_nvfp4_online_target = _qs.validate_nvfp4_online_target


# =========================================================================
# Tests — LayerQuantConfig
# =========================================================================


class TestLayerQuantConfig:
    def test_defaults(self):
        spec = LayerQuantConfig()
        assert spec.quant_type == QuantType.No
        assert spec.quant_dtype == BF16
        assert spec.is_dynamic is True
        assert spec.quant_method is None
        assert spec.is_quantized is False

    def test_no_quant_factory(self):
        spec = LayerQuantConfig.no_quant(FP8)
        assert spec.quant_type == QuantType.No
        assert spec.quant_dtype == FP8
        assert spec.is_quantized is False

    def test_is_quantized(self):
        spec = LayerQuantConfig(quant_type=QuantType.per_Token, quant_dtype=FP8)
        assert spec.is_quantized is True

    def test_nvfp4_dtype_is_explicit(self):
        spec = LayerQuantConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=NVFP4_DTYPE,
        )
        assert spec.quant_dtype == "nvfp4"

    def test_frozen(self):
        spec = LayerQuantConfig()
        with pytest.raises(AttributeError):
            spec.quant_type = QuantType.per_Token  # type: ignore[misc]


# =========================================================================
# Tests — Parser Registry
# =========================================================================


class TestParserRegistry:
    def test_quark_registered(self):
        parser = get_quant_parser("quark")
        assert isinstance(parser, QuarkParser)

    def test_online_quant_registered(self):
        parser = get_quant_parser("online_quant")
        assert isinstance(parser, QuarkOnlineParser)

    def test_modelopt_registered(self):
        parser = get_quant_parser("modelopt")
        assert isinstance(parser, ModelOptParser)

    def test_generic_fallback(self):
        parser = get_quant_parser("compressed-tensors")
        assert isinstance(parser, GenericParser)

    def test_unknown_falls_to_generic(self):
        parser = get_quant_parser("some_unknown_method")
        assert isinstance(parser, GenericParser)


class TestGenericParser:
    def test_fbgemm_fp8_uses_per_tensor_scales(self):
        parser = GenericParser()
        result = parser.parse(
            {
                "quant_method": "fbgemm_fp8",
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
            }
        )

        assert result.global_spec.quant_type == QuantType.per_Tensor
        assert result.global_spec.quant_dtype == FP8
        assert result.global_spec.is_dynamic is True
        assert result.global_spec.quant_method == "fbgemm_fp8"


# =========================================================================
# Tests — ModelOptParser
# =========================================================================


class TestModelOptParser:
    @staticmethod
    def _mixed_config():
        return {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "exclude_modules": ["language_model.model.embed_tokens", "lm_head"],
            "quantized_layers": {
                "language_model.model.layers.0.mlp.gate_proj": {"quant_algo": "MXFP8"},
                "language_model.model.layers.0.mlp.up_proj": {"quant_algo": "MXFP8"},
                "language_model.model.layers.3.block_sparse_moe.shared_experts.gate_proj": {
                    "quant_algo": "MXFP8"
                },
                "language_model.model.layers.3.block_sparse_moe.shared_experts.up_proj": {
                    "quant_algo": "MXFP8"
                },
                "language_model.model.layers.3.block_sparse_moe.experts.0.w1": {
                    "quant_algo": "NVFP4"
                },
                "language_model.model.layers.3.block_sparse_moe.experts.1.w3": {
                    "quant_algo": "NVFP4"
                },
            },
        }

    def test_mixed_mxfp8_and_nvfp4(self):
        result = ModelOptParser().parse(self._mixed_config())
        patterns = dict(result.layer_pattern_specs)

        assert result.global_spec.quant_type == QuantType.No
        dense = patterns["language_model.model.layers.0.mlp.gate_proj"]
        assert dense.quant_type == QuantType.per_1x32
        assert dense.quant_dtype == FP8
        experts = patterns["language_model.model.layers.3.block_sparse_moe.experts"]
        assert experts.quant_type == QuantType.per_1x32
        assert experts.quant_dtype == NVFP4_DTYPE
        assert (
            len(
                [
                    name
                    for name, _ in result.layer_pattern_specs
                    if ".block_sparse_moe.experts" in name
                ]
            )
            == 1
        )
        assert result.exclude_layers == [
            "language_model.model.embed_tokens",
            "lm_head",
        ]

    def test_mixed_config_remaps_wrapper_and_packed_names(self):
        hf = FakeHFConfig(
            torch_dtype=BF16,
            quantization_config=self._mixed_config(),
        )
        qcfg = QuantizationConfig(
            hf,
            online_quant_config={"global_quant_config": "mxfp4"},
        )
        qcfg.remap_layer_name(
            FakeHFConfig(model_type="minimax_m3_vl"),
            packed_modules_mapping={
                ".gate_proj": (".gate_up_proj", 0),
                ".up_proj": (".gate_up_proj", 1),
            },
            quant_exclude_name_mapping={
                "language_model.model.": "model.",
            },
        )

        dense = qcfg.get_layer_quant_config("model.layers.0.mlp.gate_up_proj")
        experts = qcfg.get_layer_quant_config("model.layers.3.block_sparse_moe.experts")
        wrapped_experts = qcfg.get_layer_quant_config(
            "language_model.model.layers.3.block_sparse_moe.experts"
        )
        wrapped_shared = qcfg.get_layer_quant_config(
            "language_model.model.layers.3.block_sparse_moe.shared_experts"
        )
        assert dense.quant_dtype == FP8
        assert experts.quant_dtype == NVFP4_DTYPE
        assert wrapped_experts == experts
        assert wrapped_shared.quant_dtype == FP8
        assert qcfg.online_quant is True
        assert (
            qcfg.get_layer_quant_config(
                "model.layers.3.block_sparse_moe.experts",
                use_online_quant=True,
            ).quant_dtype
            == FP4X2
        )
        assert "model.embed_tokens" in qcfg.exclude_layers

    @staticmethod
    def _parse_and_remap_like_m3(config):
        qcfg = QuantizationConfig(
            FakeHFConfig(torch_dtype=BF16, quantization_config=config),
            online_quant_config={"global_quant_config": "mxfp4"},
        )
        qcfg.remap_layer_name(
            FakeHFConfig(model_type="minimax_m3_vl"),
            packed_modules_mapping={
                ".gate_proj": (".gate_up_proj", 0),
                ".up_proj": (".gate_up_proj", 1),
            },
            quant_exclude_name_mapping={"language_model.model.": "model."},
        )
        return qcfg

    def test_excluding_one_expert_is_rejected(self):
        """FusedMoE resolves its experts container with check_children=True, so
        one excluded expert would build every expert in the layer as bf16."""
        config = self._mixed_config()
        config["exclude_modules"].append(
            "language_model.model.layers.3.block_sparse_moe.experts.5.w1"
        )
        with pytest.raises(ValueError, match="excluded by another"):
            self._parse_and_remap_like_m3(config)

    def test_excluding_the_router_keeps_experts_quantized(self):
        config = self._mixed_config()
        config["exclude_modules"].append(
            "language_model.model.layers.3.block_sparse_moe.gate"
        )
        qcfg = self._parse_and_remap_like_m3(config)
        experts = qcfg.get_layer_quant_config(
            "model.layers.3.block_sparse_moe.experts", check_children=True
        )
        assert experts.quant_dtype == NVFP4_DTYPE

    def test_nvfp4_without_online_target_is_rejected_at_setup(self):
        """NVFP4 without an online config fails while the config is parsed."""
        hf = FakeHFConfig(torch_dtype=BF16, quantization_config=self._mixed_config())
        with pytest.raises(ValueError, match="no online quantization config"):
            QuantizationConfig(hf)

    def test_non_nvfp4_checkpoint_needs_no_online_target(self):
        """Guards the check above from rejecting every quantized checkpoint."""
        config = self._mixed_config()
        config["quantized_layers"] = {
            "language_model.model.layers.0.mlp.gate_proj": {"quant_algo": "MXFP8"}
        }
        qcfg = QuantizationConfig(
            FakeHFConfig(torch_dtype=BF16, quantization_config=config)
        )
        assert qcfg.online_quant is False

    def test_nvfp4_layer_without_mxfp4_online_target_is_rejected(self):
        """Resolves the online target the same way the layer does when built."""
        experts = "language_model.model.layers.3.block_sparse_moe.experts"
        hf = FakeHFConfig(torch_dtype=BF16, quantization_config=self._mixed_config())
        excluded = QuantizationConfig(
            hf,
            online_quant_config={
                "global_quant_config": "mxfp4",
                "exclude_layer": [experts],
            },
        )
        with pytest.raises(ValueError, match="excluded from online quantization"):
            validate_nvfp4_online_target(excluded, experts)

        converted = QuantizationConfig(
            hf, online_quant_config={"global_quant_config": "mxfp4"}
        )
        validate_nvfp4_online_target(converted, experts)

    # `validate_nvfp4_global_scales` inspects tensor values, so it is covered in
    # tests/test_nvfp4_loading.py instead -- the `torch` this copy of
    # quant_spec closed over is a MagicMock, and every check would pass.

    def test_nvfp4_rejects_non_16_group_size(self):
        config = self._mixed_config()
        config["quantized_layers"] = {
            "model.layers.0.mlp.experts.0.w1": {
                "quant_algo": "NVFP4",
                "group_size": 32,
            }
        }
        with pytest.raises(ValueError, match="group_size=16"):
            ModelOptParser().parse(config)

    @pytest.mark.parametrize("quant_algo", ["NVFP4", "MXFP8", "FP8", ""])
    def test_only_mixed_precision_is_read(self, quant_algo):
        """A uniform `quant_algo` is rejected rather than guessed at.

        The generic heuristics would substring-match the "fp4" inside "nvfp4"
        and report group-32 MXFP4, and they read neither `quantized_layers`
        nor `exclude_modules`.
        """
        with pytest.raises(ValueError, match="MIXED_PRECISION"):
            ModelOptParser().parse(
                {
                    "quant_method": "modelopt",
                    "quant_algo": quant_algo,
                    "exclude_modules": ["lm_head"],
                }
            )


# =========================================================================
# Tests — QuarkParser
# =========================================================================


class TestQuarkParser:
    def test_per_channel_fp8(self):
        parser = QuarkParser()
        result = parser.parse(
            {
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"qscheme": "per_channel", "dtype": "fp8_e4m3"},
                    "input_tensors": {"is_dynamic": True},
                },
            }
        )
        assert result.global_spec.quant_type == QuantType.per_Token
        assert result.global_spec.quant_dtype == FP8
        assert result.global_spec.is_dynamic is True

    def test_per_group_fp4(self):
        parser = QuarkParser()
        result = parser.parse(
            {
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"qscheme": "per_group", "dtype": "fp4_e2m1"},
                    "input_tensors": {"is_dynamic": False},
                },
            }
        )
        assert result.global_spec.quant_type == QuantType.per_1x32
        assert result.global_spec.quant_dtype == FP4X2
        assert result.global_spec.is_dynamic is False

    def test_two_stage_nvfp4(self):
        parser = QuarkParser()
        result = parser.parse(
            {
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": [
                        {
                            "qscheme": "per_group",
                            "dtype": "fp4",
                            "group_size": 16,
                            "is_dynamic": False,
                            "is_scale_quant": False,
                        },
                        {
                            "qscheme": "per_tensor",
                            "dtype": "fp8_e4m3",
                            "is_dynamic": False,
                            "is_scale_quant": True,
                        },
                    ],
                    "input_tensors": [
                        {
                            "qscheme": "per_group",
                            "dtype": "fp4",
                            "group_size": 16,
                            "is_dynamic": True,
                            "is_scale_quant": False,
                        },
                        {
                            "qscheme": "per_tensor",
                            "dtype": "fp8_e4m3",
                            "is_dynamic": False,
                            "is_scale_quant": True,
                        },
                    ],
                },
            }
        )
        spec = result.global_spec
        assert spec.quant_type == QuantType.per_1x32
        assert spec.quant_dtype == NVFP4_DTYPE
        assert spec.is_dynamic is True

    @pytest.mark.parametrize(
        ("section", "stage", "field", "invalid_value"),
        [
            ("weight", 0, "is_dynamic", True),
            ("input_tensors", 0, "is_dynamic", False),
            ("input_tensors", 0, "group_size", 32),
            ("weight", 1, "is_dynamic", True),
            ("input_tensors", 1, "is_dynamic", True),
        ],
    )
    def test_two_stage_nvfp4_rejects_mismatched_stage(
        self, section, stage, field, invalid_value
    ):
        layer_config = {
            "weight": [
                {
                    "qscheme": "per_group",
                    "dtype": "fp4",
                    "group_size": 16,
                    "is_dynamic": False,
                },
                {
                    "qscheme": "per_tensor",
                    "dtype": "fp8_e4m3",
                    "is_dynamic": False,
                },
            ],
            "input_tensors": [
                {
                    "qscheme": "per_group",
                    "dtype": "fp4",
                    "group_size": 16,
                    "is_dynamic": True,
                },
                {
                    "qscheme": "per_tensor",
                    "dtype": "fp8_e4m3",
                    "is_dynamic": False,
                },
            ],
        }
        layer_config[section][stage][field] = invalid_value

        with pytest.raises(ValueError, match="recognizes only NVFP4"):
            QuarkParser().parse(
                {
                    "quant_method": "quark",
                    "global_quant_config": layer_config,
                }
            )

    def test_two_stage_nvfp4_requires_matching_input_stages(self):
        with pytest.raises(ValueError, match="both `weight` and `input_tensors`"):
            QuarkParser().parse(
                {
                    "quant_method": "quark",
                    "global_quant_config": {
                        "weight": [
                            {
                                "qscheme": "per_group",
                                "dtype": "fp4",
                                "group_size": 16,
                                "is_dynamic": False,
                            },
                            {
                                "qscheme": "per_tensor",
                                "dtype": "fp8_e4m3",
                                "is_dynamic": False,
                            },
                        ],
                        "input_tensors": {"is_dynamic": True},
                    },
                }
            )

    def test_unknown_sequential_quark_format_is_rejected(self):
        parser = QuarkParser()
        with pytest.raises(ValueError, match="recognizes only NVFP4"):
            parser.parse(
                {
                    "quant_method": "quark",
                    "global_quant_config": {
                        "weight": [
                            {
                                "qscheme": "per_group",
                                "dtype": "fp4",
                                "group_size": 32,
                            },
                            {
                                "qscheme": "per_tensor",
                                "dtype": "fp8_e4m3",
                                "is_scale_quant": True,
                            },
                        ]
                    },
                }
            )

    def test_no_input_tensors_defaults_dynamic(self):
        parser = QuarkParser()
        result = parser.parse(
            {
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"qscheme": "per_tensor", "dtype": "int8"},
                    "input_tensors": None,
                },
            }
        )
        assert result.global_spec.quant_type == QuantType.per_Tensor
        assert result.global_spec.is_dynamic is True

    def test_layer_config_parsed(self):
        parser = QuarkParser()
        result = parser.parse(
            {
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"qscheme": "per_channel", "dtype": "fp8_e4m3"},
                    "input_tensors": {"is_dynamic": True},
                },
                "layer_quant_config": {
                    "*.mlp.*": {
                        "weight": {"qscheme": "per_group", "dtype": "fp4_e2m1"},
                        "input_tensors": {"is_dynamic": False},
                    },
                },
                "exclude": ["lm_head"],
            }
        )
        assert len(result.layer_pattern_specs) == 1
        pattern, spec = result.layer_pattern_specs[0]
        assert pattern == "*.mlp.*"
        assert spec.quant_type == QuantType.per_1x32
        assert spec.quant_dtype == FP4X2
        assert result.exclude_layers == ["lm_head"]


# =========================================================================
# Tests — QuarkOnlineParser
# =========================================================================


class TestQuarkOnlineParser:
    def test_ptpc_fp8_global_config(self):
        parser = QuarkOnlineParser()
        result = parser.parse({"global_quant_config": "ptpc_fp8"})

        assert result.global_spec.quant_type == QuantType.per_Token
        assert result.global_spec.quant_dtype == FP8
        assert result.global_spec.is_dynamic is True
        assert result.global_spec.quant_method == "quark"
        assert result.layer_pattern_specs == []
        assert result.exclude_layers == []

    def test_mxfp4_layer_override_and_exclude_list(self):
        parser = QuarkOnlineParser()
        result = parser.parse(
            {
                "global_quant_config": "ptpc_fp8",
                "layer_quant_config": {"*expert*": "mxfp4"},
                "exclude_layer": ["lm_head", "*.gate.*"],
            }
        )

        assert result.global_spec.quant_type == QuantType.per_Token
        assert result.global_spec.quant_dtype == FP8
        assert len(result.layer_pattern_specs) == 1
        pattern, spec = result.layer_pattern_specs[0]
        assert pattern == "*expert*"
        assert spec.quant_type == QuantType.per_1x32
        assert spec.quant_dtype == FP4X2
        assert result.exclude_layers == ["lm_head", "*.gate.*"]

    def test_string_exclude_layer_is_preserved_as_single_pattern(self):
        parser = QuarkOnlineParser()
        result = parser.parse(
            {
                "global_quant_config": "ptpc_fp8",
                "exclude_layer": "lm_head",
            }
        )

        assert result.exclude_layers == ["lm_head"]

    def test_empty_config_returns_no_quant_defaults(self):
        parser = QuarkOnlineParser()
        result = parser.parse({})

        assert result.global_spec.quant_type == QuantType.No
        assert result.global_spec.quant_dtype == BF16
        assert result.layer_pattern_specs == []
        assert result.exclude_layers == []

    def test_invalid_online_quant_format_raises(self):
        parser = QuarkOnlineParser()

        with pytest.raises(ValueError, match="Unsupported online quant format"):
            parser.parse({"global_quant_config": "unsupported_fp8"})


# =========================================================================
# Tests — QuantizationConfig init
# =========================================================================


class TestQuantizationConfigInit:
    def test_none_config(self):
        qcfg = QuantizationConfig(config=None)
        assert qcfg.quant_method == ""
        assert qcfg.exclude_layers == []
        assert qcfg.global_quant_config.quant_type == QuantType.No
        assert qcfg.global_quant_config.is_quantized is False

    def test_config_without_quantization(self):
        hf = FakeHFConfig(torch_dtype=BF16)
        qcfg = QuantizationConfig(hf)
        assert qcfg.quant_method == ""
        assert qcfg.global_quant_config.quant_type == QuantType.No
        assert qcfg.global_quant_config.quant_dtype == BF16
        assert qcfg.online_quant is False
        assert qcfg.online_global_spec.quant_type == QuantType.No
        assert qcfg.online_layer_pattern_specs == []
        assert qcfg.online_exclude_layers == []

    def test_empty_online_quant_config_does_not_enable_online_quant(self):
        hf = FakeHFConfig(torch_dtype=BF16)
        qcfg = QuantizationConfig(hf, online_quant_config={})

        assert qcfg.online_quant is False
        assert qcfg.online_quant_config_raw == {}
        assert qcfg.online_global_spec.quant_type == QuantType.No
        assert qcfg.online_layer_pattern_specs == []
        assert qcfg.online_exclude_layers == []

    def test_online_quant_config_parses_global_layer_and_exclude(self):
        hf = FakeHFConfig(torch_dtype=BF16)
        qcfg = QuantizationConfig(
            hf,
            online_quant_config={
                "global_quant_config": "ptpc_fp8",
                "layer_quant_config": {"*expert*": "mxfp4"},
                "exclude_layer": ["lm_head", "*.gate.*"],
            },
        )

        assert qcfg.online_quant is True
        assert qcfg.online_global_spec.quant_type == QuantType.per_Token
        assert qcfg.online_global_spec.quant_dtype == FP8
        assert len(qcfg.online_layer_pattern_specs) == 1
        pattern, spec = qcfg.online_layer_pattern_specs[0]
        assert pattern == "*expert*"
        assert spec.quant_type == QuantType.per_1x32
        assert spec.quant_dtype == FP4X2
        assert qcfg.online_exclude_layers == ["lm_head", "*.gate.*"]

    def test_quark_config_parses_global_and_layer(self):
        hf = FakeHFConfig(
            torch_dtype=BF16,
            quantization_config={
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"qscheme": "per_channel", "dtype": "fp8_e4m3"},
                    "input_tensors": {"is_dynamic": True},
                },
                "layer_quant_config": {
                    "*.mlp.*": {
                        "weight": {"qscheme": "per_group", "dtype": "fp4_e2m1"},
                        "input_tensors": {"is_dynamic": False},
                    }
                },
                "exclude": ["lm_head"],
            },
        )
        qcfg = QuantizationConfig(hf)
        assert qcfg.quant_method == "quark"
        assert qcfg.global_quant_config.quant_type == QuantType.per_Token
        assert qcfg.global_quant_config.quant_dtype == FP8
        # layer pattern specs
        assert len(qcfg.layer_pattern_specs) == 1
        mlp_pattern, mlp_spec = qcfg.layer_pattern_specs[0]
        assert mlp_pattern == "*.mlp.*"
        assert mlp_spec.quant_type == QuantType.per_1x32
        assert mlp_spec.quant_dtype == FP4X2
        assert mlp_spec.is_dynamic is False

        assert qcfg.exclude_layers == ["lm_head"]


# =========================================================================
# Tests — get_layer_quant_config resolution
# =========================================================================


class TestGetLayerQuantConfig:
    def test_falls_back_to_global(self):
        qcfg = QuantizationConfig(config=None)
        qcfg.global_spec = LayerQuantConfig(
            quant_type=QuantType.per_Token, quant_dtype=FP8
        )
        result = qcfg.get_layer_quant_config("model.layers.0.self_attn.q_proj")
        assert result.quant_type == QuantType.per_Token
        assert result.quant_dtype == FP8

    def test_layer_specific_overrides_global(self):
        qcfg = QuantizationConfig(config=None)
        qcfg.global_spec = LayerQuantConfig(quant_dtype=FP8)
        qcfg.layer_pattern_specs = [
            (
                "*.mlp.*",
                LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=FP4X2),
            ),
        ]
        result = qcfg.get_layer_quant_config("model.layers.0.mlp.gate_proj")
        assert result.quant_dtype == FP4X2
        assert result.quant_type == QuantType.per_1x32

    def test_excluded_layer_returns_unquantized(self):
        qcfg = QuantizationConfig(config=None)
        qcfg.torch_dtype = BF16
        qcfg.global_spec = LayerQuantConfig(
            quant_type=QuantType.per_Token, quant_dtype=FP8
        )
        qcfg.exclude_layers = ["lm_head"]

        result = qcfg.get_layer_quant_config("lm_head")
        assert result.quant_type == QuantType.No
        assert result.quant_dtype == BF16

    def test_online_quant_resolution_uses_online_specs(self):
        qcfg = QuantizationConfig(
            FakeHFConfig(torch_dtype=BF16),
            online_quant_config={
                "global_quant_config": "ptpc_fp8",
                "layer_quant_config": {"*expert*": "mxfp4"},
                "exclude_layer": ["lm_head", "*.gate.*"],
            },
        )

        expert = qcfg.get_layer_quant_config(
            "model.layers.0.mlp.experts.0.w13_weight",
            use_online_quant=True,
        )
        assert expert.quant_type == QuantType.per_1x32
        assert expert.quant_dtype == FP4X2

        attention = qcfg.get_layer_quant_config(
            "model.layers.0.self_attn.q_proj",
            use_online_quant=True,
        )
        assert attention.quant_type == QuantType.per_Token
        assert attention.quant_dtype == FP8

        excluded = qcfg.get_layer_quant_config("lm_head", use_online_quant=True)
        assert excluded.quant_type == QuantType.No
        assert excluded.quant_dtype == BF16


# =========================================================================
# Tests — Exclude layer matching
# =========================================================================


class TestExcludeMatching:
    def _make(self, exclude_layers):
        qcfg = QuantizationConfig(config=None)
        qcfg.exclude_layers = exclude_layers
        return qcfg

    def test_empty_exclude(self):
        qcfg = self._make([])
        assert not qcfg._is_excluded("any_layer")

    def test_none_layer_name(self):
        qcfg = self._make(["lm_head"])
        assert not qcfg._is_excluded(None)

    def test_exact_match(self):
        qcfg = self._make(["lm_head"])
        assert qcfg._is_excluded("lm_head")

    def test_regex_match(self):
        qcfg = self._make(["re:model\\.layers\\..*shared_expert.*"])
        assert qcfg._is_excluded("model.layers.3.shared_expert.gate_proj")

    def test_no_match(self):
        qcfg = self._make(["lm_head"])
        assert not qcfg._is_excluded("self_attn.q_proj")

    def test_check_children_matches_child_entries(self):
        """check_children=True: parent excluded when child entries exist."""
        qcfg = self._make(
            [
                "mtp.layers.60.mlp.experts.0.gate_up_proj",
                "mtp.layers.60.mlp.experts.0.down_proj",
            ]
        )
        # Without check_children: module-level prefix does NOT match
        assert not qcfg._is_excluded("mtp.layers.60.mlp.experts")
        # With check_children: child entries trigger a match
        assert qcfg._is_excluded("mtp.layers.60.mlp.experts", check_children=True)

    def test_check_children_no_false_positive_on_siblings(self):
        """check_children must not match sibling modules."""
        qcfg = self._make(["mtp.layers.60.mlp.gate"])
        # "mlp.gate" is a sibling of "mlp.experts", not a child
        assert not qcfg._is_excluded("mtp.layers.60.mlp.experts", check_children=True)

    def test_check_children_propagates_through_get_layer_quant_config(self):
        qcfg = QuantizationConfig(config=None)
        qcfg.torch_dtype = BF16
        qcfg.global_spec = LayerQuantConfig(
            quant_type=QuantType.per_Token, quant_dtype=FP8
        )
        qcfg.exclude_layers = ["mtp.layers.60.mlp.experts.0.gate_up_proj"]

        # Without check_children: returns global FP8
        result = qcfg.get_layer_quant_config("mtp.layers.60.mlp.experts")
        assert result.quant_dtype == FP8

        # With check_children: returns BF16 (excluded)
        result = qcfg.get_layer_quant_config(
            "mtp.layers.60.mlp.experts", check_children=True
        )
        assert result.quant_dtype == BF16


class TestMatchesExclude:
    def test_exact(self):
        assert QuantizationConfig._matches_exclude("lm_head", "lm_head") is True
        assert QuantizationConfig._matches_exclude("lm_head", "other") is False

    def test_regex(self):
        assert (
            QuantizationConfig._matches_exclude(
                "model.layers.5.self_attn.q_proj",
                "re:model\\.layers\\..*self_attn.*",
            )
            is True
        )
        assert (
            QuantizationConfig._matches_exclude(
                "model.layers.5.mlp.gate_proj",
                "re:model\\.layers\\..*self_attn.*",
            )
            is False
        )

    def test_contains_mode(self):
        assert (
            QuantizationConfig._matches_exclude(
                "self_attn",
                "model.layers.0.self_attn.q_a_proj",
                check_contains=True,
            )
            is True
        )
        assert (
            QuantizationConfig._matches_exclude(
                "mlp", "self_attn.q_proj", check_contains=True
            )
            is False
        )


# =========================================================================
# Tests — remap_layer_name
# =========================================================================


class TestRemapLayerName:
    @staticmethod
    def _pattern_dict(qcfg):
        """Helper: return pattern->spec dict from layer_pattern_specs."""
        return dict(qcfg.layer_pattern_specs)

    def test_deepseek_v3_with_q_lora_rank(self):
        """Individual proj names -> fused names for deepseek_v3."""
        qcfg = QuantizationConfig(config=None)
        qcfg.layer_pattern_specs = [
            ("*.q_a_proj", LayerQuantConfig(quant_type=QuantType.per_Token)),
            ("*.gate_proj", LayerQuantConfig(quant_type=QuantType.per_1x32)),
        ]
        qcfg.exclude_layers = ["model.layers.0.q_a_proj"]

        hf = FakeHFConfig(model_type="deepseek_v3", q_lora_rank=512)
        qcfg.remap_layer_name(hf)

        pats = self._pattern_dict(qcfg)
        assert "*.fused_qkv_a_proj" in pats
        assert "*.gate_up_proj" in pats
        assert "*.q_a_proj" not in pats
        assert "model.layers.0.fused_qkv_a_proj" in qcfg.exclude_layers

    def test_qwen3_moe_splits_fused(self):
        """Fused gate_up_proj -> [gate_proj, up_proj] for qwen3_moe."""
        qcfg = QuantizationConfig(config=None)
        qcfg.layer_pattern_specs = [
            ("*.gate_up_proj", LayerQuantConfig(quant_type=QuantType.per_Token)),
        ]
        qcfg.exclude_layers = []

        hf = FakeHFConfig(model_type="qwen3_moe", mlp_only_layers=[1])
        qcfg.remap_layer_name(hf, packed_modules_mapping={})

        pats = self._pattern_dict(qcfg)
        assert "*.gate_proj" in pats
        assert "*.up_proj" in pats
        assert "*.gate_up_proj" not in pats

    def test_exclude_layers_deduplication(self):
        """gate_proj and up_proj both map to gate_up_proj -- only one remains."""
        qcfg = QuantizationConfig(config=None)
        qcfg.layer_pattern_specs = []
        qcfg.exclude_layers = [
            "model.layers.0.gate_proj",
            "model.layers.0.up_proj",
        ]

        hf = FakeHFConfig(model_type="deepseek_v3", q_lora_rank=512)
        qcfg.remap_layer_name(hf)

        assert qcfg.exclude_layers.count("model.layers.0.gate_up_proj") == 1

    def test_packed_conflict_validation_is_modelopt_mixed_only(self):
        qcfg = QuantizationConfig(config=None)
        qcfg.quant_method = "quark"
        qcfg.layer_pattern_specs = [
            (
                "model.layers.0.mlp.gate_proj",
                LayerQuantConfig(quant_type=QuantType.per_Token),
            ),
            (
                "model.layers.0.mlp.up_proj",
                LayerQuantConfig(quant_type=QuantType.per_1x32),
            ),
        ]

        qcfg.remap_layer_name(
            FakeHFConfig(model_type="minimax_m3"),
            packed_modules_mapping={
                ".gate_proj": (".gate_up_proj", 0),
                ".up_proj": (".gate_up_proj", 1),
            },
        )

        assert [pattern for pattern, _ in qcfg.layer_pattern_specs] == [
            "model.layers.0.mlp.gate_up_proj",
            "model.layers.0.mlp.gate_up_proj",
        ]

    @staticmethod
    def _modelopt_packed_config(gate_spec, up_spec):
        qcfg = QuantizationConfig(config=None)
        qcfg.quant_method = "modelopt"
        qcfg.layer_pattern_specs = [
            ("model.layers.0.mlp.gate_proj", gate_spec),
            ("model.layers.0.mlp.up_proj", up_spec),
        ]
        return qcfg

    @staticmethod
    def _remap_gate_up(qcfg):
        qcfg.remap_layer_name(
            FakeHFConfig(model_type="minimax_m3"),
            packed_modules_mapping={
                ".gate_proj": (".gate_up_proj", 0),
                ".up_proj": (".gate_up_proj", 1),
            },
        )

    def test_modelopt_packed_spec_conflict_is_rejected(self):
        """gate_proj NVFP4 + up_proj MXFP8 would build one spec for both."""
        qcfg = self._modelopt_packed_config(
            LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=NVFP4_DTYPE),
            LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=FP8),
        )

        with pytest.raises(ValueError, match="Conflicting quantization specs"):
            self._remap_gate_up(qcfg)

    def test_modelopt_partial_packed_exclusion_is_rejected(self):
        spec = LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=NVFP4_DTYPE)
        qcfg = self._modelopt_packed_config(spec, spec)
        qcfg.exclude_layers = ["model.layers.0.mlp.up_proj"]

        with pytest.raises(ValueError, match="excluded by another"):
            self._remap_gate_up(qcfg)

    def test_modelopt_agreeing_packed_sources_are_accepted(self):
        spec = LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=NVFP4_DTYPE)
        qcfg = self._modelopt_packed_config(spec, spec)

        self._remap_gate_up(qcfg)

        assert (
            qcfg.get_layer_quant_config("model.layers.0.mlp.gate_up_proj").quant_dtype
            == NVFP4_DTYPE
        )

    def test_glm_moe_dsa_remaps_like_deepseek_v3(self):
        """GLM-5 (glm_moe_dsa) uses same packed fusing as deepseek_v3."""
        qcfg = QuantizationConfig(config=None)
        qcfg.layer_pattern_specs = []
        qcfg.exclude_layers = [
            "model.layers.0.self_attn.q_a_proj",
            "model.layers.0.self_attn.kv_a_proj_with_mqa",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
        ]

        hf = FakeHFConfig(model_type="glm_moe_dsa", q_lora_rank=2048)
        qcfg.remap_layer_name(hf)

        assert "model.layers.0.self_attn.fused_qkv_a_proj" in qcfg.exclude_layers
        assert "model.layers.0.mlp.gate_up_proj" in qcfg.exclude_layers
        assert "model.layers.0.self_attn.q_a_proj" not in qcfg.exclude_layers
        assert "model.layers.0.mlp.gate_proj" not in qcfg.exclude_layers


class TestComputeHash:
    def test_hash_is_deterministic(self):
        qcfg = QuantizationConfig(config=None)
        h1 = qcfg.compute_hash()
        h2 = qcfg.compute_hash()
        assert h1 == h2
        assert isinstance(h1, str) and len(h1) == 64

    def test_different_configs_produce_different_hashes(self):
        qcfg1 = QuantizationConfig(config=None)
        qcfg2 = QuantizationConfig(config=None)
        qcfg2.global_spec = LayerQuantConfig(
            quant_type=QuantType.per_Token, quant_dtype=FP8
        )
        assert qcfg1.compute_hash() != qcfg2.compute_hash()

    def test_exclude_layers_affect_hash(self):
        qcfg1 = QuantizationConfig(config=None)
        qcfg2 = QuantizationConfig(config=None)
        qcfg2.exclude_layers = ["lm_head"]
        assert qcfg1.compute_hash() != qcfg2.compute_hash()

    def test_layer_pattern_specs_affect_hash(self):
        qcfg1 = QuantizationConfig(config=None)
        qcfg2 = QuantizationConfig(config=None)
        qcfg2.layer_pattern_specs = [
            ("*.mlp.*", LayerQuantConfig(quant_type=QuantType.per_1x32)),
        ]
        assert qcfg1.compute_hash() != qcfg2.compute_hash()


# =========================================================================
# Tests — Convenience properties
# =========================================================================


class TestConvenienceProperties:
    def test_quant_type_property(self):
        hf = FakeHFConfig(
            torch_dtype=BF16,
            quantization_config={
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"qscheme": "per_channel", "dtype": "fp8_e4m3"},
                    "input_tensors": {"is_dynamic": True},
                },
            },
        )
        qcfg = QuantizationConfig(hf)
        assert qcfg.quant_type == QuantType.per_Token
        assert qcfg.quant_dtype == FP8
        assert qcfg.is_dynamic is True


def test_get_hf_config_restores_qwen3_next_full_attention_interval(monkeypatch):
    hf = FakeHFConfig(model_type="qwen3_next")
    config_dict = {
        "model_type": "qwen3_next",
        "full_attention_interval": 4,
    }
    config_class = MagicMock()
    config_class.from_pretrained.return_value = hf
    monkeypatch.setattr(
        _m.PretrainedConfig,
        "get_config_dict",
        staticmethod(lambda _model: (config_dict, {})),
    )
    monkeypatch.setattr(
        _m.AutoConfig,
        "for_model",
        MagicMock(return_value=config_class),
    )

    result = _m.get_hf_config("Qwen/Qwen3-Next-80B-A3B-Thinking")

    assert result.full_attention_interval == 4


# =========================================================================
# Tests — will_online_requant
# =========================================================================


# MiniMaxAI/MiniMax-M3-MXFP8's own `quantization_config`, and the online config
# the nightly benchmark launches it with. A layer's format has to be decided
# from the pair: the model-wide `online_quant` flag is true for every layer
# here, including the ones the online exclude list leaves on their checkpoint
# weights.
M3_MXFP8_SOURCE = {
    "quant_method": "mxfp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [1, 32],
    "ignored_layers": ["lm_head", "model.embed_tokens", "vision_tower"],
}
M3_MXFP8_ONLINE = {
    "global_quant_config": "ptpc_fp8",
    "exclude_layer": [
        "lm_head",
        "model.embed_tokens",
        "vision_tower",
        "multi_modal_projector",
        "patch_merge_mlp",
        "*block_sparse_moe",
    ],
}
ATTENTION = "language_model.model.layers.3.self_attn.qkv_proj"
SHARED_EXPERT = (
    "language_model.model.layers.3.block_sparse_moe.shared_experts.gate_up_proj"
)


class TestWillOnlineRequant:
    def _m3_config(self):
        return QuantizationConfig(
            FakeHFConfig(torch_dtype=BF16, quantization_config=M3_MXFP8_SOURCE),
            online_quant_config=M3_MXFP8_ONLINE,
        )

    def test_none_config_never_requantizes(self):
        assert will_online_requant(None, ATTENTION, QuantType.per_1x32, FP8) is False

    def test_offline_only_config_never_requantizes(self):
        qcfg = QuantizationConfig(
            FakeHFConfig(torch_dtype=BF16, quantization_config=M3_MXFP8_SOURCE)
        )
        assert qcfg.online_quant is False
        assert will_online_requant(qcfg, ATTENTION, QuantType.per_1x32, FP8) is False

    def test_attention_linear_is_requantized(self):
        qcfg = self._m3_config()
        assert qcfg.online_quant is True
        assert will_online_requant(qcfg, ATTENTION, QuantType.per_1x32, FP8) is True

    def test_online_excluded_layer_keeps_its_checkpoint_weight(self):
        # `*block_sparse_moe` is excluded online while the checkpoint quantizes
        # everything under it but the router gate, so the model-wide flag and
        # this layer's answer disagree.
        qcfg = self._m3_config()
        assert (
            will_online_requant(qcfg, SHARED_EXPERT, QuantType.per_1x32, FP8) is False
        )

    def test_source_already_at_online_target_is_not_requantized(self):
        qcfg = self._m3_config()
        assert will_online_requant(qcfg, ATTENTION, QuantType.per_Token, FP8) is False
