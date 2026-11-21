"""Tests for CONFIG_BUILD / ARGS builder patch registrations."""

import importlib
import sys
import types
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import pytest

if "atomrl.lumen" not in sys.modules:
    _lumen = types.ModuleType("atomrl.lumen")
    _lumen.__path__ = [
        str(Path(__file__).resolve().parents[2].parent / "atomrl" / "lumen")
    ]
    sys.modules["atomrl.lumen"] = _lumen

from atomrl.lumen.patches import PatchPhase, PatchRegistry, list_patches
from atomrl.lumen.patches.builders import apply_args_patches, apply_config_build
from atomrl.lumen.patches.builders import dsv4 as dsv4_builders
from atomrl.lumen.patches.builders import llama as llama_builders
from atomrl.lumen.patches.builders import megatron_args as megatron_args_builders


class TestBuilderPatches:
    def setup_method(self):
        PatchRegistry.clear()
        importlib.reload(megatron_args_builders)
        importlib.reload(dsv4_builders)
        importlib.reload(llama_builders)

    def _load_megatron_model_builders(self):
        pytest.importorskip("torch")
        import importlib

        import atomrl.lumen.patches.builders.megatron_model as megatron_model_builders

        importlib.reload(megatron_model_builders)
        return megatron_model_builders

    def test_dsv4_builder_patches_registered(self):
        names = {
            spec.name
            for spec in list_patches(PatchPhase.CONFIG_BUILD, tags={"dsv4", "builder"})
        }
        assert names == {"dsv4_config_core", "dsv4_config_pipeline"}

    def test_apply_dsv4_config_core(self):
        config = SimpleNamespace()
        args = SimpleNamespace(vocab_size=128000, pipeline_model_parallel_size=1)
        apply_config_build(config, args, tags={"dsv4", "builder"})
        assert config.dsv4_mode is True
        assert config.vocab_size == 128000

    def test_apply_lumen_gpt_config(self):
        config = SimpleNamespace()
        args = SimpleNamespace(lumen_fp8_activation_store=True)
        apply_config_build(config, args, tags={"lumen", "builder"})
        assert config.persist_layer_norm is False
        assert config.bias_swiglu_fusion is False
        assert config.activation_func_fp8_input_store is True

    def test_dsv4_pretrain_args_registered(self):
        names = {spec.name for spec in list_patches(PatchPhase.ARGS, tags={"dsv4"})}
        assert "dsv4_pretrain_args" in names

    def test_dsv4_pretrain_args_adds_flags(self):
        parser = ArgumentParser()
        apply_args_patches(parser, tags={"dsv4"})
        args = parser.parse_args(["--dsv4-dsa-topk-backend", "flashinfer"])
        assert args.dsv4_dsa_topk_backend == "flashinfer"

    def test_common_megatron_args_registered(self):
        names = {
            spec.name
            for spec in list_patches(PatchPhase.ARGS, tags={"megatron", "lumen"})
        }
        assert "common_megatron_args" in names

    def test_common_megatron_args_adds_backend_flag(self):
        parser = ArgumentParser()
        apply_args_patches(parser, names={"common_megatron_args"})
        args = parser.parse_args([])
        assert args.backend == "megatron"

    def test_common_megatron_args_has_dedicated_fp4_switch(self):
        parser = ArgumentParser()
        apply_args_patches(parser, names={"common_megatron_args"})
        args = parser.parse_args(["--linear-fp4", "--grad-quant-type", "mxfp4"])
        assert args.linear_fp4 is True
        assert args.linear_fp8 is False
        assert args.grad_quant_type == "mxfp4"
        assert args.linear_fp8_block_size == 128

    def test_common_megatron_args_rejects_mxfp4_as_fp8_format(self):
        parser = ArgumentParser()
        apply_args_patches(parser, names={"common_megatron_args"})
        with pytest.raises(SystemExit):
            parser.parse_args(["--linear-fp8-format", "mxfp4"])

    def test_llama_pretrain_args_registered(self):
        names = {
            spec.name
            for spec in list_patches(PatchPhase.ARGS, tags={"llama", "pretrain"})
        }
        assert "llama_pretrain_args" in names

    def test_llama_pretrain_args_adds_mlperf_flags(self):
        parser = ArgumentParser()
        apply_args_patches(
            parser, names={"common_megatron_args", "llama_pretrain_args"}
        )
        args = parser.parse_args(["--size", "8b"])
        assert args.size == "8b"

    def test_gpt_model_build_patches_registered(self):
        self._load_megatron_model_builders()
        names = {
            spec.name
            for spec in list_patches(
                PatchPhase.MODEL_BUILD, tags={"megatron", "builder"}
            )
        }
        assert names >= {
            "core_attention_spec",
            "norms_in_spec",
            "mla_attention_spec",
            "model_norms",
            "fused_swiglu_mlp",
        }

    def test_core_attention_spec_patches_module_spec(self):
        megatron_model_builders = self._load_megatron_model_builders()
        from types import SimpleNamespace

        from atomrl.lumen.modules.attention_megatron import LumenDotProductAttention
        from megatron.core.transformer.spec_utils import ModuleSpec

        sa_subs = SimpleNamespace(core_attention=object())
        sa = SimpleNamespace(submodules=sa_subs)
        spec = SimpleNamespace(
            submodules=SimpleNamespace(self_attention=sa, layer_specs=None)
        )

        megatron_model_builders.patch_core_attention(spec)

        assert sa_subs.core_attention == ModuleSpec(module=LumenDotProductAttention)
