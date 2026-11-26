# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

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
        module_name = "atomrl.lumen.patches.builders.megatron_model"
        if module_name in sys.modules:
            return importlib.reload(sys.modules[module_name])
        return importlib.import_module(module_name)

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

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(), reason="ROCm GPU required"
    )
    def test_fused_swiglu_training_uses_updated_weights(self, monkeypatch):
        import torch
        import torch.nn.functional as F
        from megatron.core.transformer import mlp as megatron_mlp

        megatron_model_builders = self._load_megatron_model_builders()

        class FakeMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    gated_linear_unit=True,
                    add_bias_linear=False,
                    tensor_model_parallel_size=1,
                    activation_func=F.silu,
                    activation_func_clamp_value=None,
                    glu_linear_offset=0.0,
                )
                self.original_calls = 0
                self.linear_fc1 = torch.nn.Linear(64, 256, bias=False)
                self.linear_fc2 = torch.nn.Linear(128, 64, bias=False)

            def forward(self, hidden_states, per_token_scale=None):
                self.original_calls += 1
                gate, up = F.linear(hidden_states, self.linear_fc1.weight).chunk(
                    2, dim=-1
                )
                return F.linear(F.silu(gate) * up, self.linear_fc2.weight), None

        monkeypatch.setattr(megatron_mlp, "MLP", FakeMLP)
        mlp = FakeMLP().to(device="cuda", dtype=torch.bfloat16)
        model = torch.nn.Sequential(mlp)
        megatron_model_builders.patch_fused_swiglu_mlp(model)

        optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        output, _ = mlp(x)
        output.float().square().mean().backward()
        assert all(parameter.grad is not None for parameter in mlp.parameters())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        output_after_step, _ = mlp(x.detach())
        gate, up = F.linear(x.detach(), mlp.linear_fc1.weight).chunk(2, dim=-1)
        expected = F.linear(F.silu(gate) * up, mlp.linear_fc2.weight)
        torch.testing.assert_close(output_after_step, expected, atol=0.2, rtol=0.1)
        assert mlp.original_calls == 0

        mlp(x.detach(), per_token_scale=torch.ones(4, device="cuda"))
        assert mlp.original_calls == 1

    def test_fused_swiglu_does_not_bypass_tensor_parallel_collective(self, monkeypatch):
        import torch
        from megatron.core.transformer import mlp as megatron_mlp

        megatron_model_builders = self._load_megatron_model_builders()

        class FakeMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    gated_linear_unit=True,
                    add_bias_linear=False,
                    tensor_model_parallel_size=2,
                    activation_func=torch.nn.functional.silu,
                    activation_func_clamp_value=None,
                    glu_linear_offset=0.0,
                )
                self.linear_fc1 = torch.nn.Linear(8, 16, bias=False)
                self.linear_fc2 = torch.nn.Linear(8, 8, bias=False)

            def forward(self, hidden_states, per_token_scale=None):
                return hidden_states, None

        monkeypatch.setattr(megatron_mlp, "MLP", FakeMLP)
        mlp = FakeMLP()
        original_forward = mlp.forward.__func__
        megatron_model_builders.patch_fused_swiglu_mlp(torch.nn.Sequential(mlp))
        assert mlp.forward.__func__ is original_forward
