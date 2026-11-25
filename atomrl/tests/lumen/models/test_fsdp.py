# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.models.fsdp — FSDP shared training helpers.

Covers:
  - _rank0_print: logging with dist guard
  - add_common_fsdp_args: all argument groups and defaults
  - patch_norms: RMSNorm + LayerNorm replacement, weight copy, grad_quant
  - apply_fp8_training: QuantConfig construction, quant.enable dispatch
  - reset_fp8_state: unwrap + reset on fp8 layers
  - apply_lora: LoRA adapter application via peft
"""

import argparse
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
from atomrl.lumen.models.fsdp import (
    _rank0_print,
    add_common_fsdp_args,
    apply_fp8_training,
    apply_lora,
    patch_norms,
    reset_fp8_state,
)
from conftest import compute_snr as _compute_snr
from conftest import layernorm_ref as _layernorm_golden
from conftest import rmsnorm_ref as _rmsnorm_golden

# ===================================================================
# _rank0_print
# ===================================================================


class TestRank0Print:
    def test_logs_when_dist_not_initialized(self):
        with mock.patch("atomrl.lumen.models.fsdp.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            with mock.patch("atomrl.lumen.models.fsdp.logger") as mock_logger:
                _rank0_print("test message")
                mock_logger.info.assert_called_once_with("test message")

    def test_logs_on_rank0(self):
        with mock.patch("atomrl.lumen.models.fsdp.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_rank.return_value = 0
            with mock.patch("atomrl.lumen.models.fsdp.logger") as mock_logger:
                _rank0_print("rank0 msg")
                mock_logger.info.assert_called_once()

    def test_silent_on_non_rank0(self):
        with mock.patch("atomrl.lumen.models.fsdp.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_rank.return_value = 3
            with mock.patch("atomrl.lumen.models.fsdp.logger") as mock_logger:
                _rank0_print("should be suppressed")
                mock_logger.info.assert_not_called()


# ===================================================================
# add_common_fsdp_args
# ===================================================================


class TestAddCommonFsdpArgs:
    def _parse(self, cli_args=None):
        parser = argparse.ArgumentParser()
        add_common_fsdp_args(parser)
        return parser.parse_args(cli_args or [])

    def test_backend_default(self):
        args = self._parse()
        assert args.backend == "fsdp"

    def test_backend_megatron(self):
        args = self._parse(["--backend", "megatron"])
        assert args.backend == "megatron"

    def test_training_defaults(self):
        args = self._parse()
        assert args.micro_batch_size == 1
        assert args.gradient_accumulation_steps == 8
        assert args.max_steps == 800
        assert args.lr == 4e-4
        assert args.min_lr == 0.0
        assert args.weight_decay == 0.01
        assert args.max_grad_norm == 1.0
        assert args.log_interval == 10
        assert args.save_interval == 0
        assert args.save_dir == "./checkpoints"
        assert args.num_workers == 4

    def test_data_defaults(self):
        args = self._parse()
        assert args.train_data_path is None
        assert args.val_data_path is None
        assert args.train_samples == 10000
        assert args.val_samples == 500

    def test_fsdp_defaults(self):
        args = self._parse()
        assert args.sharding_strategy == "full_shard"

    def test_sharding_strategy_choices(self):
        for strat in ["full_shard", "shard_grad_op", "no_shard"]:
            args = self._parse(["--sharding-strategy", strat])
            assert args.sharding_strategy == strat

    def test_lora_defaults(self):
        args = self._parse()
        assert args.lora_rank == 0
        assert args.lora_alpha == 32.0
        assert args.lora_dropout == 0.1

    def test_linear_fp8_defaults(self):
        args = self._parse()
        assert args.linear_fp8 is False
        assert args.linear_fp4 is False
        assert args.linear_fp8_format == "fp8_e4m3"
        assert args.linear_fp8_scaling == "delayed"
        assert args.linear_fp8_block_size == 128
        assert args.linear_fp8_amax_algo == "max"
        assert args.linear_fp8_reduce_amax is False
        assert args.linear_fp8_amax_history == 16
        assert args.linear_fp8_margin == 0
        assert args.linear_fp8_activation is True
        assert args.linear_fp8_wgrad is True
        assert args.grad_quant_type is None

    def test_fp8_format_choices(self):
        for fmt in ["fp8_e4m3", "fp8_e5m2", "hybrid", "mxfp8"]:
            args = self._parse(["--linear-fp8-format", fmt])
            assert args.linear_fp8_format == fmt

    def test_no_linear_fp8_activation(self):
        args = self._parse(["--no-linear-fp8-activation"])
        assert args.linear_fp8_activation is False

    def test_no_linear_fp8_wgrad(self):
        args = self._parse(["--no-linear-fp8-wgrad"])
        assert args.linear_fp8_wgrad is False

    def test_linear_fp4_switch(self):
        args = self._parse(["--linear-fp4"])
        assert args.linear_fp4 is True
        assert args.linear_fp8 is False

    def test_linear_fp4_from_args_selects_mxfp4_recipe(self):
        from atomrl.lumen.config import LumenConfig

        args = self._parse(["--linear-fp4"])
        cfg = LumenConfig.from_args(args)
        assert cfg.format == "mxfp4"
        assert cfg.block_size == 32
        assert cfg.quant_config.is_quantized

    def test_fsdp_mxfp4_comm_flag(self):
        args = self._parse(["--fsdp-mxfp4-comm"])
        assert args.fsdp_mxfp4_comm is True

    def test_grad_quant_type_choices(self):
        for gq in ["fp8", "mxfp8", "mxfp4", "fp4"]:
            args = self._parse(["--grad-quant-type", gq])
            assert args.grad_quant_type == gq

    def test_first_last_layers_bf16(self):
        args = self._parse(
            [
                "--first-last-layers-bf16",
                "--num-layers-at-start-in-bf16",
                "2",
                "--num-layers-at-end-in-bf16",
                "3",
            ]
        )
        assert args.first_last_layers_bf16 is True
        assert args.num_layers_at_start_in_bf16 == 2
        assert args.num_layers_at_end_in_bf16 == 3

    def test_norm_defaults(self):
        args = self._parse()
        assert args.lumen_norm is False

    def test_warmup_defaults(self):
        args = self._parse()
        assert args.warmup_steps == 0
        assert args.val_loss_target is None

    def test_returns_parser(self):
        parser = argparse.ArgumentParser()
        result = add_common_fsdp_args(parser)
        assert result is parser

    def test_attn_fp8_defaults(self):
        args = self._parse()
        assert args.lumen_attn_backend == "auto"
        assert args.lumen_fp8_attn == "none"
        assert args.lumen_fp8_quant_type == "blockwise"

    def test_lumen_fp8_attn_choices(self):
        for scope in ["none", "dpa", "mha"]:
            args = self._parse(["--lumen-fp8-attn", scope])
            assert args.lumen_fp8_attn == scope

    def test_lumen_fp8_quant_type_blockwise2d(self):
        args = self._parse(["--lumen-fp8-quant-type", "blockwise2d"])
        assert args.lumen_fp8_quant_type == "blockwise2d"

    def test_lumen_fp8_quant_type_all_choices(self):
        for qt in [
            "dynamic",
            "delayed",
            "blockwise",
            "blockwise2d",
            "per_token",
            "none",
            "mxfp8",
        ]:
            args = self._parse(["--lumen-fp8-quant-type", qt])
            assert args.lumen_fp8_quant_type == qt

    def test_attn_backend_choices(self):
        for backend in ["auto", "triton", "csrc", "asm"]:
            args = self._parse(["--lumen-attn-backend", backend])
            assert args.lumen_attn_backend == backend


# ===================================================================
# patch_norms
# ===================================================================


class _FakeRMSNorm(nn.Module):
    __qualname__ = "RMSNorm"

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return x


_FakeRMSNorm.__name__ = "RMSNorm"


class _FakeLayerNorm(nn.Module):
    __qualname__ = "LayerNorm"

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        self.normalized_shape = (hidden_size,)

    def forward(self, x):
        return x


_FakeLayerNorm.__name__ = "LayerNorm"


class _FakeLlamaRMSNorm(nn.Module):
    __qualname__ = "LlamaRMSNorm"

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        return x


_FakeLlamaRMSNorm.__name__ = "LlamaRMSNorm"


class TestPatchNorms:
    def test_skips_when_lumen_norm_false(self):
        model = nn.Sequential(nn.Linear(8, 8))
        args = SimpleNamespace(lumen_norm=False)
        patch_norms(model, args)

    def test_skips_when_attr_missing(self):
        model = nn.Sequential(nn.Linear(8, 8))
        args = SimpleNamespace()
        patch_norms(model, args)

    def test_replaces_rmsnorm(self):
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        model = nn.Module()
        model.norm = _FakeRMSNorm(64)
        args = SimpleNamespace(lumen_norm=True, grad_quant_type=None)
        patch_norms(model, args)
        assert isinstance(model.norm, LumenRMSNorm)
        assert model.norm.weight.shape[0] == 64

    def test_replaces_layernorm(self):
        from atomrl.lumen.ops.normalization import LumenLayerNorm

        model = nn.Module()
        model.norm = _FakeLayerNorm(32)
        args = SimpleNamespace(lumen_norm=True, grad_quant_type=None)
        patch_norms(model, args)
        assert isinstance(model.norm, LumenLayerNorm)
        assert model.norm.weight.shape[0] == 32

    def test_replaces_llama_rmsnorm(self):
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        model = nn.Module()
        model.ln = _FakeLlamaRMSNorm(128)
        args = SimpleNamespace(lumen_norm=True, grad_quant_type=None)
        patch_norms(model, args)
        assert isinstance(model.ln, LumenRMSNorm)

    def test_copies_weight_data(self):
        model = nn.Module()
        orig_norm = _FakeRMSNorm(16)
        orig_norm.weight.data.fill_(3.14)
        model.norm = orig_norm
        args = SimpleNamespace(lumen_norm=True, grad_quant_type=None)
        patch_norms(model, args)
        assert torch.allclose(model.norm.weight.data, torch.full((16,), 3.14))

    def test_nested_replacement(self):
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        inner = nn.Module()
        inner.norm = _FakeRMSNorm(32)
        model = nn.Module()
        model.layer = inner
        args = SimpleNamespace(lumen_norm=True, grad_quant_type=None)
        patch_norms(model, args)
        assert isinstance(model.layer.norm, LumenRMSNorm)

    def test_multiple_norms(self):
        from atomrl.lumen.ops.normalization import LumenLayerNorm, LumenRMSNorm

        model = nn.Module()
        model.rms = _FakeRMSNorm(64)
        model.ln = _FakeLayerNorm(64)
        args = SimpleNamespace(lumen_norm=True, grad_quant_type=None)
        patch_norms(model, args)
        assert isinstance(model.rms, LumenRMSNorm)
        assert isinstance(model.ln, LumenLayerNorm)

    def test_grad_quant_type_forwarded(self):
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        model = nn.Module()
        model.norm = _FakeRMSNorm(64)
        args = SimpleNamespace(lumen_norm=True, grad_quant_type="fp8")
        patch_norms(model, args)
        assert isinstance(model.norm, LumenRMSNorm)
        assert model.norm.grad_quant_type == "fp8"

    def test_replaces_mistral_rmsnorm(self):
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        class _FakeMistralRMSNorm(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(h))
                self.variance_epsilon = 1e-6

            def forward(self, x):
                return x

        _FakeMistralRMSNorm.__name__ = "MistralRMSNorm"

        model = nn.Module()
        model.norm = _FakeMistralRMSNorm(32)
        args = SimpleNamespace(lumen_norm=True, grad_quant_type=None)
        patch_norms(model, args)
        assert isinstance(model.norm, LumenRMSNorm)

    def test_replaces_qwen2_rmsnorm(self):
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        class _FakeQwen2RMSNorm(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(h))
                self.eps = 1e-6

            def forward(self, x):
                return x

        _FakeQwen2RMSNorm.__name__ = "Qwen2RMSNorm"

        model = nn.Module()
        model.norm = _FakeQwen2RMSNorm(32)
        args = SimpleNamespace(lumen_norm=True, grad_quant_type=None)
        patch_norms(model, args)
        assert isinstance(model.norm, LumenRMSNorm)

    def test_layernorm_bias_copied(self):
        model = nn.Module()
        norm = _FakeLayerNorm(16)
        norm.weight.data.fill_(2.0)
        norm.bias.data.fill_(0.7)
        model.norm = norm
        args = SimpleNamespace(lumen_norm=True, grad_quant_type=None)
        patch_norms(model, args)
        assert torch.allclose(model.norm.weight.data, torch.full((16,), 2.0))
        if hasattr(model.norm, "bias") and model.norm.bias is not None:
            assert torch.allclose(model.norm.bias.data, torch.full((16,), 0.7))


# ===================================================================
# Patched norms — golden output verification
# ===================================================================


class TestPatchedNormGoldenOutput:
    """After patch_norms, the replaced norm should produce numerically correct output."""

    def test_patched_rmsnorm_matches_golden(self):
        hidden = 128
        model = nn.Module()
        model.norm = _FakeRMSNorm(hidden)
        model.norm.weight.data.uniform_(0.5, 1.5)
        args = SimpleNamespace(lumen_norm=True, grad_quant_type=None)
        patch_norms(model, args)

        model.norm = model.norm.cuda()
        torch.manual_seed(0)
        x = torch.randn(4, 32, hidden, device="cuda", dtype=torch.bfloat16)
        out = model.norm(x)

        golden = _rmsnorm_golden(x, model.norm.weight.data)
        snr = _compute_snr(golden, out)
        assert snr > 30, f"Patched RMSNorm vs golden SNR: {snr:.1f} dB"

    def test_patched_layernorm_matches_golden(self):
        hidden = 128
        model = nn.Module()
        model.norm = _FakeLayerNorm(hidden)
        model.norm.weight.data.uniform_(0.5, 1.5)
        model.norm.bias.data.uniform_(-0.1, 0.1)
        args = SimpleNamespace(lumen_norm=True, grad_quant_type=None)
        patch_norms(model, args)

        model.norm = model.norm.cuda().bfloat16()
        torch.manual_seed(0)
        x = torch.randn(4, 32, hidden, device="cuda", dtype=torch.bfloat16)
        out = model.norm(x)

        bias = (
            model.norm.bias.data
            if hasattr(model.norm, "bias") and model.norm.bias is not None
            else None
        )
        golden = _layernorm_golden(x, model.norm.weight.data, bias=bias)
        snr = _compute_snr(golden, out)
        assert snr > 30, f"Patched LayerNorm vs golden SNR: {snr:.1f} dB"

    def test_patched_llama_rmsnorm_matches_golden(self):
        hidden = 256
        orig_eps = 1e-5
        model = nn.Module()
        model.norm = _FakeLlamaRMSNorm(hidden, eps=orig_eps)
        model.norm.weight.data.uniform_(0.5, 1.5)
        args = SimpleNamespace(lumen_norm=True, grad_quant_type=None)
        patch_norms(model, args)

        assert (
            model.norm.eps == orig_eps
        ), "patch_norms should extract variance_epsilon correctly"

        model.norm = model.norm.cuda()
        torch.manual_seed(2)
        x = torch.randn(2, 16, hidden, device="cuda", dtype=torch.bfloat16)
        out = model.norm(x)

        golden = _rmsnorm_golden(x, model.norm.weight.data, eps=orig_eps)
        snr = _compute_snr(golden, out)
        assert snr > 30, f"Patched LlamaRMSNorm vs golden SNR: {snr:.1f} dB"

    @pytest.mark.parametrize("hidden", [64, 256, 1024, 4096])
    def test_patched_rmsnorm_various_sizes(self, hidden):
        model = nn.Module()
        model.norm = _FakeRMSNorm(hidden)
        model.norm.weight.data.uniform_(0.5, 1.5)
        args = SimpleNamespace(lumen_norm=True, grad_quant_type=None)
        patch_norms(model, args)

        model.norm = model.norm.cuda()
        torch.manual_seed(3)
        x = torch.randn(2, 8, hidden, device="cuda", dtype=torch.bfloat16)
        out = model.norm(x)

        golden = _rmsnorm_golden(x, model.norm.weight.data)
        snr = _compute_snr(golden, out)
        assert snr > 25, f"Patched RMSNorm h={hidden} vs golden SNR: {snr:.1f} dB"

    def test_patched_rmsnorm_with_grad_quant_matches_golden(self):
        hidden = 64
        model = nn.Module()
        model.norm = _FakeRMSNorm(hidden)
        model.norm.weight.data.uniform_(0.5, 1.5)
        args = SimpleNamespace(lumen_norm=True, grad_quant_type="fp8")
        patch_norms(model, args)

        model.norm = model.norm.cuda()
        torch.manual_seed(7)
        x = torch.randn(2, 16, hidden, device="cuda", dtype=torch.bfloat16)
        out = model.norm(x)

        golden = _rmsnorm_golden(x, model.norm.weight.data)
        snr = _compute_snr(golden, out)
        assert snr > 25, f"Patched RMSNorm (grad_quant=fp8) vs golden SNR: {snr:.1f} dB"


# ===================================================================
# Norm benchmarks
# ===================================================================


class TestNormBenchmark:
    """Throughput benchmarks for patched LumenNorms."""

    @pytest.mark.parametrize("hidden", [1024, 4096, 8192])
    def test_rmsnorm_throughput(self, hidden):
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        norm = LumenRMSNorm(hidden).cuda()
        x = torch.randn(4, 512, hidden, device="cuda", dtype=torch.bfloat16)

        for _ in range(3):
            norm(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 20

        start.record()
        for _ in range(iters):
            norm(x)
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / iters
        total_bytes = 4 * 512 * hidden * 2 * 2
        bw_gb_s = (total_bytes / (avg_ms / 1000.0)) / (1024**3)
        print(f"\n[RMSNorm] hidden={hidden}: {avg_ms:.3f}ms, {bw_gb_s:.1f} GB/s")


# ===================================================================
# apply_fp8_training
# ===================================================================


class TestApplyFP8Training:
    def _make_args(self, **overrides):
        defaults = {
            "linear_fp8_format": "fp8_e4m3",
            "linear_fp8_scaling": "delayed",
            "linear_fp8_block_size": 128,
            "linear_fp8_amax_algo": "max",
            "linear_fp8_reduce_amax": False,
            "linear_fp8_amax_history": 16,
            "linear_fp8_margin": 0,
            "linear_fp8_activation": True,
            "linear_fp8_wgrad": True,
            "grad_quant_type": None,
            "first_last_layers_bf16": False,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
            "use_sdma": False,
            "lumen_norm": False,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_constructs_quant_config_and_enables(self, mock_dist):
        mock_dist.is_initialized.return_value = False
        model = nn.Sequential(nn.Linear(16, 16))
        args = self._make_args()

        with mock.patch("atomrl.lumen.quantize.enable") as mock_enable:
            apply_fp8_training(model, args)
            mock_enable.assert_called_once()
            call_kwargs = mock_enable.call_args
            config = call_kwargs[1].get("config") or call_kwargs[0][1]
            assert config.format.value == "fp8_e4m3"
            assert config.scaling.value == "delayed"

    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_mxfp8_format(self, mock_dist):
        mock_dist.is_initialized.return_value = False
        model = nn.Sequential(nn.Linear(16, 16))
        args = self._make_args(
            linear_fp8_format="mxfp8", linear_fp8_scaling="blockwise"
        )

        with mock.patch("atomrl.lumen.quantize.enable") as mock_enable:
            apply_fp8_training(model, args)
            config = mock_enable.call_args[1]["config"]
            assert config.format.value == "mxfp8"
            assert config.scaling.value == "blockwise"

    def test_reduce_amax_uses_dp_group(self, tmp_path):
        """With no explicit group, amax reduction falls back to WORLD.

        The fallback lives in ``LumenConfig.enable``, which reads
        ``torch.distributed`` directly, so patching a ``dist`` name inside
        ``atomrl.lumen.models.fsdp`` would assert nothing. ``group.WORLD`` is a
        metaclass property that ``mock.patch`` cannot restore either (it leaks
        into every later test), so this stands up a real single-rank group.
        """
        import torch.distributed as dist

        model = nn.Sequential(nn.Linear(16, 16))
        args = self._make_args(linear_fp8_reduce_amax=True)

        store = dist.FileStore(str(tmp_path / "amax_store"), 1)
        dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
        try:
            with mock.patch("atomrl.lumen.quantize.enable") as mock_enable:
                apply_fp8_training(model, args)
            config = mock_enable.call_args[1]["config"]
            assert config.reduce_amax is True
            assert mock_enable.call_args[1]["dp_group"] is dist.group.WORLD
        finally:
            dist.destroy_process_group()

    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_explicit_dp_group_overrides(self, mock_dist):
        mock_dist.is_initialized.return_value = True
        model = nn.Sequential(nn.Linear(16, 16))
        args = self._make_args(linear_fp8_reduce_amax=True)
        custom_group = mock.MagicMock()

        with mock.patch("atomrl.lumen.quantize.enable") as mock_enable:
            apply_fp8_training(model, args, dp_group=custom_group)
            assert mock_enable.call_args[1]["dp_group"] == custom_group

    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_use_sdma_forwarded(self, mock_dist):
        mock_dist.is_initialized.return_value = False
        model = nn.Sequential(nn.Linear(16, 16))
        args = self._make_args(use_sdma=True)

        with mock.patch("atomrl.lumen.quantize.enable") as mock_enable:
            apply_fp8_training(model, args)
            config = mock_enable.call_args[1]["config"]
            assert config.use_sdma is True

    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_first_last_bf16_forwarded(self, mock_dist):
        mock_dist.is_initialized.return_value = False
        model = nn.Sequential(nn.Linear(16, 16))
        args = self._make_args(
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=2,
            num_layers_at_end_in_bf16=3,
        )

        with mock.patch("atomrl.lumen.quantize.enable") as mock_enable:
            apply_fp8_training(model, args)
            config = mock_enable.call_args[1]["config"]
            assert config.first_last_layers_bf16 is True
            assert config.num_layers_at_start_in_bf16 == 2
            assert config.num_layers_at_end_in_bf16 == 3

    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_patch_norms_called(self, mock_dist):
        mock_dist.is_initialized.return_value = False
        model = nn.Module()
        model.norm = _FakeRMSNorm(32)
        args = self._make_args(lumen_norm=True)

        with mock.patch("atomrl.lumen.quantize.enable"):
            apply_fp8_training(model, args)
        from atomrl.lumen.ops.normalization import LumenRMSNorm

        assert isinstance(model.norm, LumenRMSNorm)

    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_patch_norms_called_before_quant_enable(self, mock_dist):
        """patch_norms must run before quant.enable so hooks apply to patched modules."""
        mock_dist.is_initialized.return_value = False
        model = nn.Module()
        model.norm = _FakeRMSNorm(32)
        args = self._make_args(lumen_norm=True)

        call_order = []
        from atomrl.lumen.config import LumenConfig

        orig_patch_norms = LumenConfig._patch_norms

        # apply_fp8_training delegates to LumenConfig.enable, so the ordering
        # guarantee is observable there rather than on the module-level
        # patch_norms shim this test used to wrap.
        def _tracking_patch_norms(self, *a, **kw):
            call_order.append("patch_norms")
            return orig_patch_norms(self, *a, **kw)

        def _tracking_enable(*a, **kw):
            call_order.append("quant_enable")

        with (
            mock.patch.object(LumenConfig, "_patch_norms", _tracking_patch_norms),
            mock.patch("atomrl.lumen.quantize.enable", side_effect=_tracking_enable),
        ):
            apply_fp8_training(model, args)

        assert call_order == [
            "patch_norms",
            "quant_enable",
        ], f"Expected patch_norms before quant.enable, got: {call_order}"

    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_lumen_fp8_attn_dpa_sets_config(self, mock_dist):
        mock_dist.is_initialized.return_value = False
        model = nn.Sequential(nn.Linear(16, 16))
        args = self._make_args(lumen_fp8_attn="dpa")

        with mock.patch("atomrl.lumen.quantize.enable") as mock_enable:
            apply_fp8_training(model, args)
            config = mock_enable.call_args[1]["config"]
            assert config.fp8_dpa is True
            assert config.fp8_mha is False

    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_lumen_fp8_attn_mha_sets_config(self, mock_dist):
        mock_dist.is_initialized.return_value = False
        model = nn.Sequential(nn.Linear(16, 16))
        args = self._make_args(lumen_fp8_attn="mha")

        with mock.patch("atomrl.lumen.quantize.enable") as mock_enable:
            apply_fp8_training(model, args)
            config = mock_enable.call_args[1]["config"]
            assert config.fp8_dpa is True
            assert config.fp8_mha is True


# ===================================================================
# reset_fp8_state
# ===================================================================


class TestResetFP8State:
    def test_resets_fp8_initialized(self):
        module = nn.Module()
        module.fp8_initialized = True
        model = nn.Sequential(module)

        with mock.patch("atomrl.lumen.models.fsdp._rank0_print"):
            reset_fp8_state(model)
        assert module.fp8_initialized is False

    def test_resets_quant_manager(self):
        module = nn.Module()
        module._quant_manager = mock.MagicMock()
        model = nn.Sequential(module)

        with mock.patch("atomrl.lumen.models.fsdp._rank0_print"):
            reset_fp8_state(model)
        module._quant_manager.reset.assert_called_once()

    def test_resets_tl_scaling_manager(self):
        module = nn.Module()
        module._tl_scaling_manager = mock.MagicMock()
        model = nn.Sequential(module)

        with mock.patch("atomrl.lumen.models.fsdp._rank0_print"):
            reset_fp8_state(model)
        module._tl_scaling_manager.reset.assert_called_once()

    def test_unwraps_nested_module(self):
        """reset_fp8_state unwraps .module attributes (FSDP/DDP wrapping)."""
        inner = nn.Module()
        inner.fp8_initialized = True

        wrapper = nn.Module()
        wrapper.module = inner

        outer = nn.Module()
        outer.module = wrapper

        with mock.patch("atomrl.lumen.models.fsdp._rank0_print"):
            reset_fp8_state(outer)
        assert inner.fp8_initialized is False

    def test_works_with_fsdp1_module_wrapper(self):
        """Regression: reset_fp8_state still works with FSDP1 .module wrapper."""
        inner = nn.Linear(8, 4)
        inner.fp8_initialized = True
        model = nn.Module()
        model.module = inner

        with mock.patch("atomrl.lumen.models.fsdp._rank0_print"):
            reset_fp8_state(model)

        assert not getattr(inner, "fp8_initialized", False)

    def test_handles_model_without_fp8(self):
        model = nn.Sequential(nn.Linear(4, 4))
        with mock.patch("atomrl.lumen.models.fsdp._rank0_print"):
            reset_fp8_state(model)

    def test_forward_works_after_reset(self):
        module = nn.Linear(64, 64)
        module.fp8_initialized = True
        module._quant_manager = mock.MagicMock()
        model = nn.Sequential(module)

        with mock.patch("atomrl.lumen.models.fsdp._rank0_print"):
            reset_fp8_state(model)

        x = torch.randn(2, 64)
        out = model(x)
        assert out.shape == (2, 64)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ===================================================================
# apply_lora
# ===================================================================


class TestApplyLora:
    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_apply_lora_returns_peft_model(self, mock_dist):
        mock_dist.is_initialized.return_value = False

        try:
            import peft  # noqa: F401
        except ImportError:
            pytest.skip("peft not installed")

        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 1
        config.n_head = 2
        config.n_embd = 64
        model = AutoModelForCausalLM.from_config(config)

        args = SimpleNamespace(lora_rank=4, lora_alpha=16.0, lora_dropout=0.0)
        peft_model = apply_lora(model, args)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        assert trainable < total
        assert trainable > 0

    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_lora_forward_produces_valid_output(self, mock_dist):
        mock_dist.is_initialized.return_value = False

        try:
            import peft  # noqa: F401
        except ImportError:
            pytest.skip("peft not installed")

        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 1
        config.n_head = 2
        config.n_embd = 64
        model = AutoModelForCausalLM.from_config(config)

        args = SimpleNamespace(lora_rank=4, lora_alpha=16.0, lora_dropout=0.0)
        peft_model = apply_lora(model, args)

        x = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            out = peft_model(x).logits

        assert out.shape == (1, 8, config.vocab_size)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestUseSdmaArg:
    def test_add_common_fsdp_args_has_use_sdma(self):
        import argparse

        from atomrl.lumen.models.fsdp import add_common_fsdp_args

        parser = argparse.ArgumentParser()
        add_common_fsdp_args(parser)
        args = parser.parse_args([])
        assert args.use_sdma is False

    def test_add_common_fsdp_args_use_sdma_true(self):
        import argparse

        from atomrl.lumen.models.fsdp import add_common_fsdp_args

        parser = argparse.ArgumentParser()
        add_common_fsdp_args(parser)
        args = parser.parse_args(["--use-sdma"])
        assert args.use_sdma is True


class TestWrapMxfp4Comm:
    def test_wraps_aligned_mxfp4_linear(self):
        from atomrl.lumen.models.fsdp import _wrap_params_as_mxfp4_comm
        from atomrl.lumen.quantize.comm_tensor import MXFP4CommTensor

        m = nn.Linear(64, 64, bias=False)
        m._quant_enabled = True
        m._quant_scaling_type = "mxfp4"
        n = _wrap_params_as_mxfp4_comm(m, block_size=32, world_size=1)
        assert n == 1
        assert isinstance(m.weight, MXFP4CommTensor)

    def test_skips_misaligned_and_non_mxfp4(self):
        from atomrl.lumen.models.fsdp import _wrap_params_as_mxfp4_comm

        misaligned = nn.Linear(64, 31, bias=False)
        misaligned._quant_enabled = True
        misaligned._quant_scaling_type = "mxfp4"
        assert _wrap_params_as_mxfp4_comm(misaligned, block_size=32, world_size=1) == 0

        fp8 = nn.Linear(64, 64, bias=False)
        fp8._quant_enabled = True
        fp8._quant_scaling_type = "blockwise"
        assert _wrap_params_as_mxfp4_comm(fp8, block_size=32, world_size=1) == 0

    def test_world_size_alignment_skip_names_weight(self):
        from atomrl.lumen.models.fsdp import _wrap_params_as_mxfp4_comm

        model = nn.Module()
        model.proj = nn.Linear(64, 32, bias=False)
        model.proj._quant_enabled = True
        model.proj._quant_scaling_type = "mxfp4"

        with mock.patch("atomrl.lumen.models.fsdp._rank0_print") as rank0_print:
            assert _wrap_params_as_mxfp4_comm(model, block_size=32, world_size=2) == 0

        message = rank0_print.call_args.args[0]
        assert "proj.weight(32, 64)" in message
        assert "32*world_size=64" in message


class TestValidateFsdpQuantArgs:
    def _validate(self, **overrides):
        from atomrl.lumen.models.fsdp import validate_fsdp_quant_args

        args = SimpleNamespace(
            linear_fp8=False,
            linear_fp4=True,
            fsdp_version=2,
            fsdp_mxfp4_comm=True,
            fsdp_fp8_param_storage=False,
            lumen_fp8_param_gather=False,
        )
        for name, value in overrides.items():
            setattr(args, name, value)
        validate_fsdp_quant_args(args)

    def test_valid_mxfp4_comm(self):
        self._validate()

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"linear_fp4": False}, "requires --linear-fp4"),
            ({"fsdp_version": 1}, "requires --fsdp-version 2"),
            ({"fsdp_fp8_param_storage": True}, "cannot be combined"),
            ({"lumen_fp8_param_gather": True}, "cannot be combined"),
        ],
    )
    def test_rejects_invalid_mxfp4_comm(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            self._validate(**overrides)

    def test_rejects_fp8_storage_with_plain_mxfp4(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            self._validate(fsdp_mxfp4_comm=False, fsdp_fp8_param_storage=True)

    def test_rejects_fp8_and_fp4_together(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._validate(linear_fp8=True)

    def test_missing_fsdp_version_does_not_assume_fsdp2(self):
        from atomrl.lumen.models.fsdp import validate_fsdp_quant_args

        args = SimpleNamespace(
            linear_fp8=False,
            linear_fp4=True,
            fsdp_mxfp4_comm=True,
            fsdp_fp8_param_storage=False,
            lumen_fp8_param_gather=False,
        )
        with pytest.raises(ValueError, match="requires --fsdp-version 2"):
            validate_fsdp_quant_args(args)


class TestApplyFSDP2MXFP4CommRefusesNoOp:
    """`--fsdp-mxfp4-comm` matching nothing must not look like a compressed run."""

    def _args(self):
        return SimpleNamespace(
            linear_fp8=False,
            linear_fp4=True,
            fsdp_version=2,
            sharding_strategy="full_shard",
            fsdp_mxfp4_comm=True,
            fsdp_fp8_param_storage=False,
            lumen_fp8_param_gather=False,
        )

    @mock.patch("torch.distributed.fsdp.fully_shard")
    @mock.patch("torch.distributed.device_mesh.init_device_mesh")
    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_raises_when_nothing_wrapped(self, mock_dist, _mesh, _shard):
        from atomrl.lumen.models.fsdp import apply_fsdp2

        mock_dist.get_world_size.return_value = 8
        # Unquantized model: no module carries the MXFP4 patch markers.
        model = nn.Sequential(nn.Linear(64, 64, bias=False))

        with pytest.raises(ValueError, match="wrapped 0 weights"):
            apply_fsdp2(model, self._args())

    @mock.patch("torch.distributed.fsdp.fully_shard")
    @mock.patch("torch.distributed.device_mesh.init_device_mesh")
    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_reports_alignment_rule_in_error(self, mock_dist, _mesh, _shard):
        from atomrl.lumen.models.fsdp import apply_fsdp2

        mock_dist.get_world_size.return_value = 8
        # Patched but misaligned: 32 rows cannot split across 8 ranks by 32.
        model = nn.Sequential(nn.Linear(64, 32, bias=False))
        model[0]._quant_enabled = True
        model[0]._quant_scaling_type = "mxfp4"

        with pytest.raises(ValueError, match=r"N % 256 == 0 and K % 32 == 0"):
            apply_fsdp2(model, self._args())

    @mock.patch("torch.distributed.fsdp.fully_shard")
    @mock.patch("torch.distributed.device_mesh.init_device_mesh")
    @mock.patch("atomrl.lumen.models.fsdp.dist")
    def test_accepts_aligned_mxfp4_weight(self, mock_dist, _mesh, _shard):
        from atomrl.lumen.models.fsdp import apply_fsdp2

        mock_dist.get_world_size.return_value = 2
        model = nn.Sequential(nn.Linear(64, 64, bias=False))
        model[0]._quant_enabled = True
        model[0]._quant_scaling_type = "mxfp4"

        apply_fsdp2(model, self._args())


class TestRegisterQuantOptimizerHooks:
    """FSDP forward reads an all-gather buffer, so the MXFP4 weight cache's
    ``_version`` fallback cannot see optimizer steps. Losing this registration
    leaves quantized layers training against step-0 weights, silently."""

    def _model_and_optimizer(self):
        model = nn.Sequential(nn.Linear(8, 8))
        return model, torch.optim.SGD(model.parameters(), lr=0.1)

    def test_registers_for_mxfp4(self):
        from atomrl.lumen.models.fsdp import register_quant_optimizer_hooks

        model, optimizer = self._model_and_optimizer()
        with mock.patch(
            "atomrl.lumen.quantize.register_mxfp4_weight_optimizer_hooks"
        ) as reg:
            assert (
                register_quant_optimizer_hooks(
                    model, optimizer, SimpleNamespace(linear_fp4=True)
                )
                is True
            )
        reg.assert_called_once_with(model, optimizer)

    def test_skips_when_not_mxfp4(self):
        from atomrl.lumen.models.fsdp import register_quant_optimizer_hooks

        model, optimizer = self._model_and_optimizer()
        with mock.patch(
            "atomrl.lumen.quantize.register_mxfp4_weight_optimizer_hooks"
        ) as reg:
            assert (
                register_quant_optimizer_hooks(
                    model, optimizer, SimpleNamespace(linear_fp8=True, linear_fp4=False)
                )
                is False
            )
        reg.assert_not_called()

    def test_unwraps_fsdp1_module_before_registering(self):
        from atomrl.lumen.models.fsdp import register_quant_optimizer_hooks

        inner, optimizer = self._model_and_optimizer()
        wrapper = nn.Module()
        wrapper.module = inner

        with mock.patch(
            "atomrl.lumen.quantize.register_mxfp4_weight_optimizer_hooks"
        ) as reg:
            register_quant_optimizer_hooks(
                wrapper, optimizer, SimpleNamespace(linear_fp4=True)
            )
        reg.assert_called_once_with(inner, optimizer)

    def test_hook_clears_cache_on_step(self):
        from atomrl.lumen.models.fsdp import register_quant_optimizer_hooks

        model, optimizer = self._model_and_optimizer()
        model[0]._mxfp4_w_cache = ((False, False), "fp4", "scale")
        register_quant_optimizer_hooks(
            model, optimizer, SimpleNamespace(linear_fp4=True)
        )

        optimizer.step()

        assert not hasattr(model[0], "_mxfp4_w_cache")
