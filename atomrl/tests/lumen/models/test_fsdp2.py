# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for FSDP2 distributed training integration.

How to run::

    # All tests (unit + distributed); distributed tests require >= 2 GPUs:
    pytest tests/models/test_fsdp2.py -v

    # Distributed FSDP2 training test (launches torchrun --nproc_per_node=2 internally):
    pytest tests/models/test_fsdp2.py -v -k "dist"
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from unittest.mock import MagicMock, patch

import pytest  # Task 2: @pytest.mark.parametrize
import torch
from torch import nn

_DIST = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2+ GPUs required",
)

_MXFP4_DIST = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="2+ gfx950 GPUs required",
)

_BF16_TRAIN_SCRIPT = textwrap.dedent("""\
    import argparse
    import os
    import torch
    import torch.distributed as dist
    from transformers import LlamaConfig, LlamaForCausalLM

    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.manual_seed(42)

    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        vocab_size=256,
    )
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).cuda()

    args = argparse.Namespace(
        linear_fp8=False,
        sharding_strategy="full_shard",
    )
    from atomrl.lumen.models.fsdp import apply_fsdp2
    apply_fsdp2(model, args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 256, (2, 16), device="cuda")
    labels = input_ids.clone()

    try:
        losses = []
        for step in range(10):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        if rank == 0:
            assert losses[-1] < losses[0] * 0.8, (
                f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
            )
            print(f"PASS: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
""")

_FP8_TRAIN_SCRIPT = textwrap.dedent("""\
    import argparse
    import os
    import torch
    import torch.distributed as dist
    from transformers import LlamaConfig, LlamaForCausalLM

    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.manual_seed(42)

    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        vocab_size=256,
    )
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).cuda()

    args = argparse.Namespace(
        linear_fp8=True,
        sharding_strategy="full_shard",
    )
    from atomrl.lumen.models.fsdp import apply_fp8_training, apply_fsdp2
    apply_fp8_training(model, args)
    apply_fsdp2(model, args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 256, (2, 16), device="cuda")
    labels = input_ids.clone()

    try:
        losses = []
        for step in range(10):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        if rank == 0:
            assert losses[-1] < losses[0] * 0.9, (
                f"FP8 loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
            )
            print(f"PASS: FP8 loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
""")

# Frozen blockwise2d FP8 base (Blockwise2DFP8Param, all-gathered as FP8) coexisting
# with a trainable BF16 head in one fully_shard group — the mixed dtype/grad case.
_FP8_PARAM_STORAGE_SCRIPT = textwrap.dedent("""\
    import argparse
    import os
    import torch
    import torch.nn as nn
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.manual_seed(42)

    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 256, bias=False)
            self.fc2 = nn.Linear(256, 256, bias=False)
        def forward(self, x):
            return torch.relu(self.fc2(torch.relu(self.fc1(x))))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = Core()
            self.head = nn.Linear(256, 256, bias=False)   # trainable BF16
        def forward(self, x):
            return self.head(self.core(x))

    net = Net().to(torch.bfloat16).cuda()
    for p in net.core.parameters():
        p.requires_grad_(False)               # frozen base (LoRA recipe)

    from atomrl.lumen.quantize import enable
    from atomrl.lumen.quantize.config import QuantConfig
    enable(net.core, config=QuantConfig.from_str(scaling="blockwise2d", block_size=128))

    args = argparse.Namespace(
        linear_fp8=True,
        sharding_strategy="full_shard",
        fsdp_fp8_param_storage=True,
    )
    from atomrl.lumen.models.fsdp import apply_fsdp2
    from atomrl.lumen.quantize.comm_tensor import Blockwise2DFP8Param

    # Base weights must be FP8-wrapped (pre-shard); the helper runs inside apply_fsdp2.
    apply_fsdp2(net, args)

    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)

    try:
        losses = []
        for step in range(10):
            out = net(x)
            loss = (out.float() - target.float()).pow(2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        if rank == 0:
            assert all(l == l for l in losses), f"NaN loss: {losses}"
            assert losses[-1] < losses[0], (
                f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
            )
            print(f"PASS: FP8 param-storage loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
""")


_FP8_PARAM_STORAGE_NUMERICS_SCRIPT = textwrap.dedent("""\
    import argparse
    import os
    import torch
    import torch.nn as nn
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.manual_seed(0)   # same weights + input on every rank (data-parallel)

    N, K = 512, 256        # N % (128 * world_size) == 0 for world_size in {1,2,4}
    lin = nn.Linear(K, N, bias=False).to(torch.bfloat16).cuda()
    lin.weight.requires_grad_(False)
    x = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)

    from atomrl.lumen.quantize import enable
    from atomrl.lumen.quantize.config import QuantConfig, _get_float8_e4m3
    enable(lin, config=QuantConfig.from_str(scaling="blockwise2d", block_size=128))

    # Reference: dequantized FP8 weight @ x, computed from the SAME quant the
    # storage path uses (bf16 activation, no activation quant) — a loose upper bound.
    from atomrl.lumen.ops.quantize.linear import _quant_blockwise2d_weight
    from atomrl.lumen.ops.quantize.gemm_primitives import _dequant_fp8_weight
    w_bf16 = lin.weight.data.float().clone()
    fp8r, scaler = _quant_blockwise2d_weight(lin.weight.data.contiguous(), _get_float8_e4m3(), 128)
    w_deq = _dequant_fp8_weight(fp8r, scaler, 128).float()
    y_ref = (x.float() @ w_deq.t())

    model = nn.Sequential(lin)
    args = argparse.Namespace(
        linear_fp8=True, sharding_strategy="full_shard", fsdp_fp8_param_storage=True,
    )
    from atomrl.lumen.models.fsdp import apply_fsdp2
    apply_fsdp2(model, args)

    try:
        with torch.no_grad():
            y = model(x).float()
        if rank == 0:
            # Magnitude sanity: the scale-drop bug makes y ~thousands x too large.
            r = (y.abs().mean() / y_ref.abs().mean().clamp(min=1e-9)).item()
            assert 0.5 < r < 2.0, f"output magnitude off by {r:.1f}x (scale not applied?)"
            # SNR vs dequant reference (FP8 weight GEMM + activation quant).
            num = y_ref.pow(2).mean()
            den = (y - y_ref).pow(2).mean().clamp(min=1e-12)
            snr = 10 * torch.log10(num / den).item()
            assert snr > 12, f"SNR too low: {snr:.1f} dB"
            print(f"PASS: magnitude ratio {r:.3f}, SNR {snr:.1f} dB")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
""")


_MXFP4_COMM_SCRIPT = textwrap.dedent("""\
    import argparse
    import os
    import torch
    import torch.nn as nn
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 64, bias=False)

        def forward(self, x):
            return torch.nn.functional.gelu(self.proj(x))

    class ToyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = ToyTransformer().to(torch.bfloat16).cuda()
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        y_ref = x.float()
        for layer in model.layers:
            y_ref = torch.nn.functional.gelu(
                torch.nn.functional.linear(y_ref, layer.proj.weight.float())
            )

    # Follow the shipped trainer's configuration path rather than calling the
    # low-level quantizer directly.
    from atomrl.lumen.config import LumenConfig
    _, model = LumenConfig.from_args(argparse.Namespace(
        linear_fp8=False,
        linear_fp4=True,
        lora_rank=0,
    )).enable(model)

    args = argparse.Namespace(
        linear_fp8=False,
        linear_fp4=True,
        fsdp_version=2,
        sharding_strategy="full_shard",
        fsdp_mxfp4_comm=True,
        fsdp_fp8_param_storage=False,
        lumen_fp8_param_gather=False,
    )
    from atomrl.lumen.models.fsdp import apply_fsdp2
    from atomrl.lumen.quantize.comm_tensor import MXFP4CommTensor
    apply_fsdp2(model, args)

    # The local shard must retain the subclass so FSDP2 can discover the
    # pre/post-all-gather extension instead of silently communicating BF16.
    local_weight = model.layers[0].proj.weight.to_local()
    assert isinstance(local_weight, MXFP4CommTensor), type(local_weight)
    initial_local_weight = local_weight._data.detach().clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    try:
        for step in range(2):
            out = model(x)
            if step == 0:
                num = y_ref.square().mean()
                den = (out.float() - y_ref).square().mean().clamp(min=1e-12)
                snr = 10 * torch.log10(num / den).item()
                assert snr > 8, f"MXFP4 FSDP2 SNR too low: {snr:.1f} dB"
            loss = out.float().square().mean()
            assert torch.isfinite(loss), loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        updated_local_weight = model.layers[0].proj.weight.to_local()._data
        assert torch.isfinite(updated_local_weight).all()
        assert not torch.equal(updated_local_weight, initial_local_weight), (
            "trainable MXFP4CommTensor shard was not updated by the optimizer"
        )
        if rank == 0:
            print("PASS: trainable MXFP4 FSDP2 communication hook forward/backward/update")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
""")


_MXFP4_COMM_NUMERICS_SCRIPT = textwrap.dedent("""\
    import argparse
    import os
    import torch
    import torch.nn as nn
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.manual_seed(0)   # same weights + input on every rank (data-parallel)

    # N % (32 * world_size) == 0 for world_size in {1,2,4}: each rank's row shard
    # tiles the same way the full tensor does, so the gathered scales must line
    # up by dim-0 concat. A scale tensor concatenated on the wrong axis still
    # has a plausible shape here, which is why this asserts on magnitude.
    N, K = 512, 256
    lin = nn.Linear(K, N, bias=False).to(torch.bfloat16).cuda()
    x = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)

    from atomrl.lumen.ops.quantize.ops import convert_from_mxfp4_2d, convert_to_mxfp4_2d

    # Reference: quantize the FULL weight, dequantize, GEMM in BF16. The comm
    # path must land here up to per-shard rounding, not off by a scale factor.
    w_full = lin.weight.data.contiguous()
    fp4_ref, scale_ref = convert_to_mxfp4_2d(w_full)
    w_deq = convert_from_mxfp4_2d(fp4_ref, scale_ref, torch.bfloat16).float()
    y_ref = x.float() @ w_deq.t()

    from atomrl.lumen.config import LumenConfig
    model = nn.Sequential(lin)
    _, model = LumenConfig.from_args(argparse.Namespace(
        linear_fp8=False, linear_fp4=True, lora_rank=0,
    )).enable(model)

    args = argparse.Namespace(
        linear_fp8=False,
        linear_fp4=True,
        fsdp_version=2,
        sharding_strategy="full_shard",
        fsdp_mxfp4_comm=True,
        fsdp_fp8_param_storage=False,
        lumen_fp8_param_gather=False,
    )
    from atomrl.lumen.models.fsdp import apply_fsdp2
    apply_fsdp2(model, args)

    try:
        with torch.no_grad():
            y = model(x).float()
        if rank == 0:
            r = (y.abs().mean() / y_ref.abs().mean().clamp(min=1e-9)).item()
            assert 0.5 < r < 2.0, (
                f"output magnitude off by {r:.1f}x after {world_size}-rank FP4 "
                "all-gather (scales mis-concatenated or dropped?)"
            )
            num = y_ref.pow(2).mean()
            den = (y - y_ref).pow(2).mean().clamp(min=1e-12)
            snr = 10 * torch.log10(num / den).item()
            assert snr > 12, f"SNR too low: {snr:.1f} dB"
            print(f"PASS: {world_size}-rank MXFP4 gather magnitude {r:.3f}, SNR {snr:.1f} dB")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
""")


_MXFP4_CACHE_INVALIDATION_SCRIPT = textwrap.dedent("""\
    import argparse
    import os
    import torch
    import torch.nn as nn
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 64, bias=False)

        def forward(self, x):
            return torch.nn.functional.gelu(self.proj(x))

    class ToyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = ToyTransformer().to(torch.bfloat16).cuda()

    from atomrl.lumen.config import LumenConfig
    _, model = LumenConfig.from_args(argparse.Namespace(
        linear_fp8=False,
        linear_fp4=True,
        lora_rank=0,
    )).enable(model)

    # No MXFP4CommTensor here: this is the plain MXFP4 FSDP2 path, where the
    # weight the forward quantizes is an all-gather buffer whose `_version`
    # does not move when the optimizer updates the sharded parameter.
    args = argparse.Namespace(
        linear_fp8=False,
        linear_fp4=True,
        fsdp_version=2,
        sharding_strategy="full_shard",
        fsdp_mxfp4_comm=False,
        fsdp_fp8_param_storage=False,
        lumen_fp8_param_gather=False,
    )
    from atomrl.lumen.models.fsdp import apply_fsdp2, register_quant_optimizer_hooks
    apply_fsdp2(model, args)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    assert register_quant_optimizer_hooks(model, optimizer, args)

    # Same input every step, so the loss can only move if the forward sees the
    # updated weights. A stale FP4 weight cache pins it instead, which is how
    # this shipped: weights updated, loss and grad norm flat, nothing raised.
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    losses = []
    try:
        for _ in range(3):
            loss = model(x).float().square().mean()
            assert torch.isfinite(loss), loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        assert losses[1] != losses[0] and losses[2] != losses[1], (
            f"MXFP4 forward ignored optimizer updates (stale weight cache): {losses}"
        )
        if rank == 0:
            print(f"PASS: MXFP4 FSDP2 weight cache invalidated on step: {losses}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
""")


class TestFSDP2CLIArgs:

    def test_fsdp_version_default(self):
        from atomrl.lumen.models.fsdp import add_common_fsdp_args

        parser = argparse.ArgumentParser()
        add_common_fsdp_args(parser)
        args = parser.parse_args([])
        assert args.fsdp_version == 1

    def test_fsdp_version_2(self):
        from atomrl.lumen.models.fsdp import add_common_fsdp_args

        parser = argparse.ArgumentParser()
        add_common_fsdp_args(parser)
        args = parser.parse_args(["--fsdp-version", "2"])
        assert args.fsdp_version == 2


class TestApplyFSDP2:

    def test_function_exists(self):
        from atomrl.lumen.models.fsdp import apply_fsdp2

        assert callable(apply_fsdp2)

    def test_fully_shard_receives_device_mesh(self):
        from atomrl.lumen.models.fsdp import apply_fsdp2

        model = nn.Sequential(nn.Linear(8, 4))
        args = argparse.Namespace(
            linear_fp8=False,
            sharding_strategy="full_shard",
        )

        with (
            patch("torch.distributed.fsdp.fully_shard") as mock_fs,
            patch("torch.distributed.device_mesh.init_device_mesh") as mock_mesh,
            patch("atomrl.lumen.models.fsdp.dist") as mock_dist,
        ):
            mock_dist.get_world_size.return_value = 1
            mock_mesh.return_value = MagicMock()
            mock_fs.side_effect = lambda m, **kw: m
            apply_fsdp2(model, args)

            for call in mock_fs.call_args_list:
                mesh_arg = call.kwargs.get("mesh")
                assert mesh_arg is mock_mesh.return_value

    def test_no_shard_raises_with_fsdp2(self):
        from atomrl.lumen.models.fsdp import apply_fsdp2

        model = nn.Linear(8, 4)
        args = argparse.Namespace(
            linear_fp8=False,
            sharding_strategy="no_shard",
        )
        with pytest.raises(ValueError, match="no_shard is not supported"):
            apply_fsdp2(model, args)

    @pytest.mark.parametrize(
        "strategy,expected_reshard",
        [
            ("full_shard", True),
            ("shard_grad_op", False),
        ],
    )
    def test_reshard_after_forward_mapping(self, strategy, expected_reshard):
        from atomrl.lumen.models.fsdp import apply_fsdp2

        model = nn.Sequential(nn.Linear(8, 4))
        args = argparse.Namespace(
            linear_fp8=False,
            sharding_strategy=strategy,
        )

        with (
            patch("torch.distributed.fsdp.fully_shard") as mock_fs,
            patch("torch.distributed.device_mesh.init_device_mesh") as mock_mesh,
            patch("atomrl.lumen.models.fsdp.dist") as mock_dist,
        ):
            mock_dist.get_world_size.return_value = 1
            mock_mesh.return_value = MagicMock()
            mock_fs.side_effect = lambda m, **kw: m
            apply_fsdp2(model, args)

            for c in mock_fs.call_args_list:
                assert c.kwargs["reshard_after_forward"] == expected_reshard

    def test_shards_each_decoder_layer(self):
        from atomrl.lumen.models.fsdp import apply_fsdp2

        layer0 = nn.Linear(8, 8)
        layer1 = nn.Linear(8, 8)
        inner = nn.Module()
        inner.layers = nn.ModuleList([layer0, layer1])
        model = nn.Module()
        model.add_module("model", inner)
        model.add_module("lm_head", nn.Linear(8, 4))

        args = argparse.Namespace(
            linear_fp8=False,
            sharding_strategy="full_shard",
        )

        with (
            patch("torch.distributed.fsdp.fully_shard") as mock_fs,
            patch("torch.distributed.device_mesh.init_device_mesh") as mock_mesh,
            patch("atomrl.lumen.models.fsdp.dist") as mock_dist,
        ):
            mock_dist.get_world_size.return_value = 1
            mock_mesh.return_value = MagicMock()
            mock_fs.side_effect = lambda m, **kw: m
            apply_fsdp2(model, args)

            assert mock_fs.call_count == 3
            sharded_modules = [c.args[0] for c in mock_fs.call_args_list]
            assert layer0 in sharded_modules
            assert layer1 in sharded_modules
            assert model in sharded_modules


class TestResetFp8StateFSDP2:

    def test_clears_fp8_flags_without_module_wrapper(self):
        """reset_fp8_state finds FP8 modules in FSDP2 (no .module wrapper)."""
        from atomrl.lumen.models.fsdp import reset_fp8_state

        layer = nn.Linear(8, 4)
        layer.fp8_initialized = True
        model = nn.Sequential(layer)
        model.fp8_initialized = True

        reset_fp8_state(model)

        assert not getattr(model, "fp8_initialized", False)
        assert not getattr(layer, "fp8_initialized", False)


@_DIST
class TestFSDP2Integration:

    @staticmethod
    def _get_free_port():
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _run_training_script(self, script: str, timeout: int = 120):
        port = str(self._get_free_port())
        env = os.environ.copy()
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = port
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (repo_root, env.get("PYTHONPATH", "")) if part
        )
        kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            "env": env,
        }
        if sys.platform != "win32":
            kwargs["start_new_session"] = True

        # The script's own directory leads sys.path, so it must be a directory
        # we control. Writing straight into /tmp lets any stray /tmp/<name>.py
        # shadow a real package: a leftover /tmp/kernels.py once broke every
        # transformers-importing test here with an unrelated traceback.
        with tempfile.TemporaryDirectory() as script_dir:
            script_path = os.path.join(script_dir, "fsdp2_dist_case.py")
            with open(script_path, "w") as f:
                f.write(script)

            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--nproc_per_node=2",
                        "--master_addr=127.0.0.1",
                        f"--master_port={port}",
                        script_path,
                    ],
                    check=False,
                    **kwargs,
                )
            except subprocess.TimeoutExpired as exc:
                if sys.platform != "win32":
                    try:
                        os.killpg(os.getpgid(exc.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                stdout = (exc.stdout or "")[:2000]
                stderr = (exc.stderr or "")[:2000]
                pytest.fail(
                    f"Training script timed out after {timeout}s "
                    f"(port {port}).\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                )
        return result

    def test_bf16_fsdp2_overfit(self):
        """2-GPU LLaMA mini BF16 + FSDP2: loss decreases over 10 steps."""
        result = self._run_training_script(_BF16_TRAIN_SCRIPT)
        assert result.returncode == 0, (
            f"Training failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_fp8_fsdp2_overfit(self):
        """2-GPU LLaMA mini FP8 + FSDP2: loss decreases over 10 steps."""
        result = self._run_training_script(_FP8_TRAIN_SCRIPT)
        assert result.returncode == 0, (
            f"FP8 training failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_fp8_param_storage_fsdp2(self):
        """2-GPU frozen blockwise2d FP8 base (all-gathered as FP8) + trainable head."""
        result = self._run_training_script(_FP8_PARAM_STORAGE_SCRIPT)
        assert result.returncode == 0, (
            f"FP8 param-storage failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_fp8_param_storage_numerics_fsdp2(self):
        """2-GPU absolute-correctness: the all-gathered FP8 weight must apply its 2D
        scale (the scale-drop bug inflated the output ~thousands x)."""
        result = self._run_training_script(_FP8_PARAM_STORAGE_NUMERICS_SCRIPT)
        assert result.returncode == 0, (
            f"FP8 param-storage numerics failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    @_MXFP4_DIST
    def test_mxfp4_comm_fsdp2(self):
        """2-GPU MXFP4 all-gather keeps its extension and trains two steps."""
        result = self._run_training_script(_MXFP4_COMM_SCRIPT)
        assert result.returncode == 0, (
            f"MXFP4 communication failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    @_MXFP4_DIST
    def test_mxfp4_comm_fsdp2_numerics(self):
        """2-GPU absolute-correctness for the FP4 all-gather's scales.

        The per-rank scale shards must reassemble by dim-0 concat; getting that
        wrong leaves shapes plausible and magnitudes far off, which is how the
        equivalent FP8 scale-drop bug reached a run.
        """
        result = self._run_training_script(_MXFP4_COMM_NUMERICS_SCRIPT)
        assert result.returncode == 0, (
            f"MXFP4 comm numerics failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    @_MXFP4_DIST
    def test_mxfp4_fsdp2_forward_sees_optimizer_updates(self):
        """2-GPU MXFP4 without comm wrapping must re-quantize after each step.

        Covers the failure the comm-path test cannot: the weight shard updates
        correctly while the forward keeps reading a cached step-0 FP4 weight.
        """
        result = self._run_training_script(_MXFP4_CACHE_INVALIDATION_SCRIPT)
        assert result.returncode == 0, (
            f"MXFP4 weight cache invalidation failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
