# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for the in-place ROCm Megatron compatibility patches."""

from __future__ import annotations

import importlib.util
import sys
import typing
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_PATCH_SCRIPT = (
    Path(__file__).parents[2]
    / "examples"
    / "GRPO"
    / "dsv4"
    / "patch_rocm_megatron_dsv4.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "patch_rocm_megatron_dsv4", _PATCH_SCRIPT
)
assert _SPEC and _SPEC.loader
patcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(patcher)


def test_patch_tp_copy_supports_fp32_gradient_reduction(tmp_path) -> None:
    mappings = tmp_path / "megatron" / "core" / "tensor_parallel" / "mappings.py"
    mappings.parent.mkdir(parents=True)
    mappings.write_text("""
def copy_to_tensor_model_parallel_region(input_, group=None):
    group = get_tensor_model_parallel_group_if_none(group)
    return _CopyToModelParallelRegion.apply(input_, group)

class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None
""")

    assert hasattr(patcher, "patch_tp_copy_fp32_gradient_reduce")
    assert patcher.patch_tp_copy_fp32_gradient_reduce(str(tmp_path))

    patched = mappings.read_text()
    assert "all_reduce_grad_fp32=False" in patched
    assert "grad_output.float()" in patched
    assert "to(grad_output.dtype)" in patched
    assert "return grad_input, None, None" in patched
    assert not patcher.patch_tp_copy_fp32_gradient_reduce(str(tmp_path))


def test_patch_transformer_layer_wraps_tensor_attention_output(tmp_path) -> None:
    layer = tmp_path / "megatron" / "core" / "transformer" / "transformer_layer.py"
    layer.parent.mkdir(parents=True)
    layer.write_text("""
        attention_output_with_bias = self.self_attention()
        nvtx_range_pop(suffix="self_attention")

        if self.recompute_input_layernorm:
            pass
""")

    assert patcher.patch_transformer_layer(str(tmp_path))

    patched = layer.read_text()
    assert "isinstance(attention_output_with_bias, torch.Tensor)" in patched
    assert "attention_output_with_bias = (attention_output_with_bias, None)" in patched
    assert not patcher.patch_transformer_layer(str(tmp_path))


def test_patch_transformer_forward_integrates_hyper_connections(tmp_path) -> None:
    transformer = tmp_path / "megatron" / "core" / "transformer"
    transformer.mkdir(parents=True)
    block = transformer / "transformer_block.py"
    block.write_text("""
        self._build_layers()

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        # Final layer norm.
""")
    layer = transformer / "transformer_layer.py"
    layer.write_text("""
        self.bias_dropout_add_exec_handler = torch.enable_grad

        # Residual connection.
        residual = hidden_states

        nvtx_range_pop(suffix="self_attention")

        if isinstance(attention_output_with_bias, torch.Tensor):
            attention_output_with_bias = (attention_output_with_bias, None)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self._forward_pre_mlp_layernorm(hidden_states)

        else:
            return self._forward_post_mlp(mlp_output_with_bias, residual)

    def _forward_post_mlp(self, mlp_output_with_bias, residual):
        nvtx_range_push(suffix="mlp_bda")
        if using_fused_tp_inference_kernel:
            hidden_states = mlp_output_with_bias[0]
""")

    assert patcher.patch_transformer_block(str(tmp_path))
    assert patcher.patch_transformer_layer(str(tmp_path))

    block_text = block.read_text()
    assert "self.hc_util.block_expand(hidden_states)" in block_text
    assert "self.hc_util.block_head(" in block_text
    layer_text = layer.read_text()
    assert "hc_util.layer_pre(" in layer_text
    assert "hc_util.layer_post(" in layer_text
    assert "hc_ffn_post=hc_ffn_post" in layer_text


def test_patch_hybrid_optimizer_disables_foreach_temporaries(tmp_path) -> None:
    optimizer = (
        tmp_path
        / "megatron"
        / "core"
        / "optimizer"
        / "cpu_offloading"
        / "hybrid_optimizer.py"
    )
    optimizer.parent.mkdir(parents=True)
    optimizer.write_text("""
self.cpu_optimizers = [self.cpu_optimizer_cls(self.cpu_param_groups)]
self.gpu_optimizer = self.gpu_optimizer_cls(self.gpu_param_groups)
cpu_optimizers.append(cpu_optimizer_cls([_cpu_param_group]))
""")

    assert hasattr(patcher, "patch_hybrid_optimizer_disable_foreach")
    assert patcher.patch_hybrid_optimizer_disable_foreach(str(tmp_path))

    patched = optimizer.read_text()
    assert patched.count("foreach=False") == 3
    assert not patcher.patch_hybrid_optimizer_disable_foreach(str(tmp_path))


def test_patch_distrib_optimizer_routes_offloaded_grads_without_gpu_fp32_copy(
    tmp_path,
) -> None:
    optimizer = tmp_path / "megatron" / "core" / "optimizer" / "distrib_optimizer.py"
    optimizer.parent.mkdir(parents=True)
    optimizer.write_text("shard_main_param.grad = shard_model_grad.float()\n")

    assert hasattr(patcher, "patch_distrib_optimizer_grad_copy")
    assert patcher.patch_distrib_optimizer_grad_copy(str(tmp_path))

    patched = optimizer.read_text()
    assert "gpu_params_map_cpu_copy" in patched
    assert "shard_main_param.decoupled_grad = shard_model_grad" in patched
    assert "shard_model_grad.to(shard_main_param)" not in patched
    assert not patcher.patch_distrib_optimizer_grad_copy(str(tmp_path))


def _write_distrib_optimizer_checkpoint_fixture(path: Path) -> None:
    """Write the relevant p22 DistributedOptimizer source anchors."""
    path.parent.mkdir(parents=True)
    path.write_text("""
import torch

HAVE_APEX_OR_TE = True
USING_TE_OPTIMIZER = False
USING_APEX_OPTIMIZER = False
logger = __import__("logging").getLogger(__name__)


class HybridDeviceOptimizer:
    def __init__(self, sub_optimizers=None):
        self.sub_optimizers = sub_optimizers or []
        self.param_groups = [{"params": []}]

    def state_dict(self):
        return {
            "state": (
                self.sub_optimizers[0].state if self.sub_optimizers else {}
            ),
            "param_groups": [
                {"params": list(range(len(self.param_groups[0]["params"])))}
            ],
        }


class DistributedOptimizer:
    def __init__(self):
        (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,
            self.shard_fp32_groups,
            self.shard_fp32_from_float16_groups,
        ) = self._build_model_and_main_param_groups(
            self.gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges, config
        )

    def state_dict(self):
        inner_state_dict = self.optimizer.state_dict()
        state_dict = {}

        if not HAVE_APEX_OR_TE:
            steps = list(set([s["step"].item() for s in inner_state_dict["state"].values()]))
            assert len(steps) == 1
            step = steps[0]
        elif isinstance(self.optimizer, HybridDeviceOptimizer):
            step = None
            for optimizer in self.optimizer.sub_optimizers:
                if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                    if len(optimizer.state) == 0:
                        continue
                    steps = list(set([s["step"].item() for s in optimizer.state.values()]))
                    assert len(steps) == 1
                    step = steps[0]
                    break
        elif USING_TE_OPTIMIZER or USING_APEX_OPTIMIZER:
            steps = list(
                set(
                    [
                        g["step"]
                        for g in inner_state_dict["param_groups"]
                        if len(g["params"]) > 0 and "step" in g
                    ]
                )
            )
            assert len(steps) <= 1
            step = steps[0] if len(steps) == 1 else None

        state_dict["optimizer"] = {
            key: value for key, value in inner_state_dict.items() if key != "state"
        }
        for param_group in state_dict["optimizer"]["param_groups"]:
            del param_group["params"]
            if (
                USING_TE_OPTIMIZER
                or USING_APEX_OPTIMIZER
                or isinstance(self.optimizer, HybridDeviceOptimizer)
            ) and step is not None:
                param_group["step"] = int(step)
        return state_dict

    def _set_main_param_and_optimizer_states(self, model_param, tensors):
        group_index, group_order = self.model_param_group_index_map[model_param]
        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
        main_param.copy_(tensors["param"])
        self.optimizer.state[main_param]["exp_avg"].copy_(tensors["exp_avg"])
        self.optimizer.state[main_param]["exp_avg_sq"].copy_(tensors["exp_avg_sq"])

    def _get_main_param_and_optimizer_states(self, model_param):
        group_index, group_order = self.model_param_group_index_map[model_param]
        if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            sharded_model_param = self.optimizer.param_groups[group_index]["params"][
                group_order
            ]
            tensors = {
                key: value
                for key, value in self.optimizer.state[sharded_model_param].items()
            }
            tensors["param"] = tensors.pop("master_param")
            return tensors
        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
        return {"param": main_param, **self.optimizer.state[main_param]}

    def load_parameter_state_from_dp_zero(self, state_dict, *, update_legacy_format=False):
        data_parallel_world_size = self.data_parallel_group_gloo.size()
        data_parallel_rank = self.data_parallel_group_gloo.rank()
        data_parallel_group_gloo = self.data_parallel_group_gloo
        data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
            self.data_parallel_group_gloo
        )

        if data_parallel_rank == 0:
            self.split_state_dict_if_needed(state_dict)

        # Scatter tensors to all DP ranks.
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                if data_parallel_rank == 0:
                    buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
                    checkpoint_numel_unpadded = state_dict[gbuf_idx][dtype]["numel_unpadded"]
                    assert buffer_numel_unpadded == checkpoint_numel_unpadded, (
                        "Number of unpadded elements must be same"
                    )
                recv_tensors = {}
                for key in ("param", "exp_avg", "exp_avg_sq"):
                    offset_in_world_tensors = 0
                    for bucket_idx, gbuf_range_map in enumerate(
                        gbuf_range_map_for_all_buckets
                    ):
                        gbuf_world_numel = (
                            self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                        )
                        gbuf_local_numel = (
                            gbuf_world_numel // data_parallel_world_size
                        )
                        gbuf_world_numel_unpadded = (
                            self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                        )
                        recv_tensor = torch.zeros(
                            (gbuf_local_numel,), dtype=torch.float32, device="cpu"
                        )
                        if data_parallel_rank == 0:
                            world_tensors = state_dict[gbuf_idx][dtype][key]
                            start = offset_in_world_tensors
                            end = start + gbuf_world_numel_unpadded
                            world_tensor = torch.nn.functional.pad(
                                world_tensors[start:end],
                                (0, gbuf_world_numel - gbuf_world_numel_unpadded),
                            )
                            send_tensors = [
                                world_tensor[i : i + gbuf_local_numel]
                                for i in range(
                                    0, gbuf_world_numel, gbuf_local_numel
                                )
                            ]
                            offset_in_world_tensors += gbuf_world_numel_unpadded
                        else:
                            send_tensors = None
                        torch.distributed.scatter(
                            recv_tensor,
                            send_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )
                        for model_param, param_range_map in gbuf_range_map[
                            "param_map"
                        ].items():
                            if model_param not in recv_tensors:
                                recv_tensors[model_param] = {}
                            local_range = param_range_map["gbuf_local"]
                            recv_tensors[model_param][key] = recv_tensor[
                                local_range.start : local_range.end
                            ]
                for model_param, tensors in recv_tensors.items():
                    self._set_main_param_and_optimizer_states(model_param, tensors)

    @torch.no_grad()
    def load_parameter_state_from_fully_reshardable(self, state_dict: dict):
        pass

    def load_parameter_state(self, filename: str, *, update_legacy_format=False):
        state_dict = None
        if self.data_parallel_group.rank() == 0:
            state_dict = torch.load(filename)
        self.load_parameter_state_from_dp_zero(
            state_dict, update_legacy_format=update_legacy_format
        )
""")


def _load_patched_distrib_optimizer(tmp_path, *, optimize: int = 0):
    optimizer_path = (
        tmp_path / "megatron" / "core" / "optimizer" / "distrib_optimizer.py"
    )
    _write_distrib_optimizer_checkpoint_fixture(optimizer_path)
    assert patcher.patch_distrib_optimizer_hdo_checkpoint(str(tmp_path))
    patched = optimizer_path.read_text()
    namespace: dict[str, object] = {}
    exec(  # noqa: S102  # Test executes generated patch source in an isolated namespace.
        compile(
            patched,
            str(optimizer_path),
            "exec",
            optimize=optimize,
        ),
        namespace,
    )
    return optimizer_path, patched, namespace


def _checkpoint_load_instance(namespace, *, world_size: int, rank: int = 0):
    optimizer_type = namespace["DistributedOptimizer"]
    hdo_type = namespace["HybridDeviceOptimizer"]
    instance = optimizer_type.__new__(optimizer_type)

    class DataParallelGroup:
        def size(self):
            return world_size

        def rank(self):
            return rank

    class Range:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    first_model_param = object()
    second_model_param = object()
    first_main_param = torch.zeros(2)
    second_main_param = torch.zeros(3)
    hdo = hdo_type()
    hdo.param_groups = [{"params": [first_main_param, second_main_param]}]
    hdo.state = {
        first_main_param: {
            "exp_avg": torch.zeros(2),
            "exp_avg_sq": torch.zeros(2),
        },
        second_main_param: {
            "exp_avg": torch.zeros(3),
            "exp_avg_sq": torch.zeros(3),
        },
    }
    sync_observations = []

    def sync_hdo_state():
        sync_observations.append(
            (
                first_main_param.clone(),
                hdo.state[first_main_param]["exp_avg"].clone(),
                hdo.state[first_main_param]["exp_avg_sq"].clone(),
                second_main_param.clone(),
                hdo.state[second_main_param]["exp_avg"].clone(),
                hdo.state[second_main_param]["exp_avg_sq"].clone(),
            )
        )

    hdo._sync_hdo_state_to_sub_optimizers = sync_hdo_state
    instance.optimizer = hdo
    instance.config = SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False
    )
    instance.data_parallel_group_gloo = DataParallelGroup()
    instance.split_state_dict_if_needed = lambda _state_dict: None
    bucket = SimpleNamespace(
        numel_unpadded=5,
        grad_data=torch.empty(8),
    )
    instance.buffers = [SimpleNamespace(numel_unpadded=5, buckets=[bucket])]
    param_map = {
        first_model_param: {
            "gbuf_world": Range(0, 2),
            "gbuf_local": Range(0, 2),
        },
        second_model_param: {
            "gbuf_world": Range(2, 5),
            "gbuf_local": Range(2, 5),
        },
    }
    instance.gbuf_ranges = [{torch.float32: [{"param_map": param_map}]}]
    instance.model_param_group_index_map = {
        first_model_param: (0, 0),
        second_model_param: (0, 1),
    }
    state_dict = {
        0: {
            torch.float32: {
                "numel_unpadded": 5,
                "param": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
                "exp_avg": torch.tensor([11.0, 12.0, 13.0, 14.0, 15.0]),
                "exp_avg_sq": torch.tensor([21.0, 22.0, 23.0, 24.0, 25.0]),
            }
        }
    }
    return instance, hdo, state_dict, sync_observations


def _configure_dp2_checkpoint_layout(instance, state_dict) -> None:
    instance.buffers[0].numel_unpadded = 4
    instance.buffers[0].buckets[0].numel_unpadded = 4
    instance.buffers[0].buckets[0].grad_data = torch.empty(4)
    param_map = instance.gbuf_ranges[0][torch.float32][0]["param_map"]
    first_model_param = next(iter(param_map))
    instance.gbuf_ranges[0][torch.float32][0]["param_map"] = {
        first_model_param: param_map[first_model_param]
    }
    state_dict[0][torch.float32]["numel_unpadded"] = 4
    for key in ("param", "exp_avg", "exp_avg_sq"):
        state_dict[0][torch.float32][key] = state_dict[0][torch.float32][key][:4]


def test_patch_distrib_optimizer_checkpoint_matches_reordered_hdo_params(
    tmp_path,
) -> None:
    optimizer = tmp_path / "megatron" / "core" / "optimizer" / "distrib_optimizer.py"
    _write_distrib_optimizer_checkpoint_fixture(optimizer)

    assert hasattr(patcher, "patch_distrib_optimizer_hdo_checkpoint")
    assert patcher.patch_distrib_optimizer_hdo_checkpoint(str(tmp_path))

    patched = optimizer.read_text()
    assert "checkpoint order must match optimizer.param_groups" in patched
    assert (
        "self.model_param_group_index_map[model_param] = " "(group_index, group_order)"
    ) in patched
    first_patch = optimizer.read_bytes()
    assert not patcher.patch_distrib_optimizer_hdo_checkpoint(str(tmp_path))
    assert optimizer.read_bytes() == first_patch


def test_patch_distrib_optimizer_checkpoint_canonicalizes_divergent_hdo_steps(
    tmp_path, caplog
) -> None:
    _, _, namespace = _load_patched_distrib_optimizer(tmp_path)
    optimizer_type = namespace["DistributedOptimizer"]
    hdo_type = namespace["HybridDeviceOptimizer"]
    parameters = [torch.nn.Parameter(torch.tensor(float(index))) for index in range(3)]
    adam = torch.optim.Adam(parameters)
    original_steps = (25.0, 49.0, 50.0)
    for parameter, step in zip(parameters, original_steps):
        adam.state[parameter] = {
            "step": torch.tensor(step),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    hdo = hdo_type([adam])
    hdo.param_groups[0]["params"] = parameters
    instance = optimizer_type.__new__(optimizer_type)
    instance.optimizer = hdo
    namespace["HAVE_APEX_OR_TE"] = False

    with caplog.at_level("WARNING"):
        state_dict = instance.state_dict()

    assert state_dict["optimizer"]["param_groups"][0]["step"] == 50
    assert [adam.state[parameter]["step"].item() for parameter in parameters] == list(
        original_steps
    )
    assert "canonicalizing divergent HDO Adam steps" in caplog.text
    assert "minimum=25" in caplog.text
    assert "maximum=50" in caplog.text
    assert "unique_count=3" in caplog.text


def test_patch_distrib_optimizer_checkpoint_preserves_uniform_hdo_step(
    tmp_path, caplog
) -> None:
    _, _, namespace = _load_patched_distrib_optimizer(tmp_path)
    optimizer_type = namespace["DistributedOptimizer"]
    hdo_type = namespace["HybridDeviceOptimizer"]
    parameters = [torch.nn.Parameter(torch.tensor(float(index))) for index in range(2)]
    adam = torch.optim.Adam(parameters)
    for parameter in parameters:
        adam.state[parameter] = {
            "step": torch.tensor(50.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    hdo = hdo_type([adam])
    hdo.param_groups[0]["params"] = parameters
    instance = optimizer_type.__new__(optimizer_type)
    instance.optimizer = hdo
    namespace["HAVE_APEX_OR_TE"] = False

    with caplog.at_level("WARNING"):
        state_dict = instance.state_dict()

    assert state_dict["optimizer"]["param_groups"][0]["step"] == 50
    assert "canonicalizing divergent HDO Adam steps" not in caplog.text


def test_patch_distrib_optimizer_compiles_and_idempotently_patches_p22_source(
    tmp_path,
) -> None:
    optimizer = tmp_path / "megatron" / "core" / "optimizer" / "distrib_optimizer.py"
    _write_distrib_optimizer_checkpoint_fixture(optimizer)

    assert patcher.patch_distrib_optimizer_hdo_checkpoint(str(tmp_path))
    patched = optimizer.read_text()
    compile(patched, str(optimizer), "exec", optimize=2)
    method_start = patched.index("    def load_parameter_state_from_dp_zero(")
    method_end = patched.index(
        "\n    @torch.no_grad()\n"
        "    def load_parameter_state_from_fully_reshardable",
        method_start,
    )
    method_source = patched[method_start:method_end]
    fast_start = method_source.index(
        "        if (\n" "            data_parallel_world_size == 1"
    )
    fast_end = method_source.index("            return\n", fast_start)
    fast_source = method_source[fast_start : fast_end + len("            return\n")]
    assert "gbuf_world_in_bucket" in fast_source
    assert "_get_main_param_and_optimizer_states" in fast_source
    assert "final offset" in fast_source
    assert "assert " not in fast_source

    first_patch = optimizer.read_bytes()
    assert not patcher.patch_distrib_optimizer_hdo_checkpoint(str(tmp_path))
    assert optimizer.read_bytes() == first_patch


def test_patch_distrib_optimizer_checkpoint_uses_backward_compatible_mmap_load(
    tmp_path, monkeypatch
) -> None:
    _, _, namespace = _load_patched_distrib_optimizer(tmp_path)

    load_calls = []
    loaded_state = object()

    def fake_torch_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return loaded_state

    monkeypatch.setattr(torch, "load", fake_torch_load)
    optimizer_type = namespace["DistributedOptimizer"]
    instance = optimizer_type.__new__(optimizer_type)
    instance.data_parallel_group = SimpleNamespace(rank=lambda: 0)
    forwarded = []
    instance.load_parameter_state_from_dp_zero = (
        lambda state_dict, **kwargs: forwarded.append((state_dict, kwargs))
    )
    filename = str(tmp_path / "optimizer.pt")

    instance.load_parameter_state(filename, update_legacy_format=True)

    assert load_calls == [((filename,), {"mmap": True, "weights_only": False})]
    assert forwarded == [(loaded_state, {"update_legacy_format": True})]


def test_patch_distrib_optimizer_checkpoint_real_mmap_load_is_usable(
    tmp_path,
) -> None:
    _, _, namespace = _load_patched_distrib_optimizer(tmp_path)
    filename = tmp_path / "optimizer.pt"
    expected = torch.arange(16 * 1024, dtype=torch.float32)
    torch.save({"state": expected}, filename)

    optimizer_type = namespace["DistributedOptimizer"]
    instance = optimizer_type.__new__(optimizer_type)
    instance.data_parallel_group = SimpleNamespace(rank=lambda: 0)
    forwarded = []
    instance.load_parameter_state_from_dp_zero = (
        lambda state_dict, **kwargs: forwarded.append((state_dict, kwargs))
    )

    instance.load_parameter_state(str(filename))

    assert len(forwarded) == 1
    loaded, kwargs = forwarded[0]
    assert kwargs == {"update_legacy_format": False}
    torch.testing.assert_close(loaded["state"], expected)
    loaded["state"].add_(1)
    torch.testing.assert_close(loaded["state"], expected + 1)


def test_patch_distrib_optimizer_checkpoint_adds_dp1_hdo_inplace_load(
    tmp_path, monkeypatch
) -> None:
    _, _, namespace = _load_patched_distrib_optimizer(tmp_path)

    def forbidden_collective(*_args, **_kwargs):
        raise AssertionError("DP=1 HDO path must not pad or scatter")

    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda _group: [0]
    )
    monkeypatch.setattr(torch.nn.functional, "pad", forbidden_collective)
    monkeypatch.setattr(torch.distributed, "scatter", forbidden_collective)
    instance, hdo, state_dict, sync_observations = _checkpoint_load_instance(
        namespace, world_size=1
    )
    instance.load_parameter_state_from_dp_zero(state_dict)

    expected = (
        torch.tensor([1.0, 2.0]),
        torch.tensor([11.0, 12.0]),
        torch.tensor([21.0, 22.0]),
        torch.tensor([3.0, 4.0, 5.0]),
        torch.tensor([13.0, 14.0, 15.0]),
        torch.tensor([23.0, 24.0, 25.0]),
    )
    actual = (
        hdo.param_groups[0]["params"][0],
        hdo.state[hdo.param_groups[0]["params"][0]]["exp_avg"],
        hdo.state[hdo.param_groups[0]["params"][0]]["exp_avg_sq"],
        hdo.param_groups[0]["params"][1],
        hdo.state[hdo.param_groups[0]["params"][1]]["exp_avg"],
        hdo.state[hdo.param_groups[0]["params"][1]]["exp_avg_sq"],
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    assert len(sync_observations) == 1
    for synced_tensor, expected_tensor in zip(sync_observations[0], expected):
        torch.testing.assert_close(synced_tensor, expected_tensor)


def test_patch_distrib_optimizer_dp1_hdo_converts_restored_moments_to_bf16(
    tmp_path, monkeypatch
) -> None:
    _, _, namespace = _load_patched_distrib_optimizer(tmp_path)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda _group: [0]
    )
    instance, hdo, state_dict, sync_observations = _checkpoint_load_instance(
        namespace, world_size=1
    )
    hdo._lumen_streamed_adam_moment_dtype = torch.bfloat16

    instance.load_parameter_state_from_dp_zero(state_dict)

    first_param, second_param = hdo.param_groups[0]["params"]
    assert first_param.dtype == torch.float32
    assert second_param.dtype == torch.float32
    expected_moments = (
        torch.tensor([11.0, 12.0], dtype=torch.bfloat16),
        torch.tensor([21.0, 22.0], dtype=torch.bfloat16),
        torch.tensor([13.0, 14.0, 15.0], dtype=torch.bfloat16),
        torch.tensor([23.0, 24.0, 25.0], dtype=torch.bfloat16),
    )
    actual_moments = (
        hdo.state[first_param]["exp_avg"],
        hdo.state[first_param]["exp_avg_sq"],
        hdo.state[second_param]["exp_avg"],
        hdo.state[second_param]["exp_avg_sq"],
    )
    for actual, expected in zip(actual_moments, expected_moments):
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected)
    assert len(sync_observations) == 1
    synced_moments = (
        sync_observations[0][1],
        sync_observations[0][2],
        sync_observations[0][4],
        sync_observations[0][5],
    )
    assert all(moment.dtype == torch.bfloat16 for moment in synced_moments)


def test_patch_distrib_optimizer_upgrades_existing_dp1_hdo_moment_restore(
    tmp_path,
) -> None:
    optimizer_path, patched, _ = _load_patched_distrib_optimizer(tmp_path)
    moment_setup = """
            moment_dtype = getattr(
                self.optimizer, "_lumen_streamed_adam_moment_dtype", None
            )
            if moment_dtype not in (None, torch.float32, torch.bfloat16):
                raise ValueError(
                    validation_message(
                        "streamed Adam moment dtype must be float32 or bfloat16",
                        gbuf="all",
                        key="moment_dtype",
                    )
                )
"""
    moment_migration = """
                                if (
                                    key in ("exp_avg", "exp_avg_sq")
                                    and moment_dtype is not None
                                    and destination.dtype != moment_dtype
                                ):
                                    group_index, group_order = (
                                        self.model_param_group_index_map[model_param]
                                    )
                                    optimizer_param = self.optimizer.param_groups[
                                        group_index
                                    ]["params"][group_order]
                                    destination = torch.empty_like(
                                        destination,
                                        dtype=moment_dtype,
                                        device="cpu",
                                    )
                                    self.optimizer.state[optimizer_param][key] = (
                                        destination
                                    )
"""
    optimizer_path.write_text(
        patched.replace(moment_setup, "", 1).replace(moment_migration, "\n", 1)
    )

    assert patcher.patch_distrib_optimizer_hdo_checkpoint(str(tmp_path))
    upgraded = optimizer_path.read_text()
    assert moment_setup in upgraded
    assert moment_migration in upgraded


def test_patch_distrib_optimizer_dp1_hdo_rejects_mismatched_unpadded_layout_first(
    tmp_path, monkeypatch
) -> None:
    _, _, namespace = _load_patched_distrib_optimizer(tmp_path, optimize=2)
    instance, hdo, state_dict, sync_observations = _checkpoint_load_instance(
        namespace, world_size=1
    )
    state_dict[0][torch.float32]["numel_unpadded"] = 4

    def forbidden_collective(*_args, **_kwargs):
        raise AssertionError("layout validation must precede pad or scatter")

    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda _group: [0]
    )
    monkeypatch.setattr(torch.nn.functional, "pad", forbidden_collective)
    monkeypatch.setattr(torch.distributed, "scatter", forbidden_collective)

    with pytest.raises(
        RuntimeError, match="Number of unpadded elements must be same"
    ) as exc_info:
        instance.load_parameter_state_from_dp_zero(state_dict)

    message = str(exc_info.value)
    for context in (
        "gbuf=0",
        "dtype=torch.float32",
        "bucket=all",
        "key=numel_unpadded",
        "param=n/a",
    ):
        assert context in message
    targets = (
        hdo.param_groups[0]["params"][0],
        hdo.state[hdo.param_groups[0]["params"][0]]["exp_avg"],
        hdo.state[hdo.param_groups[0]["params"][0]]["exp_avg_sq"],
        hdo.param_groups[0]["params"][1],
        hdo.state[hdo.param_groups[0]["params"][1]]["exp_avg"],
        hdo.state[hdo.param_groups[0]["params"][1]]["exp_avg_sq"],
    )
    assert all(torch.count_nonzero(target).item() == 0 for target in targets)
    assert sync_observations == []


def _multi_buffer_dp1_checkpoint_layout(namespace):
    optimizer_type = namespace["DistributedOptimizer"]
    hdo_type = namespace["HybridDeviceOptimizer"]
    instance = optimizer_type.__new__(optimizer_type)

    class DataParallelGroup:
        def size(self):
            return 1

        def rank(self):
            return 0

    class Range:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    model_params = [object() for _ in range(4)]
    optimizer_params = [torch.full((2,), -1.0) for _ in model_params]
    main_params = [torch.zeros(2) for _ in model_params]
    hdo = hdo_type()
    hdo.param_groups = [{"params": optimizer_params}]
    hdo.state = {
        optimizer_param: {
            "master_param": main_param,
            "exp_avg": torch.zeros(2),
            "exp_avg_sq": torch.zeros(2),
        }
        for optimizer_param, main_param in zip(optimizer_params, main_params)
    }
    sync_observations = []
    hdo._sync_hdo_state_to_sub_optimizers = lambda: sync_observations.append("sync")
    instance.optimizer = hdo
    instance.config = SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=True
    )
    instance.data_parallel_group_gloo = DataParallelGroup()
    instance.split_state_dict_if_needed = lambda _state_dict: None
    instance.model_param_group_index_map = {
        model_param: (0, index) for index, model_param in enumerate(model_params)
    }

    def param_range(world_start, world_end, bucket_start, bucket_end):
        return {
            "gbuf_world": Range(world_start, world_end),
            "gbuf_world_in_bucket": Range(bucket_start, bucket_end),
            "gbuf_local": Range(bucket_start, bucket_end),
        }

    first_bucket = SimpleNamespace(numel_unpadded=3, grad_data=torch.empty(4))
    second_bucket = SimpleNamespace(numel_unpadded=4, grad_data=torch.empty(4))
    third_bucket = SimpleNamespace(numel_unpadded=2, grad_data=torch.empty(2))
    instance.buffers = [
        SimpleNamespace(
            numel_unpadded=7,
            buckets=[first_bucket, second_bucket],
        ),
        SimpleNamespace(numel_unpadded=2, buckets=[third_bucket]),
    ]
    first_dtype_buckets = [
        {
            "param_map": {
                model_params[0]: param_range(0, 2, 0, 2),
            }
        },
        {
            "param_map": {
                # p22's world range includes the prior padded bucket while
                # gbuf_world_in_bucket addresses the coalesced bucket slice.
                model_params[1]: param_range(4, 6, 0, 2),
                model_params[2]: param_range(6, 8, 2, 4),
            }
        },
    ]
    second_dtype_buckets = [
        {
            "param_map": {
                model_params[3]: param_range(0, 2, 0, 2),
            }
        }
    ]
    tuple_dtype = (torch.bfloat16, torch.float32)
    instance.gbuf_ranges = [
        {torch.float32: first_dtype_buckets},
        {tuple_dtype: second_dtype_buckets},
    ]

    def state(numel, base):
        values = torch.arange(base, base + numel, dtype=torch.float32)
        return {
            "numel_unpadded": numel,
            "param": values,
            "exp_avg": values + 100,
            "exp_avg_sq": values + 200,
        }

    state_dict = {
        0: {torch.float32: state(7, 1)},
        1: {tuple_dtype: state(2, 21)},
    }
    targets = [
        tensor
        for optimizer_param, main_param in zip(optimizer_params, main_params)
        for tensor in (
            main_param,
            hdo.state[optimizer_param]["exp_avg"],
            hdo.state[optimizer_param]["exp_avg_sq"],
        )
    ]
    return (
        instance,
        state_dict,
        main_params,
        optimizer_params,
        model_params,
        targets,
        sync_observations,
    )


def test_patch_distrib_optimizer_dp1_hdo_loads_p22_multibucket_multidtype_layout(
    tmp_path, monkeypatch
) -> None:
    _, _, namespace = _load_patched_distrib_optimizer(tmp_path)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda _group: [0]
    )
    (
        instance,
        state_dict,
        main_params,
        optimizer_params,
        _,
        _,
        sync_observations,
    ) = _multi_buffer_dp1_checkpoint_layout(namespace)

    def forbidden_buffer(*_args, **_kwargs):
        raise AssertionError("DP=1 HDO path must not allocate transfer buffers")

    with monkeypatch.context() as transfer_guard:
        transfer_guard.setattr(torch, "zeros", forbidden_buffer)
        transfer_guard.setattr(torch, "empty", forbidden_buffer)
        transfer_guard.setattr(torch.nn.functional, "pad", forbidden_buffer)
        transfer_guard.setattr(torch.distributed, "scatter", forbidden_buffer)
        instance.load_parameter_state_from_dp_zero(state_dict)

    expected_params = (
        [1.0, 2.0],
        [4.0, 5.0],
        [6.0, 7.0],
        [21.0, 22.0],
    )
    for main_param, optimizer_param, expected in zip(
        main_params, optimizer_params, expected_params
    ):
        torch.testing.assert_close(main_param, torch.tensor(expected))
        torch.testing.assert_close(
            instance.optimizer.state[optimizer_param]["exp_avg"],
            torch.tensor(expected) + 100,
        )
        torch.testing.assert_close(
            instance.optimizer.state[optimizer_param]["exp_avg_sq"],
            torch.tensor(expected) + 200,
        )
        torch.testing.assert_close(optimizer_param, torch.full((2,), -1.0))
    assert sync_observations == ["sync"]


@pytest.mark.parametrize(
    ("malformation", "error"),
    [
        ("aggregate_size", "bucket unpadded aggregate size"),
        ("missing_key", "missing checkpoint state exp_avg_sq"),
        ("source_length", "checkpoint state exp_avg_sq has 1 elements"),
        ("source_shape", "checkpoint state exp_avg_sq shape"),
        ("range", "invalid gbuf_world_in_bucket range"),
        ("destination_shape", "destination exp_avg_sq shape"),
    ],
)
def test_patch_distrib_optimizer_dp1_hdo_validates_complete_layout_before_copy(
    tmp_path, monkeypatch, malformation, error
) -> None:
    _, _, namespace = _load_patched_distrib_optimizer(tmp_path, optimize=2)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda _group: [0]
    )
    (
        instance,
        state_dict,
        _main_params,
        optimizer_params,
        model_params,
        targets,
        sync_observations,
    ) = _multi_buffer_dp1_checkpoint_layout(namespace)
    tuple_dtype = (torch.bfloat16, torch.float32)
    late_state = state_dict[1][tuple_dtype]
    if malformation == "aggregate_size":
        instance.buffers[1].buckets[0].numel_unpadded = 1
    elif malformation == "missing_key":
        del late_state["exp_avg_sq"]
    elif malformation == "source_length":
        late_state["exp_avg_sq"] = late_state["exp_avg_sq"][:1]
    elif malformation == "source_shape":
        late_state["exp_avg_sq"] = late_state["exp_avg_sq"].reshape(1, 2)
    elif malformation == "range":
        late_range = instance.gbuf_ranges[1][tuple_dtype][0]["param_map"][
            model_params[-1]
        ]["gbuf_world_in_bucket"]
        late_range.end = 3
    else:
        instance.optimizer.state[optimizer_params[-1]]["exp_avg_sq"] = torch.zeros(1, 2)
        targets[-1] = instance.optimizer.state[optimizer_params[-1]]["exp_avg_sq"]

    with pytest.raises((ValueError, RuntimeError), match=error) as exc_info:
        instance.load_parameter_state_from_dp_zero(state_dict)

    message = str(exc_info.value)
    assert "gbuf=1" in message
    assert f"dtype={tuple_dtype}" in message
    if malformation == "aggregate_size":
        expected_key = "metadata"
    elif malformation == "range":
        expected_key = "param"
    else:
        expected_key = "exp_avg_sq"
    assert f"key={expected_key}" in message
    if malformation in ("range", "destination_shape"):
        assert "bucket=0" in message
        assert "param=" in message
        assert "param=n/a" not in message
    else:
        assert "bucket=all" in message
        assert "param=n/a" in message
    assert all(torch.count_nonzero(target).item() == 0 for target in targets)
    assert sync_observations == []


def test_patch_distrib_optimizer_checkpoint_retains_dp2_collective_fallback(
    tmp_path, monkeypatch
) -> None:
    _, _, namespace = _load_patched_distrib_optimizer(tmp_path)

    scatter_calls = []

    def scatter(recv_tensor, send_tensors, *_args):
        scatter_calls.append((recv_tensor, send_tensors))
        recv_tensor.copy_(send_tensors[0])

    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda _group: [0, 1]
    )
    monkeypatch.setattr(torch.distributed, "scatter", scatter)
    instance, hdo, state_dict, sync_observations = _checkpoint_load_instance(
        namespace, world_size=2
    )
    _configure_dp2_checkpoint_layout(instance, state_dict)

    instance.load_parameter_state_from_dp_zero(state_dict)

    assert len(scatter_calls) == 3
    assert all(send_tensors is not None for _, send_tensors in scatter_calls)
    expected_partitions = (
        ([1.0, 2.0], [3.0, 4.0]),
        ([11.0, 12.0], [13.0, 14.0]),
        ([21.0, 22.0], [23.0, 24.0]),
    )
    for (_, send_tensors), expected in zip(scatter_calls, expected_partitions):
        assert [tensor.tolist() for tensor in send_tensors] == list(expected)
    torch.testing.assert_close(
        hdo.param_groups[0]["params"][0], torch.tensor([1.0, 2.0])
    )
    torch.testing.assert_close(
        hdo.state[hdo.param_groups[0]["params"][0]]["exp_avg"],
        torch.tensor([11.0, 12.0]),
    )
    torch.testing.assert_close(
        hdo.state[hdo.param_groups[0]["params"][0]]["exp_avg_sq"],
        torch.tensor([21.0, 22.0]),
    )
    assert len(sync_observations) == 1


def test_patch_distrib_optimizer_checkpoint_dp2_nonroot_receives_partition(
    tmp_path, monkeypatch
) -> None:
    _, _, namespace = _load_patched_distrib_optimizer(tmp_path)
    received_partitions = iter(
        (
            torch.tensor([3.0, 4.0]),
            torch.tensor([13.0, 14.0]),
            torch.tensor([23.0, 24.0]),
        )
    )
    scatter_calls = []

    def scatter(recv_tensor, send_tensors, *_args):
        scatter_calls.append(send_tensors)
        recv_tensor.copy_(next(received_partitions))

    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda _group: [0, 1]
    )
    monkeypatch.setattr(torch.distributed, "scatter", scatter)
    instance, hdo, state_dict, sync_observations = _checkpoint_load_instance(
        namespace, world_size=2, rank=1
    )
    _configure_dp2_checkpoint_layout(instance, state_dict)

    instance.load_parameter_state_from_dp_zero(None)

    assert scatter_calls == [None, None, None]
    main_param = hdo.param_groups[0]["params"][0]
    torch.testing.assert_close(main_param, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(
        hdo.state[main_param]["exp_avg"], torch.tensor([13.0, 14.0])
    )
    torch.testing.assert_close(
        hdo.state[main_param]["exp_avg_sq"], torch.tensor([23.0, 24.0])
    )
    assert len(sync_observations) == 1


def test_patch_hybrid_optimizer_streams_full_offload_sgd(tmp_path) -> None:
    optimizer = (
        tmp_path
        / "megatron"
        / "core"
        / "optimizer"
        / "cpu_offloading"
        / "hybrid_optimizer.py"
    )
    optimizer.parent.mkdir(parents=True)
    optimizer.write_text("""
class HybridDeviceOptimizer:
    def step(self, closure=None):
        self._sync_hdo_param_groups_to_sub_optimizers()

        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            self._set_sub_optimizer_grads()
""")

    assert hasattr(patcher, "patch_hybrid_optimizer_streaming_sgd")
    assert patcher.patch_hybrid_optimizer_streaming_sgd(str(tmp_path))

    patched = optimizer.read_text()
    assert "def _can_stream_full_offload_sgd(self):" in patched
    assert "def _stream_full_offload_sgd_step(self, closure=None):" in patched
    assert "if self._can_stream_full_offload_sgd():" in patched
    assert "torch.optim.SGD" in patched
    assert "staging = torch.empty(" in patched
    assert "(2, max_numel)," in patched
    assert "finish_slot(slot)" in patched
    assert "cpu_copy_map_grad" not in patched
    assert 'getattr(orig_param, "decoupled_grad", orig_param.grad)' in patched
    assert "cpu_param.mul_(1.0 - lr * weight_decay)" in patched
    assert "cpu_param.add_(staging_view, alpha=-lr)" in patched
    assert "orig_param.copy_(cpu_param" in patched
    assert not patcher.patch_hybrid_optimizer_streaming_sgd(str(tmp_path))


def _write_hybrid_optimizer_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""
import torch


class HybridDeviceOptimizer:
    def step(self, closure=None):
        self._sync_hdo_param_groups_to_sub_optimizers()

        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            self._set_sub_optimizer_grads()
""")


def test_patch_hybrid_optimizer_streams_full_offload_adam(tmp_path) -> None:
    optimizer = (
        tmp_path
        / "megatron"
        / "core"
        / "optimizer"
        / "cpu_offloading"
        / "hybrid_optimizer.py"
    )
    _write_hybrid_optimizer_fixture(optimizer)

    assert patcher.STREAMED_ADAM_CAPABILITY == ("ATOMRL_DSV4_CAPABILITY_STREAMED_ADAM")
    assert patcher.patch_hybrid_optimizer_streaming_adam(str(tmp_path))

    patched = optimizer.read_text()
    assert "from atomrl.engine.training.streamed_adam import (" in patched
    assert "AdamChunkOptions" in patched
    assert "adam_step_chunk_" in patched
    assert "initialize_adam_state" in patched
    assert "def _validate_full_offload_streaming_adam(" in patched
    assert "def _stream_full_offload_adam_step(self, closure=None):" in patched
    assert "ATOMRL_DSV4_CAPABILITY_STREAMED_ADAM = True" in patched

    adam_source = patched[
        patched.index("    def _validate_full_offload_streaming_adam(") : patched.index(
            "    def step(self, closure=None):"
        )
    ]
    assert 'getattr(self, "_lumen_streamed_adam_chunk_numel"' in adam_source
    assert "torch.empty(" in adam_source
    assert "(staging_rows, chunk_numel)" in adam_source
    assert "staging_rows = 4 if moment_dtype == torch.bfloat16 else 2" in adam_source
    assert "pin_memory=True" in adam_source
    assert "for chunk_start in range(0, cpu_param.numel(), chunk_numel):" in adam_source
    assert "finish_slot(slot)" in adam_source
    assert "adam_step_chunk_(" in adam_source
    assert "cpu_copy_map_grad" not in adam_source
    assert "_set_sub_optimizer_grads" not in adam_source
    assert "parameter_group={group_index}" in adam_source
    assert "chunk={chunk_index}" in adam_source

    first_patch = optimizer.read_text()
    assert not patcher.patch_hybrid_optimizer_streaming_adam(str(tmp_path))
    assert optimizer.read_text() == first_patch


def _patched_hybrid_optimizer_source(tmp_path: Path) -> tuple[Path, str]:
    optimizer = (
        tmp_path
        / "megatron"
        / "core"
        / "optimizer"
        / "cpu_offloading"
        / "hybrid_optimizer.py"
    )
    _write_hybrid_optimizer_fixture(optimizer)
    assert patcher.patch_hybrid_optimizer_streaming_adam(str(tmp_path))
    return optimizer, optimizer.read_text()


def test_streamed_adam_syncs_groups_before_validation_and_state_mutation(
    tmp_path,
) -> None:
    optimizer_path, patched = _patched_hybrid_optimizer_source(tmp_path)
    method_start = patched.index(
        "    def _stream_full_offload_adam_step(self, closure=None):"
    )
    method_end = patched.index("\n    def step(self, closure=None):", method_start)
    method_source = patched[method_start:method_end]
    assert method_source.index(
        "self._sync_hdo_param_groups_to_sub_optimizers()"
    ) < method_source.index("self._validate_full_offload_streaming_adam(closure)")
    assert method_source.index(
        "self._validate_full_offload_streaming_adam(closure)"
    ) < method_source.index(
        "initialize_adam_state(cpu_param, moment_dtype=moment_dtype)"
    )
    assert method_source.index(
        'getattr(\n            self, "_lumen_streamed_adam_moment_dtype"'
    ) < method_source.index(
        "initialize_adam_state(cpu_param, moment_dtype=moment_dtype)"
    )

    namespace: dict[str, object] = {}
    # Test executes generated patch source in an isolated namespace.
    exec(  # noqa: S102 -- test executes generated source
        compile(patched, str(optimizer_path), "exec"), namespace
    )
    optimizer_type = namespace["HybridDeviceOptimizer"]
    instance = optimizer_type.__new__(optimizer_type)
    group = {"lr": 1.0, "betas": (0.1, 0.2), "weight_decay": 3.0}
    observations = []

    def sync_groups():
        observations.append("sync")
        group.update(lr=0.01, betas=(0.9, 0.98), weight_decay=0.1)

    def validate(_closure):
        observations.append(
            ("validate", group["lr"], group["betas"], group["weight_decay"])
        )
        return []

    instance._sync_hdo_param_groups_to_sub_optimizers = sync_groups
    instance._validate_full_offload_streaming_adam = validate
    instance._sync_sub_optimizers_state_to_hdo = lambda: observations.append(
        "sync_state"
    )

    assert instance._stream_full_offload_adam_step() is None
    assert observations == [
        "sync",
        ("validate", 0.01, (0.9, 0.98), 0.1),
        "sync_state",
    ]
    assert instance._lumen_streamed_adam_last_h2d_event is None


def test_streamed_adam_records_completion_event_before_final_sync(
    tmp_path,
) -> None:
    _, patched = _patched_hybrid_optimizer_source(tmp_path)
    method_start = patched.index(
        "    def _stream_full_offload_adam_step(self, closure=None):"
    )
    method_end = patched.index("\n    def step(self, closure=None):", method_start)
    method_source = patched[method_start:method_end]
    copy_index = method_source.index("orig_chunk.copy_(cpu_chunk, non_blocking=True)")
    event_assignment = (
        "self._lumen_streamed_adam_last_h2d_event = " "self._h2d_stream.record_event()"
    )
    assert event_assignment in method_source
    event_index = method_source.index(event_assignment)
    final_sync_index = method_source.index(
        "self._h2d_stream.synchronize()", event_index
    )

    assert copy_index < event_index < final_sync_index


def test_streamed_adam_patch_rejects_missing_h2d_completion_event(
    tmp_path,
) -> None:
    optimizer, patched = _patched_hybrid_optimizer_source(tmp_path)
    event_assignment = (
        "            self._lumen_streamed_adam_last_h2d_event = "
        "self._h2d_stream.record_event()\n"
    )
    malformed = patched.replace(event_assignment, "", 1)
    optimizer.write_text(malformed)

    with pytest.raises(RuntimeError, match="partial.*streamed Adam"):
        patcher.patch_hybrid_optimizer_streaming_adam(str(tmp_path))

    assert optimizer.read_text() == malformed


class _FakeCudaTensor:
    def __init__(self, numel: int, *, contiguous: bool = True):
        self.is_cuda = True
        self.is_sparse = False
        self._numel = numel
        self._contiguous = contiguous
        self.grad = None
        self.decoupled_grad = None

    def numel(self):
        return self._numel

    def is_contiguous(self):
        return self._contiguous


def _streamed_adam_validation_fixture(
    tmp_path,
    *,
    cpu_contiguous: bool = True,
    original_contiguous: bool = True,
    gradient_contiguous: bool = True,
):
    optimizer_path, patched = _patched_hybrid_optimizer_source(tmp_path)
    namespace: dict[str, object] = {}
    # Test executes generated patch source in an isolated namespace.
    exec(  # noqa: S102 -- test executes generated source
        compile(patched, str(optimizer_path), "exec"), namespace
    )
    optimizer_type = namespace["HybridDeviceOptimizer"]
    instance = optimizer_type.__new__(optimizer_type)

    cpu_storage = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    if not cpu_contiguous:
        cpu_storage = cpu_storage.t()
    cpu_master = torch.nn.Parameter(cpu_storage)
    optimizer = torch.optim.AdamW(
        [cpu_master],
        lr=0.02,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.1,
        amsgrad=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=True,
    )
    gradient = _FakeCudaTensor(cpu_master.numel(), contiguous=gradient_contiguous)
    original = _FakeCudaTensor(cpu_master.numel(), contiguous=original_contiguous)
    original.decoupled_grad = gradient
    instance.offload_fraction = 1.0
    instance.gpu_optimizer = None
    instance.cpu_optimizers = [optimizer]
    instance.cpu_copys_map_gpu_param = {cpu_master: original}
    instance._lumen_streamed_adam_chunk_numel = 4
    return instance, optimizer, cpu_master


def test_streamed_adam_accepts_target_fused_adamw_metadata(tmp_path) -> None:
    instance, optimizer, cpu_master = _streamed_adam_validation_fixture(tmp_path)

    validated = instance._validate_full_offload_streaming_adam()

    assert len(validated) == 1
    options = validated[0][-1]
    assert options.lr == 0.02
    assert (options.beta1, options.beta2) == (0.9, 0.98)
    assert options.weight_decay == 0.1
    assert options.decoupled_weight_decay is True
    assert optimizer.param_groups[0]["fused"] is True
    assert optimizer.param_groups[0]["foreach"] is False
    assert cpu_master not in optimizer.state


@pytest.mark.parametrize(
    "flag",
    ["amsgrad", "differentiable", "capturable", "foreach"],
)
def test_streamed_adam_rejects_semantically_incompatible_flags(tmp_path, flag) -> None:
    instance, optimizer, cpu_master = _streamed_adam_validation_fixture(tmp_path)
    optimizer.param_groups[0][flag] = True

    with pytest.raises(RuntimeError, match=rf"parameter_group=0 flag={flag}"):
        instance._validate_full_offload_streaming_adam()

    assert cpu_master not in optimizer.state
    assert optimizer.state == {}


@pytest.mark.parametrize(
    "noncontiguous",
    ["cpu_master", "original", "gradient"],
)
def test_streamed_adam_rejects_noncontiguous_tensors_before_state_mutation(
    tmp_path, noncontiguous
) -> None:
    instance, optimizer, cpu_master = _streamed_adam_validation_fixture(
        tmp_path,
        cpu_contiguous=noncontiguous != "cpu_master",
        original_contiguous=noncontiguous != "original",
        gradient_contiguous=noncontiguous != "gradient",
    )

    with pytest.raises(
        RuntimeError,
        match=rf"contiguous.*parameter_group=0.*tensor={noncontiguous}",
    ):
        instance._validate_full_offload_streaming_adam()

    assert cpu_master not in optimizer.state
    assert optimizer.state == {}


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "staging.shape != (staging_rows, chunk_numel)",
            "staging.shape != (2, chunk_numel)",
        ),
        ("                pin_memory=True,", "                pin_memory=False,"),
        (
            "                dtype=torch.float32,",
            "                dtype=torch.float16,",
        ),
        (
            "            or staging.dtype != torch.float32\n",
            "            or staging.dtype != torch.float16\n",
        ),
        ("            or staging.dtype != torch.float32\n", ""),
        (
            "                    slot = sequence % 2",
            "                    slot = sequence % 3",
        ),
        (
            "        consumed = []",
            "        self.cpu_copy_map_grad()\n        consumed = []",
        ),
        (
            "        consumed = []",
            "        self._set_sub_optimizer_grads()\n        consumed = []",
        ),
        (
            "            if stream is None or stream is failed_stream:",
            "            if stream is None:",
        ),
        ("            orig_param.grad = None", "            pass  # grad retained"),
        (
            "            event.synchronize()",
            (
                "            orig_chunk.copy_(cpu_chunk, non_blocking=True)\n"
                "            event.synchronize()"
            ),
        ),
        (
            "        pending = [None, None]",
            (
                "        self._h2d_stream.synchronize()\n"
                "        pending = [None, None]"
            ),
        ),
        (
            (
                "            self._h2d_stream.synchronize()\n"
                "        except Exception as exc:\n"
                "            self._raise_streamed_adam_transfer_error("
            ),
            (
                "            pass  # final sync removed\n"
                "        except Exception as exc:\n"
                "            self._raise_streamed_adam_transfer_error("
            ),
        ),
        ("        def finish_slot(slot):", "        def finish_slot_broken(slot):"),
    ],
)
def test_streamed_adam_patch_rejects_malformed_full_patch(tmp_path, old, new) -> None:
    optimizer, patched = _patched_hybrid_optimizer_source(tmp_path)
    assert old in patched
    malformed = patched.replace(old, new, 1)
    optimizer.write_text(malformed)

    with pytest.raises(RuntimeError, match="partial.*streamed Adam"):
        patcher.patch_hybrid_optimizer_streaming_adam(str(tmp_path))

    assert optimizer.read_text() == malformed


def test_streamed_adam_transfer_failures_are_contextual_and_not_retried(
    tmp_path,
) -> None:
    _, patched = _patched_hybrid_optimizer_source(tmp_path)
    adam_source = patched[
        patched.index("    def _validate_full_offload_streaming_adam(") : patched.index(
            "    def step(self, closure=None):"
        )
    ]
    assert "streamed Adam D2H copy failed at " in adam_source
    assert "streamed Adam D2H event recording failed at " in adam_source
    assert "streamed Adam D2H synchronize failed at " in adam_source
    assert "streamed Adam update failed at " in adam_source
    assert "streamed Adam H2D scheduling failed at " in adam_source
    assert "streamed Adam H2D completion event recording failed at " in adam_source
    assert "streamed Adam final H2D synchronize failed at " in adam_source
    assert "parameter_group={group_index} chunk={chunk_index}" in adam_source
    assert "last_context = (group_index, chunk_index)" in adam_source
    assert "raise RuntimeError(" in adam_source
    assert ") from exc" in adam_source
    assert adam_source.count("scratch.copy_(") == 1
    assert adam_source.count("self._d2h_stream.record_event()") == 1
    assert adam_source.count("event.synchronize()") == 1
    assert adam_source.count("adam_step_chunk_(") == 1
    assert adam_source.count("orig_chunk.copy_(cpu_chunk, non_blocking=True)") == 1
    stream_source = adam_source[
        adam_source.index(
            "    def _stream_full_offload_adam_step(self, closure=None):"
        ) :
    ]
    assert stream_source.count("self._raise_streamed_adam_transfer_error(") == 7


def test_streamed_adam_transfer_error_quiesces_both_streams(
    tmp_path,
) -> None:
    optimizer_path, patched = _patched_hybrid_optimizer_source(tmp_path)
    namespace: dict[str, object] = {}
    # Test executes generated patch source in an isolated namespace.
    exec(  # noqa: S102 -- test executes generated source
        compile(patched, str(optimizer_path), "exec"), namespace
    )
    optimizer_type = namespace["HybridDeviceOptimizer"]
    instance = optimizer_type.__new__(optimizer_type)
    calls = []

    class FakeStream:
        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail

        def synchronize(self):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} quiesce failed")

    instance._d2h_stream = FakeStream("d2h", fail=True)
    instance._h2d_stream = FakeStream("h2d")
    cause = ValueError("original transfer failure")
    message = "streamed Adam D2H copy failed at parameter_group=2 chunk=5"

    with pytest.raises(RuntimeError, match=message) as exc_info:
        instance._raise_streamed_adam_transfer_error(message, cause)

    assert exc_info.value.__cause__ is cause
    assert calls == ["d2h", "h2d"]


def test_streamed_adam_final_sync_quiesces_other_stream_without_retry(
    tmp_path,
) -> None:
    optimizer_path, patched = _patched_hybrid_optimizer_source(tmp_path)
    namespace: dict[str, object] = {}
    # Test executes generated patch source in an isolated namespace.
    exec(  # noqa: S102 -- test executes generated source
        compile(patched, str(optimizer_path), "exec"), namespace
    )
    optimizer_type = namespace["HybridDeviceOptimizer"]
    instance = optimizer_type.__new__(optimizer_type)
    calls = []

    class FakeStream:
        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail

        def synchronize(self):
            calls.append(self.name)
            if self.fail:
                raise ValueError(f"{self.name} final sync failed")

    instance._d2h_stream = FakeStream("d2h")
    instance._h2d_stream = FakeStream("h2d", fail=True)
    message = (
        "streamed Adam final H2D synchronize failed at " "parameter_group=2 chunk=5"
    )

    cause = None
    try:
        instance._h2d_stream.synchronize()
    except ValueError as error:
        cause = error
        with pytest.raises(RuntimeError, match=message) as exc_info:
            instance._raise_streamed_adam_transfer_error(
                message,
                cause,
                failed_stream=instance._h2d_stream,
            )

    assert exc_info.value.__cause__ is cause
    assert calls == ["h2d", "d2h"]


def test_streamed_adam_dispatch_precedes_sgd_and_preserves_upstream(tmp_path) -> None:
    optimizer = (
        tmp_path
        / "megatron"
        / "core"
        / "optimizer"
        / "cpu_offloading"
        / "hybrid_optimizer.py"
    )
    _write_hybrid_optimizer_fixture(optimizer)
    assert patcher.patch_hybrid_optimizer_streaming_adam(str(tmp_path))
    assert patcher.patch_hybrid_optimizer_streaming_sgd(str(tmp_path))

    patched = optimizer.read_text()
    step_source = patched[patched.index("    def step(self, closure=None):") :]
    adam_dispatch = 'if streamed_optimizer_mode == "adam":'
    sgd_dispatch = 'if streamed_optimizer_mode in (None, "sgd"):'
    assert adam_dispatch in step_source
    assert sgd_dispatch in step_source
    assert step_source.index(adam_dispatch) < step_source.index(sgd_dispatch)
    assert "if self._can_stream_full_offload_sgd():" in step_source
    assert "_sync_hdo_param_groups_to_sub_optimizers()" in step_source


@pytest.mark.parametrize(
    "partial_marker",
    [
        "def _stream_full_offload_adam_step(self, closure=None):\n        pass\n",
        "ATOMRL_DSV4_CAPABILITY_STREAMED_ADAM = True\n",
        "from atomrl.engine.training.streamed_adam import AdamChunkOptions\n",
    ],
)
def test_streamed_adam_patch_rejects_partial_patch_without_writing(
    tmp_path, partial_marker
) -> None:
    optimizer = (
        tmp_path
        / "megatron"
        / "core"
        / "optimizer"
        / "cpu_offloading"
        / "hybrid_optimizer.py"
    )
    _write_hybrid_optimizer_fixture(optimizer)
    partial = optimizer.read_text().replace(
        "class HybridDeviceOptimizer:\n",
        partial_marker + "\nclass HybridDeviceOptimizer:\n",
    )
    optimizer.write_text(partial)

    with pytest.raises(RuntimeError, match="partial.*streamed Adam"):
        patcher.patch_hybrid_optimizer_streaming_adam(str(tmp_path))

    assert optimizer.read_text() == partial


def test_streamed_adam_patch_upgrades_known_fp32_only_version(tmp_path) -> None:
    optimizer, current = _patched_hybrid_optimizer_source(tmp_path)
    legacy = current.replace(
        "        moment_dtype = getattr(\n"
        '            self, "_lumen_streamed_adam_moment_dtype", torch.float32\n'
        "        )\n"
        "        if moment_dtype not in (torch.float32, torch.bfloat16):\n"
        "            raise RuntimeError(\n"
        '                "_lumen_streamed_adam_moment_dtype must be float32 or bfloat16"\n'
        "            )\n",
        "",
        1,
    )
    legacy = legacy.replace(
        "or tensor.dtype != moment_dtype", "or tensor.dtype != torch.float32", 1
    )
    legacy = legacy.replace(
        "initialize_adam_state(cpu_param, moment_dtype=moment_dtype)",
        "initialize_adam_state(cpu_param)",
        1,
    )
    legacy = legacy.replace(
        "        moment_dtype = getattr(\n"
        '            self, "_lumen_streamed_adam_moment_dtype", torch.float32\n'
        "        )\n",
        "",
        1,
    )
    legacy = legacy.replace(
        "        staging_rows = 4 if moment_dtype == torch.bfloat16 else 2\n",
        "",
        1,
    )
    legacy = legacy.replace("(staging_rows, chunk_numel)", "(2, chunk_numel)")
    workspace_block = (
        "                    exp_avg_workspace=(\n"
        "                        staging[2, : exp_avg_chunk.numel()]\n"
        "                        if moment_dtype == torch.bfloat16\n"
        "                        else None\n"
        "                    ),\n"
        "                    exp_avg_sq_workspace=(\n"
        "                        staging[3, : exp_avg_sq_chunk.numel()]\n"
        "                        if moment_dtype == torch.bfloat16\n"
        "                        else None\n"
        "                    ),\n"
    )
    legacy = legacy.replace(workspace_block, "", 1)
    optimizer.write_text(legacy)

    assert patcher.patch_hybrid_optimizer_streaming_adam(str(tmp_path))
    upgraded = optimizer.read_text()
    assert patcher._streamed_adam_patch_state(upgraded) == "complete"
    assert "_lumen_streamed_adam_moment_dtype" in upgraded
    assert "exp_avg_workspace=" in upgraded


def test_streamed_adam_patch_repairs_moment_dtype_ordering(tmp_path) -> None:
    optimizer, current = _patched_hybrid_optimizer_source(tmp_path)
    method_start = current.index(
        "    def _stream_full_offload_adam_step(self, closure=None):"
    )
    method_end = current.index("\n    def step(self, closure=None):", method_start)
    method = current[method_start:method_end]
    moment_setup = (
        "        moment_dtype = getattr(\n"
        '            self, "_lumen_streamed_adam_moment_dtype", torch.float32\n'
        "        )\n"
    )
    method = method.replace(moment_setup, "", 1)
    method = method.replace(
        "        staging_rows = 4 if moment_dtype == torch.bfloat16 else 2\n",
        moment_setup
        + "        staging_rows = 4 if moment_dtype == torch.bfloat16 else 2\n",
        1,
    )
    optimizer.write_text(current[:method_start] + method + current[method_end:])

    assert patcher.patch_hybrid_optimizer_streaming_adam(str(tmp_path))
    repaired = optimizer.read_text()
    repaired_method = repaired[
        repaired.index(
            "    def _stream_full_offload_adam_step(self, closure=None):"
        ) : repaired.index("\n    def step(self, closure=None):")
    ]
    assert repaired_method.index(moment_setup) < repaired_method.index(
        "initialize_adam_state(cpu_param, moment_dtype=moment_dtype)"
    )


def test_patch_transformer_config_admits_sqrtsoftplus_idempotently(tmp_path) -> None:
    config = tmp_path / "megatron" / "core" / "transformer" / "transformer_config.py"
    config.parent.mkdir(parents=True)
    config.write_text(
        'moe_router_score_function: Literal["softmax", "sigmoid"] = "softmax"\n'
    )

    assert patcher.patch_transformer_config(str(tmp_path))
    assert 'Literal["softmax", "sigmoid", "sqrtsoftplus"]' in config.read_text()
    assert not patcher.patch_transformer_config(str(tmp_path))


def test_patch_transformer_config_allows_sqrtsoftplus_expert_bias(tmp_path) -> None:
    config = tmp_path / "megatron" / "core" / "transformer" / "transformer_config.py"
    config.parent.mkdir(parents=True)
    config.write_text("""
moe_router_score_function: Literal["softmax", "sigmoid"] = "softmax"

if (
    self.moe_router_enable_expert_bias
    and self.moe_router_score_function != "sigmoid"
):
    raise ValueError("Expert bias only supports sigmoid")
""")

    assert patcher.patch_transformer_config(str(tmp_path))
    patched = config.read_text()
    assert (
        'self.moe_router_score_function not in ("sigmoid", "sqrtsoftplus")' in patched
    )
    assert not patcher.patch_transformer_config(str(tmp_path))


def test_patch_transformer_config_dsa_section_is_fully_idempotent(tmp_path) -> None:
    config = tmp_path / "megatron" / "core" / "transformer" / "transformer_config.py"
    config.parent.mkdir(parents=True)
    config.write_text('''
@dataclass
class TransformerConfig:
    experimental_attention_variant: Optional[
        Literal['gated_delta_net', 'dsa']
    ] = None

    ####################
    # DSA
    ####################
    dsa_indexer_n_heads: int = 64
    """Number of DSA indexer heads."""
    dsa_indexer_head_dim: int = 128
    """Dimension of each DSA indexer head."""
    dsa_indexer_topk: int = 2048
    """Number of selected DSA tokens."""

    moe_router_score_function: Literal["softmax", "sigmoid"] = "softmax"

    def __post_init__(self):
        if self.experimental_attention_variant == "dsa":
            self.dsa_mode = True
''')

    assert patcher.patch_transformer_config(str(tmp_path))
    first_patch = config.read_text()
    assert first_patch.count("# DSV4") == 1
    assert first_patch.count("dsv4_mode: bool = False") == 1
    assert first_patch.count('if self.experimental_attention_variant == "dsv4":') == 1
    assert "sqrtsoftplus" in first_patch

    assert not patcher.patch_transformer_config(str(tmp_path))
    assert config.read_text() == first_patch


def _write_unpatched_moe_routing(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""
import torch


def compute_topk(scores, topk, num_groups=None, group_topk=None):
    return torch.topk(scores, k=topk, dim=1)


def topk_routing_with_score_function(
    logits, topk, num_groups=None, group_topk=None, scaling_factor=None,
    score_function="softmax", expert_bias=None, router_replay=None,
):
    if score_function == "softmax":
        scores = torch.softmax(
            logits, dim=-1, dtype=torch.float32
        ).type_as(logits)
        probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(
                scores_for_routing, topk, num_groups, group_topk
            )
            scores = torch.gather(
                scores, dim=1, index=top_indices
            ).type_as(logits)
        else:
            scores, top_indices = compute_topk(
                scores, topk, num_groups, group_topk
            )
        probs = (
            scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
            if topk > 1
            else scores
        )
    else:
        raise ValueError(f"Invalid score_function: {score_function}")
    if scaling_factor:
        probs = probs * scaling_factor
    routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
    routing_map = torch.zeros_like(logits).int().scatter(
        1, top_indices, 1
    ).bool()
    return routing_probs, routing_map
""")


def test_patch_moe_utils_adds_unbiased_sqrtsoftplus_routing(tmp_path) -> None:
    moe_utils = tmp_path / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py"
    _write_unpatched_moe_routing(moe_utils)
    (moe_utils.parent / "router.py").write_text("class TopKRouter:\n    pass\n")

    assert patcher.patch_moe_router_score_function(str(tmp_path))

    patched = moe_utils.read_text()
    assert 'elif score_function in ("sigmoid", "sqrtsoftplus"):' in patched
    assert "torch.nn.functional.softplus(logits.float()).sqrt()" in patched
    assert "torch.sigmoid(logits.float()).type_as(logits)" in patched
    assert "scores_for_routing = scores + expert_bias" in patched
    assert "compute_topk(scores_for_routing, topk, num_groups, group_topk)" in patched
    assert (
        "scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)"
        in patched
    )
    assert (
        "probs = scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-20) "
        "if topk > 1 else scores" in patched
    )
    assert "probs = probs * scaling_factor" in patched
    assert 'if score_function == "softmax":' in patched
    assert 'if score_function == "sigmoid":' in patched
    assert not patcher.patch_moe_router_score_function(str(tmp_path))


def test_patched_sigmoid_is_bf16_and_scatter_compatible(tmp_path) -> None:
    moe_utils = tmp_path / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py"
    _write_unpatched_moe_routing(moe_utils)
    assert patcher.patch_moe_router_score_function(str(tmp_path))

    namespace: dict[str, object] = {}
    # Test executes generated patch source in an isolated namespace.
    exec(  # noqa: S102 -- test executes generated source
        compile(moe_utils.read_text(), str(moe_utils), "exec"), namespace
    )
    route = namespace["topk_routing_with_score_function"]

    logits = torch.tensor([[0.0, 4.0]], dtype=torch.bfloat16)
    expert_bias = torch.tensor([10.0, 0.0], dtype=torch.float32)
    routing_probs, routing_map = route(
        logits,
        1,
        score_function="sigmoid",
        expert_bias=expert_bias,
    )

    assert routing_map.tolist() == [[True, False]]
    assert routing_probs.dtype == torch.bfloat16
    expected = torch.sigmoid(logits[:, :1].float()).to(logits.dtype)
    torch.testing.assert_close(routing_probs[:, :1], expected)
    torch.testing.assert_close(routing_probs[:, 1:], torch.zeros_like(expected))

    fp32_logits = logits.float()
    sqrt_routing_probs, _ = route(
        fp32_logits,
        1,
        score_function="sqrtsoftplus",
        expert_bias=expert_bias,
    )
    assert sqrt_routing_probs.dtype == fp32_logits.dtype


def test_patch_moe_router_falls_back_to_compatible_router_module(tmp_path) -> None:
    router = tmp_path / "megatron" / "core" / "transformer" / "moe" / "router.py"
    _write_unpatched_moe_routing(router)

    assert patcher.patch_moe_router_score_function(str(tmp_path))
    assert "sqrtsoftplus" in router.read_text()
    assert not patcher.patch_moe_router_score_function(str(tmp_path))


def test_patch_moe_router_prefers_moe_utils_definition(tmp_path) -> None:
    moe_dir = tmp_path / "megatron" / "core" / "transformer" / "moe"
    moe_utils = moe_dir / "moe_utils.py"
    router = moe_dir / "router.py"
    _write_unpatched_moe_routing(moe_utils)
    _write_unpatched_moe_routing(router)

    assert patcher.patch_moe_router_score_function(str(tmp_path))
    assert "sqrtsoftplus" in moe_utils.read_text()
    assert "sqrtsoftplus" not in router.read_text()


def test_patch_moe_router_returns_false_when_definition_is_absent(tmp_path) -> None:
    moe_dir = tmp_path / "megatron" / "core" / "transformer" / "moe"
    moe_dir.mkdir(parents=True)
    (moe_dir / "moe_utils.py").write_text("def unrelated_utility():\n    pass\n")
    (moe_dir / "router.py").write_text("class TopKRouter:\n    pass\n")

    assert not patcher.patch_moe_router_score_function(str(tmp_path))


def test_patch_checkpoint_recompute_scopes_backward_replay_and_restores(
    tmp_path,
) -> None:
    random_py = tmp_path / "megatron" / "core" / "tensor_parallel" / "random.py"
    random_py.parent.mkdir(parents=True)
    random_py.write_text("""
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def backward(ctx, *args):
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)
        return outputs
""")

    assert patcher.patch_checkpoint_router_replay(str(tmp_path))

    patched = random_py.read_text()
    assert "RouterReplay.global_router_replay_instances" in patched
    assert "instance.router_replay_action" in patched
    assert "instance.set_router_replay_action(" in patched
    assert "RouterReplayAction.REPLAY_BACKWARD" in patched
    assert "outputs = ctx.run_function(*detached_inputs)" in patched
    assert "finally:" in patched
    assert "instance.clear_router_replay_action()" in patched
    assert "instance.set_router_replay_action(prior_action)" in patched
    assert "set_global_router_replay_action" not in patched
    assert not patcher.patch_checkpoint_router_replay(str(tmp_path))


def test_checkpoint_patch_stamps_importable_runtime_capability(tmp_path) -> None:
    random_py = tmp_path / "megatron" / "core" / "tensor_parallel" / "random.py"
    _write_checkpoint_runtime_fixture(random_py)

    assert patcher.patch_checkpoint_router_replay(str(tmp_path))
    first_patch = random_py.read_text()
    namespace: dict[str, object] = {}
    # Test executes generated patch source in an isolated namespace.
    exec(  # noqa: S102 -- test executes generated source
        compile(first_patch, str(random_py), "exec"), namespace
    )

    assert namespace["ATOMRL_R3_CAPABILITY_CHECKPOINT_REPLAY_BACKWARD"] is True
    assert not patcher.patch_checkpoint_router_replay(str(tmp_path))
    assert random_py.read_text() == first_patch


def test_router_patch_does_not_cross_into_next_function(tmp_path) -> None:
    moe_utils = tmp_path / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py"
    moe_utils.parent.mkdir(parents=True)
    original = """
def topk_routing_with_score_function(logits, topk, score_function="softmax"):
    if score_function == "softmax":
        return logits
    return logits

def unrelated(logits, topk, score_function="softmax"):
    if score_function == "softmax":
        return logits
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
        probs, top_indices = compute_topk(scores, topk)
    else:
        raise ValueError(score_function)
    return probs, top_indices
"""
    moe_utils.write_text(original)

    assert not patcher.patch_moe_router_score_function(str(tmp_path))
    assert moe_utils.read_text() == original


def test_router_patch_rejects_malformed_target_without_cross_function_write(
    tmp_path,
) -> None:
    moe_utils = tmp_path / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py"
    moe_utils.parent.mkdir(parents=True)
    original = """
def topk_routing_with_score_function(logits, topk, score_function="softmax"):
    if score_function == "softmax":
        return logits
    elif score_function == "sigmoid":
        return torch.sigmoid(logits)
    return logits

def unrelated(score_function):
    if score_function:
        return True
    else:
        raise ValueError(score_function)
"""
    moe_utils.write_text(original)

    assert not patcher.patch_moe_router_score_function(str(tmp_path))
    assert moe_utils.read_text() == original


def test_sqrtsoftplus_exact_golden_bias_normalization_and_scaling(tmp_path) -> None:
    moe_utils = tmp_path / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py"
    _write_unpatched_moe_routing(moe_utils)
    assert patcher.patch_moe_router_score_function(str(tmp_path))
    namespace: dict[str, object] = {}
    # Test executes generated patch source in an isolated namespace.
    exec(  # noqa: S102 -- test executes generated source
        compile(moe_utils.read_text(), str(moe_utils), "exec"), namespace
    )
    route = namespace["topk_routing_with_score_function"]

    logits = torch.tensor([[-2.0, 1.0, 3.0]], dtype=torch.float32)
    unbiased_scores = torch.sqrt(torch.nn.functional.softplus(logits))
    unbiased_ids = torch.topk(unbiased_scores, 2, dim=1).indices
    expert_bias = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float32)
    routing_probs, routing_map = route(
        logits,
        2,
        score_function="sqrtsoftplus",
        expert_bias=expert_bias,
        scaling_factor=1.5,
    )

    selected_ids = routing_map.nonzero(as_tuple=False)[:, 1].sort().values
    assert selected_ids.tolist() == [0, 2]
    assert set(selected_ids.tolist()) != set(unbiased_ids[0].tolist())
    expected_selected = unbiased_scores[:, [0, 2]]
    expected_selected = (
        expected_selected
        / expected_selected.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        * 1.5
    )
    torch.testing.assert_close(routing_probs[:, [0, 2]], expected_selected)
    assert routing_probs[0, 1].item() == 0.0

    underflow_logits = torch.full((1, 2), -1000.0, dtype=torch.float32)
    clamped_probs, _ = route(
        underflow_logits,
        2,
        score_function="sqrtsoftplus",
        scaling_factor=1.5,
    )
    assert torch.isfinite(clamped_probs).all()
    torch.testing.assert_close(clamped_probs, torch.zeros_like(clamped_probs))


def _write_checkpoint_runtime_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""
import torch

class CheckpointFunction(object):
    @staticmethod
    def backward(ctx, *args):
        detached_inputs = args
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)
        return outputs
""")


def _load_patched_checkpoint(tmp_path, monkeypatch):
    random_py = tmp_path / "megatron" / "core" / "tensor_parallel" / "random.py"
    _write_checkpoint_runtime_fixture(random_py)
    assert patcher.patch_checkpoint_router_replay(str(tmp_path))

    replay_backward = object()

    class ReplayAction:
        REPLAY_BACKWARD = replay_backward

    class Replay:
        global_router_replay_instances: typing.ClassVar = []

    module_names = (
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.moe",
    )
    for module_name in module_names:
        monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    replay_module_name = "megatron.core.transformer.moe.router_replay"
    replay_module = ModuleType(replay_module_name)
    replay_module.RouterReplay = Replay
    replay_module.RouterReplayAction = ReplayAction
    monkeypatch.setitem(sys.modules, replay_module_name, replay_module)

    namespace: dict[str, object] = {}
    # Test executes generated patch source in an isolated namespace.
    exec(  # noqa: S102 -- test executes generated source
        compile(random_py.read_text(), str(random_py), "exec"), namespace
    )
    return namespace["CheckpointFunction"], Replay, replay_backward


class _ReplayInstance:
    def __init__(self, action):
        self.router_replay_action = action

    def set_router_replay_action(self, action):
        self.router_replay_action = action

    def clear_router_replay_action(self):
        self.router_replay_action = None


def _write_router_replay_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""
from enum import Enum
import torch

class RouterReplayAction(Enum):
    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"

class RouterReplay:
    global_router_replay_instances = []

    def __init__(self):
        self.target_topk_idx = None
        self.recorded_topk_idx = None
        self.router_replay_action = None
        self.replay_backward_list = []
        RouterReplay.global_router_replay_instances.append(self)

    def set_target_indices(self, topk_indices):
        self.target_topk_idx = topk_indices
        self.replay_backward_list.append(topk_indices)

    def clear_indices(self):
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.replay_backward_list = []

    def set_router_replay_action(self, action):
        self.router_replay_action = action

    def get_replay_topk(self, scores, topk, num_groups=None, group_topk=None,
                        default_compute_topk=None):
        if self.router_replay_action == RouterReplayAction.REPLAY_FORWARD:
            top_indices = self.target_topk_idx
            top_indices = top_indices.to(scores.device)
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        elif self.router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
            top_indices = self.replay_backward_list.pop(0)
            top_indices = top_indices.to(scores.device)
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        return default_compute_topk(scores, topk, num_groups, group_topk)
""")


def _load_patched_router_replay(tmp_path):
    router_replay = (
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "router_replay.py"
    )
    _write_router_replay_fixture(router_replay)
    assert patcher.patch_router_replay_diagnostics(str(tmp_path))
    first_patch = router_replay.read_text()
    assert not patcher.patch_router_replay_diagnostics(str(tmp_path))
    assert router_replay.read_text() == first_patch
    namespace: dict[str, object] = {}
    # Test executes generated patch source in an isolated namespace.
    exec(  # noqa: S102 -- test executes generated source
        compile(first_patch, str(router_replay), "exec"), namespace
    )
    return namespace["RouterReplay"], namespace["RouterReplayAction"]


def test_router_replay_diagnostics_execute_success_without_cloning(tmp_path) -> None:
    replay_type, action = _load_patched_router_replay(tmp_path)
    replay = replay_type()
    first = torch.tensor([[0, 1], [1, 0]])
    second = torch.tensor([[2, 1], [0, 2]])
    scores = torch.arange(6, dtype=torch.float32).view(2, 3)

    replay.reset_recompute_diagnostics()
    for indices in (first, second):
        replay.set_target_indices(indices)
        replay.set_router_replay_action(action.REPLAY_FORWARD)
        _, selected = replay.get_replay_topk(scores, 2)
        assert replay.recompute_forward_indices[-1] is selected
    replay.set_router_replay_action(action.REPLAY_BACKWARD)
    replay.get_replay_topk(scores, 2)
    replay.get_replay_topk(scores, 2)

    assert replay.get_recompute_diagnostics() == (8, 0)
    assert replay.recompute_forward_indices == []


def test_router_replay_patch_stamps_fifo_and_diagnostic_capabilities(
    tmp_path,
) -> None:
    router_replay = (
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "router_replay.py"
    )
    _write_router_replay_fixture(router_replay)

    assert patcher.patch_router_replay_diagnostics(str(tmp_path))
    first_patch = router_replay.read_text()
    namespace: dict[str, object] = {}
    # Test executes generated patch source in an isolated namespace.
    exec(  # noqa: S102 -- test executes generated source
        compile(first_patch, str(router_replay), "exec"), namespace
    )

    assert namespace["ATOMRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO"] is True
    assert namespace["ATOMRL_R3_CAPABILITY_REPLAY_DIAGNOSTICS"] is True
    assert not patcher.patch_router_replay_diagnostics(str(tmp_path))
    assert router_replay.read_text() == first_patch


def test_runtime_capability_stamps_replace_false_values(tmp_path) -> None:
    router_replay = (
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "router_replay.py"
    )
    _write_router_replay_fixture(router_replay)
    assert patcher.patch_router_replay_diagnostics(str(tmp_path))
    router_replay.write_text(
        router_replay.read_text().replace(
            "ATOMRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO = True",
            "ATOMRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO = False",
        )
    )

    assert patcher.patch_router_replay_diagnostics(str(tmp_path))
    patched = router_replay.read_text()
    assert "ATOMRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO = True" in patched
    assert "ATOMRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO = False" not in patched
    assert not patcher.patch_router_replay_diagnostics(str(tmp_path))


def test_router_replay_diagnostics_detect_mismatched_fifo_order(tmp_path) -> None:
    replay_type, action = _load_patched_router_replay(tmp_path)
    replay = replay_type()
    first = torch.tensor([[0, 1]])
    second = torch.tensor([[2, 1]])
    scores = torch.arange(3, dtype=torch.float32).view(1, 3)

    replay.reset_recompute_diagnostics()
    for indices in (first, second):
        replay.set_target_indices(indices)
        replay.set_router_replay_action(action.REPLAY_FORWARD)
        replay.get_replay_topk(scores, 2)
    replay.replay_backward_list[:] = list(reversed(replay.replay_backward_list))
    replay.set_router_replay_action(action.REPLAY_BACKWARD)
    replay.get_replay_topk(scores, 2)
    replay.get_replay_topk(scores, 2)

    assert replay.get_recompute_diagnostics() == (4, 2)


def test_router_replay_marker_only_partial_patch_fails_without_writing(
    tmp_path,
) -> None:
    router_replay = (
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "router_replay.py"
    )
    _write_router_replay_fixture(router_replay)
    partial = router_replay.read_text().replace(
        "        self.target_topk_idx = None\n",
        "        self.target_topk_idx = None\n"
        "        self.recompute_forward_indices = []\n",
    )
    router_replay.write_text(partial)

    with pytest.raises(RuntimeError, match="partial.*diagnostic"):
        patcher.patch_router_replay_diagnostics(str(tmp_path))

    assert router_replay.read_text() == partial


def test_router_replay_malformed_full_patch_fails_without_writing(tmp_path) -> None:
    router_replay = (
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "router_replay.py"
    )
    _write_router_replay_fixture(router_replay)
    assert patcher.patch_router_replay_diagnostics(str(tmp_path))
    malformed = router_replay.read_text().replace(
        "self.recompute_forward_indices.pop(0)",
        "self.recompute_forward_indices[-1]",
    )
    router_replay.write_text(malformed)

    with pytest.raises(RuntimeError, match="partial.*diagnostic"):
        patcher.patch_router_replay_diagnostics(str(tmp_path))

    assert router_replay.read_text() == malformed


def test_checkpoint_runtime_restores_prior_actions_after_success(
    tmp_path, monkeypatch
) -> None:
    checkpoint, replay, replay_backward = _load_patched_checkpoint(
        tmp_path, monkeypatch
    )
    instances = [_ReplayInstance(None), _ReplayInstance("forward")]
    replay.global_router_replay_instances = instances

    def run_function(value):
        assert [item.router_replay_action for item in instances] == [
            replay_backward,
            replay_backward,
        ]
        return value + 1

    result = checkpoint.backward(
        SimpleNamespace(run_function=run_function), torch.tensor(2.0)
    )
    assert result.item() == 3.0
    assert [item.router_replay_action for item in instances] == [None, "forward"]


def test_checkpoint_runtime_restores_prior_actions_after_exception(
    tmp_path, monkeypatch
) -> None:
    checkpoint, replay, replay_backward = _load_patched_checkpoint(
        tmp_path, monkeypatch
    )
    instances = [_ReplayInstance("record"), _ReplayInstance(None)]
    replay.global_router_replay_instances = instances

    def run_function(_value):
        assert all(item.router_replay_action is replay_backward for item in instances)
        raise RuntimeError("recompute failed")

    with pytest.raises(RuntimeError, match="recompute failed"):
        checkpoint.backward(
            SimpleNamespace(run_function=run_function), torch.tensor(2.0)
        )
    assert [item.router_replay_action for item in instances] == ["record", None]


_OPTIONAL_PATCHERS = (
    "patch_transformer_block",
    "patch_transformer_layer",
    "patch_eav_specs",
    "patch_tp_layers",
    "patch_tp_copy_fp32_gradient_reduce",
    "patch_hybrid_optimizer_disable_foreach",
    "patch_hybrid_optimizer_streaming_adam",
    "patch_hybrid_optimizer_streaming_sgd",
    "patch_distrib_optimizer_fp32_detach",
    "patch_distrib_optimizer_grad_copy",
    "patch_distrib_optimizer_hdo_checkpoint",
)


def _disable_optional_patchers(monkeypatch) -> None:
    for name in _OPTIONAL_PATCHERS:
        monkeypatch.setattr(patcher, name, lambda _root: False)


def _write_required_main_fixtures(tmp_path) -> tuple[Path, Path, Path]:
    config = tmp_path / "megatron" / "core" / "transformer" / "transformer_config.py"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        'moe_router_score_function: Literal["softmax", "sigmoid"] = "softmax"\n'
    )
    moe_utils = tmp_path / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py"
    _write_unpatched_moe_routing(moe_utils)
    random_py = tmp_path / "megatron" / "core" / "tensor_parallel" / "random.py"
    _write_checkpoint_runtime_fixture(random_py)
    _write_router_replay_fixture(
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "router_replay.py"
    )
    return config, moe_utils, random_py


def test_main_validates_required_patches_and_reports_actual_router_source(
    tmp_path, monkeypatch, capsys
) -> None:
    _disable_optional_patchers(monkeypatch)
    _write_required_main_fixtures(tmp_path)

    patcher.main(str(tmp_path))
    first_output = capsys.readouterr().out
    assert "transformer/moe/moe_utils.py" in first_output
    assert "transformer/moe/router.py" not in first_output

    patcher.main(str(tmp_path))
    second_output = capsys.readouterr().out
    assert "transformer/moe/moe_utils.py" in second_output


def test_main_registers_streamed_adam_between_foreach_and_sgd(
    tmp_path, monkeypatch
) -> None:
    _write_required_main_fixtures(tmp_path)
    calls = []
    for name in _OPTIONAL_PATCHERS:
        monkeypatch.setattr(
            patcher,
            name,
            lambda _root, patch_name=name: calls.append(patch_name) or False,
        )

    patcher.main(str(tmp_path))

    assert calls.index("patch_hybrid_optimizer_disable_foreach") < calls.index(
        "patch_hybrid_optimizer_streaming_adam"
    )
    assert calls.index("patch_hybrid_optimizer_streaming_adam") < calls.index(
        "patch_hybrid_optimizer_streaming_sgd"
    )


def test_required_validation_rejects_config_substring_false_positive(
    tmp_path, monkeypatch
) -> None:
    _disable_optional_patchers(monkeypatch)
    config, _, _ = _write_required_main_fixtures(tmp_path)
    assert patcher.patch_moe_router_score_function(str(tmp_path))
    assert patcher.patch_checkpoint_router_replay(str(tmp_path))
    config.write_text(
        "moe_router_score_function: "
        'Literal["softmax", "sigmoid", "sqrtsoftplus_extra"] = "softmax"\n'
    )

    with pytest.raises(RuntimeError, match="sqrtsoftplus"):
        patcher._validate_required_patches(str(tmp_path))


def test_main_rejects_absent_router_score_config_field(tmp_path, monkeypatch) -> None:
    _disable_optional_patchers(monkeypatch)
    config, _, _ = _write_required_main_fixtures(tmp_path)
    config.write_text("hidden_size: int = 4096\n")

    with pytest.raises(RuntimeError, match="moe_router_score_function"):
        patcher.main(str(tmp_path))


def test_main_reports_missing_required_files_as_runtime_error(
    tmp_path, monkeypatch
) -> None:
    _disable_optional_patchers(monkeypatch)

    with pytest.raises(RuntimeError, match="Required ROCm Megatron patch inputs"):
        patcher.main(str(tmp_path))


def test_main_rejects_malformed_router_anchor_without_writing(
    tmp_path, monkeypatch
) -> None:
    _disable_optional_patchers(monkeypatch)
    _, moe_utils, _ = _write_required_main_fixtures(tmp_path)
    original = """
def topk_routing_with_score_function(logits, topk, score_function="softmax"):
    return logits

def unrelated(score_function):
    if score_function == "softmax":
        return 1
    elif score_function == "sigmoid":
        return 2
    else:
        raise ValueError(score_function)
"""
    moe_utils.write_text(original)

    with pytest.raises(RuntimeError, match="topk_routing_with_score_function"):
        patcher.main(str(tmp_path))
    assert moe_utils.read_text() == original


def test_main_rejects_missing_checkpoint_marker(tmp_path, monkeypatch) -> None:
    _disable_optional_patchers(monkeypatch)
    _, _, random_py = _write_required_main_fixtures(tmp_path)
    random_py.write_text("""
class CheckpointFunction(object):
    @staticmethod
    def backward(ctx, *args):
        return ctx.run_function(*args)
""")

    with pytest.raises(RuntimeError, match="CheckpointFunction.backward"):
        patcher.main(str(tmp_path))


def test_main_rejects_partial_router_diagnostics_without_rewriting(
    tmp_path, monkeypatch
) -> None:
    _disable_optional_patchers(monkeypatch)
    config, moe_utils, random_py = _write_required_main_fixtures(tmp_path)
    router_replay = (
        tmp_path / "megatron" / "core" / "transformer" / "moe" / "router_replay.py"
    )
    partial = router_replay.read_text().replace(
        "        self.target_topk_idx = None\n",
        "        self.target_topk_idx = None\n"
        "        self.recompute_forward_indices = []\n",
    )
    router_replay.write_text(partial)
    originals = {
        path: path.read_text() for path in (config, moe_utils, random_py, router_replay)
    }

    with pytest.raises(RuntimeError, match="partial.*diagnostic"):
        patcher.main(str(tmp_path))

    assert {
        path: path.read_text() for path in (config, moe_utils, random_py, router_replay)
    } == originals
