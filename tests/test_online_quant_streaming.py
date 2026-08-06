# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""CPU tests for online-quant streaming and completion tracking."""

import contextlib
import os
import sys
import threading
import unittest
from typing import ClassVar
from unittest import mock

import torch
from torch import nn

import atom.model_loader.loader as loader_module
import atom.model_loader.online_quant_streaming as streaming_module
from atom.model_loader.loading_core import load_weights_into_model
from atom.model_loader.online_quant_streaming import OnlineQuantStreamer

HIDDEN = 8
INTER = 4


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


class StreamLinear(nn.Module):
    """The parts of `LinearBase` the streamer talks to."""

    def __init__(self, out_features: int, in_features: int, stream: bool):
        super().__init__()
        self._stream_online_quant = stream
        self._load_device = torch.device("cpu")
        device = "meta" if stream else None
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features, device=device), requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features, device=device), requires_grad=False
        )
        for p in (self.weight, self.bias):
            p.weight_loader = self.weight_loader
        self.post_process_calls = 0

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)

    def process_weights_after_loading(self) -> None:
        self.post_process_calls += 1


class DoubleCopyLinear(StreamLinear):
    """A loader that copies into its destination twice."""

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)
        param.data.copy_(param.data)


class HostHostileLinear(StreamLinear):
    """Simulate a loader without a host kernel."""

    def __init__(self, out_features: int, in_features: int, stream: bool):
        super().__init__(out_features, in_features, stream)
        self.refusals = 0

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if getattr(self, "_stream_on_host", False):
            self.refusals += 1
            raise NotImplementedError("no kernel for host storage")
        param.data.copy_(loaded_weight)


class StreamMoE(nn.Module):
    """One fused MoE parameter pair fed one (expert, shard) tensor at a time."""

    def __init__(self, num_experts: int, stream: bool):
        super().__init__()
        self.num_experts = num_experts
        self.num_batched_experts = num_experts
        self._stream_online_quant = stream
        self._load_device = torch.device("cpu")
        device = "meta" if stream else None
        self.w13_weight = nn.Parameter(
            torch.zeros(num_experts, 2 * INTER, HIDDEN, device=device),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(num_experts, HIDDEN, INTER, device=device), requires_grad=False
        )
        for p in (self.w13_weight, self.w2_weight):
            p.weight_loader = self.weight_loader
        self.post_process_calls = 0
        self.loader_calls = 0
        self.stage_calls = 0

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        self.loader_calls += 1
        self._copy_expert_shard(
            param.data[expert_id],
            loaded_weight,
            shard_id,
        )

    @staticmethod
    def _copy_expert_shard(
        expert_data: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: str,
    ) -> None:
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
            return
        offset = 0 if shard_id == "w1" else INTER
        expert_data[offset : offset + INTER].copy_(loaded_weight)

    def process_weights_after_loading(self) -> None:
        self.post_process_calls += 1

    def stage_expert_weight(
        self,
        param,
        staging,
        loaded_weight,
        local_expert_id,
        shard_id,
        weight_name,
    ):
        self.stage_calls += 1
        self._copy_expert_shard(
            staging[local_expert_id],
            loaded_weight,
            shard_id,
        )
        return True

    def expected_batched_arrivals(self, param):
        if param is self.w13_weight:
            return self.num_batched_experts * 2
        if param is self.w2_weight:
            return self.num_batched_experts
        return None

    @staticmethod
    def _map_global_expert_id_to_local_expert_id(expert_id):
        return expert_id

    def is_batched_expert_slot(self, local_expert_id):
        return local_expert_id < self.num_batched_experts

    @staticmethod
    def batched_expert_region_numel(param, local_expert_id, shard_id):
        expert_data = param.data[local_expert_id]
        if shard_id == "w2":
            return expert_data.numel()
        return expert_data[:INTER].numel()


class SharedExpertStreamMoE(StreamMoE):
    """Treat the final slot as a separately loaded fused shared expert."""

    def __init__(self, num_experts: int, stream: bool):
        super().__init__(num_experts, stream)
        self.num_batched_experts = num_experts - 1


class ExpertParallelStreamMoE(StreamMoE):
    """Rank-zero EP storage for half of the checkpoint experts."""

    def __init__(self, num_experts: int, stream: bool):
        super().__init__(num_experts // 2, stream)

    @staticmethod
    def _map_global_expert_id_to_local_expert_id(expert_id):
        return expert_id if expert_id < NUM_EXPERTS // 2 else -1

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        local_expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if local_expert_id == -1:
            return
        self._copy_expert_shard(
            param.data[local_expert_id],
            loaded_weight,
            shard_id,
        )


class DeclineW2StreamMoE(StreamMoE):
    def stage_expert_weight(self, *args, shard_id, **kwargs):
        if shard_id == "w2":
            return False
        return super().stage_expert_weight(*args, shard_id=shard_id, **kwargs)


class DeclineW3StreamMoE(StreamMoE):
    def stage_expert_weight(self, *args, shard_id, **kwargs):
        if shard_id == "w3":
            return False
        return super().stage_expert_weight(*args, shard_id=shard_id, **kwargs)


class _Layer(nn.Module):
    def __init__(self, num_experts: int, stream: bool, linear_cls, moe_cls):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = linear_cls(HIDDEN, HIDDEN, stream)
        self.mlp = nn.Module()
        self.mlp.experts = moe_cls(num_experts, stream)


class StreamModel(nn.Module):
    """Model exposing the hooks `load_weights_into_model` looks for."""

    packed_modules_mapping: ClassVar[dict] = {}
    weights_mapping: ClassVar[dict] = {}
    moe_cls = StreamMoE

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        stream: bool,
        linear_cls=StreamLinear,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            _Layer(num_experts, stream, linear_cls, self.moe_cls)
            for _ in range(num_layers)
        )

    def streamed_modules(self) -> list[nn.Module]:
        return [m for m in self.modules() if getattr(m, "_stream_online_quant", False)]

    def get_expert_mapping(self):
        return [
            (
                "experts.w13_" if weight in ("gate_proj", "up_proj") else "experts.w2_",
                f"experts.{expert_id}.{weight}.",
                expert_id,
                shard,
            )
            for expert_id in range(self.num_experts)
            for shard, weight in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            )
        ]


class SharedExpertStreamModel(StreamModel):
    moe_cls = SharedExpertStreamMoE


class ExpertParallelStreamModel(StreamModel):
    moe_cls = ExpertParallelStreamMoE


class DeclineW2StreamModel(StreamModel):
    moe_cls = DeclineW2StreamMoE


class DeclineW3StreamModel(StreamModel):
    moe_cls = DeclineW3StreamMoE


class FusedExpertStreamModel(StreamModel):
    """A model whose checkpoint stacks all routed experts into one tensor.

    Separate from `StreamModel` so the per-expert tests keep exercising a
    dispatcher with no fused-expert hooks at all.
    """

    def detect_fused_expert_format(self, weight_name: str) -> bool:
        return "experts.gate_up_proj" in weight_name or (
            "experts.down_proj" in weight_name and ".experts." in weight_name
        )

    def get_fused_expert_mapping(self) -> list[tuple[str, str, str]]:
        return [
            ("experts.w13_weight", "experts.gate_up_proj", "w1"),
            ("experts.w2_weight", "experts.down_proj", "w2"),
        ]

    def load_fused_expert_weights(
        self,
        original_name: str,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        """Write every routed expert at once, the way a stacked checkpoint does.

        Deliberately straight into the parameter, bypassing `submit` and hence
        the streamer's arrival count -- which is what the real fused-expert path
        does, and why it needs `materialize_fused_param` to have given the
        parameter real storage first. A meta tensor would take this copy
        silently, so the resulting values are the only evidence it ran.
        """
        params_dict[name].data.copy_(loaded_weight)
        return True


def fused_checkpoint(num_layers: int, num_experts: int):
    """Attention weights per layer plus one stacked tensor per expert matrix."""
    torch.manual_seed(0)
    tensors = {}
    for layer in range(num_layers):
        base = f"model.layers.{layer}"
        tensors[f"{base}.self_attn.o_proj.weight"] = torch.randn(HIDDEN, HIDDEN)
        tensors[f"{base}.self_attn.o_proj.bias"] = torch.randn(HIDDEN)
        tensors[f"{base}.mlp.experts.gate_up_proj"] = torch.randn(
            num_experts, 2 * INTER, HIDDEN
        )
        tensors[f"{base}.mlp.experts.down_proj"] = torch.randn(
            num_experts, HIDDEN, INTER
        )
    return tensors


class HFConfig:
    def __init__(self, num_hidden_layers: int, n_routed_experts: int):
        self.num_hidden_layers = num_hidden_layers
        self.n_routed_experts = n_routed_experts
        self.num_experts = n_routed_experts


def checkpoint(num_layers: int, num_experts: int) -> dict[str, torch.Tensor]:
    """Attention weights and per-expert MoE tensors, in checkpoint order."""
    torch.manual_seed(0)
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(num_layers):
        base = f"model.layers.{layer}"
        tensors[f"{base}.self_attn.o_proj.weight"] = torch.randn(HIDDEN, HIDDEN)
        tensors[f"{base}.self_attn.o_proj.bias"] = torch.randn(HIDDEN)
        for expert_id in range(num_experts):
            p = f"{base}.mlp.experts.{expert_id}"
            tensors[f"{p}.gate_proj.weight"] = torch.randn(INTER, HIDDEN)
            tensors[f"{p}.up_proj.weight"] = torch.randn(INTER, HIDDEN)
            tensors[f"{p}.down_proj.weight"] = torch.randn(HIDDEN, INTER)
    return tensors


def iterator_over(tensors):
    """A weights iterator over a name->tensor mapping or a list of pairs.

    Pairs are what a checkpoint that repeats a name looks like from here, which a
    dict cannot express.
    """
    pairs = list(tensors.items() if isinstance(tensors, dict) else tensors)

    def _iterator(path, disable_mmap, wants=None):
        for name, tensor in pairs:
            if wants is None or wants(name):
                yield name, tensor

    return _iterator


@contextlib.contextmanager
def env_overrides(**values: str):
    previous = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


NUM_LAYERS = 2
NUM_EXPERTS = 4


def run_load(
    *,
    streaming: bool,
    host_staging: bool = True,
    num_threads: int = 1,
    tail_threads: int = 0,
    linear_cls=StreamLinear,
    num_layers: int = NUM_LAYERS,
    tensors=None,
    model_cls=StreamModel,
) -> tuple[StreamModel, OnlineQuantStreamer | None]:
    with env_overrides(
        ATOM_LOADER_NUM_THREADS=str(num_threads),
        ATOM_ONLINE_QUANT_STREAMING="1" if streaming else "0",
        ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING="1" if host_staging else "0",
        ATOM_ONLINE_QUANT_STREAMING_THREADS=str(tail_threads),
    ):
        model = model_cls(num_layers, NUM_EXPERTS, streaming, linear_cls)
        online_quant_streamer = OnlineQuantStreamer.maybe_create(model, None)
        # Stashed rather than returned so the existing two-value unpacking holds;
        # the coverage record is only interesting to a couple of tests.
        model._loaded_weights_record = load_weights_into_model(
            model=model,
            model_name_or_path="<synthetic>",
            hf_config=HFConfig(num_layers, NUM_EXPERTS),
            default_weight_loader=default_weight_loader,
            fuse_shared_expert=lambda *_args, **_kw: False,
            is_rank0=lambda: True,
            weights_iterator=iterator_over(
                checkpoint(num_layers, NUM_EXPERTS) if tensors is None else tensors
            ),
            load_fused_expert_weights_fn=getattr(
                model, "load_fused_expert_weights", None
            ),
            online_quant_streamer=online_quant_streamer,
        )
        if online_quant_streamer is not None:
            online_quant_streamer.replay_stragglers_and_report(True)
    return model, online_quant_streamer


def weights_of(model: StreamModel) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


class StreamingDifferentialTest(unittest.TestCase):
    """Every storage path must leave the same bytes in the parameters."""

    def setUp(self):
        self.baseline, _ = run_load(streaming=False)
        self.expected = weights_of(self.baseline)

    def assert_matches_baseline(self, model: StreamModel):
        actual = weights_of(model)
        self.assertEqual(sorted(self.expected), sorted(actual))
        for name, want in self.expected.items():
            self.assertFalse(actual[name].is_meta, f"{name} left on meta")
            torch.testing.assert_close(
                want, actual[name], rtol=0, atol=0, msg=f"mismatch: {name}"
            )

    def test_host_staging_matches_non_streaming(self):
        model, _ = run_load(streaming=True, host_staging=True)
        self.assert_matches_baseline(model)

    def test_buffered_replay_matches_non_streaming(self):
        model, _ = run_load(streaming=True, host_staging=False)
        self.assert_matches_baseline(model)

    def test_host_staging_with_parallel_walk_matches_non_streaming(self):
        model, _ = run_load(streaming=True, host_staging=True, num_threads=8)
        self.assert_matches_baseline(model)

    def test_host_staging_with_tail_workers_matches_non_streaming(self):
        model, _ = run_load(
            streaming=True, host_staging=True, num_threads=8, tail_threads=2
        )
        self.assert_matches_baseline(model)

    def test_loader_without_a_host_kernel_falls_back_to_the_device(self):
        model, online_quant_streamer = run_load(
            streaming=True, linear_cls=HostHostileLinear
        )
        self.assert_matches_baseline(model)
        for layer in model.model.layers:
            # The first refusal moves the whole module off host storage.
            self.assertEqual(layer.self_attn.o_proj.refusals, 1)
            self.assertEqual(layer.mlp.experts.stage_calls, 3 * NUM_EXPERTS)
        self.assertEqual(
            len(online_quant_streamer.done_module_ids),
            len(online_quant_streamer.candidates),
        )


class StreamingTriggerTest(unittest.TestCase):
    """Completion has to fire during the walk, once per module."""

    def assert_every_module_streamed(self, model, online_quant_streamer):
        streamed = model.streamed_modules()
        self.assertEqual(len(online_quant_streamer.done_module_ids), len(streamed))
        for module in streamed:
            self.assertIn(id(module), online_quant_streamer.done_module_ids)
            self.assertEqual(module.post_process_calls, 1)
        self.assertEqual(online_quant_streamer.excessive_loads, [])

    def test_every_module_completes_under_host_staging(self):
        model, online_quant_streamer = run_load(streaming=True, host_staging=True)
        self.assert_every_module_streamed(model, online_quant_streamer)

    def test_every_module_completes_under_buffered_replay(self):
        model, online_quant_streamer = run_load(streaming=True, host_staging=False)
        self.assert_every_module_streamed(model, online_quant_streamer)

    def test_every_module_completes_with_a_parallel_walk(self):
        model, online_quant_streamer = run_load(
            streaming=True, host_staging=True, num_threads=8, tail_threads=2
        )
        self.assert_every_module_streamed(model, online_quant_streamer)

    def test_a_loader_that_copies_twice_does_not_claim_early(self):
        """Repeated copies into one parameter must not claim the module early."""
        model, online_quant_streamer = run_load(
            streaming=True, linear_cls=DoubleCopyLinear
        )
        self.assert_every_module_streamed(model, online_quant_streamer)
        baseline, _ = run_load(streaming=False, linear_cls=DoubleCopyLinear)
        for name, want in weights_of(baseline).items():
            torch.testing.assert_close(
                want, dict(model.named_parameters())[name].detach(), rtol=0, atol=0
            )

    def test_host_staging_buffers_no_loader_calls(self):
        """Host staging must release checkpoint tensors immediately."""
        model, _ = run_load(streaming=True, host_staging=True)
        for module in model.streamed_modules():
            self.assertEqual(module._stream_buffer_list, [])

    def test_materialized_param_skips_module_lock_on_later_arrivals(self):
        """Only the first shard should serialize to create parameter storage."""

        class FailIfEntered:
            def __enter__(self):
                raise AssertionError("materialized fast path reacquired module lock")

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        with env_overrides(ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING="1"):
            model = StreamModel(1, NUM_EXPERTS, stream=True)
            streamer = OnlineQuantStreamer.maybe_create(model, None)
            experts = model.model.layers[0].mlp.experts
            param = experts.w13_weight

            self.assertFalse(
                streamer._ensure_storage_or_defer(experts, param),
            )
            self.assertIn(id(param), experts._stream_materialized_param_ids)
            experts._stream_lock = FailIfEntered()
            self.assertFalse(
                streamer._ensure_storage_or_defer(experts, param),
            )

    def test_fused_param_declines_later_semantic_staging(self):
        """A fused writer owns the parameter even if expert shards follow."""
        with env_overrides(ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING="1"):
            model = StreamModel(1, NUM_EXPERTS, stream=True)
            streamer = OnlineQuantStreamer.maybe_create(model, None)
            experts = model.model.layers[0].mlp.experts
            param = experts.w13_weight

            streamer.materialize_fused_param(param)
            self.assertIn(id(param), experts._stream_moe_declined_param_ids)
            self.assertTrue(experts._stream_tracking_invalid)

            source = torch.ones(INTER, HIDDEN)
            counter = streamer._run_counted(
                experts,
                param,
                experts.weight_loader,
                (param, source, "experts.w13_weight", "w1", 0),
            )

            self.assertIsNone(counter.moe_arrival)
            self.assertEqual(experts.stage_calls, 0)
            self.assertEqual(experts.loader_calls, 1)

    def test_moe_arrivals_use_semantic_regions_without_dispatch_counting(self):
        class CountingCopyCounter(streaming_module._CopyCounter):
            entries = 0

            def __enter__(self):
                type(self).entries += 1
                return super().__enter__()

        with mock.patch.object(
            streaming_module,
            "_CopyCounter",
            CountingCopyCounter,
        ):
            model, _ = run_load(
                streaming=True,
                num_layers=1,
                num_threads=1,
            )

        # MoE shards use semantic regions; only Linear params use copy counting.
        self.assertEqual(CountingCopyCounter.entries, 2)
        experts = model.model.layers[0].mlp.experts
        self.assertEqual(experts.loader_calls, 0)
        self.assertEqual(experts.stage_calls, 3 * NUM_EXPERTS)
        self.assertEqual(
            len(experts._stream_moe_arrivals[id(experts.w13_weight)]),
            2 * NUM_EXPERTS,
        )
        self.assertEqual(
            len(experts._stream_moe_arrivals[id(experts.w2_weight)]),
            NUM_EXPERTS,
        )

    def test_duplicate_moe_region_does_not_claim_module_early(self):
        tensors = list(checkpoint(1, NUM_EXPERTS).items())
        duplicate_index = next(
            i for i, (name, _) in enumerate(tensors) if ".0.gate_proj.weight" in name
        )
        tensors.insert(duplicate_index + 1, tensors[duplicate_index])

        model, online_quant_streamer = run_load(
            streaming=True,
            num_layers=1,
            tensors=tensors,
        )

        self.assert_every_module_streamed(model, online_quant_streamer)
        baseline, _ = run_load(streaming=False, num_layers=1)
        for name, want in weights_of(baseline).items():
            torch.testing.assert_close(
                want,
                dict(model.named_parameters())[name].detach(),
                rtol=0,
                atol=0,
            )

    def test_shared_expert_regions_are_not_covered_by_routed_staging(self):
        model, online_quant_streamer = run_load(
            streaming=True,
            num_layers=1,
            model_cls=SharedExpertStreamModel,
        )

        self.assert_every_module_streamed(model, online_quant_streamer)
        experts = model.model.layers[0].mlp.experts
        # Routed base slots take the staging protocol; the final shared slot
        # remains on the generic loader and contributes its own three regions.
        self.assertEqual(experts.stage_calls, 3 * (NUM_EXPERTS - 1))
        self.assertEqual(experts.loader_calls, 3)

    def test_param_that_declines_before_staging_uses_generic_counter(self):
        model, online_quant_streamer = run_load(
            streaming=True,
            num_layers=1,
            model_cls=DeclineW2StreamModel,
        )

        self.assert_every_module_streamed(model, online_quant_streamer)
        experts = model.model.layers[0].mlp.experts
        self.assertEqual(experts.loader_calls, NUM_EXPERTS)
        self.assertIn(
            id(experts.w2_weight),
            experts._stream_moe_declined_param_ids,
        )

    def test_mixed_semantic_and_declined_param_falls_back_safely(self):
        model, online_quant_streamer = run_load(
            streaming=True,
            num_layers=1,
            model_cls=DeclineW3StreamModel,
        )

        experts = model.model.layers[0].mlp.experts
        self.assertNotIn(id(experts), online_quant_streamer.done_module_ids)
        self.assertTrue(experts._stream_tracking_invalid)
        baseline, _ = run_load(
            streaming=False,
            num_layers=1,
            model_cls=DeclineW3StreamModel,
        )
        for name, want in weights_of(baseline).items():
            torch.testing.assert_close(
                want,
                dict(model.named_parameters())[name].detach(),
                rtol=0,
                atol=0,
            )


class SwapStorageVisibilityTest(unittest.TestCase):
    """Keep parameter attributes visible during concurrent storage swaps."""

    def test_attributes_stay_visible_across_a_concurrent_swap(self):
        param = nn.Parameter(torch.zeros(4, 4, device="meta"), requires_grad=False)
        sentinel = object()
        param.weight_loader = sentinel
        misses = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                if getattr(param, "weight_loader", None) is not sentinel:
                    misses.append(1)

        # Increase scheduling frequency to expose the narrow swap window.
        interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        watcher = threading.Thread(target=reader)
        watcher.start()
        try:
            for _ in range(2000):
                OnlineQuantStreamer._swap_storage(param, torch.zeros(4, 4))
        finally:
            stop.set()
            watcher.join()
            sys.setswitchinterval(interval)
        self.assertEqual(misses, [], "weight_loader disappeared during a swap")
        self.assertIs(param.weight_loader, sentinel)


class StreamingCoverageTest(unittest.TestCase):
    """What the checkpoint over- or under-delivers must not pass silently."""

    def test_missing_parameter_is_zero_filled_not_left_on_meta(self):
        tensors = checkpoint(1, NUM_EXPERTS)
        del tensors["model.layers.0.self_attn.o_proj.bias"]
        model, online_quant_streamer = run_load(
            streaming=True, num_layers=1, tensors=tensors
        )
        o_proj = model.model.layers[0].self_attn.o_proj
        self.assertFalse(o_proj.bias.data.is_meta)
        torch.testing.assert_close(o_proj.bias.data, torch.zeros_like(o_proj.bias.data))
        # It never reached its expected numel, so it fell back to the post-load
        # pass -- while the MoE beside it, which did complete, still streamed.
        self.assertNotIn(id(o_proj), online_quant_streamer.done_module_ids)
        self.assertIn(
            id(model.model.layers[0].mlp.experts),
            online_quant_streamer.done_module_ids,
        )

    def test_fused_expert_writes_get_storage_and_fall_back(self):
        """Fused writes get storage and fall back when coverage is unknown."""
        model, online_quant_streamer = run_load(
            streaming=True,
            num_layers=1,
            tensors=fused_checkpoint(1, NUM_EXPERTS),
            model_cls=FusedExpertStreamModel,
        )
        experts = model.model.layers[0].mlp.experts
        for param in (experts.w13_weight, experts.w2_weight):
            self.assertFalse(param.data.is_meta)
        self.assertNotIn(id(experts), online_quant_streamer.done_module_ids)
        self.assertEqual(experts.post_process_calls, 0)
        self.assertLessEqual(
            {
                "model.layers.0.mlp.experts.w13_weight",
                "model.layers.0.mlp.experts.w2_weight",
            },
            model._loaded_weights_record,
        )

        baseline, _ = run_load(
            streaming=False,
            num_layers=1,
            tensors=fused_checkpoint(1, NUM_EXPERTS),
            model_cls=FusedExpertStreamModel,
        )
        for name, want in weights_of(baseline).items():
            torch.testing.assert_close(
                want, dict(model.named_parameters())[name].detach(), rtol=0, atol=0
            )

    def test_load_arriving_after_the_module_was_quantized_is_dropped(self):
        """Drop source writes that arrive after quantization."""
        tensors = checkpoint(1, NUM_EXPERTS)
        weight_name = "model.layers.0.self_attn.o_proj.weight"
        wanted = tensors[weight_name].clone()
        surplus = list(tensors.items()) + [(weight_name, torch.full_like(wanted, 7.0))]
        model, online_quant_streamer = run_load(
            streaming=True, num_layers=1, tensors=surplus
        )
        weight = model.model.layers[0].self_attn.o_proj.weight
        torch.testing.assert_close(weight.detach(), wanted, rtol=0, atol=0)
        self.assertEqual(len(online_quant_streamer.excessive_loads), 1)

    def test_nonlocal_ep_loads_after_quantization_are_ignored(self):
        """Do not report non-local expert arrivals after local completion."""
        model, online_quant_streamer = run_load(
            streaming=True,
            num_layers=1,
            model_cls=ExpertParallelStreamModel,
        )
        baseline, _ = run_load(
            streaming=False,
            num_layers=1,
            model_cls=ExpertParallelStreamModel,
        )

        experts = model.model.layers[0].mlp.experts
        self.assertIn(id(experts), online_quant_streamer.done_module_ids)
        self.assertEqual(online_quant_streamer.excessive_loads, [])
        for name, want in weights_of(baseline).items():
            torch.testing.assert_close(
                want, dict(model.named_parameters())[name].detach(), rtol=0, atol=0
            )


class DraftStreamingSelectionTest(unittest.TestCase):
    def test_spec_decode_load_passes_streamer_to_loading_core(self):
        model = nn.Module()
        streamer = mock.Mock()
        streamer.done_module_ids = set()
        streamer.candidates = []

        with (
            mock.patch.object(
                loader_module.OnlineQuantStreamer,
                "maybe_create",
                return_value=streamer,
            ) as maybe_create,
            mock.patch.object(
                loader_module,
                "load_weights_into_model",
                return_value=set(),
            ) as load_weights,
            mock.patch.object(
                loader_module.torch.cuda,
                "is_available",
                return_value=False,
            ),
        ):
            loader_module.load_model(
                model,
                "<synthetic>",
                mock.Mock(),
                spec_decode=True,
            )

        maybe_create.assert_called_once_with(model, None)
        self.assertIs(
            load_weights.call_args.kwargs["online_quant_streamer"],
            streamer,
        )
        streamer.replay_stragglers_and_report.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main()
