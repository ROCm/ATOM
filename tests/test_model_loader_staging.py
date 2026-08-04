# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Characterization tests for MoE expert loading in `loading_core`.

`ATOM_LOADER_NUM_THREADS > 1` routes MoE expert weights through a CPU staging
buffer that is written back to the parameter in one large copy; `= 1` sends
every arrival through the per-expert `weight_loader`.  The two must be
indistinguishable, so the primary assertion is that the final parameter bytes
are identical.

The bug these tests are written against: a checkpoint whose experts do not all
reach the parameter through the same path.  Qwen3.5 BF16 stores routed experts
as one stacked 3D tensor but the shared expert as separate tensors, so the
staging buffer only ever owns one of the 257 expert slots — and writing such a
buffer back wholesale would zero the 256 slots the fused path already filled.

Everything here is plain CPU torch: `loading_core` and `expert_layout` import
no AITER, and the fake MoE module below calls the *real* layout functions so
the tests keep their teeth.
"""

import builtins
import os
import tempfile
import threading
import unittest
import unittest.mock
from typing import ClassVar

import safetensors.torch
import torch
from torch import nn

from atom.model_loader.expert_staging import ExpertStagingPool, _cpu_zeroable
from atom.model_loader.loading_core import load_weights_into_model
from atom.model_loader.weight_iterator import (
    _shard_tensor_names,
    safetensors_weights_iterator,
)
from atom.model_ops.fused_moe.expert_layout import (
    count_local_base_experts,
    determine_expert_map,
    expert_region,
    physical_expert_id,
)

HIDDEN = 8
INTER = 4


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


class FakeFusedMoE(nn.Module):
    """The parts of `FusedMoE` the loader talks to, on CPU.

    Slot layout mirrors the real module: routed experts first, fused shared
    experts appended after them, EPLB redundant replicas in between when
    `num_redundant_experts > 0`.
    """

    def __init__(
        self,
        num_routed: int,
        num_fused_shared: int = 1,
        num_redundant: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
    ):
        super().__init__()
        self.num_fused_shared_experts = num_fused_shared
        self.num_redundant_experts = num_redundant
        self.global_num_experts = num_routed + num_redundant
        local_routed, expert_map = determine_expert_map(
            ep_size, ep_rank, self.global_num_experts
        )
        if expert_map is not None:
            # Shared experts map to the slots right after the local routed ones,
            # then a sentinel for the fake expert id AITER's topk uses.
            expert_map = torch.cat(
                [
                    expert_map,
                    torch.tensor(
                        [local_routed + i for i in range(num_fused_shared)],
                        dtype=torch.int32,
                    ),
                    torch.tensor([-1], dtype=torch.int32),
                ]
            )
        self.expert_map = expert_map
        self.local_num_experts = local_routed + num_fused_shared
        self.w13_weight = nn.Parameter(
            torch.zeros(self.local_num_experts, 2 * INTER, HIDDEN), requires_grad=False
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(self.local_num_experts, HIDDEN, INTER), requires_grad=False
        )
        # `FusedMoE.create_weights` attaches the loader to the parameter via
        # `set_weight_attrs`; the loader looks it up there, not on the module.
        self.w13_weight.weight_loader = self.weight_loader
        self.w2_weight.weight_loader = self.weight_loader
        self.stage_calls = 0
        self.direct_calls = 0
        self.flush_calls = 0
        self.fast_path_flushes = 0

    # ── protocol the loader depends on ────────────────────────────────────

    def _map_global_expert_id_to_local_expert_id(self, global_expert_id: int) -> int:
        if self.expert_map is None:
            return global_expert_id
        return int(
            self.expert_map[
                physical_expert_id(
                    global_expert_id,
                    self.global_num_experts,
                    self.num_redundant_experts,
                    self.num_fused_shared_experts,
                )
            ]
        )

    @property
    def num_local_base_experts(self) -> int:
        return count_local_base_experts(
            expert_map=self.expert_map,
            global_num_experts=self.global_num_experts,
            num_redundant_experts=self.num_redundant_experts,
            local_num_experts=self.local_num_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
        )

    def is_batched_expert_slot(self, local_expert_id: int) -> bool:
        return local_expert_id < self.num_local_base_experts

    def flush_staged(self, param, staging, filled) -> None:
        self.flush_calls += 1
        n_base = self.num_local_base_experts
        if len(filled) == self.expected_batched_arrivals(param):
            self.fast_path_flushes += 1
            param.data[:n_base].copy_(staging[:n_base])
        else:
            for local_expert_id, shard_id in sorted(filled):
                expert_region(param.data, local_expert_id, shard_id).copy_(
                    expert_region(staging, local_expert_id, shard_id)
                )
        for slot in range(
            n_base, self.local_num_experts - self.num_fused_shared_experts
        ):
            param.data[slot].zero_()

    def expected_batched_arrivals(self, param: nn.Parameter) -> int | None:
        n_local_base = self.num_local_base_experts
        if param is self.w13_weight:
            return n_local_base * 2
        if param is self.w2_weight:
            return n_local_base
        return None

    def stage_expert_weight(
        self, param, staging, loaded_weight, local_expert_id, shard_id, weight_name
    ) -> bool:
        if shard_id not in ("w1", "w2", "w3"):
            return False
        self.stage_calls += 1
        expert_region(staging, local_expert_id, shard_id).copy_(loaded_weight)
        return True

    def weight_loader(
        self, param, loaded_weight, weight_name="", shard_id="", expert_id=0
    ) -> None:
        local_expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if local_expert_id == -1:
            return
        self.direct_calls += 1
        expert_region(param.data, local_expert_id, shard_id).copy_(loaded_weight)


class _Layers(nn.Module):
    def __init__(self, num_layers: int, **moe_kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.experts = FakeFusedMoE(**moe_kwargs)
            self.layers.append(layer)


class FakeMoEModel(nn.Module):
    """Model exposing the hooks `load_weights_into_model` looks for."""

    packed_modules_mapping: ClassVar[dict] = {}
    weights_mapping: ClassVar[dict] = {}

    def __init__(self, num_layers: int, num_routed: int, **moe_kwargs):
        super().__init__()
        self.num_routed = num_routed
        self.num_fused_shared = moe_kwargs.get("num_fused_shared", 1)
        self.model = _Layers(num_layers, num_routed=num_routed, **moe_kwargs)

    def moes(self) -> list[FakeFusedMoE]:
        return [m for m in self.modules() if isinstance(m, FakeFusedMoE)]

    def get_expert_mapping(self):
        return [
            (
                "experts.w13_" if weight in ("gate_proj", "up_proj") else "experts.w2_",
                f"experts.{expert_id}.{weight}.",
                expert_id,
                shard,
            )
            for expert_id in range(self.num_routed + self.num_fused_shared)
            for shard, weight in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            )
        ]


class FusedCkptMoEModel(FakeMoEModel):
    """Routed experts arrive as one stacked 3D tensor per layer (Qwen3.5 BF16)."""

    def detect_fused_expert_format(self, weight_name: str) -> bool:
        return "experts.gate_up_proj" in weight_name or (
            "experts.down_proj" in weight_name and ".experts." in weight_name
        )

    def get_fused_expert_mapping(self):
        return [
            ("experts.w13_weight", "experts.gate_up_proj", "w1"),
            ("experts.w2_weight", "experts.down_proj", "w2"),
        ]


class MTPDrafterModel(FakeMoEModel):
    """Drafter loaded from the target checkpoint (`mtp.*` block only)."""

    weights_mapping: ClassVar[dict] = {"mtp.": "model."}

    def remap_mtp_weight_name(self, name: str) -> str | None:
        if name.startswith("mtp."):
            return name
        if any(key in name for key in ("embed_tokens", "lm_head")):
            return name
        return None


def load_fused_expert_weights(
    original_name, name, params_dict, loaded_weight, shard_id, num_experts
):
    """Split a stacked `[E, ...]` checkpoint tensor across the expert slots."""
    param = params_dict[name]
    moe = _MODULE_OF_PARAM[id(param)]
    if shard_id == "w2":
        for expert_id in range(num_experts):
            moe.weight_loader(param, loaded_weight[expert_id], name, "w2", expert_id)
        return True
    half = loaded_weight.shape[1] // 2
    for expert_id in range(num_experts):
        moe.weight_loader(param, loaded_weight[expert_id, :half], name, "w1", expert_id)
        moe.weight_loader(param, loaded_weight[expert_id, half:], name, "w3", expert_id)
    return True


# Set by `build_model`. The real models reach the owning module via
# `model.get_submodule`; the fused loader here only needs the same lookup.
_MODULE_OF_PARAM: dict[int, FakeFusedMoE] = {}


class HFConfig:
    def __init__(self, num_hidden_layers: int, n_routed_experts: int):
        self.num_hidden_layers = num_hidden_layers
        self.n_routed_experts = n_routed_experts
        self.num_experts = n_routed_experts


# ── checkpoint builders: return an ordered list of "shards" ───────────────


def _rand(*shape) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float32)


def per_expert_shards(num_layers, num_routed, prefix="model"):
    """Routed experts as individual tensors, shared expert separate."""
    routed, shared = {}, {}
    for layer in range(num_layers):
        base = f"{prefix}.layers.{layer}.mlp"
        for expert_id in range(num_routed):
            p = f"{base}.experts.{expert_id}"
            routed[f"{p}.gate_proj.weight"] = _rand(INTER, HIDDEN)
            routed[f"{p}.up_proj.weight"] = _rand(INTER, HIDDEN)
            routed[f"{p}.down_proj.weight"] = _rand(HIDDEN, INTER)
        shared[f"{base}.shared_expert.gate_proj.weight"] = _rand(INTER, HIDDEN)
        shared[f"{base}.shared_expert.up_proj.weight"] = _rand(INTER, HIDDEN)
        shared[f"{base}.shared_expert.down_proj.weight"] = _rand(HIDDEN, INTER)
    return [routed, shared]


def fused_shards(num_layers, num_routed):
    """Routed experts stacked into one 3D tensor, shared expert separate."""
    routed, shared = {}, {}
    for layer in range(num_layers):
        base = f"model.layers.{layer}.mlp"
        routed[f"{base}.experts.gate_up_proj"] = _rand(num_routed, 2 * INTER, HIDDEN)
        routed[f"{base}.experts.down_proj"] = _rand(num_routed, HIDDEN, INTER)
        shared[f"{base}.shared_expert.gate_proj.weight"] = _rand(INTER, HIDDEN)
        shared[f"{base}.shared_expert.up_proj.weight"] = _rand(INTER, HIDDEN)
        shared[f"{base}.shared_expert.down_proj.weight"] = _rand(HIDDEN, INTER)
    return [routed, shared]


def shards_to_iterator(shards, materialized=None):
    """Turn an ordered list of shards into a `weights_iterator` callable.

    Replaces the `glob` + `safe_open` pair, whose order is filesystem- and
    lexicographic-dependent; here the read order is exactly the list order.
    `materialized`, when given, records every name the loader actually asked
    for -- i.e. everything the `wants` predicate did not reject.
    """

    def _iterator(path, disable_mmap, wants=None):
        for shard in shards:
            for name, tensor in shard.items():
                if wants is not None and not wants(name):
                    continue
                if materialized is not None:
                    materialized.append(name)
                yield name, tensor

    return _iterator


# ── harness ───────────────────────────────────────────────────────────────


def build_model(model_cls, num_layers, num_routed, **moe_kwargs):
    model = model_cls(num_layers=num_layers, num_routed=num_routed, **moe_kwargs)
    for moe in model.moes():
        _MODULE_OF_PARAM[id(moe.w13_weight)] = moe
        _MODULE_OF_PARAM[id(moe.w2_weight)] = moe
    return model


def run_load(model, shards, hf_config, num_threads, fused=False, spec_decode=False):
    os.environ["ATOM_LOADER_NUM_THREADS"] = str(num_threads)
    try:
        load_weights_into_model(
            model=model,
            model_name_or_path="<synthetic>",
            hf_config=hf_config,
            spec_decode=spec_decode,
            load_fused_expert_weights_fn=(load_fused_expert_weights if fused else None),
            default_weight_loader=default_weight_loader,
            fuse_shared_expert=lambda *_args, **_kw: True,
            is_rank0=lambda: True,
            weights_iterator=shards_to_iterator(shards),
        )
    finally:
        os.environ.pop("ATOM_LOADER_NUM_THREADS", None)
    return {name: p.detach().clone() for name, p in model.named_parameters()}


class ExpertLoadingDifferentialTest(unittest.TestCase):
    """Serial and batched loading must produce byte-identical parameters."""

    NUM_LAYERS = 2
    NUM_ROUTED = 8

    def assert_same(self, serial, batched):
        self.assertEqual(sorted(serial), sorted(batched))
        for name in serial:
            torch.testing.assert_close(
                serial[name], batched[name], rtol=0, atol=0, msg=f"mismatch: {name}"
            )

    def assert_every_slot_written(self, params):
        for name, tensor in params.items():
            for slot in range(tensor.shape[0]):
                self.assertTrue(
                    bool(tensor[slot].abs().sum() > 0),
                    f"{name} slot {slot} was never written",
                )

    def run_pair(self, model_cls, shards, fused=False, spec_decode=False, **moe_kwargs):
        hf_config = HFConfig(self.NUM_LAYERS, self.NUM_ROUTED)
        results, models = [], []
        for threads in (1, 16):
            model = build_model(
                model_cls, self.NUM_LAYERS, self.NUM_ROUTED, **moe_kwargs
            )
            results.append(
                run_load(
                    model,
                    shards,
                    hf_config,
                    threads,
                    fused=fused,
                    spec_decode=spec_decode,
                )
            )
            models.append(model)
        return results, models

    # ── (b) baseline: every arrival reaches the parameter through staging ──

    def test_per_expert_checkpoint_with_fused_shared_expert(self):
        shards = per_expert_shards(self.NUM_LAYERS, self.NUM_ROUTED)
        (serial, batched), (_, batched_model) = self.run_pair(FakeMoEModel, shards)
        self.assert_same(serial, batched)
        self.assert_every_slot_written(batched)
        # With threads=1 the staging path is dead code, so the comparison above
        # only means something if the batched run really did batch -- and did so
        # through the single-large-copy path staging exists for.
        moes = batched_model.moes()
        self.assertGreater(sum(m.stage_calls for m in moes), 0)
        self.assertEqual(
            sum(m.fast_path_flushes for m in moes),
            2 * len(moes),
            "every w13/w2 pair should have flushed via the complete-batch path",
        )

    # ── (a) the reported bug ──────────────────────────────────────────────

    def test_fused_routed_checkpoint_with_separate_shared_expert(self):
        shards = fused_shards(self.NUM_LAYERS, self.NUM_ROUTED)
        (serial, batched), _ = self.run_pair(FusedCkptMoEModel, shards, fused=True)
        self.assert_same(serial, batched)
        self.assert_every_slot_written(batched)

    # ── (g) shard read order must not matter ──────────────────────────────

    def test_shared_expert_shard_read_before_routed_shards(self):
        shards = fused_shards(self.NUM_LAYERS, self.NUM_ROUTED)
        (serial, batched), _ = self.run_pair(
            FusedCkptMoEModel, list(reversed(shards)), fused=True
        )
        self.assert_same(serial, batched)
        self.assert_every_slot_written(batched)

    # ── (c) expert parallelism ────────────────────────────────────────────

    def test_expert_parallel_with_fused_shared_expert(self):
        shards = per_expert_shards(self.NUM_LAYERS, self.NUM_ROUTED)
        (serial, batched), _ = self.run_pair(FakeMoEModel, shards, ep_size=2, ep_rank=0)
        self.assert_same(serial, batched)

    # ── (f) MTP drafter pass over the same checkpoint ─────────────────────

    def test_mtp_drafter_pass(self):
        shards = per_expert_shards(1, self.NUM_ROUTED, prefix="mtp")
        hf_config = HFConfig(1, self.NUM_ROUTED)
        results, models = [], []
        for threads in (1, 16):
            model = build_model(MTPDrafterModel, 1, self.NUM_ROUTED)
            results.append(
                run_load(model, shards, hf_config, threads, spec_decode=True)
            )
            models.append(model)
        self.assert_same(*results)
        self.assert_every_slot_written(results[1])
        moes = models[1].moes()
        self.assertGreater(sum(m.stage_calls for m in moes), 0)
        self.assertEqual(sum(m.fast_path_flushes for m in moes), 2 * len(moes))


class EPLBRedundantSlotTest(unittest.TestCase):
    """EPLB redundant replicas change where the shared expert's slot lands.

    Thread count is irrelevant here — the shared expert's global id is decided
    by a pure string rewrite — so a differential assertion would pass whether
    or not the id is right.  Assert the resolved id directly instead.
    """

    NUM_LAYERS = 1
    NUM_ROUTED = 8
    NUM_REDUNDANT = 4

    def test_shared_expert_lands_past_the_redundant_slots(self):
        model = build_model(
            FakeMoEModel,
            self.NUM_LAYERS,
            self.NUM_ROUTED,
            num_redundant=self.NUM_REDUNDANT,
            ep_size=2,
            ep_rank=0,
        )
        moe = model.moes()[0]
        shards = per_expert_shards(self.NUM_LAYERS, self.NUM_ROUTED)
        run_load(model, shards, HFConfig(self.NUM_LAYERS, self.NUM_ROUTED), 1)

        shared_slot = moe.local_num_experts - 1
        self.assertTrue(
            bool(moe.w13_weight[shared_slot].abs().sum() > 0),
            "shared expert slot was never written",
        )
        n_local_base = count_local_base_experts(
            expert_map=moe.expert_map,
            global_num_experts=moe.global_num_experts,
            num_redundant_experts=moe.num_redundant_experts,
            local_num_experts=moe.local_num_experts,
        )
        for slot in range(n_local_base, shared_slot):
            self.assertEqual(
                float(moe.w13_weight[slot].abs().sum()),
                0.0,
                f"redundant slot {slot} was overwritten by the shared expert; "
                "fill_redundant populates these after loading",
            )


class StagingBufferAllocationTest(unittest.TestCase):
    """Packed dtypes have to fall back to a raw-byte staging buffer."""

    # `zero_` on a packed fp4x2 tensor only dispatches to `fill_cpu` -- which
    # has no kernel for that dtype -- once the tensor is big enough for the
    # parallel path (torch's GRAIN_SIZE, 32768 elements). Anything smaller
    # takes a memset fast path and succeeds, so a small tensor would not
    # reproduce this at all.
    NUMEL = 1 << 15

    @unittest.skipUnless(hasattr(torch, "float4_e2m1fn_x2"), "torch has no fp4x2 dtype")
    def test_packed_dtype_falls_back_to_raw_bytes(self):
        param = nn.Parameter(
            torch.empty((4, self.NUMEL // 4), dtype=torch.float4_e2m1fn_x2),
            requires_grad=False,
        )
        with self.assertRaises(NotImplementedError):
            param.data.zero_()

        staging = ExpertStagingPool._allocate_staging(param)

        self.assertEqual(staging.dtype, torch.uint8)
        self.assertEqual(staging.shape, param.data.shape)
        self.assertEqual(int(staging.sum()), 0, "staging buffer must be zeroed")

    @unittest.skipUnless(hasattr(torch, "float4_e2m1fn_x2"), "torch has no fp4x2 dtype")
    def test_packed_dtype_allocates_exactly_one_buffer(self):
        """The doomed first allocation must not happen.

        Finding the raw-byte fallback by allocating with the packed dtype and
        letting `zero_` raise costs a full host allocation per parameter --
        pinned host memory allocates at ~4 GB/s cold, and on DeepSeek-R1 MXFP4
        every routed-expert parameter is packed. That discarded buffer
        profiled as ~40% of the loader worker pool's time.
        """
        param = nn.Parameter(
            torch.empty((4, self.NUMEL // 4), dtype=torch.float4_e2m1fn_x2),
            requires_grad=False,
        )
        # The dtype probe allocates too, and is cached; warm it first so the
        # count below sees only the staging allocation.
        _cpu_zeroable(param.data.dtype)

        dtypes_allocated = []
        real_empty = torch.empty

        def counting_empty(*args, **kwargs):
            dtypes_allocated.append(kwargs.get("dtype"))
            return real_empty(*args, **kwargs)

        with unittest.mock.patch.object(torch, "empty", counting_empty):
            staging = ExpertStagingPool._allocate_staging(param)

        self.assertEqual(staging.dtype, torch.uint8)
        self.assertEqual(
            dtypes_allocated,
            [torch.uint8],
            "packed params must go straight to a raw-byte buffer",
        )


class _OneParamMoE:
    """The smallest module satisfying the pool's protocol, for race tests."""

    def __init__(self, param: nn.Parameter, expected: int):
        self.param = param
        self.expected = expected
        self.on_staged = lambda: None

    def expected_batched_arrivals(self, param):
        return self.expected

    def _map_global_expert_id_to_local_expert_id(self, global_expert_id):
        return global_expert_id

    def is_batched_expert_slot(self, local_expert_id):
        return True

    def stage_expert_weight(
        self, param, staging, loaded_weight, local_expert_id, shard_id, weight_name
    ):
        staging[local_expert_id].copy_(loaded_weight)
        self.on_staged()
        return True

    def flush_staged(self, param, staging, filled):
        for local_expert_id, _ in filled:
            param.data[local_expert_id].copy_(staging[local_expert_id])


class DeclineDuringArrivalTest(unittest.TestCase):
    """`decline` must never strand a region an in-flight arrival is writing.

    `decline` hands the parameter to another loader path, which writes only
    regions the pool never staged. It takes the entry off the table and writes
    back what has landed -- but "what has landed" is read at that instant, so
    an arrival still in flight is invisible to it. Both windows below end with
    the expert silently missing from the parameter if the arrival does not
    write its own region back.
    """

    NUM_SLOTS = 4
    ARRIVING_SLOT = 3

    def _setup(self, expected):
        param = nn.Parameter(torch.zeros(self.NUM_SLOTS, HIDDEN), requires_grad=False)
        moe = _OneParamMoE(param, expected)
        direct = []

        def weight_loader(p, w, name, shard_id, global_expert_id):
            direct.append(global_expert_id)
            p.data[global_expert_id].copy_(w)

        param.weight_loader = weight_loader
        return param, moe, direct, ExpertStagingPool(lambda name: moe)

    def _assert_region_survived(self, param, direct, payload):
        landed = param.data[self.ARRIVING_SLOT]
        self.assertTrue(
            torch.equal(landed, payload),
            f"expert region was dropped: got {landed.tolist()}, "
            f"want {payload.tolist()} (direct_load calls: {direct})",
        )

    def test_decline_lands_while_the_buffer_is_being_allocated(self):
        """The wide window: the entry is published before its buffer exists.

        `decline` pops that entry, sees `staging is None`, and skips the flush
        entirely -- so nothing it does can cover the arrival that is at that
        moment blocked inside the allocation.
        """
        param, _moe, direct, pool = self._setup(expected=self.NUM_SLOTS)
        payload = torch.full((HIDDEN,), 7.0)

        allocating = threading.Event()
        may_finish = threading.Event()
        real_allocate = ExpertStagingPool._allocate_staging

        def blocking_allocate(p):
            allocating.set()
            may_finish.wait(timeout=5)
            return real_allocate(p)

        arrival = threading.Thread(
            target=pool.stage,
            args=(param, "m.w13_weight", "w1", self.ARRIVING_SLOT, payload),
        )
        with unittest.mock.patch.object(
            ExpertStagingPool, "_allocate_staging", staticmethod(blocking_allocate)
        ):
            arrival.start()
            self.assertTrue(allocating.wait(timeout=5), "allocation never started")
            pool.decline(param)
            may_finish.set()
            arrival.join(timeout=5)
        self.assertFalse(arrival.is_alive(), "arrival thread hung")

        pool.flush_pending()
        self._assert_region_survived(param, direct, payload)

    def test_decline_lands_between_the_copy_and_the_bookkeeping(self):
        """The narrow window: the region is staged but not yet in `filled`.

        `decline` does flush here, but over a `filled` set this region has not
        been added to yet, so the copy it makes skips exactly this slot.
        """
        param, moe, direct, pool = self._setup(expected=self.NUM_SLOTS)
        payload = torch.full((HIDDEN,), 5.0)

        # Fire the decline from inside the copy, i.e. after the bytes reach the
        # staging buffer and before `stage` records the region as filled.
        moe.on_staged = lambda: pool.decline(param)

        pool.stage(param, "m.w13_weight", "w1", self.ARRIVING_SLOT, payload)

        pool.flush_pending()
        self._assert_region_survived(param, direct, payload)

    def test_declined_before_arrival_goes_straight_to_the_weight_loader(self):
        """The ordinary case still bypasses the pool rather than double-writing."""
        param, _moe, direct, pool = self._setup(expected=self.NUM_SLOTS)
        payload = torch.full((HIDDEN,), 3.0)

        pool.decline(param)
        pool.stage(param, "m.w13_weight", "w1", self.ARRIVING_SLOT, payload)

        self.assertEqual(direct, [self.ARRIVING_SLOT])
        self._assert_region_survived(param, direct, payload)


class DrafterSkipsUnwantedTensorsTest(unittest.TestCase):
    """A drafter load must not materialize the target model's weights.

    The drafter reads the *target's* checkpoint to pick out the MTP block. With
    `ATOM_DISABLE_MMAP=true` -- which CI sets -- a shard is read whole and
    deserialized whole, so the win comes from not reading shards that hold
    nothing wanted at all (see RealSafetensorsIteratorTest). This test pins the
    loader's half of that contract: it must reject by name, up front.
    """

    NUM_LAYERS = 1
    NUM_ROUTED = 8

    def test_target_weights_are_never_materialized(self):
        # Note: this drives the injected iterator, so it pins the loader's half
        # of the contract (it asks before consuming). The production iterator's
        # half -- header-only shard inspection and the skip -- is covered by
        # RealSafetensorsIteratorTest below.
        target = per_expert_shards(self.NUM_LAYERS, self.NUM_ROUTED, prefix="model")
        drafter = per_expert_shards(self.NUM_LAYERS, self.NUM_ROUTED, prefix="mtp")
        shards = target + drafter

        materialized: list[str] = []
        model = build_model(MTPDrafterModel, self.NUM_LAYERS, self.NUM_ROUTED)
        load_weights_into_model(
            model=model,
            model_name_or_path="<synthetic>",
            hf_config=HFConfig(self.NUM_LAYERS, self.NUM_ROUTED),
            spec_decode=True,
            default_weight_loader=default_weight_loader,
            fuse_shared_expert=lambda *_args, **_kw: True,
            is_rank0=lambda: True,
            weights_iterator=shards_to_iterator(shards, materialized),
        )

        self.assertTrue(materialized, "drafter loaded nothing at all")
        self.assertFalse(
            [n for n in materialized if not n.startswith("mtp.")],
            "target-model tensors were materialized during a drafter load",
        )
        # And the drafter still got everything it needed.
        for moe in model.moes():
            self.assertTrue(bool(moe.w13_weight.abs().sum() > 0))


class RealSafetensorsIteratorTest(unittest.TestCase):
    """Exercises the production iterator against real files on disk."""

    def _write_shards(self, root: str) -> None:
        safetensors.torch.save_file(
            {
                f"model.layers.0.mlp.experts.{i}.gate_proj.weight": torch.zeros(2)
                for i in range(4)
            },
            os.path.join(root, "model-00001-of-00002.safetensors"),
        )
        safetensors.torch.save_file(
            {"mtp.layers.0.mlp.gate.weight": torch.ones(2)},
            os.path.join(root, "model-00002-of-00002.safetensors"),
        )

    def test_shard_with_nothing_wanted_is_never_opened(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_shards(root)
            wanted = os.path.join(root, "model-00002-of-00002.safetensors")
            skipped = os.path.join(root, "model-00001-of-00002.safetensors")

            opened: list[str] = []
            real_open = builtins.open

            def _tracking_open(file, *args, **kwargs):
                # The header probe opens the file too; only count reads that go
                # past it, which is what `safetensors.torch.load` does.
                if isinstance(file, str) and file.endswith(".safetensors"):
                    opened.append(file)
                return real_open(file, *args, **kwargs)

            for disable_mmap in (True, False):
                opened.clear()
                with unittest.mock.patch.object(builtins, "open", _tracking_open):
                    names = [
                        name
                        for name, _ in safetensors_weights_iterator(
                            root, disable_mmap, wants=lambda n: n.startswith("mtp.")
                        )
                    ]
                self.assertEqual(names, ["mtp.layers.0.mlp.gate.weight"])
                # The skipped shard is opened at most once (the header probe),
                # never a second time to read its tensors.
                self.assertLessEqual(
                    opened.count(skipped), 1, f"disable_mmap={disable_mmap}"
                )
                self.assertGreaterEqual(opened.count(wanted), 1)

    def test_no_predicate_yields_everything(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_shards(root)
            for disable_mmap in (True, False):
                names = {
                    name for name, _ in safetensors_weights_iterator(root, disable_mmap)
                }
                self.assertEqual(len(names), 5)

    def test_unreadable_header_falls_back_to_loading_the_shard(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_shards(root)
            self.assertIsNotNone(
                _shard_tensor_names(
                    os.path.join(root, "model-00002-of-00002.safetensors")
                )
            )
            truncated = os.path.join(root, "truncated.safetensors")
            with open(truncated, "wb") as f:
                f.write(b"\x00\x01")
            # None means "cannot tell" -- the caller must not skip the shard on
            # the strength of a failed probe.
            self.assertIsNone(_shard_tensor_names(truncated))


if __name__ == "__main__":
    unittest.main()
