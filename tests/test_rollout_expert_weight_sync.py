# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routed MoE expert weights, through the weight sync.

A model's experts arrive one tensor per expert and belong in the fused
w13_weight / w2_weight of the layer's FusedMoE -- which is neither the
incoming name nor anything `packed_modules_mapping` describes, so they used to
match nothing and be counted as skipped at debug level. The rollout then
served checkpoint-time experts with nothing saying so.

Routing them is half the job. `weight_loader` writes plain row-major bytes
over buffers the kernel reads through an aiter permutation, so the layout has
to be re-established afterwards -- and re-running
`process_weights_after_loading` is not how, which the FP8 hook here
demonstrates against the real production method.

The loading methods under test are the production ones, attached unbound: a
double that reimplemented them would pass against a loader ATOM no longer has.
"""

from multiprocessing import shared_memory
from types import SimpleNamespace

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    pytest.skip(
        "FusedMoE's loaders and aiter's shuffle need a ROCm device",
        allow_module_level=True,
    )

from aiter import QuantType, dtypes
from conftest import atom_config_double

from atom.model_engine.collective_rpc import WeightReloadRefused
from atom.model_ops import moe as moe_module
from atom.model_ops.moe import Fp8MoEMethod, FusedMoE
from atom.model_ops.utils import shuffle_weights
from atom.quantization.quark.utils import quant_weight_online
from atom.rollout import weight_sync
from atom.rollout.weight_updater import WeightUpdaterMixin
from atom.utils.selector import Family

DEVICE = torch.device("cuda")
NUM_EXPERTS = 4
HIDDEN = 64
INTERMEDIATE = 32


class _QuantMethodSpy:
    """Counts the post-load hook. A sync must never reach it."""

    def __init__(self):
        self.post_load_calls = 0

    def process_weights_after_loading(self, layer):
        self.post_load_calls += 1


class _FusedMoEDouble(nn.Module):
    """A FusedMoE's expert buffers and its real loading path, no topology."""

    weight_loader = FusedMoE.weight_loader
    _copy_expert_shard = FusedMoE._copy_expert_shard
    _load_model_weight_or_group_weight_scale = (
        FusedMoE._load_model_weight_or_group_weight_scale
    )
    _load_w13 = FusedMoE._load_w13
    _load_w2 = FusedMoE._load_w2
    _load_single_value = FusedMoE._load_single_value
    _load_g_idx = FusedMoE._load_g_idx
    _load_per_tensor_weight_scale = FusedMoE._load_per_tensor_weight_scale
    _load_per_channel_weight_scale = FusedMoE._load_per_channel_weight_scale
    _copy_quant_storage = staticmethod(FusedMoE._copy_quant_storage)
    _map_global_expert_id_to_local_expert_id = (
        FusedMoE._map_global_expert_id_to_local_expert_id
    )
    shared_expert_staging = None

    def __init__(self, *, dtype=torch.bfloat16, expert_map=None):
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.zeros(
                NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=dtype, device=DEVICE
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE, dtype=dtype, device=DEVICE),
            requires_grad=False,
        )
        self.layer_quant_config = SimpleNamespace(
            quant_dtype=dtype, quant_type=QuantType.No
        )
        self.quant_method = _QuantMethodSpy()
        self.expert_map = expert_map
        self.num_redundant_experts = 0
        self.num_fused_shared_experts = 0
        self.global_num_experts = NUM_EXPERTS
        self.local_num_experts = NUM_EXPERTS
        self.tp_size = 1
        self.tp_rank = 0
        self.use_ep = False


class _ModelDouble(nn.Module):
    """One MoE layer, named the way a checkpoint names it."""

    def __init__(self, moe, *, with_expert_mapping=True):
        super().__init__()
        mlp = nn.Module()
        mlp.experts = moe
        mlp.gate = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False, device=DEVICE)
        layer = nn.Module()
        layer.mlp = mlp
        self.layers = nn.ModuleList([layer])
        if with_expert_mapping:
            self.get_expert_mapping = lambda: FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=NUM_EXPERTS,
            )


class _Runner(WeightUpdaterMixin):
    def __init__(self, model):
        self._init_weight_updater()
        self.model = model
        self.device = DEVICE
        self.rank = 0
        self.world_size = 1
        self.label = "test"
        self.kv_clears = 0
        self.config = atom_config_double()
        self.attn_family = Family.MHA

    def clear_kv_cache(self):
        self.kv_clears += 1


def _trainer_tensors(seed, dtype=torch.bfloat16):
    """What the trainer sends: one tensor per (expert, projection)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = []
    for expert_id in range(NUM_EXPERTS):
        for proj, shape in (
            ("gate_proj", (INTERMEDIATE, HIDDEN)),
            ("up_proj", (INTERMEDIATE, HIDDEN)),
            ("down_proj", (HIDDEN, INTERMEDIATE)),
        ):
            tensors.append(
                (
                    f"layers.0.mlp.experts.{expert_id}.{proj}.weight",
                    torch.randn(shape, generator=generator).to(dtype),
                )
            )
    return tensors


def _load_from_scratch(tensors, dtype=torch.bfloat16):
    """The route a fresh model start takes: load every expert, then process.

    The reference the sync has to match. `process_weights_after_loading` for
    an unquantized MoE is `shuffle_weights` over the whole buffer.
    """
    moe = _FusedMoEDouble(dtype=dtype)
    mapping = {
        weight_name: (expert_id, shard_id)
        for _param, weight_name, expert_id, shard_id in (
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=NUM_EXPERTS,
            )
        )
    }
    for name, tensor in tensors:
        for fragment, (expert_id, shard_id) in mapping.items():
            if fragment in name:
                param = moe.w2_weight if shard_id == "w2" else moe.w13_weight
                moe.weight_loader(
                    param,
                    tensor.to(DEVICE),
                    weight_name=name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                break
    shuffle_weights(moe.w13_weight, moe.w2_weight)
    return moe


def _sync_target(dtype=torch.bfloat16):
    """A layer already through its post-load processing, holding old weights."""
    moe = _load_from_scratch(_trainer_tensors(999, dtype), dtype)
    moe.quant_method.post_load_calls = 0
    return moe


# ── routing and layout ────────────────────────────────────────────────────


def test_every_expert_tensor_is_matched():
    """96 tensors per sync on Qwen3-30B-A3B used to land in the skipped count."""
    moe = _sync_target()
    runner = _Runner(_ModelDouble(moe))

    updated = runner.update_weights(_trainer_tensors(1))

    assert updated == 3 * NUM_EXPERTS


def test_a_synced_layer_is_byte_identical_to_a_loaded_one():
    """Values *and* layout. Comparing counters cannot see the permutation."""
    tensors = _trainer_tensors(2)
    reference = _load_from_scratch(tensors)
    moe = _sync_target()

    _Runner(_ModelDouble(moe)).update_weights(tensors)

    assert torch.equal(moe.w13_weight.data, reference.w13_weight.data)
    assert torch.equal(moe.w2_weight.data, reference.w2_weight.data)


def test_two_consecutive_syncs_both_land():
    """The first sync succeeding says nothing about the second.

    Re-running the post-load hook passes this for an unquantized layer by
    luck -- a second whole-buffer shuffle is a third layout, not the loaded
    one -- so assert against a freshly loaded reference each round.
    """
    moe = _sync_target()
    runner = _Runner(_ModelDouble(moe))

    for seed in (3, 4):
        tensors = _trainer_tensors(seed)
        runner.update_weights(tensors)
        reference = _load_from_scratch(tensors)
        assert torch.equal(moe.w13_weight.data, reference.w13_weight.data)
        assert torch.equal(moe.w2_weight.data, reference.w2_weight.data)


def test_the_buffers_keep_their_addresses_across_syncs():
    """Decode graphs captured these two pointers."""
    moe = _sync_target()
    runner = _Runner(_ModelDouble(moe))
    before = (moe.w13_weight.data_ptr(), moe.w2_weight.data_ptr())

    for seed in (5, 6):
        runner.update_weights(_trainer_tensors(seed))

    assert (moe.w13_weight.data_ptr(), moe.w2_weight.data_ptr()) == before


def test_the_param_to_module_cache_still_points_at_the_live_buffers():
    """A hook that rebinds Parameters leaves this cache holding dead objects."""
    moe = _sync_target()
    runner = _Runner(_ModelDouble(moe))
    runner.update_weights(_trainer_tensors(7))

    cached = runner._get_param_to_module_mapping()["layers.0.mlp.experts.w13_weight"]

    assert cached[2] is moe.w13_weight


def test_the_post_load_hook_is_not_re_run():
    moe = _sync_target()

    _Runner(_ModelDouble(moe)).update_weights(_trainer_tensors(8))

    assert moe.quant_method.post_load_calls == 0


def test_an_expert_nobody_sent_is_left_alone():
    """Relayout is per slice. Shuffling an untouched expert corrupts it."""
    moe = _sync_target()
    untouched = moe.w13_weight.data[3].clone()
    tensors = [
        (name, tensor)
        for name, tensor in _trainer_tensors(9)
        if ".experts.3." not in name
    ]

    updated = _Runner(_ModelDouble(moe)).update_weights(tensors)

    assert updated == 3 * (NUM_EXPERTS - 1)
    assert torch.equal(moe.w13_weight.data[3], untouched)


def test_a_half_rewritten_expert_is_refused():
    """w1 without w3: the slice is half new and half old in two layouts."""
    moe = _sync_target()
    tensors = [
        (name, tensor) for name, tensor in _trainer_tensors(10) if "up_proj" not in name
    ]

    with pytest.raises(RuntimeError, match="without \\['w3'\\]"):
        _Runner(_ModelDouble(moe)).update_weights(tensors)


def test_a_dense_model_is_unaffected():
    """No get_expert_mapping: every lookup short-circuits and the gate loads."""
    moe = _sync_target()
    model = _ModelDouble(moe, with_expert_mapping=False)
    runner = _Runner(model)
    gate = torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16)

    updated = runner.update_weights([("layers.0.mlp.gate.weight", gate)])

    assert updated == 1
    assert torch.equal(model.layers[0].mlp.gate.weight.data.cpu(), gate)


# ── the combinations this path does not implement ─────────────────────────


def _expert_bytes(moe) -> list:
    return [
        getattr(moe, name).detach().view(torch.uint8).cpu().tolist()
        for name in ("w13_weight", "w2_weight")
    ]


def test_a_quantized_expert_buffer_is_refused_rather_than_miscopied():
    """bf16 into an FP8 buffer is a dtype cast with no new scale.

    `_copy_quant_storage` byte-copies between FP8 variants and otherwise
    numerically casts, so the weight changes while the scale still describes
    the old one -- and the sync reports updated. Measured before this guard:
    weight 1.0 with an existing scale of 0.125 decoded back as 0.125, an
    absolute error of 0.875, with updated=1 in the log.
    """
    moe = _FusedMoEDouble(dtype=dtypes.fp8)
    runner = _Runner(_ModelDouble(moe))
    before = _expert_bytes(moe)

    with pytest.raises(WeightReloadRefused, match="full-precision write cannot"):
        runner.update_weights(_trainer_tensors(11))
    assert _expert_bytes(moe) == before, "refused before anything was written"


def test_an_expert_parallel_layer_is_refused():
    """Expert ids then address local slots, and only some arrive here."""
    moe = _FusedMoEDouble(
        expert_map=torch.tensor([0, 1, -1, -1], dtype=torch.int32, device=DEVICE)
    )
    runner = _Runner(_ModelDouble(moe))
    before = _expert_bytes(moe)

    with pytest.raises(WeightReloadRefused, match="expert-parallel"):
        runner.update_weights(_trainer_tensors(12))
    assert _expert_bytes(moe) == before, "refused before anything was written"


def test_redundant_expert_replicas_are_refused():
    moe = _FusedMoEDouble()
    moe.num_redundant_experts = 2
    runner = _Runner(_ModelDouble(moe))
    before = _expert_bytes(moe)

    with pytest.raises(WeightReloadRefused, match="redundant"):
        runner.update_weights(_trainer_tensors(13))
    assert _expert_bytes(moe) == before, "refused before anything was written"


def test_a_buffer_that_is_not_an_expert_weight_is_refused():
    """An expert scale resolving to a real parameter must not be written."""
    moe = _sync_target()
    moe.w13_weight_scale = nn.Parameter(
        torch.ones(NUM_EXPERTS, 2, device=DEVICE), requires_grad=False
    )
    runner = _Runner(_ModelDouble(moe))
    scale = torch.ones(1, dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="w13_weight_scale"):
        runner.update_weights(
            [("layers.0.mlp.experts.0.gate_proj.weight_scale", scale)]
        )


def _torch_per_tensor_quant(x, scale, quant_dtype=None):
    return (x / scale).to(quant_dtype or dtypes.fp8), scale


def test_the_fp8_post_load_hook_cannot_be_run_twice(monkeypatch):
    """Why the sync does not re-run process_weights_after_loading.

    The real per-tensor FP8 path, not a stand-in that counts calls. It folds
    w13_weight_scale from [E, 2] down to [E] and then reads [:, 1] on the way
    back in, so the second call raises on a shape its first call produced.

    aiter's per-tensor quant kernel stands in as its torch equivalent: its JIT
    build is not present in every image, and it is not what is under test.
    The scale fold, the shuffle and the rebind around it are production code.
    """
    monkeypatch.setattr(
        moe_module, "get_hip_quant", lambda _quant_type: _torch_per_tensor_quant
    )
    method = object.__new__(Fp8MoEMethod)
    method.quant_config = SimpleNamespace(is_dynamic=True)
    method.quant_type = QuantType.per_Tensor
    method.block_quant = False
    method.channel_quant = False
    method.need_normalize_e4m3fn_to_e4m3fnuz = False

    experts = 2
    layer = SimpleNamespace(
        w13_weight=nn.Parameter(
            torch.zeros(
                experts, 2 * INTERMEDIATE, HIDDEN, dtype=dtypes.fp8, device=DEVICE
            ),
            requires_grad=False,
        ),
        w2_weight=nn.Parameter(
            torch.zeros(experts, HIDDEN, INTERMEDIATE, dtype=dtypes.fp8, device=DEVICE),
            requires_grad=False,
        ),
        w13_weight_scale=nn.Parameter(
            torch.full((experts, 2), 0.125, device=DEVICE), requires_grad=False
        ),
        w2_weight_scale=nn.Parameter(
            torch.full((experts,), 0.125, device=DEVICE), requires_grad=False
        ),
        w13_input_scale=None,
        w2_input_scale=None,
        intermediate_size_per_partition=INTERMEDIATE,
        local_num_experts=experts,
    )

    method.process_weights_after_loading(layer)
    assert layer.w13_weight_scale.shape == (experts,)

    with pytest.raises(IndexError):
        method.process_weights_after_loading(layer)


# ── the fused naming a transformers-5.x trainer emits ─────────────────────


def _fused_trainer_tensors(seed, dtype=torch.bfloat16):
    """The same weights, as one 3D tensor per layer per projection."""
    per_expert = dict(_trainer_tensors(seed, dtype))
    gate_up = torch.stack(
        [
            torch.cat(
                [
                    per_expert[f"layers.0.mlp.experts.{e}.gate_proj.weight"],
                    per_expert[f"layers.0.mlp.experts.{e}.up_proj.weight"],
                ],
                dim=0,
            )
            for e in range(NUM_EXPERTS)
        ]
    )
    down = torch.stack(
        [
            per_expert[f"layers.0.mlp.experts.{e}.down_proj.weight"]
            for e in range(NUM_EXPERTS)
        ]
    )
    return [
        ("layers.0.mlp.experts.gate_up_proj", gate_up),
        ("layers.0.mlp.experts.down_proj", down),
    ]


def test_a_fused_expert_tensor_lands_where_the_per_expert_ones_do():
    """Byte-identical to the per-expert route, and to a fresh load.

    A caller whose trainer fuses its experts should not have to rename them,
    split them, or apply the aiter permutation on ATOM's behalf -- which is
    what it takes today, because the fused name reaches `named_parameters()`
    only after the caller has renamed it, and then lands on the plain-copy
    path with no layout step at all.
    """
    reference = _load_from_scratch(_trainer_tensors(16))
    moe = _sync_target()

    updated = _Runner(_ModelDouble(moe)).update_weights(_fused_trainer_tensors(16))

    assert updated == 2
    assert torch.equal(moe.w13_weight.data, reference.w13_weight.data)
    assert torch.equal(moe.w2_weight.data, reference.w2_weight.data)


def test_two_consecutive_fused_syncs_both_land():
    moe = _sync_target()
    runner = _Runner(_ModelDouble(moe))

    for seed in (17, 18):
        runner.update_weights(_fused_trainer_tensors(seed))
        reference = _load_from_scratch(_trainer_tensors(seed))
        assert torch.equal(moe.w13_weight.data, reference.w13_weight.data)
        assert torch.equal(moe.w2_weight.data, reference.w2_weight.data)


def test_the_fused_path_keeps_the_buffer_addresses():
    moe = _sync_target()
    runner = _Runner(_ModelDouble(moe))
    before = (moe.w13_weight.data_ptr(), moe.w2_weight.data_ptr())

    runner.update_weights(_fused_trainer_tensors(19))

    assert (moe.w13_weight.data_ptr(), moe.w2_weight.data_ptr()) == before


def test_a_fused_name_that_is_not_3d_is_refused():
    moe = _sync_target()
    runner = _Runner(_ModelDouble(moe))
    flat = torch.zeros(2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="3D"):
        runner.update_weights([("layers.0.mlp.experts.gate_up_proj", flat)])


def test_a_fused_name_on_a_quantized_layer_is_refused():
    moe = _FusedMoEDouble(dtype=dtypes.fp8)
    runner = _Runner(_ModelDouble(moe))
    before = _expert_bytes(moe)

    with pytest.raises(WeightReloadRefused, match="full-precision write cannot"):
        runner.update_weights(_fused_trainer_tensors(20))
    assert _expert_bytes(moe) == before, "refused before anything was written"


def test_a_dense_gate_up_proj_is_not_mistaken_for_an_expert_one():
    """`.experts` is what makes the leaf an expert leaf.

    A dense MLP's gate_up_proj shares the leaf name and must keep going
    through packed_modules_mapping.
    """
    moe = _sync_target()
    runner = _Runner(_ModelDouble(moe))

    result = runner._apply_expert_weight(
        "layers.0.mlp.gate_up_proj",
        torch.zeros(4, 4, 4, dtype=torch.bfloat16),
        runner._get_param_to_module_mapping(),
    )

    assert result == "skipped"


# ── the transports ────────────────────────────────────────────────────────


def _pack(tensors):
    """One shared-memory bucket, laid out the way LLMEngine lays one out."""
    meta = {}
    offset = 0
    for name, tensor in tensors:
        contiguous = tensor.contiguous()
        nbytes = contiguous.numel() * contiguous.element_size()
        meta[name] = {
            "shape": tuple(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "offset": offset,
            "nbytes": nbytes,
        }
        offset += nbytes
    payload = torch.empty(offset, dtype=torch.uint8)
    for name, tensor in tensors:
        entry = meta[name]
        flat = tensor.contiguous().view(torch.uint8).reshape(-1)
        payload[entry["offset"] : entry["offset"] + entry["nbytes"]] = flat
    return meta, payload


def test_the_shm_entry_point_relayouts_only_on_the_last_bucket():
    """Split so an expert's w1 and w3 arrive in different buckets.

    Relaying out at the end of every bucket would shuffle that expert's slice
    while half of it was still the old weight.
    """
    tensors = _trainer_tensors(14)
    reference = _load_from_scratch(tensors)
    moe = _sync_target()
    runner = _Runner(_ModelDouble(moe))

    first = [t for t in tensors if "up_proj" not in t[0]]
    second = [t for t in tensors if "up_proj" in t[0]]

    for bucket, is_last in ((first, False), (second, True)):
        meta, payload = _pack(bucket)
        shm = shared_memory.SharedMemory(create=True, size=max(payload.numel(), 1))
        try:
            shm.buf[: payload.numel()] = payload.numpy().tobytes()
            updated = runner.update_weights_from_shm(shm.name, meta, is_last=is_last)
            assert updated == len(bucket)
            if not is_last:
                # Still row-major here, so it cannot match the reference yet.
                assert not torch.equal(moe.w13_weight.data, reference.w13_weight.data)
        finally:
            shm.close()
            shm.unlink()

    assert torch.equal(moe.w13_weight.data, reference.w13_weight.data)
    assert torch.equal(moe.w2_weight.data, reference.w2_weight.data)
    assert runner.kv_clears == 1


def test_the_ipc_entry_point_routes_experts_too(monkeypatch):
    """The real `update_weights_from_ipc` body.

    Only `rebuild_ipc_handle` is replaced -- opening another process's handle
    is the one thing a single-process test cannot do -- so the offset, dtype,
    view, same-device clone, is_last teardown and relayout are all the
    production code.
    """
    tensors = _trainer_tensors(15)
    reference = _load_from_scratch(tensors)
    moe = _sync_target()
    runner = _Runner(_ModelDouble(moe))

    meta, payload = _pack(tensors)
    buffer = payload.to(DEVICE)
    monkeypatch.setattr(
        weight_sync, "rebuild_ipc_handle", lambda handle, device_id=None: buffer
    )

    updated = runner.update_weights_from_ipc(
        ipc_handle=None, bucket_meta=meta, is_last=True
    )

    assert updated == len(tensors)
    assert torch.equal(moe.w13_weight.data, reference.w13_weight.data)
    assert torch.equal(moe.w2_weight.data, reference.w2_weight.data)
    assert runner._ipc_buffer is None


# ── FP8 experts, requantized by the sync ──────────────────────────────────

FP8_HIDDEN = 256
# Whole intermediate size; under TP2 a rank holds 128, the smallest a 128x128
# block-scaled layer accepts (`Fp8MoEMethod.create_weights`).
FP8_INTERMEDIATE = 256


def _fp8_method(quant_type):
    method = object.__new__(Fp8MoEMethod)
    method.quant_type = quant_type
    method.quant_dtype = dtypes.fp8
    method.block_quant = quant_type in (QuantType.per_1x128, QuantType.per_1x32)
    method.channel_quant = quant_type == QuantType.per_Token
    method.need_normalize_e4m3fn_to_e4m3fnuz = False
    return method


class _Fp8FusedMoEDouble(_FusedMoEDouble):
    """An online-quantized FusedMoE's FP8 buffers, after its post-load processing."""

    requantizes_synced_experts = FusedMoE.requantizes_synced_experts
    requantize_expert_shard = FusedMoE.requantize_expert_shard
    _load_quant_weight_scale = FusedMoE._load_quant_weight_scale
    _SYNC_REQUANTIZED_TYPES = FusedMoE._SYNC_REQUANTIZED_TYPES

    def __init__(self, quant_type, *, tp_size=1, tp_rank=0):
        nn.Module.__init__(self)
        inter = FP8_INTERMEDIATE // tp_size
        self.w13_weight = nn.Parameter(
            torch.zeros(
                NUM_EXPERTS, 2 * inter, FP8_HIDDEN, dtype=dtypes.fp8, device=DEVICE
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(
                NUM_EXPERTS, FP8_HIDDEN, inter, dtype=dtypes.fp8, device=DEVICE
            ),
            requires_grad=False,
        )
        if quant_type == QuantType.per_Token:
            scales = ((NUM_EXPERTS, 2 * inter), (NUM_EXPERTS, FP8_HIDDEN))
        else:
            scales = (
                (NUM_EXPERTS, 2 * inter // 128, FP8_HIDDEN // 128),
                (NUM_EXPERTS, FP8_HIDDEN // 128, inter // 128),
            )
        self.w13_weight_scale = nn.Parameter(
            torch.ones(scales[0], device=DEVICE), requires_grad=False
        )
        self.w2_weight_scale = nn.Parameter(
            torch.ones(scales[1], device=DEVICE), requires_grad=False
        )
        self.layer_quant_config = SimpleNamespace(
            quant_dtype=dtypes.fp8, quant_type=quant_type
        )
        self.quant_method = _fp8_method(quant_type)
        self.layer_name = "layers.0.mlp.experts"
        self.intermediate_size_per_partition = inter
        self.expert_map = None
        self.num_redundant_experts = 0
        self.num_fused_shared_experts = 0
        self.global_num_experts = NUM_EXPERTS
        self.local_num_experts = NUM_EXPERTS
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.use_ep = False


def _fp8_trainer_tensors(seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = []
    for expert_id in range(NUM_EXPERTS):
        for proj, shape in (
            ("gate_proj", (FP8_INTERMEDIATE, FP8_HIDDEN)),
            ("up_proj", (FP8_INTERMEDIATE, FP8_HIDDEN)),
            ("down_proj", (FP8_HIDDEN, FP8_INTERMEDIATE)),
        ):
            tensors.append(
                (
                    f"layers.0.mlp.experts.{expert_id}.{proj}.weight",
                    (torch.randn(shape, generator=generator) * 0.02).to(torch.bfloat16),
                )
            )
    return tensors


def _quantize(weight, quant_type):
    return quant_weight_online(
        weight.to(DEVICE).contiguous(),
        online_quant_type=quant_type,
        online_quant_dtype=dtypes.fp8,
    )


def _fp8_reference(tensors, quant_type, *, tp_size=1, tp_rank=0):
    """What the load's online quantization leaves for these weights, worked out
    without the layer: this rank's rows of w1 and w3 and columns of w2, each
    quantized on its own -- except a per-channel w2 under TP, scaled over the
    whole row the load gathers -- stacked, then shuffled as the load does."""
    weights = dict(tensors)
    rows = FP8_INTERMEDIATE // tp_size
    own = slice(rows * tp_rank, rows * (tp_rank + 1))
    w13, w13_scale, w2, w2_scale = [], [], [], []
    for e in range(NUM_EXPERTS):
        halves = [
            _quantize(
                weights[f"layers.0.mlp.experts.{e}.{proj}.weight"][own], quant_type
            )
            for proj in ("gate_proj", "up_proj")
        ]
        w13.append(torch.cat([q for q, _ in halves]))
        w13_scale.append(torch.cat([s for _, s in halves]))
        down = weights[f"layers.0.mlp.experts.{e}.down_proj.weight"]
        if quant_type == QuantType.per_Token and tp_size > 1:
            q, s = _quantize(down, quant_type)
            q = q[:, own]
        else:
            q, s = _quantize(down[:, own], quant_type)
        w2.append(q)
        w2_scale.append(s)
    w13 = nn.Parameter(torch.stack(w13), requires_grad=False)
    w2 = nn.Parameter(torch.stack(w2), requires_grad=False)
    shuffle_weights(w13, w2)
    squeeze = quant_type == QuantType.per_Token
    return (
        w13.data,
        torch.stack([s.squeeze(-1) if squeeze else s for s in w13_scale]),
        w2.data,
        torch.stack([s.squeeze(-1) if squeeze else s for s in w2_scale]),
    )


def _fp8_sync_target(quant_type, **tp):
    """A layer holding some other FP8 weights, through the post-load shuffle."""
    moe = _Fp8FusedMoEDouble(quant_type, **tp)
    old = _fp8_reference(_fp8_trainer_tensors(999), quant_type, **tp)
    for param, value in zip(
        (moe.w13_weight, moe.w13_weight_scale, moe.w2_weight, moe.w2_weight_scale),
        old,
    ):
        param.data.copy_(value)
    return moe


def _fp8_state(moe):
    return (
        moe.w13_weight.data.view(torch.uint8),
        moe.w13_weight_scale.data,
        moe.w2_weight.data.view(torch.uint8),
        moe.w2_weight_scale.data,
    )


def _assert_matches(moe, reference):
    for got, want in zip(_fp8_state(moe), reference):
        want = want.view(torch.uint8) if want.element_size() == 1 else want
        assert torch.equal(got, want)


_FP8_CASES = [
    pytest.param(QuantType.per_1x128, {}, id="per_1x128"),
    pytest.param(QuantType.per_Token, {}, id="per_Token"),
    pytest.param(QuantType.per_1x128, {"tp_size": 2, "tp_rank": 1}, id="per_1x128-tp2"),
    pytest.param(QuantType.per_Token, {"tp_size": 2, "tp_rank": 1}, id="per_Token-tp2"),
]


@pytest.mark.parametrize("quant_type, tp", _FP8_CASES)
def test_an_fp8_sync_matches_a_fresh_online_quantization(quant_type, tp):
    """Weights, scales and layout, byte for byte. Before the sync could
    requantize, these weights were cast into the buffers under the old scales
    -- or, since #3058, refused."""
    tensors = _fp8_trainer_tensors(21)
    moe = _fp8_sync_target(quant_type, **tp)

    updated = _Runner(_ModelDouble(moe)).update_weights(tensors)

    assert updated == 3 * NUM_EXPERTS
    _assert_matches(moe, _fp8_reference(tensors, quant_type, **tp))


@pytest.mark.parametrize("quant_type, tp", _FP8_CASES[:2])
def test_two_fp8_syncs_both_land_in_the_same_buffers(quant_type, tp):
    moe = _fp8_sync_target(quant_type, **tp)
    runner = _Runner(_ModelDouble(moe))
    before = [t.data_ptr() for t in _fp8_state(moe)]

    for seed in (22, 23):
        tensors = _fp8_trainer_tensors(seed)
        runner.update_weights(tensors)
        _assert_matches(moe, _fp8_reference(tensors, quant_type, **tp))

    assert [t.data_ptr() for t in _fp8_state(moe)] == before


@pytest.mark.parametrize("quant_type, tp", _FP8_CASES)
def test_fused_fp8_tensors_land_where_the_per_expert_ones_do(quant_type, tp):
    """A fused tensor is this rank's shard: under TP2, its rows of gate and of
    up, and its columns of down. A per-channel w2 under TP is scaled over the
    whole row, which that shard does not hold (refused below), so there
    down_proj goes one tensor per expert."""
    per_expert = _fp8_trainer_tensors(24)
    weights = dict(per_expert)
    tp_size, tp_rank = tp.get("tp_size", 1), tp.get("tp_rank", 0)
    rows = FP8_INTERMEDIATE // tp_size
    own = slice(rows * tp_rank, rows * (tp_rank + 1))

    def expert(e, proj):
        return weights[f"layers.0.mlp.experts.{e}.{proj}.weight"]

    gate_up = torch.stack(
        [
            torch.cat([expert(e, "gate_proj")[own], expert(e, "up_proj")[own]])
            for e in range(NUM_EXPERTS)
        ]
    )
    if quant_type == QuantType.per_Token and tp_size > 1:
        down = [(name, t) for name, t in per_expert if ".down_proj." in name]
    else:
        down = [
            (
                "layers.0.mlp.experts.down_proj",
                torch.stack(
                    [expert(e, "down_proj")[:, own] for e in range(NUM_EXPERTS)]
                ),
            )
        ]
    moe = _fp8_sync_target(quant_type, **tp)

    _Runner(_ModelDouble(moe)).update_weights(
        [("layers.0.mlp.experts.gate_up_proj", gate_up), *down]
    )

    _assert_matches(moe, _fp8_reference(per_expert, quant_type, **tp))


def test_a_global_fused_tensor_under_tp_is_refused():
    """`requantize_expert_shard` takes each expert's leading rows of a
    rank-local shard, so a global gate_up_proj was written as rank 0's rows on
    rank 1 too, with nothing said."""
    moe = _fp8_sync_target(QuantType.per_1x128, tp_size=2, tp_rank=1)
    before = [t.clone() for t in _fp8_state(moe)]
    gate_up = torch.ones(
        NUM_EXPERTS, 2 * FP8_INTERMEDIATE, FP8_HIDDEN, dtype=torch.bfloat16
    )

    with pytest.raises(NotImplementedError, match="this rank's"):
        _Runner(_ModelDouble(moe)).update_weights(
            [("layers.0.mlp.experts.gate_up_proj", gate_up)]
        )
    assert all(torch.equal(a, b) for a, b in zip(_fp8_state(moe), before))


def test_an_fp8_expert_sent_to_an_fp8_layer_is_refused():
    """Requantized, its FP8 values would be read as the weight and the scale
    they were stored under dropped; experts' scales are not synced."""
    moe = _fp8_sync_target(QuantType.per_1x128)
    before = [t.clone() for t in _fp8_state(moe)]
    tensors = [(name, t.to(dtypes.fp8)) for name, t in _fp8_trainer_tensors(26)]

    with pytest.raises(NotImplementedError, match="requantizes from full precision"):
        _Runner(_ModelDouble(moe)).update_weights(tensors)
    assert all(torch.equal(a, b) for a, b in zip(_fp8_state(moe), before))


def test_a_rank_local_per_channel_w2_under_tp_is_refused():
    """A fused down_proj holds this rank's columns; a per-channel scale needs
    the whole row."""
    moe = _fp8_sync_target(QuantType.per_Token, tp_size=2, tp_rank=1)
    down = torch.zeros(
        NUM_EXPERTS, FP8_HIDDEN, FP8_INTERMEDIATE // 2, dtype=torch.bfloat16
    )

    with pytest.raises(NotImplementedError, match="whole row"):
        _Runner(_ModelDouble(moe)).update_weights(
            [("layers.0.mlp.experts.down_proj", down)]
        )


def test_an_mxfp8_layer_is_still_refused():
    """Gate/up-interleaved weights with shuffled E8M0 scales: not a layout a
    sync rewrites one shard at a time."""
    moe = _Fp8FusedMoEDouble(QuantType.per_1x128)
    moe.quant_method = _fp8_method(QuantType.per_1x32)
    before = _expert_bytes(moe)

    with pytest.raises(WeightReloadRefused, match="full-precision write cannot"):
        _Runner(_ModelDouble(moe)).update_weights(_fp8_trainer_tensors(25))
    assert _expert_bytes(moe) == before, "refused before anything was written"
