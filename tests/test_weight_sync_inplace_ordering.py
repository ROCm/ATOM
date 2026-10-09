# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A weight is not overwritten before its readers are done.

The update writes the parameter buffer in place so a captured decode graph
keeps reading a valid address, and the price is writing into a buffer that may
still be in use. What matters is that the wait sits with the write: waiting
once per update instead still lost five of seven weight syncs in a DAPO smoke.

Which is why there is no test here for "the wait happened". The failure that
shipped was a wait that happened and did nothing -- `_post_process_fp8_weight`
carried the only fence on the FP8 direct-copy path and ran one line *after*
`param.data.copy_` -- so every test below asserts the buffer still held its old
bytes when the fence fired, and covers each write path separately, because four
of the six had no fence at all.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom.rollout.weight_updater import WeightUpdaterMixin

# `atom.model_ops.linear` and `atom.model_ops.utils` import aiter, which
# resolves the chip architecture through rocminfo. Everything that does not
# reach them runs on a CPU box, which is what CI is.
needs_aiter = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the quant_type dispatch imports aiter, which reads the chip "
    "architecture out of rocminfo",
)

HIDDEN = 8
INTERMEDIATE = 4
EXPERTS = 2


def _updater(model=None, world_size=1):
    class _Updater(WeightUpdaterMixin):
        device = torch.device("cpu")
        label = "test"
        rank = 0

        def clear_kv_cache(self):
            pass

    updater = _Updater()
    updater.world_size = world_size
    updater.model = model if model is not None else nn.Module()
    return updater


def _record_fences(monkeypatch):
    """Capture each fenced buffer's contents at the moment its fence fired.

    A fence that runs after its write sees the new bytes; one that runs before
    sees the old ones. That difference is the bug, so it is what gets asserted.
    """
    import atom.rollout.weight_updater as wu

    seen = []
    monkeypatch.setattr(
        wu.WeightUpdaterMixin,
        "_await_readers_of",
        lambda self, param: seen.append((param, param.data.clone())),
        raising=True,
    )
    return seen


def _linear_model(shape=(4, 4), dtype=torch.bfloat16, weight_loader=None):
    """One parameter, named the way a checkpoint names it."""
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
    if weight_loader is not None:
        layer.weight_loader = weight_loader
    model = nn.Module()
    model.layer = layer
    return model, layer.weight


def _assert_fenced_before_write(seen, param, old, new):
    """The one fence named this parameter, saw the old bytes, and the new bytes
    landed after it."""
    assert [p for p, _ in seen] == [param], "expected exactly one fence, on this param"
    assert torch.equal(seen[0][1], old), "the fence ran after the write it fences"
    assert torch.equal(param.data, new), "the write did not land"


# ── update_weights: every branch of the dispatch ───────────────────────────


def test_bf16_direct_copy_waits_first(monkeypatch):
    """`param.data.copy_(tensor)` for a plain shape-match had no fence at all,
    on all three of the named-tensor, SHM and IPC entry points."""
    model, param = _linear_model()
    updater = _updater(model)
    seen = _record_fences(monkeypatch)
    old = param.data.clone()
    new = torch.full((4, 4), 7.0, dtype=torch.bfloat16)

    assert updater.update_weights([("layer.weight", new)]) == 1

    _assert_fenced_before_write(seen, param, old, new)


def test_weight_loader_fallback_waits_first(monkeypatch):
    """A module's own loader narrows and copies into the buffer it is handed,
    so its write is as in-place as ours."""
    written = []

    def weight_loader(param, tensor):
        written.append(tensor.shape)
        param.data.copy_(tensor[: param.shape[0]])

    model, param = _linear_model(weight_loader=weight_loader)
    updater = _updater(model)
    seen = _record_fences(monkeypatch)
    old = param.data.clone()
    # Not a shape match, so the dispatch falls through to the loader.
    incoming = torch.full((6, 4), 3.0, dtype=torch.bfloat16)

    assert updater.update_weights([("layer.weight", incoming)]) == 1

    assert written == [torch.Size([6, 4])]
    _assert_fenced_before_write(
        seen, param, old, torch.full((4, 4), 3.0, dtype=torch.bfloat16)
    )


def test_tp_sharded_copy_waits_first(monkeypatch):
    """`_try_shard_weight` is the last branch and wrote unfenced."""
    model, param = _linear_model()
    updater = _updater(model, world_size=2)
    seen = _record_fences(monkeypatch)
    old = param.data.clone()
    # Twice the rows, so rank 0 takes the first half.
    incoming = torch.cat(
        [
            torch.full((4, 4), 5.0, dtype=torch.bfloat16),
            torch.full((4, 4), 9.0, dtype=torch.bfloat16),
        ]
    )

    assert updater.update_weights([("layer.weight", incoming)]) == 1

    _assert_fenced_before_write(
        seen, param, old, torch.full((4, 4), 5.0, dtype=torch.bfloat16)
    )


# ── the MoE expert path, which carried no fence anywhere ───────────────────


def _moe_model(dtype=torch.bfloat16, fused_leaves=False):
    """A layer holding the fused expert buffers, plus a recording loader.

    `FusedMoE`'s real loader needs a device; what is under test here is the
    fence in front of whatever loader is reached, so a stub that writes the
    slice it is given is the right stand-in.
    """
    written = []

    def weight_loader(param, tensor, weight_name=None, shard_id=None, expert_id=0):
        written.append((shard_id, expert_id))
        lo = 0 if shard_id == "w1" else INTERMEDIATE
        rows = slice(lo, lo + INTERMEDIATE)
        if tensor.dim() == 3:
            # A fused tensor covers every expert; the loader is handed one
            # half of the intermediate dim at a time, (E, I, H).
            param.data[:, rows].copy_(tensor)
        else:
            param.data[expert_id, rows].copy_(tensor)

    experts = nn.Module()
    experts.w13_weight = nn.Parameter(
        torch.zeros(EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=dtype),
        requires_grad=False,
    )
    experts.weight_loader = weight_loader
    experts.expert_map = None
    experts.num_redundant_experts = 0
    mlp = nn.Module()
    mlp.experts = experts
    model = nn.Module()
    model.mlp = mlp
    if not fused_leaves:
        model.get_expert_mapping = lambda: [
            ("experts.w13_weight", "experts.0.gate_proj.weight", 0, "w1"),
        ]
    return model, experts.w13_weight, written


def test_per_expert_weight_waits_first(monkeypatch):
    """`_check_expert_sync_supported` requires an unquantized MoE, so the
    experts never reach `_post_process_fp8_weight` -- the helper that used to
    carry the only wait. The headline feature of the PR could not be fenced."""
    model, param, written = _moe_model()
    updater = _updater(model)
    seen = _record_fences(monkeypatch)
    old = param.data.clone()

    result = updater._apply_expert_weight(
        "mlp.experts.0.gate_proj.weight",
        torch.full((INTERMEDIATE, HIDDEN), 2.0, dtype=torch.bfloat16),
        updater._get_param_to_module_mapping(),
    )

    assert result == "updated"
    assert written == [("w1", 0)]
    assert [p for p, _ in seen] == [param]
    assert torch.equal(seen[0][1], old), "the fence ran after the loader wrote"
    assert param.data[0, :INTERMEDIATE].eq(2.0).all()


def test_fused_expert_tensor_waits_before_each_half(monkeypatch):
    """One (E, 2I, H) tensor drives the loader once per half, and both halves
    write the same live buffer."""
    model, param, written = _moe_model(fused_leaves=True)
    updater = _updater(model)
    seen = _record_fences(monkeypatch)
    old = param.data.clone()

    result = updater._apply_fused_expert_weight(
        "mlp.experts.gate_up_proj",
        torch.full((EXPERTS, 2 * INTERMEDIATE, HIDDEN), 4.0, dtype=torch.bfloat16),
        updater._get_param_to_module_mapping(),
    )

    assert result == "updated"
    assert [shard for shard, _ in written] == ["w1", "w3"]
    assert [p for p, _ in seen] == [param, param]
    assert torch.equal(seen[0][1], old), "the first half wrote before its fence"


@needs_aiter
def test_expert_relayout_waits_first(monkeypatch):
    """The relayout is the second in-place write to these slices, and the one
    a graph is most likely to catch half-done: a half-permuted expert reads as
    plausible garbage rather than as an error."""
    import atom.rollout.weight_updater as wu

    model, param, _ = _moe_model()
    updater = _updater(model)
    seen = _record_fences(monkeypatch)
    order = []
    monkeypatch.setattr(
        wu.WeightUpdaterMixin,
        "_await_readers_of",
        lambda self, p: (order.append("wait"), seen.append((p, p.data.clone())))[0],
        raising=True,
    )
    import atom.model_ops.utils as utils_mod

    monkeypatch.setattr(
        utils_mod,
        "shuffle_expert_slices",
        lambda *a, **k: order.append("shuffle"),
        raising=True,
    )
    updater._pending_expert_relayout[(model.mlp.experts, "w13_weight")] = {
        0: {"w1", "w3"}
    }

    updater._finalize_expert_weight_sync()

    assert order == ["wait", "shuffle"]
    assert [p for p, _ in seen] == [param]


# ── the FP8 post-process ──────────────────────────────────────────────────


def test_post_process_does_not_wait_when_it_writes_nothing(monkeypatch):
    """The fence belongs to the write, so a call that decides not to write
    must not fence either -- otherwise the cost is paid per parameter for a
    guarantee nothing needed."""
    seen = _record_fences(monkeypatch)
    param = nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16), requires_grad=False)
    module = SimpleNamespace(weight_scale=None, quant_type=None)

    _updater()._post_process_fp8_weight(module, param)

    assert seen == []


@needs_aiter
def test_post_process_waits_before_the_shuffle(monkeypatch):
    from aiter import QuantType, dtypes

    import atom.model_ops.utils as utils_mod

    order = []
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1")
    monkeypatch.setattr(
        utils_mod, "shuffle_weights", lambda *a, **k: order.append("shuffle")
    )
    import atom.rollout.weight_updater as wu

    monkeypatch.setattr(
        wu.WeightUpdaterMixin,
        "_await_readers_of",
        lambda self, p: order.append("wait"),
        raising=True,
    )
    param = nn.Parameter(torch.zeros(32, 64, dtype=torch.uint8), requires_grad=False)
    module = SimpleNamespace(
        weight_scale=None,
        quant_type=QuantType.per_1x128,
        params_dtype=dtypes.fp8,
        needs_preshuffled_weight=False,
        need_normalize_e4m3fn_to_e4m3fnuz=False,
    )

    _updater()._post_process_fp8_weight(module, param)

    assert order == ["wait", "shuffle"]


@needs_aiter
def test_requantize_waits_before_the_first_write(monkeypatch):
    from aiter import QuantType

    order = []
    import atom.rollout.weight_updater as wu

    monkeypatch.setattr(
        wu.WeightUpdaterMixin,
        "_await_readers_of",
        lambda self, p: order.append("wait"),
        raising=True,
    )
    monkeypatch.setattr(
        wu.WeightUpdaterMixin,
        "_post_process_fp8_weight",
        lambda self, m, p: order.append("post"),
        raising=True,
    )
    param = nn.Parameter(
        torch.zeros(4, 4, dtype=torch.float8_e4m3fnuz), requires_grad=False
    )
    module = SimpleNamespace(
        weight_scale=nn.Parameter(torch.ones(4, 1), requires_grad=False),
        quant_type=QuantType.per_Token,
    )

    _updater()._requantize_fp8_weight(
        module, "weight", param, torch.ones(4, 4, dtype=torch.float32)
    )

    assert order == ["wait", "post"], "the requantize must fence before it writes"


# ── the e4m3fnuz conversion is not a repeatable transform ─────────────────


def test_an_already_converted_weight_is_not_converted_again():
    """`need_normalize_e4m3fn_to_e4m3fnuz` is a static property of the layer --
    `params_dtype == torch.float8_e4m3fnuz`, set once in `create_weights` --
    and nothing clears it after the load. Re-running the conversion per sync
    rebuilds `weight_scale` as `scale * 2.0`, so it doubles again every time
    (3.0 -> 6.0 -> 12.0) and the dequantized weight comes out 2**N too large
    after N syncs. The fresh allocation also moves the scale's address out from
    under a captured decode graph.
    """
    param = nn.Parameter(
        torch.zeros(4, 4, dtype=torch.float8_e4m3fnuz), requires_grad=False
    )
    weight_scale = nn.Parameter(torch.full((4, 1), 3.0), requires_grad=False)
    module = SimpleNamespace(
        weight_scale=weight_scale,
        quant_type=None,
        need_normalize_e4m3fn_to_e4m3fnuz=True,
    )
    scale_ptr = weight_scale.data_ptr()
    param_ptr = param.data_ptr()

    for _ in range(3):
        _updater()._post_process_fp8_weight(module, param)

    assert weight_scale.data.flatten()[0].item() == 3.0
    assert weight_scale.data_ptr() == scale_ptr, "the scale's address moved"
    assert param.data_ptr() == param_ptr, "the weight's address moved"
    assert param.dtype == torch.float8_e4m3fnuz


@needs_aiter
def test_converting_an_e4m3fn_weight_keeps_both_addresses():
    """When the conversion does have work to do, it writes the scale in place.

    The weight's own rebind stays: `normalize_e4m3fn_to_e4m3fnuz` fixes its
    bytes through an int8 view of the same storage and hands back that storage
    with a reinterpreted dtype, so the address a captured graph holds does not
    move. The scale is a new tensor, and that one does.
    """
    param = nn.Parameter(
        torch.zeros(4, 4, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    weight_scale = nn.Parameter(torch.full((4, 1), 3.0), requires_grad=False)
    module = SimpleNamespace(
        weight_scale=weight_scale,
        quant_type=None,
        need_normalize_e4m3fn_to_e4m3fnuz=True,
    )
    scale_ptr = weight_scale.data_ptr()
    param_ptr = param.data_ptr()

    _updater()._post_process_fp8_weight(module, param)

    assert weight_scale.data.flatten()[0].item() == 6.0, "converted exactly once"
    assert weight_scale.data_ptr() == scale_ptr
    assert param.data_ptr() == param_ptr
    assert param.dtype == torch.float8_e4m3fnuz


# ── the wait itself ───────────────────────────────────────────────────────


def test_the_wait_is_a_no_op_off_device():
    """The mixin's methods run unbound on CPU stand-ins throughout the tests,
    and a host that never allocated on a device has nothing to wait for."""
    calls = []
    torch_sync = torch.cuda.synchronize
    try:
        torch.cuda.synchronize = lambda *a, **k: calls.append(a)
        WeightUpdaterMixin._await_readers_of(
            SimpleNamespace(), nn.Parameter(torch.zeros(2), requires_grad=False)
        )
    finally:
        torch.cuda.synchronize = torch_sync

    assert calls == []


def test_the_wait_targets_the_parameters_own_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: calls.append(d))

    param = SimpleNamespace(device=torch.device("cuda", 3))
    WeightUpdaterMixin._await_readers_of(SimpleNamespace(), param)

    assert calls == [torch.device("cuda", 3)]
