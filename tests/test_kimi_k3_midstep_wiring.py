# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Kimi-K3's midstep checkpoints, wired from the builder to the kernels.

`test_kda_midstep_snapshots.py` pins what `atom.models.kimi_k3.midstep`
computes; this pins where it is called. Both modules involved import AITER at
load, so they are imported against `aiter_stub`'s stand-ins for AITER and
Triton, and run on the CPU:

  * `_KimiMLAGDNCommon.finish_prefill` attaches the step's `KdaSnapshots` to
    the GDN metadata its parent built: interior targets and an end target, an
    end target alone, or none (a step without reservations, or the switch off,
    where it does not ask aiter for the grid);
  * `KimiKDAAttention._forward_impl`'s prefill path hands the interior targets
    and a `snapshot_out` buffer to `chunk_kimi_delta_attn` through `_run_kda`,
    then gives `write_state_checkpoints` that buffer with the targets' `h_rows`,
    once the runtime slots hold the final state. Without interior targets the
    kernel gets no snapshot arguments at all.

Only the kernels, the conv and gather ops, and the parent builder hook are
stand-ins. `einops` is not an ATOM dependency and ATOM's CPU job does not
install it; where it is missing, a stand-in serves the three patterns this path
uses.
"""

import dataclasses
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_under_stubs

H, HD = 2, 4  # KDA heads and head dim of the layer under test
LP = H * HD  # the q / k / v width
CONV_KERNEL = 4
NUM_SLOTS = 9
GRID = 32  # aiter's FLASH_KDA_CHUNK, which the builder reads from aiter
# Row 0 resumes at token 64 and runs 96 more: interior checkpoints 32 and 64
# tokens in, and one at its end. Row 1 is a fresh 48-token prompt with one 32
# tokens in. Row 2 reserved none. Positions are absolute, as the scheduler
# reserves them.
CACHED = [64, 0, 0]
SCHEDULED = [96, 48, 20]
RUNTIME = [1, 2, 3]
SAVES = [[(4, 96), (5, 128), (6, 160)], [(7, 32)], []]
END_ONLY = [[(6, 160)], [], []]
NO_SAVES = [[], [], []]
T = sum(SCHEDULED)

_PATTERNS = {
    "t (h d) -> 1 t h d": lambda x, d: x.reshape(1, x.shape[0], -1, d),
    "t (h d) -> t h d": lambda x, d: x.reshape(x.shape[0], -1, d),
    "t h d -> t (h d)": lambda x: x.reshape(x.shape[0], -1),
}
_ABSENT = object()


def _rearrange(tensor, pattern, **axes):
    if pattern not in _PATTERNS:
        raise NotImplementedError(f"the einops stand-in has no {pattern!r}")
    return _PATTERNS[pattern](tensor, **axes)


@contextmanager
def _einops_if_missing():
    try:
        import einops  # noqa: F401
    except ImportError:
        pass
    else:
        yield
        return
    previous = sys.modules.get("einops", _ABSENT)
    stand_in = types.ModuleType("einops")
    stand_in.rearrange = _rearrange
    sys.modules["einops"] = stand_in
    try:
        yield
    finally:
        if previous is _ABSENT:
            sys.modules.pop("einops", None)
        else:
            sys.modules["einops"] = previous


@pytest.fixture(scope="module")
def k3():
    with _einops_if_missing(), atom_under_stubs() as load:
        yield SimpleNamespace(
            backend=load("atom.model_ops.attentions.kimi_mla_gdn_attn"),
            gdn=load("atom.model_ops.attentions.gdn_attn"),
            model=load("atom.models.kimi_k3.model"),
            ops=load("atom.model_ops.kimi_k3"),
            kda=load("aiter.ops.triton.kimi_delta_attn"),
        )


# ── the builder ──────────────────────────────────────────────────────────────


def _no_grid():
    raise ImportError("an aiter without ROCm/aiter#7413 has no FLASH_KDA_CHUNK")


def finish_prefill(k3, monkeypatch, saves, *, builds_gdn_metadata=True, switch=True):
    """`finish_prefill` on a Kimi builder holding only what it reads, over a
    parent hook that builds the GDN metadata as `GDNStateMixin` does (or leaves
    None, for a prefill without block tables). Returns that metadata. With the
    switch off, asking aiter for the grid raises."""
    monkeypatch.setenv("ATOM_K3_KDA_STATE_MIDSTEP", "1" if switch else "0")
    monkeypatch.setattr(
        k3.backend, "flash_kda_chunk", (lambda: GRID) if switch else _no_grid
    )
    cls = k3.backend.KimiAiterMLAGDNMetadataBuilder
    mro = cls.__mro__
    parent = next(
        c
        for c in mro[mro.index(k3.backend._KimiMLAGDNCommon) + 1 :]
        if "finish_prefill" in vars(c)
    )
    calls = []

    def parent_finish_prefill(self, batch, attn_metadata, running_bs, running_tokens):
        calls.append(batch)
        if builds_gdn_metadata:
            attn_metadata.gdn_metadata = SimpleNamespace(kda_snapshots=None)

    monkeypatch.setattr(parent, "finish_prefill", parent_finish_prefill)
    builder = object.__new__(cls)
    builder.model_runner = SimpleNamespace(
        mamba_k_cache=SimpleNamespace(shape=(2, NUM_SLOTS)),
        config=SimpleNamespace(
            hf_config=SimpleNamespace(linear_conv_kernel_dim=CONV_KERNEL)
        ),
    )
    builder.device = torch.device("cpu")
    batch = SimpleNamespace(
        state_save_all=saves,
        state_rows=[0, 1, 2],
        num_cached_tokens=CACHED,
        num_scheduled_tokens=SCHEDULED,
        state_slots_committed=RUNTIME,
    )
    attn_metadata = SimpleNamespace(gdn_metadata=None)
    builder.finish_prefill(batch, attn_metadata, len(SCHEDULED), T)
    assert calls == [batch], "the parent hook runs, once, before the attach"
    return attn_metadata.gdn_metadata


def test_the_builder_attaches_interior_and_end_targets(k3, monkeypatch):
    snaps = finish_prefill(k3, monkeypatch, SAVES).kda_snapshots
    # Interior targets first, in the order the kernel snapshots them.
    assert snaps.targets == ((0, 32), (0, 64), (1, 32))
    assert snaps.rows.tolist() == [0, 0, 1, 0]
    assert snaps.slots.tolist() == [4, 5, 7, 6]
    assert snaps.offs.tolist() == [32, 64, 32, 96]
    assert snaps.is_end.tolist() == [0, 0, 0, 1]
    assert snaps.runtime_slots.tolist() == [1, 1, 2, 1]
    assert snaps.h_rows.tolist() == [0, 1, 2, 0]
    assert snaps.cu_seqlens.tolist() == [0, 96, 144, 164]
    assert snaps.conv_len == CONV_KERNEL - 1


def test_a_step_without_reservations_attaches_none(k3, monkeypatch):
    assert finish_prefill(k3, monkeypatch, NO_SAVES).kda_snapshots is None


def test_with_the_switch_off_the_builder_does_not_ask_aiter_for_the_grid(
    k3, monkeypatch
):
    metadata = finish_prefill(k3, monkeypatch, NO_SAVES, switch=False)
    assert metadata.kda_snapshots is None


def test_a_prefill_without_gdn_metadata_is_left_alone(k3, monkeypatch):
    assert finish_prefill(k3, monkeypatch, SAVES, builds_gdn_metadata=False) is None


def test_the_gdn_metadata_declares_the_field(k3):
    """The KDA layer reads `kda_snapshots` directly, so native ATOM's GDN
    metadata declares it, None unless the builder attached a step's."""
    fields = {f.name: f for f in dataclasses.fields(k3.gdn.GDNAttentionMetadata)}
    assert fields["kda_snapshots"].default is None


# ── the KDA layer ────────────────────────────────────────────────────────────


def _layer(k3):
    """A `KimiKDAAttention` holding only what its prefill path reads."""
    layer = k3.model.KimiKDAAttention.__new__(k3.model.KimiKDAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_num = 0
    layer.local_proj_size = LP
    layer.num_local_heads = H
    layer.head_dim = HD
    layer.fuse_f_a_quant = False
    layer.activation = "silu"
    layer.conv_weight = torch.zeros(3 * LP, CONV_KERNEL)
    layer.A_log = torch.zeros(H)
    layer.dt_bias = torch.zeros(H * HD)
    layer._kda_gate_lower_bound = None
    fused = torch.randn(T, 4 * LP + H + HD, generator=torch.Generator().manual_seed(0))
    layer.in_proj = lambda x, x_scale=None: fused
    layer.f_b_proj = lambda x, x_scale=None: torch.zeros(x.shape[0], H * HD)
    layer.o_norm = lambda out, gate, is_prefill=False: out
    layer.o_proj = lambda x, x_scale=None: x
    return layer, fused


def run_prefill(k3, monkeypatch, snaps):
    """One prefill forward of the layer over the three rows, with `snaps` as
    the GDN metadata's `kda_snapshots`; the kernels record their arguments."""
    layer, fused = _layer(k3)
    ssm = torch.randn(NUM_SLOTS, H, HD, HD, generator=torch.Generator().manual_seed(1))
    conv = torch.zeros(NUM_SLOTS, 3 * LP, CONV_KERNEL - 1 + 2)
    metadata = SimpleNamespace(
        num_actual_tokens=T,
        num_prefills=len(SCHEDULED),
        num_decodes=0,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor(RUNTIME),
        non_spec_state_indices_in_tensor=None,
        non_spec_query_start_loc=torch.tensor([0, 96, 144, 164], dtype=torch.int32),
        has_initial_state=torch.tensor([True, False, False]),
        kda_snapshots=snaps,
    )
    context = SimpleNamespace(
        attn_metadata=SimpleNamespace(gdn_metadata=metadata),
        kv_cache_data={"layer_0": SimpleNamespace(k_cache=conv, v_cache=ssm)},
    )
    run = SimpleNamespace(
        kernel=[], writer=[], runtime_at_write=[], ssm=ssm, conv=conv, fused=fused
    )

    def chunk_kimi_delta_attn(**kw):
        run.kernel.append(kw)
        if "snapshot_out" in kw:
            for i in range(len(kw["snapshot_targets"])):
                kw["snapshot_out"][i] = 100 + i
        return torch.zeros(1, T, H, HD), kw["initial_state"] + 1

    def write_state_checkpoints(**kw):
        run.writer.append(kw)
        run.runtime_at_write.append(kw["ssm_state"][RUNTIME].clone())

    monkeypatch.setattr(k3.model, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        k3.model,
        "causal_conv1d_fn",
        lambda x, *args, **kwargs: (x[:LP].T, x[LP : 2 * LP].T, x[2 * LP :].T),
    )
    monkeypatch.setattr(
        k3.ops,
        "gather_kda_initial_state",
        lambda pool, indices, has_initial_state: pool[indices].clone(),
    )
    monkeypatch.setattr(k3.kda, "chunk_kimi_delta_attn", chunk_kimi_delta_attn)
    monkeypatch.setattr(k3.model, "write_state_checkpoints", write_state_checkpoints)
    run.before = ssm.clone()
    out = layer._forward_impl(torch.zeros(T, 16))
    assert out.shape == (T, H * HD)
    return run


def test_the_prefill_snapshots_its_targets_and_writes_their_rows(k3, monkeypatch):
    snaps = finish_prefill(k3, monkeypatch, SAVES).kda_snapshots
    run = run_prefill(k3, monkeypatch, snaps)

    (kernel,) = run.kernel
    assert kernel["snapshot_targets"] == ((0, 32), (0, 64), (1, 32))
    snap_out = kernel["snapshot_out"]
    # One state per interior target, in the final state's dtype and layout.
    assert snap_out.shape == (3, H, HD, HD) and snap_out.dtype == run.ssm.dtype
    assert snap_out.is_contiguous()
    assert kernel["cu_seqlens"].tolist() == [0, 96, 144, 164]

    (writer,) = run.writer
    assert writer["h"] is snap_out, "the writer reads the buffer the kernel filled"
    assert [writer["h"][i].unique().item() for i in range(3)] == [100, 101, 102]
    assert writer["h_rows"] is snaps.h_rows
    assert writer["h_rows"].tolist() == [0, 1, 2, 0]
    assert writer["slots"].tolist() == [4, 5, 7, 6]
    assert writer["is_end"].tolist() == [0, 0, 0, 1]
    assert writer["ssm_state"] is run.ssm and writer["conv_state"] is run.conv
    assert torch.equal(writer["x"], run.fused[:, : 3 * LP])
    assert writer["conv_state_len"] == CONV_KERNEL - 1
    # The end target is copied from its runtime slot, so the final state has
    # to be there before the writer runs.
    (runtime,) = run.runtime_at_write
    assert torch.equal(runtime, run.before[RUNTIME] + 1)


def test_an_end_target_alone_passes_no_snapshot_to_the_kernel(k3, monkeypatch):
    snaps = finish_prefill(k3, monkeypatch, END_ONLY).kda_snapshots
    run = run_prefill(k3, monkeypatch, snaps)
    (kernel,) = run.kernel
    assert "snapshot_targets" not in kernel and "snapshot_out" not in kernel
    (writer,) = run.writer
    assert writer["h"] is run.ssm, "without an interior target it reads the pool"
    assert writer["slots"].tolist() == [6] and writer["is_end"].tolist() == [1]


def test_without_snapshots_the_kernel_is_called_as_before(k3, monkeypatch):
    """`kda_snapshots` None: the switch off, or a step without reservations."""
    run = run_prefill(k3, monkeypatch, None)
    (kernel,) = run.kernel
    assert "snapshot_targets" not in kernel and "snapshot_out" not in kernel
    assert run.writer == []
