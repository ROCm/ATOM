# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A decode-time PAGE checkpoint under ReplaySSM carries the folded state.

Between flushes the SSM plane lags the record buffer, but the image is resumed
by prefill, which reads that plane alone. Runs the shipped store/restore path
over the real ReplaySSM, copy and chunk kernels.
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "runs the ReplaySSM and PAGE copy kernels; needs a real GPU",
        allow_module_level=True,
    )

from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    CheckpointStoreOp,
    PagedStateCheckpointSpec,
)
from atom.model_ops.attentions.backends import AttentionMetadataBuilder
from atom.model_ops.attentions.gdn_attn import GDNStateMixin
from atom.model_ops.attentions.pool_layout.page_unit_geometry import (
    PageUnitGeometryMixin,
)
from atom.model_ops.fla_ops.chunk import chunk_gated_delta_rule
from atom.model_ops.fla_ops.replayssm import (
    replayssm_buffer_shapes,
    replayssm_commit,
    replayssm_gated_delta_rule,
)

DEV = "cuda"
LAYERS, SLOTS = 2, 6
LIVE, PARKED, DST, DST2, REF = 0, 1, 2, 3, 4
H, K, V = 4, 128, 128
T = 4  # verify window: anchor + 3 drafts
# Step 5 below flushes; the reference step taken after any of them never does.
CAP = 4 * T
CONV = (3, 64)
ROWS, BLOCK, ENTRY, UNITS = 2, 16, 1024, 32
LAYOUT = "test-gdn-replayssm"


class _Copier:
    """What the store/restore path reads, with the shipped methods bound."""


for _name in (
    "execute_paged_state_copies",
    "_fold_replay_records",
    "_validate_paged_state_op",
    "_checkpoint_copy_plan",
    "_checkpoint_slot_bases",
    "_checkpoint_plane_tensors",
    "_checkpoint_plane_shapes",
    "_checkpoint_layer_ranges",
    "_checkpoint_segment_sizes",
    "_checkpoint_num_slots",
    "checkpoint_image_bytes",
    "_checkpoint_descriptor_device",
    "_is_kda",
):
    setattr(_Copier, _name, getattr(GDNStateMixin, _name))
for _name in (
    "_page_unit_index_cache",
    "_page_unit_kv_cache",
    "_page_unit_regions",
    "_page_unit_bases",
    "_page_unit_stream_sizes",
):
    setattr(_Copier, _name, getattr(PageUnitGeometryMixin, _name))
_Copier._checkpoint_staging = AttentionMetadataBuilder._checkpoint_staging


def _copier() -> _Copier:
    sk, su, sg = replayssm_buffer_shapes(CAP, H, K, V, False)
    bf16 = torch.bfloat16
    runner = SimpleNamespace(
        mamba_k_cache=torch.randn((LAYERS, SLOTS) + CONV, dtype=bf16, device=DEV),
        mamba_v_cache=(
            torch.randn((LAYERS, SLOTS, H, K, V), dtype=bf16, device=DEV) * 0.01
        ),
        replayssm_buf_k=torch.zeros((LAYERS, SLOTS) + sk, dtype=bf16, device=DEV),
        replayssm_buf_u=torch.zeros((LAYERS, SLOTS) + su, dtype=bf16, device=DEV),
        replayssm_buf_g=torch.zeros(
            (LAYERS, SLOTS) + sg, dtype=torch.float32, device=DEV
        ),
        # Every slot as a prefill leaves it.
        replayssm_write_pos=torch.full((SLOTS,), -1, dtype=torch.int32, device=DEV),
        kv_cache=torch.zeros(
            (ROWS, UNITS, BLOCK, ENTRY), dtype=torch.uint8, device=DEV
        ),
        block_size=BLOCK,
        config=SimpleNamespace(
            num_kvcache_blocks=UNITS,
            max_num_seqs=4,
            hf_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
        ),
    )
    c = _Copier()
    c.model_runner = runner
    c.kv_pool = SimpleNamespace(cache=SimpleNamespace(view=lambda _n: runner.kv_cache))
    c.replayssm = True
    c.replayssm_max_query_len = T
    c.num_state_layers = lambda: LAYERS
    c._state_shape_for_runner = lambda: (CONV, (H, K, V))
    c._state_dtypes = lambda: (bf16, bf16)
    image = c.checkpoint_image_bytes()
    runner.state_runtime = SimpleNamespace(
        checkpoint_spec=PagedStateCheckpointSpec(
            ROWS * BLOCK * ENTRY, image, LAYOUT, image
        )
    )
    return c


def _window():
    q = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=DEV)
    k = torch.randn_like(q)
    v = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=DEV)
    g = -torch.rand(1, T, H, dtype=torch.float32, device=DEV) * 0.1
    beta = torch.rand(1, T, H, dtype=torch.bfloat16, device=DEV) * 0.5 + 0.25
    return q, k, v, g, beta


def _cu():
    return torch.tensor([0, T], dtype=torch.int32, device=DEV)


def _decode(c, slot, accepted, windows):
    """One decode step as the runner takes it: commit, then every layer."""
    r = c.model_runner
    idx = torch.tensor([slot], dtype=torch.int32, device=DEV)
    replayssm_commit(
        r.replayssm_write_pos,
        idx,
        torch.tensor([accepted], dtype=torch.int32, device=DEV),
        T,
        CAP,
    )
    return [
        replayssm_gated_delta_rule(
            *windows[layer],
            ckpt=r.mamba_v_cache[layer],
            buf_k=r.replayssm_buf_k[layer],
            buf_u=r.replayssm_buf_u[layer],
            buf_g=r.replayssm_buf_g[layer],
            write_pos=r.replayssm_write_pos,
            slot_idx=idx,
            cu_seqlens=_cu(),
            max_query_len=T,
            use_qk_l2norm_in_kernel=True,
        ).clone()
        for layer in range(LAYERS)
    ]


def _round_trip(c, pairs):
    """Store each `src` in one call, then restore each image into its `dst`."""
    spec = c.model_runner.state_runtime.checkpoint_spec
    n = spec.units_per_checkpoint
    units = [tuple(range(i * n, (i + 1) * n)) for i in range(len(pairs))]
    c.execute_paged_state_copies(
        [
            CheckpointStoreOp(src, u, spec.op_bytes, LAYOUT)
            for (src, _), u in zip(pairs, units)
        ],
        [],
    )
    c.execute_paged_state_copies(
        [],
        [
            CheckpointRestoreOp(dst, u, spec.op_bytes, LAYOUT)
            for (_, dst), u in zip(pairs, units)
        ],
    )
    torch.cuda.synchronize()


def _slot(c, slot):
    r = c.model_runner
    planes = (
        r.mamba_k_cache,
        r.mamba_v_cache,
        r.replayssm_buf_k,
        r.replayssm_buf_u,
        r.replayssm_buf_g,
    )
    return [p[:, slot].clone() for p in planes] + [r.replayssm_write_pos[slot].clone()]


def _rel_l2(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("steps", [1, 2, 3, 5, 6])
def test_a_decode_store_resumes_as_the_window_s_first_token(steps):
    torch.manual_seed(steps)
    c = _copier()
    r = c.model_runner
    accepted = 1
    for i in range(steps):
        _decode(c, LIVE, accepted, [_window() for _ in range(LAYERS)])
        accepted = 1 + i % T
    live = _slot(c, LIVE)
    # The checkpoint position is the last window's first token: the live slot
    # with only that one accepted.
    for plane, state in zip(
        (
            r.mamba_k_cache,
            r.mamba_v_cache,
            r.replayssm_buf_k,
            r.replayssm_buf_u,
            r.replayssm_buf_g,
        ),
        live,
    ):
        plane[:, REF] = state
    r.replayssm_write_pos[REF] = live[-1]
    parked = _slot(c, PARKED)

    _round_trip(c, [(LIVE, DST), (PARKED, DST2)])

    for before, after in zip(live, _slot(c, LIVE)):
        assert torch.equal(before, after), "the store changed the live slot"
    assert torch.equal(r.mamba_k_cache[:, DST], live[0])
    # A prefill-parked slot has no records to fold.
    assert torch.equal(r.mamba_v_cache[:, DST2], parked[1])

    windows = [_window() for _ in range(LAYERS)]
    want = _decode(c, REF, 1, windows)
    for layer in range(LAYERS):
        got, _ = chunk_gated_delta_rule(
            *windows[layer],
            initial_state=r.mamba_v_cache[layer, DST : DST + 1].clone(),
            output_final_state=True,
            cu_seqlens=_cu(),
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        err = _rel_l2(got, want[layer])
        assert err < 2e-2, f"layer {layer} resumes elsewhere: rel L2 {err:.4f}"


def test_without_replayssm_the_plane_is_copied_as_is():
    torch.manual_seed(0)
    c = _copier()
    c.replayssm = False
    r = c.model_runner
    r.replayssm_write_pos.zero_()
    r.replayssm_buf_k.normal_()
    r.replayssm_buf_u.normal_()
    before = r.mamba_v_cache[:, LIVE].clone()
    _round_trip(c, [(LIVE, DST)])
    assert torch.equal(r.mamba_v_cache[:, DST], before)
