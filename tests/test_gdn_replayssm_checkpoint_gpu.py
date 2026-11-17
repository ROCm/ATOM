# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A decode-time PAGE checkpoint is the image of its verify step.

A decode leaves the live slot ahead of the boundary it files: the conv window
sits `step` columns in, and under ReplaySSM the SSM plane lags the records
since the last flush. The image is resumed by prefill, which reads the SSM
plane and the first `kernel - 1` conv columns alone, so the store builds the
step's image in a scratch slot. Runs the shipped store/restore path over the
real ReplaySSM, copy and chunk kernels.
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
from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder
from atom.model_ops.fla_ops.chunk import chunk_gated_delta_rule
from atom.model_ops.fla_ops.replayssm import (
    replayssm_buffer_shapes,
    replayssm_commit,
    replayssm_gated_delta_rule,
    replayssm_materialize,
)
from atom.model_ops.qwen4_exp.ops.ple import (
    advance_ngram_state,
    dilated_causal_conv1d,
)

DEV = "cuda"
LAYERS, SLOTS = 2, 7
LIVE, PARKED, DST, DST2, REF, SCRATCH, SCRATCH2 = range(7)
H, K, V = 4, 128, 128
T = 4  # verify window: anchor + 3 drafts
CAP = 4 * T
WIDTH = 3  # conv kernel - 1
CONV = (WIDTH + T - 1, 64)
ROWS, BLOCK, ENTRY, UNITS = 2, 16, 1024, 32
LAYOUT = "test-gdn-replayssm"


class _Copier:
    """What the store/restore path reads, with the shipped methods bound."""


for _name in (
    "execute_paged_state_copies",
    "_build_decode_images",
    "_materialize",
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
        config=SimpleNamespace(num_kvcache_blocks=UNITS, max_num_seqs=4),
    )
    c = _Copier()
    c.model_runner = runner
    c.kv_pool = SimpleNamespace(cache=SimpleNamespace(view=lambda _n: runner.kv_cache))
    c.replayssm = True
    c.replayssm_max_query_len = T
    c.num_spec = T - 1
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


def _planes(c):
    r = c.model_runner
    return (
        r.mamba_k_cache,
        r.mamba_v_cache,
        r.replayssm_buf_k,
        r.replayssm_buf_u,
        r.replayssm_buf_g,
    )


def _slot(c, slot):
    r = c.model_runner
    return [p[:, slot].clone() for p in _planes(c)] + [
        r.replayssm_write_pos[slot].clone()
    ]


def _round_trip(c, stores):
    """Store each `(src, step, scratch, dst)` in one call, then restore."""
    spec = c.model_runner.state_runtime.checkpoint_spec
    n = spec.units_per_checkpoint
    units = [tuple(range(i * n, (i + 1) * n)) for i in range(len(stores))]
    c.execute_paged_state_copies(
        [
            CheckpointStoreOp(src, u, spec.op_bytes, LAYOUT, step, scratch)
            for (src, step, scratch, _), u in zip(stores, units)
        ],
        [],
    )
    c.execute_paged_state_copies(
        [],
        [
            CheckpointRestoreOp(dst, u, spec.op_bytes, LAYOUT)
            for (*_, dst), u in zip(stores, units)
        ],
    )
    torch.cuda.synchronize()


def _rel_l2(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("decodes", [1, 3, 5])
@pytest.mark.parametrize("step", [0, 1, 3])
def test_a_decode_image_resumes_as_its_verify_step(decodes, step):
    """Step 0 is what deferred output files; a later step, undeferred."""
    torch.manual_seed(10 * decodes + step)
    c = _copier()
    r = c.model_runner
    accepted = 1
    for i in range(decodes):
        _decode(c, LIVE, accepted, [_window() for _ in range(LAYERS)])
        accepted = 1 + i % T
    live = _slot(c, LIVE)
    # The reference: the live slot with that step's records committed.
    for plane, state in zip(_planes(c), live):
        plane[:, REF] = state
    r.replayssm_write_pos[REF] = live[-1]
    # A slot a prefill wrote whole, with a previous tenant's records left over.
    for p in _planes(c)[2:]:
        p[:, PARKED].normal_()
    parked = _slot(c, PARKED)

    _round_trip(c, [(LIVE, step, SCRATCH, DST), (PARKED, 0, SCRATCH2, DST2)])

    for before, after in zip(live, _slot(c, LIVE)):
        assert torch.equal(before, after), "the store changed the live slot"
    assert torch.equal(
        r.mamba_k_cache[:, DST, :WIDTH], live[0][:, step : step + WIDTH]
    ), "the conv window is not the step's"
    assert torch.equal(r.mamba_v_cache[:, DST2], parked[1])
    assert torch.equal(r.mamba_k_cache[:, DST2], parked[0])

    windows = [_window() for _ in range(LAYERS)]
    want = _decode(c, REF, step + 1, windows)
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


@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize("h, step", [(5, 0), (5, 2), (10, 0), (10, 3)])
def test_materialize_folds_exactly_the_step_s_records(kda, h, step):
    """Against the recurrence written out; h = 10 is a flushing forward."""
    torch.manual_seed(h + step)
    slots, hv = 3, 4
    sk, su, sg = replayssm_buffer_shapes(CAP, hv, K, V, kda)
    buf_k = torch.randn((slots,) + sk, dtype=torch.bfloat16, device=DEV) * 0.1
    buf_u = torch.randn((slots,) + su, dtype=torch.bfloat16, device=DEV) * 0.1
    buf_g = -torch.rand((slots,) + sg, dtype=torch.float32, device=DEV) * 0.1
    state = (hv, V, K) if kda else (hv, K, V)
    ckpt = torch.randn((slots,) + state, dtype=torch.bfloat16, device=DEV) * 0.1
    write_pos = torch.tensor([h, -1, -1], dtype=torch.int32, device=DEV)
    ix = torch.tensor([0], dtype=torch.int64, device=DEV)
    replayssm_materialize(
        ckpt, buf_k, buf_u, buf_g, write_pos, ix, ix + 2, ix + step, T
    )

    n = (0 if h + 2 * T > CAP else h) + step + 1
    g = buf_g[0, :, :n].float()  # [HV, n] or [HV, n, K]
    if not kda:
        g = g[..., None].expand(hv, n, K)
    c = g.cumsum(1)
    w = torch.exp(c[:, -1:] - c)  # [HV, n, K]
    kw = buf_k[0, :, :n].float() * w
    s = torch.einsum("hjk,hjv->hkv", kw, buf_u[0, :, :n].float())
    decay = torch.exp(c[:, -1])  # [HV, K]
    s0 = ckpt[0].float().transpose(1, 2) if kda else ckpt[0].float()
    want = s0 * decay[:, :, None] + s
    got = ckpt[2].float().transpose(1, 2) if kda else ckpt[2].float()
    assert _rel_l2(got, want) < 1e-2


PLE_KERNEL, NGRAM, PLE_CHANNELS = 4, 3, 16


def _qwen38_builder():
    """`_copier`'s slots behind Qwen3.8's builder, with its two PLE windows."""
    c = object.__new__(Qwen4ExpMetadataBuilder)
    c.__dict__.update(vars(_copier()))
    r = c.model_runner
    r.config.torch_dtype = torch.bfloat16
    r.config.hf_config = SimpleNamespace(
        ple_layer_ids=[0],
        ple_conv_kernel_size=PLE_KERNEL,
        ngram_size=NGRAM,
        hidden_size=PLE_CHANNELS // 2,
        hc_count=2,
    )
    c.ngram_context_len = NGRAM - 1
    c.eos_token_id = 0
    history = (PLE_KERNEL - 1) * NGRAM
    r.ple_conv_state = torch.randn(
        SLOTS, PLE_CHANNELS, history + c.num_spec, dtype=torch.bfloat16, device=DEV
    )
    r.ple_ngram_state = torch.randint(
        1, 1000, (SLOTS, NGRAM - 1 + c.num_spec), dtype=torch.int64, device=DEV
    )
    return c


@pytest.mark.parametrize("step", [0, 1, 3])
def test_a_qwen38_decode_image_resumes_its_ple_windows_at_the_verify_step(step):
    """The next verify would read each PLE window `step` columns into the live
    slot; a prefill resumed from the image reads it from column 0. Both reads
    of the same new tokens have to agree."""
    torch.manual_seed(step)
    c = _qwen38_builder()
    r = c.model_runner
    _decode(c, LIVE, 1, [_window() for _ in range(LAYERS)])
    c._build_decode_images(
        [SimpleNamespace(src_slot=LIVE, scratch_slot=SCRATCH, step=step)]
    )

    def one(value):
        return torch.tensor([value], dtype=torch.int32, device=DEV)

    has = torch.tensor([True], device=DEV)
    accepted = one(step + 1)
    x = torch.randn(T, PLE_CHANNELS, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(PLE_CHANNELS, PLE_KERNEL, dtype=torch.bfloat16, device=DEV)
    conv = r.ple_conv_state
    want = dilated_causal_conv1d(
        x, w, conv.clone(), _cu(), one(LIVE), one(REF), has, NGRAM, accepted
    )
    got = dilated_causal_conv1d(
        x, w, conv.clone(), _cu(), one(SCRATCH), one(REF), has, NGRAM
    )
    assert torch.equal(got, want), "the PLE conv resumes elsewhere"

    ids = torch.randint(1, 1000, (T,), dtype=torch.int64, device=DEV)
    ngram = r.ple_ngram_state
    want = advance_ngram_state(
        ids, _cu(), ngram.clone(), one(LIVE), one(REF), has, 0, NGRAM - 1, accepted
    )
    got = advance_ngram_state(
        ids, _cu(), ngram.clone(), one(SCRATCH), one(REF), has, 0, NGRAM - 1
    )
    assert torch.equal(got, want), "the n-gram context resumes elsewhere"
    assert torch.equal(
        r.mamba_k_cache[:, SCRATCH, :WIDTH],
        r.mamba_k_cache[:, LIVE, step : step + WIDTH],
    ), "the GDN conv window is not the step's"
