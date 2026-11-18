# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A decode-time PAGE checkpoint is the image of its verify step, as prefill
resumes it. Runs the shipped store/restore path over the real kernels."""

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
from atom.model_ops.fla_ops.fused_recurrent import fused_recurrent_gated_delta_rule
from atom.model_ops.fla_ops.replayssm import (
    replayssm_buffer_shapes,
    replayssm_commit,
    replayssm_gated_delta_rule,
    replayssm_materialize,
)
from atom.model_ops.mamba_ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)

DEV = "cuda"
LAYERS, SLOTS = 2, 8
LIVE, PARKED, DST, DST2, DST3, REF, SCRATCH, SCRATCH2 = range(8)
# Without ReplaySSM a request holds a slot per verify step.
SPEC_SLOTS = (LIVE, PARKED, DST2, DST3)
H, K, V = 4, 128, 128
T = 4  # verify window: anchor + 3 drafts
CAP = 4 * T
WIDTH = 3  # conv kernel - 1
CK, CV = 128, 256  # conv q/k and v widths, as the conv kernels tile them
CONV = (WIDTH + T - 1, 2 * CK + CV)
ROWS, BLOCK, ENTRY, UNITS = 2, 16, 1024, 32
LAYOUT = "test-gdn-replayssm"


class _Copier(PageUnitGeometryMixin, GDNStateMixin):
    """The shipped store/restore path, built without a model runner."""

    _checkpoint_staging = AttentionMetadataBuilder._checkpoint_staging


def _copier(kda=False) -> _Copier:
    """GDN keeps the state `[HV, K, V]`; KDA transposed, with a per-K gate."""
    sk, su, sg = replayssm_buffer_shapes(CAP, H, K, V, kda)
    state = (H, V, K) if kda else (H, K, V)
    bf16 = torch.bfloat16
    runner = SimpleNamespace(
        mamba_k_cache=torch.randn((LAYERS, SLOTS) + CONV, dtype=bf16, device=DEV),
        mamba_v_cache=torch.randn((LAYERS, SLOTS) + state, dtype=bf16, device=DEV)
        * 0.01,
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
    c = _Copier.__new__(_Copier)
    c.model_runner = runner
    c.kv_pool = SimpleNamespace(cache=SimpleNamespace(view=lambda _n: runner.kv_cache))
    c.replayssm = True
    c.replayssm_max_query_len = T
    c.num_spec = T - 1
    c.num_state_layers = lambda: LAYERS
    c._state_shape_for_runner = lambda: (CONV, state)
    c._state_dtypes = lambda: (bf16, bf16)
    c.conv_weight = torch.randn(CONV[1], WIDTH + 1, dtype=bf16, device=DEV) * 0.3
    image = c.checkpoint_image_bytes()
    runner.state_runtime = SimpleNamespace(
        checkpoint_spec=PagedStateCheckpointSpec(
            ROWS * BLOCK * ENTRY, image, LAYOUT, image
        )
    )
    return c


def _ix(*vals):
    return torch.tensor(vals, dtype=torch.int32, device=DEV)


def _cu():
    return _ix(0, T)


def _window():
    q = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=DEV)
    k = torch.randn_like(q)
    v = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=DEV)
    g = -torch.rand(1, T, H, dtype=torch.float32, device=DEV) * 0.1
    beta = torch.rand(1, T, H, dtype=torch.bfloat16, device=DEV) * 0.5 + 0.25
    return q, k, v, g, beta


def _conv_pool(c, layer):
    # Allocated `[slots, window, dim]`, handed to the kernels `[slots, dim, window]`.
    return c.model_runner.mamba_k_cache[layer].transpose(-1, -2)


def _decode(c, slot, accepted, windows, xs):
    """One decode step as the runner takes it: commit, then every layer's conv
    update (rolled by the previous step's accepts) and recurrence."""
    r = c.model_runner
    replayssm_commit(r.replayssm_write_pos, _ix(slot), _ix(accepted), T, CAP)
    ssm, conv = [], []
    for layer in range(LAYERS):
        conv.append(
            torch.cat(
                causal_conv1d_update(
                    xs[layer],
                    _conv_pool(c, layer),
                    c.conv_weight,
                    CK,
                    CV,
                    activation="silu",
                    conv_state_indices=_ix(slot),
                    num_accepted_tokens=_ix(accepted),
                    query_start_loc=_cu(),
                    max_query_len=T,
                ),
                dim=-1,
            )
        )
        ssm.append(
            replayssm_gated_delta_rule(
                *windows[layer],
                ckpt=r.mamba_v_cache[layer],
                buf_k=r.replayssm_buf_k[layer],
                buf_u=r.replayssm_buf_u[layer],
                buf_g=r.replayssm_buf_g[layer],
                write_pos=r.replayssm_write_pos,
                slot_idx=_ix(slot),
                cu_seqlens=_cu(),
                max_query_len=T,
                use_qk_l2norm_in_kernel=True,
            ).clone()
        )
    return ssm, conv


def _decode_spec(c, accepted, windows, xs):
    """The same step without ReplaySSM: each verify step's state goes to its
    own slot, and the next verify resumes from the accepted one."""
    r = c.model_runner
    ssm, conv = [], []
    for layer in range(LAYERS):
        conv.append(
            torch.cat(
                causal_conv1d_update(
                    xs[layer],
                    _conv_pool(c, layer),
                    c.conv_weight,
                    CK,
                    CV,
                    activation="silu",
                    conv_state_indices=_ix(LIVE),
                    num_accepted_tokens=_ix(accepted),
                    query_start_loc=_cu(),
                    max_query_len=T,
                ),
                dim=-1,
            )
        )
        o, _ = fused_recurrent_gated_delta_rule(
            *windows[layer],
            initial_state=r.mamba_v_cache[layer],
            inplace_final_state=True,
            cu_seqlens=_cu(),
            ssm_state_indices=_ix(*SPEC_SLOTS)[None],
            num_accepted_tokens=_ix(accepted),
            use_qk_l2norm_in_kernel=True,
        )
        ssm.append(o.clone())
    return ssm, conv


def _step_inputs():
    return (
        [_window() for _ in range(LAYERS)],
        [
            torch.randn(T, CONV[1], dtype=torch.bfloat16, device=DEV)
            for _ in range(LAYERS)
        ],
    )


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


def _round_trip(c, stores, step_slot=None):
    """Store each `(src, step, scratch, dst)` in one call, then restore."""
    spec = c.model_runner.state_runtime.checkpoint_spec
    n = spec.units_per_checkpoint
    units = [tuple(range(i * n, (i + 1) * n)) for i in range(len(stores))]
    c.execute_paged_state_copies(
        [
            CheckpointStoreOp(
                src,
                u,
                spec.op_bytes,
                LAYOUT,
                step,
                scratch,
                src if step_slot is None else step_slot,
            )
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


def _fold(s0, buf_k, buf_u, buf_g, n, kda):
    """The state after `n` records, the recurrence's closed form, `[HV, K, V]`."""
    hv, k = s0.shape[0], buf_k.shape[-1]
    g = buf_g[:, :n].float()  # [HV, n] or [HV, n, K]
    if not kda:
        g = g[..., None].expand(hv, n, k)
    c = g.cumsum(1)
    kw = buf_k[:, :n].float() * torch.exp(c[:, -1:] - c)
    s = torch.einsum("hjk,hjv->hkv", kw, buf_u[:, :n].float())
    s0 = s0.float().transpose(1, 2) if kda else s0.float()
    return s0 * torch.exp(c[:, -1])[:, :, None] + s


@pytest.mark.parametrize("decodes", [1, 3, 5])
@pytest.mark.parametrize("step", [0, 1, 3])
def test_a_decode_image_resumes_as_its_verify_step(decodes, step):
    """Step 0 is what deferred output files; a later step, undeferred."""
    torch.manual_seed(10 * decodes + step)
    c = _copier()
    r = c.model_runner
    accepted = 1
    for i in range(decodes):
        _decode(c, LIVE, accepted, *_step_inputs())
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

    _round_trip(
        c,
        [
            (LIVE, step, SCRATCH, DST),
            (PARKED, 0, SCRATCH2, DST2),
            (PARKED, -1, -1, DST3),  # a prefill boundary, copied as is
        ],
    )

    for before, after in zip(live, _slot(c, LIVE)):
        assert torch.equal(before, after), "the store changed the live slot"
    cut = CONV[0] - step
    assert torch.equal(
        r.mamba_k_cache[:, DST],
        torch.cat([live[0][:, step:], live[0][:, cut:]], dim=1),
    ), "the conv image is not the step's window over the slot's own tail"
    for dst in (DST2, DST3):
        assert torch.equal(r.mamba_v_cache[:, dst], parked[1])
        assert torch.equal(r.mamba_k_cache[:, dst], parked[0])

    windows, xs = _step_inputs()
    _assert_resumes(c, windows, xs, *_decode(c, REF, step + 1, windows, xs))


def _assert_resumes(c, windows, xs, want_ssm, want_conv):
    """A prefill over the next window from DST matches the live continuation."""
    r = c.model_runner
    for layer in range(LAYERS):
        got, _ = chunk_gated_delta_rule(
            *windows[layer],
            initial_state=r.mamba_v_cache[layer, DST : DST + 1].clone(),
            output_final_state=True,
            cu_seqlens=_cu(),
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        err = _rel_l2(got, want_ssm[layer])
        assert err < 2e-2, f"layer {layer} SSM resumes elsewhere: rel L2 {err:.4f}"
        got_conv = torch.cat(
            causal_conv1d_fn(
                xs[layer].T,
                c.conv_weight,
                None,
                conv_states=_conv_pool(c, layer),
                query_start_loc=_cu(),
                k_dim_size=CK,
                v_dim_size=CV,
                cache_indices=_ix(DST),
                has_initial_state=torch.tensor([True], device=DEV),
                activation="silu",
            ),
            dim=-1,
        )
        err = _rel_l2(got_conv, want_conv[layer])
        assert err < 1e-2, f"layer {layer} conv resumes elsewhere: rel L2 {err:.4f}"


@pytest.mark.parametrize("decodes", [1, 3])
@pytest.mark.parametrize("step", [0, 1, 3])
def test_without_replayssm_the_image_is_the_accepted_step_s_slot(decodes, step):
    torch.manual_seed(100 + 10 * decodes + step)
    c = _copier()
    c.replayssm = False
    r = c.model_runner
    accepted = 1
    for i in range(decodes):
        _decode_spec(c, accepted, *_step_inputs())
        accepted = 1 + i % T
    conv, ssm = (p[:, list(SPEC_SLOTS)].clone() for p in _planes(c)[:2])

    _round_trip(c, [(LIVE, step, SCRATCH, DST)], step_slot=SPEC_SLOTS[step])

    assert torch.equal(r.mamba_k_cache[:, list(SPEC_SLOTS)], conv)
    assert torch.equal(r.mamba_v_cache[:, list(SPEC_SLOTS)], ssm)
    assert torch.equal(r.mamba_v_cache[:, DST], ssm[:, step])
    cut = CONV[0] - step
    assert torch.equal(
        r.mamba_k_cache[:, DST],
        torch.cat([conv[:, 0, step:], conv[:, 0, cut:]], dim=1),
    )
    windows, xs = _step_inputs()
    _assert_resumes(c, windows, xs, *_decode_spec(c, step + 1, windows, xs))


@pytest.mark.parametrize("h, step", [(5, 0), (5, 2), (10, 3)])
def test_a_kda_decode_image_is_its_verify_step(h, step):
    """The KDA layout through the same build; h = 10 is a flushing forward."""
    torch.manual_seed(h + step)
    c = _copier(kda=True)
    r = c.model_runner
    for p in _planes(c)[2:]:
        p[:, LIVE].normal_(std=0.1)
    r.replayssm_buf_g[:, LIVE] = -r.replayssm_buf_g[:, LIVE].abs()
    r.replayssm_write_pos[LIVE] = h
    live = _slot(c, LIVE)

    _round_trip(c, [(LIVE, step, SCRATCH, DST)])

    for before, after in zip(live, _slot(c, LIVE)):
        assert torch.equal(before, after), "the store changed the live slot"
    assert torch.equal(
        r.mamba_k_cache[:, DST, :WIDTH], live[0][:, step : step + WIDTH]
    ), "the conv window is not the step's"
    n = (0 if h + 2 * T > CAP else h) + step + 1
    for layer in range(LAYERS):
        want = _fold(*(x[layer] for x in live[1:5]), n, kda=True)
        got = r.mamba_v_cache[layer, DST].float().transpose(1, 2)
        assert _rel_l2(got, want) < 1e-2


@pytest.mark.parametrize(
    "k, v, dtype",
    [(K, V, torch.bfloat16), (64, 128, torch.bfloat16), (K, V, torch.float32)],
)
@pytest.mark.parametrize("kda", [False, True])
@pytest.mark.parametrize("h, step", [(5, 0), (5, 2), (10, 0), (10, 3)])
def test_materialize_folds_exactly_the_step_s_records(kda, h, step, k, v, dtype):
    """Against the closed form; h = 10 is a flushing forward."""
    torch.manual_seed(h + step)
    slots, hv = 3, 4
    sk, su, sg = replayssm_buffer_shapes(CAP, hv, k, v, kda)
    buf_k = torch.randn((slots,) + sk, dtype=torch.bfloat16, device=DEV) * 0.1
    buf_u = torch.randn((slots,) + su, dtype=torch.bfloat16, device=DEV) * 0.1
    buf_g = -torch.rand((slots,) + sg, dtype=torch.float32, device=DEV) * 0.1
    state = (hv, v, k) if kda else (hv, k, v)
    ckpt = torch.randn((slots,) + state, dtype=dtype, device=DEV) * 0.1
    write_pos = torch.tensor([h, -1, -1], dtype=torch.int32, device=DEV)
    ix = torch.tensor([0], dtype=torch.int64, device=DEV)
    replayssm_materialize(
        ckpt, buf_k, buf_u, buf_g, write_pos, ix, ix + 2, ix + step, T
    )

    n = (0 if h + 2 * T > CAP else h) + step + 1
    want = _fold(ckpt[0], buf_k[0], buf_u[0], buf_g[0], n, kda)
    got = ckpt[2].float().transpose(1, 2) if kda else ckpt[2].float()
    assert _rel_l2(got, want) < 1e-2
