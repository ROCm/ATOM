"""Step-0 differential test for a batched GDN target-verify path.

The SGLang plugin implements DFLASH target verify as a per-draft-step Python
loop (``atom/plugin/sglang/attention_backend/attention_gdn.py``), while ATOM's
native GDN backend already owns a batched spec path that folds the whole draft
block into a single kernel launch (``atom/model_ops/attention_gdn.py``, the
``spec_sequence_masks is not None`` branch). The plugin cannot reach it: its
``_build_gdn_metadata`` returns ``None`` for TARGET_VERIFY and hard-codes every
spec field to ``None``.

These tests check, with the real kernels, whether the two call patterns are
numerically interchangeable, and whether SGLang's buffer layouts can be
addressed the way ATOM's batched kernels expect:

* ``intermediate_ssm`` is contiguous ``[layer, slot, step, HV, K, V]``, so a
  per-layer slice can be viewed as a flat ``[slot * step, HV, K, V]`` pool and
  addressed by a 2D ``ssm_state_indices`` table.
* ``intermediate_conv_window`` is already SGLang's deduplicated sliding-window
  view over a physical ``[slot, dim, D + K - 2]`` buffer -- the same wide window
  ATOM's spec ``causal_conv1d_update`` writes.

The negative-control tests exist so that a future refactor cannot make the
equivalence tests pass vacuously: they assert that realistic adapter bugs are
still detected. See section 17 of ``ATOM_Qwen3.5_DFLASH_TP8复现与实现报告.md``.

Nothing here touches the plugin's production path.
"""

from __future__ import annotations

import pytest
import torch

from atom.model_ops.attention_gdn import fused_gdn_gating
from atom.model_ops.fla_ops import fused_recurrent_gated_delta_rule
from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_update

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GDN kernels require a GPU"
)

# Qwen3.5-397B per-rank GDN shape at TP8.
NUM_K_HEADS = 16 // 8
NUM_V_HEADS = 64 // 8
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4
STATE_LEN = CONV_KERNEL - 1

K_DIM = NUM_K_HEADS * HEAD_K_DIM
V_DIM = NUM_V_HEADS * HEAD_V_DIM
CONV_DIM = 2 * K_DIM + V_DIM  # q, k and v are packed into mixed_qkv

NUM_SLOTS = 24

# Draft lengths spanning the sweep in section 16 of the report.
DRAFT_LENS = [4, 8, 16]
BATCH_SIZES = [1, 3]


class _Inputs:
    """One synthetic target-verify batch plus the live conv/SSM state."""

    def __init__(self, bs: int, draft: int, seed: int = 1234):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        self.bs = bs
        self.draft = draft
        self.mixed = torch.randn(
            bs, draft, CONV_DIM, device="cuda", dtype=torch.bfloat16, generator=gen
        )
        self.live_ssm = torch.randn(
            NUM_SLOTS,
            NUM_V_HEADS,
            HEAD_K_DIM,
            HEAD_V_DIM,
            device="cuda",
            dtype=torch.float32,
            generator=gen,
        )
        self.live_conv = torch.randn(
            NUM_SLOTS,
            CONV_DIM,
            STATE_LEN,
            device="cuda",
            dtype=torch.bfloat16,
            generator=gen,
        )
        # Deliberately non-contiguous, unsorted slots, as in a real batch.
        self.cache_indices = torch.tensor(
            [5, 2, 9, 17, 1, 13][:bs], device="cuda", dtype=torch.int32
        )
        a = torch.randn(
            bs * draft, NUM_V_HEADS, device="cuda", dtype=torch.float32, generator=gen
        )
        b = torch.randn(
            bs * draft, NUM_V_HEADS, device="cuda", dtype=torch.float32, generator=gen
        )
        a_log = torch.randn(
            NUM_V_HEADS, device="cuda", dtype=torch.float32, generator=gen
        )
        dt_bias = torch.randn(
            NUM_V_HEADS, device="cuda", dtype=torch.float32, generator=gen
        )
        # Gating is elementwise per token, so one call covers both paths; each
        # path then consumes the same values in sequence-major order.
        g, beta = fused_gdn_gating(a_log, a, b, dt_bias)
        self.g = g.reshape(1, bs, draft, NUM_V_HEADS)
        self.beta = beta.reshape(1, bs, draft, NUM_V_HEADS)
        self.conv_weight = torch.randn(
            CONV_DIM, CONV_KERNEL, device="cuda", dtype=torch.bfloat16, generator=gen
        )
        self.conv_bias = torch.randn(
            CONV_DIM, device="cuda", dtype=torch.bfloat16, generator=gen
        )

    def slot_table(self) -> torch.Tensor:
        """2D ``[seq, step]`` table into a flat ``slot * draft`` pool."""
        steps = torch.arange(self.draft, device="cuda")
        return (
            self.cache_indices.long().unsqueeze(1) * self.draft + steps.unsqueeze(0)
        ).to(torch.int32)


def _split_qkv(packed: torch.Tensor, num_tokens: int):
    query = packed[..., :K_DIM].reshape(1, num_tokens, NUM_K_HEADS, HEAD_K_DIM)
    key = packed[..., K_DIM : 2 * K_DIM].reshape(1, num_tokens, NUM_K_HEADS, HEAD_K_DIM)
    value = packed[..., 2 * K_DIM :].reshape(1, num_tokens, NUM_V_HEADS, HEAD_V_DIM)
    return query, key, value


# --------------------------------------------------------------------------
# SSM recurrent
# --------------------------------------------------------------------------
def _ssm_per_step_loop(inp: _Inputs):
    """What the plugin does today: one kernel triple per draft step."""
    ssm = inp.live_ssm.clone()
    out = torch.zeros(
        inp.bs, inp.draft, NUM_V_HEADS, HEAD_V_DIM, device="cuda", dtype=torch.float32
    )
    inter = torch.zeros(
        inp.bs,
        inp.draft,
        NUM_V_HEADS,
        HEAD_K_DIM,
        HEAD_V_DIM,
        device="cuda",
        dtype=torch.float32,
    )
    cu_step = torch.arange(inp.bs + 1, device="cuda", dtype=torch.int32)
    for step in range(inp.draft):
        query, key, value = _split_qkv(inp.mixed[:, step, :], inp.bs)
        step_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=inp.g[:, :, step, :],
            beta=inp.beta[:, :, step, :],
            initial_state=ssm,
            inplace_final_state=True,
            cu_seqlens=cu_step,
            ssm_state_indices=inp.cache_indices,
            use_qk_l2norm_in_kernel=True,
        )
        out[:, step] = step_out.squeeze(0).float()
        inter[:, step] = ssm[inp.cache_indices].float()
    return out, inter


def _ssm_batched(inp: _Inputs, slot_table: torch.Tensor | None = None, seed=True):
    """One kernel launch over a flat slot pool with a 2D index table."""
    if slot_table is None:
        slot_table = inp.slot_table()
    pool = torch.zeros(
        NUM_SLOTS * inp.draft,
        NUM_V_HEADS,
        HEAD_K_DIM,
        HEAD_V_DIM,
        device="cuda",
        dtype=torch.float32,
    )
    if seed:
        # The kernel loads h0 once, before its internal step loop, so seeding
        # column 0 with the live state and letting step 0 overwrite that same
        # slot is safe.
        pool[slot_table[:, 0].long()] = inp.live_ssm[inp.cache_indices].float()
    query, key, value = _split_qkv(
        inp.mixed.reshape(inp.bs * inp.draft, CONV_DIM), inp.bs * inp.draft
    )
    cu_block = torch.arange(
        0, (inp.bs + 1) * inp.draft, inp.draft, device="cuda", dtype=torch.int32
    )
    out, _ = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=inp.g.reshape(1, inp.bs * inp.draft, NUM_V_HEADS),
        beta=inp.beta.reshape(1, inp.bs * inp.draft, NUM_V_HEADS),
        initial_state=pool,
        inplace_final_state=True,
        cu_seqlens=cu_block,
        ssm_state_indices=slot_table,
        use_qk_l2norm_in_kernel=True,
    )
    out = out.squeeze(0).reshape(inp.bs, inp.draft, NUM_V_HEADS, HEAD_V_DIM).float()
    return out, pool[slot_table.long()].float()


@pytest.mark.parametrize("draft", DRAFT_LENS)
@pytest.mark.parametrize("bs", BATCH_SIZES)
def test_ssm_per_step_loop_matches_batched_slot_table(bs: int, draft: int):
    """The batched call reproduces the loop bit-for-bit, outputs and every
    per-step intermediate state alike."""
    inp = _Inputs(bs, draft)
    out_a, inter_a = _ssm_per_step_loop(inp)
    out_b, inter_b = _ssm_batched(inp)

    # Observed exactly equal on gfx942; assert it, so any drift is caught.
    torch.testing.assert_close(out_b, out_a, rtol=0, atol=0)
    torch.testing.assert_close(inter_b, inter_a, rtol=0, atol=0)


@pytest.mark.parametrize("draft", DRAFT_LENS)
def test_batched_ssm_leaves_live_state_untouched(draft: int):
    """The batched path writes only into the slot pool, so the plugin's
    snapshot/restore of the live SSM state becomes unnecessary."""
    inp = _Inputs(3, draft)
    live_before = inp.live_ssm.clone()
    _ssm_batched(inp)
    torch.testing.assert_close(inp.live_ssm, live_before, rtol=0, atol=0)


def test_ssm_equivalence_detects_collapsed_slot_table():
    """Negative control: a naive adapter reusing one slot for every step (what
    a 1D index table would do) must NOT compare equal.

    Note the outputs alone stay identical -- only the per-step state dump
    differs -- so an equivalence test that checks outputs only would be
    vacuous."""
    inp = _Inputs(3, 8)
    _out_a, inter_a = _ssm_per_step_loop(inp)
    collapsed = (
        (inp.cache_indices.long().unsqueeze(1) * inp.draft)
        .expand(inp.bs, inp.draft)
        .contiguous()
        .to(torch.int32)
    )
    _out_b, inter_b = _ssm_batched(inp, slot_table=collapsed)
    assert not torch.allclose(inter_b, inter_a, rtol=1e-3, atol=1e-3)


def test_ssm_equivalence_detects_missing_live_state_seed():
    """Negative control: forgetting to seed column 0 with the live state must
    NOT compare equal."""
    inp = _Inputs(3, 8)
    out_a, inter_a = _ssm_per_step_loop(inp)
    out_b, inter_b = _ssm_batched(inp, seed=False)
    assert not torch.allclose(out_b, out_a, rtol=1e-3, atol=1e-3)
    assert not torch.allclose(inter_b, inter_a, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------
# Conv
# --------------------------------------------------------------------------
def _conv_per_step_loop(inp: _Inputs):
    """Plugin behaviour today: one narrow-window update per draft step."""
    conv = inp.live_conv.clone()
    qkv = torch.zeros(inp.bs, inp.draft, CONV_DIM, device="cuda", dtype=torch.bfloat16)
    windows = torch.zeros(
        inp.bs, inp.draft, CONV_DIM, STATE_LEN, device="cuda", dtype=torch.bfloat16
    )
    for step in range(inp.draft):
        query, key, value = causal_conv1d_update(
            inp.mixed[:, step, :].contiguous(),
            conv,
            inp.conv_weight,
            K_DIM,
            V_DIM,
            inp.conv_bias,
            "silu",
            conv_state_indices=inp.cache_indices,
            validate_data=False,
        )
        qkv[:, step] = torch.cat([query, key, value], dim=-1)
        windows[:, step] = conv[inp.cache_indices]
    return qkv, windows


def _conv_batched(inp: _Inputs, seed: bool = True):
    """One spec call writing the shared wide window, mirroring SGLang's
    deduplicated ``intermediate_conv_window`` physical buffer."""
    shared_win = inp.draft + STATE_LEN - 1
    conv = torch.zeros(
        NUM_SLOTS, CONV_DIM, shared_win, device="cuda", dtype=torch.bfloat16
    )
    if seed:
        conv[inp.cache_indices, :, :STATE_LEN] = inp.live_conv[inp.cache_indices]
    query_start_loc = torch.arange(
        0, (inp.bs + 1) * inp.draft, inp.draft, device="cuda", dtype=torch.int32
    )
    query, key, value = causal_conv1d_update(
        inp.mixed.reshape(inp.bs * inp.draft, CONV_DIM),
        conv,
        inp.conv_weight,
        K_DIM,
        V_DIM,
        inp.conv_bias,
        "silu",
        conv_state_indices=inp.cache_indices.reshape(inp.bs, 1).to(torch.int32),
        num_accepted_tokens=torch.ones(inp.bs, device="cuda", dtype=torch.int32),
        query_start_loc=query_start_loc,
        max_query_len=inp.draft,
        validate_data=False,
    )
    qkv = torch.cat([query, key, value], dim=-1).reshape(inp.bs, inp.draft, CONV_DIM)
    # SGLang's view: step t's window is phys[..., t : t + STATE_LEN].
    windows = torch.stack(
        [conv[inp.cache_indices, :, t : t + STATE_LEN] for t in range(inp.draft)],
        dim=1,
    )
    return qkv, windows


@pytest.mark.parametrize("draft", DRAFT_LENS)
@pytest.mark.parametrize("bs", BATCH_SIZES)
def test_conv_per_step_windows_match_wide_window_slices(bs: int, draft: int):
    """One wide-window spec call reproduces both the projected q/k/v and every
    per-step conv window that the loop materialises."""
    inp = _Inputs(bs, draft)
    qkv_a, win_a = _conv_per_step_loop(inp)
    qkv_b, win_b = _conv_batched(inp)

    torch.testing.assert_close(qkv_b, qkv_a, rtol=0, atol=0)
    torch.testing.assert_close(win_b, win_a, rtol=0, atol=0)


def test_conv_equivalence_detects_missing_history_seed():
    """Negative control: not seeding the wide window with the live conv history
    must NOT compare equal."""
    inp = _Inputs(3, 8)
    qkv_a, win_a = _conv_per_step_loop(inp)
    qkv_b, win_b = _conv_batched(inp, seed=False)
    assert not torch.allclose(qkv_b.float(), qkv_a.float(), rtol=1e-2, atol=1e-2)
    assert not torch.allclose(win_b.float(), win_a.float(), rtol=1e-2, atol=1e-2)


def test_conv_equivalence_detects_off_by_one_window_slice():
    """Negative control: shifting the sliding-window slice by one column must
    NOT compare equal."""
    inp = _Inputs(3, 8)
    _qkv_a, win_a = _conv_per_step_loop(inp)
    shared_win = inp.draft + STATE_LEN - 1
    conv = torch.zeros(
        NUM_SLOTS, CONV_DIM, shared_win, device="cuda", dtype=torch.bfloat16
    )
    conv[inp.cache_indices, :, :STATE_LEN] = inp.live_conv[inp.cache_indices]
    query_start_loc = torch.arange(
        0, (inp.bs + 1) * inp.draft, inp.draft, device="cuda", dtype=torch.int32
    )
    causal_conv1d_update(
        inp.mixed.reshape(inp.bs * inp.draft, CONV_DIM),
        conv,
        inp.conv_weight,
        K_DIM,
        V_DIM,
        inp.conv_bias,
        "silu",
        conv_state_indices=inp.cache_indices.reshape(inp.bs, 1).to(torch.int32),
        num_accepted_tokens=torch.ones(inp.bs, device="cuda", dtype=torch.int32),
        query_start_loc=query_start_loc,
        max_query_len=inp.draft,
        validate_data=False,
    )
    shifted = torch.stack(
        [
            conv[inp.cache_indices, :, t + 1 : t + 1 + STATE_LEN]
            for t in range(inp.draft - 1)
        ],
        dim=1,
    )
    assert not torch.allclose(
        shifted.float(), win_a[:, : inp.draft - 1].float(), rtol=1e-2, atol=1e-2
    )


# --------------------------------------------------------------------------
# Layout prerequisite
# --------------------------------------------------------------------------
def test_intermediate_ssm_layout_is_flattenable():
    """SGLang allocates intermediate_ssm as a single contiguous
    ``[layer, slot, step, HV, K, V]`` tensor, so a per-layer slice can be
    viewed as the flat ``[slot * step, ...]`` pool the 2D index table needs.

    This mirrors the allocation in SGLang's
    ``MambaPool.__init__`` (memory_pool.py); if that ever stops being
    contiguous the conversion layer must copy instead of view."""
    draft = 8
    layers, slots = 2, NUM_SLOTS
    cache = torch.zeros(
        layers, slots, draft, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, device="cuda"
    )
    per_layer = cache[0]
    assert per_layer.is_contiguous()
    flat = per_layer.view(slots * draft, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)
    # flat[slot * draft + step] must alias cache[0, slot, step].
    marker = torch.randn(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, device="cuda")
    flat[7 * draft + 3] = marker
    torch.testing.assert_close(cache[0, 7, 3], marker, rtol=0, atol=0)
