"""Forward-path coverage for the batched (non-ReplaySSM) GDN TARGET_VERIFY path.

With speculative decoding (MTP, DFLASH, ...), TARGET_VERIFY goes through
ReplaySSM by default; this path runs only with ``ATOM_ENABLE_REPLAYSSM=0``.
Every test here forces ReplaySSM off.

``test_gdn_target_verify_batched_equiv.py`` compares the batched call pattern
with a per-step loop, using its own stand-in calls. This module drives the
real ``SGLangGatedDeltaNet.forward()`` TARGET_VERIFY path and asserts the
plugin's observable contract:

* ``core_attn_out`` (pad rows zero)
* nonzero ``intermediate_ssm[req, step]`` contents for each live request at
  steps 1..draft-1 (step 0 is host-seeded) -- what SGLang's
  ``fused_mamba_state_scatter_with_mask`` commits from
* nonzero ``intermediate_conv_window[0][req, step]`` contents for each live
  request at steps 1..draft-1
* the live ``conv`` / ``temporal`` state, which must come back untouched
* one conv and one SSM kernel call

Run on a GPU; the kernels have no CPU fallback.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from atom.plugin.sglang.attention_backend.attention_gdn import SGLangGatedDeltaNet
from atom.plugin.sglang.runtime import bind_current_forward_batch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GDN kernels require a GPU"
)


@pytest.fixture(autouse=True)
def _replayssm_off(monkeypatch):
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "0")


# Qwen3.5-397B per-rank GDN shape at TP8.
NUM_K_HEADS = 16
NUM_V_HEADS = 64
TP_SIZE = 8
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4
STATE_LEN = CONV_KERNEL - 1

K_DIM = (NUM_K_HEADS // TP_SIZE) * HEAD_K_DIM
V_DIM = (NUM_V_HEADS // TP_SIZE) * HEAD_V_DIM
CONV_DIM = 2 * K_DIM + V_DIM

NUM_SLOTS = 24
LAYER_NUM = 7


class _TargetVerifyMode:
    @staticmethod
    def is_target_verify():
        return True


def _make_layer_cache(bs: int, draft: int, gen: torch.Generator):
    return SimpleNamespace(
        conv=[
            torch.randn(
                NUM_SLOTS,
                CONV_DIM,
                STATE_LEN,
                device="cuda",
                dtype=torch.bfloat16,
                generator=gen,
            )
        ],
        temporal=torch.randn(
            NUM_SLOTS,
            NUM_V_HEADS // TP_SIZE,
            HEAD_K_DIM,
            HEAD_V_DIM,
            device="cuda",
            dtype=torch.float32,
            generator=gen,
        ),
        intermediate_ssm=torch.zeros(
            NUM_SLOTS,
            draft,
            NUM_V_HEADS // TP_SIZE,
            HEAD_K_DIM,
            HEAD_V_DIM,
            device="cuda",
            dtype=torch.float32,
        ),
        intermediate_conv_window=[_dedup_conv_window_view(NUM_SLOTS, draft)],
    )


def _dedup_conv_window_view(num_slots: int, draft: int) -> torch.Tensor:
    """SGLang's deduplicated sliding-window intermediate_conv_window, per layer.

    One shared [slot, dim, D + K - 2] physical row exposed as an overlapping
    [slot, D, dim, K - 1] view; see MambaPool.__init__ in SGLang's
    memory_pool.py. GDN target verify on ROCm requires this layout (linear
    draft chain), so the tests exercise it rather than the dense fallback.
    """
    shared_win = draft + STATE_LEN - 1
    phys = torch.zeros(
        num_slots, CONV_DIM, shared_win, device="cuda", dtype=torch.bfloat16
    )
    return phys.as_strided(
        (num_slots, draft, CONV_DIM, STATE_LEN),
        (phys.stride(0), phys.stride(2), phys.stride(1), phys.stride(2)),
    )


def _make_impl(gen: torch.Generator) -> SGLangGatedDeltaNet:
    impl = SGLangGatedDeltaNet.__new__(SGLangGatedDeltaNet)
    torch.nn.Module.__init__(impl)
    impl.layer_num = LAYER_NUM
    impl.tp_size = TP_SIZE
    impl.num_k_heads = NUM_K_HEADS
    impl.num_v_heads = NUM_V_HEADS
    impl.head_k_dim = HEAD_K_DIM
    impl.head_v_dim = HEAD_V_DIM
    impl.activation = "silu"
    impl.A_log = torch.randn(
        NUM_V_HEADS // TP_SIZE, device="cuda", dtype=torch.float32, generator=gen
    )
    impl.dt_bias = torch.randn(
        NUM_V_HEADS // TP_SIZE, device="cuda", dtype=torch.float32, generator=gen
    )
    impl.conv1d = SimpleNamespace(
        weight=torch.randn(
            CONV_DIM,
            1,
            CONV_KERNEL,
            device="cuda",
            dtype=torch.bfloat16,
            generator=gen,
        ),
        bias=torch.randn(CONV_DIM, device="cuda", dtype=torch.bfloat16, generator=gen),
    )
    return impl


def _run_once(bs: int, draft: int, slots=None, flydsl_decode: bool = False):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20260804)

    layer_cache = _make_layer_cache(bs, draft, gen)
    impl = _make_impl(gen)
    if flydsl_decode:
        impl.gdn_flydsl_policy = SimpleNamespace(decode=True)
    # Deliberately unsorted, non-contiguous mamba slots.
    if slots is None:
        slots = [5, 2, 9, 17, 1, 13][:bs]
    cache_indices = torch.tensor(slots, device="cuda", dtype=torch.int32)
    linear_backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(mamba_cache_indices=cache_indices),
        req_to_token_pool=SimpleNamespace(
            mamba2_layer_cache=lambda _layer_id: layer_cache
        ),
    )
    forward_batch = SimpleNamespace(
        forward_mode=_TargetVerifyMode(),
        spec_info=SimpleNamespace(draft_token_num=draft),
        batch_size=bs,
        attn_backend=SimpleNamespace(linear_attn_backend=linear_backend),
    )

    num_tokens = bs * draft
    mixed_qkv = torch.randn(
        num_tokens, CONV_DIM, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    a = torch.randn(
        num_tokens,
        NUM_V_HEADS // TP_SIZE,
        device="cuda",
        dtype=torch.float32,
        generator=gen,
    )
    b = torch.randn(
        num_tokens,
        NUM_V_HEADS // TP_SIZE,
        device="cuda",
        dtype=torch.float32,
        generator=gen,
    )
    core_attn_out = torch.zeros(
        num_tokens,
        NUM_V_HEADS // TP_SIZE,
        HEAD_V_DIM,
        device="cuda",
        dtype=torch.float32,
    )

    conv_before = layer_cache.conv[0].clone()
    ssm_before = layer_cache.temporal.clone()

    with bind_current_forward_batch(forward_batch):
        out = impl.forward(mixed_qkv, b, a, core_attn_out, f"layers.{LAYER_NUM}")

    return {
        "out": out.clone(),
        "intermediate_ssm": layer_cache.intermediate_ssm[:bs].clone(),
        "intermediate_conv": layer_cache.intermediate_conv_window[0][:bs].clone(),
        "conv_live": layer_cache.conv[0],
        "ssm_live": layer_cache.temporal,
        "conv_before": conv_before,
        "ssm_before": ssm_before,
    }


@pytest.mark.parametrize("draft", [4, 8, 16])
@pytest.mark.parametrize("bs", [1, 3])
def test_batched_path_populates_verify_buffers(bs: int, draft: int):
    """Both buffers start at zero. Exclude step 0 because forward seeds the
    SSM state there before the kernel; steps 1.. must be kernel writes."""
    result = _run_once(bs, draft)
    assert result["out"].shape[0] == bs * draft
    for name in ("intermediate_ssm", "intermediate_conv"):
        buf = result[name]
        assert buf.shape[:2] == (bs, draft)
        written = buf[:, 1:].flatten(2).abs().amax(-1) > 0
        assert written.all(), f"{name} has unwritten (req, step) entries"


@pytest.mark.parametrize("draft", [4, 8, 16])
def test_live_state_unchanged_after_verify(draft: int):
    """SGLang commits accepted steps itself from the intermediate buffers."""
    result = _run_once(3, draft)
    torch.testing.assert_close(
        result["conv_live"], result["conv_before"], rtol=0, atol=0
    )
    torch.testing.assert_close(result["ssm_live"], result["ssm_before"], rtol=0, atol=0)


def test_batched_ssm_path_uses_one_kernel_call(monkeypatch):
    import atom.plugin.sglang.attention_backend.attention_gdn as gdn_mod

    calls = {"n": 0}
    real = gdn_mod.fused_recurrent_gated_delta_rule

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(gdn_mod, "fused_recurrent_gated_delta_rule", counting)

    _run_once(3, 8)
    assert calls["n"] == 1, f"batched SSM should run once, got {calls['n']} calls"


@pytest.mark.parametrize(
    "flydsl_decode", [False, True], ids=["plain_kv", "flydsl_logical_kv"]
)
@pytest.mark.parametrize("draft", [4, 16])
def test_pad_rows_zeroed_and_real_rows_unchanged(monkeypatch, draft, flydsl_decode):
    """CUDA-graph pad rows (cache index -1) skip the conv, so their q/k/v are
    whatever `torch.empty` held. Fill them with NaN to make that deterministic;
    the SSM kernel must still give those rows zeros and leave real rows alone.
    """
    import atom.plugin.sglang.attention_backend.attention_gdn as gdn_mod

    real_slots = [5, 2, 9, 17]
    padded_slots = [5, -1, 9, -1]
    bs = len(padded_slots)
    pad_rows = torch.tensor([s < 0 for s in padded_slots], device="cuda")
    real_rows = ~pad_rows

    real_conv = gdn_mod.causal_conv1d_update

    def conv_with_nan_pads(*args, **kwargs):
        outs = real_conv(*args, **kwargs)
        rows_in = kwargs["conv_state_indices_in"].view(-1) < 0
        for t in outs:
            t.view(bs, draft, -1)[rows_in] = float("nan")
        return outs

    monkeypatch.setattr(gdn_mod, "causal_conv1d_update", conv_with_nan_pads)
    padded = _run_once(bs, draft, slots=padded_slots, flydsl_decode=flydsl_decode)
    monkeypatch.setattr(gdn_mod, "causal_conv1d_update", real_conv)
    reference = _run_once(bs, draft, slots=real_slots, flydsl_decode=flydsl_decode)

    out = padded["out"].view(bs, draft, *padded["out"].shape[1:])
    ref = reference["out"].view(bs, draft, *reference["out"].shape[1:])
    assert torch.equal(out[pad_rows], torch.zeros_like(out[pad_rows]))
    assert torch.equal(out[real_rows], ref[real_rows])
    assert torch.equal(
        padded["intermediate_ssm"][real_rows], reference["intermediate_ssm"][real_rows]
    )
    # Step 0 holds the host seed ssm_states[-1]; the kernel must not write the
    # pad rows' later steps.
    pad_steps = padded["intermediate_ssm"][pad_rows][:, 1:]
    assert torch.equal(pad_steps, torch.zeros_like(pad_steps))
    torch.testing.assert_close(padded["ssm_live"], padded["ssm_before"], rtol=0, atol=0)


# --------------------------------------------------------------------------
# Batched conv: recovering SGLang's physical wide-window buffer
# --------------------------------------------------------------------------
def _sglang_dedup_conv_window(num_slots: int, draft: int):
    """Replicate SGLang's deduplicated intermediate_conv_window allocation.

    Mirrors MambaPool.__init__ (memory_pool.py): one shared
    [layer, slot, dim, D + K - 2] physical buffer plus an overlapping
    as_strided view of logical shape [layer, slot, D, dim, K - 1].
    """
    shared_win = draft + STATE_LEN - 1
    phys = torch.zeros(
        1, num_slots, CONV_DIM, shared_win, device="cuda", dtype=torch.bfloat16
    )
    view = phys.as_strided(
        (phys.shape[0], phys.shape[1], draft, CONV_DIM, STATE_LEN),
        (
            phys.stride(0),
            phys.stride(1),
            phys.stride(3),
            phys.stride(2),
            phys.stride(3),
        ),
    )
    return phys, view


@pytest.mark.parametrize("draft", [4, 8, 16])
def test_conv_window_phys_recovery_matches_allocation(draft: int):
    """The stride-based recovery must return the same storage SGLang allocated."""
    phys, view = _sglang_dedup_conv_window(NUM_SLOTS, draft)
    recovered = SGLangGatedDeltaNet._spec_conv_window_phys(view[0], draft)

    assert recovered is not None
    assert recovered.shape == phys[0].shape
    assert recovered.stride() == phys[0].stride()
    marker = torch.randn(CONV_DIM, device="cuda", dtype=torch.bfloat16)
    recovered[3, :, 2] = marker
    torch.testing.assert_close(phys[0, 3, :, 2], marker, rtol=0, atol=0)
    # And the logical view must see it as step 2's window column 0.
    torch.testing.assert_close(view[0, 3, 2, :, 0], marker, rtol=0, atol=0)


def test_conv_window_phys_recovery_rejects_dense_layout():
    dense = torch.zeros(
        NUM_SLOTS, 8, CONV_DIM, STATE_LEN, device="cuda", dtype=torch.bfloat16
    )
    with pytest.raises(RuntimeError, match="deduplicated sliding-window"):
        SGLangGatedDeltaNet._spec_conv_window_phys(dense, 8)


def test_conv_window_phys_recovery_rejects_wrong_draft_len():
    _phys, view = _sglang_dedup_conv_window(NUM_SLOTS, 8)
    with pytest.raises(RuntimeError, match="draft dimension"):
        SGLangGatedDeltaNet._spec_conv_window_phys(view[0], 16)


def test_default_conv_path_uses_one_kernel_call(monkeypatch):
    import atom.plugin.sglang.attention_backend.attention_gdn as gdn_mod

    calls = {"n": 0}
    real = gdn_mod.causal_conv1d_update

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(gdn_mod, "causal_conv1d_update", counting)

    _run_once(3, 8)
    assert calls["n"] == 1, f"batched conv should run once, got {calls['n']} calls"


def test_default_path_rejects_dense_conv_layout(monkeypatch):
    def dense(num_slots: int, draft: int) -> torch.Tensor:
        return torch.zeros(
            num_slots, draft, CONV_DIM, STATE_LEN, device="cuda", dtype=torch.bfloat16
        )

    monkeypatch.setitem(globals(), "_dedup_conv_window_view", dense)
    with pytest.raises(RuntimeError, match="deduplicated sliding-window"):
        _run_once(3, 8)
