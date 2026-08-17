# SPDX-License-Identifier: MIT
"""Unit tests for DeepSeek-V4 DSpark drafter (Phase 1).

Covers the self-contained, GPU-free pieces: Markov head + Confidence head
numerics, and SpeculativeConfig DSpark detection/routing.
"""

import torch

from atom.models.deepseek_v4_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    _dspark_block_sparse_attention,
    _dspark_block_topk_idxs,
)


def _block_attn(q, kv, attn_sink, valid_target, scale):
    """Block attention with the gather indices the block plan would supply.

    Production builds ``topk_idxs`` once per block in ``_build_block_plan`` and
    shares it across stages; these tests exercise a single call, so they derive
    it the same way here.
    """
    B, T = q.shape[0], q.shape[1]
    W = kv.shape[1] - T
    topk_idxs = _dspark_block_topk_idxs(B, T, W, valid_target, q.device)
    return _dspark_block_sparse_attention(
        q, kv, attn_sink, valid_target, topk_idxs, scale
    )


def test_markov_head_shapes_and_factorization():
    V, r = 64, 8
    head = DSparkMarkovHead(vocab_size=V, rank=r)
    tokens = torch.tensor([0, 3, 63, 17])
    bias, embed = head(tokens)
    assert bias.shape == (4, V)
    assert embed.shape == (4, r)
    # bias must equal W1[x] @ W2^T exactly (low-rank factorization, paper Eq.5).
    w1 = head.markov_w1.weight  # [V, r]
    w2 = head.markov_w2.weight  # [V, r]
    expected = w1[tokens].float() @ w2.float().t()
    torch.testing.assert_close(bias, expected, rtol=1e-5, atol=1e-5)


def test_markov_head_conditioning_is_token_specific():
    # Different previous tokens must yield different biases (the whole point of
    # injecting intra-block dependency to fix multi-modal collision).
    V, r = 32, 4
    head = DSparkMarkovHead(vocab_size=V, rank=r)
    b0, _ = head(torch.tensor([0]))
    b1, _ = head(torch.tensor([1]))
    assert not torch.allclose(b0, b1)


def test_confidence_head_range_and_input_concat():
    hidden, r = 16, 8
    head = DSparkConfidenceHead(hidden_size=hidden, rank=r)
    h = torch.randn(5, hidden)
    m = torch.randn(5, r)
    c = head(h, m)
    assert c.shape == (5,)
    assert torch.all(c > 0) and torch.all(c < 1)
    # Matches sigmoid(proj([h; m])).
    expected = torch.sigmoid(head.proj(torch.cat([h, m], dim=-1).float()).squeeze(-1))
    torch.testing.assert_close(c, expected, rtol=1e-5, atol=1e-5)


def test_semi_autoregressive_bias_changes_argmax():
    # The Markov bias should be able to flip the next-token argmax away from the
    # base-logit argmax (this is how it suppresses cross-mode collisions).
    V, r = 10, 4
    head = DSparkMarkovHead(vocab_size=V, rank=r)
    torch.nn.init.zeros_(head.markov_w1.weight)
    torch.nn.init.zeros_(head.markov_w2.weight)
    # Make token 7 -> strong bias toward vocab id 2.
    head.markov_w1.weight.data[7, 0] = 1.0
    head.markov_w2.weight.data[2, 0] = 5.0
    base = torch.zeros(1, V)
    base[0, 9] = 1.0  # base prefers id 9
    bias, _ = head(torch.tensor([7]))
    combined = base + bias
    assert int(base.argmax(-1)) == 9
    assert int(combined.argmax(-1)) == 2


def _real_hf_config_override():
    """Load the real SpeculativeConfig.hf_config_override despite conftest stubs.

    conftest stubs ``atom.config`` to dodge heavy imports, so we exec the real
    source by file path under a throwaway module name and restore the stub.
    Returns None (→ test skips) if the module can't be imported in this sandbox.
    """
    import importlib.util
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "atom", "config.py")
    spec = importlib.util.spec_from_file_location("_atom_config_real", path)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.modules.get("atom.config")
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    finally:
        if saved is not None:
            sys.modules["atom.config"] = saved
    return getattr(mod, "SpeculativeConfig", None)


def test_speculative_config_detects_dspark():
    """A config with dspark_block_size routes to the DSpark draft arch and skips
    the serial-MTP n_predict=1 rewrite."""
    import types

    import pytest

    SpeculativeConfig = _real_hf_config_override()
    if SpeculativeConfig is None:
        pytest.skip("atom.config not importable in this sandbox")

    hf = types.SimpleNamespace(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_markov_rank=512,
        dspark_target_layer_ids=[58, 59, 60],
        num_nextn_predict_layers=3,
    )
    hf.update = lambda d: [setattr(hf, k, v) for k, v in d.items()]
    SpeculativeConfig.hf_config_override(hf, model_path=None)
    assert hf.model_type == "deepseek_v4_dspark"
    assert hf.architectures == ["DeepseekV4DSparkModel"]


def test_speculative_config_mtp_not_misrouted_to_dspark():
    """A plain V4 MTP config (no dspark_block_size) still routes to MTP."""
    import types

    import pytest

    SpeculativeConfig = _real_hf_config_override()
    if SpeculativeConfig is None:
        pytest.skip("atom.config not importable in this sandbox")

    hf = types.SimpleNamespace(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        num_nextn_predict_layers=1,
    )
    hf.update = lambda d: [setattr(hf, k, v) for k, v in d.items()]
    SpeculativeConfig.hf_config_override(hf, model_path=None)
    assert hf.model_type == "deepseek_v4_mtp"
    assert hf.architectures == ["DeepseekV4MTPModel"]


def test_block_sparse_attention_is_bidirectional_within_block():
    # The block is decoded in one parallel pass, so every draft query position
    # sees every draft KV column, including ones after itself.
    B, T, H, D, W = 1, 4, 2, 8, 3
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D)
    kv = torch.randn(B, W + T, D)
    sink = torch.zeros(H)
    valid_target = torch.ones(B, W, dtype=torch.bool)
    out_full = _block_attn(q, kv, sink, valid_target, D**-0.5)
    # Perturb the LAST draft KV row: every position, not just the last, must move.
    kv2 = kv.clone()
    kv2[:, -1] = 0.0
    out2 = _block_attn(q, kv2, sink, valid_target, D**-0.5)
    for t in range(T):
        assert not torch.allclose(
            out_full[:, t], out2[:, t]
        ), f"draft position {t} did not see the last draft column (causal leak)"
    # Symmetrically, perturbing the FIRST draft column must move every position.
    kv3 = kv.clone()
    kv3[:, W] = 0.0
    out3 = _block_attn(q, kv3, sink, valid_target, D**-0.5)
    for t in range(T):
        assert not torch.allclose(out_full[:, t], out3[:, t])


def test_block_topk_idxs_encode_the_same_mask_as_the_torch_path():
    # The torch fallback ignores topk_idxs and production honours only
    # topk_idxs, so nothing else pins the two encodings together. Assert the
    # gather indices directly — drift is invisible on CPU otherwise.
    B, T, W = 2, 4, 5
    valid_target = torch.ones(B, W, dtype=torch.bool)
    valid_target[0, :2] = False  # request 0 has two unpopulated window slots
    idxs = _dspark_block_topk_idxs(B, T, W, valid_target, torch.device("cpu"))
    assert idxs.shape == (B, T, W + T)
    assert idxs.dtype == torch.int32
    win, draft = idxs[..., :W], idxs[..., W:]
    for b in range(B):
        for m in range(T):
            # Window half: the slot's own index where valid, -1 where not.
            expected_win = torch.where(
                valid_target[b],
                torch.arange(W, dtype=torch.int32),
                torch.full((W,), -1, dtype=torch.int32),
            )
            torch.testing.assert_close(win[b, m], expected_win)
            # Draft half: the WHOLE block, every query row, never masked.
            torch.testing.assert_close(
                draft[b, m], W + torch.arange(T, dtype=torch.int32)
            )


def test_block_sparse_attention_respects_window_validity():
    # Invalid (future/empty) window slots must be masked out.
    B, T, H, D, W = 1, 2, 1, 4, 4
    torch.manual_seed(1)
    q = torch.randn(B, T, H, D)
    kv = torch.randn(B, W + T, D)
    sink = torch.zeros(H)
    all_valid = torch.ones(B, W, dtype=torch.bool)
    some_valid = all_valid.clone()
    some_valid[:, -2:] = False  # invalidate 2 window slots
    o_all = _block_attn(q, kv, sink, all_valid, D**-0.5)
    o_some = _block_attn(q, kv, sink, some_valid, D**-0.5)
    # Changing which window slots are valid must change the output.
    assert not torch.allclose(o_all, o_some)
    # But masking out slots that were already absent (none) is a no-op.
    o_again = _block_attn(q, kv, sink, some_valid, D**-0.5)
    torch.testing.assert_close(o_some, o_again, rtol=1e-5, atol=1e-5)


def test_block_sparse_attention_sink_absorbs_probability():
    # A large positive sink logit should pull probability mass off the real
    # keys, shrinking the output magnitude toward zero (sink has zero value).
    B, T, H, D, W = 1, 1, 1, 4, 2
    torch.manual_seed(2)
    q = torch.randn(B, T, H, D)
    kv = torch.randn(B, W + T, D)
    valid = torch.ones(B, W, dtype=torch.bool)
    o_no_sink = _block_attn(q, kv, torch.tensor([-30.0]), valid, 1.0)
    o_big_sink = _block_attn(q, kv, torch.tensor([30.0]), valid, 1.0)
    assert o_big_sink.abs().sum() < o_no_sink.abs().sum()


# NOTE: the draft RoPE norm-preservation test was removed with the hand-written
# `_apply_dspark_rope_hf` helper. The draft now applies RoPE via the shared aiter
# fused kernel (`attn.rotary_emb.forward`, GPT-J interleaved) — the same op the V4
# target uses and covers — so there is no DSpark-specific RoPE path left to unit
# test here (it needs a GPU + a real _V4RoPE, out of scope for these CPU tests).


# ---- Phase 2: confidence-scheduled verification (Hardware-Aware Scheduler) ----


def test_survival_probabilities_monotone_cumprod():
    from atom.spec_decode.dspark_scheduler import survival_probabilities

    c = torch.tensor([[0.9, 0.8, 0.5], [1.0, 0.5, 0.5]])
    a = survival_probabilities(c)
    # cumulative product
    torch.testing.assert_close(a, torch.tensor([[0.9, 0.72, 0.36], [1.0, 0.5, 0.25]]))
    # monotonically non-increasing along block axis
    assert torch.all(a[:, 1:] <= a[:, :-1] + 1e-6)


def test_sts_calibration_is_order_preserving():
    from atom.spec_decode.dspark_scheduler import calibrate_confidence

    c = torch.tensor([[0.6, 0.9, 0.3, 0.95]])
    T = torch.tensor([2.0, 2.0, 2.0, 2.0])
    cal = calibrate_confidence(c, T)
    # Temperature scaling on the logit preserves the ranking within a row.
    assert torch.argsort(c[0]).tolist() == torch.argsort(cal[0]).tolist()
    # T=None is a no-op.
    torch.testing.assert_close(calibrate_confidence(c, None), c.clamp(1e-6, 1 - 1e-6))


def test_scheduler_flat_sps_keeps_all_high_confidence():
    # With a FLAT sps (no batch penalty) and high confidence, throughput keeps
    # rising as we admit tokens, so the scheduler verifies the whole block.
    from atom.spec_decode.dspark_scheduler import schedule_prefix_lengths

    conf = torch.tensor([[0.99, 0.99, 0.99, 0.99, 0.99]])
    sps = torch.ones(64)  # flat → admitting always raises tau*SPS
    ell = schedule_prefix_lengths(conf, sps, early_stop=True)
    assert ell == [5]


def test_scheduler_prunes_low_confidence_suffix():
    # High prefix, collapsing suffix: cumulative survival of late positions ~0,
    # so admitting them past the SPS penalty stops helping → truncated.
    from atom.spec_decode.dspark_scheduler import schedule_prefix_lengths

    conf = torch.tensor([[0.95, 0.9, 0.05, 0.05, 0.05]])
    # Steeply decreasing SPS so each extra verified token costs throughput.
    sps = torch.linspace(1.0, 0.1, steps=16)
    ell = schedule_prefix_lengths(conf, sps, early_stop=True)
    assert 0 <= ell[0] <= 2  # keeps the confident prefix, drops the dead suffix


def test_scheduler_heavy_load_shrinks_budget():
    # Same confidence, but a sharper SPS dropoff (heavier load) must verify
    # fewer or equal tokens than a gentle dropoff (load-adaptive behavior).
    from atom.spec_decode.dspark_scheduler import schedule_prefix_lengths

    conf = torch.tensor([[0.9, 0.85, 0.8, 0.75, 0.7]])
    gentle = torch.linspace(1.0, 0.9, steps=16)
    sharp = torch.linspace(1.0, 0.2, steps=16)
    ell_gentle = schedule_prefix_lengths(conf, gentle, early_stop=True)
    ell_sharp = schedule_prefix_lengths(conf, sharp, early_stop=True)
    assert ell_sharp[0] <= ell_gentle[0]


def test_scheduler_multi_request_global_topk():
    # Two requests: one confident, one weak. Under a batch penalty the scheduler
    # should give the confident request more verify budget than the weak one.
    from atom.spec_decode.dspark_scheduler import schedule_prefix_lengths

    conf = torch.tensor(
        [
            [0.97, 0.95, 0.93, 0.9, 0.88],  # strong
            [0.4, 0.2, 0.1, 0.05, 0.02],  # weak
        ]
    )
    sps = torch.linspace(1.0, 0.3, steps=32)
    ell = schedule_prefix_lengths(conf, sps, early_stop=True)
    assert ell[0] >= ell[1]


# ---------------------------------------------------------------------------
# AF_PIECEWISE generic contract (atom/model_ops/attn_ffn_piecewise.py)
# ---------------------------------------------------------------------------


def test_zero_copy_buffers_stage_returns_a_fixed_address():
    from atom.model_ops.attn_ffn_piecewise import ZeroCopyBuffers

    bufs = ZeroCopyBuffers(max_tokens=8)
    bufs.alloc("q", width=4, dtype=torch.float32, device="cpu")
    assert bufs.fits(8) and not bufs.fits(9)

    first = bufs.stage("q", torch.ones(3, 4))
    second = bufs.stage("q", torch.full((3, 4), 2.0))
    # Same address across steps is the entire point: the graph was captured
    # reading it. A fresh tensor per step would replay against stale memory.
    assert first.data_ptr() == second.data_ptr()
    torch.testing.assert_close(second, torch.full((3, 4), 2.0))
    # Unregistered name / None tensor pass through rather than raising: an input
    # absent in this config (no indexer, say) still flows through the signature.
    assert bufs.stage("not_registered", torch.ones(2)) is not None
    assert bufs.stage("q", None) is None


def test_zero_copy_names_resolve_by_name_not_position():
    from atom.model_ops.attn_ffn_piecewise import resolve_zero_copy_names

    names = ("x", "q", "positions")
    assert resolve_zero_copy_names(names, None, default_excluded=("positions",)) == {
        "x",
        "q",
    }
    # Env override selects explicitly; empty means copy everything.
    assert resolve_zero_copy_names(names, "x,q") == {"x", "q"}
    assert resolve_zero_copy_names(names, "") == frozenset()
    # A name that is not an input is a typo, not a silently-ignored entry --
    # which is exactly what the old positional index set could not detect.
    try:
        resolve_zero_copy_names(names, "postions")
        assert False, "expected ValueError for an unknown name"
    except ValueError as e:
        assert "postions" in str(e)


def test_v4_core_inputs_come_from_the_signature_and_are_all_zero_copy():
    # There is no second declaration to drift from any more: the inputs and
    # their order ARE core's parameter list. Pin the resolved contract instead,
    # since the runner expands staged inputs by it.
    import types

    import pytest

    try:
        from atom.models.deepseek_v4 import V4AttnFfn
    except ImportError as e:
        if "aiter" not in str(e):
            raise
        pytest.skip(f"requires aiter to import deepseek_v4: {e}")

    assert V4AttnFfn.core_input_names() == (
        "x",
        "q",
        "kv_pre",
        "qr",
        "qr_scale",
        "positions",
        "idx_q_quant",
        "idx_weights",
        "idx_q_scale",
    )

    inst = V4AttnFfn.__new__(V4AttnFfn)
    V4AttnFfn.__init__(inst, layer=types.SimpleNamespace(), runner=None, enabled=False)
    # positions included: its old exclusion was a stale dspark_ragged_lens_gpu
    # (71a3e941), not positions itself.
    assert set(inst.zero_copy_names) == set(inst.input_names)


def test_core_with_var_kwargs_is_rejected():
    # **kwargs carries no order, and the runner expands by order. Better to fail
    # at class use than to mis-assign inputs at replay.
    from atom.model_ops.attn_ffn_piecewise import AttnFfnPiecewise

    class _Bad(AttnFfnPiecewise):
        def core(self, **named):
            return None

    try:
        _Bad.core_input_names()
        assert False, "expected TypeError for **kwargs core"
    except TypeError as e:
        assert "explicitly" in str(e)
