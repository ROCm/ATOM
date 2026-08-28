# SPDX-License-Identifier: MIT
"""Unit tests for DeepSeek-V4 DSpark drafter (Phase 1).

Covers the self-contained, GPU-free pieces: Markov head + Confidence head
numerics, and SpeculativeConfig DSpark detection/routing.
"""

import types

import pytest

pytest.importorskip("aiter", reason="the compiled draft imports aiter at module load")

import torch

from atom.model_ops.v4_kernels.dspark_fp8_indices import DSparkIndexBuffers
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
# AF_PIECEWISE generic contract (atom/utils/attn_ffn_piecewise.py)
# ---------------------------------------------------------------------------


def _core_probe(
    *, piecewise, capturing, graph_ready, dummy=False, capture=True, decode=True
):
    """Drive a decorated core once against fake collaborators and report which
    of capture / replay / deliver / bare-core it chose.

    `capture` is the mode gate (AF_PIECEWISE on). Off, the core never records a
    graph of its own -- plain PIECEWISE, eager core plus a stabilised output.
    `decode` is the eligibility gate: prefill is never captured, whatever the
    mode, because its shapes are one-off."""
    import types

    from atom.utils.attn_ffn_piecewise import piecewise_core

    log = []

    class FakeRunner:
        def has_graph(self, key):
            return graph_ready

        def input_buffers(self, inputs, input_names, upstream):
            # (what the graph reads, what replay has to refresh)
            return dict(inputs), {}

        def capture(self, key, read_from, refresh, core, out_buf):
            log.append("capture")

        def replay(self, key, inputs):
            log.append("replay")
            return "replayed"

    class FakeOutputs:
        def slot(self, key, sample):
            return sample

        def deliver(self, key, out):
            log.append("deliver")
            return out

    @piecewise_core()
    def core(layer, *, x):
        log.append("core")
        return x

    fc = types.SimpleNamespace(
        context=types.SimpleNamespace(is_dummy_run=dummy),
        # `_is_decode` reads `attn_metadata.state` by NAME, so a stub enum-alike
        # is enough and this file needs no AttnState import.
        attn_metadata=types.SimpleNamespace(
            state=types.SimpleNamespace(name="DECODE" if decode else "PREFILL_NATIVE"),
            max_seqlen_q=1,
        ),
        in_hipgraph=capturing,
    )
    core(
        types.SimpleNamespace(layer_name="l0"),
        runner=FakeRunner(),
        outputs=FakeOutputs(),
        piecewise=piecewise,
        capture=capture,
        forward_context=fc,
        x=torch.ones(4),
    )
    return log


def test_decorated_core_picks_capture_replay_or_eager():
    # The decision table, pinned. Every row that is not a replay must end in a
    # deliver: whatever computed the output still owes it to the fixed address
    # the downstream dense piece reads.
    P = _core_probe

    # Not piecewise: no graph cache, and no slot to deliver to either.
    assert P(piecewise=False, capturing=False, graph_ready=False) == ["core"]
    assert P(piecewise=False, capturing=True, graph_ready=True) == ["core"]

    # PREFILL is never captured, whatever the mode: its shapes are one-off and
    # the compressor's prefill plan is sliced to an actual count, not a
    # graph-fixed capacity. It still owes the downstream piece a deliver.
    #
    # This replaced a `num_tokens <= 512` bound, which was trying to say the
    # same thing in the wrong units -- at DSpark q=6 it also cut every decode
    # above bs~85, silently disabling AF for the three largest buckets.
    assert P(piecewise=True, capturing=False, graph_ready=True, decode=False) == [
        "core",
        "deliver",
    ]
    assert P(piecewise=True, capturing=True, graph_ready=False, decode=False) == [
        "core",
        "deliver",
    ]

    # Capture pass, first time this key appears: warm up, record, then compute
    # the real answer (the recording fed on clones).
    assert P(piecewise=True, capturing=True, graph_ready=False) == [
        "core",
        "capture",
        "core",
        "deliver",
    ]
    # Real step with a graph ready: replay only.
    assert P(piecewise=True, capturing=False, graph_ready=True) == ["replay"]
    # Capture pass revisiting a key, and a real step never captured: eager.
    assert P(piecewise=True, capturing=True, graph_ready=True) == ["core", "deliver"]
    assert P(piecewise=True, capturing=False, graph_ready=False) == ["core", "deliver"]

    # A dummy run never touches the cache, but still owes the slot.
    assert P(piecewise=True, capturing=True, graph_ready=False, dummy=True) == [
        "core",
        "deliver",
    ]

    # Plain PIECEWISE (capture gate off): the core never records or replays a
    # graph of its own, whatever the capture pass / cache says -- it runs eager
    # and only stabilises its output. This is the path the three-way branch in
    # v4_attn_compress used to hand-code outside the decorator.
    assert P(piecewise=True, capturing=True, graph_ready=True, capture=False) == [
        "core",
        "deliver",
    ]
    assert P(piecewise=True, capturing=False, graph_ready=True, capture=False) == [
        "core",
        "deliver",
    ]


def test_non_tensor_params_are_passthrough_not_graph_inputs():
    # A bool/int/enum config arg on a core is forwarded to the body untouched
    # (and baked at capture), NOT treated as a graph input to clone or capture
    # on. This is what lets a model keep its own config flags in its signature --
    # e.g. V4's `compressor_already_launched` -- without the decorator choking on
    # a non-tensor. The split comes off the annotations.
    import torch as _t

    from atom.utils.attn_ffn_piecewise import piecewise_core

    seen = {}

    @piecewise_core()
    def core(layer, *, x: _t.Tensor, flag: bool = False, n: int = 0):
        seen["flag"] = flag
        seen["n"] = n
        return x

    # Only the tensor is a graph input; the annotated non-tensors are config.
    assert core.input_names == ("x",)
    assert core.passthrough_names == ("flag", "n")
    # piecewise=False just runs the body, and the config args reach it as passed.
    out = core(object(), piecewise=False, x=_t.ones(3), flag=True, n=7)
    assert out.shape == (3,)
    assert seen == {"flag": True, "n": 7}


def test_runner_copies_only_what_is_not_zero_copy():
    import torch as _t

    from atom.utils.cuda_graph import CudagraphCaptureRunner

    r = CudagraphCaptureRunner()
    inputs = {"x": _t.ones(4, 8), "positions": _t.arange(4)}
    read_from, refresh = r.input_buffers(inputs, ("x", "positions"), frozenset({"x"}))
    # A zero-copy input is captured on the caller's own tensor and never
    # refreshed -- that is what makes its producer responsible for the address.
    assert read_from["x"] is inputs["x"]
    assert "x" not in refresh
    # Everything else is cloned, and the clone is what replay copies into.
    assert read_from["positions"] is not inputs["positions"]
    assert refresh["positions"] is read_from["positions"]


def test_v4_core_copies_nothing_per_step():
    # Nothing is copied, and structurally so: the core is the batch-shaped half
    # alone, and its inputs all come from the dense piece immediately upstream,
    # whose graph writes them to the same address every replay. `positions` cost
    # ~5pts when captured on (8f86bbaf) and is not an input here at all any
    # more -- both its readers (`_qk_norm_rope`, `_fill_csa_paged_compress`) are
    # token-shaped and stayed in the dense pieces. If a future change moves a
    # token-shaped reader back INTO the core, this is what should stop it.
    import pytest

    try:
        from atom.models.deepseek_v4 import DeepseekV4Attention

        core = DeepseekV4Attention._attn_compress
    except ImportError as e:
        if "aiter" not in str(e):
            raise
        pytest.skip(f"requires aiter to import deepseek_v4: {e}")

    assert set(core.zero_copy_names) == set(core.input_names)


def test_copy_per_step_rejects_a_bare_string():
    # `("positions")` is a string, not a 1-tuple; `frozenset` of it is a set of
    # characters, which subtracts nothing from the input names -- so the entry
    # silently means "copy nothing" while reading as its opposite. That shipped.
    import pytest

    from atom.utils.attn_ffn_piecewise import piecewise_core

    with pytest.raises(TypeError, match="trailing comma"):

        @piecewise_core(copy_per_step="positions")
        def core(layer, *, x):
            return x


def test_topk_cannot_reach_qr_when_the_projection_was_handed_in():
    # `_attn_pre` passes None for `qr`/`qr_scale` on the AF path, which is only
    # safe because the core's single use of them -- `Indexer.topk` -- never
    # reaches them once `pre_q_quant` is given. That short-circuit is the whole
    # justification, so assert it directly rather than trusting the read.
    import types

    import pytest

    try:
        from atom.models.deepseek_v4 import Indexer
    except ImportError as e:
        if "aiter" not in str(e) and "forward_context" not in str(e):
            raise
        pytest.skip(f"deepseek_v4 not importable here: {e}")

    def _explode(*a, **k):
        raise AssertionError("forward_batched ran -- qr would have been read")

    sentinel = object()
    stub = types.SimpleNamespace(
        score_topk_from=lambda q, w, s: (q, w, s),
        forward_batched=_explode,
    )
    out = Indexer.topk(
        stub,
        None,  # x_full
        _explode,  # qr_full: reading it at all is a failure
        None,  # positions
        _explode,  # qr_full_scale
        pre_q_quant=sentinel,
        pre_weights="w",
        pre_q_scale="s",
    )
    assert out == (sentinel, "w", "s")


def test_v4_core_inputs_come_from_the_signature():
    # There is no second declaration to drift from: the inputs and their order
    # ARE the core's parameter list, minus the leading layer. The runner clones
    # and refreshes them by it.
    import pytest

    try:
        from atom.models.deepseek_v4 import DeepseekV4Attention

        core = DeepseekV4Attention._attn_compress
    except ImportError as e:
        if "aiter" not in str(e):
            raise
        pytest.skip(f"requires aiter to import deepseek_v4: {e}")

    # The core is the ONE batch-shaped kernel now: the compressor, whose grid is
    # `graph_bs * per_seq_bound`. `x` is all it needs. The indexer top-k left
    # with the FP4 default -- its varqlen path is token-shaped -- and everything
    # else token-shaped was already in the dense pieces. Nine inputs to one.
    assert core.input_names == ("x",)


def test_core_with_var_kwargs_is_rejected():
    # **kwargs carries no order, and the runner expands by order. Better to fail
    # at decoration than to mis-assign inputs at replay.
    from atom.utils.attn_ffn_piecewise import piecewise_core

    try:

        @piecewise_core()
        def _bad(layer, **named):
            return None

        assert False, "expected TypeError for a **kwargs core"
    except TypeError as e:
        assert "explicitly" in str(e)


# ----------------------------------------------------------------------------
# torch.compile boundary
#
# These guard the two SILENT failure modes of compiling the draft: a baked
# `is_dummy_run` (draft reads a zero KV window forever -> acceptance collapses,
# output stays correct) and a decorator that quietly does nothing at all.
# ----------------------------------------------------------------------------


def test_support_torch_compile_is_actually_applied():
    # Adding the decorator is a NO-OP unless dispatch reaches it: it replaces
    # __call__ -> forward, so the class must define `forward` (not the old
    # `forward_spec`) and the ctor must take `atom_config`.
    import inspect

    from atom.models.deepseek_v4_dspark import _DSparkInner
    from atom.utils.decorators import TorchCompileWrapperWithCustomDispatcher

    assert TorchCompileWrapperWithCustomDispatcher in _DSparkInner.__bases__
    assert hasattr(_DSparkInner, "forward")
    assert not hasattr(_DSparkInner, "forward_spec")

    # dynamic_arg_dims inference marks dim 0 of every param annotated exactly
    # torch.Tensor, and raises at import time if it finds none.
    params = inspect.signature(_DSparkInner.forward).parameters
    tensor_args = [n for n, p in params.items() if p.annotation is torch.Tensor]
    assert tensor_args == ["input_ids", "positions"]

    # The decorator's replacement __init__ is (self, atom_config, **kwargs), so
    # anything passed positionally after atom_config would fail to construct.
    init = inspect.signature(_DSparkInner.__init__).parameters
    assert list(init) == ["self", "atom_config", "kwargs"]


def test_block_plan_takes_no_is_dummy_run():
    # _build_block_plan is traced. Its old is_dummy_run branch guarded nothing
    # (the live branch is pure tensor arithmetic over `positions`), but it WOULD
    # bake from the warmup trace and zero the window permanently. Keep it gone.
    import inspect

    from atom.models.deepseek_v4_dspark import _build_block_plan

    assert "is_dummy_run" not in inspect.signature(_build_block_plan).parameters


def test_traced_region_reads_no_per_step_globals():
    # Under CompilationLevel >= DYNAMO_ONCE the custom dispatcher evaluates no
    # guards, so any per-step mutable global read in the traced region is frozen
    # at whatever the first call saw. Turn the invariant into a build failure
    # instead of a code-review hope.
    #
    # Scanned via AST, not text: identifiers only, so prose in docstrings and
    # comments (which necessarily discuss is_dummy_run) doesn't trip it.
    #
    # DSparkLayer.dspark_attention is deliberately absent: it sits behind the
    # opaque torch.ops.aiter.dspark_block_attention op, runs eagerly every step,
    # and its forward-context reads therefore cannot bake.
    import ast
    import inspect
    import textwrap

    from atom.models import deepseek_v4_dspark as m

    traced = [
        m._DSparkInner.forward,
        m._build_block_plan,
        m._dspark_block_topk_idxs,
        m.DSparkLayer.forward_block,
    ]
    banned = {"is_dummy_run", "get_forward_context", "environ"}
    for fn in traced:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
        leaked = used & banned
        assert not leaked, f"{fn.__qualname__} reads {sorted(leaked)}"


def test_num_draft_change_raises():
    # num_draft is a python int, so it is baked into the compiled graph. It is
    # constant in practice (min(mtp_k, window_size)); fail loudly rather than
    # replay a graph built for another width.
    from atom.models.deepseek_v4_dspark import DeepseekV4DSpark

    class _StubInner:
        def __init__(self):
            self.calls = []

        def forward(self, input_ids, positions, num_draft):
            # Mirrors the real contract: the compiled region returns hidden
            # state, not tokens.
            self.calls.append(num_draft)
            return "normed", "hc_hidden"

        __call__ = forward

        @staticmethod
        def head_and_sample(normed, hc_hidden, anchor_ids):
            return "draft", "conf"

    class _StubCtx:
        is_dummy_run = False

    stub = DeepseekV4DSpark.__new__(DeepseekV4DSpark)
    stub.model = _StubInner()
    stub.block_size = 5
    stub._compiled_num_draft = None

    import atom.utils.forward_context as fc

    ids = torch.zeros(2, dtype=torch.int32)
    pos = torch.zeros(2, dtype=torch.int64)
    saved = fc.get_forward_context
    fc.get_forward_context = lambda: type("_FC", (), {"context": _StubCtx()})()
    try:
        assert stub.forward_spec(ids, pos, num_draft=5) == ("draft", "conf")
        try:
            stub.forward_spec(ids, pos, num_draft=6)
            assert False, "expected ValueError on draft-width change"
        except ValueError as e:
            assert "5 -> 6" in str(e)
    finally:
        fc.get_forward_context = saved


def test_write_context_kv_stays_eager_and_still_writes():
    # write_context_kv is deliberately NOT compiled -- the decorator replaces
    # only __call__, so every other method is untouched. That is what keeps its
    # is_dummy_run early-return, its wildly dynamic num_tokens, and swa_write's
    # variable Triton grid out of the traced region.
    #
    # It lives on DSparkDraftModel (the wrapper's base) rather than on the inner
    # module: the inner is the compiled one, and nothing about absorbing the
    # target's context belongs inside the traced block forward. Assert it is
    # reachable as a plain method, is not the compiled entry point, and still
    # fans out to one write per stage.
    from atom.models.deepseek_v4_dspark import DeepseekV4DSpark, _DSparkInner
    from atom.models.dspark_draft import DSparkDraftModel

    assert callable(DeepseekV4DSpark.write_context_kv)
    assert DeepseekV4DSpark.write_context_kv is DSparkDraftModel.write_context_kv
    # The traced entry point is the inner's forward, and this is not it.
    assert DeepseekV4DSpark.write_context_kv is not _DSparkInner.forward

    calls = []

    class _StageStub:
        def write_context_kv(self, ctx_hidden, positions):
            calls.append(ctx_hidden.shape[0])

    class _StubCtx:
        is_dummy_run = False

    stages = [_StageStub(), _StageStub(), _StageStub()]

    class _Wrapper(DSparkDraftModel):
        # project_context is stage 0's main_proj/main_norm; stub it so the test
        # covers the fan-out, not the projection.
        def project_context(self, aux_concat):
            return aux_concat

        @property
        def context_layers(self):
            return stages

    import atom.utils.forward_context as fc

    saved = fc.get_forward_context
    fc.get_forward_context = lambda: type("_FC", (), {"context": _StubCtx()})()
    try:
        DSparkDraftModel.write_context_kv(_Wrapper(), torch.zeros(4, 2), None)
    finally:
        fc.get_forward_context = saved
    assert calls == [4, 4, 4], "one rolling-KV write per stage"


def test_attention_is_reached_only_through_the_opaque_op():
    # The fused qk_norm_rope_maybe_quant lazily JIT-builds a flydsl kernel, and
    # Dynamo cannot trace the builder (it hits function.__new__). Tracing into
    # dspark_attention graph-breaks, and the break splits the forward into two
    # Dynamo graphs -- the second trips "VllmBackend can only be called once".
    #
    # The V4 target calls the identical kernel safely because its call site is
    # behind torch.ops.aiter.v4_attn_compress. Mirror that, and keep it mirrored.
    import ast
    import inspect
    import textwrap

    import torch as _t

    from atom.models import deepseek_v4_dspark as m

    op = _t.ops.aiter.dspark_block_attention
    assert op is not None
    # Registered as a split point: backends._split_judge_func tests the
    # attribute, not compilation_config.splitting_ops.
    assert getattr(op, "spliting_op", False) is True

    # forward_block (traced) must go through the op, never call the method.
    tree = ast.parse(textwrap.dedent(inspect.getsource(m.DSparkLayer.forward_block)))
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                calls.add(fn.attr)
    assert "dspark_block_attention" in calls
    assert "dspark_attention" not in calls, (
        "forward_block must reach attention through the opaque op; calling the "
        "method directly puts the flydsl JIT builder back inside the trace"
    )


def test_opaque_attention_fake_impl_matches_real_output_shape():
    # A wrong fake impl mis-sizes every downstream op in the compiled graph.
    # dspark_attention returns [B, T, dim] -- same shape as its input x.
    from torch._subclasses.fake_tensor import FakeTensorMode

    from atom.models.deepseek_v4_dspark import _dspark_block_attention_fake

    B, T, W, dim = 2, 5, 128, 64
    with FakeTensorMode():
        x = torch.empty(B, T, dim, dtype=torch.bfloat16)
        out = _dspark_block_attention_fake(
            x,
            torch.empty(B, dtype=torch.int64),
            torch.empty(B, T, dtype=torch.int64),
            torch.empty(B, W, dtype=torch.bool),
            torch.empty(B, T, W + T, dtype=torch.int32),
            "layer",
        )
    assert out.shape == (B, T, dim)
    assert out.dtype == torch.bfloat16


def test_lm_head_and_markov_sampler_are_outside_the_compiled_region():
    # Under TP, ParallelLMHead's aiter all_gather lazily JIT-loads an aiter
    # module on first call, and tracing that loader reaches
    # shutil.which() -> posix.stat, which Dynamo cannot trace. Same
    # graph-break-then-"VllmBackend can only be called once" failure as the
    # attention path. The V4 target draws the line in the same place: its LM
    # head lives in compute_logits, outside the decorated model's forward.
    import ast
    import inspect
    import textwrap

    from atom.models.deepseek_v4_dspark import _DSparkInner

    tree = ast.parse(textwrap.dedent(inspect.getsource(_DSparkInner.forward)))
    attrs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attrs.add(node.attr)
    assert "get_logits" not in attrs, "LM head must not be traced"
    assert "forward_head" not in attrs, "Markov sampler must not be traced"

    # That the wrapper actually runs them is asserted where it can be
    # observed -- `test_forward_spec_passes_the_batch_through_unpadded` records
    # `head_and_sample`'s arguments, which only happens if it was called.

    # head_and_sample is a plain method: the decorator replaces only __call__.
    assert callable(_DSparkInner.head_and_sample)


@pytest.mark.parametrize("is_dummy", [True, False], ids=["dummy", "real"])
@pytest.mark.parametrize("B", [1, 4])
def test_forward_spec_passes_the_batch_through_unpadded(is_dummy, B):
    """No pad, on any batch, dummy or not.

    `forward_spec` used to repeat a B==1 dummy run to B==2: warmup was the
    first traced call and `mark_dynamic` cannot make a size-1 dim dynamic.
    Under `--enable-dp-attention` that pad ran 2*T MoE rows against a
    `dp_metadata` sized for B==1 and tripped reduce_scatterv, so #1915 fixed
    it upstream instead -- `warmup_model` gives a block drafter >= 2 sequences,
    so the first trace is already B>=2 -- and dropped the pad entirely.

    The two tests that covered the pad were left behind asserting a `repeat(2)`
    that no longer happens and an `"is_dummy and" in src` that is no longer in
    the source. Both were red from the day #1915 landed and nobody was told,
    because this module `importorskip`s aiter and CI has none. This is the
    contract that replaced them, and it is read off the call rather than off
    the source: the pad would show up here as a batch the inner model did not
    ask for.

    `is_dummy` is still varied because "dummy runs are not special either" is
    half of what changed.

    Batch padding lives in the drafter now, not here: `propose` rounds the block
    up to the target's captured running_bs before calling this
    (`test_propose_drafts_at_the_captured_graph_bs`). This end stays pass-through.
    """
    from atom.models.deepseek_v4_dspark import DeepseekV4DSpark

    T = 5
    seen = {}

    class _StubInner:
        def __call__(self, input_ids, positions, num_draft):
            seen["B"] = input_ids.shape[0]
            seen["positions"] = positions.tolist()
            return torch.zeros(input_ids.shape[0] * num_draft, 4), torch.zeros(
                input_ids.shape[0], num_draft, 4
            )

        forward = __call__

        @staticmethod
        def head_and_sample(normed, hc_hidden, anchor_ids):
            seen["normed_rows"] = normed.shape[0]
            seen["hc_B"] = hc_hidden.shape[0]
            seen["anchor_B"] = anchor_ids.shape[0]
            return "draft", "conf"

    stub = DeepseekV4DSpark.__new__(DeepseekV4DSpark)
    stub.model = _StubInner()
    stub.block_size = T
    stub._compiled_num_draft = None

    import atom.utils.forward_context as fc

    saved = fc.get_forward_context
    fc.get_forward_context = lambda: type(
        "_FC", (), {"context": type("_C", (), {"is_dummy_run": is_dummy})()}
    )()
    try:
        stub.forward_spec(
            torch.full((B,), 7, dtype=torch.int32),
            torch.full((B,), 11, dtype=torch.int64),
            num_draft=T,
        )
    finally:
        fc.get_forward_context = saved

    assert seen["B"] == B
    # Outputs always come back sliced to the real batch.
    assert (seen["normed_rows"], seen["hc_B"], seen["anchor_B"]) == (B * T, B, B)


# --------------------------------------------------------------------------- #
# DSpark's wiring into the draft-graph machine. The machine's own invariants
# live in tests/test_draft_graph.py, which needs no aiter and so runs in CI.
# --------------------------------------------------------------------------- #

_GRAPH_BS = [1, 2, 4, 8, 16, 32, 48, 64, 128, 256]


def _stub_forward_context(*, scheduled_bs, target_bs, use_cudagraph=True):
    context = types.SimpleNamespace(
        scheduled_bs=scheduled_bs,
        running_bs=target_bs,
        # Rows, not sequences -- a DSpark ragged step leaves these wildly apart.
        # The drafter pads in SEQUENCES, so a stub that agreed would prove
        # nothing about which of the two it reads.
        running_tokens=target_bs * 337,
        is_dummy_run=False,
        is_draft=False,
        positions=None,
        forward_mode=types.SimpleNamespace(
            use_cudagraph=use_cudagraph, running_bs=target_bs
        ),
    )
    # `prepare_decode` publishes the ring slots at the PADDED batch, so the stub
    # does too -- the block slices to that length and nothing stages it.
    attn_metadata = types.SimpleNamespace(
        state_slot_out=torch.arange(max(target_bs, scheduled_bs), dtype=torch.int32)
        + 100
    )
    return types.SimpleNamespace(context=context, attn_metadata=attn_metadata)


def _proposer_with_graph_bs(monkeypatch, *, eplb=False, mtp_k=5, window=128):
    """A DSparkProposer carrying only what the block pass reads."""
    from atom.spec_decode.dspark_proposer import DSparkProposer

    monkeypatch.setattr(DSparkProposer, "_with_draft", False, raising=False)
    monkeypatch.setattr(
        DSparkProposer, "aux_for", lambda self, h: [torch.zeros(1)], raising=False
    )
    monkeypatch.setattr(
        DSparkProposer,
        "_refresh_dp_metadata",
        lambda self, fc, n: None,
        raising=False,
    )
    monkeypatch.setattr(DSparkProposer, "verify_scheduler", None, raising=False)

    p = DSparkProposer.__new__(DSparkProposer)
    p.config = types.SimpleNamespace(
        max_num_seqs=256,
        eplb_enable=eplb,
        # What `set_forward_context` reads; the warm goes through it.
        parallel_config=types.SimpleNamespace(data_parallel_size=1),
        compilation_config=types.SimpleNamespace(static_forward_context={}),
    )
    p.device = torch.device("cpu")
    p.mtp_k = mtp_k
    p.model = types.SimpleNamespace(vocab_size=1024, window_size=window)
    p.runner = types.SimpleNamespace(
        capture_sizes=list(_GRAPH_BS),
        attn_metadata_builder=types.SimpleNamespace(
            row_ids=torch.arange(p.config.max_num_seqs + 1, dtype=torch.int32)
        ),
    )
    p._build_draft_graphs()
    return p


def _run_propose(p, fc, real_bs, monkeypatch, seen):
    import atom.spec_decode.dspark_proposer as mod

    def _backbone(ids, pos, num_draft):
        seen["B"] = ids.shape[0]
        seen["positions_B"] = pos.shape[0]
        seen["slots"] = fc.attn_metadata.state_slot_out.clone()
        return ("normed", ids.shape[0])

    class _Inner:
        # A real class, not SimpleNamespace: __call__ is looked up on the type.
        __call__ = staticmethod(_backbone)

        # The real bundle, not a stub: `mask_pad_tail` is what keeps a padded
        # block from scattering draft KV, and `seen["batch_ids"]` is the only
        # place these tests can watch it.
        bufs = DSparkIndexBuffers.allocate(
            p.config.max_num_seqs, p.mtp_k, int(p.model.window_size), p.device
        )

        @classmethod
        def index_buffers(cls, draft, window, device):
            return cls.bufs

        @staticmethod
        def head_and_sample(normed, hc_hidden, anchor_ids):
            seen["batch_ids"] = _Inner.bufs.batch_ids.clone()
            return (
                torch.zeros(hc_hidden, p.mtp_k, dtype=torch.int32),
                torch.zeros(hc_hidden, p.mtp_k),
            )

    p.model.model = _Inner()
    monkeypatch.setattr(mod, "get_forward_context", lambda: fc)
    return p.propose(
        target_token_ids=None,
        target_positions=torch.arange(real_bs, dtype=torch.int64) * 7 + 3,
        target_hidden_states=torch.zeros(real_bs, 4),
        num_reject_tokens=None,
        next_token_ids=torch.full((real_bs,), 5, dtype=torch.int32),
        last_token_indices=torch.arange(real_bs, dtype=torch.int64),
    )


@pytest.mark.parametrize(
    "real_bs,expect_B", [(44, 48), (50, 64), (1, 1), (64, 64), (35, 48)]
)
def test_propose_drafts_at_the_captured_graph_bs(monkeypatch, real_bs, expect_B):
    """The block runs at the target's running_bs, not the live batch size.

    Without this the drafter hands aiter a fresh width on every distinct decode
    batch and flydsl builds a new hgemm for each -- in-process, so the stall
    lands mid-serve and every restart pays it again.
    """
    seen = {}
    p = _proposer_with_graph_bs(monkeypatch)
    fc = _stub_forward_context(scheduled_bs=real_bs, target_bs=expect_B)
    out = _run_propose(p, fc, real_bs, monkeypatch, seen)

    assert seen["B"] == expect_B
    assert seen["positions_B"] == expect_B
    # The block does not touch the target's ring slots: they arrive already at
    # the padded length, so there is nothing to install and nothing to restore.
    assert seen["slots"].shape[0] >= expect_B
    assert seen["slots"][:real_bs].tolist() == list(range(100, 100 + real_bs))
    assert out.shape[0] == real_bs

    # ...but the rows it fabricated must scatter no draft KV. Their ring slot is
    # the 0 `prepare_decode` fills that tail with, and 0 is a real position, so
    # an unmasked pad row writes into another request's window.
    t = p.mtp_k
    ids = seen["batch_ids"]
    assert ids[: real_bs * t].tolist() == [i // t for i in range(real_bs * t)]
    assert ids[real_bs * t : expect_B * t].tolist() == [-1] * (expect_B - real_bs) * t


@pytest.mark.parametrize(
    "cudagraph,eplb,why",
    [
        (False, False, "eager: the target pinned no wider batch to follow"),
        (True, True, "pad rows would poison the expert-load histogram"),
    ],
)
def test_propose_leaves_the_batch_alone_when_nothing_pins_a_wider_one(
    monkeypatch, cudagraph, eplb, why
):
    seen = {}
    p = _proposer_with_graph_bs(monkeypatch, eplb=eplb)
    fc = _stub_forward_context(scheduled_bs=44, target_bs=48, use_cudagraph=cudagraph)
    out = _run_propose(p, fc, 44, monkeypatch, seen)
    assert seen["B"] == 44, why
    assert out.shape[0] == 44


def test_propose_pads_a_dp_sync_dummy_exactly_like_the_rank_with_work(monkeypatch):
    """`is_dummy_run` is per-rank, so the draft's width must not read it.

    One DP step runs the rank holding sequences for real while the others run
    dummies purely to reach the collectives. A width that shrank on the dummies
    would put two shapes into one MoE all_gather.
    """
    widths = []
    for dummy in (False, True):
        seen = {}
        p = _proposer_with_graph_bs(monkeypatch)
        fc = _stub_forward_context(scheduled_bs=44, target_bs=48)
        fc.context.is_dummy_run = dummy
        _run_propose(p, fc, 44, monkeypatch, seen)
        widths.append(seen["B"])
    assert widths == [48, 48]


def test_the_token_map_is_usable_before_any_step_has_padded():
    """The startup sweep runs the block before `propose` ever calls
    `mask_pad_tail`, so `allocate` may not leave this undefined: the scatter
    reads it as a liveness gate and would drop a random subset of the warm.
    """
    bufs = DSparkIndexBuffers.allocate(8, 2, 4, torch.device("cpu"))
    assert bufs.batch_ids.tolist() == [i // 2 for i in range(16)]


def test_the_warm_marks_its_context_as_a_draft(monkeypatch):
    """A capture bakes every Python branch taken while it is made.

    `is_draft` gates the aux-capture hook away from the draft's own forward
    (`Drafter._make_aux_hook`), and the draft shares the target's embedding --
    so a warm that leaves it False records that hook copying the draft's own
    embedding over the buffer it exists to protect, on every replay after.
    """
    seen = {}
    p = _proposer_with_graph_bs(monkeypatch)
    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "0")  # capture needs a GPU
    fc = _stub_forward_context(scheduled_bs=8, target_bs=8)
    assert fc.context.is_draft is False, "the capture builder leaves it off"

    class _Inner:
        @staticmethod
        def __call__(ids, pos, num_draft):
            seen["is_draft"] = fc.context.is_draft
            return ("normed", ids.shape[0])

        @staticmethod
        def head_and_sample(normed, hc_hidden, anchor_ids):
            return None, None

    p.model.model = _Inner()
    import atom.spec_decode.dspark_proposer as mod

    monkeypatch.setattr(mod, "get_forward_context", lambda: fc)
    p.runner.graph_pool = None
    p.runner.rank = 0
    p.warmup_draft_graphs(lambda bs: (fc.attn_metadata, fc.context), None)
    assert seen["is_draft"] is True


def test_the_pad_sentinel_lifts_off_a_row_that_becomes_real_again():
    """The half a single step cannot show: the batch shrinks, then grows.

    `batch_ids` outlives the step, so marking without restoring would leave the
    -1 on rows the next, larger batch fills with real requests -- and a -1 there
    drops that request's draft KV silently, which reads as lost acceptance
    rather than as a bug.
    """
    bufs = DSparkIndexBuffers.allocate(8, 2, 4, torch.device("cpu"))
    row_ids = torch.arange(9, dtype=torch.int32)

    bufs.mask_pad_tail(row_ids, 2, 6)
    assert bufs.batch_ids[:12].tolist() == [0, 0, 1, 1] + [-1] * 8

    bufs.mask_pad_tail(row_ids, 5, 6)
    assert bufs.batch_ids[:12].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1, -1]


def test_confidence_is_sliced_back_before_the_verify_scheduler(monkeypatch):
    """compute_ell reads its batch size off confidence.shape and zips the ell it
    returns against batch.req_ids by position, so a pad row silently shifts every
    request's verify length."""
    from atom.spec_decode.dspark_proposer import DSparkProposer

    seen = {}
    p = _proposer_with_graph_bs(monkeypatch)
    sched = types.SimpleNamespace(
        compute_ell=lambda conf: seen.setdefault("ell_bs", conf.shape[0]),
        set_last_ell=lambda ell: None,
    )
    monkeypatch.setattr(DSparkProposer, "verify_scheduler", sched, raising=False)
    fc = _stub_forward_context(scheduled_bs=44, target_bs=48)
    _run_propose(p, fc, 44, monkeypatch, seen)
    assert seen["B"] == 48
    assert seen["ell_bs"] == 44


def test_the_block_wires_both_its_backbone_and_its_head_into_the_pass(monkeypatch):
    """DSpark's own wiring, not the machine's ordering (that is in
    tests/test_draft_graph.py): the head must be the pass's EPILOGUE, so warming
    reaches it. It has its own per-shape flydsl builder, and leaving it out is
    how `hipModuleLoadData` went 0 -> 4 on the reproducer once.
    """
    ran = []
    p = _proposer_with_graph_bs(monkeypatch)
    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "0")  # capture needs a GPU

    class _Inner:
        @staticmethod
        def __call__(ids, pos, num_draft):
            ran.append("backbone")
            return ("normed", ids.shape[0])

        @staticmethod
        def head_and_sample(normed, hc_hidden, anchor_ids):
            ran.append("head")
            return None, None

    p.model.model = _Inner()
    fc = _stub_forward_context(scheduled_bs=8, target_bs=8)
    import atom.spec_decode.dspark_proposer as mod

    monkeypatch.setattr(mod, "get_forward_context", lambda: fc)
    p.block.warmup(8)
    assert ran == ["backbone", "head"]

    # ...and the forward really is only the backbone, so `capture_epilogue=False`
    # would keep the LM head's all-gather out of a capture.
    ran.clear()
    p.block.forward(
        8,
        anchor_ids=torch.zeros(8, dtype=torch.int32),
        anchor_positions=torch.zeros(8, dtype=torch.int64),
    )
    assert ran == ["backbone"]


@pytest.mark.parametrize("window", [128, 64, 7])
def test_the_block_warms_where_the_whole_rolling_window_is_valid(monkeypatch, window):
    """The seed decides which shape the warm compiles, not merely that it runs.

    An anchor at position 0 masks every window slot but the last, so the warm
    builds a near-empty-window variant and steady-state decode still meets its
    own shape fresh -- the same defect class as leaving the LM head out of the
    warm, and just as invisible: the pass runs, the graph captures, only the
    flydsl build moves back into serving.

    Asserted against the model's own ``window_size`` at three widths, so a seed
    that happened to write the literal 128 would not pass.
    """
    seen = {}
    p = _proposer_with_graph_bs(monkeypatch, window=window)
    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "0")  # capture needs a GPU

    class _Inner:
        @staticmethod
        def __call__(ids, pos, num_draft):
            seen["pos"] = pos.clone()
            return ("normed", ids.shape[0])

        @staticmethod
        def head_and_sample(normed, hc_hidden, anchor_ids):
            return None, None

    p.model.model = _Inner()
    fc = _stub_forward_context(scheduled_bs=8, target_bs=8)
    import atom.spec_decode.dspark_proposer as mod

    monkeypatch.setattr(mod, "get_forward_context", lambda: fc)
    p.block.warmup(8)

    assert seen["pos"].shape[0] == 8
    assert (seen["pos"] >= window).all(), seen["pos"].tolist()


def test_warming_the_block_on_a_dummy_context_is_refused(monkeypatch):
    """A dummy context carries an all-zero rolling window, and the seed cannot
    tell: it writes positions and reads the ring slots off whatever context it
    is handed. Warming there compiles against zeros and shows up only as lost
    acceptance downstream, so the pass refuses instead.

    The armed half is the second call: the same warm on a real context reaches
    the backbone, so the raise below is the guard firing and not the stub
    failing to be reachable.
    """
    reached = []
    p = _proposer_with_graph_bs(monkeypatch)
    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "0")

    class _Inner:
        @staticmethod
        def __call__(ids, pos, num_draft):
            reached.append("backbone")
            return ("normed", ids.shape[0])

        @staticmethod
        def head_and_sample(normed, hc_hidden, anchor_ids):
            return None, None

    p.model.model = _Inner()
    fc = _stub_forward_context(scheduled_bs=8, target_bs=8)
    import atom.spec_decode.dspark_proposer as mod

    monkeypatch.setattr(mod, "get_forward_context", lambda: fc)

    fc.context.is_dummy_run = True
    with pytest.raises(AssertionError, match="dummy"):
        p.block.warmup(8)
    assert reached == []

    fc.context.is_dummy_run = False
    p.block.warmup(8)
    assert reached == ["backbone"]


def test_the_separate_draft_model_path_declares_no_draft_graph(monkeypatch):
    """Kimi-K3 must declare NO pass, not merely an unpaddable one.

    Its draft carries neither `window_size` nor `model.head_and_sample`, which
    the warmup and epilogue reach for -- and `warmup` runs both BEFORE it
    consults the pad/capture gates. So an unpaddable-but-declared pass still
    takes the startup sweep through `_block_warmup_inputs` and dies with an
    AttributeError, which is what leaving this at `pads = False` used to do.
    """
    from atom.spec_decode.dspark_proposer import DSparkProposer

    p = _proposer_with_graph_bs(monkeypatch)
    assert p.draft_graphs and p.block.pads

    monkeypatch.setattr(DSparkProposer, "_with_draft", True, raising=False)
    p._build_draft_graphs()
    assert p.draft_graphs == ()
    # None, not absent and not the pass a previous build left behind: rebuilding
    # is what the flavor probes in these tests do, and `propose` reads this.
    assert p.block is None


def test_qk_norm_rope_short_circuits_dummy_run():
    # warmup_model runs BEFORE allocate_kv_cache, so the SWA plane this kernel
    # writes into is not bound yet. the attention has always guarded dummy_run;
    # called from `_attn_pre` this sits UPSTREAM of that guard and needs its
    # own. Getting that wrong died in `flydsl_hca_compress_attn` on a None
    # kv_cache, so the guard is asserted where it now has to live.
    import types

    import pytest

    try:
        from atom.models.deepseek_v4 import DeepseekV4Attention
    except ImportError as e:
        # `forward_context`: conftest stubs `atom.config` for the rest of the
        # suite and `module_dispatch_ops` then cannot resolve
        # `get_current_cudagraph_runtime_mode`, which is why the sibling V4
        # tests here only pass when this file runs alone. Anything else is a
        # real import break and must not be skipped past.
        if "aiter" not in str(e) and "forward_context" not in str(e):
            raise
        pytest.skip(f"deepseek_v4 not importable here: {e}")

    import atom.models.deepseek_v4 as v4

    T, H, D, RD = 4, 2, 64, 16
    layer = types.SimpleNamespace(
        n_local_heads=H, head_dim=D, rope_head_dim=RD, kv_fp8=False
    )
    fc = types.SimpleNamespace(context=types.SimpleNamespace(is_dummy_run=True))
    saved = v4.get_forward_context
    v4.get_forward_context = lambda: fc
    try:
        # No attn_metadata, no kv_norm, no rotary_emb, no swa_plane on the stub:
        # reaching past the guard is an AttributeError, which is the assertion.
        qkn = DeepseekV4Attention._qk_norm_rope(
            layer,
            torch.zeros(T, H * D),
            torch.zeros(T, D),
            torch.zeros(T, dtype=torch.int32),
        )
    finally:
        v4.get_forward_context = saved

    # Shapes still have to be right -- warmup's output is consumed downstream,
    # and the op's fake impl promises exactly these.
    assert qkn.q_sa.shape == (T, H, D) and qkn.kv.shape == (T, D)
    assert torch.all(qkn.q_sa == 0) and torch.all(qkn.kv == 0)


def test_core_still_binds_the_inputs_that_arrive_none():
    # A core whose signature spans more than one call shape gets None for the
    # inputs of the shape it is not in -- under the narrow split `_attn_compress` is
    # handed the paged Q and not `q`/`kv_pre`. Those parameters still have to be
    # BOUND: dropping them from the call instead was a missing-argument
    # TypeError that only showed up at cudagraph capture on real hardware.
    import types

    from atom.utils.attn_ffn_piecewise import piecewise_core

    seen = {}

    @piecewise_core()
    def core(layer, *, x: torch.Tensor | None, y: torch.Tensor | None):
        seen["x"], seen["y"] = x, y
        return y if x is None else x

    class _Outputs:
        def deliver(self, key, out):
            return out

    fc = types.SimpleNamespace(
        context=types.SimpleNamespace(is_dummy_run=False),
        attn_metadata=object(),
        in_hipgraph=False,
    )
    y = torch.ones(4)
    out = core(
        types.SimpleNamespace(layer_name="l0"),
        runner=None,
        outputs=_Outputs(),
        piecewise=True,
        capture=False,
        forward_context=fc,
        x=None,
        y=y,
    )
    assert seen == {"x": None, "y": y}
    assert out is y


def test_core_rejects_an_all_none_call():
    # The row count comes off an input, so there has to be one. Saying so beats
    # an AttributeError on None deep in the runner.
    import types

    import pytest

    from atom.utils.attn_ffn_piecewise import piecewise_core

    @piecewise_core()
    def core(layer, *, x: torch.Tensor | None):
        return x

    fc = types.SimpleNamespace(
        context=types.SimpleNamespace(is_dummy_run=False),
        attn_metadata=object(),
        in_hipgraph=False,
    )
    with pytest.raises(ValueError, match="every tensor input None"):
        core(
            types.SimpleNamespace(layer_name="l0"),
            runner=None,
            outputs=None,
            piecewise=True,
            capture=False,
            forward_context=fc,
            x=None,
        )


def test_every_half_of_the_core_short_circuits_dummy_run():
    # warmup_model runs BEFORE allocate_kv_cache: the SWA plane and the
    # Compressor/Indexer caches are all unbound. The attention has always guarded
    # dummy_run, but BOTH halves of it now also run from `_attn_pre`, a graph
    # piece earlier -- outside that guard. Each needs its own, and guarding one
    # and not the other is exactly the bug that shipped twice: both times it
    # died in `flydsl_hca_compress_attn` on a None kv_cache.
    #
    # So this asserts the property over EVERY entry point `_attn_pre` can reach
    # under AF, rather than over whichever one was most recently added.
    import types

    import pytest

    try:
        from atom.models.deepseek_v4 import DeepseekV4Attention
    except ImportError as e:
        if "aiter" not in str(e) and "forward_context" not in str(e):
            raise
        pytest.skip(f"deepseek_v4 not importable here: {e}")

    import atom.models.deepseek_v4 as v4

    T, H, D, RD = 4, 2, 64, 16
    q, kv_pre = torch.zeros(T, H * D), torch.zeros(T, D)
    positions = torch.zeros(T, dtype=torch.int32)

    def _explode(*a, **k):
        raise AssertionError("a paged kernel ran: the KV caches are not bound yet")

    # Deliberately bare: anything that reaches past the guard touches an
    # attribute this stub does not have, or an exploding collaborator.
    layer = types.SimpleNamespace(
        n_local_heads=H,
        head_dim=D,
        rope_head_dim=RD,
        kv_fp8=False,
        compress_ratio=4,
        skip_topk=False,
        indexer=types.SimpleNamespace(topk=_explode),
        maybe_compressors_async=_explode,
        _fill_csa_paged_compress=_explode,
    )
    fc = types.SimpleNamespace(context=types.SimpleNamespace(is_dummy_run=True))
    saved = v4.get_forward_context
    v4.get_forward_context = lambda: fc
    try:
        qkn = DeepseekV4Attention._qk_norm_rope(layer, q, kv_pre, positions)
        x = torch.zeros(T, D)
        # `piecewise=False` is the decorator's eager route -- it just runs the
        # body, which is what has the guard.
        assert DeepseekV4Attention._attn_compress(layer, piecewise=False, x=x) is None
        o = DeepseekV4Attention._sparse_attention(layer, qkn, positions)
        assert o.shape == (T, H * D) and torch.all(o == 0)
    finally:
        v4.get_forward_context = saved

    # Shapes still have to be right: warmup's output is consumed downstream and
    # the op's fake impl promises exactly these.
    assert qkn.q_sa.shape == (T, H, D) and qkn.kv.shape == (T, D)
    assert torch.all(qkn.q_sa == 0) and torch.all(qkn.kv == 0)


def test_custom_op_bodies_only_call_methods_that_exist():
    # The custom ops resolve their layer out of `static_forward_context` and
    # then call methods on it, so a method the model no longer has is not a
    # NameError at import -- it is an AttributeError inside a compiled graph, on
    # the GPU, at capture time. That shipped: a op left behind by a reverted
    # design kept calling `self._paged_pre_ran_upstream()` after the method was
    # deleted, and every test here passed because none of them reach `_attn_pre`.
    #
    # So this reads the ops' own bodies and checks each `self.<name>` against
    # the class. Cheap, and it fails at the moment the method disappears.
    import ast
    import inspect

    import pytest

    try:
        import atom.models.deepseek_v4 as v4
    except ImportError as e:
        if "aiter" not in str(e) and "forward_context" not in str(e):
            raise
        pytest.skip(f"deepseek_v4 not importable here: {e}")

    tree = ast.parse(inspect.getsource(v4))
    # The module-level functions registered as custom ops: named in a
    # `direct_register_custom_op` call, or decorated with `mark_spliting_op`.
    registered = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name not in ("direct_register_custom_op", "mark_spliting_op"):
            continue
        for kw in node.keywords:
            if kw.arg in ("op_func", "gen_fake") and isinstance(kw.value, ast.Name):
                registered.add(kw.value.id)
    # `mark_spliting_op` is a decorator, so its op is the function it decorates.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(
            getattr(getattr(d, "func", d), "id", None) == "mark_spliting_op"
            for d in node.decorator_list
        ):
            registered.add(node.name)

    assert registered, "found no custom ops to check -- the scan is broken"

    missing = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in registered:
            continue
        for sub in ast.walk(node):
            # CALLS only. Instance attributes (`self.kv_fp8`, set in __init__)
            # are not on the class and would all read as missing; methods are,
            # and a missing method is what this is for.
            if not isinstance(sub, ast.Call):
                continue
            fn = sub.func
            if (
                isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "self"
                and not callable(getattr(v4.DeepseekV4Attention, fn.attr, None))
            ):
                missing.append(f"{node.name} -> self.{fn.attr}()")

    assert not missing, (
        "custom op bodies call methods DeepseekV4Attention does not have; these "
        "raise inside a compiled graph on the GPU, not at import:\n  "
        + "\n  ".join(sorted(set(missing)))
    )


def test_every_op_fake_agrees_with_its_body():
    # A fake impl that promises a different shape than the body returns does not
    # fail at the op -- it fails as `assert_size_stride` deep inside the compiled
    # graph, on the GPU, with a message about strides. That shipped: the core's
    # no-top-k stand-in returned [T, 1] while its fake promised [T, index_topk].
    #
    # Both are reachable here: the fakes are plain functions, and every body has
    # a dummy_run short-circuit that returns the same shape the real path does
    # (it has to -- warmup output feeds the layers downstream). So compare them.
    import types

    import pytest

    try:
        import atom.models.deepseek_v4 as v4
    except ImportError as e:
        if "aiter" not in str(e) and "forward_context" not in str(e):
            raise
        pytest.skip(f"deepseek_v4 not importable here: {e}")

    T, H, D, RD, TOPK = 4, 2, 64, 16, 32
    layer = types.SimpleNamespace(
        n_local_heads=H,
        head_dim=D,
        rope_head_dim=RD,
        kv_fp8=False,
        compress_ratio=4,
        skip_topk=False,
        indexer=types.SimpleNamespace(index_topk=TOPK),
    )
    q, kv_pre = torch.zeros(T, H * D), torch.zeros(T, D)
    x, positions = torch.zeros(T, D), torch.zeros(T, dtype=torch.int32)

    cfg = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(static_forward_context={"L": layer})
    )
    fc = types.SimpleNamespace(context=types.SimpleNamespace(is_dummy_run=True))
    saved_cfg, saved_fc = v4.get_current_atom_config, v4.get_forward_context
    v4.get_current_atom_config = lambda: cfg
    v4.get_forward_context = lambda: fc
    try:
        A = v4.DeepseekV4Attention
        cases = [
            (
                "v4_qk_norm_rope",
                [t.shape for t in v4._v4_qk_norm_rope_fake(q, kv_pre, positions, "L")],
                [
                    t.shape
                    for t in v4._qkn_to_list(
                        A._qk_norm_rope(layer, q, kv_pre, positions), layer.kv_fp8
                    )
                ],
            ),
            (
                "v4_attn_compress",
                [v4._v4_attn_compress_fake(x, "L")],
                [A._attn_compress(layer, piecewise=False, x=x)],
            ),
            (
                "v4_sparse_attention",
                [
                    v4._v4_sparse_attention_fake(
                        # a real Q: the fake sizes on it, not on `positions`
                        torch.zeros(T, H, D),
                        None,
                        None,
                        None,
                        None,
                        None,
                        positions,
                        None,
                        None,
                        None,
                        "L",
                    ).shape
                ],
                [
                    A._sparse_attention(
                        layer,
                        # a real Q: the body sizes on it and asserts it is there
                        v4.QKNormRopeOut(q_sa=torch.zeros(T, H, D)),
                        positions,
                    ).shape
                ],
            ),
        ]
    finally:
        v4.get_current_atom_config, v4.get_forward_context = saved_cfg, saved_fc

    for name, fake_shapes, real_shapes in cases:
        assert fake_shapes == real_shapes, (
            f"{name}: fake promises {fake_shapes}, body returns {real_shapes}. "
            "These must match or the compiled graph fails on assert_size_stride."
        )


def test_paged_post_refuses_a_missing_q():
    # `forward()`'s narrow branch always calls `v4_sparse_attention`, so whatever
    # fills its Q must run on EVERY narrow path. Gating `v4_qk_norm_rope` on
    # AF_PIECEWISE alone left plain PIECEWISE with no QK-norm at all and a None
    # Q going into an aiter kernel -- a GPU-side failure, and invisible to every
    # test here because none of them reach `_attn_pre`.
    #
    # So the shape of that mistake is asserted on the CPU side instead.
    import types

    import pytest

    try:
        from atom.models.deepseek_v4 import DeepseekV4Attention, QKNormRopeOut
    except ImportError as e:
        if "aiter" not in str(e) and "forward_context" not in str(e):
            raise
        pytest.skip(f"deepseek_v4 not importable here: {e}")

    import atom.models.deepseek_v4 as v4

    T, H, D = 4, 2, 64
    layer = types.SimpleNamespace(
        n_local_heads=H,
        head_dim=D,
        rope_head_dim=16,
        kv_fp8=False,
        compress_ratio=4,
        skip_topk=False,
        indexer=types.SimpleNamespace(index_topk=8),
    )
    # is_dummy_run False, or the short-circuit fires before the assertion.
    fc = types.SimpleNamespace(
        context=types.SimpleNamespace(is_dummy_run=False), attn_metadata=object()
    )
    saved = v4.get_forward_context
    v4.get_forward_context = lambda: fc
    try:
        with pytest.raises(AssertionError, match="did not run upstream"):
            DeepseekV4Attention._sparse_attention(
                layer,
                QKNormRopeOut(),  # every field None -- the broken-gate shape
                torch.zeros(T, dtype=torch.int32),
            )
    finally:
        v4.get_forward_context = saved


def test_qk_norm_rope_shapes_match_what_the_paged_kernel_asserts():
    # `_qkn_placeholder` feeds both the op's fake impl and the dummy_run
    # stand-in, so comparing those two to each other proves nothing -- they were
    # BOTH 448 wide while the real kernel produced 512, and it surfaced as
    # `assert_size_stride` inside a compiled graph on an fp8-KV run (the bf16
    # runs never touched that branch).
    #
    # So check against the CONSUMER's contract instead: the asm path of
    # `sparse_attn_v4_paged_decode` asserts its Q on these exact constants. The
    # 2buff packed width is NOT `head_dim - rope_head_dim` -- that is
    # `V4_DIM_NOPE`; the packed row adds the inline e8m0 scale and padding.
    import types

    import pytest

    try:
        import atom.models.deepseek_v4 as v4
        from atom.model_ops.v4_kernels.v4_quant import (
            V4_DIM_NOPE,
            V4_DIM_QK_PACKED,
            V4_DIM_ROPE,
        )
    except ImportError as e:
        if "aiter" not in str(e) and "forward_context" not in str(e):
            raise
        pytest.skip(f"deepseek_v4 not importable here: {e}")

    assert V4_DIM_QK_PACKED != V4_DIM_NOPE, (
        "the trap this pins is gone -- packed and nope now coincide, so "
        "deriving one from head_dim would no longer be wrong"
    )

    T, H, D, RD = 4, 2, 512, V4_DIM_ROPE
    q = torch.zeros(T, H * D)

    def layer(kv_fp8):
        return types.SimpleNamespace(
            n_local_heads=H, head_dim=D, rope_head_dim=RD, kv_fp8=kv_fp8
        )

    fp8 = v4._qkn_placeholder(layer(True), q, T, zeros=True)
    assert fp8.q_packed.shape == (T, H, V4_DIM_QK_PACKED)
    assert fp8.q_rope.shape == (T, H, V4_DIM_ROPE)
    assert fp8.k_packed.shape == (T, 1, V4_DIM_QK_PACKED)
    assert fp8.k_rope.shape == (T, 1, V4_DIM_ROPE)
    assert fp8.q_sa is None and fp8.kv is None

    bf16 = v4._qkn_placeholder(layer(False), q, T, zeros=True)
    assert bf16.q_sa.shape == (T, H, D) and bf16.kv.shape == (T, D)
    assert bf16.q_packed is None and bf16.q_rope is None

    # And the op's return list has to carry the active layout's four / two.
    assert len(v4._qkn_to_list(fp8, True)) == 4
    assert len(v4._qkn_to_list(bf16, False)) == 2


# ---------------------------------------------------------------------------
# `kv_indices_{swa,csa,hca}` carry no information in their own length
#
# `_attach_v4_paged_decode_meta` publishes them whole, not sliced to
# `indptr_np[T]`. That length is a cumsum of per-token KV spans, so it varies
# step to step at a FIXED num_tokens and no graph key pins it -- and a cudagraph
# bakes it. Neither consumer reads it: the paged kernel walks
# `kv_indices[indptr[t]:indptr[t+1]]`, `csa_translate_pack`'s grid comes from
# `topk_local.shape`. Both are exercised below against an exact and an oversized
# buffer and required to agree.
# ---------------------------------------------------------------------------

ENVELOPE_ROWS = 8
CSA_BLOCK_CAPACITY = 64
WINDOW_SIZE = 128
SLACK = 97  # deliberately not a round number, and not a multiple of index_topk


def _decode_batch(bs: int, tokens_per_seq: int, index_topk: int):
    """A ragged decode batch: per-token spans differ, and a CG pad tail follows.

    `positions` drives `skip = min(pos + 1, WINDOW_SIZE)` inline, so varying
    them across tokens is what makes `valid_k` -- and therefore the exact
    destination length -- data-dependent in the first place.
    """
    g = torch.Generator().manual_seed(bs * 1000 + tokens_per_seq)
    t_real = bs * tokens_per_seq
    t_pad = t_real + 3  # CG padding: batch_id -1, contributes nothing

    batch_id = torch.full((t_pad,), -1, dtype=torch.int32)
    batch_id[:t_real] = torch.repeat_interleave(
        torch.arange(bs, dtype=torch.int32), tokens_per_seq
    )
    positions = torch.zeros(t_pad, dtype=torch.int32)
    positions[:t_real] = torch.randint(
        WINDOW_SIZE, WINDOW_SIZE * 6, (t_real,), generator=g, dtype=torch.int32
    )

    # Slice length per token = skip + valid_k, with valid_k ragged across tokens.
    skip = torch.minimum(
        positions[:t_real].to(torch.int64) + 1, torch.tensor(WINDOW_SIZE)
    )
    valid_k = torch.randint(1, index_topk + 1, (t_real,), generator=g)
    spans = torch.zeros(t_pad, dtype=torch.int64)
    spans[:t_real] = skip + valid_k

    indptr = torch.zeros(t_pad + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(spans, 0).to(torch.int32)

    topk_local = torch.randint(
        0, CSA_BLOCK_CAPACITY * 4, (t_pad, index_topk), generator=g, dtype=torch.int32
    )
    block_tables = torch.randint(
        1, 5000, (max(bs, 1), 16), generator=g, dtype=torch.int32
    )
    return topk_local, block_tables, positions, indptr, batch_id, int(indptr[t_pad])


def _run_writer(dest_len: int, batch) -> torch.Tensor:
    from atom.model_ops.v4_kernels.csa_translate_pack import (
        csa_translate_pack_reference,
    )

    topk_local, block_tables, positions, indptr, batch_id, _ = batch
    dest = torch.full((dest_len,), -7, dtype=torch.int32)
    csa_translate_pack_reference(
        topk_local,
        block_tables,
        positions,
        indptr,
        batch_id,
        None,
        dest,
        envelope_rows=ENVELOPE_ROWS,
        csa_block_capacity=CSA_BLOCK_CAPACITY,
        window_size=WINDOW_SIZE,
    )
    return dest


def test_writer_ignores_the_destination_length():
    batch = _decode_batch(bs=5, tokens_per_seq=6, index_topk=32)
    exact = batch[-1]

    tight = _run_writer(exact, batch)
    loose = _run_writer(exact + SLACK, batch)

    torch.testing.assert_close(loose[:exact], tight)
    assert torch.all(loose[exact:] == -7), (
        "an oversized destination must leave its tail untouched -- a writer that "
        "sized anything off `kv_indices.numel()` would have scribbled into it"
    )


def test_writer_is_length_invariant_across_shapes():
    """The same, over batch shapes whose exact lengths differ widely."""
    for bs, tokens_per_seq, index_topk in [
        (1, 1, 16),
        (3, 4, 32),
        (8, 6, 64),
        (17, 2, 128),
    ]:
        batch = _decode_batch(bs, tokens_per_seq, index_topk)
        exact = batch[-1]
        tight = _run_writer(exact, batch)
        loose = _run_writer(exact + SLACK, batch)
        torch.testing.assert_close(
            loose[:exact],
            tight,
            msg=f"bs={bs} tokens_per_seq={tokens_per_seq} topk={index_topk}",
        )


def test_reader_ignores_the_indices_length():
    """Attention output is identical over an exact vs an oversized `kv_indices`.

    The oversized tail is filled with in-range but WRONG slot ids, so a reader
    that walked past `indptr[T]` would change its answer rather than fault.
    """
    torch.manual_seed(0)
    t, heads, dim, pages = 6, 4, 32, 512

    spans = torch.tensor([3, 1, 7, 0, 4, 2])
    indptr = torch.zeros(t + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(spans, 0).to(torch.int32)
    exact = int(indptr[t])

    q = torch.randn(t, heads, dim)
    unified_kv = torch.randn(pages, dim)
    attn_sink = torch.randn(heads)
    tight = torch.randint(0, pages, (exact,), dtype=torch.int32)
    loose = torch.cat([tight, torch.randint(0, pages, (SLACK,), dtype=torch.int32)])

    from atom.model_ops.v4_kernels.paged_decode import (
        sparse_attn_v4_paged_decode_reference,
    )

    out_tight = sparse_attn_v4_paged_decode_reference(
        q, unified_kv, tight, indptr, attn_sink, dim**-0.5
    )
    out_loose = sparse_attn_v4_paged_decode_reference(
        q, unified_kv, loose, indptr, attn_sink, dim**-0.5
    )
    torch.testing.assert_close(out_tight, out_loose)


def test_a_void_op_survives_only_because_split_ops_are_not_compiled():
    # `v4_attn_compress` returns nothing. That is only safe because a split op's
    # submodule is the one piece the backend leaves uncompiled --
    # `submod_names_to_compile` excludes `is_splitting_graph` -- so it never
    # reaches AOT autograd, which is the layer that drops an effect-free call.
    #
    # Both halves of that are pinned here: the DCE layer, and the exclusion. Get
    # either wrong and the compressor silently stops running, with no error.
    import torch

    from atom.utils.custom_register import direct_register_custom_op

    calls = []

    def _op(x: torch.Tensor) -> None:
        calls.append(1)

    def _fake(x: torch.Tensor) -> None:
        return None

    direct_register_custom_op(
        op_name="_void_probe", op_func=_op, mutates_args=[], fake_impl=_fake
    )

    def f(x):
        torch.ops.aiter._void_probe(x)
        return x * 2

    if not torch.cuda.is_available():
        import pytest

        pytest.skip("the op registers a CUDA kernel")
    x = torch.randn(8, device="cuda")
    seen = {}
    for backend in ("eager", "aot_eager"):
        g = torch.compile(f, backend=backend, dynamic=False)
        g(x)
        calls.clear()
        g(x)
        seen[backend] = len(calls)
        torch._dynamo.reset()
    assert seen["eager"] == 1, "Dynamo alone keeps it; if not, the premise moved"
    assert seen["aot_eager"] == 0, (
        "AOT no longer drops an effect-free op -- then a void op would be safe "
        "anywhere and `v4_attn_compress` need not stay a split op"
    )

    import inspect

    from atom.utils import backends

    src = inspect.getsource(backends)
    assert "if not item.is_splitting_graph" in src, (
        "the backend no longer excludes split-op submodules from compilation, "
        "so `v4_attn_compress` would go through AOT and be DCE'd"
    )


def test_sparse_attention_sizes_on_the_q_not_positions():
    # The attention output feeds the mHC residual stream, which is sized by the
    # hidden state. The Q descends from the hidden state, `positions` does not
    # have to: under `--enable-dp-attention` a step where any rank is prefilling
    # takes the variable-length path and the two counts can differ. Sizing on
    # `positions` then surfaced as an aiter `residual_in shape mismatch` deep
    # inside a compiled piece -- expected 6, got 5 -- with nothing near the
    # cause. Both the body and the fake are pinned, since disagreeing is its own
    # class of failure.
    import types

    import pytest

    try:
        import atom.models.deepseek_v4 as v4
    except ImportError as e:
        if "aiter" not in str(e) and "forward_context" not in str(e):
            raise
        pytest.skip(f"deepseek_v4 not importable here: {e}")

    T_Q, T_POS, H, D = 5, 6, 2, 64  # deliberately different
    layer = types.SimpleNamespace(
        n_local_heads=H, head_dim=D, rope_head_dim=16, kv_fp8=False
    )
    qkn = v4.QKNormRopeOut(q_sa=torch.zeros(T_Q, H, D), kv=torch.zeros(T_Q, D))
    positions = torch.zeros(T_POS, dtype=torch.int32)

    cfg = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(static_forward_context={"L": layer})
    )
    fc = types.SimpleNamespace(context=types.SimpleNamespace(is_dummy_run=True))
    saved_cfg, saved_fc = v4.get_current_atom_config, v4.get_forward_context
    v4.get_current_atom_config, v4.get_forward_context = (lambda: cfg), (lambda: fc)
    try:
        body = v4.DeepseekV4Attention._sparse_attention(layer, qkn, positions)
        fake = v4._v4_sparse_attention_fake(
            qkn.q_sa, qkn.kv, None, None, None, None, positions, None, None, None, "L"
        )
    finally:
        v4.get_current_atom_config, v4.get_forward_context = saved_cfg, saved_fc

    assert body.shape == (T_Q, H * D), f"body followed positions: {body.shape}"
    assert fake.shape == (T_Q, H * D), f"fake followed positions: {fake.shape}"
