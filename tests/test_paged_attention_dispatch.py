# SPDX-License-Identifier: MIT
"""Envelopes of the two paged decode kernels.

Launch policy only -- `gluon_decode_over_limit` is integer arithmetic and needs
no device. It still needs triton and aiter to be importable, because it lives
beside the kernel wrappers it describes and `atom.model_ops.base_attention`
pulls both at import. The CPU CI runner installs neither (`pre-checks.yaml`
installs cpu torch and pytest), so this file self-skips there, the same way
every other attention test in this directory does. Its coverage comes from a
GPU environment.

What is worth asserting here is the coupling to aiter, since nothing else
watches it: the gluon kernel picks its register layout from a table keyed on
next_pow2(query_group_size), and past the last arm the variable is simply never
bound -- a Triton compile error with nothing in it about speculative length.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

pytest.importorskip("triton", reason="base_attention defines @triton.jit kernels")
pytest.importorskip("aiter", reason="base_attention imports the AITER runtime")
# The wiring tests monkeypatch "aiter.ops.flydsl.pa_decode.plan_pa_decode" by
# string, which pytest resolves by importing the module -- raising=False covers
# a missing attribute, not a missing module. Without #4332 those tests ERROR
# instead of skipping.
# The gate compares against aiter's own fp8 alias, which differs by arch
# (e4m3fnuz on gfx942, e4m3fn on gfx950); take it from aiter, never hardcode.
import aiter as _aiter

_FP8 = _aiter.dtypes.fp8


def _fake_ctx(n):
    """A stand-in for context_lens carrying every field the builder inspects.

    Deliberately not a bare SimpleNamespace(shape=...): refresh_flydsl_plan also
    checks dtype and device, mirroring what plan_pa_decode rejects, and a fake
    that is missing those stops exercising the guard it is meant to pass.
    """
    import torch

    return SimpleNamespace(
        shape=(n,),
        device=SimpleNamespace(index=0),
        dtype=torch.int32,
        is_cuda=True,
        ndim=1,
        is_contiguous=lambda: True,
    )


try:
    _HAS_FLYDSL_PA = importlib.util.find_spec("aiter.ops.flydsl.pa_decode") is not None
except ModuleNotFoundError:
    # find_spec imports the parent package first; a missing one raises rather
    # than returning None, which would make this whole file a collection error.
    _HAS_FLYDSL_PA = False
try:
    import torch as _torch

    _HAS_CUDA = _torch.cuda.is_available()
except (ImportError, RuntimeError):
    # torch may be absent, or present against a driver that fails to initialise.
    _HAS_CUDA = False

from aiter.ops.triton.gluon import pa_decode_gluon

from atom.model_ops.base_attention import (
    PA_ASM_MAX_QUERY_GROUP_SIZE,
    PA_DENSE_SPLIT_MAX,
    PA_DENSE_SPLIT_TARGET_WG,
    PA_GLUON_MAX_QUERY_GROUP_SIZE,
    PA_GLUON_MAX_QUERY_LEN,
    dense_decode_splits,
    gluon_decode_over_limit,
)

# aiter pa_decode_gluon.py:134-168 -- the arms `register_bases` is defined for,
# plus a separate path below 16. There is no 128 arm and no else.
AITER_GLUON_GROUP_ARMS = (16, 32, 64)

# aiter pa_ps.py:69 -- the C++ PS reduce is built for 1..64 partitions. Past it
# the launcher falls to flydsl, whose module-level wrapper takes no `stream`
# argument, and the TypeError that raises is not caught by the `except
# ImportError` that would otherwise reach the Triton kernel. So there is no
# fallback: decode aborts.
AITER_PS_REDUCE_MAX_PARTITIONS = 64


class TestGluonEnvelope:
    """Shapes the gluon decode kernel takes, and the ones it cannot."""

    @pytest.mark.parametrize(
        "max_qlen, num_heads, num_kv_heads, over, why",
        [
            (1, 16, 1, False, "M3 dense at tp4, no drafting"),
            (
                4,
                16,
                1,
                False,
                "M3 dense at tp4 with 3 draft tokens: 16*4=64, on the limit",
            ),
            (5, 16, 1, True, "one more draft token: past both limits at once"),
            (4, 32, 2, False, "M3 dense at tp2 -- ratio is still 16"),
            (
                5,
                8,
                1,
                True,
                "gqa=8 reaches the query-length limit before the group one",
            ),
            (
                3,
                17,
                1,
                True,
                "ratio 17 rounds to 32, 3 rounds to 4: 128, no arm for it",
            ),
            (2, 64, 1, True, "ratio 64 doubled by two query positions"),
        ],
    )
    def test_known_shapes(self, max_qlen, num_heads, num_kv_heads, over, why):
        assert gluon_decode_over_limit(max_qlen, num_heads, num_kv_heads) is over, why

    def test_rounds_up_rather_than_using_the_raw_product(self):
        """17 heads over 1 kv head at qlen 3 is 51 -- under the raw 64 limit, but
        the kernel indexes its table with next_pow2, and 128 has no arm."""
        assert 3 * (17 // 1) <= PA_GLUON_MAX_QUERY_GROUP_SIZE
        assert gluon_decode_over_limit(3, 17, 1) is True

    # 0 and -1 pass with or without the clamp -- they are boundary shapes, not
    # evidence. -5 and -100 are: unclamped their bit_length alone synthesises a
    # 128- and 2048-wide group out of what is really one position.
    @pytest.mark.parametrize("max_qlen", [0, -1, -5, -100])
    def test_non_positive_query_length_is_clamped(self, max_qlen):
        """A sentinel or unset length must not synthesise a large group."""
        assert gluon_decode_over_limit(max_qlen, 16, 1) is False
        assert gluon_decode_over_limit(max_qlen, 16, 1) == gluon_decode_over_limit(
            1, 16, 1
        )

    def test_monotone_in_query_length(self):
        """Once a shape is past the envelope, longer cannot bring it back."""
        seen_over = False
        for qlen in range(1, 12):
            over = gluon_decode_over_limit(qlen, 16, 1)
            assert not (seen_over and not over), f"qlen={qlen} came back under"
            seen_over |= over
        assert seen_over, "16 heads should leave the envelope within 12 positions"


class TestEnvelopeConstants:
    """The constants against the kernel sources they were read from."""

    def test_gluon_group_limit_is_the_last_layout_arm(self):
        assert PA_GLUON_MAX_QUERY_GROUP_SIZE == max(AITER_GLUON_GROUP_ARMS)

    def test_the_split_ceiling_stays_inside_what_the_ps_reduce_was_built_for(self):
        """The bound the rule is written against, not the one it declares.

        Every other test here reads its limit off PA_DENSE_SPLIT_MAX, so raising
        that constant past what aiter serves leaves them all green while decode
        aborts. This is the one that goes red.
        """
        assert PA_DENSE_SPLIT_MAX <= AITER_PS_REDUCE_MAX_PARTITIONS

    def test_the_split_constants_are_positive(self):
        """`1 << (x.bit_length() - 1)` raises on 0, and both are tuning knobs."""
        assert PA_DENSE_SPLIT_TARGET_WG >= 1
        assert PA_DENSE_SPLIT_MAX >= 1

    def test_every_reachable_group_has_an_arm(self):
        """Anything reported as safe must land on an arm, not between two."""
        for qlen in range(1, PA_GLUON_MAX_QUERY_LEN + 1):
            for ratio in (1, 2, 4, 8, 16, 32, 64):
                if gluon_decode_over_limit(qlen, ratio, 1):
                    continue
                qlen_p2 = 1 << (qlen - 1).bit_length()
                group_p2 = qlen_p2 * max(16 // qlen_p2, 1 << (ratio - 1).bit_length())
                assert group_p2 in AITER_GLUON_GROUP_ARMS, (
                    f"qlen={qlen} ratio={ratio} passes as safe but needs a "
                    f"{group_p2}-wide layout, which aiter does not define"
                )

    def test_asm_envelope_is_inside_gluon(self):
        """ASM tops out lower, so a shape it declines still has somewhere to go.

        asm_pa.cu:113-116 carries `# mtp * gqa <= 16` as a source comment.
        """
        assert PA_ASM_MAX_QUERY_GROUP_SIZE < PA_GLUON_MAX_QUERY_GROUP_SIZE


class TestDenseDecodeSplits:
    """How finely the dense decode splits the KV, and what bounds it.

    Integer arithmetic, no device. `dense_decode_splits` imports the aiter
    heuristic inside its body, so monkeypatching the module attribute reaches it
    -- which is what lets the cases aiter cannot produce today be tested at all.
    """

    def test_batch_one_asks_for_the_ceiling(self):
        """Reverting to the bare heuristic returns 8 and turns this red."""
        assert dense_decode_splits(1, 1) == PA_DENSE_SPLIT_MAX

    def test_a_full_grid_is_left_to_the_heuristic(self, monkeypatch):
        """Past TARGET_WG the added term is 1, so the heuristic must win outright.

        Red if max() becomes min(), or if TARGET_WG grows past the grid.
        """
        monkeypatch.setattr(
            pa_decode_gluon, "get_recommended_splits", lambda seqs, heads: 3
        )
        assert dense_decode_splits(128, 4) == 3

    def test_never_below_the_heuristic(self, monkeypatch):
        """Guards the direction: this is what makes the change unable to regress."""
        monkeypatch.setattr(
            pa_decode_gluon, "get_recommended_splits", lambda seqs, heads: 8
        )
        for num_seqs in (1, 2, 4, 8, 16, 64, 256):
            assert dense_decode_splits(num_seqs, 1) >= 8

    @pytest.mark.parametrize("num_seqs", range(1, 40))
    def test_only_ever_asks_for_a_power_of_two_above_the_heuristic(
        self, num_seqs, monkeypatch
    ):
        """Every split count past the heuristic's own range is a power of two.

        The C++ PS reduce compiles one variant per distinct count, so a
        continuous cdiv would add ~11 of them, each a first-use hipcc on the
        request path under eager decode. Red if the rounding is dropped: n=5
        would ask for 26.
        """
        monkeypatch.setattr(
            pa_decode_gluon, "get_recommended_splits", lambda seqs, heads: 1
        )
        s = dense_decode_splits(num_seqs, 1)
        assert s & (s - 1) == 0, f"n={num_seqs} asked for {s}"

    def test_stays_inside_the_ps_reduce_contract(self, monkeypatch):
        """The C++ PS reduce is built for 1..64 and there is no usable fallback.

        Past it `launch_pa_decode_ps_reduce_flydsl` is called with a `stream`
        kwarg its module-level signature does not take, and the resulting
        TypeError is not caught by the `except ImportError` that would otherwise
        reach the Triton kernel. So the ceiling has to clamp the result, not just
        the term this function adds -- red if it moves back inside the max().
        """
        monkeypatch.setattr(
            pa_decode_gluon, "get_recommended_splits", lambda seqs, heads: 128
        )
        assert 1 <= dense_decode_splits(1, 1) <= PA_DENSE_SPLIT_MAX

    def test_kv_heads_count_toward_the_grid(self, monkeypatch):
        """The grid is (num_seqs, num_kv_heads, splits), so both dims fill it.

        Red if num_kv_heads is dropped from the product: 4 x 4 would then be read
        as 4 and ask for TARGET_WG // 4 instead of // 16.
        """
        monkeypatch.setattr(
            pa_decode_gluon, "get_recommended_splits", lambda seqs, heads: 1
        )
        assert dense_decode_splits(4, 4) == dense_decode_splits(16, 1)
        assert dense_decode_splits(4, 4) == PA_DENSE_SPLIT_TARGET_WG // 16


class _Layer:
    """Just the attributes _dispatch_decode reads.

    It touches six of them plus two env flags and returns a bound method, so the
    routing table can be driven without a device -- which the envelope tests
    above cannot do, and which is the gap a sliding-window regression slipped
    through once already.
    """

    def __init__(self, sliding_window=-1, num_heads=16, num_kv_heads=1, **flags):
        self.sliding_window = sliding_window
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.use_triton_attn = flags.get("use_triton_attn", False)
        self.use_flash_layout = flags.get("use_flash_layout", False)
        for name in (
            "paged_attention_unified",
            "paged_attention_triton",
            "paged_attention_asm",
            "paged_attention_persistent_asm",
        ):
            setattr(self, name, name)


def _route(monkeypatch, max_qlen, block_size=128, unified=False, force=False, **kw):
    from atom.model_ops import attention_mha as mha

    monkeypatch.setattr(mha.envs, "ATOM_USE_UNIFIED_ATTN", unified)
    monkeypatch.setattr(mha.envs, "ATOM_FORCE_ATTN_TRITON", force)
    monkeypatch.setattr(
        mha,
        "get_current_atom_config",
        lambda: SimpleNamespace(kv_cache_block_size=block_size),
    )
    return mha.PagedAttentionImpl._dispatch_decode(_Layer(**kw), max_qlen)


class TestDecodeRouting:
    """Which backend _dispatch_decode picks, for the shapes that reach it."""

    def test_m3_dense_production_shape_stays_on_gluon(self, monkeypatch):
        """TP4, 3 draft tokens. The shape this change is measured on."""
        assert _route(monkeypatch, 4) == "paged_attention_triton"

    def test_no_drafting_still_reaches_asm(self, monkeypatch):
        assert _route(monkeypatch, 1) == "paged_attention_asm"

    def test_past_gluon_falls_back_to_unified(self, monkeypatch):
        assert _route(monkeypatch, 5) == "paged_attention_unified"

    def test_past_asm_but_within_gluon_takes_gluon(self, monkeypatch):
        """4 x 16 = 64 clears gluon and is four times ASM's envelope.

        Without the check it would reach run_pa_fwd_asm, where an unmatched mtp
        silently re-runs with mtp=1 rather than refusing.
        """
        assert _route(monkeypatch, 4) == "paged_attention_triton"
        assert _route(monkeypatch, 1, num_heads=64) == "paged_attention_triton"

    @pytest.mark.parametrize("max_qlen", [1, 4])
    def test_sliding_window_honours_unified_env(self, monkeypatch, max_qlen):
        """The env has to reach the sliding-window branch too.

        It returns before the ATOM_USE_UNIFIED_ATTN block below it, so dropping
        the flag from this one expression silently moves sliding-window layers
        onto a kernel whose output dtype the caller has already fixed as fp8.
        """
        assert (
            _route(monkeypatch, max_qlen, sliding_window=128, unified=True)
            == "paged_attention_unified"
        )

    def test_sliding_window_without_the_env_uses_gluon(self, monkeypatch):
        assert _route(monkeypatch, 4, sliding_window=128) == "paged_attention_triton"

    def test_flash_layout_routes_to_unified(self, monkeypatch):
        assert (
            _route(monkeypatch, 1, use_flash_layout=True) == "paged_attention_unified"
        )

    def test_force_triton_takes_unified(self, monkeypatch):
        """ATOM_FORCE_ATTN_TRITON short-circuits the block-256 ASM route."""
        assert (
            _route(monkeypatch, 1, block_size=256, unified=True, force=True)
            == "paged_attention_unified"
        )

    def test_use_triton_attn_diverts_off_asm(self, monkeypatch):
        """Same shape reaches ASM without the flag, so this arm is load-bearing."""
        assert _route(monkeypatch, 1) == "paged_attention_asm"
        assert _route(monkeypatch, 1, use_triton_attn=True) == "paged_attention_triton"

    def test_a_sentinel_query_length_routes_as_one(self, monkeypatch):
        """Clamped at the top, so both gates see the same value.

        Unclamped, `0 * ratio > 16` is false and this would reach ASM instead.
        """
        assert _route(monkeypatch, 0, num_heads=64) == _route(
            monkeypatch, 1, num_heads=64
        )

    def test_persistent_asm_is_not_bounded_by_the_run_pa_fwd_envelope(
        self, monkeypatch
    ):
        """pa_persistent_fwd is a different kernel with its own table.

        4 x 16 = 64 is past run_pa_fwd_asm's 16, but that says nothing about
        the persistent path, so the block-256 route must still be taken.
        """
        assert (
            _route(monkeypatch, 4, block_size=256, unified=True)
            == "paged_attention_persistent_asm"
        )


class _FakePlan:
    """Just the fields the op and the scratch helper read off a real plan."""

    def __init__(self, capacity=512, max_partitions=256, num_kv_heads=1):
        self.capacity = capacity
        self.max_partitions = max_partitions
        self.num_kv_heads = num_kv_heads


@pytest.mark.skipif(not _HAS_FLYDSL_PA, reason="needs aiter #4332 (FlyDSL pa_decode)")
class TestWorkPlanWiring:
    """How aiter #5546's planner is wired in, not what it computes.

    The numerics are aiter's own op_tests' job. What nothing else watches is the
    wiring, and every case below is one this tree got wrong once: the ceiling
    taken from the static split count, which switched the planner off in all but
    name while still paying for it; and the planner reaching the sparse call
    sites, where the context is a fixed topk window and there is no unevenness
    to rebalance.
    """

    def test_ceiling_is_left_at_the_aiter_default(self, monkeypatch):
        """`max_partitions` must not be passed at all.

        Red the moment anyone routes the static split count -- or any other
        value -- into the plan's ceiling. `get_recommended_splits` hands every
        request the same count and documents itself as "not a variable-work
        scheduler"; the plan's ceiling is an upper bound the planner divides
        under a workgroup budget. Feeding one into the other clamps the long
        request to the short requests' share. Every number measured on this
        branch was measured at the default, so changing it is a perf claim
        nothing here backs.
        """
        from atom.model_ops.attentions import aiter_attention as aa

        seen = {}

        def fake_plan(context_lens, num_kv_heads, **kwargs):
            seen.update(kwargs)
            return _FakePlan()

        monkeypatch.setattr("aiter.ops.flydsl.pa_decode.plan_pa_decode", fake_plan)
        builder = aa.AiterAttentionMetadataBuilder.__new__(
            aa.AiterAttentionMetadataBuilder
        )
        builder._flydsl_kv_heads = 1
        builder._flydsl_plans = {}
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL", True)
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL_PLAN", True)

        assert builder.refresh_flydsl_plan(_fake_ctx(8), create=True) is not None
        assert "max_partitions" not in seen, f"ceiling was set: {seen}"

    def test_planner_off_returns_no_plan(self, monkeypatch):
        """With the env off the op must see None, not a stale plan."""
        from atom.model_ops.attentions import aiter_attention as aa

        builder = aa.AiterAttentionMetadataBuilder.__new__(
            aa.AiterAttentionMetadataBuilder
        )
        builder._flydsl_kv_heads = 1
        builder._flydsl_plans = {}
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL", True)
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL_PLAN", False)
        ctx = _fake_ctx(8)
        assert builder.refresh_flydsl_plan(ctx) is None

    def test_batch_past_the_planner_limit_falls_back(self, monkeypatch):
        """M3's sparse prefill-as-decode folds query tokens into num_seqs.

        It reaches 32768, past what plan_pa_decode accepts. Letting that raise
        killed a worker 90 s into a run while the server kept answering
        /metrics, so the client sat in warmup until it timed out.
        """
        from atom.model_ops.attentions import aiter_attention as aa
        from atom.model_ops.base_attention import _FLYDSL_PLAN_MAX_BATCH

        builder = aa.AiterAttentionMetadataBuilder.__new__(
            aa.AiterAttentionMetadataBuilder
        )
        builder._flydsl_kv_heads = 1
        builder._flydsl_plans = {}
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL", True)
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL_PLAN", True)
        ctx = SimpleNamespace(
            shape=(_FLYDSL_PLAN_MAX_BATCH + 1,), device=SimpleNamespace(index=0)
        )
        assert builder.refresh_flydsl_plan(ctx) is None

    def test_only_the_dense_call_site_passes_a_plan(self):
        """Which call sites hand in a plan IS the boundary.

        There is no flag any more: the parameter defaults to None, so a site
        that does not pass one gets the static path. The two MiniMax-M3 sparse
        sites read a fixed topk window (nothing to rebalance, and measured a net
        loss), and the vLLM/SGLang bridges run under someone else's forward
        context entirely.
        """
        import inspect

        from atom.model_ops import attention_mha
        from atom.model_ops.base_attention import run_pa_decode
        from atom.model_ops.minimax_m3 import sparse_attn

        param = inspect.signature(run_pa_decode).parameters["work_plan"]
        assert param.default is None, "a plan must be opt-in, per call site"
        assert "work_plan=" in inspect.getsource(attention_mha)
        assert "work_plan=" not in inspect.getsource(sparse_attn)
        # Read the bridges as text, never import them: they pull in vllm /
        # sglang, which a CPU-only checkout does not have, and the claim being
        # tested is about the source anyway.
        import pathlib

        import atom

        root = pathlib.Path(atom.__file__).parent
        for rel in (
            "plugin/vllm/attention/layer_mha.py",
            "plugin/sglang/attention_backend/full_attention/full_attention_backend.py",
        ):
            f = root / rel
            assert f.is_file(), f"{rel} moved; this boundary is no longer checked"
            assert "work_plan=" not in f.read_text(), rel

    def test_every_draft_pass_refreshes_the_plan(self):
        """Weak on purpose, and the weakness is the point.

        Each draft pass advances context_lens by a token, so reusing the
        target's plan points the kernel at KV ranges that no longer match --
        wrong output, not merely slower. Driving prepare_mtp_decode for real
        needs a model runner, so this only pins that the call is there; if it
        ever needs to be stronger, that is the cost.
        """
        import inspect

        from atom.model_ops.attentions import aiter_attention as aa

        src = inspect.getsource(aa.AiterAttentionMetadataBuilder.prepare_mtp_decode)
        assert "refresh_flydsl_plan" in src

    @pytest.mark.skipif(not _HAS_CUDA, reason="allocates real scratch buffers")
    def test_scratch_is_per_plan_not_per_shape(self):
        """Two plans must never share buffers, however alike their shapes.

        capacity is a constant under the workgroup budget at kv_heads=1, so a
        shape-only key hands every capture rung -- and, under TBO, two
        concurrent ubatches -- the same three tensors to write at once. Wrong
        logits, no error, and it vanishes under HIP_LAUNCH_BLOCKING.
        """
        import torch

        from atom.model_ops.base_attention import (
            _FLYDSL_PLAN_SCRATCH,
            _flydsl_plan_scratch,
        )

        dev = torch.device("cuda", 0)
        before = dict(_FLYDSL_PLAN_SCRATCH)
        try:
            plan = _FakePlan(capacity=512)
            first = _flydsl_plan_scratch(plan, 4, 16, 128, torch.bfloat16, dev)
            again = _flydsl_plan_scratch(plan, 4, 16, 128, torch.bfloat16, dev)
            assert first[0] is again[0], "the same plan must reuse its buffers"

            twin = _FakePlan(capacity=512)
            other = _flydsl_plan_scratch(twin, 4, 16, 128, torch.bfloat16, dev)
            assert (
                other[0] is not first[0]
            ), "a second plan of identical shape must get its own buffers"

            big = _flydsl_plan_scratch(
                _FakePlan(capacity=1024), 4, 16, 128, torch.bfloat16, dev
            )
            assert big[0].shape[1] == 1024
        finally:
            # The cache is a module global with no eviction; leaving ~25 MB of
            # live CUDA buffers behind would charge every later test for it.
            _FLYDSL_PLAN_SCRATCH.clear()
            _FLYDSL_PLAN_SCRATCH.update(before)

    def test_one_plan_per_batch_and_never_replaced(self, monkeypatch):
        """Decode replays captured graphs, one per capture-ladder size.

        A single plan slot would be rebuilt every time the batch moved to
        another rung, leaving the graph captured for the previous rung pointing
        at freed tensors. Red if the dict goes back to one slot: `first` would
        come back a different object after the batch changed and returned.
        """
        from atom.model_ops.attentions import aiter_attention as aa

        def fake_plan(context_lens, num_kv_heads, **kw):
            return kw.get("plan") or _FakePlan()

        monkeypatch.setattr(
            "aiter.ops.flydsl.pa_decode.plan_pa_decode", fake_plan, raising=False
        )
        builder = aa.AiterAttentionMetadataBuilder.__new__(
            aa.AiterAttentionMetadataBuilder
        )
        builder._flydsl_kv_heads = 1
        builder._flydsl_plans = {}
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL", True)
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL_PLAN", True)

        ctx = _fake_ctx

        first = builder.refresh_flydsl_plan(ctx(8), create=True)
        assert builder.refresh_flydsl_plan(ctx(8)) is first, "same rung must reuse"
        other = builder.refresh_flydsl_plan(ctx(16), create=True)
        assert other is not first, "a different rung needs its own plan"
        assert (
            builder.refresh_flydsl_plan(ctx(8)) is first
        ), "returning to a rung must hand back the plan its graph captured"

    def test_a_plan_built_for_another_batch_is_refused(self):
        """The guard that keeps a shape mismatch from killing the worker.

        aiter validates `reduce_info.shape == (num_seqs, 2)` and raises. The
        plan is built by the metadata builder for the batch it saw, which is
        not necessarily the one this call runs, so the op checks first and
        falls back to the static path. Red if the guard goes away.
        """
        from atom.model_ops.base_attention import flydsl_plan_matches

        plan = _FakePlan()
        plan.reduce_info = SimpleNamespace(shape=(32, 2))
        assert flydsl_plan_matches(plan, 32, 1)
        assert not flydsl_plan_matches(plan, 20, 1), "batch mismatch must refuse"
        assert not flydsl_plan_matches(plan, 32, 2), "kv-head mismatch must refuse"

    def test_capture_builder_attaches_a_plan(self):
        """Weak on purpose: a source check, and the reason it is here.

        Decode runs from captured graphs. If the plan is absent when the graph
        is captured, the static path is what gets recorded and every later
        refresh is work on a graph that never reads it -- silently, with no
        error and a plausible-looking benchmark. Driving the real capture needs
        a model runner, so this only pins that the attach is present.
        """
        import inspect

        from atom.model_ops.attentions import aiter_attention as aa

        src = inspect.getsource(
            aa.AiterAttentionMetadataBuilder.build_for_cudagraph_capture
        )
        assert "flydsl_work_plan" in src

    def test_only_the_capture_builders_mint_a_plan(self):
        """The value of `create`, at every site, in every builder.

        Six calls refresh a plan and two mint one; which is which is the whole
        contract. A sed that moved `create=True` onto a runtime path once left
        every test here green, and so did deleting it from the GDN builder --
        one check read the keyword without its value, the other read one file.
        So walk the AST of all three.
        """
        import ast
        import pathlib

        import atom

        root = pathlib.Path(atom.__file__).parent / "model_ops" / "attentions"
        sites = []

        def visit(node, where):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                where = node.name
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "refresh_flydsl_plan"
            ):
                create = {k.arg: k.value for k in node.keywords}.get("create")
                sites.append((f, where, getattr(create, "value", None) is True))
            for child in ast.iter_child_nodes(node):
                visit(child, where)

        for f in ("aiter_attention.py", "gdn_attn.py", "qwen4_exp_attn.py"):
            visit(ast.parse((root / f).read_text()), None)

        assert sites, "nothing calls refresh_flydsl_plan; this test moved"
        minting = {(f, fn) for f, fn, mints in sites if mints}
        assert minting == {
            ("aiter_attention.py", "build_for_cudagraph_capture"),
            ("gdn_attn.py", "build_for_cudagraph_capture"),
        }, f"plans may only be minted during capture, got {sorted(minting)}"
        assert len(sites) - len(minting) >= 4, "a runtime refresh site went missing"

    def test_plans_are_only_minted_during_capture(self, monkeypatch):
        """A runtime batch that was never captured gets the static path.

        aiter's planner takes batch as a tl.constexpr, so a new value costs a
        kernel specialization -- 65-72 ms cold -- and a plan that can never be
        freed, since some captured graph may have baked its pointers in.
        Minting one mid-serving would be a stall for a batch no graph replays.
        """
        from atom.model_ops.attentions import aiter_attention as aa

        built = []

        def fake_plan(context_lens, num_kv_heads, **kw):
            built.append(context_lens.shape[0])
            return _FakePlan()

        monkeypatch.setattr("aiter.ops.flydsl.pa_decode.plan_pa_decode", fake_plan)
        builder = aa.AiterAttentionMetadataBuilder.__new__(
            aa.AiterAttentionMetadataBuilder
        )
        builder._flydsl_kv_heads = 1
        builder._flydsl_plans = {}
        builder._flydsl_plan_unplanned = False
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL", True)
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL_PLAN", True)

        assert (
            builder.refresh_flydsl_plan(_fake_ctx(31)) is None
        ), "an uncaptured batch must not mint a plan"
        assert built == [], "no planner call may happen outside capture"
        assert builder.refresh_flydsl_plan(_fake_ctx(31), create=True) is not None
        assert built == [31]
        assert (
            builder.refresh_flydsl_plan(_fake_ctx(31)) is not None
        ), "once captured, replay refreshes it without create="

    def test_the_unplanned_notice_waits_for_a_capture(self, monkeypatch):
        """One shot, spent on the batch that means something.

        Before the first capture every call lands on the static path -- profile
        run, eager warmup -- so logging there burns the notice on a step that
        says nothing, and the batch that really is uncaptured mid-serving then
        goes unreported.
        """
        from atom.model_ops.attentions import aiter_attention as aa

        monkeypatch.setattr(
            "aiter.ops.flydsl.pa_decode.plan_pa_decode",
            lambda *a, **kw: _FakePlan(),
        )
        builder = aa.AiterAttentionMetadataBuilder.__new__(
            aa.AiterAttentionMetadataBuilder
        )
        builder._flydsl_kv_heads = 1
        builder._flydsl_plans = {}
        builder._flydsl_plan_unplanned = False
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL", True)
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL_PLAN", True)

        builder.refresh_flydsl_plan(_fake_ctx(31))
        assert not builder._flydsl_plan_unplanned, "nothing captured yet; stay quiet"
        builder.refresh_flydsl_plan(_fake_ctx(8), create=True)
        builder.refresh_flydsl_plan(_fake_ctx(31))
        assert builder._flydsl_plan_unplanned, "an uncaptured batch after capture"

    def test_plan_is_built_for_the_row_count_the_op_derives(self):
        """The row count the builder plans for must be the one the op passes.

        The op derives ``num_seqs`` as ``q.shape[0] // max_seqlen_q`` -- that is
        ``running_bs``, the batch rounded up to a cudagraph capture size -- and
        aiter validates ``reduce_info.shape == (num_seqs, 2)``. A
        ``scheduled_bs`` slice agrees only when the batch happens to land on a
        ladder rung; at c20 (20 -> 32) it raises and kills the worker. This tree
        shipped that slice once, and nothing else watches this seam: the shape
        guard in the op downgrades a mismatch to the static path, so the defect
        would come back as a silent slowdown instead of a crash.

        AST rather than behaviour, because ``prepare_decode`` needs a model
        runner to drive. It is exact about the one thing that went wrong -- the
        expression handed to ``refresh_flydsl_plan`` -- and goes red on any
        slice at the two target sites, or on a draft slice that stops naming
        ``running_bs``.
        """
        import ast
        import inspect

        from atom.model_ops.attentions import aiter_attention as aa

        tree = ast.parse(inspect.getsource(aa))
        args = {}
        calls_by_fn = {}
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "refresh_flydsl_plan"
                ):
                    args.setdefault(fn.name, []).append(node.args[0])
                    calls_by_fn.setdefault(fn.name, []).append(node)

        # Who may mint a plan, and who may only refresh one. Minting is a
        # kernel specialization aiter caches per batch value; doing it at
        # runtime stalls a step for a batch no captured graph will replay, and
        # NOT doing it at capture time means no plan is ever created and the
        # planner is silently off. Neither shows up in any behavioural test --
        # driving either path needs a model runner.
        creates = {
            fn: any(
                any(kw.arg == "create" for kw in call.keywords)
                for call in calls_by_fn[fn]
            )
            for fn in calls_by_fn
        }
        assert creates.get(
            "build_for_cudagraph_capture"
        ), "capture must pass create=; without it no plan is ever minted"
        assert not creates.get("prepare_decode"), (
            "replay must not mint plans: that is a mid-serving kernel "
            "specialization for a batch no graph replays"
        )
        assert not creates.get("prepare_mtp_decode"), "same, for the draft pass"

        # Target passes: the whole buffer, which is already running_bs long
        # with the padded tail zeroed. A zero-length row gets no work.
        for name in ("prepare_decode", "build_for_cudagraph_capture"):
            assert name in args, f"{name} no longer builds a plan"
            for arg in args[name]:
                assert isinstance(arg, ast.Attribute) and arg.attr == "context_lens", (
                    f"{name} must hand refresh_flydsl_plan the whole "
                    f"context_lens, got {ast.dump(arg)}"
                )

        # Draft pass: its own buffer, sliced to running_bs -- never scheduled_bs.
        assert "prepare_mtp_decode" in args, "draft no longer refreshes the plan"
        for arg in args["prepare_mtp_decode"]:
            assert isinstance(
                arg, ast.Subscript
            ), f"draft plan must be sliced to running_bs, got {ast.dump(arg)}"
            upper = getattr(arg.slice, "upper", None)
            assert (
                isinstance(upper, ast.Name) and upper.id == "running_bs"
            ), f"draft plan must be sliced to running_bs, got {ast.dump(arg)}"

    def test_gluon_is_the_default_and_the_env_short_circuits(self, monkeypatch):
        """The env is the first gate, and off is the default.

        #4332 is unmerged and the kernel is not fully tested, so gluon ships and
        FlyDSL is the opt-in. Two things regress independently: someone flips
        the default, or someone drops the env from the dispatch -- which is how
        this tree ran with FlyDSL as the only decode backend and no way back.

        The second half is a source check: reaching the dispatch needs real
        tensors on a GPU. It is exact about the one thing that matters, that the
        env is consulted before the capability check and short-circuits it.
        """
        import inspect

        import atom.model_ops.base_attention as ba
        from atom.utils import envs

        # delenv, not os.environ.pop: pop does not restore, and every later
        # test in the session would then see whatever this one left behind.
        monkeypatch.delenv("ATOM_PA_FLYDSL", raising=False)
        assert envs.ATOM_PA_FLYDSL is False, "FlyDSL must be opt-in"

        src = inspect.getsource(ba.run_pa_decode)
        assert (
            "envs.ATOM_PA_FLYDSL and _flydsl_pa_decode_num_seqs" in src
        ), "the env gate is gone, or no longer short-circuits the capability check"

    def test_lengths_aiter_would_reject_build_no_plan(self, monkeypatch):
        """int64 or host lengths must fall back, not raise from plan_pa_decode.

        This builder serves several models; one of them handing int64 lengths
        would raise on the first decode step of a run that looked fine at
        startup.
        """
        import torch

        from atom.model_ops.attentions import aiter_attention as aa

        builder = aa.AiterAttentionMetadataBuilder.__new__(
            aa.AiterAttentionMetadataBuilder
        )
        builder._flydsl_kv_heads = 1
        builder._flydsl_plans = {}
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL", True)
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL_PLAN", True)

        for field, bad in (
            ("dtype", torch.int64),
            ("is_cuda", False),
            ("ndim", 2),
            ("is_contiguous", lambda: False),
        ):
            ctx = _fake_ctx(8)
            setattr(ctx, field, bad)
            assert builder.refresh_flydsl_plan(ctx) is None, f"{field} must refuse"
        assert not builder._flydsl_plans

    def test_planner_needs_flydsl(self, monkeypatch):
        """With FlyDSL off the builder must not build a plan either.

        The plan only feeds the FlyDSL kernel. Building one anyway is a refresh
        kernel every step that nothing reads, and it would make the "flydsl work
        plan" log line -- the evidence an A/B arm is checked with -- appear on a
        run that is entirely gluon.
        """
        from atom.model_ops.attentions import aiter_attention as aa

        builder = aa.AiterAttentionMetadataBuilder.__new__(
            aa.AiterAttentionMetadataBuilder
        )
        builder._flydsl_kv_heads = 1
        builder._flydsl_plans = {}
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL", False)
        monkeypatch.setattr(aa.envs, "ATOM_PA_FLYDSL_PLAN", True)

        ctx = _fake_ctx(8)
        assert builder.refresh_flydsl_plan(ctx) is None
        assert not builder._flydsl_plans, "a plan was built with FlyDSL off"


class TestFlyDSLCapabilityGate:
    """`_flydsl_pa_decode_num_seqs` promises a fallback, not an exception.

    Every clause it checks is a hard reject inside aiter, so a clause that goes
    missing does not degrade to gluon -- it raises from inside the kernel call
    and kills the worker, while the server keeps answering /metrics and the
    client warms up until it times out. This gate had no coverage at all, and
    two of aiter's rejects were in fact missing from it.
    """

    @staticmethod
    def _call(**over):
        """A call that passes every clause, with one field overridable."""
        import torch

        from atom.model_ops.base_attention import _flydsl_pa_decode_num_seqs

        q = over.pop("q", None)
        if q is None:
            q = torch.empty(16, 16, 128, device="meta", dtype=torch.bfloat16)
        kw = {
            "output": (
                over.pop("output", None) if "output" in over else torch.empty_like(q)
            ),
            "q": q,
            "k_cache": torch.empty(4, 1, 8, 128, 16, device="meta", dtype=_FP8),
            "v_cache": torch.empty(4, 1, 8, 128, 16, device="meta", dtype=_FP8),
            "block_tables": torch.empty(4, 64, device="meta", dtype=torch.int32),
            "context_lens": torch.empty(4, device="meta", dtype=torch.int32),
            "max_seqlen_q": 4,
            "max_context_partition_num": 32,
            "context_partition_size": 256,
            "compute_type": _FP8,
            "q_scale": None,
            "alibi_slopes": None,
            "sinks": None,
            "sliding_window": -1,
            "ps": True,
        }
        kw.update(over)
        return _flydsl_pa_decode_num_seqs(**kw)

    def test_the_baseline_call_is_accepted(self, monkeypatch):
        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        assert self._call() == 4

    @pytest.mark.parametrize(
        "field,value",
        [
            ("sliding_window", 128),
            ("ps", False),
            ("compute_type", "bf16"),
            ("context_partition_size", 128),
            ("max_context_partition_num", 512),
            ("q_scale", "present"),
            ("alibi_slopes", "present"),
            ("sinks", "present"),
        ],
    )
    def test_each_aiter_reject_falls_back(self, monkeypatch, field, value):
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        if value == "bf16":
            value = torch.bfloat16
        elif value == "present":
            value = torch.empty(1, device="meta")
        assert self._call(**{field: value}) is None, f"{field}={value} must fall back"

    @pytest.mark.parametrize("block_size,ok", [(16, True), (128, True), (256, False)])
    def test_block_size_outside_aiters_whitelist_falls_back(
        self, monkeypatch, block_size, ok
    ):
        """--block-size 256 reaches this site whenever use_triton_attn is set.

        aiter takes block_size from dim -2 and rejects anything but 16/64/128;
        the guard used to check only dim -1, so a 256-block deployment routed
        into FlyDSL and died there.
        """
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        k = torch.empty(4, 1, 8, block_size, 16, device="meta", dtype=_FP8)
        assert (self._call(k_cache=k) == 4) is ok

    def test_unsupported_arch_falls_back(self, monkeypatch):
        """aiter builds FlyDSL for gfx942/gfx950 only and raises on the rest."""
        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: False
        )
        assert self._call() is None

    def test_row_count_must_divide_evenly(self, monkeypatch):
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        odd = torch.empty(15, 16, 128, device="meta", dtype=torch.bfloat16)
        assert self._call(q=odd) is None

    @pytest.mark.parametrize("head_dim", [192, 96, 256])
    def test_head_dim_the_cache_does_not_encode_falls_back(self, monkeypatch, head_dim):
        """Two clauses, and the second is the one that was missing.

        192 and 96 fail ATOM's own whitelist. 256 passes it -- and used to be
        asserted as accepted -- but the baseline cache is (.., 8, 128, 16),
        whose num_hgroups encodes head_dim 128, and aiter rejects a q that
        disagrees with its cache. Pinning that acceptance as the contract is
        what would have kept the clause from ever being added.
        """
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        q = torch.empty(16, 16, head_dim, device="meta", dtype=torch.bfloat16)
        assert self._call(q=q) is None

    def test_head_dim_matching_the_cache_is_accepted(self, monkeypatch):
        """The positive half: 256 is fine once the cache encodes 256."""
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        q = torch.empty(16, 16, 256, device="meta", dtype=torch.bfloat16)
        k = torch.empty(4, 1, 16, 128, 16, device="meta", dtype=_FP8)
        v = torch.empty(4, 1, 16, 128, 16, device="meta", dtype=_FP8)
        assert self._call(q=q, k_cache=k, v_cache=v) == 4

    def test_zero_query_length_falls_back_instead_of_dividing(self, monkeypatch):
        """`divmod(rows, 0)` is a ZeroDivisionError, not a fallback."""
        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        assert self._call(max_seqlen_q=0) is None

    def test_more_rows_than_sequences_falls_back(self, monkeypatch):
        """The upper bound is what stops the per-sequence slices going short.

        `context_lens[:n]` with n past the buffer does not raise -- torch hands
        back whatever is there -- so without this clause FlyDSL would be given
        a rectangle that silently disagrees with the batch.
        """
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        q = torch.empty(32, 16, 128, device="meta", dtype=torch.bfloat16)
        assert self._call(q=q) is None

    @pytest.mark.parametrize("shape", [(4, 1, 8, 128), (4, 1, 8, 128, 8)])
    def test_cache_layout_outside_page16_falls_back(self, monkeypatch, shape):
        """A bf16 cache gives x == 8 on the last axis; aiter needs 16."""
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        k = torch.empty(*shape, device="meta", dtype=_FP8)
        assert self._call(k_cache=k) is None

    @pytest.mark.parametrize("head_dim,hgroups", [(96, 6), (1152, 72)])
    def test_head_dim_outside_atoms_own_whitelist_falls_back(
        self, monkeypatch, head_dim, hgroups
    ):
        """The whitelist clause alone, with the cache made to agree.

        The cases above fail the cache clause too, so deleting the whitelist
        left them green. 96 is neither 64 nor a multiple of 128; 1152 is a
        multiple but past aiter's 1024 ceiling.
        """
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        q = torch.empty(16, 16, head_dim, device="meta", dtype=torch.bfloat16)
        k = torch.empty(4, 1, hgroups, 128, 16, device="meta", dtype=_FP8)
        assert self._call(q=q, k_cache=k, v_cache=k) is None

    def test_a_q_dtype_the_kernel_has_no_arm_for_falls_back(self, monkeypatch):
        """aiter picks its scale from q.dtype and raises on anything else."""
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        q = torch.empty(16, 16, 128, device="meta", dtype=torch.float32)
        assert self._call(q=q) is None

    def test_an_output_that_disagrees_with_q_falls_back(self, monkeypatch):
        """aiter writes through the output pointer as if it were q's twin."""
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        other_dtype = torch.empty(16, 16, 128, device="meta", dtype=torch.float16)
        assert self._call(output=other_dtype) is None
        other_shape = torch.empty(32, 16, 128, device="meta", dtype=torch.bfloat16)
        assert self._call(output=other_shape) is None

    def test_a_non_contiguous_head_dim_axis_falls_back(self, monkeypatch):
        """Both axes: aiter checks q and output separately (pa_decode:443,448)."""
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        skewed = torch.empty(
            16, 128, 16, device="meta", dtype=torch.bfloat16
        ).transpose(1, 2)
        assert skewed.shape == (16, 16, 128) and skewed.stride(2) != 1
        contiguous = torch.empty(16, 16, 128, device="meta", dtype=torch.bfloat16)
        assert self._call(q=skewed, output=contiguous) is None
        assert self._call(output=skewed) is None

    def test_q_heads_that_do_not_divide_over_kv_heads_fall_back(self, monkeypatch):
        """The kernel gives each kv head a whole group; 16 over 3 has no split."""
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        k = torch.empty(4, 3, 8, 128, 16, device="meta", dtype=_FP8)
        assert self._call(k_cache=k, v_cache=k) is None

    def test_a_v_cache_of_another_dtype_falls_back(self, monkeypatch):
        """One compute_type covers both caches; a split pair raises inside aiter."""
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        v = torch.empty(4, 1, 8, 128, 16, device="meta", dtype=torch.bfloat16)
        assert self._call(v_cache=v) is None

    @pytest.mark.parametrize(
        "field", ["k_cache", "v_cache", "block_tables", "context_lens"]
    )
    def test_a_non_contiguous_argument_falls_back(self, monkeypatch, field):
        """aiter demands all four contiguous (pa_decode.py:464-470).

        Nothing in this tree hands one over -- every cache view is a `.view()`,
        which raises rather than returning a skewed tensor -- but the SGLang
        bridge takes its pool from someone else and ATOM_PA_FLYDSL routes it
        too, so the gate promises a fallback it could not keep.
        """
        import torch

        monkeypatch.setattr(
            "atom.model_ops.base_attention._flydsl_arch_supported", lambda: True
        )
        skewed = {
            "k_cache": lambda: torch.empty(
                4, 1, 8, 16, 128, device="meta", dtype=_FP8
            ).transpose(3, 4),
            "v_cache": lambda: torch.empty(
                4, 1, 8, 16, 128, device="meta", dtype=_FP8
            ).transpose(3, 4),
            "block_tables": lambda: torch.empty(
                64, 4, device="meta", dtype=torch.int32
            ).t(),
            "context_lens": lambda: torch.empty(8, device="meta", dtype=torch.int32)[
                ::2
            ],
        }[field]()
        assert not skewed.is_contiguous()
        assert self._call(**{field: skewed}) is None

    @pytest.mark.skipif(not _HAS_FLYDSL_PA, reason="needs aiter #4332")
    def test_the_arch_probe_binding_exists(self):
        """The binding, not the module name.

        Every other test here monkeypatches `_flydsl_arch_supported`, so the
        one thing none of them touches is whether it can be computed at all.
        aiter moving `get_gfx_runtime` leaves the module importable, leaves all
        of these green, and makes a guard that exists to prevent an exception
        inside aiter raise one itself on the first decode.
        """
        import importlib

        mod = importlib.import_module("aiter.jit.utils.chip_info")
        assert callable(getattr(mod, "get_gfx_runtime", None))


class TestFlyDSLConstantsMatchAiter:
    """The mirrored limits are copies; a test is what keeps them copies.

    Same reasoning as TestEnvelopeConstants: read aiter's source rather than
    import it, so this runs without a GPU and fails loudly on a rename.
    """

    @staticmethod
    def _aiter_src(rel):
        import pathlib

        import aiter

        return (pathlib.Path(aiter.__file__).parent / rel).read_text()

    @pytest.mark.skipif(not _HAS_FLYDSL_PA, reason="needs aiter #4332")
    def test_block_size_whitelist(self):
        from atom.model_ops.base_attention import _FLYDSL_PA_BLOCK_SIZES

        src = self._aiter_src("ops/flydsl/pa_decode.py")
        assert "block_size not in (16, 64, 128)" in src, (
            "aiter's block_size whitelist moved; _FLYDSL_PA_BLOCK_SIZES is now "
            "either over-rejecting or no longer protecting the call"
        )
        assert _FLYDSL_PA_BLOCK_SIZES == (16, 64, 128)

    @pytest.mark.skipif(not _HAS_FLYDSL_PA, reason="needs aiter #4332")
    def test_supported_archs(self):
        from atom.model_ops.base_attention import _FLYDSL_PA_ARCHS

        src = self._aiter_src("ops/flydsl/pa_decode.py")
        for arch in _FLYDSL_PA_ARCHS:
            assert f'"{arch}"' in src, f"aiter no longer names {arch}"

    @pytest.mark.skipif(not _HAS_FLYDSL_PA, reason="needs aiter #4332")
    def test_tile_size_and_batch_cap(self):
        """The two limits the comment claimed were pinned and were not.

        _FLYDSL_PA_TILE is the only context_partition_size aiter accepts, and
        _FLYDSL_PLAN_MAX_BATCH is where its planner stops. Drift either way is
        silent: too strict falls back more than it must, too loose hands aiter
        a call it raises on.
        """
        from atom.model_ops.base_attention import (
            _FLYDSL_PA_TILE,
            _FLYDSL_PLAN_MAX_BATCH,
        )

        src = self._aiter_src("ops/flydsl/pa_decode.py")
        assert (
            f"context_partition_size={_FLYDSL_PA_TILE}" in src
        ), "aiter's accepted partition size moved"
        plan_src = self._aiter_src("ops/flydsl/kernels/pa_decode_plan.py")
        assert (
            str(_FLYDSL_PLAN_MAX_BATCH) in plan_src
        ), "aiter's planner batch cap moved"
