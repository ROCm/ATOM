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

from types import SimpleNamespace

import pytest

pytest.importorskip("triton", reason="base_attention defines @triton.jit kernels")
pytest.importorskip("aiter", reason="base_attention imports the AITER runtime")

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
        request to the short requests' share.
        """
        from atom.model_ops.attentions import aiter_attention as aa

        seen = {}

        def fake_plan(context_lens, num_kv_heads, **kwargs):
            seen.update(kwargs)
            return _FakePlan()

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

        ctx = SimpleNamespace(shape=(8,), device=SimpleNamespace(index=0))
        assert builder.refresh_flydsl_plan(ctx) is not None
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
        ctx = SimpleNamespace(shape=(8,), device=SimpleNamespace(index=0))
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

    def test_only_the_dense_call_site_opts_in(self):
        """The planner is per call site, the way the split count already is.

        Red if the default flips, if a sparse site starts asking, or if the
        dense one stops. On uniform lengths the planner is a measured loss
        (0.56x at B8/257) that a larger ceiling does not rescue, and the two
        sparse sites are 57 of the 63 pa_decode calls in a step.
        """
        import inspect

        from atom.model_ops import attention_mha
        from atom.model_ops.base_attention import run_pa_decode_gluon
        from atom.model_ops.minimax_m3 import sparse_attn

        param = inspect.signature(run_pa_decode_gluon).parameters["allow_flydsl_plan"]
        assert param.default is False
        assert "allow_flydsl_plan=True" in inspect.getsource(attention_mha)
        assert "allow_flydsl_plan" not in inspect.getsource(sparse_attn)

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

    def test_scratch_is_keyed_by_capacity(self):
        """A refresh that grows the plan must not reuse the old buffers.

        Red if capacity leaves the key: the second call would hand back buffers
        sized for 512 while the kernel writes 1024 rows.
        """
        import torch

        from atom.model_ops.base_attention import _flydsl_plan_scratch

        dev = torch.device("cuda", 0)
        small = _flydsl_plan_scratch(
            _FakePlan(capacity=512), 4, 16, 128, torch.bfloat16, dev
        )
        again = _flydsl_plan_scratch(
            _FakePlan(capacity=512), 4, 16, 128, torch.bfloat16, dev
        )
        big = _flydsl_plan_scratch(
            _FakePlan(capacity=1024), 4, 16, 128, torch.bfloat16, dev
        )
        assert small[0] is again[0], "same shape should reuse"
        assert big[0] is not small[0], "a grown capacity must not reuse"
        assert big[0].shape[1] == 1024

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

        def ctx(n):
            return SimpleNamespace(shape=(n,), device=SimpleNamespace(index=0))

        first = builder.refresh_flydsl_plan(ctx(8))
        assert builder.refresh_flydsl_plan(ctx(8)) is first, "same rung must reuse"
        other = builder.refresh_flydsl_plan(ctx(16))
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

    def test_gluon_is_the_default_and_the_env_short_circuits(self):
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
        import os

        import atom.model_ops.base_attention as ba
        from atom.utils import envs

        os.environ.pop("ATOM_PA_FLYDSL", None)
        assert envs.ATOM_PA_FLYDSL is False, "FlyDSL must be opt-in"

        src = inspect.getsource(ba.run_pa_decode_gluon)
        assert (
            "envs.ATOM_PA_FLYDSL and _flydsl_pa_decode_num_seqs" in src
        ), "the env gate is gone, or no longer short-circuits the capability check"

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

        ctx = SimpleNamespace(shape=(8,), device=SimpleNamespace(index=0))
        assert builder.refresh_flydsl_plan(ctx) is None
        assert not builder._flydsl_plans, "a plan was built with FlyDSL off"
