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
