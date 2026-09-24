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

import sys
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


def _call_v4_native_fp8_decode(paged_decode):
    marker = object()
    return paged_decode.sparse_attn_v4_paged_decode(
        None,
        marker,
        marker,
        marker,
        marker,
        1.0,
        unified_kv_rope=marker,
        q_packed_in=marker,
        q_rope_in=marker,
        qo_indptr=marker,
        query_group=4,
        kv_kind="csa",
    )


def _call_v4_native_fp8_prefill(
    paged_prefill,
    *,
    heads=16,
    tokens=4,
    prefix_has_sentinel=False,
    max_seqlen_q=127,
    max_seqlen_k=1151,
):
    marker = object()
    packed = SimpleNamespace(
        shape=(tokens, heads, 512),
        view=lambda *_shape: marker,
    )
    return paged_prefill.sparse_attn_v4_paged_prefill(
        None,
        marker,
        marker,
        marker,
        None,
        marker,
        marker,
        marker,
        1.0,
        unified_kv_rope=marker,
        q_packed=packed,
        q_rope=marker,
        k_packed=packed,
        k_rope=packed,
        prefix_has_sentinel=prefix_has_sentinel,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )


class TestV4NativeFp8PrefillRouting:
    def _patch_backends(self, monkeypatch, arch="gfx950"):
        from atom.model_ops.v4_kernels import paged_prefill

        monkeypatch.setattr(paged_prefill, "get_gfx_runtime", lambda: arch)
        monkeypatch.setattr(paged_prefill, "_HAS_PREFILL_ASM", False)
        monkeypatch.setattr(
            paged_prefill,
            "pa_sparse_prefill_fp8_opus",
            lambda *args, **kwargs: "opus",
        )
        monkeypatch.setitem(
            sys.modules,
            "atom.model_ops.v4_kernels.paged_prefill_fp8_flydsl",
            SimpleNamespace(
                sparse_attn_v4_paged_prefill_fp8_flydsl=(
                    lambda *args, **kwargs: "flydsl"
                )
            ),
        )
        return paged_prefill

    def test_qualified_short_context_routes_to_flydsl(self, monkeypatch):
        paged_prefill = self._patch_backends(monkeypatch)
        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "1")
        monkeypatch.setenv("ATOM_V4_FLYDSL_FP8_PREFILL", "1")
        monkeypatch.delenv("ATOM_FORCE_V4_PREFILL_OPUS", raising=False)
        assert (
            _call_v4_native_fp8_prefill(
                paged_prefill,
                heads=128,
                tokens=2048,
                max_seqlen_q=127,
                max_seqlen_k=1278,
            )
            == "flydsl"
        )

    @pytest.mark.parametrize(
        "max_seqlen_q,max_seqlen_k",
        [(128, 1278), (127, 4096)],
    )
    def test_unmeasured_prefill_ranges_stay_on_opus(
        self, monkeypatch, max_seqlen_q, max_seqlen_k
    ):
        paged_prefill = self._patch_backends(monkeypatch)
        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "1")
        monkeypatch.setenv("ATOM_V4_FLYDSL_FP8_PREFILL", "1")
        monkeypatch.delenv("ATOM_FORCE_V4_PREFILL_OPUS", raising=False)
        assert (
            _call_v4_native_fp8_prefill(
                paged_prefill,
                heads=128,
                tokens=2048,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
            == "opus"
        )

    def test_p8192_crossover_stays_on_opus(self, monkeypatch):
        paged_prefill = self._patch_backends(monkeypatch)
        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "1")
        monkeypatch.setenv("ATOM_V4_FLYDSL_FP8_PREFILL", "1")
        monkeypatch.delenv("ATOM_FORCE_V4_PREFILL_OPUS", raising=False)
        assert (
            _call_v4_native_fp8_prefill(
                paged_prefill,
                heads=128,
                tokens=2048,
                max_seqlen_k=8192,
            )
            == "opus"
        )

    def test_flydsl_prefill_rejects_sentinel_prefix(self, monkeypatch):
        paged_prefill = self._patch_backends(monkeypatch)
        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "1")
        monkeypatch.setenv("ATOM_V4_FLYDSL_FP8_PREFILL", "1")
        monkeypatch.delenv("ATOM_FORCE_V4_PREFILL_OPUS", raising=False)
        assert (
            _call_v4_native_fp8_prefill(
                paged_prefill,
                heads=128,
                prefix_has_sentinel=True,
            )
            == "opus"
        )

    @pytest.mark.parametrize(
        "master,flydsl,force_opus,arch",
        [
            ("1", "0", "0", "gfx950"),
            ("0", "1", "0", "gfx950"),
            ("1", "1", "1", "gfx950"),
            ("1", "1", "0", "gfx1250"),
        ],
    )
    def test_unqualified_fp8_prefill_keeps_aiter(
        self, monkeypatch, master, flydsl, force_opus, arch
    ):
        paged_prefill = self._patch_backends(monkeypatch, arch=arch)
        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", master)
        monkeypatch.setenv("ATOM_V4_FLYDSL_FP8_PREFILL", flydsl)
        monkeypatch.setenv("ATOM_FORCE_V4_PREFILL_OPUS", force_opus)
        assert _call_v4_native_fp8_prefill(paged_prefill) == "opus"


class TestV4NativeFp8Routing:
    @pytest.mark.parametrize(
        "min_q,max_q,expected",
        [
            (4, 4, 4),
            (7, 7, 7),
            (1, 7, 1),
            (1, 4, 1),
            (3, 4, 1),
            (2, 2, 1),
            (1, 1, 1),
        ],
    )
    def test_only_tuned_rectangular_widths_are_exposed(self, min_q, max_q, expected):
        from atom.model_ops.v4_kernels.paged_decode import v4_decode_query_group

        assert v4_decode_query_group(min_q, max_q) == expected

    def test_env_one_routes_native_fp8_to_triton(self, monkeypatch):
        from atom.model_ops.v4_kernels import paged_decode, paged_decode_fp8_triton

        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "1")
        monkeypatch.setenv("ATOM_V4_TRITON_HYBRID_DECODE", "0")
        monkeypatch.setattr(
            paged_decode_fp8_triton,
            "sparse_attn_v4_paged_decode_fp8_triton_auto",
            lambda *args, **kwargs: "triton",
        )
        monkeypatch.setattr(
            paged_decode,
            "_sparse_attn_v4_paged_decode_asm",
            lambda *args, **kwargs: "aiter",
        )
        assert _call_v4_native_fp8_decode(paged_decode) == "triton"

    def test_env_zero_routes_native_fp8_to_aiter(self, monkeypatch):
        from atom.model_ops.v4_kernels import paged_decode, paged_decode_fp8_triton

        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "0")
        monkeypatch.setattr(
            paged_decode_fp8_triton,
            "sparse_attn_v4_paged_decode_fp8_triton_auto",
            lambda *args, **kwargs: "triton",
        )
        monkeypatch.setattr(
            paged_decode,
            "_sparse_attn_v4_paged_decode_asm",
            lambda *args, **kwargs: "aiter",
        )
        assert _call_v4_native_fp8_decode(paged_decode) == "aiter"

    @pytest.mark.parametrize(
        "tokens,heads,query_group,kv_kind,expected",
        [
            (7, 128, 7, "hca", "aiter"),
            (63, 128, 7, "csa", "triton"),
            (70, 128, 7, "csa", "triton"),
            (84, 16, 7, "dspark", "triton"),
            (64, 16, 4, "hca", "triton"),
        ],
    )
    def test_hybrid_routes_only_measured_winners(
        self,
        monkeypatch,
        tokens,
        heads,
        query_group,
        kv_kind,
        expected,
    ):
        from atom.model_ops.v4_kernels import paged_decode, paged_decode_fp8_triton

        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "1")
        monkeypatch.setenv("ATOM_V4_TRITON_HYBRID_DECODE", "1")
        monkeypatch.delenv("ATOM_V4_TRITON_NATIVE_BF16_V", raising=False)
        monkeypatch.setattr(paged_decode, "_device_arch", lambda _index: "gfx950")
        monkeypatch.setattr(
            paged_decode_fp8_triton,
            "sparse_attn_v4_paged_decode_fp8_triton_auto",
            lambda *args, **kwargs: "triton",
        )
        monkeypatch.setattr(
            paged_decode,
            "_sparse_attn_v4_paged_decode_asm",
            lambda *args, **kwargs: "aiter",
        )
        marker = object()
        q_packed = SimpleNamespace(
            shape=(tokens, heads, 512), device=SimpleNamespace(index=0)
        )
        result = paged_decode.sparse_attn_v4_paged_decode(
            None,
            marker,
            marker,
            marker,
            marker,
            1.0,
            unified_kv_rope=marker,
            q_packed_in=q_packed,
            q_rope_in=marker,
            qo_indptr=marker,
            query_group=query_group,
            kv_kind=kv_kind,
        )
        assert result == expected

    def test_hybrid_routes_native_b6_hca_to_triton(self, monkeypatch):
        from atom.model_ops.v4_kernels import paged_decode, paged_decode_fp8_triton

        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "1")
        monkeypatch.setenv("ATOM_V4_TRITON_HYBRID_DECODE", "1")
        monkeypatch.setenv("ATOM_V4_TRITON_NATIVE_BF16_V", "1")
        monkeypatch.setattr(paged_decode, "_device_arch", lambda _index: "gfx950")
        monkeypatch.setattr(
            paged_decode_fp8_triton,
            "sparse_attn_v4_paged_decode_fp8_triton_auto",
            lambda *args, **kwargs: "triton",
        )
        monkeypatch.setattr(
            paged_decode,
            "_sparse_attn_v4_paged_decode_asm",
            lambda *args, **kwargs: "aiter",
        )
        marker = object()
        q_packed = SimpleNamespace(
            shape=(6 * 7, 128, 512), device=SimpleNamespace(index=0)
        )
        result = paged_decode.sparse_attn_v4_paged_decode(
            None,
            marker,
            marker,
            marker,
            marker,
            1.0,
            unified_kv_rope=marker,
            q_packed_in=q_packed,
            q_rope_in=marker,
            qo_indptr=marker,
            query_group=7,
            kv_kind="hca",
        )
        assert result == "triton"

    def test_b6_hca_routes_to_graphsafe_flydsl(self, monkeypatch):
        from atom.model_ops.v4_kernels import paged_decode

        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "1")
        monkeypatch.setenv("ATOM_V4_FLYDSL_FP8_DECODE", "1")
        monkeypatch.setenv("ATOM_V4_TRITON_HYBRID_DECODE", "1")
        monkeypatch.setattr(paged_decode, "_device_arch", lambda _index: "gfx950")
        monkeypatch.setitem(
            sys.modules,
            "atom.model_ops.v4_kernels.paged_decode_fp8_flydsl",
            SimpleNamespace(
                sparse_attn_v4_paged_decode_fp8_flydsl_auto=(
                    lambda *args, **kwargs: "flydsl"
                )
            ),
        )
        monkeypatch.setattr(
            paged_decode,
            "_sparse_attn_v4_paged_decode_asm",
            lambda *args, **kwargs: "aiter",
        )
        marker = object()
        q_packed = SimpleNamespace(
            shape=(6 * 7, 128, 512), device=SimpleNamespace(index=0)
        )
        result = paged_decode.sparse_attn_v4_paged_decode(
            None,
            marker,
            marker,
            marker,
            marker,
            1.0,
            unified_kv_rope=marker,
            q_packed_in=q_packed,
            q_rope_in=marker,
            qo_indptr=marker,
            empty_kv_indptr=marker,
            query_group=7,
            kv_kind="hca",
        )
        assert result == "flydsl"

    def test_csa_b5_b32_routes_to_graphsafe_flydsl(self, monkeypatch):
        from atom.model_ops.v4_kernels import paged_decode

        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "1")
        monkeypatch.delenv("ATOM_V4_FLYDSL_FP8_DECODE", raising=False)
        monkeypatch.setattr(paged_decode, "_device_arch", lambda _index: "gfx950")
        monkeypatch.setitem(
            sys.modules,
            "atom.model_ops.v4_kernels.paged_decode_fp8_flydsl",
            SimpleNamespace(
                sparse_attn_v4_paged_decode_fp8_flydsl_auto=(
                    lambda *args, **kwargs: "flydsl"
                )
            ),
        )
        marker = object()
        for requests in (5, 6, 7, 8, 9, 10, 16, 17, 18, 32):
            q_packed = SimpleNamespace(
                shape=(requests * 7, 128, 512), device=SimpleNamespace(index=0)
            )
            result = paged_decode.sparse_attn_v4_paged_decode(
                None,
                marker,
                marker,
                marker,
                marker,
                1.0,
                unified_kv_rope=marker,
                q_packed_in=q_packed,
                q_rope_in=marker,
                qo_indptr=marker,
                empty_kv_indptr=marker,
                query_group=7,
                kv_kind="csa",
            )
            assert result == "flydsl"

    @pytest.mark.parametrize("requests", [4, 33])
    def test_csa_outside_flydsl_envelope_keeps_triton(self, monkeypatch, requests):
        from atom.model_ops.v4_kernels import paged_decode, paged_decode_fp8_triton

        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "1")
        monkeypatch.delenv("ATOM_V4_FLYDSL_FP8_DECODE", raising=False)
        monkeypatch.setattr(paged_decode, "_device_arch", lambda _index: "gfx950")
        monkeypatch.setitem(
            sys.modules,
            "atom.model_ops.v4_kernels.paged_decode_fp8_flydsl",
            SimpleNamespace(
                sparse_attn_v4_paged_decode_fp8_flydsl_auto=(
                    lambda *args, **kwargs: "flydsl"
                )
            ),
        )
        monkeypatch.setattr(
            paged_decode_fp8_triton,
            "sparse_attn_v4_paged_decode_fp8_triton_auto",
            lambda *args, **kwargs: "triton",
        )
        marker = object()
        q_packed = SimpleNamespace(
            shape=(requests * 7, 128, 512), device=SimpleNamespace(index=0)
        )
        result = paged_decode.sparse_attn_v4_paged_decode(
            None,
            marker,
            marker,
            marker,
            marker,
            1.0,
            unified_kv_rope=marker,
            q_packed_in=q_packed,
            q_rope_in=marker,
            qo_indptr=marker,
            empty_kv_indptr=marker,
            query_group=7,
            kv_kind="csa",
        )
        assert result == "triton"

    def test_flydsl_decode_master_zero_keeps_aiter(self, monkeypatch):
        from atom.model_ops.v4_kernels import paged_decode

        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "0")
        monkeypatch.setenv("ATOM_V4_FLYDSL_FP8_DECODE", "1")
        monkeypatch.setattr(paged_decode, "_device_arch", lambda _index: "gfx950")
        monkeypatch.setattr(
            paged_decode,
            "_sparse_attn_v4_paged_decode_asm",
            lambda *args, **kwargs: "aiter",
        )
        marker = object()
        q_packed = SimpleNamespace(
            shape=(6 * 7, 128, 512), device=SimpleNamespace(index=0)
        )
        result = paged_decode.sparse_attn_v4_paged_decode(
            None,
            marker,
            marker,
            marker,
            marker,
            1.0,
            unified_kv_rope=marker,
            q_packed_in=q_packed,
            q_rope_in=marker,
            qo_indptr=marker,
            empty_kv_indptr=marker,
            query_group=7,
            kv_kind="hca",
        )
        assert result == "aiter"

    @pytest.mark.parametrize(
        "kv_kind,tokens,kv_len,expected",
        [
            ("hca", 6 * 7, 2048, (13, 14, 10, 50, 11, 64)),
            ("csa", 5 * 7, 384, (6, 6, 3, 20, 0, 0)),
            ("csa", 6 * 7, 1152, (6, 6, 3, 20, 0, 0)),
            ("csa", 7 * 7, 384, (6, 6, 4, 12, 0, 0)),
            ("csa", 8 * 7, 1152, (6, 6, 4, 12, 0, 0)),
            ("csa", 9 * 7, 640, (3, 8, 6, 12, 0, 0)),
            ("csa", 10 * 7, 384, (3, 8, 6, 12, 0, 0)),
            ("csa", 12 * 7, 1152, (3, 8, 6, 12, 0, 0)),
            ("csa", 13 * 7, 384, (2, 9, 6, 12, 0, 0)),
            ("csa", 16 * 7, 1152, (2, 9, 6, 12, 0, 0)),
            ("csa", 17 * 7, 384, (2, 10, 6, 12, 0, 0)),
            ("csa", 18 * 7, 1152, (2, 10, 6, 12, 0, 0)),
            ("csa", 19 * 7, 384, (2, 9, 6, 12, 0, 0)),
            ("csa", 32 * 7, 1152, (2, 9, 6, 12, 0, 0)),
        ],
    )
    def test_flydsl_auto_selects_measured_schedule(
        self, monkeypatch, kv_kind, tokens, kv_len, expected
    ):
        from atom.model_ops.v4_kernels import paged_decode_fp8_flydsl as fp8

        captured = {}

        def candidate(*args, **kwargs):
            captured.update(kwargs)
            return "flydsl"

        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_flydsl_graphsafe",
            candidate,
        )
        q_packed = SimpleNamespace(shape=(tokens, 128, 512))
        kv_indices = SimpleNamespace(numel=lambda: tokens * kv_len)
        result = fp8.sparse_attn_v4_paged_decode_fp8_flydsl_auto(
            q_packed,
            object(),
            object(),
            object(),
            kv_indices,
            object(),
            object(),
            object(),
            1.0,
            query_group=7,
            kv_kind=kv_kind,
        )
        assert result == "flydsl"
        assert (
            captured["max_splits"],
            captured["split_tiles"],
            captured["split_tiles_short"],
            captured["split_short_max_tiles"],
            captured["split_tiles_mid"],
            captured["split_mid_max_tiles"],
        ) == expected
        assert captured["waves_per_eu"] == (
            0 if kv_kind == "csa" and tokens <= 6 * 7 else 1
        )

    def test_default_hybrid_does_not_route_other_architectures_to_aiter(
        self, monkeypatch
    ):
        from atom.model_ops.v4_kernels import paged_decode, paged_decode_fp8_triton

        monkeypatch.setenv("ATOM_USE_TRITON_ATTN", "1")
        monkeypatch.delenv("ATOM_V4_TRITON_HYBRID_DECODE", raising=False)
        monkeypatch.setattr(paged_decode, "_device_arch", lambda _index: "gfx1250")
        monkeypatch.setattr(
            paged_decode_fp8_triton,
            "sparse_attn_v4_paged_decode_fp8_triton_auto",
            lambda *args, **kwargs: "triton",
        )
        monkeypatch.setattr(
            paged_decode,
            "_sparse_attn_v4_paged_decode_asm",
            lambda *args, **kwargs: "aiter",
        )
        marker = object()
        q_packed = SimpleNamespace(shape=(7, 128, 512), device=SimpleNamespace(index=0))
        result = paged_decode.sparse_attn_v4_paged_decode(
            None,
            marker,
            marker,
            marker,
            marker,
            1.0,
            unified_kv_rope=marker,
            q_packed_in=q_packed,
            q_rope_in=marker,
            qo_indptr=marker,
            query_group=7,
            kv_kind="hca",
        )
        assert result == "triton"

    @pytest.mark.parametrize("query_group", [1, 2, 3])
    def test_only_supported_query_groups_use_query_fusion(
        self, monkeypatch, query_group
    ):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(("regular", kwargs)) or "regular",
        )
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton_query_group",
            lambda *args, **kwargs: (
                calls.append(("query_group", kwargs)) or "query_group"
            ),
        )
        q = SimpleNamespace(shape=(64, 16, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            query_group=query_group,
            kv_kind="csa",
        )
        assert result == "regular"
        assert calls[0][0] == "regular"

    @pytest.mark.parametrize(
        "requests,expected",
        [
            (1, (16, 16)),
            (2, (16, 16)),
            (3, (32, 8)),
            (5, (32, 8)),
            (6, (16, 4)),
            (12, (16, 4)),
            (13, (16, 2)),
            (24, (16, 2)),
            (25, (16, 1)),
        ],
    )
    def test_dspark_uses_short_window_tuned_dispatch(
        self, monkeypatch, requests, expected
    ):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(("regular", kwargs)) or "regular",
        )
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton_query_group",
            lambda *args, **kwargs: (
                calls.append(("query_group", kwargs)) or "query_group"
            ),
        )
        q = SimpleNamespace(shape=(requests * 6, 16, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            kv_kind="dspark",
        )
        assert result == "regular"
        _, kwargs = calls[0]
        block_k, splits = expected
        assert kwargs["block_h"] == 16
        assert kwargs["block_k"] == block_k
        assert kwargs["kv_splits"] == splits
        assert kwargs["num_stages"] == 2
        assert kwargs["matrix_instr_nonkdim"] == 16
        assert kwargs["use_mxfp8_qk"] is True
        assert kwargs["reduce_num_warps"] == 1
        assert kwargs["fp16_partials"] is True

    @pytest.mark.parametrize("tokens", [7, 14])
    def test_q7_c1_c2_use_regular_split16(self, monkeypatch, tokens):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(("regular", kwargs)) or "regular",
        )
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton_query_group",
            lambda *args, **kwargs: (
                calls.append(("query_group", kwargs)) or "query_group"
            ),
        )
        q = SimpleNamespace(shape=(tokens, 16, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            query_group=7,
            kv_kind="hca",
        )
        assert result == "regular"
        _, kwargs = calls[0]
        assert kwargs["block_h"] == 16
        assert kwargs["block_k"] == 16
        assert kwargs["kv_splits"] == 16
        assert kwargs["num_stages"] == 2
        assert kwargs["num_warps"] == 4
        assert kwargs["use_mxfp8_qk"] is True
        assert kwargs["reduce_num_warps"] == 4
        assert kwargs["fp16_partials"] is True

    @pytest.mark.parametrize(
        "tokens,expected_split",
        [(28, 16), (56, 16), (112, 8)],
    )
    def test_q7_c4_plus_uses_four_query_stripes(
        self,
        monkeypatch,
        tokens,
        expected_split,
    ):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(("regular", kwargs)) or "regular",
        )
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton_query_group",
            lambda *args, **kwargs: (
                calls.append(("query_group", kwargs)) or "query_group"
            ),
        )
        q = SimpleNamespace(shape=(tokens, 16, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            query_group=7,
            kv_kind="hca",
        )
        assert result == "query_group"
        _, kwargs = calls[0]
        assert kwargs["query_group"] == 7
        assert kwargs["fused_query_group"] == 4
        assert kwargs["block_k"] == 16
        assert kwargs["kv_splits"] == expected_split
        assert kwargs["num_stages"] == 2
        assert kwargs["num_warps"] == 4
        assert kwargs["use_mxfp8_qk"] is True
        assert kwargs["reduce_num_warps"] == 1
        assert kwargs["fp16_partials"] is True

    @pytest.mark.parametrize(
        "tokens,expected_split,expected_block_k,expected_stages",
        [
            (7, 16, 64, 1),
            (14, 8, 32, 1),
            (21, 4, 64, 1),
            (28, 4, 64, 1),
        ],
    )
    def test_q7_dp_hca_low_batch_uses_query_and_head_tiled_triton(
        self,
        monkeypatch,
        tokens,
        expected_split,
        expected_block_k,
        expected_stages,
    ):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(("regular", kwargs)) or "regular",
        )
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton_query_group",
            lambda *args, **kwargs: (
                calls.append(("query_group", kwargs)) or "query_group"
            ),
        )
        q = SimpleNamespace(shape=(tokens, 128, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            query_group=7,
            kv_kind="hca",
        )
        assert result == "query_group"
        _, kwargs = calls[0]
        assert kwargs["query_group"] == 7
        assert kwargs["fused_query_group"] == 4
        assert kwargs["block_h"] == 16
        assert kwargs["block_k"] == expected_block_k
        assert kwargs["kv_splits"] == expected_split
        assert kwargs["num_stages"] == expected_stages
        assert kwargs["num_warps"] == 4
        assert kwargs["waves_per_eu"] == 1
        assert kwargs["matrix_instr_nonkdim"] == 0
        assert kwargs["use_mxfp8_qk"] is True
        assert kwargs.get("use_native_bf16_v", False) is False
        assert kwargs["reduce_num_warps"] == 1
        assert kwargs["fp16_partials"] is True

    @pytest.mark.parametrize(
        "requests,expected_splits",
        [
            (1, 16),
            (2, 8),
            (3, 8),
            (4, 8),
        ],
    )
    def test_q7_dp_hca_native_bf16_v_uses_regular_low_batch_kernel(
        self, monkeypatch, requests, expected_splits
    ):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(fp8, "_ENABLE_NATIVE_BF16_V", True)
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(("regular", kwargs)) or "regular",
        )
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton_query_group",
            lambda *args, **kwargs: (
                calls.append(("query_group", kwargs)) or "query_group"
            ),
        )
        q = SimpleNamespace(shape=(requests * 7, 128, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            query_group=7,
            kv_kind="hca",
        )
        assert result == "regular"
        _, kwargs = calls[0]
        assert kwargs["block_h"] == 64
        assert kwargs["block_k"] == 32
        assert kwargs["kv_splits"] == expected_splits
        assert kwargs["num_stages"] == 2
        assert kwargs["num_warps"] == 4
        assert kwargs["waves_per_eu"] == 1
        assert kwargs["matrix_instr_nonkdim"] == 16
        assert kwargs["use_mxfp8_qk"] is True
        assert kwargs["use_native_bf16_v"] is True
        assert kwargs["reduce_num_warps"] == 1
        assert kwargs["fp16_partials"] is True

    @pytest.mark.parametrize(
        "requests,expected",
        [
            (5, (64, 8, 1, 0)),
            (6, (32, 8, 2, 16)),
            (7, (32, 2, 2, 16)),
            (8, (32, 2, 2, 16)),
            (9, (32, 2, 2, 16)),
            (10, (32, 7, 2, 16)),
            (11, (32, 3, 2, 16)),
            (12, (32, 3, 2, 16)),
            (13, (32, 4, 2, 16)),
            (14, (32, 2, 2, 16)),
            (16, (32, 2, 2, 16)),
            (32, (32, 2, 2, 16)),
        ],
    )
    def test_q7_dp_hca_b5_plus_uses_qh64_tuned_dispatch(
        self, monkeypatch, requests, expected
    ):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(("regular", kwargs)) or "regular",
        )
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton_query_group",
            lambda *args, **kwargs: (
                calls.append(("query_group", kwargs)) or "query_group"
            ),
        )
        q = SimpleNamespace(shape=(requests * 7, 128, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            query_group=7,
            kv_kind="hca",
        )
        assert result == "regular"
        _, kwargs = calls[0]
        block_k, splits, stages, matrix = expected
        assert kwargs["block_h"] == 64
        assert kwargs["block_k"] == block_k
        assert kwargs["kv_splits"] == splits
        assert kwargs["num_stages"] == stages
        assert kwargs["matrix_instr_nonkdim"] == matrix
        assert kwargs["use_mxfp8_qk"] is True
        assert kwargs["use_native_bf16_v"] is False
        assert kwargs["reduce_num_warps"] == 1
        assert kwargs["fp16_partials"] is True

    def test_q7_dp_hca_native_bf16_v_requires_explicit_opt_in(self, monkeypatch):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(fp8, "_ENABLE_NATIVE_BF16_V", True)
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(kwargs) or "regular",
        )
        q = SimpleNamespace(shape=(5 * 7, 128, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            query_group=7,
            kv_kind="hca",
        )
        assert result == "regular"
        assert calls[0]["use_native_bf16_v"] is True

    @pytest.mark.parametrize(
        "packed_uniform_splits,packed_query_group,split_tiles,error",
        [
            (-1, 7, 1, r"packed_uniform_splits must be in \[0, kv_splits\]"),
            (5, 7, 1, r"packed_uniform_splits must be in \[0, kv_splits\]"),
            (3, 0, 1, "requires packed_query_group and split_tiles"),
            (3, 7, 0, "requires packed_query_group and split_tiles"),
        ],
    )
    def test_packed_uniform_split_validation(
        self,
        packed_uniform_splits,
        packed_query_group,
        split_tiles,
        error,
    ):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        q_packed = SimpleNamespace(
            dtype=fp8.dtypes.fp8,
            shape=(7, 128, 512),
            dim=lambda: 3,
        )
        q_rope = SimpleNamespace(dtype=fp8.torch.bfloat16, shape=(7, 128, 64))
        kv_packed = SimpleNamespace(
            dtype=fp8.dtypes.fp8,
            shape=(16, 512),
            dim=lambda: 2,
        )
        kv_rope = SimpleNamespace(dtype=fp8.torch.bfloat16, shape=(16, 64))

        with pytest.raises(ValueError, match=error):
            fp8.sparse_attn_v4_paged_decode_fp8_triton(
                q_packed,
                q_rope,
                kv_packed,
                kv_rope,
                object(),
                object(),
                object(),
                1.0,
                kv_splits=4,
                split_tiles=split_tiles,
                packed_query_group=packed_query_group,
                packed_uniform_splits=packed_uniform_splits,
            )

    def test_q7_dp_hca_native_bf16_v_b6_uses_packed_adaptive_split(self, monkeypatch):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(fp8, "_ENABLE_NATIVE_BF16_V", True)
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(kwargs) or "regular",
        )
        q = SimpleNamespace(shape=(6 * 7, 128, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            query_group=7,
            kv_kind="hca",
        )

        assert result == "regular"
        assert calls == [
            {
                "block_h": 64,
                "block_k": 64,
                "kv_splits": 11,
                "num_stages": 2,
                "num_warps": 4,
                "waves_per_eu": 1,
                "matrix_instr_nonkdim": 16,
                "use_mxfp8_qk": True,
                "use_native_bf16_v": True,
                "split_tiles": 9,
                "split_tiles_short": 7,
                "split_short_max_tiles": 32,
                "packed_query_group": 7,
                "reduce_d_chunk": 512,
                "reduce_num_warps": 1,
                "fp16_partials": True,
            }
        ]

    @pytest.mark.parametrize(
        "requests,expected",
        [
            (1, (16, 16, 2, 16)),
            (2, (16, 8, 2, 16)),
            (3, (32, 4, 3, 16)),
            (4, (32, 4, 3, 16)),
            (5, (64, 4, 1, 0)),
            (6, (32, 3, 3, 16)),
            (7, (32, 2, 3, 16)),
            (8, (32, 2, 3, 16)),
            (9, (32, 2, 3, 16)),
            (10, (64, 3, 2, 16)),
            (11, (64, 3, 2, 16)),
            (12, (64, 3, 2, 16)),
            (13, (64, 1, 2, 16)),
            (14, (64, 1, 2, 16)),
            (16, (64, 1, 2, 16)),
        ],
    )
    def test_q7_dp_csa_uses_qh64_tuned_dispatch(self, monkeypatch, requests, expected):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(("regular", kwargs)) or "regular",
        )
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton_query_group",
            lambda *args, **kwargs: (
                calls.append(("query_group", kwargs)) or "query_group"
            ),
        )
        q = SimpleNamespace(shape=(requests * 7, 128, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            query_group=7,
            kv_kind="csa",
            gfx950_native_v=True,
        )
        assert result == "regular"
        _, kwargs = calls[0]
        block_k, splits, stages, matrix_nonkdim = expected
        assert kwargs["block_h"] == 64
        assert kwargs["block_k"] == block_k
        assert kwargs["kv_splits"] == splits
        assert kwargs["num_stages"] == stages
        assert kwargs["matrix_instr_nonkdim"] == matrix_nonkdim
        assert kwargs["use_mxfp8_qk"] is True
        assert kwargs["use_native_bf16_v"] is (requests >= 10)
        assert kwargs["schedule_hint"] == ("attention" if requests >= 10 else "none")
        assert kwargs["fp16_partials"] is True

    def test_c16_csa_uses_tuned_q4_split16(self, monkeypatch):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(("regular", kwargs)) or "regular",
        )
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton_query_group",
            lambda *args, **kwargs: (
                calls.append(("query_group", kwargs)) or "query_group"
            ),
        )
        q = SimpleNamespace(shape=(64, 16, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            query_group=4,
            kv_kind="csa",
        )
        assert result == "query_group"
        _, kwargs = calls[0]
        assert kwargs["query_group"] == 4
        assert kwargs["kv_splits"] == 16
        assert kwargs["matrix_instr_nonkdim"] == 16
        assert kwargs["use_mxfp8_qk"] is False
        assert kwargs["fp16_partials"] is True

    def test_c16_hca_uses_tuned_regular_split4(self, monkeypatch):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(("regular", kwargs)) or "regular",
        )
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton_query_group",
            lambda *args, **kwargs: (
                calls.append(("query_group", kwargs)) or "query_group"
            ),
        )
        q = SimpleNamespace(shape=(64, 16, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            query_group=4,
            kv_kind="hca",
        )
        assert result == "regular"
        _, kwargs = calls[0]
        assert kwargs["block_h"] == 16
        assert kwargs["block_k"] == 16
        assert kwargs["kv_splits"] == 4
        assert kwargs["num_stages"] == 3
        assert kwargs["num_warps"] == 8
        assert kwargs["fp16_partials"] is True

    def test_c64_hca_uses_capture_safe_q4_fp16_partials(self, monkeypatch):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(("regular", kwargs)) or "regular",
        )
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton_query_group",
            lambda *args, **kwargs: (
                calls.append(("query_group", kwargs)) or "query_group"
            ),
        )
        q = SimpleNamespace(shape=(256, 16, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            query_group=4,
            kv_kind="hca",
        )
        assert result == "query_group"
        _, kwargs = calls[0]
        assert kwargs["query_group"] == 4
        assert kwargs["kv_splits"] == 4
        assert kwargs["block_k"] == 16
        assert kwargs["num_stages"] == 3
        assert kwargs["fp16_partials"] is True

    @pytest.mark.parametrize("kv_kind", ["csa", "hca"])
    def test_q4_dp_heads_use_regular_triton(self, monkeypatch, kv_kind):
        from atom.model_ops.v4_kernels import paged_decode_fp8_triton as fp8

        calls = []
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton",
            lambda *args, **kwargs: calls.append(("regular", kwargs)) or "regular",
        )
        monkeypatch.setattr(
            fp8,
            "sparse_attn_v4_paged_decode_fp8_triton_query_group",
            lambda *args, **kwargs: (
                calls.append(("query_group", kwargs)) or "query_group"
            ),
        )
        q = SimpleNamespace(shape=(256, 128, 512))
        result = fp8.sparse_attn_v4_paged_decode_fp8_triton_auto(
            q,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1.0,
            query_group=4,
            kv_kind=kv_kind,
        )
        assert result == "regular"
        assert calls[0][0] == "regular"
