"""Unit tests for the profiler label taxonomy (`build_run_label`).

Pure-function tests — no GPU / no forward pass needed. Guards the label
contract that `tools/parse_trace.py` and the capture-trace skill depend on.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from atom.model_engine.run_labels import build_run_label


@dataclass
class _FakeBatch:
    total_tokens_num: int = 0
    total_seqs_num_prefill: int = 0
    total_seqs_num_decode: int = 0
    num_spec_step: int = 0
    context_lens: object = None
    num_scheduled_tokens: object = None


def prefill_batch(tok, ctx):
    return _FakeBatch(
        total_tokens_num=tok,
        total_seqs_num_prefill=len(ctx),
        context_lens=np.asarray(ctx, dtype=np.int64),
    )


def decode_batch(tok, d, p=0, spec=0):
    return _FakeBatch(
        total_tokens_num=tok,
        total_seqs_num_prefill=p,
        total_seqs_num_decode=d,
        num_spec_step=spec,
        context_lens=np.zeros(d, dtype=np.int64),
    )


class TestKindPrefix:
    def test_real_prefill(self):
        lbl = build_run_label(
            is_prefill=True,
            use_cudagraph=False,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=2,
            running_bs=2,
            batch=prefill_batch(14721, [7803, 6918]),
        )
        assert lbl.startswith("prefill[")
        assert "tok=14721" in lbl and "ctx=[7803, 6918]" in lbl

    def test_real_cudagraph_decode(self):
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=64,
            running_bs=64,
            batch=decode_batch(64, d=64),
        )
        assert lbl.startswith("decode[")
        assert " d=64" in lbl

    def test_real_eager_decode(self):
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=False,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=300,
            running_bs=300,
            batch=decode_batch(300, d=300),
        )
        assert lbl.startswith("eager_decode[")

    def test_dummy_decode_distinct_from_real(self):
        dummy = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=True,
            tbo_on=False,
            scheduled_bs=1,
            running_bs=1,
            batch=decode_batch(1, d=1),
        )
        real = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=1,
            running_bs=1,
            batch=decode_batch(1, d=1),
        )
        assert dummy.startswith("dummy_decode[")
        assert real.startswith("decode[")
        # The whole point: dummy must NOT be mistaken for a real decode.
        assert not dummy.startswith("decode[")

    def test_dummy_prefill(self):
        lbl = build_run_label(
            is_prefill=True,
            use_cudagraph=False,
            is_dummy=True,
            tbo_on=False,
            scheduled_bs=1,
            running_bs=1,
            batch=prefill_batch(8192, [8192]),
        )
        assert lbl.startswith("dummy_prefill[")
        assert not lbl.startswith("prefill[")

    def test_dummy_eager_decode(self):
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=False,
            is_dummy=True,
            tbo_on=False,
            scheduled_bs=1,
            running_bs=1,
            batch=decode_batch(1, d=1),
        )
        assert lbl.startswith("dummy_eager_decode[")


class TestFields:
    def test_tbo_field(self):
        lbl = build_run_label(
            is_prefill=True,
            use_cudagraph=False,
            is_dummy=False,
            tbo_on=True,
            scheduled_bs=3,
            running_bs=3,
            batch=prefill_batch(16384, [7000, 6000, 3384]),
        )
        assert lbl.endswith("tbo=1]")

    def test_spec_and_p_fields(self):
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=128,
            running_bs=128,
            batch=decode_batch(128, d=126, p=2, spec=3),
        )
        assert " p=2" in lbl and " d=126" in lbl and " spec=3" in lbl

    def test_ctx_truncation_many_seqs(self):
        lbl = build_run_label(
            is_prefill=True,
            use_cudagraph=False,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=8,
            running_bs=8,
            batch=prefill_batch(8000, list(range(8))),
        )
        assert "...+5" in lbl  # 8 seqs → first 3 shown + "...+5"

    def test_no_batch(self):
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=16,
            running_bs=16,
            batch=None,
        )
        assert lbl == "decode[bs=16]"


class TestGraphBs:
    def test_padded_graph_shows_real_over_graph(self):
        # CUDAGraph replays bs=128 for a real batch of 117 → "bs=117/128".
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=117,
            running_bs=128,
            batch=decode_batch(117, d=117),
        )
        assert lbl.startswith("decode[bs=117/128 ")

    def test_matching_graph_bs_omits_slash(self):
        # No padding (graph bs == real bs) → plain "bs=128".
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=128,
            running_bs=128,
            batch=decode_batch(128, d=128),
        )
        assert lbl.startswith("decode[bs=128 ")
        assert "/" not in lbl

    def test_the_single_number_is_what_ran(self):
        """The unslashed form reports `running_bs`, not `scheduled_bs`.

        They are equal here -- `decide` ceils to a captured size, so the
        cudagraph path never runs narrower than it scheduled -- which is
        exactly why reading the wrong one is invisible in the output. Pinned
        by construction instead: a caller that passed only `scheduled_bs`
        would not compile.
        """
        import inspect

        params = inspect.signature(build_run_label).parameters
        assert params["running_bs"].default is inspect.Parameter.empty, (
            "every step has a running_bs; making it optional invites a caller "
            "to omit it and silently label a padded step as unpadded"
        )
        with pytest.raises(TypeError):
            build_run_label(
                is_prefill=False,
                use_cudagraph=True,
                is_dummy=False,
                tbo_on=False,
                scheduled_bs=64,
                batch=decode_batch(64, d=64),
            )

    def test_dummy_decode_padded(self):
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=True,
            tbo_on=False,
            scheduled_bs=1,
            running_bs=8,
            batch=decode_batch(1, d=1),
        )
        assert lbl.startswith("dummy_decode[bs=1/8 ")

    def test_graph_bs_ignored_on_eager_path(self):
        # running_bs only applies to the CUDAGraph path; eager decode ignores it.
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=False,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=300,
            running_bs=512,
            batch=decode_batch(300, d=300),
        )
        assert lbl.startswith("eager_decode[bs=300 ")
        assert "/" not in lbl

    def test_parse_trace_regex_extracts_real_bs(self):
        # tools/parse_trace.py uses re.search(r"bs=(\d+)"), which must still pull
        # the REAL batch (117), not the graph pad (128), from "bs=117/128".
        import re

        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=117,
            running_bs=128,
            batch=decode_batch(117, d=117),
        )
        assert int(re.search(r"bs=(\d+)", lbl).group(1)) == 117


def _eager(is_prefill, tok, running_tokens, ctx, **kw):
    return build_run_label(
        is_prefill=is_prefill,
        use_cudagraph=False,
        is_dummy=False,
        tbo_on=False,
        scheduled_bs=len(ctx),
        running_bs=len(ctx),
        batch=(prefill_batch if is_prefill else decode_batch_ctx)(tok, ctx),
        running_tokens=running_tokens,
        **kw,
    )


def decode_batch_ctx(tok, ctx):
    return _FakeBatch(
        total_tokens_num=tok,
        total_seqs_num_decode=len(ctx),
        context_lens=np.asarray(ctx, dtype=np.int64),
    )


class TestPaddedTokens:
    """`tok=<scheduled>/<running>` wherever the forward runs rows past the
    scheduled ones, as `bs` shows a padded graph batch."""

    def test_a_prefill_padded_to_a_token_graph_bucket(self):
        assert "tok=155/256 ctx=48283" in _eager(True, 155, 256, [48283])

    def test_a_decode_padded_to_a_peers_prefill_bucket(self):
        lbl = _eager(False, 12, 3072, [84991, 169353])
        assert lbl.startswith("eager_decode[bs=2 tok=12/3072 ")

    def test_an_unpadded_step_shows_one_number(self):
        for running in (7424, None):
            lbl = _eager(True, 7424, running, [200192])
            assert "tok=7424 " in lbl and "/" not in lbl

    def test_a_decode_graph_shows_its_rectangle(self):
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=2,
            running_bs=3,
            batch=decode_batch(12, d=2, spec=5),
            running_tokens=18,
        )
        assert lbl.startswith("decode[bs=2/3 tok=12/18 d=2 spec=5")

    def test_parse_trace_still_reads_the_scheduled_tokens(self):
        import re

        lbl = _eager(False, 12, 3072, [84991, 169353])
        assert int(re.search(r"tok=(\d+)", lbl).group(1)) == 12


class TestDpRanks:
    """`dp=[...]`: what every DP rank scheduled and whether it prefills, this
    rank starred -- what a decode beside a prefilling peer was padded for."""

    def test_a_decode_beside_a_prefilling_peer(self):
        lbl = _eager(
            False, 12, 3072, [84991, 169353], dp=((12, False), (2237, True)), dp_rank=0
        )
        assert lbl.endswith(" dp=[12d*,2237p]]")

    def test_the_prefilling_rank_stars_itself(self):
        lbl = _eager(
            True, 2237, 3072, [200199, 86838], dp=((12, False), (2237, True)), dp_rank=1
        )
        assert " dp=[12d,2237p*]" in lbl

    def test_a_decode_graph_step_carries_it_too(self):
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=True,
            scheduled_bs=2,
            running_bs=3,
            batch=decode_batch(12, d=2, spec=5),
            running_tokens=18,
            dp=((12, False), (6, False)),
            dp_rank=0,
        )
        assert lbl.endswith(" dp=[12d*,6d] tbo=1]")

    def test_one_rank_has_no_dp_field(self):
        assert "dp=" not in _eager(True, 155, 256, [48283])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


class TestPrefillAttentionSpan:
    def test_nctx_is_new_tokens_times_mean_attended_context(self):
        # seq A: 1000 cached + 3000 new (ctx 4000); seq B: 0 cached + 2000 new (ctx 2000)
        batch = _FakeBatch(
            total_tokens_num=5000,
            total_seqs_num_prefill=2,
            context_lens=np.asarray([4000, 2000], dtype=np.int64),
            num_scheduled_tokens=np.asarray([3000, 2000], dtype=np.int32),
        )
        lbl = build_run_label(
            is_prefill=True,
            use_cudagraph=False,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=2,
            batch=batch,
            running_bs=2,
        )
        # 3000 * (4000 - 1500) + 2000 * (2000 - 1000) = 9.5M
        assert "nctx=9.5M" in lbl

    def test_no_nctx_without_scheduled_tokens(self):
        lbl = build_run_label(
            is_prefill=True,
            use_cudagraph=False,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=1,
            batch=prefill_batch(100, [100]),
            running_bs=1,
        )
        assert "nctx=" not in lbl
