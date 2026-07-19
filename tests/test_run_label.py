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
            bs=2,
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
            bs=64,
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
            bs=300,
            batch=decode_batch(300, d=300),
        )
        assert lbl.startswith("eager_decode[")

    def test_dummy_decode_distinct_from_real(self):
        dummy = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=True,
            tbo_on=False,
            bs=1,
            batch=decode_batch(1, d=1),
        )
        real = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            bs=1,
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
            bs=1,
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
            bs=1,
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
            bs=3,
            batch=prefill_batch(16384, [7000, 6000, 3384]),
        )
        assert lbl.endswith("tbo=1]")

    def test_spec_and_p_fields(self):
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            bs=128,
            batch=decode_batch(128, d=126, p=2, spec=3),
        )
        assert " p=2" in lbl and " d=126" in lbl and " spec=3" in lbl

    def test_ctx_truncation_many_seqs(self):
        lbl = build_run_label(
            is_prefill=True,
            use_cudagraph=False,
            is_dummy=False,
            tbo_on=False,
            bs=8,
            batch=prefill_batch(8000, list(range(8))),
        )
        assert "...+5" in lbl  # 8 seqs → first 3 shown + "...+5"

    def test_no_batch(self):
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            bs=16,
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
            bs=117,
            graph_bs=128,
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
            bs=128,
            graph_bs=128,
            batch=decode_batch(128, d=128),
        )
        assert lbl.startswith("decode[bs=128 ")
        assert "/" not in lbl

    def test_graph_bs_below_real_bs_omits_slash(self):
        # Guard is "graph padded ABOVE real" (graph_bs > bs). A graph_bs < bs
        # never happens in production (the cudagraph path always has
        # graph_bs >= bs), but if it did we keep the plain real bs rather than
        # emit a misleading "bs=128/64".
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            bs=128,
            graph_bs=64,
            batch=decode_batch(128, d=128),
        )
        assert lbl.startswith("decode[bs=128 ")
        assert "/" not in lbl

    def test_graph_bs_none_is_backward_compatible(self):
        # Omitting graph_bs keeps the legacy single-number form.
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            bs=64,
            batch=decode_batch(64, d=64),
        )
        assert lbl.startswith("decode[bs=64 ")

    def test_dummy_decode_padded(self):
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=True,
            tbo_on=False,
            bs=1,
            graph_bs=8,
            batch=decode_batch(1, d=1),
        )
        assert lbl.startswith("dummy_decode[bs=1/8 ")

    def test_graph_bs_ignored_on_eager_path(self):
        # graph_bs only applies to the CUDAGraph path; eager decode ignores it.
        lbl = build_run_label(
            is_prefill=False,
            use_cudagraph=False,
            is_dummy=False,
            tbo_on=False,
            bs=300,
            graph_bs=512,
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
            bs=117,
            graph_bs=128,
            batch=decode_batch(117, d=117),
        )
        assert int(re.search(r"bs=(\d+)", lbl).group(1)) == 117


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
