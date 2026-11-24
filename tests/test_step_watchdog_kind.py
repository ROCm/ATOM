# SPDX-License-Identifier: MIT
"""The step watchdog names a long forward by its kind: Mono kernels run only in
decode-only steps, so a long prefill or mixed forward must not read as a Mono stall."""
from types import SimpleNamespace

from atom.model_engine.engine_core import EngineCore


def _batch(prefill, decode, blocked=False):
    return SimpleNamespace(
        total_seqs_num_prefill=prefill, total_seqs_num_decode=decode, mono_blocked=blocked
    )


def test_forward_kind():
    assert EngineCore._forward_kind(_batch(1, 0)) == "prefill"
    assert EngineCore._forward_kind(_batch(9, 0)) == "prefill"
    assert EngineCore._forward_kind(_batch(2, 3)) == "mixed"
    assert EngineCore._forward_kind(_batch(0, 1)) == "decode"
    assert EngineCore._forward_kind(_batch(0, 4, blocked=True)) == "decode (Mono off)"
