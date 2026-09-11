# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The two gates that decide whether MiniMax-M3 indexer CP runs.

They are separate on purpose and fail differently:

* ``indexer_cp_unsupported_reason`` is the TOPOLOGY gate. It answers "would the
  CP chain be correct here", and ``Config.__post_init__`` uses it to clear the
  flag with a warning -- a fallback, never a raise, because this is a
  performance feature.
* ``indexer_cp_enabled`` is the FRAMEWORK gate. It answers "does the host that
  is about to run the model implement the CP chain at all".

The framework gate earns a test because it is not a documentation nicety. Every
host -- native, vLLM, SGLang -- runs ATOM's own ``models/minimax_m3.py`` and
``model_ops/linear.py``, so the flag widens the fused index-Q projection
*underneath* whichever bridge is hosting it. A bridge that then reshapes on an
assumed width reads 4x the rows it should. SGLang is exactly that bridge today,
which is why it must keep answering False.
"""

import pytest

from atom.config import indexer_cp_unsupported_reason

M3 = ["MiniMaxM3SparseForCausalLM"]


# ─────────────────────────────────────────────────────── topology gate ──


@pytest.mark.parametrize(
    "arches, tp, kv_heads, block, dcp, tbo, expected",
    [
        # The supported case: TP4 on M3's 4 KV heads, 128-block, no DCP, no TBO.
        (M3, 4, 4, 128, 1, False, None),
        (["LlamaForCausalLM"], 4, 4, 128, 1, False, "not a MiniMax-M3 model"),
        # DCP is EXCLUSIVE with this feature, not a prerequisite: real DCP
        # shards the KV cache itself and M3 has no DCP-aware attention path.
        (M3, 4, 4, 128, 2, False, "decode_context_parallel_size > 1"),
        # Below the square case a rank holds >1 kv head, which both the
        # candidate merge and the gluon decode kernel reject.
        (M3, 2, 4, 128, 1, False, "!= num_key_value_heads"),
        # Above it the CP group is a strided subset of TP; not wired.
        (M3, 8, 4, 128, 1, False, "!= num_key_value_heads"),
        (M3, 4, 4, 64, 1, False, "sparse_block_size 64 != 128"),
        # Two ubatch threads issuing all-to-alls on one group with no ordering
        # discipline deadlock.
        (M3, 4, 4, 128, 1, True, "TBO"),
    ],
)
def test_topology_gate_truth_table(arches, tp, kv_heads, block, dcp, tbo, expected):
    reason = indexer_cp_unsupported_reason(arches, tp, kv_heads, block, dcp, tbo)
    if expected is None:
        assert reason is None
    else:
        assert reason is not None and expected in reason


def test_topology_gate_does_not_reject_speculative_decode():
    """Spec decode is the configuration this feature is FOR, not a blocker.

    The whole chain takes ``max_query_len`` as a runtime argument. An earlier
    revision rejected spec, which silently served the TP path under
    ``--method eagle3`` and left every measurable arm at <=25% MMA occupancy --
    making the feature look marginal. The signature no longer even accepts a
    speculative config, so re-coupling them takes a deliberate edit.
    """
    import inspect

    params = list(inspect.signature(indexer_cp_unsupported_reason).parameters)
    assert not any("spec" in p for p in params), (
        "the indexer-CP gate must not depend on speculative decoding; "
        f"got parameters {params}"
    )


def test_topology_gate_reasons_are_human_readable():
    """Every reason is logged verbatim, so it has to name its own cause."""
    assert "num_key_value_heads" in indexer_cp_unsupported_reason(
        M3, 2, 4, 128, 1, False
    )
    assert "decode_context_parallel_size" in indexer_cp_unsupported_reason(
        M3, 4, 4, 128, 2, False
    )


# ────────────────────────────────────────────────────── framework gate ──


@pytest.fixture
def framework():
    """Set the plugin framework for one test and restore it afterwards.

    ``_set_framework_backbone`` writes a module global that every later test in
    the process would otherwise inherit -- and reading it as "vllm" makes
    unrelated config code take the plugin branch.
    """
    from atom.plugin import prepare

    original = prepare._CURRENT_FRAMEWORK
    yield prepare._set_framework_backbone
    prepare._CURRENT_FRAMEWORK = original


@pytest.mark.parametrize("bridge", ["sglang", "sgl", "rtpllm"])
def test_bridges_without_a_cp_chain_are_disabled(framework, bridge):
    """SGLang and rtpllm call the TP kernel directly and issue no all-to-all.

    They must answer False BEFORE the config is consulted -- a config read would
    make the answer depend on a flag the operator can set, and the point of this
    gate is that they cannot.
    """
    from atom.distributed.indexer_cp import indexer_cp_enabled

    framework(bridge)
    assert indexer_cp_enabled() is False
