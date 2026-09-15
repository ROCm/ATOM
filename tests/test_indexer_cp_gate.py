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
import torch

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
    ``--method eagle3`` -- so every arm anyone measured was a no-spec arm, and
    the feature looked marginal. The signature no longer even accepts a
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


# ───────────────────────────────────────────────────── compilation key ──


def _hash_with(indexer_dcp_only: bool) -> str:
    """``Config.compute_hash`` over a stand-in carrying only what it reads.

    Constructing a real ``Config`` needs a model directory, which this suite
    runs without. ``compute_hash`` is a pure function of the attributes it
    touches, so binding it to a namespace that supplies exactly those is a
    faithful call -- and it fails loudly (AttributeError) if the factor list
    ever grows a field this stub does not model, rather than silently drifting.
    """
    from types import SimpleNamespace

    from atom.config import Config

    stub = SimpleNamespace(
        quant_config=None,
        compilation_config=None,
        parallel_config=None,
        tensor_parallel_size=4,
        prefill_context_parallel_size=1,
        dcp_config=SimpleNamespace(indexer_dcp_only=indexer_dcp_only),
        enable_dp_attention=False,
        index_cache_dtype="fp8",
        hf_config=SimpleNamespace(),
    )
    return Config.compute_hash(stub)


def test_the_two_indexer_modes_do_not_share_a_compiled_artifact():
    """CP widens the fused QKV output, so it must key the compilation cache.

    ``index_q`` is this rank's one index head under TP and all of them under CP
    (``minimax_m3.py`` picks the width, ``linear.py`` allocates it), which
    changes the traced graph and the captured buffer strides. Two runs of the
    same model and source otherwise hash identically, so without this factor
    the second mode loads the first mode's artifact and trips
    ``assert_size_stride`` at runtime -- the same hazard the
    ``prefill_context_parallel_size`` and ``ATOM_REPLICATE_VOCAB_EMBED``
    factors beside it were added for.
    """
    assert _hash_with(True) != _hash_with(False)


def test_the_compilation_key_is_stable_for_one_mode():
    """Guard against keying on something unstable, which would never cache."""
    assert _hash_with(False) == _hash_with(False)
    assert _hash_with(True) == _hash_with(True)


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


PLUGIN_BRIDGES = ["vllm", "sglang", "sgl", "rtpllm"]


@pytest.mark.parametrize("bridge", PLUGIN_BRIDGES)
def test_bridges_without_a_cp_chain_are_disabled(framework, bridge):
    """Every bridge calls the TP kernel directly and issues no all-to-all.

    They must answer False BEFORE the config is consulted -- a config read would
    make the answer depend on a flag the operator can set, and the point of this
    gate is that they cannot.

    ``vllm`` leads the list because it is the one that was missing. The module
    docstring names the vLLM bridge as a TP-path host, ``is_plugin_mode``
    covers it, and ``plugin/vllm/attention/minimax_m3_attnetion.py`` is one of
    the two files that would ``view`` a widened index_q at the wrong width --
    so omitting it let exactly the regression the docstring warns about pass.
    """
    from atom.distributed.indexer_cp import indexer_cp_enabled

    framework(bridge)
    assert indexer_cp_enabled() is False


def test_every_plugin_framework_is_covered_by_this_table():
    """The truth table must track ``_SUPPORTED_FRAMEWORKS_FOR_PLUGIN_MODE``.

    Enumerating bridges by hand is how ``vllm`` went missing. A bridge added to
    that tuple later now fails here instead of silently entering the CP path.
    """
    from atom.plugin import prepare

    declared = {f.lower() for f in prepare._SUPPORTED_FRAMEWORKS_FOR_PLUGIN_MODE}
    missing = declared - set(PLUGIN_BRIDGES)
    assert not missing, f"plugin frameworks absent from the truth table: {missing}"


# ─────────────────────────────────────────────────────── exchange transport ──


class _CaComm:
    def __init__(self, disabled=False, accepts=True):
        self.disabled = disabled
        self._accepts = accepts

    def should_custom_ag(self, _tensor):
        return self._accepts


class _Group:
    """A TP group stub whose rank is a TRAP.

    Reading ``rank_in_group`` raises, which is the point: the transport
    predicate must not be able to depend on it. See
    ``_can_all_gather``'s docstring -- two ranks answering differently put one
    in ``custom_all_gather`` and the other in ``all_to_all_single`` on the same
    group, which deadlocks silently rather than failing.
    """

    def __init__(self, ca_comm):
        from types import SimpleNamespace

        self.device_communicator = SimpleNamespace(ca_comm=ca_comm)

    @property
    def rank_in_group(self):
        raise AssertionError("the transport decision must not depend on rank")


# A sentinel, so ``ca_comm=None`` can mean what it means in production -- the
# group has no custom-all-reduce comm at all -- rather than "use the default".
_AVAILABLE = object()


def _can(keys_bytes=1024, ca_comm=_AVAILABLE):
    from atom.distributed.indexer_cp import _can_all_gather

    keys = torch.zeros(keys_bytes // 8, dtype=torch.int64)
    comm = _CaComm() if ca_comm is _AVAILABLE else ca_comm
    return _can_all_gather(keys, _Group(comm))


def test_a_small_payload_takes_the_all_gather():
    assert _can() is True


def test_a_payload_over_the_threshold_takes_the_all_to_all():
    """Above the measured 512KB crossover the all-gather's 4x traffic dominates."""
    from atom.distributed.indexer_cp import _ALLGATHER_MAX_PAYLOAD_BYTES

    assert _can(keys_bytes=_ALLGATHER_MAX_PAYLOAD_BYTES + 8) is False
    assert _can(keys_bytes=_ALLGATHER_MAX_PAYLOAD_BYTES) is True


def test_the_repo_wide_custom_all_gather_opt_out_is_honoured(monkeypatch):
    """``ATOM_USE_CUSTOM_ALL_GATHER=0`` must reach this transport too.

    ``embed_head.py`` passes the same variable as ``use_custom``. Ignoring it
    here meant an operator who disabled aiter's custom gather everywhere else
    still got it in the indexer exchange, with no way to turn it off.
    """
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "0")
    assert _can() is False
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "1")
    assert _can() is True


@pytest.mark.parametrize(
    "ca_comm",
    [None, _CaComm(disabled=True), _CaComm(accepts=False)],
    ids=["absent", "disabled", "refused"],
)
def test_an_unavailable_custom_gather_falls_back_to_the_all_to_all(ca_comm):
    """Availability is decided, never discovered by catching an exception.

    Each of these is a state aiter's own assertions would have raised on. The
    all-to-all is correct at every size, so falling back is free -- but it has
    to be a decision every rank reaches identically, not a rescued failure.
    """
    assert _can(ca_comm=ca_comm) is False


def test_topology_gate_clears_the_flag_in_plugin_mode():
    """The gate, not only the runtime check, must reject plugin mode.

    ``indexer_cp_enabled`` returning False is enough for CORRECTNESS but not for
    honesty: with the flag still set, ``__post_init__`` logs "indexer_dcp_only
    enabled" and ``compute_hash`` keys on a mode the model never runs. That log
    line is what a CP-vs-TP A/B is diagnosed from, so a gate that leaves it
    saying the opposite of what ran is its own bug.
    """
    reason = indexer_cp_unsupported_reason(M3, 4, 4, 128, 1, False, True)
    assert reason is not None and "plugin" in reason
    # Native is still the supported case -- the new term defaults to off.
    assert indexer_cp_unsupported_reason(M3, 4, 4, 128, 1, False) is None
