"""How the connector picks an offload transport, and what ``mp`` demands first.

Both resolvers guard the same fact: LMCache's multiprocess registration
addresses every KV cache group by paged block, so a hybrid model's per-slot
recurrent state cannot ride along as one more group. Under ``mp`` it gets an
LMCache engine and a host pool of its own, which is a second allocation the
operator has to know about -- hence an option with no default rather than a
sized-for-you pool.

Exercised unbound, off a stub, because everything either reads is the option
bag and the mamba group list. Constructing a real connector would drag in a
vLLM config, a model and a device for no added coverage.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector

_resolve_backend = AtomLMCacheOffloadConnector._resolve_offload_backend
_resolve_state_size = AtomLMCacheOffloadConnector._resolve_state_cpu_size_gb


def _stub(*, mamba_groups=(), **options):
    stub = SimpleNamespace(
        _config=SimpleNamespace(kv_transfer_config=dict(options)),
        _mamba_groups=list(mamba_groups),
        _state_cpu_size_gb=0.0,
    )
    # Mirrors __init__'s order: the size is resolved first, and the backend
    # refusal reads the resolved value. Tests that feed a bad size call the
    # resolver directly, so a stub that cannot be built for them would hide
    # the very error they assert on.
    try:
        stub._state_cpu_size_gb = _resolve_state_size(stub)
    except ValueError:
        pass
    return stub


def test_backend_defaults_to_inproc():
    assert _resolve_backend(_stub()) == "inproc"


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="must be 'inproc' or 'mp'"):
        _resolve_backend(_stub(**{"atom.offload.backend": "zmq"}))


def test_backend_is_case_and_space_insensitive():
    assert _resolve_backend(_stub(**{"atom.offload.backend": " MP "})) == "mp"


def test_mp_is_allowed_for_a_model_with_no_recurrent_group():
    assert _resolve_backend(_stub(**{"atom.offload.backend": "mp"})) == "mp"


def test_inproc_hybrid_needs_no_state_pool_size():
    # The in-process path shares the one pool, so there is no second
    # allocation to declare.
    stub = _stub(mamba_groups=[(1, object())])
    assert _resolve_backend(stub) == "inproc"


def test_mp_hybrid_without_a_state_pool_size_is_refused():
    stub = _stub(mamba_groups=[(1, object())], **{"atom.offload.backend": "mp"})

    with pytest.raises(ValueError, match="lmcache.mp.state_cpu_size_gb"):
        _resolve_backend(stub)


def test_mp_hybrid_with_a_state_pool_size_is_allowed():
    stub = _stub(
        mamba_groups=[(1, object())],
        **{"atom.offload.backend": "mp", "lmcache.mp.state_cpu_size_gb": 8},
    )

    assert _resolve_backend(stub) == "mp"


def test_state_pool_size_defaults_to_zero_meaning_unset():
    assert _resolve_state_size(_stub()) == 0.0


def test_state_pool_size_accepts_a_string_of_digits():
    # Every option arrives through a CLI JSON blob, so a quoted number is the
    # normal spelling, not an edge case.
    stub = _stub(**{"lmcache.mp.state_cpu_size_gb": "12.5"})
    assert _resolve_state_size(stub) == 12.5


def test_non_numeric_state_pool_size_is_refused():
    stub = _stub(**{"lmcache.mp.state_cpu_size_gb": "big"})

    with pytest.raises(ValueError, match="must be a number of GiB"):
        _resolve_state_size(stub)


def test_negative_state_pool_size_is_refused():
    stub = _stub(**{"lmcache.mp.state_cpu_size_gb": -1})

    with pytest.raises(ValueError, match="must not be negative"):
        _resolve_state_size(stub)


_replication = AtomLMCacheOffloadConnector._page_tp_replication_factor


def _layout_stub(spec, *, tp=4, group_id=0, num_groups=1):
    groups = [SimpleNamespace(kv_cache_spec=None) for _ in range(num_groups)]
    groups[group_id] = SimpleNamespace(kv_cache_spec=spec)
    return SimpleNamespace(
        _kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        _attn_group_id=group_id,
        _config=SimpleNamespace(tensor_parallel_size=tp),
    )


def _mla_spec():
    """An ``MLAAttentionSpec`` without its constructor's arguments.

    The method asks ``isinstance``, nothing else, and the real dataclass wants
    a full cache geometry that has no bearing on the answer.
    """
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    return object.__new__(MLAAttentionSpec)


def _full_spec():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return object.__new__(FullAttentionSpec)


def test_mla_group_is_declared_replicated_across_the_tp_group():
    assert _replication(_layout_stub(_mla_spec(), tp=4)) == 4


def test_non_mla_group_stays_sharded():
    # vLLM splits KV heads across TP, so the ranks' bytes differ.
    assert _replication(_layout_stub(_full_spec(), tp=4)) == 1


def test_replication_reads_the_attention_group_not_group_zero():
    # On a hybrid model group 0 can be the recurrent one; answering from it
    # would describe a cache this layout does not carry.
    stub = _layout_stub(_mla_spec(), tp=8, group_id=1, num_groups=2)
    assert _replication(stub) == 8


def test_no_groups_is_the_sharded_answer():
    stub = SimpleNamespace(
        _kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        _attn_group_id=0,
        _config=SimpleNamespace(tensor_parallel_size=4),
    )
    assert _replication(stub) == 1


def test_tp1_replication_is_one_even_for_mla():
    assert _replication(_layout_stub(_mla_spec(), tp=1)) == 1


def _merged_spec(members):
    """A ``UniformTypeKVCacheSpecs`` carrying `members`, built like `_mla_spec`."""
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    merged = object.__new__(UniformTypeKVCacheSpecs)
    object.__setattr__(
        merged,
        "kv_cache_specs",
        {f"layer.{i}": member for i, member in enumerate(members)},
    )
    return merged


def test_merged_all_mla_group_is_replicated():
    # GLM-5.3's shape: latent layers and DSA indexer layers merged into one
    # spec because their page sizes differ.
    stub = _layout_stub(_merged_spec([_mla_spec(), _mla_spec()]), tp=4)
    assert _replication(stub) == 4


def test_merged_group_with_one_sharded_member_stays_sharded():
    stub = _layout_stub(_merged_spec([_mla_spec(), _full_spec()]), tp=4)
    assert _replication(stub) == 1


def test_empty_merged_group_stays_sharded():
    assert _replication(_layout_stub(_merged_spec([]), tp=4)) == 1
