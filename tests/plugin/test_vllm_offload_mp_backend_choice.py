"""How the connector picks an offload transport, and what it refuses.

Exercised unbound, off a stub, because everything either reads is the option
bag and the group lists. Constructing a real connector would drag in a vLLM
config, a model and a device for no added coverage.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector

_resolve_backend = AtomLMCacheOffloadConnector._resolve_offload_backend
_rides = AtomLMCacheOffloadConnector._state_rides_mp.fget
_refuse_v41 = AtomLMCacheOffloadConnector._refuse_v41_without_ride


def _stub(*, mamba_groups=(), v41=False, checkpoint_groups=(), **options):
    stub = SimpleNamespace(
        _config=SimpleNamespace(kv_transfer_config=dict(options)),
        _mamba_groups=list(mamba_groups),
        _is_deepseek_v41=v41,
        _v41_checkpoint_group_ids=tuple(checkpoint_groups),
    )
    stub._offload_backend = _resolve_backend(stub)
    return stub


def test_backend_defaults_to_inproc():
    assert _resolve_backend(_stub()) == "inproc"


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="must be 'inproc' or 'mp'"):
        _stub(**{"atom.offload.backend": "zmq"})


def test_backend_is_case_and_space_insensitive():
    assert _resolve_backend(_stub(**{"atom.offload.backend": " MP "})) == "mp"


@pytest.mark.parametrize(
    "option",
    [
        "lmcache.mp.state_transport",
        "lmcache.mp.state_cpu_size_gb",
        "atom.offload.v41.state_interval",
    ],
)
def test_a_retired_state_pool_option_is_refused_not_ignored(option):
    with pytest.raises(ValueError, match="no longer"):
        _stub(**{"atom.offload.backend": "mp", option: 1})


def test_mp_is_allowed_for_a_model_with_no_recurrent_group():
    stub = _stub(**{"atom.offload.backend": "mp"})
    assert stub._offload_backend == "mp"
    assert not _rides(stub)


def test_inproc_hybrid_keeps_its_state_in_process():
    assert not _rides(_stub(mamba_groups=[(1, object())]))


def test_mp_hybrid_rides_the_page_object():
    stub = _stub(mamba_groups=[(1, object())], **{"atom.offload.backend": "mp"})
    assert _rides(stub)


def test_v41_rides_its_checkpoint_groups():
    stub = _stub(v41=True, checkpoint_groups=(1, 2), **{"atom.offload.backend": "mp"})
    _refuse_v41(stub)
    assert _rides(stub)


def test_v41_refuses_the_in_process_backend():
    stub = _stub(v41=True, checkpoint_groups=(1, 2))
    with pytest.raises(ValueError, match="LMCache MP server only"):
        _refuse_v41(stub)


def test_v41_refuses_mp_without_checkpoint_groups():
    stub = _stub(v41=True, **{"atom.offload.backend": "mp"})
    with pytest.raises(ValueError, match="state checkpoint groups"):
        _refuse_v41(stub)


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
