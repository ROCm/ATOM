"""The checkpoint-image group is not a recurrent cache to hand to the tier.

It is declared as a `MambaSpec` so vLLM allocates and prefix-caches blocks
for it, which is enough for `find_mamba_groups` to pick it up. Left in, the
recurrent leg built `KdaStore`s for it and fed them to the V4.1 leg, which
expects its own shape: `AttributeError: 'KdaStore' object has no attribute
'req_id'` on the first request, on every rank, engine down.
"""

from copy import copy
from types import SimpleNamespace

import pytest
from atom.plugin.vllm.kv_transfer.connector import _without_v41_image_groups
from atom.plugin.vllm.v41_state_image import expected_layer_names


def _cfg(*layer_name_lists):
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=list(names)) for names in layer_name_lists
        ]
    )


def _image_names(n=8):
    class _S:
        units = n

    return list(expected_layer_names(_S()))[:n]


def test_an_all_image_group_is_dropped():
    names = _image_names()
    cfg = _cfg(["model.layers.0.self_attn"], names)
    kept = _without_v41_image_groups([(1, "spec")], cfg)
    assert kept == []


def test_a_real_recurrent_group_is_kept():
    cfg = _cfg(["model.layers.0.self_attn"], ["model.layers.3.linear_attn"])
    kept = _without_v41_image_groups([(1, "spec")], cfg)
    assert kept == [(1, "spec")]


def test_a_mixed_group_is_kept():
    """Dropping a group that also holds real recurrent layers would silently
    stop offloading them. Only an all-image group is ours."""
    names = _image_names(2) + ["model.layers.3.linear_attn"]
    cfg = _cfg(["model.layers.0.self_attn"], names)
    kept = _without_v41_image_groups([(1, "spec")], cfg)
    assert kept == [(1, "spec")]


def test_no_config_leaves_the_groups_alone():
    kept = _without_v41_image_groups([(1, "spec")], SimpleNamespace())
    assert kept == [(1, "spec")]


def test_the_image_group_is_not_an_attention_group_either():
    """Dropping it from the recurrent leg alone moved it to the other side:
    `expected exactly one non-recurrent KV cache group, found [0, 1]`, and
    the engine refused to start. It is neither."""
    from atom.plugin.vllm.kv_transfer.connector import _v41_image_group_ids

    cfg = _cfg(["model.layers.0.self_attn"], _image_names())
    assert _v41_image_group_ids(cfg) == {1}


def test_a_model_without_images_has_none():
    cfg = _cfg(["model.layers.0.self_attn"], ["model.layers.3.linear_attn"])
    from atom.plugin.vllm.kv_transfer.connector import _v41_image_group_ids

    assert _v41_image_group_ids(cfg) == set()


def test_the_proxy_group_is_claimed_by_name_not_by_elimination():
    """The codec registers the proxy's planes, so the proxy's group is the
    one it owns. Resolving it by eliminating everything else needs a new
    clause for every group anyone adds: DSpark's draft KV was the second
    one and the engine refused to start with `found [0, 1]`.

    Also pins the ordering. `_is_deepseek_v41` is a class attribute
    defaulting to False, and it used to be assigned AFTER this resolution
    ran -- which made the branch below dead without raising anything.
    """
    from atom.plugin.vllm.deepseek_v41_bridge import (
        ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME,
    )
    from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector

    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._is_deepseek_v41 = True
    c._mamba_groups = []
    cfg = _cfg(["draft.layers.0.self_attn"], [ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME])
    assert c._resolve_attention_group(cfg) == 1

    cfg = _cfg([ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME], ["draft.layers.0.self_attn"])
    assert c._resolve_attention_group(cfg) == 0


def test_a_v41_run_with_no_proxy_group_says_so():
    from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector

    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._is_deepseek_v41 = True
    c._mamba_groups = []
    cfg = _cfg(["a.self_attn"], ["b.self_attn"])
    try:
        c._resolve_attention_group(cfg)
    except ValueError as exc:
        assert "proxy layer" in str(exc)
    else:
        raise AssertionError("a V4.1 run with no proxy group was accepted")


def test_the_scheduler_is_told_the_block_size_of_the_group_it_strides():
    """`cache_config.block_size` is engine-wide and becomes the SMALLEST
    group's once a second group exists, but every block table this
    scheduler indexes belongs to the attention group.

    With DSpark the draft's 32 replaced the proxy's 256 and both arithmetics
    that read it went wrong the same way: a 1345-token prompt's six blocks
    priced at 192 tokens, so the save frontier floored to zero chunks, and
    the backing check then called six blocks too few to cover [0, 1280).
    Neither raised anything; the tier was simply never written.
    """
    from types import SimpleNamespace

    from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector

    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._attn_block_size = 256
    c._scheduler = SimpleNamespace(block_size=32, virtual_block_size=32)
    c._reconcile_scheduler_block_size()
    assert c._scheduler.block_size == 256
    assert c._scheduler.virtual_block_size == 256

    # The arithmetic that was wrong, as arithmetic.
    blocks, chunk = 6, 256
    covered = blocks * c._scheduler.block_size
    assert covered == 1536
    assert (min(1345, covered) // chunk) * chunk == 1280  # was 0
    assert blocks >= -(-1280 // c._scheduler.virtual_block_size)  # was False


def test_the_dcp_factor_is_carried_not_dropped():
    from types import SimpleNamespace

    from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector

    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._attn_block_size = 256
    c._scheduler = SimpleNamespace(block_size=32, virtual_block_size=64)
    c._reconcile_scheduler_block_size()
    assert c._scheduler.virtual_block_size == 512


def test_a_single_group_run_is_left_alone():
    """No second group, so the engine-wide value is already the right one
    and this must not touch it."""
    from types import SimpleNamespace

    from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector

    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._attn_block_size = 256
    c._scheduler = SimpleNamespace(block_size=256, virtual_block_size=256)
    c._reconcile_scheduler_block_size()
    assert c._scheduler.block_size == 256


@pytest.mark.parametrize("backend", ["inproc", "mp"])
@pytest.mark.parametrize("role_name", ["SCHEDULER", "WORKER"])
@pytest.mark.parametrize("proxy_first", [False, True])
@pytest.mark.parametrize("engine_block_size", [32, 256])
@pytest.mark.parametrize("layout", ["v41-draft", "hybrid", "single"])
def test_target_geometry_is_resolved_before_transport_construction(
    monkeypatch, backend, role_name, proxy_first, engine_block_size, layout
):
    """Changing scheduler arithmetic after construction leaves MP keys stale.

    Record the real PAGE namespace at the transport constructor boundary:
    the worker publishes 256-token PAGE blocks, not the draft's global 32.
    No transport is opened and no GPU is initialized.
    """
    import sys
    from types import ModuleType

    from atom.kv_transfer.offload.config import build_page_namespace
    from atom.plugin.vllm.deepseek_v41_bridge import (
        ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME,
    )
    from atom.plugin.vllm.kv_transfer import connector as mod

    cls = mod.AtomLMCacheOffloadConnector
    monkeypatch.setattr(mod.KVConnectorBase_V1, "__init__", lambda *a, **kw: None)
    monkeypatch.setattr(cls, "_init_kda_planner", lambda *a: None)
    monkeypatch.setattr(cls, "_init_v41_planner", lambda *a: None)
    monkeypatch.setenv("ATOM_V41_ALLOW_SPECULATIVE_OFFLOAD", "1")
    patch = ModuleType("atom.plugin.vllm.kv_transfer.hybrid_invalid_blocks_patch")
    patch.apply_vllm_hybrid_invalid_blocks_patch = lambda: None
    monkeypatch.setitem(sys.modules, patch.__name__, patch)

    captured = []
    chunks = SimpleNamespace(chunk_size=256)

    def transport(config):
        captured.append(
            (config.kv_cache_block_size, build_page_namespace(config, chunks, 4))
        )
        return SimpleNamespace(
            block_size=config.kv_cache_block_size,
            virtual_block_size=config.kv_cache_block_size,
        )

    for path, names in (
        ("atom.kv_transfer.offload.mp.connector", ["LMCacheMPConnector"]),
        ("atom.kv_transfer.offload.mp.scheduler", ["LMCacheMPConnectorScheduler"]),
        (
            "atom.kv_transfer.offload.dense.connector",
            ["DenseOffloadConnector", "DenseOffloadScheduler"],
        ),
    ):
        module = ModuleType(path)
        for name in names:
            setattr(module, name, transport)
        monkeypatch.setitem(sys.modules, path, module)

    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=engine_block_size, cache_dtype="auto"),
        model_config=SimpleNamespace(
            model="test-v41",
            dtype="bfloat16",
            hf_config=SimpleNamespace(
                model_type="deepseek_v41" if layout == "v41-draft" else "test"
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=4, world_size=4),
        speculative_config=SimpleNamespace(method="dspark", num_speculative_tokens=5),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config={"atom.offload.backend": backend},
        ),
    )
    draft = SimpleNamespace(
        layer_names=["draft.layers.0.self_attn"],
        kv_cache_spec=SimpleNamespace(block_size=32),
    )
    proxy = SimpleNamespace(
        layer_names=[ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME],
        kv_cache_spec=SimpleNamespace(block_size=256),
    )
    groups = [proxy, draft] if proxy_first else [draft, proxy]
    expected_group = 0 if proxy_first else 1
    expected_block_size = 256
    if layout == "hybrid":
        # Exercise the non-V4.1 resolver without depending on MambaSpec's API.
        monkeypatch.setattr(
            mod,
            "find_mamba_groups",
            lambda _: [(1 - expected_group, draft.kv_cache_spec)],
        )
    elif layout == "single":
        groups = [proxy]
        proxy.kv_cache_spec.block_size = engine_block_size
        expected_group = 0
        expected_block_size = engine_block_size
    c = cls(
        cfg,
        getattr(mod.KVConnectorRole, role_name),
        SimpleNamespace(kv_cache_groups=groups),
    )
    registered = copy(c._config)
    registered.kv_cache_block_size = expected_block_size
    assert captured == [
        (expected_block_size, build_page_namespace(registered, chunks, 4))
    ]
    assert c._config.kv_cache_block_size == expected_block_size
    assert cfg.cache_config.block_size == engine_block_size
    assert draft.kv_cache_spec.block_size == 32
    assert c._attn_group_id == expected_group


def test_the_reconcile_runs_after_the_scheduler_exists():
    """It early-returns when there is no scheduler, so calling it before one
    is built makes it a no-op that logs nothing. That is the second bug of
    this exact shape in this file -- `_is_deepseek_v41` was read by a
    resolution that ran before its assignment -- so the ordering is pinned
    here rather than left to the next reader to notice."""
    import inspect

    from atom.plugin.vllm.kv_transfer import connector as mod

    src = inspect.getsource(mod.AtomLMCacheOffloadConnector.__init__)
    build = src.index("self._scheduler = ")
    call = src.index("self._reconcile_scheduler_block_size()")
    assert call > build, (
        "_reconcile_scheduler_block_size must run after a scheduler exists; "
        "before it, it returns silently and changes nothing"
    )
