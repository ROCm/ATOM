# SPDX-License-Identifier: MIT
"""Kimi-K3 LMCache offload over the MP server: the KDA state rides the PAGE object.

The state at a boundary is one block per mamba group, and vLLM aliases mamba
layer ``j`` onto MLA plane ``j``, so it is PAGE units a transfer names as a
native-state image -- the mechanism DeepSeek-V4.1's checkpoints use.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector
from atom.plugin.vllm.kv_transfer.kda_ride import (
    KdaRidePlanner,
    kda_ride_geometry,
    kda_ride_page_order,
    publish_kda_ride_state,
    refuse_kda_ride_without_prerequisites,
)


def _planner(*, block=1536, chunk=1536):
    return KdaRidePlanner(
        group_ids=(1, 2, 3),
        mamba_block_size=block,
        hash_block_size=block,
        chunk_size=chunk,
    )


def test_a_hit_is_capped_to_the_chunk_grid_and_only_inside_its_lookup():
    """A save carries the state of the chunk edge it ends on, nothing finer;
    an untested hit would restore KV past the last state the tier holds."""
    p = _planner()
    p.begin_lookup(SimpleNamespace(request_id="r", num_tokens=1 << 20))
    assert p.cap_hit(SimpleNamespace(id="r"), 4000) == 3072
    assert p.cap_hit(SimpleNamespace(id="other"), 4000) == 0
    p.end_lookup()
    assert p.cap_hit(SimpleNamespace(id="r"), 4000) == 0


@pytest.mark.parametrize(
    "computed, group_blocks, expected",
    [
        (3072, ([9], [3, 4], [5, 6], [7, 8]), (4, 6, 8)),
        (3000, ([9], [3, 4], [5, 6], [7, 8]), None),  # not on a mamba block
        (3072, ([9], [3, 4], [5, 0], [7, 8]), None),  # one group has no block
    ],
)
def test_a_load_names_its_state_block_in_every_group_or_nothing(
    computed, group_blocks, expected
):
    """Row `(n - 1) // block`, the one `preprocess_mamba` reads, in all groups."""
    p = _planner()
    request = SimpleNamespace(request_id="r")
    assert p.resolve_load(request, computed, group_blocks=group_blocks) is (
        expected is not None
    )
    assert p._ride_boundaries.get("r", {}).get(computed) == expected


# -- image geometry and the worker's aliasing check --------------------------


def _mamba_spec(*, block, page, state):
    from vllm.v1.kv_cache_interface import MambaSpec

    spec = MambaSpec.__new__(MambaSpec)
    for name, value in {
        "block_size": block,
        "shapes": ((state,),),
        "dtypes": (torch.int8,),
        "page_size_padded": page,
        "num_speculative_blocks": 0,
    }.items():
        object.__setattr__(spec, name, value)
    return spec


def _kv_cache_config(*, attention, mamba, page, block=1536, state=832512, index=0):
    """vLLM's groups: MLA layers (each with a DSA index cache when *index* is
    its page bytes), then the mamba groups."""
    pages = {f"mla.{j}": page for j in range(attention)}
    if index:
        pages.update({f"mla.{j}.index_cache": index for j in range(attention)})
    groups = [
        SimpleNamespace(
            layer_names=list(pages),
            kv_cache_spec=SimpleNamespace(
                kv_cache_specs={
                    name: SimpleNamespace(page_size_bytes=size)
                    for name, size in pages.items()
                }
            ),
        )
    ]
    for g, layers in enumerate(mamba):
        groups.append(
            SimpleNamespace(
                layer_names=[f"kda.{g}.{j}" for j in range(layers)],
                kv_cache_spec=_mamba_spec(block=block, page=page, state=state),
            )
        )
    mamba_groups = [(1 + g, groups[1 + g].kv_cache_spec) for g in range(len(mamba))]
    return SimpleNamespace(kv_cache_groups=groups), mamba_groups


def test_k3s_image_is_three_units_over_the_leading_23_layers():
    pytest.importorskip("vllm")
    config, mamba = _kv_cache_config(attention=29, mamba=(23, 23, 23), page=884736)
    geometry = kda_ride_geometry(config, 0, mamba)
    assert geometry.state_layers == 23
    assert geometry.spec.page_unit_bytes == 23 * 884736
    assert geometry.spec.image_bytes == 3 * 23 * 884736
    assert geometry.spec.units_per_checkpoint == 3


_N, _PAGE, _STATE = 5, 64, 48  # blocks, page bytes, state bytes per block


def _pool(*, alias=True, index=0, extra_first=False):
    """vLLM's allocation in miniature: one buffer, every group layer-outer
    from byte 0, so mamba layer j shares attention layer j's bytes; index
    caches after the MLA planes, as vLLM lays out MLA with a DSA indexer.
    *extra_first* publishes the one layer no mamba layer aliases ahead of the
    others, as K3's layer names sort with DSpark; the ride's page order
    (`kda_ride_page_order`) puts the aliased layers back in front."""
    from atom.config import KVCacheTensor
    from atom.plugin.vllm.kv_transfer.mp_page_layout import build_mp_transfer_tensors

    buffer = torch.zeros(4 * _N * (_PAGE + index), dtype=torch.uint8)
    tensors = []
    for j in range(4):
        k = buffer[j * _N * _PAGE :][: _N * _PAGE].view(_N, _PAGE)
        at = 4 * _N * _PAGE + j * _N * index
        idx = buffer[at:][: _N * index].view(_N, index) if index else None
        tensors.append(KVCacheTensor(layer_num=j, k_cache=k, index_cache=idx))
    state_buffer = buffer if alias else torch.zeros_like(buffer)
    kda = [
        [
            torch.as_strided(
                state_buffer.view(torch.int8),
                (_N, 1, 1, _STATE),
                (_PAGE, _STATE, _STATE, 1),
                j * _N * _PAGE,
            )
            for j in range(3)
        ]
        for _ in range(2)
    ]
    if extra_first:
        tensors = [tensors[3], *tensors[:3]]
    if alias:
        tensors = kda_ride_page_order(tensors, kda)
    transfer = build_mp_transfer_tensors(tensors, num_blocks=_N)
    config, groups = _kv_cache_config(
        attention=4, mamba=(3, 3), page=_PAGE, block=16, state=_STATE, index=index
    )
    return transfer, kda, kda_ride_geometry(config, 0, groups)


@pytest.mark.parametrize(
    "index, extra_first",
    [(0, False), (16, False), (0, True)],
    ids=["mla", "mla+dsa_index", "unaliased_layer_sorts_first"],
)
def test_the_state_rides_as_the_native_layout_registers_it(index, extra_first):
    """Image unit ``j`` of the layout the worker registers holds mamba group
    ``j``'s state in the block that unit names; a DSA index plane between the
    MLA planes is carried along, and a layer no mamba layer aliases is moved
    out of the image's planes however it sorts."""
    pytest.importorskip("vllm")
    from atom.kv_transfer.offload.mp.native_state_layout import (
        build_native_state_mp_layout,
    )

    transfer, kda, geometry = _pool(index=index, extra_first=extra_first)
    per_layer = 2 if index else 1
    publish_kda_ride_state(transfer, kda, geometry)
    assert transfer.paged_state_region_count == 3 * per_layer
    assert transfer.native_state_tp_replication_factor == 1
    # One block pool: each group writes only the blocks vLLM gave it.
    owned = {0: (1, 3), 1: (2, 4)}
    for g, layers in enumerate(kda):
        for j, tensor in enumerate(layers):
            for block in owned[g]:
                tensor[block].fill_(10 * g + j + 1)
    layout = build_native_state_mp_layout(
        transfer, block_size=16, chunk_size=16, num_blocks=_N
    )
    units = (3, 2)  # group 0's block 3, group 1's block 2
    pages = len(transfer.pages)
    for ordinal, unit in enumerate(units):
        for j in range(3):
            alias = layout.tensors[pages + 3 * per_layer * ordinal + per_layer * j]
            got = alias[unit, 0, :_STATE]
            assert torch.equal(got.view(torch.int8), kda[ordinal][j][unit].flatten())
            assert int(got[0]) == 10 * ordinal + j + 1


def test_a_state_not_in_the_page_planes_is_refused():
    pytest.importorskip("vllm")
    transfer, kda, geometry = _pool(alias=False)
    with pytest.raises(ValueError, match="does not start where one"):
        publish_kda_ride_state(transfer, kda, geometry)
    assert getattr(transfer, "paged_state_checkpoint_spec", None) is None


# -- the MP scheduler --------------------------------------------------------


def test_a_save_carries_the_kda_state_at_its_end_and_pins_it():
    """Three units, one per mamba group, from the blocks vLLM cached under the
    boundary's hash; pinned until the save leaves the scheduler."""
    from atom.kv_transfer.disaggregation.types import ConnectorCompletion
    from atom.kv_transfer.offload.chunked_scheduler import (
        DENSE_PAGE_STORE_CHANNEL,
        ChunkedOffloadSchedulerBase,
    )
    from atom.kv_transfer.offload.metadata import NativeStateTransfer
    from atom.plugin.vllm.kv_transfer.ride_mp_scheduler import StateRideMPScheduler

    s = StateRideMPScheduler.__new__(StateRideMPScheduler)
    ChunkedOffloadSchedulerBase.__init__(
        s,
        SimpleNamespace(
            kv_cache_block_size=256, kv_transfer_config={"kv_role": "kv_both"}
        ),
        chunk_size=256,
        lookup_client=SimpleNamespace(
            prepare_retrieve=lambda *_a: None, clear_lookup_status=lambda _sid: None
        ),
    )
    s._init_plugin_state()
    s._mp_adapter = SimpleNamespace(end_session=lambda _session: None)
    s._scheduler_deadline_s = 3600.0
    s._transfer_seen_at = {}
    s._finished_frontiers = {}
    s._min_load_tokens = 0
    blocks = {b: SimpleNamespace(block_id=b, ref_cnt=1) for b in (41, 42, 43)}
    freed = []
    pool = SimpleNamespace(
        blocks=blocks,
        get_cached_block=lambda h, _g: (
            [blocks[b] for b in (41, 42, 43)] if h == b"h512" else None
        ),
        touch=lambda _blocks: None,
        free_blocks=lambda bs: freed.append(tuple(b.block_id for b in bs)),
    )
    planner = _planner(block=256, chunk=256)
    planner.bind_gpu_block_pool(pool)
    s.bind_planner(planner, lambda seq: seq._request)
    request = SimpleNamespace(
        request_id="r0", block_hashes=(b"h256", b"h512"), num_tokens=512
    )
    s.update_state_after_alloc(
        SimpleNamespace(
            id="r0",
            num_prompt_tokens=512,
            num_cached_tokens=512,
            token_ids=list(range(512)),
            block_table=[10, 11],
            _request=request,
        )
    )
    [save] = s.build_connector_meta().requests
    assert save.native_state == NativeStateTransfer(
        (41, 42, 43), 512, int.from_bytes(b"h512", "big")
    )
    assert planner._ride_pins == {save.save_operation: (41, 42, 43)}
    s.connector_completion(
        ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, save.save_operation, True)
    )
    s.build_connector_meta()
    assert planner._ride_pins == {} and freed == [(41, 42, 43)]


@pytest.mark.parametrize(
    "cache, policy, match",
    [
        ({"mamba_cache_mode": "all"}, "recompute", "mamba-cache-mode align"),
        ({"enable_prefix_caching": False}, "recompute", "enable-prefix-caching"),
        ({}, "fail", "kv_load_failure_policy='recompute'"),
    ],
)
def test_the_ride_refuses_a_deployment_it_cannot_serve(cache, policy, match):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            **{"mamba_cache_mode": "align", "enable_prefix_caching": True, **cache}
        ),
        kv_transfer_config=SimpleNamespace(kv_load_failure_policy=policy),
    )
    with pytest.raises(ValueError, match=match):
        refuse_kda_ride_without_prerequisites(config)


@pytest.mark.parametrize("kimi_k3, rides", [(True, True), (False, False)])
def test_only_kimi_k3s_mamba_state_rides_mp(kimi_k3, rides):
    """Qwen's GDN layers report K3's mamba type; only K3 is sized for the ride,
    and `__init__` refuses any other mamba model on MP."""
    stub = SimpleNamespace(
        _offload_backend="mp",
        _is_deepseek_v41=False,
        _v41_checkpoint_group_ids=(),
        _mamba_groups=[(1, None)],
        _is_kimi_k3=kimi_k3,
    )
    assert AtomLMCacheOffloadConnector._state_rides_mp.fget(stub) is rides
