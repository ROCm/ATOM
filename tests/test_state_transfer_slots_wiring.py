# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2027, Advanced Micro Devices, Inc. All rights reserved.

"""Who sizes the state transfer slots, and who fills them (needs aiter)."""

import types
from unittest import mock

import pytest
import torch

gdn_attn = pytest.importorskip("atom.model_ops.attentions.gdn_attn")
model_runner = pytest.importorskip("atom.model_engine.model_runner")


def builder(kv_transfer_config):
    config = types.SimpleNamespace(kv_transfer_config=kv_transfer_config)
    return types.SimpleNamespace(model_runner=types.SimpleNamespace(config=config))


@pytest.mark.parametrize(
    "cfg, slots",
    [
        (
            {
                "kv_connector": "moriio",
                "kv_role": "kv_producer",
                "state_transfer_slots": 8,
            },
            8,
        ),
        ({"kv_connector": "moriio", "state_transfer_slots": 8}, 8),
        (
            {
                "kv_connector": "moriio",
                "kv_role": "kv_consumer",
                "state_transfer_slots": 8,
            },
            0,
        ),
        (
            {
                "kv_connector": "mooncake",
                "kv_role": "kv_producer",
                "state_transfer_slots": 8,
            },
            0,
        ),
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {
                        "kv_connector": "moriio",
                        "kv_role": "kv_producer",
                        "state_transfer_slots": 8,
                    },
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                ],
            },
            0,
        ),
        ({"kv_connector": "moriio", "kv_role": "kv_producer"}, 0),
        ({}, 0),
    ],
)
def test_only_a_moriio_producer_sizes_transfer_slots(cfg, slots):
    assert gdn_attn.GDNStateMixin.state_transfer_slots(builder(cfg)) == slots


@pytest.mark.parametrize("decode_images", [True, False])
def test_transfer_slots_come_on_top_of_the_decode_image_spares(decode_images):
    """Both are extra entries of the state pool; neither takes the other's."""
    builder = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(
            config=types.SimpleNamespace(
                enable_prefix_caching=decode_images,
                state_checkpoint_interval_tokens=8192,
            )
        ),
        state_transfer=lambda: types.SimpleNamespace(
            decode_keep=gdn_attn.KEEP_SNAPSHOT
        ),
        _state_shape_for_runner=lambda: ((3, 16), (2, 4, 4)),
        _state_dtypes=lambda: (torch.bfloat16, torch.bfloat16),
        num_state_layers=lambda: 2,
        _replayssm_bytes_per_slot=lambda: 0,
        slots_per_req=lambda: 1,
        state_transfer_slots=lambda: 8,
    )
    spec = gdn_attn.GDNStateMixin.state_spec(builder)
    spares = gdn_attn.DECODE_SNAPSHOT_SCRATCH_SLOTS if decode_images else 0
    assert spec.extra_entries == spares + 8


def test_the_runner_copies_staged_state_before_the_connector_sees_it():
    calls = []
    runner = types.SimpleNamespace(
        attn_metadata_builder=types.SimpleNamespace(
            relocate_state_slots=lambda pairs: calls.append(("copy", pairs))
        )
    )
    connector = types.SimpleNamespace(
        start_load_kv=lambda meta: calls.append(("connector", meta))
    )
    meta = types.SimpleNamespace(state_stages={9: (1, 4), 10: (2, 5)})
    with mock.patch.object(model_runner, "get_kvconnector", return_value=connector):
        model_runner.ModelRunner.process_kvconnector_output(runner, meta)
    assert calls == [("copy", [(1, 4), (2, 5)]), ("connector", meta)]
