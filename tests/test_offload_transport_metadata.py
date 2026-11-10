# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MP wire metadata remains usable after IPC without changing other transports."""

import array
import pickle
import subprocess
import sys

import pytest

from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata
from atom.kv_transfer.offload.mp.metadata import MPOffloadMetadata, MPReqMeta
from atom.kv_transfer.offload.mp.token_shipping import (
    ShippedTokens,
    TokenPrefix,
    WorkerTokens,
)


def test_mp_metadata_roundtrip_and_drops_only_work():
    metadata = MPOffloadMetadata()
    metadata.add_request(
        MPReqMeta(7, ShippedTokens(0, 3, array.array("i", [11, 12, 13])), [])
    )
    wire = pickle.loads(pickle.dumps(metadata))
    copies = WorkerTokens()
    copies.resolve(wire)
    assert isinstance(wire.requests[0].token_ids, TokenPrefix)
    assert wire.requests[0].token_ids.tolist() == [11, 12, 13]
    later = MPOffloadMetadata()
    later.add_request(MPReqMeta(7, ShippedTokens(3, 4, array.array("i", [14])), []))
    copies.resolve(pickle.loads(pickle.dumps(later)))
    assert wire.requests[0].token_ids.tolist() == [11, 12, 13]
    drop = MPOffloadMetadata()
    drop.token_drops = (7,)
    assert drop.has_work()
    copies.resolve(pickle.loads(pickle.dumps(drop)))
    assert not copies._held
    assert not hasattr(LMCacheOffloadMetadata(), "token_drops")
    assert not LMCacheOffloadMetadata().has_work()


def test_mp_missing_generation_is_rejected_before_any_token_reader():
    metadata = MPOffloadMetadata()
    metadata.add_request(MPReqMeta(7, ShippedTokens(3, 4, array.array("i", [14])), []))
    with pytest.raises(RuntimeError, match="holds 0 tokens"):
        WorkerTokens().resolve(pickle.loads(pickle.dumps(metadata)))


def test_state_capability_does_not_import_gpu_codecs():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from atom.kv_transfer.offload.capabilities import "
                "replays_state; assert not replays_state(None); "
                "assert 'torch' not in sys.modules; "
                "assert 'atom.kv_transfer.offload.page_codec' not in sys.modules"
            ),
        ],
        check=True,
        timeout=10,
    )


@pytest.mark.parametrize("bad", ["missing_pages", "slots", "zero_blocks"])
def test_page_codec_validates_shared_transfer_geometry(bad):
    from atom.kv_transfer.disaggregation.types import (
        KVTransferRegion,
        KVTransferTensors,
        PageRegion,
    )
    from atom.kv_transfer.offload.page_codec import pure_page_codec

    region = KVTransferRegion(0, 128, 16)
    tensors = KVTransferTensors(pages=[PageRegion(region)])
    if bad == "missing_pages":
        tensors.pages.clear()
        match = "non-empty block_regions"
    elif bad == "slots":
        tensors.slot_regions.append(region)
        match = "empty SLOT"
    else:
        match = "positive num_blocks"
    with pytest.raises(ValueError, match=match):
        pure_page_codec(tensors)
