from types import SimpleNamespace

import torch

from atom.plugin.sglang.attention_backend.attention_gdn import (
    SGLangGDNForwardContext,
)


class _DecodeMode:
    @staticmethod
    def is_target_verify():
        return False

    @staticmethod
    def is_decode_or_idle():
        return True

    @staticmethod
    def is_extend():
        return False


class _MambaPool:
    @staticmethod
    def get_mamba_indices(req_pool_indices):
        return req_pool_indices + 10


def test_build_gdn_metadata_reconstructs_missing_forward_metadata():
    pool = _MambaPool()
    forward_batch = SimpleNamespace(
        forward_mode=_DecodeMode(),
        batch_size=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        req_to_token_pool=pool,
    )
    linear_backend = SimpleNamespace(
        forward_metadata=None,
        req_to_token_pool=pool,
    )

    metadata = SGLangGDNForwardContext._build_gdn_metadata(
        forward_batch, linear_backend
    )

    assert metadata is not None
    assert torch.equal(
        metadata.non_spec_query_start_loc,
        torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    assert torch.equal(
        metadata.non_spec_state_indices_tensor,
        torch.tensor([10, 11], dtype=torch.int32),
    )
