import random
from types import SimpleNamespace

import torch


class _WeakConfig:
    pass


from atom.distributed.pcp_utils import (
    pcp_build_token_row_map,
    pcp_owned_request_rows,
    pcp_sparse_prefill_reindex,
    pcp_split_true_decodes_and_context,
)


def test_mixed_request_major_rows_share_one_explicit_mapping():
    # Two one-token decode requests followed by variable-length prefills.
    query_lens = torch.tensor([1, 1, 3, 2, 5], dtype=torch.int32)
    query_start = torch.cat(
        [torch.zeros(1, dtype=torch.int32), query_lens.cumsum(0)]
    )
    num_tokens = int(query_start[-1])
    req_ids = torch.repeat_interleave(
        torch.arange(query_lens.numel(), dtype=torch.int32), query_lens
    )
    sparse_lens = torch.arange(1, num_tokens + 1, dtype=torch.int32)
    slots = torch.arange(100, 100 + num_tokens, dtype=torch.int64)

    for rank in range(4):
        row_map = pcp_build_token_row_map(num_tokens, 4, rank)
        main = pcp_sparse_prefill_reindex(
            sparse_lens, req_ids, slots, 2048, row_map=row_map
        )

        assert torch.equal(main["owned_q"].cpu(), row_map.owned_global_rows)
        real_rows = row_map.owned_global_rows[
            row_map.owned_global_rows < num_tokens
        ]
        assert torch.equal(
            main["req_id_per_token"][: real_rows.numel()], req_ids[real_rows]
        )

        # Decode request rows and each prefill request range are slices of the same
        # compact local ordering used by main sparse metadata.
        ranges = [(0, 2)] + [
            (int(query_start[i]), int(query_start[i + 1]))
            for i in range(2, query_lens.numel())
        ]
        rebuilt = []
        for start, end in ranges:
            local_start, local_end = row_map.local_range(start, end)
            rows = row_map.owned_global_rows[local_start:local_end]
            assert torch.all((rows >= start) & (rows < end))
            rebuilt.extend(rows.tolist())
        assert rebuilt == real_rows.tolist()


def test_real_owned_rows_exclude_cp_padding_before_physical_conversion():
    num_tokens = 2049
    for rank in range(4):
        row_map = pcp_build_token_row_map(num_tokens, 4, rank)
        real_owned = row_map.owned_global_rows[
            row_map.owned_global_rows < num_tokens
        ]
        assert torch.all(real_owned < num_tokens)
        assert real_owned.tolist() == list(range(rank, num_tokens, 4))
        assert real_owned.numel() in (512, 513)


def test_plugin_cp_owner_key_survives_registry_reconstruction():
    from atom.plugin.vllm.attention import cp_gather

    hf = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"], num_hidden_layers=78
    )
    first = _WeakConfig()
    first.hf_config = hf
    first.model = "/models/glm-5.2"
    compiled_key = cp_gather.stable_cp_owner_key(first)

    # Simulate a new process loading a compiled graph containing compiled_key.
    rebuilt = _WeakConfig()
    rebuilt.hf_config = hf
    rebuilt.model = "/models/glm-5.2"
    assert cp_gather.stable_cp_owner_key(rebuilt) == compiled_key

    # Same static role intentionally replaces the prior model's registry entries.
    second = _WeakConfig()
    second.hf_config = hf
    second.model = "/models/glm-5.2"
    assert cp_gather.stable_cp_owner_key(second) == compiled_key

    # Target and draft instances remain isolated even for the same checkpoint.
    draft = _WeakConfig()
    draft.hf_config = hf
    draft.model = "/models/glm-5.2"
    draft._vllm_cp_model_role = "draft"
    assert cp_gather.stable_cp_owner_key(draft) != compiled_key


def test_plugin_cp_owner_key_accepts_unhashable_config():
    from atom.plugin.vllm.attention.cp_gather import stable_cp_owner_key

    class UnhashableConfig:
        __hash__ = None

    config = UnhashableConfig()
    config.hf_config = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"], num_hidden_layers=78
    )
    config.model = "/models/glm-5.2"
    assert stable_cp_owner_key(config) == stable_cp_owner_key(config)
    assert "role=target" in stable_cp_owner_key(config)


def test_vllm_decode_bucket_with_multitoken_prefill_is_reclassified():
    # Real crash shape: upstream thresholding placed a qlen>1 prefilling row in
    # its decode bucket. Plugin CP must retain only the true generated-token prefix.
    starts = torch.tensor([0, 1, 2, 4, 9], dtype=torch.int32)
    is_prefilling = torch.tensor([False, False, True, True])
    assert pcp_split_true_decodes_and_context(starts, is_prefilling) == (
        2,
        2,
        2,
        False,
    )


def test_multitoken_nonprefill_is_spec_verify_not_context_extend():
    starts = torch.tensor([0, 1, 4, 9], dtype=torch.int32)
    is_prefilling = torch.tensor([False, False, True])
    assert pcp_split_true_decodes_and_context(starts, is_prefilling) == (
        1,
        1,
        2,
        True,
    )


def test_four_phase_mixed_token_to_request_rows():
    # decode -> short extend -> long extend -> first prefill
    query_lens = torch.tensor([1, 1, 2, 5, 3], dtype=torch.int32)
    starts = torch.cat([torch.zeros(1, dtype=torch.int32), query_lens.cumsum(0)])
    request_segments = ((0, 2), (2, 3), (3, 4), (4, 5))

    for rank in range(4):
        row_map = pcp_build_token_row_map(int(starts[-1]), 4, rank)
        rebuilt = []
        for request_start, request_end in request_segments:
            tokens, requests = pcp_owned_request_rows(
                row_map, starts, request_start, request_end
            )
            expected = torch.searchsorted(starts[1:], tokens, right=True)
            assert torch.equal(requests, expected)
            assert torch.all(
                (requests >= request_start) & (requests < request_end)
            )
            rebuilt.extend(tokens.tolist())
        expected_real = row_map.owned_global_rows[
            row_map.owned_global_rows < int(starts[-1])
        ]
        assert rebuilt == expected_real.tolist()


def test_token_row_map_property_random_variable_prefills():
    rng = random.Random(20260731)
    for _ in range(100):
        cp_size = rng.choice((2, 4, 8))
        num_decodes = rng.randrange(0, 9)
        prefill_lens = [rng.randrange(2, 40) for _ in range(rng.randrange(1, 9))]
        query_lens = [1] * num_decodes + prefill_lens
        starts = [0]
        for length in query_lens:
            starts.append(starts[-1] + length)
        num_tokens = starts[-1]

        all_owned = []
        for rank in range(cp_size):
            row_map = pcp_build_token_row_map(num_tokens, cp_size, rank)
            all_owned.extend(row_map.owned_global_rows.tolist())

            compact_ranges = []
            for start, end in zip(starts, starts[1:]):
                local_start, local_end = row_map.local_range(start, end)
                compact_ranges.extend(
                    row_map.owned_global_rows[local_start:local_end].tolist()
                )
            expected_real = row_map.owned_global_rows[
                row_map.owned_global_rows < num_tokens
            ].tolist()
            assert compact_ranges == expected_real
            for local_row, global_row in enumerate(expected_real):
                assert int(row_map.global_to_local[global_row]) == local_row

        padded = ((num_tokens + cp_size - 1) // cp_size) * cp_size
        assert sorted(all_owned) == list(range(padded))
