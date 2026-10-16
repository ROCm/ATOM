# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github/scripts/atomesh/minimax_bounded_gather.py"
)
spec = importlib.util.spec_from_file_location("bounded_gather", SCRIPT)
bounded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounded)
try:
    import torch
except ImportError:
    torch = None


class RequestGroupsTests(unittest.TestCase):
    def test_stalled_batch_fits_original_contexts_into_eight_gib(self):
        cached = [
            440192,
            68352,
            75904,
            100992,
            168448,
            227072,
            128,
            128,
            92672,
            70656,
            210816,
            695296,
            653312,
            682496,
            511104,
        ]
        new = [
            1674,
            730,
            11224,
            2951,
            303,
            271,
            505,
            374,
            3604,
            598,
            2587,
            2692,
            2217,
            2223,
            768,
        ]
        cu_q, cu_k = [0], [0]
        for old, added in zip(cached, new):
            cu_q.append(cu_q[-1] + added)
            cu_k.append(cu_k[-1] + old + added)
        self.assertEqual(cu_k[-1], 4030289)
        groups = bounded.request_groups(cu_q, cu_k, 1048576)
        self.assertEqual(
            [i for start, end in groups for i in range(start, end)], list(range(15))
        )
        self.assertLessEqual(
            max(cu_k[end] - cu_k[start] for start, end in groups), 1048576
        )
        self.assertGreater(len(groups), 1)

    def test_context_is_never_truncated_to_fit_workspace(self):
        with self.assertRaisesRegex(ValueError, "One request exceeds"):
            bounded.request_groups([0, 2], [0, 1048577], 1048576)

    def test_request_boundary_exact_fit_and_invalid_geometry(self):
        self.assertEqual(
            bounded.request_groups([0, 2, 5, 6], [0, 7, 10, 19], 10), [(0, 2), (2, 3)]
        )
        with self.assertRaises(ValueError):
            bounded.request_groups([0, 4], [0, 3], 10)


@unittest.skipIf(
    torch is None, "CPU torch is needed for numerical attention verification"
)
class AttentionGroupingTests(unittest.TestCase):
    def test_grouped_attention_matches_full_paged_reference_without_mutation(self):
        torch.manual_seed(23)
        q_lengths = torch.tensor([2, 3, 1, 4, 2], dtype=torch.int32)
        k_lengths = torch.tensor([7, 3, 9, 6, 2], dtype=torch.int32)
        cu_q = torch.cat(
            [torch.zeros(1, dtype=torch.int32), q_lengths.cumsum(0).to(torch.int32)]
        )
        cu_k = torch.cat(
            [torch.zeros(1, dtype=torch.int32), k_lengths.cumsum(0).to(torch.int32)]
        )
        blocks = torch.randperm(25).reshape(5, 5)
        starts = torch.tensor([0, 1, 3, 0, 2], dtype=torch.int32)
        k_cache = torch.randn(25, 4, 2, 4, dtype=torch.float64)
        v_cache = torch.randn_like(k_cache)
        k_scale = torch.rand(25, 4, 2, 1, dtype=torch.float64) + 0.2
        v_scale = torch.rand_like(k_scale) + 0.2
        q = torch.randn(int(cu_q[-1]), 2, 4, dtype=torch.float64)
        meta = SimpleNamespace(
            has_cached=True,
            total_kv=int(cu_k[-1]),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            block_tables=blocks,
            seq_starts=starts,
            context_lens=k_lengths,
            batch_id_per_k_token=torch.repeat_interleave(
                torch.arange(5, dtype=torch.int32), k_lengths
            ),
            max_seqlen_q=4,
            max_seqlen_k=9,
            _ci_prefill_cu_q=cu_q.numpy().copy(),
            _ci_prefill_cu_k=cu_k.numpy().copy(),
        )
        ctx = SimpleNamespace(
            context=SimpleNamespace(is_draft=True), attn_metadata=meta
        )
        original = {
            key: value.clone()
            for key, value in vars(meta).items()
            if isinstance(value, torch.Tensor)
        }

        def reference(queries, md):
            # Independent per-request causal attention, including shuffled
            # physical pages, nonzero seq_starts and per-token dequant scales.
            outputs = []
            for i, (a, b) in enumerate(zip(md.cu_seqlens_q[:-1], md.cu_seqlens_q[1:])):
                count = int(md.context_lens[i])
                pos = torch.arange(count) + md.seq_starts[i]
                pages, offsets = md.block_tables[i, pos // 4], pos % 4
                keys = k_cache[pages, offsets] * k_scale[pages, offsets]
                values = v_cache[pages, offsets] * v_scale[pages, offsets]
                sub_q = queries[int(a) : int(b)]
                logits = torch.einsum("qhd,khd->hqk", sub_q, keys) / 2.0
                mask = torch.arange(count)[None, :] > torch.arange(len(sub_q))[
                    :, None
                ] + count - len(sub_q)
                logits.masked_fill_(mask, float("-inf"))
                outputs.append(torch.einsum("hqk,khd->qhd", logits.softmax(-1), values))
            return torch.cat(outputs)

        expected = reference(q, meta)
        seen = []
        max_bytes = 10 * 2 * 4 * 8 * 2

        class Attention:
            def prefill_attention(self, q, k, v, kc, vc, ks, vs, context):
                result = bounded.bounded_prefill_attention(
                    self, q, k, v, kc, vc, ks, vs, context, max_bytes
                )
                if result is not None:
                    return result
                md = context.attn_metadata
                seen.append(md.total_kv)
                torch.testing.assert_close(
                    md.batch_id_per_k_token,
                    torch.repeat_interleave(
                        torch.arange(len(md.context_lens), dtype=torch.int32),
                        md.context_lens,
                    ),
                )
                torch.testing.assert_close(
                    md.cu_seqlens_k[1:] - md.cu_seqlens_k[:-1], md.context_lens
                )
                return reference(q, md)

        actual = Attention().prefill_attention(
            q, q, q, k_cache, v_cache, k_scale, v_scale, ctx
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(seen, [10, 9, 8])
        self.assertIs(ctx.attn_metadata, meta)
        for key, value in original.items():
            torch.testing.assert_close(getattr(meta, key), value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
