# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import unittest
from copy import deepcopy
from math import prod

from atom.distributed.pp_host_fence.contract import plan_submission
from atom.distributed.pp_host_fence.k3_edge_schema import edge_schemas, routes


class TestHostFencePolicy(unittest.TestCase):
    def setUp(self):
        self.main = {
            "model_type": "kimi_k3",
            "text_config": {
                "num_hidden_layers": 93,
                "hidden_size": 7168,
                "attn_res_block_size": 12,
            },
        }
        self.draft = {
            "target_layer_ids": [2, 23, 47, 71, 89],
            "target_hidden_size": 7168,
            "target_num_hidden_layers": 93,
        }
        self.identities = [
            {
                "ppRank": i,
                "hostname": "left" if i < 4 else "right",
                "bootId": "boot-left" if i < 4 else "boot-right",
                "gpuUuid": f"gpu-{i}",
            }
            for i in range(8)
        ]

    def test_aux_columns_follow_actual_layer_boundaries(self):
        schemas = edge_schemas(self.main, self.draft)
        self.assertEqual([s.aux_taps for s in schemas], [1, 1, 2, 2, 3, 4, 4])

    def test_residual_bank_follows_main_layer_boundaries(self):
        schemas = edge_schemas(self.main, self.draft)
        self.assertEqual([s.blocks for s in schemas], [1, 2, 3, 4, 5, 6, 8])
        for schema, blocks in zip(schemas, [1, 2, 3, 4, 5, 6, 8]):
            for rows in (0, 17, 4097, 16384):
                with self.subTest(blocks=blocks, rows=rows):
                    fields = schema.fields(rows)
                    self.assertEqual(fields["block_residual"], (rows, blocks, 7168))
                    descriptors = {
                        key: {
                            "shape": shape,
                            "dtype": "bfloat16",
                            "device": "cuda",
                            "contiguous": True,
                            "bytes": prod(shape) * 2,
                        }
                        for key, shape in fields.items()
                    }
                    plan = plan_submission(schema, descriptors)
                    self.assertEqual(
                        plan["copyBytes"], sum(prod(s) * 2 for s in fields.values())
                    )
                    if blocks != 8:
                        descriptors["block_residual"].update(
                            shape=(rows, 8, 7168), bytes=rows * 8 * 7168 * 2
                        )
                        with self.assertRaises(ValueError):
                            plan_submission(schema, descriptors)

    def test_maximum_buffers_use_wide_byte_offsets(self):
        schema = edge_schemas(self.main, self.draft)[6]
        meta = schema.metadata()
        self.assertEqual(meta["fields"][1]["capacityBytes"], 1879048192)
        self.assertEqual(meta["slotBytes"], 3053453312)
        self.assertEqual(meta["ringBytes"], 9160359936)
        self.assertGreater(2 * meta["slotBytes"], 2**32)

    def test_model_and_draft_mismatch_refused(self):
        for key, value in (
            ("attn_res_block_size", 13),
            ("num_hidden_layers", 94),
            ("hidden_size", 4096),
        ):
            main = deepcopy(self.main)
            main["text_config"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                edge_schemas(main, self.draft)
        for key, value in (
            ("target_hidden_size", 4096),
            ("target_num_hidden_layers", 94),
            ("target_layer_ids", [2, 24, 47, 71, 89]),
        ):
            draft = dict(self.draft, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                edge_schemas(self.main, draft)

    def test_only_same_host_edges_select_ipc(self):
        selected = routes(self.identities)
        self.assertEqual(
            [x["transport"] for x in selected],
            ["hip-ipc-memory"] * 3 + ["rccl"] + ["hip-ipc-memory"] * 3,
        )

    def test_unknown_or_aliased_device_cannot_exchange_handles(self):
        for key in ("hostname", "bootId", "gpuUuid"):
            ids = deepcopy(self.identities)
            ids[1][key] = "unknown"
            with self.subTest(key=key), self.assertRaises(ValueError):
                routes(ids)
        ids = deepcopy(self.identities)
        ids[1]["gpuUuid"] = ids[0]["gpuUuid"]
        with self.assertRaises(ValueError):
            routes(ids)

    def test_rank_omission_or_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            routes(self.identities[:-1])
        ids = deepcopy(self.identities)
        ids[7]["ppRank"] = 6
        with self.assertRaises(ValueError):
            routes(ids)
