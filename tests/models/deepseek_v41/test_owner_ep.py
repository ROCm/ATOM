# SPDX-License-Identifier: MIT
"""CPU ownership checks for the actual EP4 deployment validator."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "owner_ep",
    Path(__file__).resolve().parents[3] / "atom/models/deepseek_v41/owner_ep.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MappingTests(unittest.TestCase):
    def test_complete_partition_and_reject_cross_rank_or_poisoned_sentinel(self):
        ownership = []
        for rank in range(4):
            mapping = [-1] * 385
            mapping[rank * 96 : (rank + 1) * 96] = range(96)
            mask = [int(v >= 0) for v in mapping]
            module.validate_mapping(mapping, mask, rank)
            ownership.extend(i for i, value in enumerate(mapping) if value >= 0)
            with self.assertRaises(ValueError):
                module.validate_mapping(mapping, mask, (rank + 1) % 4)
            corrupt = mapping[:]
            corrupt[-1] = 0
            with self.assertRaises(ValueError):
                module.validate_mapping(corrupt, mask, rank)
            bad_mask = mask[:]
            bad_mask[rank * 96] = 0
            with self.assertRaises(ValueError):
                module.validate_mapping(mapping, bad_mask, rank)
        self.assertEqual(ownership, list(range(384)))


if __name__ == "__main__":
    unittest.main()
