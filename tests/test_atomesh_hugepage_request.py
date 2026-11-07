# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""pd_submit.sh's sbatch --comment that asks the Spur prolog for a Store
cell's 2 MiB HugeTLB pages (mooncake_hugepage_request)."""

import re
import subprocess
import unittest
from pathlib import Path

SUBMIT_SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_submit.sh"
)


def request(env):
    """Runs mooncake_hugepage_request alone with the given role env."""
    text = SUBMIT_SCRIPT.read_text()
    function = re.search(
        r"^mooncake_hugepage_request\(\) \{\n.*?^\}\n", text, re.MULTILINE | re.DOTALL
    ).group(0)
    script = "set -euo pipefail\n" + function + "mooncake_hugepage_request\n"
    result = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class HugepageRequestTest(unittest.TestCase):
    def test_single_node_owners_ask_for_their_pages_on_each_numa_node(self):
        self.assertEqual(
            request(
                {
                    "ATOMESH_PREFILL_ENV_MOONCAKE_STORE": "1",
                    "ATOMESH_PREFILL_ENV_MOONCAKE_STORE_OWNERS": "0:768:rdma4,rdma5;1:768:rdma6,rdma7",
                }
            ),
            "hugepages_2m=node0:393216,node1:393216",
        )

    def test_launcher_default_owners_when_unset(self):
        self.assertEqual(
            request({"ATOMESH_PREFILL_ENV_MOONCAKE_STORE": "1"}),
            "hugepages_2m=node0:393216,node1:393216",
        )

    def test_two_node_owners_sum_per_numa_node_and_take_the_larger_node(self):
        self.assertEqual(
            request(
                {
                    "ATOMESH_PREFILL_ENV_MOONCAKE_STORE": "1",
                    "ATOMESH_PREFILL_ENV_MOONCAKE_STORE_OWNERS": "0:240:rdma0;0:240:rdma1;0:240:rdma2;0:240:rdma3",
                    "ATOMESH_PREFILL_ENV_MOONCAKE_STORE_DECODE_OWNERS": "0:200:rdma0;1:100:rdma4",
                }
            ),
            "hugepages_2m=node0:491520,node1:51200",
        )

    def test_no_request_without_store_or_with_hugetlb_off(self):
        self.assertEqual(request({}), "")
        self.assertEqual(
            request(
                {
                    "ATOMESH_PREFILL_ENV_MOONCAKE_STORE": "1",
                    "ATOMESH_PREFILL_ENV_MOONCAKE_STORE_HUGETLB": "0",
                }
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()
