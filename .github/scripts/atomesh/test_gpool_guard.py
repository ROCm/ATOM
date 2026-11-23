import copy
import unittest

from pd_gpool_guard import validate_lease


class PoolGuardTest(unittest.TestCase):
    def test_requires_owned_full_node_without_yield(self):
        lease_id = "L123abc"
        status = {
            "leases": {
                lease_id: {
                    "user": "hexwang",
                    "alloc": [{"node": "g07", "gpus": list(range(8))}],
                }
            },
            "flags": {lease_id: {"yield": False}},
        }
        self.assertEqual(
            validate_lease(status, lease_id, "pit2-p03-g07", "hexwang@amd.com", ""),
            status["leases"][lease_id],
        )
        for case in ("missing", "owner", "node", "gpus", "yield", "hold"):
            changed = copy.deepcopy(status)
            mine = ""
            if case == "missing":
                changed["leases"].clear()
            elif case == "owner":
                changed["leases"][lease_id]["user"] = "another-user"
            elif case == "node":
                changed["leases"][lease_id]["alloc"][0]["node"] = "g08"
            elif case == "gpus":
                changed["leases"][lease_id]["alloc"][0]["gpus"] = [0, 1, 2, 3]
            elif case == "yield":
                changed["flags"][lease_id]["yield"] = True
            else:
                mine = "node g07 no longer held by the team"
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                validate_lease(
                    changed, lease_id, "pit2-p03-g07", "hexwang@amd.com", mine
                )


if __name__ == "__main__":
    unittest.main()
