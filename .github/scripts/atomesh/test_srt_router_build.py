import copy
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from pd_srt_functional import (
    configurations,
    prefill_remap_preamble,
    verified_router_image,
)

CONTRACT = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / ".github/benchmark/pr5090-router-ab/single-node-functional.json"
    ).read_text()
)


class RouterBuildTest(unittest.TestCase):
    def device_config(self, order):
        return configurations(
            CONTRACT | {"prefill_device_order": order},
            "/model",
            Path("/run"),
            Path("/experiment"),
            {
                "host-ip": "10.0.0.1",
                "socket-ifname": "eth0",
                "rdma-devices": "ionic_0",
                "rdma-gid-index": "1",
                "rdma-tc": "104",
            },
            "1000:1000",
            "sha256:" + "b" * 64,
        )

    def test_prefill_device_order_cannot_escape_its_gpu_allocation(self):
        for order in (
            [],
            [0, 1, 2, 4],
            [0, 1, 1, 3],
            [False, 1, 2, 3],
            "0,1,3,2",
            None,
        ):
            with self.subTest(order=order), self.assertRaises(ValueError):
                self.device_config(order)

    @unittest.skipUnless(
        shutil.which("bash"), "bash is required for the launch preamble"
    )
    def test_prefill_remap_preserves_decode_and_nonworker_masks(self):
        recipe, _cluster = self.device_config([0, 1, 3, 2])
        self.assertTrue(recipe["engine"]["set_visible_devices"])
        for mask, expected in (
            ("0,1,2,3", "0,1,3,2"),
            ("4,5,6,7", "4,5,6,7"),
            (None, "unset"),
        ):
            env = dict(os.environ)
            if mask is None:
                env.pop("HIP_VISIBLE_DEVICES", None)
            else:
                env["HIP_VISIBLE_DEVICES"] = mask
            command = (
                prefill_remap_preamble([0, 1, 3, 2])
                + '; printf "\\n%s\\n" "${HIP_VISIBLE_DEVICES-unset}"'
            )
            output = subprocess.check_output(
                [shutil.which("bash"), "-c", command], env=env, text=True
            )
            self.assertEqual(output.strip().splitlines()[-1], expected)

    def test_baseline_requires_its_own_image_and_version_evidence(self):
        contract = CONTRACT | {
            "router_arm": "baseline",
            "baseline_router_version": "0.1.16",
        }
        manifest = {
            "kind": "baseline",
            "base_image": contract["router_images"]["baseline"],
            "image_id": "sha256:" + "c" * 64,
            "version": "0.1.16",
            "binary_sha256": "d" * 64,
            "runtime_verified": True,
        }
        self.assertEqual(
            verified_router_image(contract, manifest), manifest["image_id"]
        )
        for key, value in (
            ("kind", "atomesh"),
            ("version", "0.1.15"),
            ("base_image", contract["router_images"]["atomesh"]),
            ("runtime_verified", False),
            ("binary_sha256", ""),
            ("image_id", "latest"),
        ):
            with self.subTest(field=key), self.assertRaises(ValueError):
                verified_router_image(contract, manifest | {key: value})

    def test_only_the_verified_image_is_selected(self):
        build = CONTRACT["router_build"]
        manifest = {
            "commit": build["source_sha"],
            "lockfile_sha256": build["lockfile_sha256"],
            "profile": build["profile"],
            "cargo": "cargo " + build["rust_toolchain"] + " (fixture)",
            "base_image": CONTRACT["router_images"]["atomesh"],
            "binary_sha256": "a" * 64,
            "runtime_verified": True,
            "image_id": "sha256:" + "b" * 64,
        }
        self.assertEqual(
            verified_router_image(CONTRACT, manifest), manifest["image_id"]
        )
        for key, value in (
            ("commit", "0" * 40),
            ("lockfile_sha256", "0" * 64),
            ("runtime_verified", False),
            ("binary_sha256", ""),
            ("cargo", "cargo 1.94.0 (old)"),
            ("profile", "ci"),
            ("image_id", CONTRACT["router_images"]["atomesh"]),
            ("image_id", "sha256:" + "z" * 64),
        ):
            with self.subTest(field=key, value=value):
                stale = copy.deepcopy(manifest)
                stale[key] = value
                with self.assertRaises(ValueError):
                    verified_router_image(CONTRACT, stale)


if __name__ == "__main__":
    unittest.main()
