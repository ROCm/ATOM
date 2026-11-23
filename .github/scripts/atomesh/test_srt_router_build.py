import copy
import json
import unittest
from pathlib import Path

from pd_srt_functional import verified_router_image

CONTRACT = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / ".github/benchmark/pr5090-router-ab/single-node-functional.json"
    ).read_text()
)


class RouterBuildTest(unittest.TestCase):
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
