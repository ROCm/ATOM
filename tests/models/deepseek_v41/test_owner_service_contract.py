# SPDX-License-Identifier: MIT
"""Admission for combined owner decode and retained prefill, CPU only."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "owner_service_contract",
    Path(__file__).resolve().parents[3]
    / "atom/models/deepseek_v41/owner_service_contract.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
owner_prefill_enabled = module.owner_prefill_enabled


def config():
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="dspark", num_speculative_tokens=5),
        tensor_parallel_size=4,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_dp_attention=False,
        enable_tbo=False,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=False,
        index_cache_dtype="fp4",
        kv_cache_dtype="bf16",
        enable_decoder_swa_bounded_replay=True,
    )


class OwnerPrefillTests(unittest.TestCase):
    def test_ep_requires_verified_modules_and_keeps_deployment_constraints(self):
        deployment = config()
        deployment.enable_expert_parallel = True
        arguments = {"owner_service": True, "retained_prefill": True}
        self.assertFalse(owner_prefill_enabled(deployment, **arguments))
        self.assertTrue(
            owner_prefill_enabled(deployment, ep4_verified=True, **arguments)
        )
        self.assertFalse(owner_prefill_enabled(deployment, ep4_verified=1, **arguments))
        deployment.parallel_config.data_parallel_size = 2
        self.assertFalse(
            owner_prefill_enabled(deployment, ep4_verified=True, **arguments)
        )
        deployment.parallel_config.data_parallel_size = 1
        deployment.speculative_config.num_speculative_tokens = 6
        self.assertFalse(
            owner_prefill_enabled(deployment, ep4_verified=True, **arguments)
        )

    def test_explicit_combination(self):
        value = config()
        self.assertTrue(
            owner_prefill_enabled(value, owner_service=True, retained_prefill=True)
        )
        self.assertFalse(
            owner_prefill_enabled(value, owner_service=False, retained_prefill=True)
        )
        self.assertFalse(
            owner_prefill_enabled(value, owner_service=True, retained_prefill=False)
        )

    def test_other_deployments_keep_their_refusal(self):
        for field, value in (
            ("tensor_parallel_size", 2),
            ("enable_dp_attention", True),
            ("enable_tbo", True),
            ("prefill_context_parallel_size", 2),
            ("decode_context_parallel_size", 2),
            ("enable_expert_parallel", True),
            ("index_cache_dtype", "fp8"),
            ("kv_cache_dtype", "fp4"),
            ("enable_decoder_swa_bounded_replay", False),
        ):
            with self.subTest(field=field):
                deployment = config()
                setattr(deployment, field, value)
                self.assertFalse(
                    owner_prefill_enabled(
                        deployment, owner_service=True, retained_prefill=True
                    )
                )

    def test_other_proposals_keep_their_refusal(self):
        deployment = config()
        for method, width in (("eagle3", 5), ("dspark", 3), ("dspark", 6)):
            with self.subTest(method=method, width=width):
                deployment.speculative_config = SimpleNamespace(
                    method=method, num_speculative_tokens=width
                )
                self.assertFalse(
                    owner_prefill_enabled(
                        deployment, owner_service=True, retained_prefill=True
                    )
                )

    def test_disabled_path_does_not_require_new_configuration(self):
        self.assertFalse(
            owner_prefill_enabled(
                SimpleNamespace(), owner_service=False, retained_prefill=False
            )
        )


if __name__ == "__main__":
    unittest.main()
