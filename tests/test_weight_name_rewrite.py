# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for checkpoint-name rewriting.

Nine rules run in a fixed order to turn an on-disk tensor name into the
parameter it belongs to, and getting the order wrong is the failure mode that
motivated pulling them out of the loading loop: dropping every name containing
"mtp" before `mtp_remap` has run throws away a whole drafter checkpoint, since
the drafter's remap rewrites `mtp.*` into names that no longer say "mtp".

Pure string manipulation, no torch state, no AITER.
"""

import unittest

from atom.model_loader.weight_names import (
    CheckpointNameRewriter,
    WeightsMapper,
    extract_expert_target_and_id,
    have_shared_expert,
    shared_expert_prefixes,
)


def rewriter(**kwargs) -> CheckpointNameRewriter:
    kwargs.setdefault("num_hidden_layers", 4)
    kwargs.setdefault("n_routed_experts", 8)
    kwargs.setdefault("fuse_shared_expert", lambda *_: True)
    return CheckpointNameRewriter(**kwargs)


class WeightsMapperCompatibilityTest(unittest.TestCase):
    def test_mapper_is_already_unstacked(self):
        mapper = WeightsMapper(
            orig_to_new_prefix={"model.language_model.": "language_model.model."}
        )

        unstacked_mapper = mapper.get_unstacked_mapper()

        self.assertIs(unstacked_mapper, mapper)
        self.assertEqual(
            unstacked_mapper.apply_list(
                ["model.language_model.layers.0.self_attn.q_proj"]
            ),
            ["language_model.model.layers.0.self_attn.q_proj"],
        )


class DroppedTensorsTest(unittest.TestCase):
    def test_drops_scale_and_frequency_tensors(self):
        r = rewriter()
        self.assertIsNone(r.rewrite("model.layers.0.self_attn.kv_scale"))
        self.assertIsNone(r.rewrite("model.layers.0.self_attn.rotary_emb.inv_freq"))

    def test_drops_model_declared_prefixes(self):
        r = rewriter(skip_weight_prefixes=["model.visual."])
        self.assertIsNone(r.rewrite("model.visual.blocks.0.mlp.linear_fc1.weight"))
        self.assertIsNotNone(r.rewrite("model.layers.0.mlp.gate.weight"))

    def test_drops_layers_past_the_configured_depth(self):
        r = rewriter(num_hidden_layers=4)
        self.assertIsNotNone(r.rewrite("model.layers.3.mlp.gate.weight"))
        self.assertIsNone(r.rewrite("model.layers.4.mlp.gate.weight"))

    def test_keeps_extra_layers_when_loading_a_drafter(self):
        # The drafter's own block sits at an index past the target's depth.
        r = rewriter(num_hidden_layers=4, spec_decode=True)
        self.assertIsNotNone(r.rewrite("model.layers.4.mlp.gate.weight"))

    def test_mapper_veto_drops_the_tensor(self):
        r = rewriter(weights_mapper=WeightsMapper(orig_to_new_prefix={"aux.": None}))
        self.assertIsNone(r.rewrite("aux.something.weight"))


class MTPOrderingTest(unittest.TestCase):
    """The MTP filter must run before the model's own remap, not after."""

    MTP_NAME = "mtp.layers.0.mlp.experts.3.gate_proj.weight"

    @staticmethod
    def _drafter_rewriter():
        return rewriter(
            spec_decode=True,
            mtp_remap=lambda n: n if n.startswith("mtp.") else None,
            weights_mapping={"mtp.": "model."},
        )

    def test_target_pass_drops_the_drafter_block(self):
        self.assertIsNone(rewriter().rewrite(self.MTP_NAME))

    def test_drafter_pass_keeps_and_renames_it(self):
        self.assertEqual(
            self._drafter_rewriter().rewrite(self.MTP_NAME),
            "model.layers.0.mlp.experts.3.gate_proj.weight",
        )

    def test_drafter_pass_drops_the_target_block(self):
        r = self._drafter_rewriter()
        self.assertIsNone(r.rewrite("model.layers.0.mlp.experts.3.gate_proj.weight"))


class SharedExpertFusionTest(unittest.TestCase):
    SHARED = "model.layers.0.mlp.shared_expert.gate_proj.weight"

    def test_rewrites_into_the_slot_after_the_routed_experts(self):
        self.assertEqual(
            rewriter(n_routed_experts=8).rewrite(self.SHARED),
            "model.layers.0.mlp.experts.8.gate_proj.weight",
        )

    def test_left_alone_when_the_model_keeps_shared_experts_standalone(self):
        r = rewriter(disable_fused_shared_loading=True)
        self.assertEqual(r.rewrite(self.SHARED), self.SHARED)

    def test_left_alone_when_the_quant_configs_differ(self):
        r = rewriter(fuse_shared_expert=lambda *_: False)
        self.assertEqual(r.rewrite(self.SHARED), self.SHARED)

    def test_ffn_naming_keeps_its_module_prefix(self):
        name = "model.layers.0.ffn.shared_experts.gate_proj.weight"
        self.assertEqual(
            rewriter(n_routed_experts=8).rewrite(name),
            "model.layers.0.ffn.experts.8.gate_proj.weight",
        )

    def test_missing_expert_count_is_reported(self):
        r = rewriter(n_routed_experts=None)
        with self.assertRaises(AttributeError):
            r.rewrite(self.SHARED)

    def test_prefixes_used_for_the_fusion_decision(self):
        matching = have_shared_expert(self.SHARED)
        self.assertEqual(matching, "mlp.shared_expert.")
        self.assertEqual(
            shared_expert_prefixes(self.SHARED, matching),
            ("model.layers.0.mlp.shared_expert", "model.layers.0.mlp.experts"),
        )


class SubstitutionTest(unittest.TestCase):
    def test_weight_scale_inv_is_renamed(self):
        self.assertEqual(
            rewriter().rewrite("model.layers.0.mlp.gate.weight_scale_inv"),
            "model.layers.0.mlp.gate.weight_scale",
        )

    def test_mapper_runs_before_the_substring_map(self):
        r = rewriter(
            weights_mapper=WeightsMapper(orig_to_new_prefix={"raw.": "model."}),
            weights_mapping={"model.": "backbone."},
        )
        self.assertEqual(r.rewrite("raw.embed.weight"), "backbone.embed.weight")


class ExpertTargetExtractionTest(unittest.TestCase):
    def test_extracts_the_expert_id_and_fused_name(self):
        self.assertEqual(
            extract_expert_target_and_id("model.layers.10.mlp.experts.100.w2_bias"),
            ("model.layers.10.mlp.experts.w2_bias", 100),
        )

    def test_ignores_names_without_an_expert_index(self):
        self.assertIsNone(extract_expert_target_and_id("model.layers.10.mlp.gate"))
        self.assertIsNone(extract_expert_target_and_id("model.embed_tokens.weight"))


if __name__ == "__main__":
    unittest.main()
