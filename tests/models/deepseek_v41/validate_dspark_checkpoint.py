# SPDX-License-Identifier: MIT
"""TP4 draft diagnostics with real target features and independent draft weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from atom.examples.deepseek_v41_offline import (
    initialize_parallel,
    load_offline_model,
    prepare_engram,
)
from atom.models.deepseek_v41.dspark import DeepseekV41DSpark
from tests.models.deepseek_v41.checkpoint_draft_reference import (
    checkpoint_draft_reference,
)

from atom.model_loader.loader import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/DeepSeek-V4.1-Flash")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = initialize_parallel()
    report = {
        "completed": False,
        "tp": torch.distributed.get_world_size(),
        "not_a_runtime_acceptance": True,
        "cases": [],
    }
    with load_offline_model(args.model, 512) as (target, tokenizer, mapping, host):
        config = target.config
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                draft = DeepseekV41DSpark(config, max_length=512)
        finally:
            torch.set_default_dtype(previous)
        loaded = load_model(draft, args.model, config, spec_decode=True)
        report["loaded_draft_parameters"] = len(loaded)
        report["draft_stages"] = len(draft.mtp)
        report["experts_per_stage"] = [
            layer.ffn.experts.global_num_experts for layer in draft.mtp
        ]
        draft.share_with_target(target)
        captured = {}
        hooks = []
        for layer_id in config.dspark_target_layer_ids:
            assert target.layers[layer_id].engram is None

            def capture(module, inputs, layer_id=layer_id):
                captured[layer_id] = inputs[0].residual.mean(dim=-2).detach().clone()

            hooks.append(target.layers[layer_id].register_forward_pre_hook(capture))
        seed = tokenizer.encode(
            "Explain why a private sliding window must restore the accepted prefix after speculative decoding.",
            add_special_tokens=True,
        )
        cases = [
            seed,
            (seed * 20)[:129],
            (seed * 20)[:257],
            tokenizer.encode(
                "请用中文解释推测解码为什么不能保留被拒绝的状态。",
                add_special_tokens=True,
            ),
        ]
        with (
            checkpoint_draft_reference(args.model, config, 512) as (source, reference),
            torch.inference_mode(),
            torch.device("cuda"),
        ):
            native_stages, reference_stages = {}, {}
            stage_hooks = []
            for stage, layer in enumerate(draft.mtp):

                def capture_native(module, inputs, output, stage=stage):
                    native_stages[stage] = output.collapse().float().clone()

                stage_hooks.append(layer.register_forward_hook(capture_native))
            for stage, layer in enumerate(source.mtp):

                def capture_reference(module, inputs, output, stage=stage):
                    reference_stages[stage] = module.hc_pre(*output).float().clone()

                stage_hooks.append(layer.register_forward_hook(capture_reference))
            for case_id, tokens in enumerate(cases):
                embeddings, _ = prepare_engram(
                    tokens, 0, np.full((1, 3), -1, dtype=np.int64), mapping, host
                )
                logits = target(
                    torch.tensor([tokens], device="cuda"),
                    target.new_cache(1),
                    embeddings,
                )
                anchors = logits.argmax(-1)
                main_hidden = torch.cat(
                    [captured[i] for i in config.dspark_target_layer_ids], dim=-1
                )
                count = len(tokens)
                x, main_x = source.mtp[0].forward_embed(main_hidden[:, :-1], anchors)
                pre = reference.make_identity_pre_mix(x, config.hc_mult)
                for layer in source.mtp:
                    x, pre = layer(x, 0, pre, main_x)
                x, main_x = source.mtp[0].forward_embed(main_hidden[:, -1:], anchors)
                pre = reference.make_identity_pre_mix(x, config.hc_mult)
                for layer in source.mtp:
                    x, pre = layer(x, count - 1, pre, main_x)
                expected_hidden = source.mtp[-1].hc_pre(x, pre)
                expected_ids, expected_logits, expected_confidence = source.mtp[
                    -1
                ].forward_head(x, pre, anchors)
                projected = draft.project_context_kv(
                    main_hidden, torch.arange(count, device="cuda")[None]
                )
                start = max(0, count - draft.window_size)
                kept_positions = torch.arange(start, count, device="cuda")
                positions = torch.full(
                    (1, draft.window_size), -1, device="cuda", dtype=torch.int64
                )
                slots = kept_positions % draft.window_size
                positions[:, slots] = kept_positions
                context_kv = {}
                for layer, values in projected.items():
                    window = values.new_zeros(1, draft.window_size, config.head_dim)
                    window[:, slots] = values[:, start:]
                    context_kv[layer] = window
                out = draft.draft_hidden(
                    anchors,
                    torch.tensor([count - 1], device="cuda"),
                    context_kv,
                    positions,
                )
                actual_ids, actual_confidence = draft.head_and_sample(
                    out, anchors, config.dspark_block_size
                )
                teacher_logits = draft.head(out[0])
                for i in range(config.dspark_block_size):
                    bias, _ = draft.mtp[-1].markov_head(expected_ids[:, i])
                    teacher_logits[:, i] += bias
                log_p, log_q = expected_logits.log_softmax(
                    -1
                ), teacher_logits.log_softmax(-1)
                kl = (log_p.exp() * (log_p - log_q)).sum(-1)
                row = {
                    "case": case_id,
                    "tokens": count,
                    "input_sha256": hashlib.sha256(
                        json.dumps(tokens).encode()
                    ).hexdigest(),
                    "reference_draft_ids": expected_ids[:, 1:].tolist(),
                    "draft_ids": actual_ids.tolist(),
                    "matching_draft_tokens": int(
                        (actual_ids == expected_ids[:, 1:]).sum()
                    ),
                    "hidden_relative_l2": float(
                        (out[1].float() - expected_hidden.float()).norm()
                        / expected_hidden.float().norm()
                    ),
                    "teacher_forced_kl": kl.tolist(),
                    "max_confidence_error": float(
                        (actual_confidence - expected_confidence.sigmoid()).abs().max()
                    ),
                    "stage_hidden_relative_l2": [
                        float(
                            (native_stages[i] - reference_stages[i]).norm()
                            / reference_stages[i].norm()
                        )
                        for i in range(len(draft.mtp))
                    ],
                    "context_compared_slots": slots.tolist(),
                    "context_relative_l2": [
                        float(
                            (
                                context_kv[layer.attn.spec.layer_id][:, slots].float()
                                - source.mtp[i].attn.window_kv_cache[:, slots].float()
                            ).norm()
                            / source.mtp[i]
                            .attn.window_kv_cache[:, slots]
                            .float()
                            .norm()
                        )
                        for i, layer in enumerate(draft.mtp)
                    ],
                }
                assert (
                    torch.isfinite(teacher_logits).all()
                    and torch.isfinite(actual_confidence).all()
                )
                all_ids = [None] * report["tp"]
                torch.distributed.all_gather_object(all_ids, actual_ids.tolist())
                assert all(value == all_ids[0] for value in all_ids)
                report["cases"].append(row)
                if rank == 0:
                    print(json.dumps(row), flush=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
            for hook in stage_hooks:
                hook.remove()
        for hook in hooks:
            hook.remove()
    report["completed"] = True
    if rank == 0:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
