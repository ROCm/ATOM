# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZhangDanyang-AMD
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Shared ATOM rollout adapter for miles- and slime-style hosts."""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _prompt_token_ids(sample: Any, tokenizer: Any) -> list[int]:
    if sample.tokens:
        return list(sample.tokens)
    prompt = sample.prompt
    if isinstance(prompt, str):
        return list(tokenizer.encode(prompt))
    if isinstance(prompt, list):
        encoded = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            **getattr(sample, "apply_chat_template_kwargs", {}),
        )
        if hasattr(encoded, "keys"):
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return list(encoded)
    raise TypeError(f"unsupported prompt type: {type(prompt).__name__}")


def _apply_output(
    sample: Any,
    output: dict[str, Any],
    *,
    prompt_ids: list[int],
    tokenizer: Any,
    max_tokens: int,
    rollout_id: int,
) -> Any:
    token_ids = list(output.get("token_ids", []))
    logprobs = output.get("logprobs")
    text = output.get("text", "")
    sample.tokens = [*prompt_ids, *token_ids]
    sample.response = text or tokenizer.decode(token_ids, skip_special_tokens=True)
    sample.response_length = len(token_ids)
    sample.rollout_log_probs = (
        [float(value) for value in logprobs] if logprobs else None
    )
    sample.status = (
        sample.Status.TRUNCATED
        if len(token_ids) >= max_tokens
        else sample.Status.COMPLETED
    )
    sample.rollout_id = rollout_id
    return sample


def generate_rollout(
    framework: str,
    args: Any,
    rollout_id: int,
    data_source: Any,
    *,
    evaluation: bool = False,
):
    """Generate framework-native samples with public ATOM manager APIs."""
    manager = getattr(args, "_atomrl_atom_manager", None)
    if manager is None:
        raise RuntimeError(
            "args._atomrl_atom_manager is not set; call setup_atom_manager first"
        )

    base_types = importlib.import_module(f"{framework}.rollout.base_types")
    if evaluation:
        return base_types.RolloutFnEvalOutput(data={})

    import ray

    processing = importlib.import_module(f"{framework}.utils.processing_utils")
    model_path = getattr(args, "hf_checkpoint", None) or getattr(
        args, "model_name", None
    )
    if not model_path:
        raise ValueError("args must define 'hf_checkpoint' or 'model_name'")
    tokenizer = processing.load_tokenizer(
        model_path,
        trust_remote_code=True,
    )
    max_tokens = int(getattr(args, "rollout_max_response_len", 2048))
    batch_size = int(getattr(args, "rollout_batch_size", 1))
    servers = manager.servers
    if not servers:
        raise RuntimeError("ATOM manager has no rollout servers")

    manager.wake_all(tags=["weights", "kv_cache"])
    try:
        selector = rollout_id if framework == "vime" else batch_size
        sample_groups = data_source.get_samples(selector)
        results = []
        sample_index = 0
        for group in sample_groups:
            group_out = []
            for sample in group:
                prompt_ids = _prompt_token_ids(sample, tokenizer)
                sample_rollout_id = (
                    sample.index if framework in {"miles", "slime"} else rollout_id
                )
                sampling_params = {
                    "max_tokens": max_tokens,
                    "temperature": getattr(args, "temperature", 1.0),
                    "top_p": getattr(args, "top_p", 1.0),
                }
                server = servers[sample_index % len(servers)]
                sample_index += 1
                output = ray.get(
                    server.generate.remote(
                        prompt=prompt_ids,
                        sampling_params=sampling_params,
                    )
                )
                group_out.append(
                    _apply_output(
                        sample,
                        output,
                        prompt_ids=prompt_ids,
                        tokenizer=tokenizer,
                        max_tokens=max_tokens,
                        rollout_id=sample_rollout_id,
                    )
                )
            results.append(group_out)
        return base_types.RolloutFnTrainOutput(samples=results)
    finally:
        manager.sleep_all()
