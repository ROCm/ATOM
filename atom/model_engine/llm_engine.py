# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import itertools
import logging
import time
from dataclasses import fields
from typing import List, Optional, Union

from atom.config import Config
from atom.model_engine.engine_core_mgr import CoreManager
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams
from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger("atom")


def _load_tokenizer(model: str, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=True, trust_remote_code=trust_remote_code
    )
    probe = "Hello world 你好"
    if tokenizer.decode(tokenizer.encode(probe), skip_special_tokens=True) != probe:
        logger.warning(
            "AutoTokenizer round-trip failed, falling back to PreTrainedTokenizerFast"
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model)
    return tokenizer


class LLMEngine:

    def __init__(self, model, tokenizer=None, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        data_parallel_size = kwargs.get("data_parallel_size", 1)
        config = Config(model, **config_kwargs)
        self.tokenizer = tokenizer or _load_tokenizer(
            config.model, config.trust_remote_code
        )
        config.bos_token_id = self.tokenizer.bos_token_id
        config.eos_token_id = self.tokenizer.eos_token_id
        stop_token_ids = set(config.stop_token_ids)
        # separate eos_token_id from stop_token_ids
        stop_token_ids.discard(config.eos_token_id)
        config.stop_token_ids = list(stop_token_ids)
        # Set data parallel size in config
        config.parallel_config.data_parallel_size = data_parallel_size
        self.data_parallel_size = data_parallel_size
        self.rquest_ids = set()
        self.io_processor = InputOutputProcessor(
            config, self.tokenizer, config.kv_cache_block_size
        )
        self.core_mgr = CoreManager(config)
        self._step_lock = None
        self._pending_results = {}
        import json

        kv_config_str = kwargs.get("kv_transfer_config", "{}")
        try:
            config.kv_transfer_config = json.loads(kv_config_str)
            logger.info(f"KV transfer config loaded: {config.kv_transfer_config}")
        except json.JSONDecodeError:
            config.kv_transfer_config = {}
        logger.info(
            f"LLMEngine init with {self.data_parallel_size} data parallel ranks"
        )
        logger.info(
            f"LLMEngine init with {self.data_parallel_size} data parallel ranks"
        )

    def close(self):
        """Shut down engine and release all GPU resources."""
        if hasattr(self, "core_mgr"):
            self.core_mgr.close()

    def add_request(
        self,
        prompt_or_tokens_list: List[Union[str, List[int]]],
        sampling_params_list: SamplingParams | List[SamplingParams],
        stream_callback=None,
    ):
        # if sampling params is not list, use it for all prompts
        if not isinstance(sampling_params_list, list):
            sampling_params_iter = itertools.repeat(sampling_params_list)
        else:
            # otherwise check num elements first
            if len(prompt_or_tokens_list) != len(sampling_params_list):
                raise ValueError(
                    f"number of elements in prompt_or_tokens_list and sampling_params_list is different: "
                    f"{len(prompt_or_tokens_list)=} vs {len(sampling_params_list)=}"
                )
            sampling_params_iter = sampling_params_list

        # Handle stream_callback
        if stream_callback is not None and not isinstance(stream_callback, list):
            stream_callback_iter = itertools.repeat(stream_callback)
        elif isinstance(stream_callback, list):
            if len(stream_callback) != len(prompt_or_tokens_list):
                raise ValueError(
                    f"number of elements in prompt_or_tokens_list and stream_callback is different: "
                    f"{len(prompt_or_tokens_list)=} vs {len(stream_callback)=}"
                )
            stream_callback_iter = stream_callback
        else:
            stream_callback_iter = itertools.repeat(None)

        reqs = []
        for prompt, sampling_param, callback in zip(
            prompt_or_tokens_list, sampling_params_iter, stream_callback_iter
        ):
            req = self.io_processor.preprocess(
                prompt, sampling_param, stream_callback=callback
            )
            reqs.append(req)
        self.core_mgr.add_request(reqs)

    def step(self) -> list[Sequence]:
        seqs = self.core_mgr.get_output()
        return seqs

    def is_finished(self):
        return not self.io_processor.has_pending_requests()

    def generate(
        self,
        # prompts: list[str] | list[list[int]],
        prompts: list[str],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[str]:
        # Reset round-robin counter to ensure consistent DP not core dump
        self.core_mgr._rr_counter = 0

        self.add_request(prompts, sampling_params)
        outputs = {}
        while not self.is_finished() and (
            self.core_mgr.is_alive() or self.core_mgr.is_rest()
        ):
            seqs = self.step()
            outs = self.io_processor.postprocess(seqs)
            outputs.update(outs)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        return outputs

    def start_profile(self):
        self.core_mgr.send_utility_command("start_profile")
        logger.info("Profiling started")

    def stop_profile(self):
        self.core_mgr.send_utility_command("stop_profile")

    def print_mtp_statistics(self):
        self.core_mgr.send_utility_command("get_mtp_stats")


class InputOutputProcessor:

    def __init__(self, config, tokenizer, block_size):
        self.config = config
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.requests = {}
        self.mamba_enabled = False
        self.num_speculative_tokens = 0
        if (
            hasattr(self.config, "speculative_config")
            and self.config.speculative_config is not None
        ):
            self.num_speculative_tokens = (
                self.config.speculative_config.num_speculative_tokens
            )
        mamba_model_types = {"qwen3_next", "qwen3_5_text", "qwen3_5_moe_text"}
        if self.config.hf_config.model_type in mamba_model_types:
            self.mamba_enabled = True

    def preprocess(
        self,
        prompt_or_tokens: str | list[int],
        sampling_params: SamplingParams,
        stream_callback=None,
        kv_transfer_params=None,
    ):
        """responsible for:
        1) Tokenize
        2) Create Sequence object

        Single-sequence entry point. Rejects ``sampling_params.n > 1`` so that
        callers which expect exactly one ``Sequence`` back cannot silently
        drop the other siblings. Use :meth:`preprocess_fanout` for n > 1.
        """
        if getattr(sampling_params, "n", 1) > 1:
            raise ValueError(
                "preprocess() returns a single Sequence; for SamplingParams.n > 1 "
                "call preprocess_fanout() and manage the returned list."
            )
        seqs = self.preprocess_fanout(
            prompt_or_tokens,
            sampling_params,
            stream_callback=stream_callback,
            kv_transfer_params=kv_transfer_params,
        )
        return seqs[0]

    def preprocess_fanout(
        self,
        prompt_or_tokens: str | list[int],
        sampling_params: SamplingParams,
        stream_callback=None,
        stream_callbacks: Optional[List] = None,
        kv_transfer_params=None,
        parent_request_id: Optional[str] = None,
    ) -> List[Sequence]:
        """Tokenize once and materialize ``sampling_params.n`` Sequences.

        Returns a list of length ``n``. For ``n == 1`` this is functionally
        equivalent to the legacy single-sequence path. For ``n > 1``:

        * The prompt is tokenized a single time and the token list is copied
          into each sibling (``Sequence`` copies internally, so mutations stay
          isolated).
        * Every sibling is marked ``needs_independent_noise=True`` so the
          sampler generates fresh per-row noise instead of reusing the cached
          shared exponential tensor. Without this, siblings with identical
          logits would emit identical tokens.
        * Per-sibling ``stream_callbacks`` can be supplied to route streaming
          deltas to independent queues (one per choice index). Falls back to
          the scalar ``stream_callback`` for every sibling.
        """
        n = max(1, int(getattr(sampling_params, "n", 1)))

        tokens = (
            self.tokenizer.encode(prompt_or_tokens)
            if isinstance(prompt_or_tokens, str)
            else prompt_or_tokens
        )

        stop_token_sequences = []
        if sampling_params.stop_strings:
            stops = (
                [sampling_params.stop_strings]
                if isinstance(sampling_params.stop_strings, str)
                else sampling_params.stop_strings
            )
            for stop_str in stops:
                stop_tokens = self.tokenizer.encode(stop_str, add_special_tokens=False)
                if stop_tokens:
                    stop_token_sequences.append(stop_tokens)

        if stream_callbacks is not None and len(stream_callbacks) != n:
            raise ValueError(
                f"stream_callbacks length {len(stream_callbacks)} does not match n={n}"
            )

        seqs: List[Sequence] = []
        for i in range(n):
            cb = (
                stream_callbacks[i] if stream_callbacks is not None else stream_callback
            )
            seq = Sequence(
                tokens,
                self.block_size,
                sampling_params,
                stop_token_sequences,
                stream_callback=cb,
                num_draft_tokens=self.num_speculative_tokens,
                mamba_enabled=self.mamba_enabled,
                kv_transfer_params=kv_transfer_params,
                needs_independent_noise=(n > 1),
                parent_request_id=parent_request_id,
                sibling_index=i,
            )
            seq.arrive_time = time.time()
            self.requests[seq.id] = seq
            seqs.append(seq)

        if n == 1:
            logger.info(
                f"Request {seqs[0].id} arrived, input tokens: {len(tokens)}, "
                f"pending requests: {len(self.requests)}"
            )
        else:
            logger.info(
                f"Request {parent_request_id or seqs[0].id} fanned out into "
                f"{n} siblings ({seqs[0].id}..{seqs[-1].id}), "
                f"input tokens: {len(tokens)}, "
                f"pending requests: {len(self.requests)}"
            )
        return seqs

    def postprocess(self, reqs: List[Sequence]):
        """responsible for:
        1) Compute stats for logging
        2) Detokenize"""
        outputs = {}
        for req in reqs:
            self.requests.pop(req.id)
            output_str = self.tokenizer.decode(req.completion_token_ids)
            req.leave_time = time.time()

            # Calculate TTFT (Time To First Token) and TPOT (Time Per Output Token)
            ttft = 0.0
            tpot = 0.0
            if req.first_token_time > 0:
                ttft = req.first_token_time - req.arrive_time
                # Calculate TPOT only if there are multiple output tokens
                if req.num_completion_tokens > 1:
                    tpot = (req.leave_time - req.first_token_time) / (
                        req.num_completion_tokens - 1
                    )

            logger.info(
                f"Request {req.id} finished with reason {req.leave_reason}. "
                f"Input tokens: {req.num_prompt_tokens}, output tokens: {req.num_completion_tokens}, "
                f"latency: {req.leave_time - req.arrive_time:.2f}s, "
                f"TTFT: {ttft:.3f}s, TPOT: {tpot:.3f}s"
                # f"{req.completion_token_ids}"
            )
            outputs[req.id] = {
                "text": output_str,
                "token_ids": req.completion_token_ids,
                "latency": req.leave_time - req.arrive_time,
                "finish_reason": req.leave_reason,
                "num_tokens_input": req.num_prompt_tokens,
                "num_tokens_output": req.num_completion_tokens,
                "ttft": ttft,  # Time to first token in seconds
                "tpot": tpot,  # Time per output token in seconds
            }
        return outputs

    def has_pending_requests(self):
        return len(self.requests) > 0
