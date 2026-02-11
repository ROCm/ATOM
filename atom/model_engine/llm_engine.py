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
from atom.model_engine.weight_sync import load_weights_via_shm
from atom.sampling_params import SamplingParams
from transformers import AutoTokenizer

logger = logging.getLogger("atom")


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        data_parallel_size = kwargs.get('data_parallel_size', 1)
        trust_remote_code = kwargs.get('trust_remote_code', False)
        config = Config(model, **config_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, 
            use_fast=True,
            trust_remote_code=trust_remote_code
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
            self.tokenizer, config.kv_cache_block_size
        )
        self.core_mgr = CoreManager(config)
        self._step_lock = None
        self._pending_results = {}
        logger.info(
            f"LLMEngine init with {self.data_parallel_size} data parallel ranks"
        )

    def add_request(
        self,
        prompt_or_tokens_list: List[Union[str, List[int]]],
        sampling_params_list: SamplingParams | List[SamplingParams],
        stream_callback=None,
        request_ids: Optional[list[str]] = None,
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

        # Handle request_ids
        if request_ids is not None:
            if len(request_ids) != len(prompt_or_tokens_list):
                raise ValueError(
                    "number of elements in prompt_or_tokens_list and request_ids is different: "
                    f"{len(prompt_or_tokens_list)=} vs {len(request_ids)=}"
                )
            request_id_iter = iter(request_ids)
        else:
            request_id_iter = itertools.repeat(None)
        
        reqs = []
        for prompt, sampling_param, callback, request_id in zip(
            prompt_or_tokens_list, sampling_params_iter, stream_callback_iter, request_id_iter
        ):
            req = self.io_processor.preprocess(
                prompt,
                sampling_param,
                stream_callback=callback,
                request_id=request_id,
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
        request_ids: Optional[list[str]] = None,
    ) -> list[str]:
        # Reset round-robin counter to ensure consistent DP not core dump
        self.core_mgr._rr_counter = 0
        
        self.add_request(prompts, sampling_params, request_ids=request_ids)
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


    def wake_up(self, tags: List[str] = None):
        """
        Resume resources in GPU memory.
        """
        if tags is None:
            tags = ["weights", "kv_cache"]
        
        logger.info(f"LLMEngine wake_up: tags={tags}")
        self.core_mgr.broadcast_utility_command("resume_memory", tags=tags)

    def sleep(self, level: int = 1):
        """
        Release resources to free GPU memory.
        """
        logger.info(f"LLMEngine sleep: level={level}")
        
        if level >= 1:
            self.core_mgr.broadcast_utility_command(
                "release_memory", tags=["kv_cache"]
            )
        if level >= 2:
            self.core_mgr.broadcast_utility_command(
                "release_memory", tags=["weights"]
            )

    def load_weights(self, weights, bucket_size_mb: int = 2048):
        load_weights_via_shm(self.core_mgr, weights, bucket_size_mb)


class InputOutputProcessor:

    def __init__(self, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.requests = {}
        self._external_to_internal: dict[str, int] = {}
        self._internal_to_external: dict[int, str] = {}

    def preprocess(
        self,
        prompt_or_tokens: str | list[int],
        sampling_params: SamplingParams,
        stream_callback=None,
        request_id: Optional[str] = None,
    ):
        """responsible for:
        1) Tokenize
        2) Create Sequence object"""
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
                # Encode the full stop string as a sequence of tokens
                stop_tokens = self.tokenizer.encode(stop_str, add_special_tokens=False)
                if stop_tokens:
                    stop_token_sequences.append(stop_tokens)
        
        seq = Sequence(
            tokens,
            self.block_size,
            sampling_params,
            stop_token_sequences,
            stream_callback=stream_callback,
            request_id=request_id,
        )
        seq.arrive_time = time.time()
        self.requests[seq.id] = seq
        if request_id is not None:
            self._external_to_internal[request_id] = seq.id
            self._internal_to_external[seq.id] = request_id
        print(
            f"Request {seq.id} arrived, input tokens: {len(tokens)}, pending requests: {len(self.requests)}"
        )
        return seq

    def postprocess(self, reqs: List[Sequence]):
        """responsible for:
        1) Compute stats for logging
        2) Detokenize"""
        outputs = {}
        for req in reqs:
            self.requests.pop(req.id)
            external_request_id = self._internal_to_external.pop(req.id, None)
            if external_request_id is not None:
                self._external_to_internal.pop(external_request_id, None)
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

            print(
                f"Request {req.id} finished with reason {req.leave_reason}. "
                f"Input tokens: {req.num_prompt_tokens}, output tokens: {req.num_completion_tokens}, "
                f"latency: {req.leave_time - req.arrive_time:.2f}s, "
                f"TTFT: {ttft:.3f}s, TPOT: {tpot:.3f}s"
                # f"{req.completion_token_ids}"
            )
            outputs[req.id] = {
                "text": output_str,
                "token_ids": req.completion_token_ids,
                "logprobs": req.logprobs if req.return_logprobs else None,
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
