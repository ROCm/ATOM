# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import itertools
import logging
import time
from dataclasses import fields
from typing import List, Union, AsyncGenerator, Dict, Optional

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from atom.config import Config
from atom.model_engine.engine_core_mgr import CoreManager
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

logger = logging.getLogger("atom")


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        data_parallel_size = kwargs.get('data_parallel_size', 1)
        config = Config(model, **config_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.bos_token_id = self.tokenizer.bos_token_id
        config.eos_token_id = self.tokenizer.eos_token_id
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
        logger.info(f"LLMEngine init with {self.data_parallel_size} data parallel ranks")

    def add_request(
        self, 
        prompt_or_tokens_list: List[Union[str, List[int]]], 
        sampling_params_list: SamplingParams | List[SamplingParams],
        stream_callback=None
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
        for prompt, sampling_param, callback in zip(prompt_or_tokens_list, sampling_params_iter, stream_callback_iter):
            req = self.io_processor.preprocess(prompt, sampling_param, stream_callback=callback)
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
        logger.info("Profiling stopped. Trace files should be generated.")


class InputOutputProcessor:

    def __init__(self, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.requests = {}

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
            stops = [sampling_params.stop_strings] if isinstance(sampling_params.stop_strings, str) else sampling_params.stop_strings
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
            request_id_str=request_id,
        )
        seq.arrive_time = time.time()
        self.requests[seq.id] = seq
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
            output_str = self.tokenizer.decode(req.completion_token_ids)
            req.leave_time = time.time()
            
            # Calculate TTFT (Time To First Token) and TPOT (Time Per Output Token)
            ttft = 0.0
            tpot = 0.0
            if req.first_token_time > 0:
                ttft = req.first_token_time - req.arrive_time
                # Calculate TPOT only if there are multiple output tokens
                if req.num_completion_tokens > 1:
                    tpot = (req.leave_time - req.first_token_time) / (req.num_completion_tokens - 1)
            
            print(
                f"Request {req.id} finished with reason {req.leave_reason}. "
                f"Input tokens: {req.num_prompt_tokens}, output tokens: {req.num_completion_tokens}, "
                f"latency: {req.leave_time - req.arrive_time:.2f}s, "
                f"TTFT: {ttft:.3f}s, TPOT: {tpot:.3f}s"
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


class AsyncLLMEngine(LLMEngine):
    def __init__(self, *args, **kwargs):
        # Force asyncio mode so CoreManager builds async queues/handler.
        kwargs["asyncio_mode"] = True
        super().__init__(*args, **kwargs)
        self._running = True

    async def add_request_async(
        self,
        prompt_or_tokens_list: List[Union[str, List[int]]],
        sampling_params_list: SamplingParams | List[SamplingParams],
        stream_callback=None,
    ):
        """Add requests without blocking the event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.add_request,
            prompt_or_tokens_list,
            sampling_params_list,
            stream_callback,
        )

    async def step_async(self) -> list[Sequence]:
        """Async counterpart of step() using CoreManager async queue."""
        return await self.core_mgr.get_output_async()

    async def generate_batch_async(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[dict]:
        """Async version of generate that returns full outputs once ready."""
        await self.add_request_async(prompts, sampling_params)
        outputs: Dict[int, Dict] = {}

        while not self.is_finished() and (
            self.core_mgr.is_alive() or self.core_mgr.is_rest()
        ):
            seqs = await self.step_async()
            outs = self.io_processor.postprocess(seqs)
            outputs.update(outs)

        return [outputs[seq_id] for seq_id in sorted(outputs)]

    async def generate_async(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str
    ) -> AsyncGenerator[Dict, None]:
        """Stream incremental outputs via the engine's stream_callback without dropping tokens."""
        loop = asyncio.get_running_loop()
        output_queue: asyncio.Queue[RequestOutput] = asyncio.Queue()

        arrival_time = time.time()
        prompt_tokens = len(self.tokenizer.encode(prompt))
        first_token_time: float | None = None
        completion_token_ids: list[int] = []

        seq_id = None
        finished = False
        try:
            def stream_callback(request_output: RequestOutput) -> None:
                nonlocal first_token_time
                if first_token_time is None:
                    first_token_time = time.time()
                loop.call_soon_threadsafe(output_queue.put_nowait, request_output)

            req = await loop.run_in_executor(
                None,
                self.io_processor.preprocess,
                prompt,
                sampling_params,
                stream_callback,
                request_id,
            )
            seq_id = req.id
            self.core_mgr.add_request([req])

            while True:
                req_output = await output_queue.get()

                completion_token_ids.extend(req_output.output_tokens)
                text = self.tokenizer.decode(
                    completion_token_ids, skip_special_tokens=True
                )

                ttft = (
                    0.0
                    if first_token_time is None
                    else first_token_time - arrival_time
                )
                tpot = 0.0
                if first_token_time is not None and len(completion_token_ids) > 1:
                    elapsed_since_first = time.time() - first_token_time
                    tpot = elapsed_since_first / (len(completion_token_ids) - 1)

                latency = time.time() - arrival_time if req_output.finished else 0.0

                result = {
                    "text": text,
                    "token_ids": completion_token_ids.copy(),
                    "finished": req_output.finished,
                    "finish_reason": req_output.finish_reason,
                    "ttft": ttft,
                    "tpot": tpot,
                    "latency": latency,
                    "num_tokens_input": prompt_tokens,
                    "num_tokens_output": len(completion_token_ids),
                }

                yield result

                finished = req_output.finished
                if finished:
                    break

        finally:
            if seq_id is not None:
                if not finished:
                    self.io_processor.requests.pop(seq_id, None)
    
    #TODO add tokenizer process to handle tokens combination for utf8 decoding
    def _postprocess_sequence(self, seq: Sequence) -> Dict:
        # Set leave_time when sequence is finished
        if seq.is_finished and seq.leave_time == 0.0:
            seq.leave_time = time.time()
        
        # Remove from pending requests when sequence is finished
        if seq.is_finished:
            self.io_processor.requests.pop(seq.id, None)

        ttft = 0.0
        tpot = 0.0
        latency = 0.0
        
        if seq.first_token_time > 0:
            ttft = seq.first_token_time - seq.arrive_time
            if seq.num_completion_tokens > 1:
                leave_time = seq.leave_time if seq.leave_time > 0 else time.time()
                tpot = (leave_time - seq.first_token_time) / (seq.num_completion_tokens - 1)
        
        # Calculate latency if sequence is finished
        if seq.is_finished and seq.leave_time > 0:
            latency = seq.leave_time - seq.arrive_time
        
        return {
            "text": self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=True),
            "token_ids": seq.completion_token_ids.copy(), 
            "finished": seq.is_finished,
            "finish_reason": getattr(seq, "leave_reason", None),
            "ttft": ttft,
            "tpot": tpot,
            "latency": latency,
        }

    # TODO enable abort request (needs request_id to seq cleanup if added)
    
    async def close(self):
        self._running = False
        if self._output_task is not None:
            self._output_task.cancel()
            try:
                await self._output_task
            except asyncio.CancelledError:
                pass
        self.core_mgr.close()