import logging
import time
from dataclasses import fields

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from atom.config import Config
from atom.model_engine.engine_core_mgr import CoreManager
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

logger = logging.getLogger("atom")


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos_token_id = self.tokenizer.eos_token_id

        self.rquest_ids = set()
        self.io_processor = InputOutputProcessor(
            self.tokenizer, config.kvcache_block_size
        )
        self.core_mgr = CoreManager(config)
        logger.info("LLMEngine init")

    def add_request(
        self, prompt_or_tokens: str | list[int], sampling_params: SamplingParams
    ):
        # 1. convet prompt to request
        req = self.io_processor.preprocess(prompt_or_tokens, sampling_params)

        # 2. add req to scheduler
        self.core_mgr.add_request(req)

    def step(self):
        # seqs, is_prefill = self.scheduler.schedule()
        # token_ids = self.model_runner.call("run", seqs, is_prefill)
        # self.scheduler.postprocess(seqs, token_ids)
        # outputs = [
        #     (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        # ]
        # num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        # return outputs, num_tokens

        seq = self.core_mgr.get_output()
        return seq

    def is_finished(self):
        return not self.io_processor.has_pending_requests()

    def generate(
        self,
        # prompts: list[str] | list[list[int]],
        prompts: list[str],
        sampling_params: SamplingParams | list[SamplingParams],
        enable_profiling: bool = False,
    ) -> list[str]:
        # # Start profiling for all ranks if enabled
        # if enable_profiling:
        #     self.model_runner.call("start_profiler")
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        while not self.is_finished() and (
            self.core_mgr.is_alive() or self.core_mgr.is_rest()
        ):
            seq = self.step()
            out = self.io_processor.postprocess(seq)
            outputs[seq.id] = out
        # # Stop profiling for all ranks if enabled
        # if enable_profiling:
        #     self.model_runner.call("stop_profiler")
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        return outputs


class InputOutputProcessor:

    def __init__(self, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.requests = {}

    def preprocess(
        self, prompt_or_tokens: str | list[int], sampling_params: SamplingParams
    ):
        """responsible for:
        1) Tokenize
        2) Create Sequence object"""
        tokens = (
            self.tokenizer.encode(prompt_or_tokens)
            if isinstance(prompt_or_tokens, str)
            else prompt_or_tokens
        )
        seq = Sequence(tokens, self.block_size, sampling_params)
        seq.arrive_time = time.time()
        self.requests[seq.id] = seq
        print(
            f"Request {seq.id} arrived, input tokens: {len(tokens)}, pending requests: {len(self.requests)}"
        )
        return seq

    def postprocess(self, req: Sequence):
        """responsible for:
        1) Compute stats for logging
        2) Detokenize"""
        self.requests.pop(req.id)
        output_str = self.tokenizer.decode(req.completion_token_ids)
        req.leave_time = time.time()
        print(
            f"Request {req.id} finished with reason {req.leave_reason}. "
            f"Input tokens: {req.num_prompt_tokens}, output tokens: {req.num_completion_tokens}, "
            f"latency: {req.leave_time - req.arrive_time:.2f}s"
        )
        return {
            "text": output_str,
            "token_ids": req.completion_token_ids,
            "latency": req.leave_time - req.arrive_time,
            "finish_reason": req.leave_reason,
            "num_tokens_input": req.num_prompt_tokens,
            "num_tokens_output": req.num_completion_tokens,
        }

    def has_pending_requests(self):
        return len(self.requests) > 0
