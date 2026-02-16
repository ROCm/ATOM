# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
import time
from itertools import chain
from typing import Any, Optional, Union
import math

import numpy as np
import torch
import torch.profiler as torch_profiler
import tqdm
from aiter import destroy_dist_env, dtypes, init_dist_env
from aiter.dist.parallel_state import (
    get_dp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
)
from aiter.dist.utils import get_distributed_init_method
from atom.config import Config, KVCacheTensor, set_current_atom_config
from atom.model_engine.scheduler import ScheduledBatch, ScheduledBatchOutput
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType
from atom.model_loader.loader import load_model
from atom.model_ops.rejection_sampler import RejectionSampler
from atom.model_ops.sampler import Sampler
from atom.spec_decode.eagle import EagleProposer
from atom.utils import (
    CpuGpuBuffer,
    get_hf_text_config,
    init_exit_handler,
    resolve_obj_by_qualname,
)
from atom.utils.forward_context import (
    Context,
    DPMetadata,
    SpecDecodeMetadata,
    get_forward_context,
    reset_forward_context,
    set_forward_context,
    set_kv_cache_data,
)
from atom.utils.selector import get_attn_backend

logger = logging.getLogger("atom")

support_model_arch_dict = {
    "Qwen3ForCausalLM": "atom.models.qwen3.Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM": "atom.models.qwen3_moe.Qwen3MoeForCausalLM",
    "LlamaForCausalLM": "atom.models.llama.LlamaForCausalLM",
    "MixtralForCausalLM": "atom.models.mixtral.MixtralForCausalLM",
    "DeepseekV3ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "DeepseekV32ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "GptOssForCausalLM": "atom.models.gpt_oss.GptOssForCausalLM",
    "Glm4MoeForCausalLM": "atom.models.glm4_moe.Glm4MoeForCausalLM",
    "Qwen3NextForCausalLM": "atom.models.qwen3_next.Qwen3NextForCausalLM",
}
# seed = 34567
# np.random.seed(seed)
# torch.cuda.manual_seed_all(seed)


class tokenIDProcessor:

    def __init__(
        self,
        max_num_batched_tokens: int,
        device: torch.device,
        use_spec: bool = False,
        num_speculative_tokens: int = 0,
    ):
        """Asynchronously copy the sampled_token_ids tensor to the host."""
        # self.is_deferred_out = False
        self.is_deferred_out = True
        self.input_ids = CpuGpuBuffer(
            max_num_batched_tokens + 1, dtype=torch.int32, device=device
        )
        self.input_ids_loc = CpuGpuBuffer(
            max_num_batched_tokens, dtype=torch.int64, device=device
        )
        self.use_spec = use_spec
        if self.use_spec:
            self.num_speculative_tokens = num_speculative_tokens

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_event = torch.cuda.Event()
        self.async_copy_stream = torch.cuda.Stream()
        self.clean()

    def send_to_cpu_async(
        self,
        gpu_tensor: torch.Tensor,
        cpu_tensor_handle,
        data_ready: torch.cuda.Event,
        copy_done: Optional[torch.cuda.Event] = None,
    ):
        copy_done = copy_done or torch.cuda.Event()
        with torch.cuda.stream(self.async_copy_stream):
            data_ready.wait(stream=self.async_copy_stream)
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            copy_done.record(self.async_copy_stream)
        cpu_tensor_handle.append((cpu_tensor, copy_done))

    def recv_async_output(self, cpu_tensor_handle) -> list[int]:
        if not cpu_tensor_handle:
            return []
        cpu_tensor, event = cpu_tensor_handle.pop(0)
        event.synchronize()
        token_ids = cpu_tensor.tolist()
        return token_ids

    def send_to_cpu_async_draft(self, gpu_tensor: torch.Tensor):
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.async_copy_stream):
            self.async_copy_stream.wait_stream(default_stream)
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.async_copy_stream)
        self.draft_token_ids_cpu.append((cpu_tensor, event))

    def recv_async_output_draft(self) -> list[int]:
        if not self.draft_token_ids_cpu:
            return []
        token_ids, event = self.draft_token_ids_cpu.pop(0)
        event.synchronize()
        return token_ids.tolist()

    def send_bonus_to_cpu_async(self, gpu_tensor: torch.Tensor, req_ids: list[int]):
        default_stream = torch.cuda.current_stream()
        self.bonus_req_ids.append(req_ids)
        with torch.cuda.stream(self.async_copy_stream):
            self.async_copy_stream.wait_stream(default_stream)
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            self.async_copy_event.record(self.async_copy_stream)
        self.bonus_tokens_cpu.append(cpu_tensor)

    def recv_bonus_async(self) -> Optional[np.ndarray]:
        if not self.bonus_tokens_cpu:
            return None
        self.async_copy_event.synchronize()
        bonus_list = self.bonus_tokens_cpu.pop(0).numpy()
        req_ids = self.bonus_req_ids.pop(0)
        for i, req_id in enumerate(req_ids):
            self.bonus_tokens_map[req_id] = bonus_list[i]
        return self.bonus_tokens_map

    def clean(self):
        self.token_ids_cpu: list[torch.Tensor] = []

        self.prev_batch: Optional[ScheduledBatch] = None

        self.pre_num_decode_token_per_seq = 1
        self.draft_token_ids: Optional[torch.Tensor] = None
        self.draft_token_ids_cpu: list[torch.Tensor] = []
        self.bonus_tokens_cpu: list[torch.Tensor] = (
            []
        )  # Async queue for num_bonus_tokens
        self.bonus_tokens_map: map[int, list[int]] = {}
        self.bonus_req_ids: list[list[int]] = []
        self.mapped_bonus_list: Optional[list[int]] = (
            None  # Mapped to current batch order
        )

    def _process_token_id(self, token_id) -> tuple[int, ...]:
        """Helper function: process a single token_id, handling list and non-list cases.

        Optimized: eliminates double traversal (removed 'in' check before 'index').
        Returns tuple for better performance and immutability.
        """
        if isinstance(token_id, list):
            try:
                idx = token_id.index(-1)
                return tuple(token_id[:idx])
            except ValueError:
                # No -1 found, return the entire list as tuple
                return tuple(token_id)
        else:
            return (token_id,)

    def prepare_sampled_ids(
        self,
        batch: ScheduledBatch,
        sampled_token_ids: torch.Tensor,
        sync_event: torch.cuda.Event,
    ) -> dict[int, tuple[int, ...]]:
        if not self.is_deferred_out:
            token_ids = sampled_token_ids.tolist()
            req_ids = batch.req_ids
            ret = {
                seq_id: self._process_token_id(token_id)
                for seq_id, token_id in zip(req_ids, token_ids)
            }
            ret[-1] = 0  # is_deferred_out flag
            return ret

        token_ids = self.recv_async_output(self.token_ids_cpu)
        self.send_to_cpu_async(sampled_token_ids, self.token_ids_cpu, sync_event)
        token_id_dict = {}
        self.prev_req_ids = None
        if self.prev_batch is not None:
            self.prev_req_ids = self.prev_batch.req_ids
            token_id_dict = {
                seq_id: self._process_token_id(token_id)
                for seq_id, token_id in zip(self.prev_req_ids, token_ids)
            }
        else:
            # first time, no previous tokens
            token_ids = {}

        self.prev_batch = batch
        self.prev_token_ids = sampled_token_ids
        token_id_dict[-1] = 1

        return token_id_dict

    def get_token_locations(
        self, batch: ScheduledBatch
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        prev_req_ids = self.prev_batch.req_ids
        cur_req_ids = batch.req_ids
        num_prev = len(prev_req_ids)
        num_cur = len(cur_req_ids)

        prev_id_to_idx = dict(zip(prev_req_ids, range(num_prev)))

        deferred_curr = np.empty(num_cur, dtype=np.intp)
        deferred_prev = np.empty(num_cur, dtype=np.intp)
        new_curr = np.empty(num_cur, dtype=np.intp)
        n_deferred = 0
        n_new = 0

        for cur_idx in range(num_cur):
            prev_idx = prev_id_to_idx.get(cur_req_ids[cur_idx])
            if prev_idx is not None:
                deferred_curr[n_deferred] = cur_idx
                deferred_prev[n_deferred] = prev_idx
                n_deferred += 1
            else:
                new_curr[n_new] = cur_idx
                n_new += 1

        deferred_curr = deferred_curr[:n_deferred]
        deferred_prev = deferred_prev[:n_deferred]
        new_curr = new_curr[:n_new]

        is_all_same = (
            n_new == 0
            and n_deferred == num_prev
            and np.array_equal(deferred_curr, deferred_prev)
        )

        return deferred_curr, deferred_prev, new_curr, is_all_same

    def prepare_input_ids(
        self,
        batch: ScheduledBatch,
    ) -> torch.Tensor:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""
        scheduled_tokens = batch.scheduled_tokens  # tokens per req
        token_nums = batch.num_scheduled_tokens
        total_tokens = batch.total_tokens_num
        total_tokens_prefill = batch.total_tokens_num_prefill
        total_tokens_decode = batch.total_tokens_num_decode
        total_reqs_prefill = batch.total_seqs_num_prefill
        total_reqs_decode = batch.total_seqs_num_decode
        """for prefill: all input ids are new"""
        start_loc = 0
        for tokens, new_token_num in zip(
            scheduled_tokens[:total_reqs_prefill], token_nums[:total_reqs_prefill]
        ):
            self.input_ids.np[start_loc : start_loc + new_token_num] = tokens
            start_loc += new_token_num
        self.input_ids.copy_to_gpu(total_tokens_prefill)

        # TODO: remove this when we support mixed prefill and decode in one batch
        if total_reqs_prefill > 0:
            return self.input_ids.gpu[:total_tokens_prefill]

        if not self.is_deferred_out:
            if self.use_spec:
                scheduled_tokens = scheduled_tokens[
                    total_reqs_prefill : total_reqs_prefill + total_reqs_decode
                ]
                draft_values = batch.scheduled_spec_decode_tokens.values()
                mtp_k = self.num_speculative_tokens
                token_ids = np.fromiter(
                    chain.from_iterable(
                        (tokens[0], *draft)
                        for tokens, draft in zip(scheduled_tokens, draft_values)
                    ),
                    dtype=np.int32,
                    count=total_reqs_decode * (mtp_k + 1),
                )
            else:
                token_ids = [
                    token
                    for tokens in scheduled_tokens[
                        total_reqs_prefill : total_reqs_prefill + total_reqs_decode
                    ]
                    for token in tokens
                ]
            self.input_ids.np[:total_tokens_decode] = token_ids
            return self.input_ids.copy_to_gpu(total_tokens_decode)

        """for decode: input ids are from prev_sampled_token_ids"""
        deferred_curr_indices, deferred_prev_indices, new_curr_indices, is_all_same = (
            self.get_token_locations(batch)
        )
        num_deferred_seqs = len(deferred_curr_indices)
        num_new_seqs = len(new_curr_indices)

        # Calculate token counts: in MTP mode, each seq has multiple tokens
        if self.use_spec:
            tokens_per_seq = self.num_speculative_tokens + 1
            num_deferred_tokens = num_deferred_seqs * tokens_per_seq
            num_new_tokens = num_new_seqs * tokens_per_seq
        else:
            tokens_per_seq = 1
            num_deferred_tokens = num_deferred_seqs
            num_new_tokens = num_new_seqs

        # Receive and map bonus_list to current batch order
        prev_bonus_map = self.recv_bonus_async()
        total_seqs = num_deferred_seqs + num_new_seqs
        if prev_bonus_map is not None and num_deferred_seqs > 0:
            # Map: prev_bonus_list[prev_idx] → mapped_bonus_list[curr_idx]
            # New seqs get default value (tokens_per_seq - 1)

            # -1 means this seq do not have prev bonus which means its prefill
            # otherwise, all seq should have prev bonus result
            self.mapped_bonus_list = [-1] * total_seqs
            for i, req_id in enumerate(batch.req_ids):
                if req_id in prev_bonus_map:
                    self.mapped_bonus_list[i] = prev_bonus_map.pop(req_id)
        else:
            self.mapped_bonus_list = None

        if is_all_same:
            # All requests are the same, only deferred tokens
            if self.use_spec:
                # MTP mode: combine prev_token_ids and draft_token_ids
                if (
                    self.draft_token_ids is not None
                    and self.pre_num_decode_token_per_seq > 1
                ):
                    combined = torch.cat(
                        [
                            self.prev_token_ids.unsqueeze(1),  # (num_seqs, 1)
                            self.draft_token_ids,  # (num_seqs, mtp_n_grams-1)
                        ],
                        dim=1,
                    ).reshape(
                        -1
                    )  # (num_deferred_tokens,)
                else:
                    combined = self.prev_token_ids
                self.input_ids.gpu[:num_deferred_tokens] = combined
            else:
                # Non-MTP mode: only prev_token_ids
                self.input_ids.gpu[:num_deferred_tokens] = self.prev_token_ids
        else:
            """
            (1) prev_batch=[301], cur_batch=[0..255, 301] → Layout: [301 prefill | new | deferred]
            (2) prev_batch=[0..255], cur_batch=[0..253, 256, 257] → Layout: [deferred | new 256, 257] when conc > max_num_seq
            """
            is_prev_prefill = self.prev_batch.total_tokens_num_prefill > 0
            new_decode_front = (
                is_prev_prefill
                and np.array_equal(new_curr_indices, np.arange(num_new_seqs))
                and np.array_equal(
                    deferred_curr_indices,
                    np.arange(num_new_seqs, num_new_seqs + num_deferred_seqs),
                )
            )

            gathered_tokens = None
            # old requests (deferred)
            if num_deferred_seqs > 0:
                self.input_ids_loc.np[:num_deferred_seqs] = deferred_prev_indices
                deferred_indices_gpu = self.input_ids_loc.copy_to_gpu(num_deferred_seqs)
                gathered_prev = torch.gather(
                    self.prev_token_ids,
                    0,
                    deferred_indices_gpu,
                )
                if self.use_spec:
                    # MTP mode: combine prev_token_ids and draft_token_ids
                    if (
                        self.draft_token_ids is not None
                        and self.pre_num_decode_token_per_seq > 1
                    ):
                        # draft_token_ids is 2D (num_seqs, mtp_n_grams-1), use direct indexing
                        gathered_draft = self.draft_token_ids[deferred_indices_gpu]
                        gathered_tokens = torch.cat(
                            [
                                gathered_prev.unsqueeze(1),  # (num_deferred_seqs, 1)
                                gathered_draft,  # (num_deferred_seqs, mtp_n_grams-1)
                            ],
                            dim=1,
                        ).reshape(
                            -1
                        )  # (num_deferred_tokens,)
                    else:
                        # normal decode (fallback)
                        gathered_tokens = gathered_prev
                else:
                    # Non-MTP mode: only prev_token_ids
                    gathered_tokens = gathered_prev

            if new_decode_front:
                # Layout: [new | deferred]
                if gathered_tokens is not None:
                    self.input_ids.gpu[
                        num_new_tokens : num_new_tokens + num_deferred_tokens
                    ] = gathered_tokens
                if num_new_tokens > 0:
                    if self.use_spec:
                        # MTP mode: combine scheduled_tokens and draft_tokens
                        # new_decode_front=True means new_curr_indices == list(range(num_new_seqs))
                        # so we can use sequential indexing
                        token_ids = [
                            token
                            for tokens in scheduled_tokens[
                                total_reqs_prefill : total_reqs_prefill + num_new_seqs
                            ]
                            for token in tokens
                        ]
                        draft_values = batch.scheduled_spec_decode_tokens.values()

                        self.input_ids.np[:num_new_tokens] = token_ids
                        self.input_ids.copy_to_gpu(num_new_tokens)
                    else:
                        # Non-MTP mode: flatten scheduled_tokens for new requests
                        token_ids = [
                            token
                            for tokens in scheduled_tokens[
                                total_reqs_prefill : total_reqs_prefill + num_new_seqs
                            ]
                            for token in tokens
                        ]
                        self.input_ids.np[:num_new_tokens] = token_ids
                        self.input_ids.copy_to_gpu(num_new_tokens)
            else:
                # Layout: [deferred | new] - deferred at front, new is from previous finished prefill and waiting for decode
                if num_new_tokens > 0:
                    if self.use_spec:
                        # MTP mode: combine scheduled_tokens and draft_tokens
                        # For new_decode_front=False, use new_curr_indices to get the right sequences
                        new_token_ids = []
                        for idx in new_curr_indices:
                            req_idx = total_reqs_prefill + idx
                            tokens = scheduled_tokens[req_idx]
                            new_token_ids.extend(tokens)
                    else:
                        # Non-MTP mode: only first token from scheduled_tokens
                        new_token_ids = [
                            scheduled_tokens[idx][0] for idx in new_curr_indices
                        ]
                    self.input_ids.np[:num_new_tokens] = new_token_ids
                    self.input_ids.gpu[
                        num_deferred_tokens : num_deferred_tokens + num_new_tokens
                    ].copy_(self.input_ids.cpu[:num_new_tokens], non_blocking=True)
                if gathered_tokens is not None:
                    self.input_ids.gpu[:num_deferred_tokens] = gathered_tokens

        return self.input_ids.gpu[:total_tokens]

    def prepare_draft_ids(
        self, batch: ScheduledBatch, draft_token_ids: torch.Tensor
    ) -> dict[int, list[int]]:
        if not self.is_deferred_out:
            draft_token_ids = draft_token_ids.tolist()
            req_ids = batch.req_ids
            ret = {
                seq_id: token_id for seq_id, token_id in zip(req_ids, draft_token_ids)
            }
            ret[-1] = 0
        else:
            self.draft_token_ids = draft_token_ids
            self.pre_num_decode_token_per_seq = self.num_speculative_tokens + 1
            token_ids = self.recv_async_output_draft()
            self.send_to_cpu_async_draft(draft_token_ids)
            ret = (
                {
                    seq_id: token_id
                    for seq_id, token_id in zip(self.prev_req_ids, token_ids)
                }
                if self.prev_req_ids is not None
                else {}
            )
            ret[-1] = 1
        return ret


class ModelRunner:

    def __init__(self, rank: int, config: Config):
        self.config = config
        set_current_atom_config(config)
        hf_config = config.hf_config
        self.block_size = config.kv_cache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.label = f"Model Runner{rank}/{self.world_size}"
        self.hf_text_config = get_hf_text_config(hf_config)
        if self.hf_text_config.model_type in ["llama"] and self.config.torch_dtype in [
            torch.bfloat16,
            torch.float16,
        ]:
            os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"
        self.use_mla = self.is_deepseek_mla()
        self.use_gdn = self.is_qwen_next()
        self.is_deepseek_v32 = (
            hasattr(hf_config, "index_topk") if self.use_mla else False
        )
        # Calculate local device rank considering both TP and DP
        # When data parallelism is enabled on the same node, different DP ranks
        # need to use different sets of GPUs
        dp_rank_local = config.parallel_config.data_parallel_rank_local
        if dp_rank_local is None:
            dp_rank_local = 0
        local_device_rank = dp_rank_local * config.tensor_parallel_size + rank
        num_gpus = torch.cuda.device_count()
        if local_device_rank >= num_gpus:
            raise ValueError(
                f"Calculated local_device_rank={local_device_rank} exceeds available GPUs ({num_gpus}). "
            )

        device = torch.device(f"cuda:{local_device_rank}")
        logger.info(
            f"ModelRunner rank={rank}, dp_rank_local={dp_rank_local}, local_device_rank={local_device_rank}, device={device}"
        )
        self.device = device

        # Initialize profiler for this rank
        self.profiler = None
        self.profiler_dir = None
        if config.torch_profiler_dir is not None:
            # Create rank-specific profiler directory
            if dp_rank_local > 0 or config.parallel_config.data_parallel_size > 1:
                rank_name = f"dp{dp_rank_local}_tp{rank}"
            else:
                rank_name = f"rank_{rank}"
            self.profiler_dir = os.path.join(config.torch_profiler_dir, rank_name)
            os.makedirs(self.profiler_dir, exist_ok=True)

        self.graph_bs = [0]  # for eager fallback

        torch.cuda.set_device(self.device)
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = str(self.config.port)
        distributed_init_method = get_distributed_init_method(
            config.parallel_config.data_parallel_master_ip,
            config.parallel_config.data_parallel_base_port,
        )
        init_dist_env(
            config.tensor_parallel_size,
            rankID=rank,
            backend="nccl",
            distributed_init_method=distributed_init_method,
            data_parallel_size=config.parallel_config.data_parallel_size,
            data_parallel_rank=config.parallel_config.data_parallel_rank,
        )
        init_exit_handler(self)
        default_dtype = self.config.torch_dtype
        torch.set_default_dtype(default_dtype)
        torch.set_default_device(self.device)
        self.attn_backend = get_attn_backend(
            self.block_size,
            use_mla=self.use_mla,
            use_gdn=self.use_gdn,
        )
        if self.config.speculative_config and get_pp_group().is_last_rank:
            from atom.utils.backends import set_model_tag

            with set_model_tag("drafter"):
                self.drafter = EagleProposer(self.config, self.device, self)
            self.rejection_sampler = RejectionSampler()
            self.mtp_total_draft_tokens = 0
            self.mtp_total_accepted_tokens = 0
        self.num_speculative_tokens = (
            self.drafter.mtp_k if hasattr(self, "drafter") else 0
        )
        self.tokenID_processor = tokenIDProcessor(
            self.config.max_num_batched_tokens,
            self.device,
            hasattr(self, "drafter"),
            self.num_speculative_tokens,
        )
        self.sampler = Sampler()
        self.arange_np = np.arange(
            max(
                self.config.max_num_seqs + 1,
                self.config.max_model_len,
                self.config.max_num_batched_tokens,
            ),
            dtype=np.int64,
        )

        model_class = resolve_obj_by_qualname(support_model_arch_dict[hf_config.architectures[0]])  # type: ignore
        self.model = model_class(config)
        torch.set_default_device(None)
        load_model(self.model, config.model, config.hf_config, config.load_dummy)
        logger.info(f"Model load done: {config.model}")

        if hasattr(self, "drafter"):
            logger.info("Loading drafter model...")
            self.drafter.load_model(self.model)
        torch.set_default_device(self.device)
        self.allocate_forward_vars()
        self.attn_metadata_builder = self.attn_backend.get_builder_cls()(self)
        self.physical_block_size = self.attn_metadata_builder.block_size
        self.forward_done_event = torch.cuda.Event()
        self.warmup_model()
        logger.info(f"Model warmup done: {config.model}")

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.config.compilation_config.level == 1:
            self.model = torch.compile(self.model, fullgraph=True, backend="eager")

    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_mtp",
        ):
            return self.hf_text_config.kv_lora_rank is not None
        elif self.hf_text_config.model_type == "eagle":
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return (
                self.hf_text_config.model.model_type in ("deepseek_v2", "deepseek_v3")
                and self.hf_text_config.kv_lora_rank is not None
            )
        return False

    def is_qwen_next(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in ("qwen3_next", "qwen3_next_mtp"):
            return True
        return False

    def get_mtp_statistics(self) -> dict:
        if hasattr(self, "mtp_total_draft_tokens"):
            acceptance_rate = (
                self.mtp_total_accepted_tokens / self.mtp_total_draft_tokens
                if self.mtp_total_draft_tokens > 0
                else 0.0
            )
            return {
                "total_draft_tokens": self.mtp_total_draft_tokens,
                "total_accepted_tokens": self.mtp_total_accepted_tokens,
                "acceptance_rate": acceptance_rate,
            }
        return {
            "total_draft_tokens": 0,
            "total_accepted_tokens": 0,
            "acceptance_rate": 0.0,
        }

    def reset_mtp_statistics(self):
        if hasattr(self, "mtp_total_draft_tokens"):
            self.mtp_total_draft_tokens = 0
            self.mtp_total_accepted_tokens = 0

    def _make_buffer(
        self, *size: Union[int, torch.SymInt], dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        # Bfloat16 torch tensors cannot be directly cast to a numpy array, so
        # if a bfloat16 buffer is needed without a corresponding numpy array,
        # don't bother instantiating the numpy array.
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=True, with_numpy=numpy
        )

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: Optional[np.dtype] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        if not self.enforce_eager:
            self.graphs = self.graph_pool = None  # type: ignore
        destroy_dist_env()
        return True

    def start_profiler(self):
        """
        Start profiling for this rank.

        The ATOM_PROFILER_MORE environment variable controls detailed profiling features:
        - Set to "1" to enable record_shapes, with_stack, and profile_memory.
        - Set to "0" or unset to disable these features (default).
        """
        if self.profiler_dir is not None and self.profiler is None:
            enable_detailed_profiling = os.environ.get("ATOM_PROFILER_MORE", "0") == "1"
            self.profiler = torch_profiler.profile(
                activities=[
                    torch_profiler.ProfilerActivity.CPU,
                    torch_profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=enable_detailed_profiling,
                with_stack=enable_detailed_profiling,
                profile_memory=enable_detailed_profiling,
                on_trace_ready=torch_profiler.tensorboard_trace_handler(
                    self.profiler_dir, use_gzip=True
                ),
            )
            self.profiler.__enter__()

    def stop_profiler(self):
        """Stop profiling for this rank"""
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
        return True

    def debug(self, *args: Any):
        if self.rank == 0:
            logger.info(*args)

    def dummy_execution(self):
        """Execute dummy decode batch for DP synchronization."""
        num_tokens_original = 1

        seq = Sequence([0] * num_tokens_original, block_size=self.block_size)
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.DECODE
        seq.block_table = [0]
        bs = 1

        dummy_batch = ScheduledBatch(
            seqs={seq.id: seq},
            num_scheduled_tokens=np.array([num_tokens_original], dtype=np.int32),
            total_tokens_num=num_tokens_original,  # original value
            total_tokens_num_decode=num_tokens_original,
            total_seqs_num=1,
            total_seqs_num_decode=1,
            is_dummy_run=True,
        )

        bs = self.prepare_inputs(dummy_batch)
        actual_num_tokens = dummy_batch.total_tokens_num

        # self.tokenID_processor.input_ids.np[:actual_num_tokens] = [0] * actual_num_tokens
        # self.tokenID_processor.input_ids.copy_to_gpu(actual_num_tokens)
        # input_ids = self.tokenID_processor.input_ids.gpu[:actual_num_tokens]
        # input_ids = torch.zeros(actual_num_tokens, dtype=torch.int32, device=self.device)
        self.forward_vars["input_ids"].gpu[:bs].zero_()
        input_ids = self.forward_vars["input_ids"].gpu[:bs]

        self.run_model(input_ids)

        reset_forward_context()
        logger.debug(
            f"{self.label}: dummy batch executed with {actual_num_tokens} tokens"
        )
        return True

    def dummy_prefill_execution(self, num_tokens: int):
        """
        Execute dummy prefill batch for DP synchronization.
        """
        if num_tokens <= 0:
            num_tokens = 1
        seq = Sequence([0] * num_tokens, block_size=self.block_size)
        seqs = {seq.id: seq}

        dummy_batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=np.array([num_tokens], dtype=np.int32),
            total_tokens_num=num_tokens,
            total_tokens_num_prefill=num_tokens,
            total_seqs_num=1,
            total_seqs_num_prefill=1,
            is_dummy_run=True,
        )

        bs = self.prepare_inputs(dummy_batch)

        # self.tokenID_processor.input_ids.np[:num_tokens] = [0] * num_tokens
        # self.tokenID_processor.input_ids.copy_to_gpu(num_tokens)
        # input_ids = self.tokenID_processor.input_ids.gpu[:num_tokens]
        # input_ids= torch.zeros(num_tokens, dtype=torch.int32, device=self.device)
        self.forward_vars["input_ids"].gpu[:bs].zero_()
        input_ids = self.forward_vars["input_ids"].gpu[:bs]

        # not exe run_model and synchronize: acc 0.79

        with torch.no_grad():
            self.run_model(input_ids)

        torch.cuda.synchronize()

        reset_forward_context()

        logger.info(
            f"{self.label}: dummy PREFILL batch executed with {num_tokens} tokens"
        )
        return True

    def warmup_model(self):
        start_time = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        dp_size = get_dp_group().world_size
        warmup_max_tokens = max_num_batched_tokens // dp_size

        num_seqs = min(warmup_max_tokens // max_model_len, self.config.max_num_seqs)

        if num_seqs == 0:
            num_seqs = 1
            seq_len = min(warmup_max_tokens, max_model_len)
            if seq_len == 0:
                seq_len = 1
            logger.warning(
                f"{self.label}: DP size={dp_size} too large, warmup_max_tokens={warmup_max_tokens} < max_model_len={max_model_len}. "
                f"Using {num_seqs} seq with length {seq_len} for warmup."
            )
        else:
            seq_len = max_model_len

        seqs = [
            Sequence([0] * seq_len, block_size=self.block_size) for _ in range(num_seqs)
        ]
        seqs = {seq.id: seq for seq in seqs}

        num_scheduled_tokens = np.array([seq_len] * num_seqs, dtype=np.int32)
        total_tokens_num = int(num_scheduled_tokens.sum())

        dummy_batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_tokens_num=total_tokens_num,
            total_tokens_num_prefill=total_tokens_num,
            total_seqs_num=num_seqs,
            total_seqs_num_prefill=num_seqs,
            is_dummy_run=True,
        )
        self.forward(dummy_batch)
        self.tokenID_processor.clean()
        torch.cuda.empty_cache()
        logger.info(
            f"{self.label}: warmup_model {time.time() - start_time:.2f} seconds with {num_seqs} reqs {total_tokens_num} tokens"
        )

    def allocate_forward_vars(self):
        config = self.config
        hidden_size = config.hf_config.hidden_size
        hidden_type = config.torch_dtype
        self.max_bs = self.config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        i32_kwargs = {"dtype": torch.int32, "device": self.device}
        f32_kwargs = {"dtype": torch.float, "device": self.device}

        # TODO: remove it in forward_context
        self.forward_vars = {
            "input_ids": self.tokenID_processor.input_ids,
            "positions": CpuGpuBuffer(self.max_num_batched_tokens, **i64_kwargs),
            "temperatures": CpuGpuBuffer(self.max_bs, **f32_kwargs),
            # Keep enough space for MTP decode (max_q_len > 1).
            "outputs": torch.empty(
                self.max_num_batched_tokens, hidden_size, dtype=hidden_type
            ),
        }
        if hasattr(self, "drafter"):
            self.forward_vars["mtp_k"] = self.drafter.mtp_k
            self.forward_vars["num_accepted_tokens"] = CpuGpuBuffer(
                self.max_bs, **i32_kwargs
            )

    def get_num_blocks(self):
        torch.set_default_device(self.device)
        config = self.config
        hf_config = config.hf_config
        if not hasattr(hf_config, "head_dim") or hf_config.head_dim is None:
            hf_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        planned = int(total * config.gpu_memory_utilization)
        torch.set_default_device("cpu")
        if hf_config.num_key_value_heads >= self.world_size:
            assert hf_config.num_key_value_heads % self.world_size == 0
            num_kv_heads = hf_config.num_key_value_heads // self.world_size
        else:
            assert self.world_size % hf_config.num_key_value_heads == 0
            num_kv_heads = 1
        if self.use_mla:
            block_bytes = (
                hf_config.num_hidden_layers
                * self.block_size
                * 576
                * dtypes.d_dtypes[config.kv_cache_dtype].itemsize
            )
            if self.is_deepseek_v32:
                index_dim = hf_config.index_head_dim + 4
                aligned_index_dim = ((index_dim + 15) // 16) * 16
                block_bytes += (
                    hf_config.num_hidden_layers
                    * self.block_size
                    * aligned_index_dim
                    * dtypes.fp8.itemsize
                )
        elif self.is_qwen_next():
            self.full_attention_interval = hf_config.full_attention_interval
            self.num_full_attn = (
                hf_config.num_hidden_layers // self.full_attention_interval
            )
            self.num_gdn_attn_state = hf_config.num_hidden_layers - self.num_full_attn

            # full attention bytes
            block_bytes = (
                2
                * hf_config.num_hidden_layers
                * self.physical_block_size
                * num_kv_heads
                * hf_config.head_dim
                * dtypes.d_dtypes[config.kv_cache_dtype].itemsize
            )

            # gdn attn bytes
            mamba_shape = self.gated_delta_net_state_shape(
                get_tp_group().world_size,
                hf_config.linear_num_key_heads,
                hf_config.linear_num_value_heads,
                hf_config.linear_key_head_dim,
                hf_config.linear_value_head_dim,
                hf_config.linear_conv_kernel_dim,
                self.num_speculative_tokens,
            )

            one_layer_byte = (
                sum(math.prod(subtuple) for subtuple in mamba_shape)
                * dtypes.d_dtypes[config.kv_cache_dtype].itemsize
            )
            block_bytes = block_bytes + self.num_gdn_attn_state * one_layer_byte
        else:
            block_bytes = (
                2
                * hf_config.num_hidden_layers
                * self.block_size
                * num_kv_heads
                * hf_config.head_dim
                * dtypes.d_dtypes[config.kv_cache_dtype].itemsize
            )
        available_for_kv = min((planned - max(peak, current)), free)
        num_kvcache_blocks = available_for_kv // block_bytes
        assert num_kvcache_blocks > 0, (
            f"Not enough memory for KV cache with block size({self.block_size}). "
            f"At least 1 block ({block_bytes / (1 << 20):.2f}MB) is required, "
            f"but available memory is {free / (1 << 20):.2f}MB "
            f"(planned: {planned / (1 << 30):.2f}GB ({total/ (1 << 30):.2f}GB*{config.gpu_memory_utilization}), "
            f"used: {used / (1 << 30):.2f}GB, "
            f"peak: {peak / (1 << 30):.2f}GB, "
            f"current: {current / (1 << 30):.2f}GB)"
        )
        return num_kvcache_blocks

    def allocate_kv_cache(self, num_kvcache_blocks):
        config = self.config
        config.num_kvcache_blocks = num_kvcache_blocks
        hf_config = config.hf_config
        self.num_physical_kvcache_blocks = (
            num_kvcache_blocks * self.attn_metadata_builder.block_ratio
        )
        if hf_config.num_key_value_heads >= self.world_size:
            assert hf_config.num_key_value_heads % self.world_size == 0
            num_kv_heads = hf_config.num_key_value_heads // self.world_size
        else:
            assert self.world_size % hf_config.num_key_value_heads == 0
            num_kv_heads = 1

        # Calculate total number of layers (target + draft)
        total_num_layers = hf_config.num_hidden_layers
        num_draft_layers = 0
        if self.config.speculative_config and hasattr(self, "drafter"):
            draft_hf_config = self.config.speculative_config.draft_model_hf_config
            # For MTP, use num_nextn_predict_layers instead of num_hidden_layers
            num_draft_layers = getattr(draft_hf_config, "num_nextn_predict_layers", 1)
            total_num_layers += num_draft_layers
            logger.info(
                f"Allocating KV cache for {hf_config.num_hidden_layers} target layers + "
                f"{num_draft_layers} draft (MTP) layers = {total_num_layers} total layers"
            )

        if self.use_mla:
            self.kv_cache = torch.zeros(
                total_num_layers,
                self.num_physical_kvcache_blocks,
                self.physical_block_size,
                576,
                dtype=dtypes.d_dtypes[config.kv_cache_dtype],
                device="cuda",
            )
            if self.is_deepseek_v32:
                # Align last dimension to 16 bytes for fp8 (1 byte per element)
                # to avoid unaligned memory access in torch inductor
                index_dim = hf_config.index_head_dim + 4
                aligned_index_dim = ((index_dim + 15) // 16) * 16
                self.index_cache = torch.zeros(
                    hf_config.num_hidden_layers,
                    self.num_physical_kvcache_blocks,
                    self.physical_block_size,
                    aligned_index_dim,
                    dtype=dtypes.fp8,
                    device="cuda",
                )
        elif self.is_qwen_next():

            self.kv_cache = torch.zeros(
                2,
                self.num_full_attn + num_draft_layers,
                self.num_physical_kvcache_blocks,
                self.physical_block_size,
                num_kv_heads,
                hf_config.head_dim,
                dtype=dtypes.d_dtypes[config.kv_cache_dtype],
                device="cuda",
            )

            self.kv_scale = torch.zeros(
                2,
                self.num_full_attn + num_draft_layers,
                self.num_physical_kvcache_blocks,
                num_kv_heads,
                self.physical_block_size,
                dtype=dtypes.fp32,
                device="cuda",
            )

            mamba_shape = self.gated_delta_net_state_shape(
                get_tp_group().world_size,
                hf_config.linear_num_key_heads,
                hf_config.linear_num_value_heads,
                hf_config.linear_key_head_dim,
                hf_config.linear_key_head_dim,
                hf_config.linear_conv_kernel_dim,
                self.num_speculative_tokens,  # self.num_spec,
            )
            self.mamba_k_cache = torch.zeros(
                (self.num_gdn_attn_state, self.num_physical_kvcache_blocks)
                + mamba_shape[0],
                dtype=dtypes.d_dtypes[config.kv_cache_dtype],
                device="cuda",
            )
            self.mamba_v_cache = torch.zeros(
                (self.num_gdn_attn_state, self.num_physical_kvcache_blocks)
                + mamba_shape[1],
                dtype=dtypes.d_dtypes[config.kv_cache_dtype],
                device="cuda",
            )
        else:
            self.kv_cache = torch.zeros(
                2,
                hf_config.num_hidden_layers,
                self.num_physical_kvcache_blocks,
                self.physical_block_size,
                num_kv_heads,
                hf_config.head_dim,
                dtype=dtypes.d_dtypes[config.kv_cache_dtype],
                device="cuda",
            )

            self.kv_scale = torch.zeros(
                2,
                hf_config.num_hidden_layers,
                self.num_physical_kvcache_blocks,
                num_kv_heads,
                self.physical_block_size,
                dtype=dtypes.fp32,
                device="cuda",
            )
        # Build KVCacheConfig
        # lirong TODO: This is a simple solution to build KVCacheConfig,
        # models with only one type of attention, but not support multi-type of attention models.
        # We need to support it by kv_cache_group in the future.

        # Prepare list of models to bind KV cache
        models_to_bind = [("target", self.model)]
        if self.config.speculative_config and hasattr(self, "drafter"):
            models_to_bind.append(("draft", self.drafter.model))

        kv_cache_tensors = []
        layer_id = 0
        x = 16 // self.kv_cache.element_size()
        for model_name, model in models_to_bind:
            logger.info(
                f"Binding KV cache for {model_name} model starting at layer_id={layer_id}"
            )

            for module in model.modules():
                # Since use attention base and there are child in attention, add base condition
                if hasattr(module, "base_attention"):
                    if hasattr(module, "use_mla") and not module.use_mla:
                        # Non-MLA attention
                        if self.is_qwen_next():
                            attn_idx = layer_id // self.full_attention_interval
                        else:
                            attn_idx = layer_id
                        k_cache = self.kv_cache[0, attn_idx].view(
                            self.num_physical_kvcache_blocks,
                            num_kv_heads,
                            hf_config.head_dim // x,
                            self.physical_block_size,
                            x,
                        )
                        v_cache = self.kv_cache[1, attn_idx].view(
                            self.num_physical_kvcache_blocks,
                            num_kv_heads,
                            hf_config.head_dim,
                            self.physical_block_size,
                        )
                        module.max_model_len = self.config.max_model_len
                        if config.kv_cache_dtype == "fp8":
                            module.k_scale = self.kv_scale[0, attn_idx]
                            module.v_scale = self.kv_scale[1, attn_idx]

                        k_scale = module.k_scale
                        v_scale = module.v_scale

                        # Store in KVCacheTensor
                        kv_cache_tensor = KVCacheTensor(
                            layer_num=layer_id,
                            k_cache=k_cache,
                            v_cache=v_cache,
                            k_scale=k_scale,
                            v_scale=v_scale,
                        )
                        kv_cache_tensors.append(kv_cache_tensor)

                        module.k_cache = k_cache
                        module.v_cache = v_cache

                        layer_id += 1
                    elif hasattr(module, "use_mla") and module.use_mla:
                        # MLA attention
                        kv_cache = self.kv_cache[layer_id].view(
                            self.num_physical_kvcache_blocks * self.physical_block_size,
                            1,
                            576,
                        )
                        module.max_model_len = self.config.max_model_len
                        if self.is_deepseek_v32 and module.indexer is not None:
                            # Use aligned dimension to avoid memory copy in torch inductor
                            module.indexer.k_cache.kv_cache[0] = self.index_cache[
                                layer_id
                            ].view(
                                self.num_physical_kvcache_blocks
                                * self.physical_block_size,
                                1,
                                aligned_index_dim,
                            )
                        # Store in KVCacheTensor
                        kv_cache_tensor = KVCacheTensor(
                            layer_num=layer_id,
                            k_cache=kv_cache,
                            v_cache=None,
                            k_scale=None,
                            v_scale=None,
                        )
                        kv_cache_tensors.append(kv_cache_tensor)

                        module.kv_cache = kv_cache
                        module.max_model_len = self.config.max_model_len
                        layer_id += 1
                elif hasattr(module, "base_linear_attention"):
                    gdn_idx = (
                        layer_id
                        // self.full_attention_interval
                        * (self.full_attention_interval - 1)
                        + layer_id % self.full_attention_interval
                    )
                    mamba_k_cache = self.mamba_k_cache[gdn_idx]
                    mamba_v_cache = self.mamba_v_cache[gdn_idx]
                    kv_cache_tensor = KVCacheTensor(
                        layer_num=layer_id,
                        k_cache=mamba_k_cache,
                        v_cache=mamba_v_cache,
                        k_scale=None,
                        v_scale=None,
                    )
                    kv_cache_tensors.append(kv_cache_tensor)
                    layer_id += 1

        # Store KVCacheConfig
        kv_cache_data = {
            f"layer_{i}": kv_cache_tensor
            for i, kv_cache_tensor in enumerate(kv_cache_tensors)
        }
        # vllm use register_kv_caches to register kv_cache_data. We just set it to global here
        set_kv_cache_data(kv_cache_data)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return True

    def gated_delta_net_state_shape(
        self,
        tp_world_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        num_spec: int = 0,
    ):
        conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
        conv_state_shape = (
            conv_dim // tp_world_size,
            conv_kernel_size - 1 + num_spec,
        )

        conv_state_shape = conv_state_shape[1], conv_state_shape[0]

        temporal_state_shape = (
            num_v_heads // tp_world_size,
            head_k_dim,
            head_v_dim,
        )
        return conv_state_shape, temporal_state_shape

    def get_dp_padding(self, num_tokens: int) -> tuple[int, Optional[torch.Tensor]]:
        dp_size = self.config.parallel_config.data_parallel_size
        dp_rank = self.config.parallel_config.data_parallel_rank

        # For DP: Don't pad when setting enforce_eager.
        # This lets us set enforce_eager on the prefiller in a P/D setup and
        # still use CUDA graphs (enabled by this padding) on the decoder.
        #
        # TODO(tms) : There are many cases where padding is enabled for
        # prefills, causing unnecessary and excessive padding of activations.

        if dp_size == 1:
            # Early exit.
            return 0, None
        num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
            num_tokens, dp_size, dp_rank
        )
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp).item()

        return max_tokens_across_dp_cpu - num_tokens, num_tokens_across_dp

    def _preprocess(self, batch: ScheduledBatch):
        num_input_tokens = batch.total_tokens_num
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad
        return num_input_tokens, num_tokens_across_dp

    def prepare_inputs(self, batch: ScheduledBatch, input_ids: torch.Tensor = None):
        is_prefill = batch.total_tokens_num_prefill > 0
        bs = batch.total_seqs_num
        num_scheduled_tokens = np.asarray(batch.num_scheduled_tokens)
        cu_seqlens_q, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
        num_input_tokens, num_tokens_across_dp = self._preprocess(batch)
        self.forward_vars["cu_seqlens_q"].np[1 : bs + 1] = cu_seqlens_q
        if not is_prefill:
            scheduled_bs = batch.total_seqs_num_decode
            # num_pad, num_tokens_across_dp = self.get_dp_padding(scheduled_bs)
            # padded_scheduled_bs = scheduled_bs + num_pad
            # TODO rename num_input_tokens to actual bs in currrent rank?
            padded_scheduled_bs = num_input_tokens
            # for MTP, we need to divide by (mtp_k + 1) to get the actual batch size
            if hasattr(self, "drafter"):
                padded_scheduled_bs = padded_scheduled_bs // (self.drafter.mtp_k + 1)
            bs = (
                padded_scheduled_bs
                if self.enforce_eager
                else next(
                    (x for x in self.graph_bs if x >= padded_scheduled_bs),
                    padded_scheduled_bs,
                )
                # Use cudagraph and padding to batch_size, if bs > graph_bs, use eager mode
            )
            assert (
                bs >= padded_scheduled_bs
            ), f"current decode {padded_scheduled_bs=} > max graph_bs{bs}"
            self.forward_vars["cu_seqlens_q"].np[scheduled_bs + 1 : bs + 1] = (
                self.forward_vars["cu_seqlens_q"].np[scheduled_bs]
            )
        attn_metadata, positions = self.attn_metadata_builder.build(batch, bs)
        context_bs = batch.total_seqs_num_prefill if is_prefill else scheduled_bs

        # graph_bs should be batch size (number of sequences), not token count
        graph_bs = num_input_tokens if is_prefill else bs
        context = Context(
            positions=positions,
            is_prefill=is_prefill,
            batch_size=context_bs,
            graph_bs=graph_bs,
        )
        actual_num_tokens = batch.total_tokens_num

        spec_decode_metadata = None
        if not is_prefill and hasattr(self, "drafter"):
            scheduled_bs = batch.total_seqs_num_decode
            spec_decode_metadata = self._calc_spec_decode_metadata(
                self.drafter.mtp_k,
                num_scheduled_tokens[:scheduled_bs],
                cu_seqlens_q[:scheduled_bs],
                input_ids,
            )

        set_forward_context(
            attn_metadata=attn_metadata,
            atom_config=self.config,
            context=context,
            num_tokens=actual_num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            spec_decode_metadata=spec_decode_metadata,
        )
        return graph_bs

    def prepare_sample(self, batch: ScheduledBatch) -> torch.Tensor:
        bs = batch.total_seqs_num
        buffer = self.forward_vars["temperatures"]
        buffer.np[:bs] = batch.temperatures
        return buffer.copy_to_gpu(bs)

    def prepare_model(self, batch: ScheduledBatch):
        total_tokens_num = batch.total_tokens_num
        assert total_tokens_num > 0

        temperatures = self.prepare_sample(batch)
        input_ids = self.tokenID_processor.prepare_input_ids(batch)
        self.prepare_inputs(batch, input_ids)
        return (
            input_ids,
            temperatures,
        )

    def run_model(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        forward_context = get_forward_context()
        context = forward_context.context
        bs = context.batch_size
        is_prefill = context.is_prefill
        positions = context.positions
        if is_prefill or self.enforce_eager or bs > self.graph_bs[-1]:
            hidden_states = self.model(input_ids, positions)
        else:
            graph_bs = context.graph_bs
            max_q_len = forward_context.attn_metadata.max_seqlen_q
            graph_key = (graph_bs, max_q_len)
            self.graphs[graph_key].replay()
            num_tokens = context.batch_size * max_q_len
            hidden_states = self.forward_vars["outputs"][:num_tokens]
        logits = self.model.compute_logits(hidden_states)

        return logits, hidden_states

    def postprocess(
        self,
        batch: ScheduledBatch,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        # following for draft
        hidden_states: torch.Tensor,
    ) -> ScheduledBatchOutput:
        spec_decode_metadata = get_forward_context().spec_decode_metadata
        num_bonus_tokens: Optional[torch.Tensor] = None
        if spec_decode_metadata is None:
            sampled_tokens = self.sampler(logits, temperatures)
        else:
            assert logits is not None
            bonus_logits_indices = spec_decode_metadata.bonus_logits_indices
            target_logits_indices = spec_decode_metadata.target_logits_indices

            bonus_logits = torch.index_select(logits, 0, bonus_logits_indices)
            target_logits = torch.index_select(logits, 0, target_logits_indices)
            bonus_token_ids = self.sampler(
                logits=bonus_logits,
                temperatures=temperatures,
            )
            # Validate shapes match expectations
            if target_logits.shape[0] != len(spec_decode_metadata.draft_token_ids):
                raise ValueError(
                    f"Shape mismatch: target_logits.shape[0]={target_logits.shape[0]} "
                    f"but len(draft_token_ids)={len(spec_decode_metadata.draft_token_ids)}. "
                    f"target_logits_indices shape={spec_decode_metadata.target_logits_indices.shape}, "
                    f"logits.shape[0]={logits.shape[0]}"
                )

            sampled_tokens, num_bonus_tokens = self.rejection_sampler.forward(
                spec_decode_metadata,
                target_logits,
                bonus_token_ids,
            )

        if get_tp_group().world_size > 1 and self.tokenID_processor.is_deferred_out:
            sampled_tokens = get_tp_group().broadcast(sampled_tokens, src=0)

        self.forward_done_event.record()
        token_ids = self.tokenID_processor.prepare_sampled_ids(
            batch, sampled_tokens, self.forward_done_event
        )
        draft_token_ids: Optional[torch.Tensor] = None
        if self.tokenID_processor.is_deferred_out and hasattr(self, "drafter"):
            if spec_decode_metadata is None:
                next_token_ids = sampled_tokens
            else:
                next_token_ids = torch.gather(
                    sampled_tokens, 1, num_bonus_tokens.view(-1, 1)
                ).view(-1)
                # self.debug(f"{sampled_tokens=}")
                # self.debug(f"{num_bonus_tokens=}")
                # self.debug(f"{next_token_ids=}")
                self.tokenID_processor.prev_token_ids = next_token_ids
                self.tokenID_processor.num_bonus_tokens = num_bonus_tokens
                self.tokenID_processor.send_bonus_to_cpu_async(
                    num_bonus_tokens,
                    batch.req_ids,
                )  # Async copy to CPU

            draft_token_ids = self.propose_draft_token_ids(
                batch,
                self.tokenID_processor.input_ids.gpu[1 : batch.total_tokens_num + 1],
                hidden_states,
                next_token_ids,
            )

        return ScheduledBatchOutput(
            token_ids,
            draft_token_ids,
            is_deferred_out=self.tokenID_processor.is_deferred_out,
        )

    @torch.inference_mode()
    def forward(self, batch: ScheduledBatch) -> ScheduledBatchOutput:
        input_ids, temperatures = self.prepare_model(batch)
        logits, hidden_states = self.run_model(input_ids)
        fwd_output = self.postprocess(
            batch,
            logits,
            temperatures,
            hidden_states,
        )
        reset_forward_context()
        return fwd_output

    def _calc_spec_decode_metadata(
        self,
        num_spec_steps: int,
        num_sampled_tokens: np.ndarray,
        cu_num_sampled_tokens: np.ndarray,
        input_ids: torch.Tensor,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]

        scheduled_bs = len(num_sampled_tokens)

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        num_draft_tokens = np.full(scheduled_bs, num_spec_steps, dtype=np.int32)
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange
        # self.debug(f"{target_logits_indices=}")

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        )
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        )

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = torch.index_select(input_ids, 0, bonus_logits_indices)

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_spec_steps=num_spec_steps,
            num_draft_tokens_np=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
        )
        return metadata

    def propose_draft_token_ids(
        self,
        batch: ScheduledBatch,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ):
        # num_scheduled_tokens = batch.total_tokens_num
        forward_context = get_forward_context()

        positions = forward_context.context.positions
        spec_decode_metadata = forward_context.spec_decode_metadata
        if spec_decode_metadata is None:
            last_token_offset = 1
        else:
            num_bonus_tokens = self.tokenID_processor.num_bonus_tokens
            last_token_offset = 1 + self.drafter.mtp_k - num_bonus_tokens

        assert isinstance(self.drafter, EagleProposer)

        last_token_indices = self.drafter.prepare_inputs(
            batch.total_seqs_num, last_token_offset
        )

        draft_token = self.drafter.propose(
            target_token_ids=input_ids,
            target_positions=positions,
            target_hidden_states=hidden_states,
            next_token_ids=next_token_ids,
            last_token_indices=last_token_indices,
        )
        return self.tokenID_processor.prepare_draft_ids(batch, draft_token)

    @torch.inference_mode()
    def capture_cudagraph(self):
        start_time = time.time()
        # self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        if self.config.compilation_config.cudagraph_capture_sizes:
            self.graph_bs = self.config.compilation_config.cudagraph_capture_sizes
        else:
            cuda_graph_sizes = self.config.compilation_config.cuda_graph_sizes
            if len(cuda_graph_sizes) == 1:
                self.graph_bs = [1, 2, 4, 8] + [
                    i for i in range(16, cuda_graph_sizes[0] + 1, 16)
                ]
            elif len(cuda_graph_sizes) > 1:
                self.graph_bs = cuda_graph_sizes
        self.graph_bs.sort(reverse=True)

        assert (
            self.graph_bs[0] <= self.config.max_num_seqs
        ), "cudagraph capture sizes must be less than max_num_seqs."

        input_ids = self.forward_vars["input_ids"].gpu
        positions = self.forward_vars["positions"].gpu
        outputs = self.forward_vars["outputs"]

        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = dict()
        self.graph_pool = None

        with graph_capture() as gc:
            capture_range = (
                tqdm.tqdm(self.graph_bs) if self.rank == 0 else self.graph_bs
            )
            max_q_len = self.drafter.mtp_k + 1 if hasattr(self, "drafter") else 1
            for bs in capture_range:
                if self.rank == 0:
                    capture_range.set_description(f"Capturing {bs=}, {max_q_len=}")
                graph = torch.cuda.CUDAGraph()

                cu_seqlens_q = np.arange(
                    0, (bs + 1) * max_q_len, max_q_len, dtype=np.int32
                )
                self.forward_vars["cu_seqlens_q"].np[: bs + 1] = cu_seqlens_q
                self.forward_vars["cu_seqlens_q"].copy_to_gpu(bs + 1)

                num_tokens = bs * max_q_len
                # Use a simple, safe position pattern for capture.
                self.forward_vars["positions"].np[:num_tokens] = (
                    np.arange(num_tokens, dtype=np.int64) % max_q_len
                )

                attn_metadata, context = (
                    self.attn_metadata_builder.build_for_cudagraph_capture(bs)
                )
                num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
                num_tokens += num_pad
                set_forward_context(
                    attn_metadata=attn_metadata,
                    atom_config=self.config,
                    context=context,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                )

                outputs[:num_tokens] = self.model(
                    input_ids[:num_tokens], positions[:num_tokens]
                )  # warmup

                with torch.cuda.graph(graph, self.graph_pool, stream=gc.stream):
                    outputs[:num_tokens] = self.model(
                        input_ids[:num_tokens], positions[:num_tokens]
                    )  # capture
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[(bs, max_q_len)] = graph
                torch.cuda.synchronize()
        self.graph_bs.sort(reverse=False)
        return time.time() - start_time, self.graph_bs
