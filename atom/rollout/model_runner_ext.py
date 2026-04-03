# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
from typing import Optional

import numpy as np
import torch

from atom.model_engine.model_runner import ModelRunner, tokenIDProcessor
from atom.model_engine.scheduler import ScheduledBatch, ScheduledBatchOutput
from atom.rollout.memory_manager import MemoryManagerMixin
from atom.rollout.weight_updater import WeightUpdaterMixin
from aiter.dist.parallel_state import get_tp_group

logger = logging.getLogger("atom")


class LogprobsTokenIDProcessor(tokenIDProcessor):
    """tokenIDProcessor extended with logprobs support for RLHF."""

    def clean(self):
        super().clean()
        self.logprobs_cpu: list[Optional[torch.Tensor]] = []

    def send_to_cpu_async(
        self,
        gpu_tensor: torch.Tensor,
        cpu_tensor_handle,
        data_ready: torch.cuda.Event,
        copy_done: Optional[torch.cuda.Event] = None,
        gpu_logprobs: Optional[torch.Tensor] = None,
    ):
        copy_done = copy_done or torch.cuda.Event()
        with torch.cuda.stream(self.async_copy_stream):
            data_ready.wait(stream=self.async_copy_stream)
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            cpu_logprobs = (
                gpu_logprobs.to("cpu", non_blocking=True)
                if gpu_logprobs is not None
                else None
            )
            copy_done.record(self.async_copy_stream)
        cpu_tensor_handle.append((cpu_tensor, copy_done))
        self.logprobs_cpu.append(cpu_logprobs)

    def recv_logprobs(self) -> Optional[list[float]]:
        """Pop and return the earliest logprobs from the async copy queue."""
        if not self.logprobs_cpu:
            return None
        logprob_tensor = self.logprobs_cpu.pop(0)
        if logprob_tensor is not None:
            return logprob_tensor.tolist()
        return None

    def prepare_sampled_ids_with_logprobs(
        self,
        batch: ScheduledBatch,
        sampled_token_ids: torch.Tensor,
        sync_event: torch.cuda.Event,
        sampled_logprobs: Optional[torch.Tensor] = None,
    ) -> tuple[list[int], list[tuple[int, ...]], Optional[dict[int, float]]]:
        """Extended prepare_sampled_ids that also returns logprobs.

        Returns:
            (req_ids, token_ids, logprobs_map) where logprobs_map is
            {seq_id: logprob} or None.
        """
        if not self.is_deferred_out:
            token_ids = sampled_token_ids.tolist()
            req_ids = batch.req_ids
            if token_ids and isinstance(token_ids[0], list):
                processed = self._batch_process_token_ids(token_ids)
            else:
                processed = [(tid,) for tid in token_ids]
            logprobs_map = None
            if sampled_logprobs is not None:
                lp = sampled_logprobs.tolist()
                logprobs_map = dict(zip(req_ids, lp))
            return req_ids, processed, logprobs_map

        token_ids = self.recv_async_output(self.token_ids_cpu).tolist()
        logprobs = self.recv_logprobs()
        self.send_to_cpu_async(
            sampled_token_ids, self.token_ids_cpu, sync_event,
            gpu_logprobs=sampled_logprobs,
        )
        req_ids_out: list[int] = []
        processed_out: list[tuple[int, ...]] = []
        logprobs_map = None
        self.prev_req_ids = None
        if self.prev_batch is not None:
            self.prev_req_ids = self.prev_batch.req_ids
            req_ids_out = self.prev_req_ids
            if token_ids and isinstance(token_ids[0], list):
                processed_out = self._batch_process_token_ids(token_ids)
            else:
                processed_out = [(tid,) for tid in token_ids]
            if logprobs is not None:
                logprobs_map = dict(zip(self.prev_req_ids, logprobs))

        self.prev_batch = batch
        self.prev_token_ids = sampled_token_ids

        return req_ids_out, processed_out, logprobs_map


class RLHFModelRunner(ModelRunner, WeightUpdaterMixin, MemoryManagerMixin):
    """ModelRunner with RLHF extensions (weight sync + memory lifecycle + logprobs).

    Used when ATOM is driven by an external RLHF framework.
    Pure inference deployments use the base ModelRunner, which carries no
    RLHF-specific code.
    """

    # Environment variable whose value is a comma-separated list of physical
    # GPU indices assigned to this DP rank (e.g. "2,3").  When set, each DP
    # rank's ModelRunners form an independent NCCL world with TP only.
    # Frameworks may set this via their own env vars; the adapter layer is
    # responsible for mapping to ATOM_DP_DEVICE_MAP before constructing the
    # runner.
    DP_DEVICE_MAP_ENV = "ATOM_DP_DEVICE_MAP"

    def _init_dp_isolation(self, config, rank: int):
        """Resolve DP rank → device mapping when device-level isolation is active.

        When ``DP_DEVICE_MAP_ENV`` is set, each DP rank's ModelRunners form
        an independent NCCL world scoped to TP only.
        """
        dp_rank_local = config.parallel_config.data_parallel_rank_local or 0
        num_gpus = torch.cuda.device_count()
        device_map = os.environ.get(self.DP_DEVICE_MAP_ENV)
        dp_isolated = device_map is not None

        if dp_isolated:
            device_indices = [int(x) for x in device_map.split(",")]
            local_device_rank = device_indices[rank]
            dp_port = config.parallel_config.data_parallel_base_port + dp_rank_local * 100
            effective_dp_rank, effective_dp_size = 0, 1
        else:
            local_device_rank = dp_rank_local * config.tensor_parallel_size + rank
            dp_port = config.parallel_config.data_parallel_base_port
            effective_dp_rank = config.parallel_config.data_parallel_rank
            effective_dp_size = config.parallel_config.data_parallel_size

        if local_device_rank >= num_gpus:
            raise ValueError(
                f"local_device_rank={local_device_rank} exceeds available GPUs ({num_gpus}), "
                f"dp_isolated={dp_isolated}, dp_rank_local={dp_rank_local}, tp_rank={rank}"
            )

        return local_device_rank, dp_port, effective_dp_rank, effective_dp_size, num_gpus

    def __init__(self, rank: int, config):
        # Check if DP device-level isolation is active — if so, override
        # config before the base class __init__ sets up distributed.
        device_map = os.environ.get(self.DP_DEVICE_MAP_ENV)
        if device_map is not None:
            (
                local_device_rank,
                dp_port,
                effective_dp_rank,
                effective_dp_size,
                num_gpus,
            ) = self._init_dp_isolation(config, rank)

            # Temporarily patch config so base __init__ uses our values
            orig_dp_rank = config.parallel_config.data_parallel_rank
            orig_dp_size = config.parallel_config.data_parallel_size
            orig_dp_rank_local = config.parallel_config.data_parallel_rank_local
            orig_dp_base_port = config.parallel_config.data_parallel_base_port
            config.parallel_config.data_parallel_rank = effective_dp_rank
            config.parallel_config.data_parallel_size = effective_dp_size
            config.parallel_config.data_parallel_rank_local = (
                0 if effective_dp_size == 1 else orig_dp_rank_local
            )
            config.parallel_config.data_parallel_base_port = dp_port

            # Pre-set HIP_VISIBLE_DEVICES so aiter doesn't restrict GPU visibility
            if "HIP_VISIBLE_DEVICES" not in os.environ:
                os.environ["HIP_VISIBLE_DEVICES"] = ",".join(
                    str(i) for i in range(num_gpus)
                )

        super().__init__(rank, config)

        # Restore original config values
        if device_map is not None:
            config.parallel_config.data_parallel_rank = orig_dp_rank
            config.parallel_config.data_parallel_size = orig_dp_size
            config.parallel_config.data_parallel_rank_local = orig_dp_rank_local
            config.parallel_config.data_parallel_base_port = orig_dp_base_port

    def warmup_model(self):
        """Override to replace tokenIDProcessor before warmup runs forward()."""
        self.tokenID_processor = LogprobsTokenIDProcessor(
            self,
            self.config.max_num_batched_tokens,
            hasattr(self, "drafter"),
            self.num_spec_tokens,
        )
        super().warmup_model()

    def allocate_kv_cache(self, num_kvcache_blocks):
        """Override to use TP-only barrier instead of global barrier.

        Each DP rank manages its own KV cache independently via its Engine
        Core, so a global barrier would deadlock when DP ranks execute
        resume_memory at different times.  In isolated DP mode
        (effective_dp_size=1), the global barrier is equivalent to TP
        barrier anyway.
        """
        return super().allocate_kv_cache(num_kvcache_blocks)

    def postprocess(
        self,
        batch: ScheduledBatch,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks,
        top_ps,
        all_greedy: bool,
        hidden_states: torch.Tensor,
    ) -> ScheduledBatchOutput:
        """Override to add logprobs computation."""
        from atom.utils.forward_context import get_forward_context, reset_forward_context
        from atom.model_engine.model_runner import Sampler

        spec_decode_metadata = get_forward_context().spec_decode_metadata
        bs = batch.total_seqs_num
        if spec_decode_metadata is None:
            sampled_tokens = self.sampler(
                logits, temperatures, top_ks, top_ps, all_greedy
            )
            num_reject_tokens = self.tokenID_processor.default_num_rejected_tokens[:bs]
            next_token_locs = num_reject_tokens
        else:
            assert logits is not None
            bonus_logits_indices = spec_decode_metadata.bonus_logits_indices
            target_logits_indices = spec_decode_metadata.target_logits_indices

            bonus_logits = torch.index_select(logits, 0, bonus_logits_indices)
            target_logits = torch.index_select(logits, 0, target_logits_indices)
            bonus_token_ids = self.sampler(
                logits=bonus_logits,
                temperatures=temperatures,
                top_ks=top_ks,
                top_ps=top_ps,
                all_greedy=all_greedy,
            )

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
            num_reject_tokens = self.drafter.mtp_k - num_bonus_tokens
            next_token_locs = num_bonus_tokens

        if get_tp_group().world_size > 1 and self.tokenID_processor.is_deferred_out:
            sampled_tokens = get_tp_group().broadcast(sampled_tokens, src=0)

        # Compute logprobs if any sequence requested them
        need_logprobs = any(batch.return_logprobs)
        sampled_logprobs = None
        if need_logprobs:
            logits_fp32 = logits.float()
            safe_temps = torch.where(
                temperatures <= 0, torch.ones_like(temperatures), temperatures
            )
            scaled_logits = logits_fp32 / safe_temps.view(-1, 1)
            log_probs = torch.log_softmax(scaled_logits, dim=-1)
            sampled_logprobs = log_probs.gather(
                -1, sampled_tokens.to(torch.long).unsqueeze(-1)
            ).squeeze(-1)
            if get_tp_group().world_size > 1 and self.tokenID_processor.is_deferred_out:
                sampled_logprobs = get_tp_group().broadcast(sampled_logprobs, src=0)

        self.forward_done_event.record()
        prev_batch = self.tokenID_processor.prev_batch
        req_ids_out, token_ids_out, logprobs_map = (
            self.tokenID_processor.prepare_sampled_ids_with_logprobs(
                batch, sampled_tokens, self.forward_done_event, sampled_logprobs
            )
        )

        draft_token_ids: Optional[np.ndarray] = None
        if self.tokenID_processor.is_deferred_out:
            if hasattr(self, "drafter"):
                prev_rejected_num = self.tokenID_processor.prev_rejected_num
                prev_bonus_num = self.tokenID_processor.prev_bonus_num
                self.tokenID_processor.send_mtp_status_to_cpu_async(
                    num_reject_tokens, next_token_locs, self.forward_done_event
                )
                next_token_ids = torch.gather(
                    sampled_tokens.view(bs, -1), 1, next_token_locs.view(-1, 1)
                ).view(bs)
                self.tokenID_processor.prev_token_ids = next_token_ids
                draft_token_ids = self.propose_draft_token_ids(
                    batch,
                    self.tokenID_processor.input_ids.gpu[
                        1 : batch.total_tokens_num + 1
                    ],
                    hidden_states,
                    next_token_ids,
                    num_reject_tokens,
                )

            elif prev_batch is not None:
                prev_rejected_num = np.zeros(prev_batch.total_seqs_num, dtype=np.int32)
                prev_bonus_num = np.zeros(prev_batch.total_seqs_num, dtype=np.int32)
            else:
                prev_rejected_num = np.zeros(0, dtype=np.int32)
                prev_bonus_num = np.zeros(0, dtype=np.int32)
        else:
            prev_rejected_num = np.zeros(batch.total_seqs_num, dtype=np.int32)
            prev_bonus_num = np.zeros(batch.total_seqs_num, dtype=np.int32)

        return ScheduledBatchOutput(
            req_ids=req_ids_out,
            token_ids=token_ids_out,
            draft_token_ids=draft_token_ids,
            is_deferred_out=self.tokenID_processor.is_deferred_out,
            num_rejected=prev_rejected_num,
            num_bonus=prev_bonus_num,
            logprobs=logprobs_map,
        )
