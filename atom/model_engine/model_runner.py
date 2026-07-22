# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import bisect
import gc
import inspect
import logging
import math
import os
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.profiler as torch_profiler
import tqdm
from aiter import destroy_dist_env, init_dist_env
from aiter.dist.parallel_state import (
    get_dp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
)
from aiter.dist.utils import get_distributed_init_method
from atom.config import Config, CUDAGraphMode, set_current_atom_config
from atom.kv_transfer.disaggregation import KVConnectorOutput
from atom.model_engine.run_labels import build_run_label
from atom.model_engine.scheduler import ScheduledBatch, ScheduledBatchOutput
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType
from atom.model_loader.loader import load_model
from atom.model_ops.eplb import (
    initialize_eplb_runtime,
    with_eplb_forward_monitor,
)
from atom.model_ops.rejection_sampler import RejectionSampler
from atom.model_ops.sampler import SAMPLER_EPS, Sampler
from atom.spec_decode.eagle import EagleProposer
from atom.utils import (
    CpuGpuBuffer,
    envs,
    get_hf_text_config,
    init_exit_handler,
    resolve_obj_by_qualname,
)
from atom.utils.cuda_graph import BatchDescriptor
from atom.utils.forward_context import (
    Context,
    DPMetadata,
    ForwardMode,
    get_forward_context,
    get_kvconnector,
    reset_forward_context,
    set_forward_context,
    set_kv_cache_data,
)
from atom.utils.selector import get_attn_backend
from atom.utils.tbo import (
    UBatchSlice,
    UBatchWrapper,
    local_tbo_precompute,
    maybe_create_ubatch_slices,
    sync_dp_metadata,
)
from atom.distributed.pcp_utils import (
    PcpBalGroup,
    pcp_allgather_rerange,
    pcp_pad_len,
    pcp_round_robin_split,
)
from torch.profiler import record_function

logger = logging.getLogger("atom")

support_model_arch_dict = {
    "Qwen3ForCausalLM": "atom.models.qwen3.Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM": "atom.models.qwen3_moe.Qwen3MoeForCausalLM",
    "LlamaForCausalLM": "atom.models.llama.LlamaForCausalLM",
    "MixtralForCausalLM": "atom.models.mixtral.MixtralForCausalLM",
    "DeepseekV3ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "DeepseekV32ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "DeepseekV4ForCausalLM": "atom.models.deepseek_v4.DeepseekV4ForCausalLM",
    "GptOssForCausalLM": "atom.models.gpt_oss.GptOssForCausalLM",
    "GlmMoeDsaForCausalLM": "atom.models.deepseek_v2.GlmMoeDsaForCausalLM",
    "Glm4MoeForCausalLM": "atom.models.glm4_moe.Glm4MoeForCausalLM",
    "Qwen3NextForCausalLM": "atom.models.qwen3_next.Qwen3NextForCausalLM",
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MultimodalModel",
    "Qwen3_5MoeForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MoeMultimodalModel",
    "KimiK25ForConditionalGeneration": "atom.models.kimi_k25.KimiK25ForCausalLM",
    "MiniMaxM2ForCausalLM": "atom.models.minimax_m2.MiniMaxM2ForCausalLM",
    "MiMoV2ForCausalLM": "atom.models.mimo_v2.MiMoV2ForCausalLM",
    "MiMoV2FlashForCausalLM": "atom.models.mimo_v2.MiMoV2ForCausalLM",
    "Mistral3ForConditionalGeneration": "atom.models.mistral3.Mistral3TextOnly",
    "MistralForCausalLM": "atom.models.mistral3.Mistral3ForCausalLM",
    "MiniMaxM3SparseForCausalLM": "atom.models.minimax_m3.MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration": "atom.models.minimax_m3.MiniMaxM3SparseForConditionalGeneration",
}
# seed = 34567
# np.random.seed(seed)
# torch.cuda.manual_seed_all(seed)


class tokenIDProcessor:

    def __init__(
        self,
        runner: "ModelRunner",
        max_num_batched_tokens: int,
        use_spec: bool = False,
        num_spec_tokens: int = 0,
    ):
        """Asynchronously copy the sampled_token_ids tensor to the host."""
        self.is_deferred_out = True

        self.runner = runner
        device = runner.device
        self.input_ids = CpuGpuBuffer(
            max_num_batched_tokens + 1, dtype=torch.int32, device=device
        )
        self.input_ids_loc = CpuGpuBuffer(
            max_num_batched_tokens, dtype=torch.int64, device=device
        )
        self.use_spec = use_spec
        self.num_spec_tokens = num_spec_tokens

        self.async_copy_stream = torch.cuda.Stream(runner.device)
        self.default_num_rejected_tokens = torch.zeros(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )
        self.clean()

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

    def recv_async_output(self, cpu_tensor_handle) -> torch.Tensor:
        if not cpu_tensor_handle:
            return torch.empty(0, dtype=torch.int32, device="cpu")
        cpu_tensor, event = cpu_tensor_handle.pop(0)
        event.synchronize()
        return cpu_tensor

    def recv_logprobs(self) -> Optional[list[float]]:
        """Pop and return the earliest logprobs from the async copy queue.
        Must be called after recv_async_output (which synchronizes the event).
        """
        if not self.logprobs_cpu:
            return None
        logprob_tensor = self.logprobs_cpu.pop(0)
        if logprob_tensor is not None:
            return logprob_tensor.tolist()
        return None

    def send_to_cpu_async_draft(self, gpu_tensor: torch.Tensor):
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.async_copy_stream):
            self.async_copy_stream.wait_stream(default_stream)
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.async_copy_stream)
        self.draft_token_ids_cpu.append((cpu_tensor, event))

    def recv_async_output_draft(self) -> np.ndarray:
        if not self.draft_token_ids_cpu:
            return np.array([], dtype=np.int32)
        token_ids, event = self.draft_token_ids_cpu.pop(0)
        event.synchronize()
        return token_ids.numpy()

    def send_mtp_status_to_cpu_async(
        self,
        num_rejected: torch.Tensor,
        num_bonus: torch.Tensor,
        data_ready: torch.cuda.Event,
    ):
        # rejected num and bonus num are slightly different info for mtp
        # take mtp=1 for example:
        #   first decode after prefill have 0 rej, 0 bonus
        #   prev acc decode have 0 rej, 1 bonus
        #   prev rej decode have 1 rej, 0 bonus
        # It is clear that only rejected number is not sufficient for all status tracking, bonus number is also needed.
        # Single Event for both copies (vs. per-tensor send_to_cpu_async) so the
        # consumer pops one queue entry and synchronizes once instead of twice.
        copy_done = torch.cuda.Event()
        with torch.cuda.stream(self.async_copy_stream):
            data_ready.wait(stream=self.async_copy_stream)
            cpu_num_rejected = num_rejected.to("cpu", non_blocking=True)
            cpu_num_bonus = num_bonus.to("cpu", non_blocking=True)
            copy_done.record(self.async_copy_stream)
        self.pending_mtp_status_copies.append(
            (cpu_num_rejected, cpu_num_bonus, copy_done)
        )

    def recv_mtp_status_async(
        self,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.pending_mtp_status_copies:
            return None, None
        cpu_num_rejected, cpu_num_bonus, copy_done = self.pending_mtp_status_copies.pop(
            0
        )
        copy_done.synchronize()
        return cpu_num_rejected.numpy(), cpu_num_bonus.numpy()

    def clean(self):
        self.token_ids_cpu: list[torch.Tensor] = []
        self.logprobs_cpu: list[Optional[torch.Tensor]] = []

        self.prev_batch: Optional[ScheduledBatch] = None
        self.prev_token_ids: Optional[torch.Tensor] = None

        self.pre_num_decode_token_per_seq = 1
        self.draft_token_ids: Optional[torch.Tensor] = None
        self.draft_token_ids_cpu: list[torch.Tensor] = []
        # Queue of (cpu_num_rejected, cpu_num_bonus, copy_done_event) — async
        # D2H copies fired by send_mtp_status_to_cpu_async, drained by
        # recv_mtp_status_async after the event syncs.
        self.pending_mtp_status_copies: list[
            tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]
        ] = []
        self.mapped_bonus_list: Optional[list[int]] = (
            None  # Mapped to current batch order
        )
        self.num_rejected: Optional[np.ndarray] = None
        self.num_bonus: Optional[np.ndarray] = None

    @staticmethod
    def _batch_process_token_ids(token_ids: list) -> list[tuple[int, ...]]:
        """Batch process token_ids: vectorized -1 truncation using numpy."""
        arr = np.array(token_ids, dtype=np.int64)
        mask = arr == -1
        if not mask.any():
            # No -1 sentinel in any row, convert each row to tuple directly
            return [tuple(row) for row in arr.tolist()]
        # Per-row: find first -1, truncate
        # Use argmax on mask; rows without -1 get 0, disambiguate with ~mask.any(axis=1)
        has_sentinel = mask.any(axis=1)
        first_neg = mask.argmax(axis=1)
        result = []
        rows = arr.tolist()
        for i, row in enumerate(rows):
            if has_sentinel[i]:
                result.append(tuple(row[: first_neg[i]]))
            else:
                result.append(tuple(row))
        return result

    def prepare_sampled_ids(
        self,
        batch: ScheduledBatch,
        sampled_token_ids: torch.Tensor,
        sync_event: torch.cuda.Event,
        sampled_logprobs: Optional[torch.Tensor] = None,
    ) -> tuple[dict[int, tuple[int, ...]], Optional[dict[int, float]]]:
        if not self.is_deferred_out:
            token_ids = sampled_token_ids.tolist()
            req_ids = batch.req_ids
            if token_ids and isinstance(token_ids[0], list):
                processed = self._batch_process_token_ids(token_ids)
            else:
                processed = [(tid,) for tid in token_ids]
            ret = dict(zip(req_ids, processed))
            ret[-1] = 0  # is_deferred_out flag
            logprobs_map = None
            if sampled_logprobs is not None:
                logprobs = sampled_logprobs.tolist()
                logprobs_map = {
                    seq_id: logprob for seq_id, logprob in zip(req_ids, logprobs)
                }
            return ret, logprobs_map

        token_ids = self.recv_async_output(self.token_ids_cpu)
        logprobs = self.recv_logprobs()
        self.send_to_cpu_async(
            sampled_token_ids,
            self.token_ids_cpu,
            sync_event,
            gpu_logprobs=sampled_logprobs,
        )
        token_id_dict = {}
        logprobs_map = None
        self.prev_req_ids = None
        if self.prev_batch is not None:
            self.prev_req_ids = self.prev_batch.req_ids
            token_ids_list = (
                token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
            )
            if token_ids_list and isinstance(token_ids_list[0], list):
                processed = self._batch_process_token_ids(token_ids_list)
            else:
                processed = [(tid,) for tid in token_ids_list]
            token_id_dict = dict(zip(self.prev_req_ids, processed))
            if logprobs is not None:
                logprobs_map = {
                    seq_id: logprob
                    for seq_id, logprob in zip(self.prev_req_ids, logprobs)
                }
        else:
            # first time, no previous tokens
            token_ids = {}
            logprobs_map = None

        self.prev_batch = batch
        self.prev_token_ids = sampled_token_ids
        token_id_dict[-1] = 1
        return token_id_dict, logprobs_map

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
        GPU need to be copied into the corresponding slots into input_ids.
        """
        scheduled_tokens = batch.scheduled_tokens  # tokens per req
        total_tokens = batch.total_tokens_num
        total_tokens_prefill = batch.total_tokens_num_prefill
        total_tokens_decode = batch.total_tokens_num_decode
        total_reqs_prefill = batch.total_seqs_num_prefill
        """for prefill: all input ids are new"""
        self.input_ids.np[:total_tokens_prefill] = scheduled_tokens[
            :total_tokens_prefill
        ]
        self.input_ids.copy_to_gpu(total_tokens_prefill)

        self.prev_rejected_num, self.prev_bonus_num = self.recv_mtp_status_async()

        # TODO: remove this when we support mixed prefill and decode in one batch
        if total_reqs_prefill > 0:
            return self.input_ids.gpu[:total_tokens_prefill]

        if not self.is_deferred_out:
            token_ids = scheduled_tokens[
                total_tokens_prefill : total_tokens_prefill + total_tokens_decode
            ]
            if self.use_spec:
                if (
                    getattr(batch, "dynamic_spec_query_tokens_per_req", None)
                    is not None
                ):
                    # RAGGED: scheduled_tokens is already the flat [anchor, drafts...]
                    # so no rectangular reshape/overwrite is needed.
                    pass
                else:
                    token_ids[:, 1:] = batch.scheduled_spec_decode_tokens

            self.input_ids.np[:total_tokens_decode] = token_ids
            return self.input_ids.copy_to_gpu(total_tokens_decode)

        # PD consumer first decode: no prior prefill step initialized
        # prev_batch, so use scheduled_tokens directly for this step.
        if self.prev_batch is None:
            token_ids = scheduled_tokens[
                total_tokens_prefill : total_tokens_prefill + total_tokens_decode
            ]
            self.input_ids.np[:total_tokens_decode] = token_ids
            return self.input_ids.copy_to_gpu(total_tokens_decode)

        """for decode: input ids are from prev_sampled_token_ids"""
        deferred_curr_indices, deferred_prev_indices, new_curr_indices, is_all_same = (
            self.get_token_locations(batch)
        )
        num_deferred_seqs = len(deferred_curr_indices)
        num_new_seqs = len(new_curr_indices)

        # Calculate token counts: in MTP mode, each seq has multiple tokens.
        # num_spec_query_tokens is the single source of truth (= mtp_k+1 for
        # plain MTP, or the DSpark q-bucket when shrunk this step). See
        # ScheduledBatch.num_spec_query_tokens.
        _per_req = getattr(batch, "dynamic_spec_query_tokens_per_req", None)
        if self.use_spec and _per_req is not None and is_all_same:
            _pr = np.asarray(_per_req)
            tokens_per_seq = int(batch.num_spec_query_tokens)
            num_deferred_tokens = int(_pr[deferred_curr_indices].sum())
            num_new_tokens = (
                int(_pr[new_curr_indices].sum()) if len(new_curr_indices) else 0
            )
        elif self.use_spec:
            tokens_per_seq = batch.num_spec_query_tokens
            num_deferred_tokens = num_deferred_seqs * tokens_per_seq
            num_new_tokens = num_new_seqs * tokens_per_seq
        else:
            tokens_per_seq = 1
            num_deferred_tokens = num_deferred_seqs
            num_new_tokens = num_new_seqs

        # Receive and map bonus_list to current batch order
        self.num_rejected = batch.num_rejected
        self.num_bonus = batch.num_bonus
        if num_deferred_seqs > 0 and self.prev_rejected_num is not None:
            # Map: prev_bonus_list[prev_idx] → mapped_bonus_list[curr_idx]
            self.num_rejected[deferred_curr_indices] = self.prev_rejected_num[
                deferred_prev_indices
            ]
            self.num_bonus[deferred_curr_indices] = self.prev_bonus_num[
                deferred_prev_indices
            ]

        # DSpark dynamic: per-req lengths differ, build input_ids by scattering each
        # seq's [anchor, drafts...] into its cu-offset segment.
        ragged_lens = getattr(batch, "dynamic_spec_query_tokens_per_req", None)
        if ragged_lens is not None and is_all_same and self.use_spec:
            self._ragged_fill_deferred_all_same(batch, ragged_lens, num_deferred_tokens)
            input_ids = self.input_ids.gpu[:total_tokens]
            return input_ids

        if is_all_same:
            # All requests are the same, only deferred tokens
            if self.use_spec:
                # MTP mode: combine prev_token_ids and draft_token_ids
                if (
                    self.draft_token_ids is not None
                    and self.pre_num_decode_token_per_seq > 1
                ):
                    # DSpark: self.draft_token_ids carries full mtp_k
                    # columns from the previous step, but this step's q-bucket
                    # wants only tokens_per_seq (= q) per seq.
                    draft_cols = self.draft_token_ids
                    n_draft = tokens_per_seq - 1
                    if n_draft < draft_cols.shape[1]:
                        draft_cols = draft_cols[:, :n_draft]
                    combined = torch.cat(
                        [
                            self.prev_token_ids.unsqueeze(1),  # (num_seqs, 1)
                            draft_cols,  # (num_seqs, q-1)
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
                        n_draft = tokens_per_seq - 1
                        if n_draft < gathered_draft.shape[1]:
                            gathered_draft = gathered_draft[:, :n_draft]
                        gathered_tokens = torch.cat(
                            [
                                gathered_prev.unsqueeze(1),  # (num_deferred_seqs, 1)
                                gathered_draft,  # (num_deferred_seqs, q-1)
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
                    token_ids = scheduled_tokens[
                        total_tokens_prefill : total_tokens_prefill + num_new_tokens
                    ].reshape(num_new_seqs, tokens_per_seq)
                    if self.use_spec:
                        token_ids[:, 1:] = batch.scheduled_spec_decode_tokens[
                            :num_new_seqs
                        ]
                    self.input_ids.np[:num_new_tokens] = token_ids.flatten()
                    self.input_ids.copy_to_gpu(num_new_tokens)
            else:
                # Layout: [deferred | new] - deferred at front, new is from previous finished prefill and waiting for decode
                if num_new_tokens > 0:
                    # Convert seq-level indices to token-level indices
                    new_token_indices = (
                        new_curr_indices[:, None] * tokens_per_seq
                        + np.arange(tokens_per_seq)
                    ).flatten()
                    new_token_ids = scheduled_tokens[new_token_indices].reshape(
                        num_new_seqs, tokens_per_seq
                    )
                    if self.use_spec:
                        # MTP mode: combine scheduled_tokens and draft_tokens
                        draft_tokens = batch.scheduled_spec_decode_tokens[
                            new_curr_indices
                        ]
                        new_token_ids[:, 1:] = draft_tokens
                    self.input_ids.np[:num_new_tokens] = new_token_ids.flatten()
                    self.input_ids.gpu[
                        num_deferred_tokens : num_deferred_tokens + num_new_tokens
                    ].copy_(self.input_ids.cpu[:num_new_tokens], non_blocking=True)
                if gathered_tokens is not None:
                    self.input_ids.gpu[:num_deferred_tokens] = gathered_tokens
        input_ids = self.input_ids.gpu[:total_tokens]
        return input_ids

    def _ragged_fill_deferred_all_same(self, batch, ragged_lens, num_deferred_tokens):
        """Fill input_ids for the all-same deferred decode step under RAGGED.

        Layout per seq i (length ragged_lens[i] = ell_i+1):
          [ anchor_i (= prev_token_ids[i]),  draft_i[0 .. ell_i-1] ]
        anchor from self.prev_token_ids [bs]; drafts from self.draft_token_ids
        [bs, mtp_k] (full columns, sliced to ell_i-1). Scatter into the flat
        input_ids buffer at per-seq cu offsets. Done on CPU then one H2D — the
        token counts are tiny (Σ ell_i+1 ≤ bs*(mtp_k+1)).
        """
        lens = np.asarray(ragged_lens, dtype=np.int64)
        bs = lens.shape[0]
        cu = np.zeros(bs + 1, dtype=np.int64)
        np.cumsum(lens, out=cu[1:])
        total = int(cu[-1])
        assert total <= num_deferred_tokens, (
            f"ragged total {total} > num_deferred_tokens {num_deferred_tokens} "
            f"(graph bucket capacity); ragged must fit within bs*q_eff"
        )
        anchors = self.prev_token_ids.detach().to("cpu").numpy()  # [bs]
        drafts = (
            self.draft_token_ids.detach().to("cpu").numpy()
            if self.draft_token_ids is not None
            else None
        )  # [bs, mtp_k]
        flat = self.input_ids.np
        for i in range(bs):
            s = int(cu[i])
            flat[s] = anchors[i]
            d = int(lens[i]) - 1
            if d > 0 and drafts is not None:
                flat[s + 1 : s + 1 + d] = drafts[i, :d]
        # FLAT graph tail-padding. Under CUDAGraph the captured grid processes
        # C = effective_bs * q_eff tokens (effective_bs = the graph bs bucket
        # >= bs), but this ragged step has only Σ = total real tokens (Σ ≤ C).
        # The graph reads the static input_ids buffer out to C, so [Σ:C] must
        # hold a LEGAL vocab id (0) — stale ids would OOB the embedding gather.
        # Compute C the same way ForwardMode will (smallest graph_bs >= bs) ×
        # q_eff. Eager (no graph) → fill_to == total (no-op beyond the Σ fill).
        q_eff = int(getattr(batch, "num_spec_query_tokens", 1))
        fill_to = num_deferred_tokens
        if not self.runner.enforce_eager:
            # smallest captured graph_bs >= bs (graph_bs is sorted descending)
            gbs = next((g for g in reversed(self.runner.graph_bs) if g >= bs), None)
            if gbs is not None:
                fill_to = max(fill_to, int(gbs) * q_eff)
        if fill_to > total:
            flat[total:fill_to] = 0
        self.input_ids.copy_to_gpu(fill_to)

    def prepare_draft_ids(
        self, batch: ScheduledBatch, draft_token_ids: torch.Tensor
    ) -> np.ndarray:
        if not self.is_deferred_out:
            ret = draft_token_ids.numpy()
        else:
            self.draft_token_ids = draft_token_ids
            self.pre_num_decode_token_per_seq = self.num_spec_tokens + 1
            token_ids = self.recv_async_output_draft()
            self.send_to_cpu_async_draft(draft_token_ids)
            ret = (
                token_ids
                if self.prev_req_ids is not None
                else np.array([], dtype=np.int32)
            )
        return ret


class ModelRunner:

    def __init__(self, rank: int, config: Config):
        self.config = config
        self.mark_trace = getattr(config, "mark_trace", False)
        from atom.utils.graph_marker import set_graph_marker_enabled

        set_graph_marker_enabled(self.mark_trace)
        set_current_atom_config(config)
        hf_config = config.hf_config
        self.block_size = config.kv_cache_block_size
        self.kv_cache_dtype = config.kv_cache_dtype
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
        self.use_v4 = self.is_deepseek_v4()
        rope_parameters = getattr(self.hf_text_config, "rope_parameters", None) or {}
        self.use_mrope = "mrope_section" in rope_parameters
        self.is_deepseek_v32 = (
            hasattr(hf_config, "index_topk") if self.use_mla else False
        )
        # Initialize profiler for this rank (before _setup_device_and_distributed
        # so that dp config fields are still at their original values)
        self.profiler = None
        self.profiler_dir = None
        dp_rank_local = config.parallel_config.data_parallel_rank_local or 0
        if dp_rank_local > 0 or config.parallel_config.data_parallel_size > 1:
            self.rank_name = f"dp{dp_rank_local}_tp{rank}"
        else:
            self.rank_name = f"rank_{rank}"
        if config.torch_profiler_dir is not None:
            self.profiler_dir = os.path.join(config.torch_profiler_dir, self.rank_name)
            os.makedirs(self.profiler_dir, exist_ok=True)

        self._setup_device_and_distributed(rank, config)

        self.graph_bs = [0]  # for eager fallback
        # PIECEWISE cudagraph state, populated by capture_cudagraph. Empty when
        # capture never ran (enforce_eager), so the ragged-bucket paths no-op.
        self._piecewise_captured_tokens: set[int] = set()
        self._piecewise_sorted_tokens: list[int] = []

        init_exit_handler(self)
        default_dtype = self.config.torch_dtype
        torch.set_default_dtype(default_dtype)
        torch.set_default_device(self.device)
        self.attn_backend = get_attn_backend(
            self.block_size,
            use_mla=self.use_mla,
            use_gdn=self.use_gdn,
            use_v4=self.use_v4,
        )
        use_spec = bool(self.config.speculative_config) and get_pp_group().is_last_rank
        self.num_spec_tokens = (
            self.config.speculative_config.num_speculative_tokens if use_spec else 0
        )
        self.eagle3_mode = (
            self.config.speculative_config is not None
            and self.config.speculative_config.method == "eagle3"
        )

        self.use_aux_hidden_state_outputs = False
        self.use_dspark_aux_capture = False
        self._aux_hidden_states = None
        self.tokenID_processor = tokenIDProcessor(
            self,
            self.config.max_num_batched_tokens,
            use_spec,
            self.num_spec_tokens,
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
        # The model construction depends on quant_config,
        # so we must complete the remapping for layers before constructing the model.
        config.quant_config.remap_layer_name(
            config.hf_config,
            packed_modules_mapping=getattr(model_class, "packed_modules_mapping", {}),
            quant_exclude_name_mapping=getattr(
                model_class, "quant_exclude_name_mapping", {}
            ),
        )
        self.model = model_class(config)
        fused_shared_expert_load_fn = None
        if hasattr(self.model, "load_fused_expert_weights"):
            fused_shared_expert_load_fn = self.model.load_fused_expert_weights
        torch.set_default_device(None)
        load_start = time.perf_counter()
        load_model(
            self.model,
            config.model,
            config.hf_config,
            config.load_dummy,
            load_fused_expert_weights_fn=fused_shared_expert_load_fn,
        )
        load_elapsed = time.perf_counter() - load_start
        logger.info(
            f"[{self.rank_name}] Model load done: {config.model} "
            f"(weights loaded in {load_elapsed:.2f}s)"
        )

        # Optional debug instrumentation; no-op when env vars unset.
        # See atom/utils/debug_helper/.
        from atom.utils.debug_helper import (
            install_block_forward_hooks,
            maybe_dump_weights_and_exit,
        )

        _n_fwd_hooks = install_block_forward_hooks(self.model)
        if _n_fwd_hooks > 0:
            logger.info(f"[ATOM_FWD_DUMP] {_n_fwd_hooks} Block forward hooks installed")
        maybe_dump_weights_and_exit(self.model)

        if self.config.speculative_config and get_pp_group().is_last_rank:
            from atom.utils.backends import set_model_tag

            torch.set_default_device(self.device)
            with set_model_tag("drafter"):
                self.drafter = EagleProposer(self.config, self.device, self)
            self.rejection_sampler = RejectionSampler()
            torch.set_default_device(None)
            logger.info("Loading drafter model...")
            self.drafter.load_model(self.model)

        if self.eagle3_mode and self.config.speculative_config.use_aux_hidden_state:
            aux_ids = self.config.speculative_config.eagle3_aux_layer_ids
            if not aux_ids and hasattr(
                self.model, "get_eagle3_aux_hidden_state_layers"
            ):
                aux_ids = list(self.model.get_eagle3_aux_hidden_state_layers())
            if aux_ids:
                self.model.set_aux_hidden_state_layers(tuple(aux_ids))
                self.use_aux_hidden_state_outputs = True
                logger.info(f"Eagle3 aux hidden state layers: {aux_ids}")

        # DSpark draft consumes target hidden states from configured target
        # layers (dspark_target_layer_ids, e.g. [58,59,60]). The reference
        # captures the per-layer mHC residual reduced over the hc axis
        # (mean over dim=1: [N, hc, dim] -> [N, dim]) and concatenates the
        # selected layers. V4ForCausalLM is @support_torch_compile (must not be
        # edited), so we capture via forward hooks on the target decoder layers.
        if (
            self.config.speculative_config
            and get_pp_group().is_last_rank
            and getattr(self.config.speculative_config, "use_dspark", lambda: False)()
        ):
            self._install_dspark_aux_hooks()

        torch.set_default_device(self.device)
        self.async_execute_stream = torch.cuda.Stream(self.device)
        self.allocate_forward_vars()
        self.attn_metadata_builder = self.attn_backend.get_builder_cls()(
            model_runner=self
        )
        self.physical_block_size = self.attn_metadata_builder.block_size
        # Sanity-check: any builder that allocates a per-request cache must
        # have its model_type listed in `InputOutputProcessor`'s
        # `per_req_cache_model_types` set; otherwise sequences will be
        # constructed with `has_per_req_cache=False`, the BlockManager will
        # never assign them a slot, and the builder will silently read
        # tensor[-1] on first decode. Catch the misconfiguration up front
        # rather than producing wrong outputs at inference time.
        if self.attn_metadata_builder.compute_per_req_cache_bytes() > 0:
            from atom.model_engine.llm_engine import InputOutputProcessor as _IOProc

            mt = self.config.hf_config.model_type
            known = _IOProc._per_req_cache_model_types()  # noqa: SLF001
            assert mt in known, (
                f"Attention builder {type(self.attn_metadata_builder).__name__} "
                f"reports per_req_cache_bytes>0 but model_type={mt!r} is not in "
                f"InputOutputProcessor.per_req_cache_model_types ({sorted(known)}). "
                "Add it to the set or sequences will not be assigned slots "
                "(silent corruption)."
            )
        if config.enable_tbo:
            dp_gather_scatter = (
                config.enable_dp_attention and not config.enable_expert_parallel
            )
            self.model = UBatchWrapper(
                self.model,
                attn_metadata_builder=self.attn_metadata_builder,
                dp_gather_scatter=dp_gather_scatter,
            )
            logger.info("TBO enabled: model wrapped with UBatchWrapper")
        self.forward_done_event = torch.cuda.Event()
        initialize_eplb_runtime(self)
        self.warmup_model()
        logger.info(f"Model warmup done: {config.model}")

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.config.compilation_config.level == 1:
            self.model = torch.compile(self.model, fullgraph=True, backend="eager")
            if hasattr(self, "drafter"):
                self.drafter.model = torch.compile(
                    self.drafter.model, fullgraph=True, backend="eager"
                )

    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_mtp",
            "glm_moe_dsa",
            "kimi_k2",
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
        elif self.hf_text_config.model_type in (
            "qwen3_next",
            "qwen3_next_mtp",
            "qwen3_5_text",
            "qwen3_5_moe_text",
        ):
            return True
        return False

    def is_deepseek_v4(self) -> bool:
        # NOTE: `hf_text_config.model_type` reads "deepseek_v3" for V4 because
        # `_CONFIG_REGISTRY` maps deepseek_v4 → deepseek_v3 (V4 reuses V3 schema).
        # Use `architectures` (preserved by get_hf_config:567) instead. Covers
        # both target (DeepseekV4ForCausalLM[NextN]) and draft (whose model_type
        # SpeculativeConfig stamps as deepseek_v4_mtp).
        arches = getattr(self.hf_text_config, "architectures", None) or []
        if any("DeepseekV4" in str(a) for a in arches):
            return True
        return getattr(self.hf_text_config, "model_type", None) in (
            "deepseek_v4",
            "deepseek_v4_mtp",
        )

    def is_mimo_v2(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            "mimo_v2",
            "mimo_v2_flash",
        ):
            return True
        return False

    def _setup_device_and_distributed(self, rank: int, config: Config):
        # Calculate local device rank considering both TP and DP
        # When data parallelism is enabled on the same node, different DP ranks
        # need to use different sets of GPUs
        dp_rank_local = config.parallel_config.data_parallel_rank_local or 0
        local_device_rank = (
            dp_rank_local
            * config.tensor_parallel_size
            * config.prefill_context_parallel_size
            + rank
        )
        num_gpus = torch.cuda.device_count()
        if local_device_rank >= num_gpus:
            raise ValueError(
                f"Calculated local_device_rank={local_device_rank} exceeds available GPUs ({num_gpus}). "
            )

        self.device = torch.device(f"cuda:{local_device_rank}")
        logger.info(
            f"ModelRunner rank={rank}, dp_rank_local={dp_rank_local}, "
            f"local_device_rank={local_device_rank}, device={self.device}"
        )

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
            prefill_context_model_parallel_size=config.prefill_context_parallel_size,
        )

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
        # 1. Destroy distributed env (NCCL + CustomAllreduce + process groups)
        #    Must happen while ops module is still alive for CustomAllreduce cleanup.
        destroy_dist_env()
        # 2. Release CUDA graphs
        if not self.enforce_eager:
            self.graphs = self.graph_pool = None  # type: ignore
        if isinstance(self.model, UBatchWrapper):
            self.model.tbo_graphs.clear()
        # 3. Release GPU tensors
        for attr in (
            "kv_cache",
            "kv_scale",
            "index_cache",
            "mamba_k_cache",
            "mamba_v_cache",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "drafter"):
            del self.drafter
        torch.cuda.empty_cache()
        return True

    def start_profiler(self, trace_name: Optional[str] = None):
        """
        Start profiling for this rank.

        The ATOM_PROFILER_MORE environment variable controls detailed profiling features:
        - Set to "1" to enable record_shapes, with_stack, and profile_memory.
        - Set to "0" or unset to disable these features (default).
        """
        if self.profiler_dir is not None and self.profiler is None:
            enable_detailed_profiling = envs.ATOM_PROFILER_MORE
            model_name = os.path.basename(self.config.model.rstrip("/"))
            safe_model_name = "".join(
                c if c.isalnum() or c in ("_", "-", ".") else "_" for c in model_name
            )
            worker_name = safe_model_name or "trace"
            if isinstance(trace_name, str) and trace_name:
                worker_name = "".join(
                    c if c.isalnum() or c in ("_", "-", ".") else "_"
                    for c in trace_name
                )
            if worker_name == "capture_graph":
                if safe_model_name:
                    worker_name = f"{worker_name}_{safe_model_name}"
            output_prefix = os.path.join(self.profiler_dir, worker_name)

            def _on_trace_ready(prof):
                import gzip as _gzip

                # Use a short human-readable timestamp in file name.
                ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                ms = int((time.time() % 1) * 1000)
                output_path = f"{output_prefix}_ts_{ts}_{ms:03d}.pt.trace.json.gz"
                tmp_json_path = output_path[:-3]
                try:
                    t0 = time.monotonic()
                    prof.export_chrome_trace(tmp_json_path)
                    # Chunked gzip: read 64 MB at a time to avoid loading
                    # the entire JSON (~30 GB) into memory at once.
                    with (
                        open(tmp_json_path, "rb") as src,
                        _gzip.open(output_path, "wb") as dst,
                    ):
                        while chunk := src.read(64 * 1024 * 1024):
                            dst.write(chunk)
                    os.remove(tmp_json_path)
                    sz = os.path.getsize(output_path)
                    logger.info(
                        "Rank %d: trace exported to %s (%.1f MB, %.1fs)",
                        self.rank,
                        output_path,
                        sz / 1e6,
                        time.monotonic() - t0,
                    )
                except Exception:
                    logger.exception(
                        "Rank %d: failed to export trace to %s",
                        self.rank,
                        output_path,
                    )
                    for p in (tmp_json_path, output_path):
                        if os.path.exists(p):
                            os.remove(p)

            self.profiler = torch_profiler.profile(
                activities=[
                    torch_profiler.ProfilerActivity.CPU,
                    torch_profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=enable_detailed_profiling,
                with_stack=enable_detailed_profiling,
                profile_memory=enable_detailed_profiling,
                on_trace_ready=_on_trace_ready,
            )
            self.profiler.__enter__()
            logger.info(
                "Rank %d: profiler started (detailed=%s, dir=%s)",
                self.rank,
                enable_detailed_profiling,
                self.profiler_dir,
            )
        return True

    def stop_profiler(self):
        """Stop profiling for this rank.

        Returns a dict with ``trace_dir`` and ``elapsed`` so the caller
        can report where the trace was written.
        """
        if self.profiler is None:
            return {"trace_dir": self.profiler_dir, "elapsed": 0.0}
        t0 = time.monotonic()
        logger.info("Rank %d: stopping profiler...", self.rank)
        try:
            self.profiler.__exit__(None, None, None)
        except Exception:
            logger.exception("Rank %d: profiler stop failed", self.rank)
        finally:
            self.profiler = None
        elapsed = round(time.monotonic() - t0, 1)
        logger.info(
            "Rank %d: profiler stop completed in %.1fs",
            self.rank,
            elapsed,
        )
        return {"trace_dir": self.profiler_dir, "elapsed": elapsed}

    def debug(self, *args: Any):
        if self.rank == 0:
            logger.info(*args)

    def _install_dspark_aux_hooks(self) -> None:
        """Capture DSpark target hidden states via forward hooks.

        DSpark's draft reads, for each configured target layer L, the layer's
        output mHC residual reduced over the hc axis (mean(dim=1):
        [N, hc, dim] -> [N, dim]), and concatenates the selected layers into
        [N, len(target_layers)*dim].

        The captured per-layer [N, dim] tensors are stashed on the runner and
        read out in run_model as ``self._aux_hidden_states`` (list, target order).
        """
        spec_cfg = self.config.speculative_config
        draft_cfg = spec_cfg.draft_model_hf_config
        target_layer_ids = tuple(
            int(i) for i in getattr(draft_cfg, "dspark_target_layer_ids", ())
        )
        if not target_layer_ids:
            raise ValueError(
                "DSpark requires dspark_target_layer_ids on the draft config."
            )

        base = getattr(self.model, "language_model", self.model)
        inner = base.model  # DeepseekV4Model
        layers = inner.layers
        hidden_size = self.config.hf_config.hidden_size
        max_tokens = self.config.max_num_batched_tokens

        self._dspark_target_layer_ids = target_layer_ids
        self._dspark_aux_buffers = [
            torch.zeros(
                max_tokens,
                hidden_size,
                device=self.device,
                dtype=self.config.torch_dtype,
            )
            for _ in target_layer_ids
        ]
        # Map layer id -> buffer index (closed over by each hook; read-only).
        layer_to_buf = {lid: i for i, lid in enumerate(target_layer_ids)}

        def _make_hook(buf_idx: int, block):
            buffer = self._dspark_aux_buffers[buf_idx]

            def _hook(_module, _inputs, output):
                # output is the HCState returned by Block.forward.
                residual = getattr(output, "residual", None)
                x_prev = getattr(output, "x_prev", None)
                post = getattr(output, "post_mix", None)
                comb = getattr(output, "comb_mix", None)
                if residual is None:
                    return
                if x_prev is not None and post is not None and comb is not None:
                    # Synthesize the post-layer residual [N, hc, dim].
                    out_res = block.hc_post(x_prev, residual, post, comb)
                else:
                    out_res = residual
                # Reduce over the hc axis to [N, dim] and write in-place into the
                # fixed buffer (cudagraph-safe; no host-side dict mutation).
                reduced = out_res.mean(dim=1)
                buffer[: reduced.shape[0]].copy_(reduced)

            return _hook

        n_layers = len(layers)
        for lid in target_layer_ids:
            if lid < 0 or lid >= n_layers:
                raise ValueError(
                    f"dspark_target_layer_id {lid} out of range [0,{n_layers})."
                )
            layers[lid].register_forward_hook(
                _make_hook(layer_to_buf[lid], layers[lid])
            )

        self.use_dspark_aux_capture = True
        logger.info(f"DSpark aux capture hooks on target layers: {target_layer_ids}")

    def _collect_dspark_aux(self, num_tokens: int) -> None:
        """Assemble captured per-layer aux tensors (sliced to num_tokens)."""
        if not getattr(self, "use_dspark_aux_capture", False):
            return
        self._aux_hidden_states = [buf[:num_tokens] for buf in self._dspark_aux_buffers]

    def _run_dummy_drafter(self, hidden_states, draft_bs=None):
        """Run drafter forward for DP synchronization (no real proposal)."""
        if not hasattr(self, "drafter"):
            return
        forward_context = get_forward_context()
        forward_context.context.is_draft = True
        if draft_bs is None:
            draft_bs = forward_context.context.graph_bs
        for i in range(self.drafter.mtp_k):
            self.drafter._refresh_dp_metadata(forward_context, hidden_states.shape[0])
            hidden_states = self.drafter.model(
                input_ids=torch.zeros(
                    hidden_states.shape[0],
                    dtype=torch.int32,
                    device=self.device,
                ),
                positions=torch.zeros(
                    hidden_states.shape[0],
                    dtype=torch.int64,
                    device=self.device,
                ),
                hidden_states=hidden_states,
            )
            if i == 0:
                hidden_states = hidden_states[:draft_bs]
                # pad_for_all_gather uses graph_bs * 1, consistent with
                # ranks running propose
                forward_context.attn_metadata.max_seqlen_q = 1

    def dummy_execution(self):
        """Execute dummy decode batch for DP synchronization."""
        # num_tokens_original = 1
        has_drafter = hasattr(self, "drafter")
        mtp_k = self.drafter.mtp_k if has_drafter else 0
        mtp_factor = mtp_k + 1
        num_tokens_original = mtp_factor

        seq = Sequence([0] * num_tokens_original, block_size=self.block_size, id=-1)
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.DECODE
        seq.block_table = [0]

        spec_tokens = {seq.id: np.zeros(mtp_k, dtype=np.int32)} if mtp_k > 0 else None
        dummy_batch = ScheduledBatch(
            seqs={seq.id: seq},
            num_scheduled_tokens=np.array([num_tokens_original], dtype=np.int32),
            total_tokens_num=num_tokens_original,
            total_tokens_num_decode=num_tokens_original,
            total_seqs_num=1,
            total_seqs_num_decode=1,
            is_dummy_run=True,
            num_spec_step=mtp_k,
            scheduled_spec_decode_tokens=spec_tokens,
        )

        self.forward(dummy_batch)
        logger.debug(
            f"{self.label}: dummy batch executed with {dummy_batch.total_tokens_num} tokens"
        )
        return True

    def dummy_prefill_execution(self, num_tokens: int, num_reqs: int = 1):
        """Execute dummy prefill batch for DP synchronization."""
        if num_reqs < 1:
            num_reqs = 1
        if num_tokens < num_reqs:
            num_tokens = num_reqs
        # Distribute tokens evenly across requests
        base = num_tokens // num_reqs
        remainder = num_tokens % num_reqs
        tokens_per_seq = [base + (1 if i < remainder else 0) for i in range(num_reqs)]

        seqs = {}
        for t in tokens_per_seq:
            seq = Sequence([0] * t, block_size=self.block_size)
            seqs[seq.id] = seq

        dummy_batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=np.array(tokens_per_seq, dtype=np.int32),
            total_tokens_num=num_tokens,
            total_tokens_num_prefill=num_tokens,
            total_seqs_num=num_reqs,
            total_seqs_num_prefill=num_reqs,
            is_dummy_run=True,
        )

        bs = self.prepare_inputs(dummy_batch)
        self.forward_vars["input_ids"].gpu[:bs].zero_()
        input_ids = self.forward_vars["input_ids"].gpu[:bs]

        with torch.no_grad():
            logits, hidden_states = self.run_model(input_ids)
            self._run_dummy_drafter(hidden_states, draft_bs=1)

        torch.cuda.synchronize()
        reset_forward_context()

        logger.info(
            f"{self.label}: dummy PREFILL batch executed with {num_tokens} tokens, {num_reqs} reqs"
        )
        # TODO: initialize KV connector during warmup
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
        if self.config.enable_dp_attention:
            warmup_max_tokens = max_num_batched_tokens
        else:
            warmup_max_tokens = max_num_batched_tokens // dp_size

        pcp_size = self.config.prefill_context_parallel_size
        if pcp_size > 1:
            warmup_max_tokens = max(1, warmup_max_tokens // pcp_size)

        num_seqs = min(warmup_max_tokens // max_model_len, self.config.max_num_seqs)

        if num_seqs == 0:
            num_seqs = 1
            seq_len = min(warmup_max_tokens, max_model_len)
            if seq_len == 0:
                seq_len = 1
            logger.warning(
                f"{self.label}: dp_size={dp_size}, dp_attn={self.config.enable_dp_attention}, "
                f"warmup_max_tokens={warmup_max_tokens} < max_model_len={max_model_len}. "
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
            "top_ks": CpuGpuBuffer(self.max_bs, **i32_kwargs),
            "top_ps": CpuGpuBuffer(self.max_bs, **f32_kwargs),
            # Keep enough space for MTP decode (max_q_len > 1).
            # `extra_output_dims` lets a model insert dims between N and dim
            # (e.g. DeepSeek-V4 returns the un-reduced mHC residual
            # [N, hc_mult, dim] from forward, with hc_head + LM head deferred
            # to compute_logits). Default `()` keeps the standard 2D layout.
            "outputs": torch.empty(
                self.max_num_batched_tokens,
                *getattr(self.model, "extra_output_dims", ()),
                hidden_size,
                dtype=hidden_type,
            ),
        }
        if self.use_mrope:
            self.forward_vars["mrope_positions"] = CpuGpuBuffer(
                3, self.max_num_batched_tokens, **i64_kwargs
            )
        if hasattr(self, "drafter"):
            self.forward_vars["mtp_k"] = self.drafter.mtp_k
            self.forward_vars["num_accepted_tokens"] = CpuGpuBuffer(
                self.max_bs, **i32_kwargs
            )

    def _get_num_kv_heads(self):
        """Return the per-rank number of KV heads."""
        hf_config = self.config.hf_config
        if hf_config.num_key_value_heads >= self.world_size:
            assert hf_config.num_key_value_heads % self.world_size == 0
            return hf_config.num_key_value_heads // self.world_size
        else:
            assert self.world_size % hf_config.num_key_value_heads == 0
            return 1

    def _mrope_positions_view(self, num_tokens: int) -> torch.Tensor:
        return self.forward_vars["mrope_positions"].gpu.as_strided(
            (3, num_tokens), (num_tokens, 1)
        )

    def _get_total_num_layers(self):
        """Return total layer count including draft (MTP) layers.

        Drafts that own an independent KV cache via their own builder
        (e.g. Eagle3 MHA draft on an MLA target) account for their layers
        through that builder, so they are NOT added here. Only MTP-style
        drafts that share the target's KV pool contribute.
        """
        total = self.config.hf_config.num_hidden_layers
        if self.config.speculative_config and hasattr(self, "drafter"):
            if not hasattr(self, "eagle3_draft_builder"):
                draft_hf = self.config.speculative_config.draft_model_hf_config
                total += getattr(draft_hf, "num_nextn_predict_layers", 1)
        return total

    def _compute_block_bytes(self):
        """Per-block bytes for the unified KV pool budget.

        Sum across all attention builders attached to this runner: the
        target builder always, plus an optional `eagle3_draft_builder`
        when a heterogeneous spec-decode draft owns its own KV pool. Each
        builder knows its own tensor layout (MLA 576-dim packed, GDN-hybrid
        full-attn-only, MiMo-V2 per-layer-type, standard MHA split-K/V,
        Eagle3 independent MHA). Per-request cache bytes are accounted
        for separately via `compute_per_req_cache_bytes()`.
        """
        block_bytes = self.attn_metadata_builder.compute_block_bytes()
        if hasattr(self, "eagle3_draft_builder"):
            block_bytes += self.eagle3_draft_builder.compute_block_bytes()
        return block_bytes

    def _estimate_cudagraph_overhead(self):
        """Estimate GPU memory consumed by CUDA graph capture.

        CUDA graphs allocate a shared memory pool for intermediate activations.
        The pool size is roughly the peak activation memory during a single
        forward pass. We estimate this from the gap between warmup peak and
        current (steady-state) allocation.

        Returns 0 when enforce_eager is set (no CUDA graphs).
        """
        if self.config.enforce_eager:
            return 0
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        activation_bytes = max(peak - current, 0)

        # PIECEWISE pool ~ per_token * Σ(captured num_tokens). per_token from model
        # geometry (hidden*dtype*layers*k), not a magic constant. Under-reserve is
        # safe: capture re-checks live free mem per bucket and skips oversized.
        if self._piecewise_cg_active():
            cap_sizes = self.config.compilation_config.cudagraph_capture_sizes or [
                self.config.max_num_seqs
            ]
            # Captured num_tokens shapes are bs * q. The capture loop uses
            # full_q_len = mtp_k+1 for ANY spec-decode drafter (plain MTP or
            # DSpark), and q=1 for non-spec. Mirror that here or the estimate
            # under-counts Σtok by a factor of q (plain MTP q=4 -> 4x under ->
            # pool est 8.5GB vs actual 33GB -> OOM). DSpark additionally captures
            # multiple q-buckets, so fold its whole bucket set in.
            if hasattr(self, "drafter"):
                full_q = self.drafter.mtp_k + 1
                q_buckets = self._dspark_capture_q_buckets(full_q)
                if (
                    getattr(self.drafter, "dspark_confidence_schedule", False)
                    and os.environ.get("ATOM_PIECEWISE_FINE_TOKENS", "0") == "1"
                ):
                    q_buckets = sorted(set(q_buckets) | set(range(1, full_q + 1)))
            else:
                q_buckets = [1]
            per_token_bytes = self._piecewise_per_token_bytes()
            dp_size = self.config.parallel_config.data_parallel_size
            # Cap the reserved buckets at a fraction of the KV budget so a huge
            # capture list can't starve KV. Use the utilization budget (not raw
            # total) as the reference — it tracks the configured memory envelope.
            budget = self.config.gpu_memory_utilization * torch.cuda.mem_get_info()[1]
            target_reserve = 0.15 * budget
            all_shapes = sorted({bs * q for bs in cap_sizes for q in q_buckets})
            # Mirror the capture-loop DP+spec num_tokens cap (see capture_cudagraph)
            # so the reservation only counts buckets we actually capture.
            if dp_size > 1 and hasattr(self, "drafter"):
                _dp_cap = int(os.environ.get("ATOM_PIECEWISE_DP_MAX_TOKENS", "512"))
                all_shapes = [s for s in all_shapes if s <= _dp_cap]
            captured = []
            acc = 0
            for num_tokens in all_shapes:
                if captured and per_token_bytes * (acc + num_tokens) > target_reserve:
                    break
                captured.append(num_tokens)
                acc += num_tokens
            overhead = int(per_token_bytes * acc)
            logger.info(
                "PIECEWISE cudagraph mem estimate: n_shapes=%d/%d Σtok=%d "
                "per_token=%.3fMB -> overhead=%.2fGB",
                len(captured),
                len(all_shapes),
                acc,
                per_token_bytes / (1 << 20),
                overhead / (1 << 30),
            )
            return overhead
        overhead = activation_bytes * 0.2
        # DSpark RAGGED captures one graph set PER q-bucket, so scale by the
        # number of captured buckets (the pool grows ~linearly with bucket
        # count, each bucket ~one graph set). This stays a safe upper bound:
        # measured per-bucket pool (~1.4GB) << 0.2*act.
        if hasattr(self, "drafter") and getattr(
            self.drafter, "dspark_confidence_schedule", False
        ):
            # Match the capture loop's bucket source so we count the graphs
            # actually captured; the pool grows ~linearly with bucket count.
            buckets = self._dspark_capture_q_buckets(self.drafter.mtp_k + 1)
            n_buckets = len(buckets)
            overhead = activation_bytes * 0.2 * n_buckets
            logger.info(
                "DSpark cudagraph mem estimate: buckets=%s n=%d act=%.2fGB "
                "-> overhead=%.2fGB",
                buckets,
                n_buckets,
                activation_bytes / (1 << 30),
                overhead / (1 << 30),
            )
        return int(overhead)

    def get_num_blocks(self) -> dict[str, int]:
        torch.set_default_device(self.device)
        config = self.config
        hf_config = config.hf_config
        if not hasattr(hf_config, "head_dim") or hf_config.head_dim is None:
            hf_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads

        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # weights + peak activation tensors (PyTorch allocator high-water).
        peak_torch = max(peak, current)
        # RCCL/NCCL buffers etc. held outside the allocator: device-used minus
        # torch-reserved. Ignoring it over-allocates KV and OOMs at runtime.
        non_torch = max((total - free) - torch.cuda.memory_reserved(), 0)

        cudagraph_overhead = self._estimate_cudagraph_overhead()
        safety_margin = int(total * 0.02)

        budget = int(total * config.gpu_memory_utilization)
        non_kv_overhead = peak_torch + non_torch + cudagraph_overhead + safety_margin
        available_for_kv_budget = budget - non_kv_overhead

        # Physical clamp: never exceed what's actually free on the GPU.
        # This prevents OOM when other processes share the GPU.
        available_for_kv = min(available_for_kv_budget, free)

        torch.set_default_device("cpu")

        block_bytes = self._compute_block_bytes()

        # Per-request cache (e.g. GDN recurrent state, future DeepseekV4 ring
        # buffer + compressor state): deduct its tensor memory from the KV
        # pool budget. The actual layout / shape is owned by the attention
        # builder; ModelRunner only does sizing math.
        per_req_cache_bytes = self.attn_metadata_builder.compute_per_req_cache_bytes()
        slots_per_req = self.attn_metadata_builder.slots_per_req()
        max_per_req_cache_slots = (
            config.max_num_seqs * slots_per_req if per_req_cache_bytes > 0 else 0
        )
        per_req_cache_tensor_bytes = max_per_req_cache_slots * per_req_cache_bytes
        available_for_pool = available_for_kv - per_req_cache_tensor_bytes
        if available_for_pool <= 0:
            # Minimum gpu_memory_utilization that makes the budget just cover the
            # per-request cache tensor (available_for_kv_budget ==
            # per_req_cache_tensor_bytes). Rounded UP to the next 0.01 so the
            # printed value is actually sufficient, not the exact threshold.
            min_util = (non_kv_overhead + per_req_cache_tensor_bytes) / total
            min_util_hint = math.ceil(min_util * 100) / 100
            base_msg = (
                f"Per-request cache tensor "
                f"({per_req_cache_tensor_bytes / (1 << 30):.2f}GB for "
                f"{max_per_req_cache_slots} slots) exceeds available KV budget "
                f"({available_for_kv / (1 << 30):.2f}GB) at "
                f"--gpu-memory-utilization {config.gpu_memory_utilization:.2f}."
            )
            if available_for_kv_budget > free:
                # The physical free-memory clamp is the binding limit, not the
                # utilization budget — raising --gpu-memory-utilization won't help.
                fix_msg = (
                    f" Only {free / (1 << 30):.2f}GB is physically free on the GPU "
                    f"(other processes may be holding memory); raising "
                    f"--gpu-memory-utilization will NOT help. Free GPU memory or "
                    f"reduce --max-num-seqs (currently {config.max_num_seqs})."
                )
            elif min_util_hint <= 1.0:
                fix_msg = (
                    f" Set --gpu-memory-utilization >= {min_util_hint:.2f} "
                    f"(this only zeroes out the deficit; use a higher value for "
                    f"actual KV capacity) or reduce --max-num-seqs "
                    f"(currently {config.max_num_seqs})."
                )
            else:
                fix_msg = (
                    f" Even --gpu-memory-utilization 1.0 is insufficient "
                    f"(would need {min_util:.2f}); reduce --max-num-seqs "
                    f"(currently {config.max_num_seqs}) or free GPU memory."
                )
            raise RuntimeError(base_msg + fix_msg)
        per_req_cache_equiv_blocks = (
            math.ceil(per_req_cache_bytes / block_bytes)
            if per_req_cache_bytes > 0
            else 0
        )

        # Store for BlockManager and allocate_kv_cache.
        # Note the distinction:
        #   - per_req_cache_equiv_blocks: block-equivalents charged to the
        #     unified pool per request (memory accounting)
        #   - num_per_req_cache_groups: BlockManager free-list size; one
        #     group == one request occupies `slots_per_req` contiguous
        #     tensor slots
        #   - max_per_req_cache_slots (runner-only): TENSOR slot dimension
        #     == groups × slots_per_req (groups != slots in general)
        config.per_req_cache_equiv_blocks = per_req_cache_equiv_blocks
        config.num_per_req_cache_groups = (
            config.max_num_seqs if per_req_cache_bytes > 0 else 0
        )
        self.max_per_req_cache_slots = max_per_req_cache_slots

        # paged-SWA: some attention backends carve a SEPARATE windowed/prefix-
        # cached SWA pool out of the KV budget. The SWA bytes that
        # `compute_block_bytes` charges per compressed block move into a
        # `num_swa_blocks`-sized pool (window-freed, so far smaller than the
        # compressed pool), and the freed budget grows `num_kvcache_blocks`.
        # Whether this applies is a builder capability — `swa_pool_block_bytes()`
        # returns >0 only for backends with a separate SWA pool — so the runner
        # stays model-agnostic (no architecture check here). Under
        # PD/disaggregation the SWA pool is transferred per-request by
        # seq.swa_block_table (only the live window, i.e. the last ~128-token
        # block); see get_kv_transfer_tensors.
        b = self.attn_metadata_builder
        swa_block_bytes = b.swa_pool_block_bytes()
        if swa_block_bytes > 0:
            num_swa_blocks = b.swa_pool_num_blocks(
                config.max_num_seqs, config.max_model_len
            )
            swa_reserved = num_swa_blocks * swa_block_bytes
            # block_bytes (from _compute_block_bytes) currently includes the SWA
            # term; strip it so the compressed pool is sized on compressed bytes.
            compressed_block_bytes = block_bytes - swa_block_bytes
            num_kvcache_blocks = max(
                0, (available_for_pool - swa_reserved) // compressed_block_bytes
            )
            config.num_swa_blocks = int(num_swa_blocks)
            config.swa_window_size = int(
                getattr(hf_config, "sliding_window", 128) or 128
            )
            self.num_swa_blocks = int(num_swa_blocks)
            logger.info(
                f"paged-SWA pool: num_swa_blocks={num_swa_blocks}, "
                f"swa_block_bytes={swa_block_bytes}, "
                f"swa_reserved={swa_reserved / (1 << 30):.2f}GB, "
                f"compressed_block_bytes={compressed_block_bytes}, "
                f"num_kvcache_blocks={num_kvcache_blocks}"
            )
        else:
            config.num_swa_blocks = 0
            config.swa_window_size = 0
            self.num_swa_blocks = 0
            num_kvcache_blocks = available_for_pool // block_bytes

        logger.info(
            f"Memory budget: total_gpu={total / (1 << 30):.2f}GB, "
            f"free={free / (1 << 30):.2f}GB, "
            f"utilization={config.gpu_memory_utilization}, "
            f"budget={budget / (1 << 30):.2f}GB, "
            f"peak_torch={peak_torch / (1 << 30):.2f}GB, "
            f"non_torch={non_torch / (1 << 30):.2f}GB, "
            f"cudagraph_est={cudagraph_overhead / (1 << 30):.2f}GB, "
            f"safety={safety_margin / (1 << 30):.2f}GB, "
            f"available_for_kv={available_for_kv / (1 << 30):.2f}GB, "
            f"block_bytes={block_bytes}, "
            f"num_kvcache_blocks={num_kvcache_blocks}"
        )
        if per_req_cache_bytes > 0:
            logger.info(
                f"Per-req cache pool: bytes_per_slot="
                f"{per_req_cache_bytes / (1 << 20):.2f}MB, "
                f"max_slots={max_per_req_cache_slots}, "
                f"tensor_total={per_req_cache_tensor_bytes / (1 << 30):.2f}GB, "
                f"equiv_blocks_per_req={per_req_cache_equiv_blocks}, "
                f"pool_blocks={num_kvcache_blocks}"
            )

        # Concurrent-capacity table: at each context-length percentage of
        # max_model_len, how many requests can simultaneously hold their
        # KV in the pool. Per-req block usage = ceil(ctx_len/block_size);
        # per-req state cache is in its own pre-allocated tensor (already
        # excluded from `num_kvcache_blocks` at sizing time), so it adds
        # no per-block cost. Concurrency is also capped by
        # max_per_req_cache_slots (state buffer slot count).
        max_model_len = config.max_model_len
        cap = (
            max_per_req_cache_slots if per_req_cache_bytes > 0 else config.max_num_seqs
        )
        pct_lines = []
        for pct in (10, 30, 50, 70, 90, 100):
            ctx = max(1, max_model_len * pct // 100)
            blocks_per_req = math.ceil(ctx / self.block_size)
            block_bound = (
                num_kvcache_blocks // blocks_per_req if blocks_per_req > 0 else 0
            )
            max_conc = min(cap, block_bound) if cap > 0 else block_bound
            bound_label = (
                "slots" if cap > 0 and max_conc == cap < block_bound else "blocks"
            )
            pct_lines.append(
                f"  {pct:>3}% ({ctx:>7} tok): {blocks_per_req:>6} blk/req "
                f"→ max_concurrent={max_conc:<5} (bound by {bound_label})"
            )
        logger.info(
            f"Concurrent capacity vs context length "
            f"(max_model_len={max_model_len}, block_size={self.block_size}, "
            f"max_slots={cap}, pool_blocks={num_kvcache_blocks}):\n"
            + "\n".join(pct_lines)
        )

        assert num_kvcache_blocks > 0, (
            f"Not enough memory for KV cache with block size({self.block_size}). "
            f"At least 1 block ({block_bytes / (1 << 20):.2f}MB) is required, "
            f"but available_for_kv={available_for_kv / (1 << 20):.2f}MB "
            f"(budget={budget / (1 << 30):.2f}GB, "
            f"peak_torch={peak_torch / (1 << 30):.2f}GB, "
            f"non_torch={non_torch / (1 << 30):.2f}GB, "
            f"cudagraph_est={cudagraph_overhead / (1 << 30):.2f}GB, "
            f"safety={safety_margin / (1 << 30):.2f}GB, "
            f"free={free / (1 << 30):.2f}GB)"
        )
        return {
            "num_kvcache_blocks": num_kvcache_blocks,
            "per_req_cache_equiv_blocks": per_req_cache_equiv_blocks,
            "num_per_req_cache_groups": (
                config.max_num_seqs if per_req_cache_bytes > 0 else 0
            ),
            # paged-SWA: get_num_blocks runs in the RUNNER subprocess, so its
            # config.num_swa_blocks isn't visible to the engine process that
            # builds BlockManager. Propagate via block_info (mirrors the
            # per_req_cache fields) so BlockManager.swa_enabled matches the
            # attn builder's SWA pool.
            "num_swa_blocks": int(getattr(config, "num_swa_blocks", 0)),
            "swa_window_size": int(getattr(config, "swa_window_size", 0)),
        }

    def allocate_kv_cache(self, num_kvcache_blocks):
        pre_alloc = torch.cuda.memory_stats()["allocated_bytes.all.current"]

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
        # Promote to self so attention builders' build_kv_cache_tensor()
        # hooks can access it without re-deriving from hf_config.
        self.num_kv_heads = num_kv_heads
        self.aligned_index_dim = None  # set below for DeepSeek-V3.2

        # Calculate total number of layers (target + draft)
        total_num_layers = hf_config.num_hidden_layers
        num_draft_layers = 0
        if self.config.speculative_config and hasattr(self, "drafter"):
            draft_hf_config = self.config.speculative_config.draft_model_hf_config
            if hasattr(self, "eagle3_draft_builder"):
                # Heterogeneous draft (e.g. Eagle3 MHA on MLA target) owns
                # its own KV pool via its builder; don't add to target's count.
                num_draft_layers = draft_hf_config.num_hidden_layers
                logger.info(
                    f"Allocating KV cache for {hf_config.num_hidden_layers} target layers + "
                    f"{num_draft_layers} Eagle3 draft layers (separate non-MLA cache)"
                )
            else:
                # For MTP, use num_nextn_predict_layers instead of num_hidden_layers
                num_draft_layers = getattr(
                    draft_hf_config, "num_nextn_predict_layers", 1
                )
                total_num_layers += num_draft_layers
                logger.info(
                    f"Allocating KV cache for {hf_config.num_hidden_layers} target layers + "
                    f"{num_draft_layers} draft (MTP) layers = {total_num_layers} total layers"
                )

        # Primary KV cache allocation (model-agnostic, delegated to the
        # attention builder). Each builder owns its tensor layout: MLA →
        # single 576-dim per layer; GDN-hybrid → only num_full_attn rows;
        # MiMo-V2 → defer per-module; standard MHA → split-K/V `[2, L, ...]`.
        # Returned tensors are setattr'd on `self` under their conventional
        # names (kv_cache, kv_scale, index_cache, aligned_index_dim,
        # _kv_layer_cache_store) so binding code and downstream consumers
        # find them where they expect.
        main_kv = self.attn_metadata_builder.allocate_kv_cache_tensors(
            num_kv_heads, num_draft_layers
        )
        for name, value in main_kv.items():
            setattr(self, name, value)

        # Heterogeneous draft (e.g. Eagle3 MHA alongside an MLA target) owns
        # its own KV pool through a sibling builder; same protocol as above,
        # tensors land under namespaced keys (eagle3_kv_cache, eagle3_kv_scale).
        if hasattr(self, "eagle3_draft_builder"):
            draft_kv = self.eagle3_draft_builder.allocate_kv_cache_tensors(
                num_kv_heads, num_draft_layers
            )
            for name, value in draft_kv.items():
                setattr(self, name, value)

        # Per-request cache allocation (model-agnostic, delegated to the
        # attention metadata builder). For GDN this returns
        # `{"mamba_k_cache": ..., "mamba_v_cache": ...}`; for stateless
        # attentions it returns an empty dict (no-op). Tensors are setattr'd
        # on `self` so model layers can access them as `model_runner.<name>`.
        if self.max_per_req_cache_slots > 0:
            per_req_tensors = self.attn_metadata_builder.allocate_per_req_cache(
                self.max_per_req_cache_slots
            )
            for name, tensor in per_req_tensors.items():
                setattr(self, name, tensor)

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
        # Promote to self so the attention builder's build_kv_cache_tensor()
        # can access it without recomputing from drafter state. Heterogeneous
        # drafts (Eagle3 MHA) own their own layer space via their builder.
        # Eagle3 MLA drafts (K2.6) share the target's MLA pool but still
        # appear as one extra layer at index num_hidden_layers. In both Eagle3
        # variants the eagle3 draft model has no `.model.mtp_start_layer_idx`,
        # so only MTP-style drafts take the first branch.
        is_eagle3 = (
            self.config.speculative_config is not None
            and self.config.speculative_config.method == "eagle3"
        )
        self.mtp_start_layer_idx = (
            self.drafter.model.model.mtp_start_layer_idx
            if hasattr(self, "drafter") and not is_eagle3
            else hf_config.num_hidden_layers
        )
        for model_name, model in models_to_bind:
            logger.info(
                f"Binding KV cache for {model_name} model starting at layer_id={layer_id}"
            )

            for module in model.modules():
                # Drafts that own an independent KV pool (Eagle3) bind through
                # their sibling builder first; for unrecognized modules it
                # returns None and we fall through to the target builder.
                if model_name == "draft" and hasattr(self, "eagle3_draft_builder"):
                    kv_cache_tensor = self.eagle3_draft_builder.build_kv_cache_tensor(
                        layer_id, module
                    )
                    if kv_cache_tensor is not None:
                        kv_cache_tensors.append(kv_cache_tensor)
                        layer_id += 1
                        continue

                # Per-attention-type binding is owned by the attention
                # metadata builder; ModelRunner only walks modules and
                # collects the resulting KVCacheTensor entries. The builder
                # returns None for modules it does not recognize (so a
                # sibling module like nn.LayerNorm is silently skipped),
                # and increments through MHA / MLA / GDN / V3.2-indexer
                # internally.
                kv_cache_tensor = self.attn_metadata_builder.build_kv_cache_tensor(
                    layer_id, module
                )
                if kv_cache_tensor is not None:
                    kv_cache_tensors.append(kv_cache_tensor)
                    layer_id += 1

        # Store KVCacheConfig
        kv_cache_data = {
            f"layer_{i}": kv_cache_tensor
            for i, kv_cache_tensor in enumerate(kv_cache_tensors)
        }
        transfer_tensors = self.attn_metadata_builder.get_kv_transfer_tensors()
        if hasattr(self, "eagle3_draft_builder") and transfer_tensors is not None:
            draft_regions = self.eagle3_draft_builder.get_kv_transfer_tensors()
            if draft_regions:
                transfer_tensors.block_regions.extend(draft_regions)
        # Pass the physical block count so the offload connector can byte-slice
        # MLA's token-major latent cache (shape[0] is tokens, not blocks there).
        set_kv_cache_data(
            kv_cache_data,
            config,
            transfer_tensors,
            num_blocks=self.num_physical_kvcache_blocks,
        )

        # Cross-validate: compare estimated vs actual KV cache allocation.
        # `actual_kv_bytes` includes BOTH the unified pool tensors (counted by
        # `block_bytes × num_blocks`) AND the per-request cache tensors (state
        # buffers + SWA window prefix embedded in unified_kv). The budget
        # math in `get_num_blocks()` reserves both separately, so the cross-
        # check must mirror that — otherwise it spuriously fires for any
        # backend with non-zero `compute_per_req_cache_bytes()` (V4, GDN).
        post_alloc = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        actual_kv_bytes = post_alloc - pre_alloc
        # paged-SWA: SWA moved to its own num_swa_blocks pool, so the
        # compressed pool is sized on (block_bytes - swa_block_bytes); add the
        # SWA pool separately. (non-V4 → num_swa_blocks=0, reduces to the
        # original formula.)
        _nswa = getattr(self, "num_swa_blocks", 0)
        _swa_bb = (
            self.attn_metadata_builder.swa_pool_block_bytes()
            if _nswa > 0 and hasattr(self.attn_metadata_builder, "swa_pool_block_bytes")
            else 0
        )
        expected_kv_bytes = (
            (self._compute_block_bytes() - _swa_bb) * num_kvcache_blocks
            + _swa_bb * _nswa
            + self.attn_metadata_builder.compute_per_req_cache_bytes()
            * self.max_per_req_cache_slots
        )
        if expected_kv_bytes > 0:
            diff_pct = abs(actual_kv_bytes - expected_kv_bytes) / expected_kv_bytes
            # 3% threshold: budget formula matches allocation exactly, but the
            # measured `post_alloc - pre_alloc` includes allocator alignment
            # (round to 256 B / 16 MiB segments) and ephemeral init buffers
            # from `_zero_state` / `_neg_inf_state` views, accounting for ~2%
            # noise on multi-GiB pools. Lower thresholds spuriously fire.
            if diff_pct > 0.03:
                logger.warning(
                    f"KV cache allocation mismatch: "
                    f"expected={expected_kv_bytes / (1 << 30):.3f}GB, "
                    f"actual={actual_kv_bytes / (1 << 30):.3f}GB, "
                    f"diff={diff_pct:.1%}"
                )

        # Skip on single-rank: a world_size==1 barrier is a no-op but still
        # forces lazy NCCL communicator creation (CUDA-allocs its buffers),
        # which can OOM/fail on single-card runs. The process group stays
        # initialized so get_tp_group() and friends keep working.
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.barrier()
        return True

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
        max_tokens_across_dp = int(torch.max(num_tokens_across_dp))

        return max_tokens_across_dp - num_tokens, num_tokens_across_dp

    def _maybe_create_tbo_slices(
        self,
        batch,
        is_prefill,
        scheduled_bs,
        actual_num_tokens,
        num_scheduled_tokens,
        tbo_collective_active: bool,
    ):
        """Create TBO ubatch slices when the collective DP decision is True.

        With the packed-reduce path the eligibility (local + cross-DP AND)
        is decided in ``_preprocess``; here we just realise the split.
        """
        if not tbo_collective_active:
            return None

        tbo_num_reqs = batch.total_seqs_num_prefill if is_prefill else scheduled_bs
        # tbo_collective_active is the OR-reduced cross-DP decision: this rank
        # is committed to splitting even if it's below ATOM_TBO_PREFILL_MIN_TOKENS
        # (a peer cleared the bar). force=True bypasses the local min-token gate
        # so we don't desync from peers and hang.
        ubatch_slices = maybe_create_ubatch_slices(
            num_reqs=tbo_num_reqs,
            num_tokens=actual_num_tokens,
            is_prefill=is_prefill,
            num_scheduled_tokens=num_scheduled_tokens if is_prefill else None,
            force=True,
        )
        if ubatch_slices is not None:
            logger.debug(
                f"[TBO] splitting {'prefill' if is_prefill else 'decode'} batch: "
                f"num_reqs={tbo_num_reqs}, ubatches={len(ubatch_slices)}"
            )
        return ubatch_slices

    def _preprocess(
        self,
        batch: ScheduledBatch,
        num_scheduled_tokens: Optional[np.ndarray] = None,
        dspark_shape: Optional[tuple[int, int, int]] = None,
    ):
        """Per-step DP sync: token padding, prefill fan-out, TBO decision.

        Thin wrapper over :func:`atom.utils.tbo.sync_dp_metadata` (the
        actual collective) and :func:`atom.utils.tbo.local_tbo_precompute`
        (the rank-local TBO eligibility / per-ubatch token split).

        ``dspark_shape`` (local q, decode_bs, total_tokens) folds DSpark's graph-shape
        DP-MAX into this same all_gather so the two per-step collectives become
        one; the reduced max is returned as the 7th tuple element for the caller
        to apply via ``_apply_dspark_shape_max``.

        Returns:
            (num_input_tokens, num_tokens_across_dp, dp_uniform_decode,
             max_tokens, tbo_collective_active, ub_max_tokens_across_dp,
             dspark_shape_max)
        """
        num_input_tokens = batch.total_tokens_num
        is_prefill = batch.total_tokens_num_prefill > 0
        tbo_on = self.config.enable_tbo
        dp_size = self.config.parallel_config.data_parallel_size

        # Rank-local TBO precompute (needed for both dp==1 fast path and
        # the cross-DP packed gather below). `meets_min_tokens` = this rank's
        # prefill reached the min-token bar (e.g. 8k), OR-reduced across DP;
        # `can_split` = structurally splittable, AND-reduced across DP.
        local_meets_min_tokens, local_can_split, local_ub0, local_ub1 = (
            False,
            False,
            0,
            0,
        )
        if tbo_on:
            if num_scheduled_tokens is None:
                num_scheduled_tokens = np.asarray(batch.num_scheduled_tokens)
            local_meets_min_tokens, local_can_split, local_ub0, local_ub1 = (
                local_tbo_precompute(
                    self.config, batch, is_prefill, num_scheduled_tokens
                )
            )

        # PCP+TBO prefill: split requests into two GROUPS at a request boundary
        # (never split a sequence's tokens), so each ubatch = "non-TBO PCP on a
        # request subset". Requires num_reqs >= 2 (request-boundary split needs
        # two non-empty groups); bs=1 falls back to non-TBO.
        pcp_size = self.config.prefill_context_parallel_size
        # True for eligible PCP+TBO request-boundary split prefill; read by
        # build_ubatch / run_model / prepare_prefill to route the per-group path.
        self._pcp_tbo_balanced_active = False
        # Per-group descriptors; reset each step, set in prepare_inputs
        # request-boundary-split branch. Guards run_model/build_ubatch against
        # stale values.
        self._pcp_bal_groups = None
        if tbo_on and is_prefill and pcp_size > 1 and not batch.is_dummy_run:
            num_prefill_reqs = batch.total_seqs_num_prefill
            n_prefill = batch.total_tokens_num_prefill
            # Rough local sizing for TBO eligibility. PCP is always dp=1, so the
            # dp_size<=1 fast path below returns local_eligible verbatim as
            # tbo_collective_active; local_ub0/ub1 are only used by the dp>1
            # sync path (never hit under PCP).
            local_tokens = n_prefill // pcp_size
            local_eligible = num_prefill_reqs >= 2 and local_tokens >= 2
            local_ub0 = local_tokens // 2
            local_ub1 = local_tokens - local_ub0
            self._pcp_tbo_balanced_active = local_eligible

        # PCP+TBO prefill: split requests into two GROUPS at a request boundary
        # (never split a sequence's tokens), so each ubatch = "non-TBO PCP on a
        # request subset". Requires num_reqs >= 2 (request-boundary split needs
        # two non-empty groups); bs=1 falls back to non-TBO.
        pcp_size = self.config.prefill_context_parallel_size
        # True for eligible PCP+TBO request-boundary split prefill; read by
        # build_ubatch / run_model / prepare_prefill to route the per-group path.
        self._pcp_tbo_balanced_active = False
        # Per-group descriptors; reset each step, set in prepare_inputs
        # request-boundary-split branch. Guards run_model/build_ubatch against
        # stale values.
        self._pcp_bal_groups = None
        if tbo_on and is_prefill and pcp_size > 1 and not batch.is_dummy_run:
            num_prefill_reqs = batch.total_seqs_num_prefill
            n_prefill = batch.total_tokens_num_prefill
            # Rough local sizing for TBO eligibility. PCP is always dp=1, so the
            # dp_size<=1 fast path below returns local_eligible verbatim as
            # tbo_collective_active; local_ub0/ub1 are only used by the dp>1
            # sync path (never hit under PCP).
            local_tokens = n_prefill // pcp_size
            local_eligible = num_prefill_reqs >= 2 and local_tokens >= 2
            local_ub0 = local_tokens // 2
            local_ub1 = local_tokens - local_ub0
            self._pcp_tbo_balanced_active = local_eligible

        if dp_size <= 1:
            # Single-rank: TBO decision is purely local; no collective needed.
            # Both bits must hold (reached min-tokens AND able to split).
            # dp_uniform_decode=True mirrors the DP-disabled case in the
            # multi-rank branch (`not enable_dp_attention` => True) and the
            # Context default — otherwise single-GPU/TP-only decode would
            # be forced into eager and lose the CUDAGraph decode path.
            return (
                num_input_tokens,
                None,
                True,
                num_input_tokens,
                local_meets_min_tokens and local_can_split,
                None,
                dspark_shape,
            )

        sync = sync_dp_metadata(
            dp_group=get_dp_group().cpu_group,
            dp_size=dp_size,
            num_input_tokens=num_input_tokens,
            is_prefill=is_prefill,
            tbo_on=tbo_on,
            local_meets_min_tokens=local_meets_min_tokens,
            local_can_split=local_can_split,
            local_ub_tokens=(local_ub0, local_ub1),
            dspark_shape=dspark_shape,
        )

        max_tokens = int(sync.num_tokens_across_dp.max())
        dp_uniform_decode = (not sync.any_rank_has_prefill) or (
            not self.config.enable_dp_attention
        )
        if dp_uniform_decode:
            # CUDAGraph path: all ranks pad to the same max for fixed-size all_gather.
            num_input_tokens = max_tokens
        # else: variable-length path — each rank keeps its own token count.

        return (
            num_input_tokens,
            sync.num_tokens_across_dp,
            dp_uniform_decode,
            max_tokens,
            sync.tbo_collective_active,
            sync.ub_max_tokens_across_dp,
            sync.dspark_shape_max,
        )

    def _dspark_apply_q_bucket(self, batch: ScheduledBatch) -> None:
        """Shrink this decode step's verify length to one CUDA-graph bucket q.

        q = quantize_up(max ell_i + 1) over the batch (ell_i = last step's
        per-req schedule). All seqs then forward q tokens (anchor + q-1 drafts)
        instead of mtp_k+1, and replay picks the (bs, q) graph; the dropped
        draft suffix is re-drafted next step -> lossless.

        Mutates only the worker's batch copy (counts + scheduled_spec_decode_
        tokens truncated to q-1); KV stays reserved at mtp_k+1. No-op unless
        DSpark confidence scheduling is on and this is a pure-decode batch.
        """
        # Idempotency guard: prepare_model calls this before prepare_input_ids,
        # and prepare_inputs (also reachable standalone for dummy/warmup) calls
        # it again — only the first application must shrink the batch.
        if getattr(batch, "_dspark_q_applied", False):
            return
        if not (
            hasattr(self, "drafter")
            and getattr(self.drafter, "dspark_confidence_schedule", False)
        ):
            return
        if batch.total_tokens_num_prefill > 0:
            return  # mixed/prefill step: keep full length
        batch._dspark_q_applied = True
        scheduled_bs = batch.total_seqs_num_decode
        if scheduled_bs <= 0:
            return
        full_q = self.drafter.mtp_k + 1

        # {req_id: ell} from the PREVIOUS step's propose() (verify_scheduler,
        # same process). The worker batch copy has req_ids but NOT the
        # scheduler-side `seqs` dict, so look ell up by req_id. A request with no
        # prior ell (new this step) -> full length (never under-verify).
        verify_scheduler = self.drafter.verify_scheduler
        by_req = (
            verify_scheduler.ell_by_req if verify_scheduler is not None else None
        ) or {}
        if not by_req:
            return

        # ==== RAGGED path (paper §5.2 avoid-padding) — FULLY INDEPENDENT =====
        # This branch is hoisted ABOVE the q-bucket early-return so it never
        # depends on dspark.q_buckets. Each decode seq forwards its own
        # ell_r+1 tokens (no batch-level pad to a single q). num_scheduled_tokens
        # becomes a true ragged array; all V4 attn metadata/kernels are already
        # per-token + marker-driven, so this is the only construction change.
        # Graph replay picks a (bs, q_eff) graph captured from the independent
        # dspark.ragged_graph_sizes set. Anchor lower bound (q>=num_bonus+1)
        # is applied PER REQUEST so each seg can hold its own anchor.
        if self.config.dspark.ragged:
            self._dspark_apply_ragged(batch, scheduled_bs, full_q, by_req)
            return
        # ====================================================================

        # ---- Q-BUCKET path (older batch-uniform padding scheme) ------------
        from atom.spec_decode.dspark_scheduler import (
            quantize_to_bucket,
            resolve_q_buckets,
        )

        buckets = resolve_q_buckets(self.config.dspark.q_buckets, full_q)
        if buckets == [full_q]:
            return  # no smaller buckets configured -> Phase-1 behavior

        max_ell = 0
        for rid in batch.req_ids[:scheduled_bs]:
            ell = by_req.get(rid)
            max_ell = full_q - 1 if ell is None else max(max_ell, int(ell))
            if max_ell >= full_q - 1:
                break

        # Lower bound q >= max_num_bonus + 1: ell is only the PREDICTED accept
        # count, but the anchor sits at the PREVIOUS step's ACTUAL num_bonus. If
        # q-1 < num_bonus the anchor falls outside the shrunk segment and the
        # draft propose scatter/index_select goes OOB. No-op when num_bonus is
        # unavailable (first decode step).
        max_num_bonus = 0
        num_bonus_arr = getattr(batch, "num_bonus", None)
        if num_bonus_arr is not None:
            nb = np.asarray(num_bonus_arr)[:scheduled_bs]
            if nb.size > 0:
                max_num_bonus = int(nb.max())
        need = max(max_ell + 1, max_num_bonus + 1)
        q = quantize_to_bucket(need, buckets)
        if q >= full_q:
            return  # no shrink possible this step

        # Rebuild scheduled_tokens (flat [seq0 tokens | seq1 tokens | ...]) to the
        # new q-per-seq layout BEFORE rewriting the counts (need the old per-seq
        # lengths to slice). Pure-decode step (we returned early on prefill), so
        # the array is entirely decode segments. Keep the first q of each seq's
        # segment: token[0] is the anchor; the rest are placeholders overwritten
        # by token_ids[:, 1:] = scheduled_spec_decode_tokens downstream.
        old_nst = np.asarray(batch.num_scheduled_tokens, dtype=np.int32)
        sched = np.asarray(batch.scheduled_tokens)
        old_cu = np.zeros(scheduled_bs + 1, dtype=np.int64)
        np.cumsum(old_nst[:scheduled_bs], out=old_cu[1:])
        new_sched = np.empty(scheduled_bs * q, dtype=sched.dtype)
        for i in range(scheduled_bs):
            start = int(old_cu[i])
            new_sched[i * q : (i + 1) * q] = sched[start : start + q]
        batch.scheduled_tokens = new_sched

        # Rewrite decode token counts to q (anchor + q-1 drafts) per seq.
        nst = old_nst.copy()
        prefill_tok = int(batch.total_tokens_num_prefill)
        nst[:scheduled_bs] = q
        batch.num_scheduled_tokens = nst
        batch.total_tokens_num_decode = int(nst[:scheduled_bs].sum())
        batch.total_tokens_num = prefill_tok + batch.total_tokens_num_decode
        # Publish the chosen q as the single source of truth (see ScheduledBatch
        # .num_spec_query_tokens). All downstream length consumers read this.
        batch.num_spec_query_tokens = q
        # Truncate each request's draft block to q-1 (regular matrix: all seqs q-1).
        spec = batch.scheduled_spec_decode_tokens
        if spec is not None and getattr(spec, "size", 0) > 0:
            batch.scheduled_spec_decode_tokens = np.ascontiguousarray(spec[:, : q - 1])

    def _dspark_apply_ragged(self, batch, scheduled_bs, full_q, by_req):
        """DSpark per-request RAGGED verify (paper §5.2 avoid-padding).

        Sets num_scheduled_tokens[i] = len_i PER REQUEST (no batch-level pad to a
        single q), where len_i = max(ell_i, max_num_bonus) + 1, clamped to
        [1, full_q]. Downstream V4 attn is marker-driven (cu_seqlens etc.) so a
        ragged num_scheduled_tokens flows through unchanged; dropped draft suffix
        is re-drafted next step -> lossless. KV stays reserved at mtp_k+1.
        """
        old_nst = np.asarray(batch.num_scheduled_tokens, dtype=np.int32)

        tp = getattr(self, "tokenID_processor", None)
        prev_b = getattr(tp, "prev_batch", None) if tp is not None else None
        cur_req = list(batch.req_ids[:scheduled_bs])
        prev_req = list(prev_b.req_ids) if prev_b is not None else None
        # is_all_same premise: previous batch is exactly this decode set, same
        # order (no new/prefill seqs, no reorder). Any deviation → boundary step.
        if prev_req is None or prev_req != cur_req:
            return  # boundary / reorder step: skip ragged, stay rectangular

        num_bonus_arr = getattr(batch, "num_bonus", None)
        nb = (
            np.asarray(num_bonus_arr)[:scheduled_bs]
            if num_bonus_arr is not None
            else None
        )
        max_nb = int(nb.max()) if nb is not None and nb.size > 0 else 0

        # Per-request forward length = max(ell_i, max_num_bonus) + 1, in [1, full_q].
        new_len = np.empty(scheduled_bs, dtype=np.int32)
        any_shrink = False
        for i, rid in enumerate(batch.req_ids[:scheduled_bs]):
            ell = by_req.get(rid)
            ell_i = full_q - 1 if ell is None else int(ell)
            ell_i = max(ell_i, max_nb)
            li = ell_i + 1
            if li < 1:
                li = 1
            elif li > full_q:
                li = full_q
            new_len[i] = li
            if li < int(old_nst[i]):
                any_shrink = True

        from atom.spec_decode.dspark_scheduler import (
            quantize_to_bucket,
            resolve_q_buckets,
        )

        if not any_shrink:
            return  # nothing to shrink this step -> Phase-1 layout

        # Rebuild scheduled_tokens (flat) to the ragged per-seq layout: keep the
        # first new_len[i] of each seq's old segment (token[0]=anchor, rest=draft
        # placeholders already populated by the scheduler from seq.token_ids).
        sched = np.asarray(batch.scheduled_tokens)
        old_cu = np.zeros(scheduled_bs + 1, dtype=np.int64)
        np.cumsum(old_nst[:scheduled_bs], out=old_cu[1:])
        new_cu = np.zeros(scheduled_bs + 1, dtype=np.int64)
        np.cumsum(new_len, out=new_cu[1:])
        total_new = int(new_cu[-1])
        new_sched = np.empty(total_new, dtype=sched.dtype)
        for i in range(scheduled_bs):
            s_old = int(old_cu[i])
            s_new = int(new_cu[i])
            new_sched[s_new : s_new + new_len[i]] = sched[s_old : s_old + new_len[i]]
        batch.scheduled_tokens = new_sched

        nst = old_nst.copy()
        nst[:scheduled_bs] = new_len
        batch.num_scheduled_tokens = nst
        prefill_tok = int(batch.total_tokens_num_prefill)
        batch.total_tokens_num_decode = total_new
        batch.total_tokens_num = prefill_tok + total_new
        # Two sources of truth (TRUE FLAT, paper §5.2): tokens are flat-packed
        # [0:Σ] with the per-seq ragged new_len.
        #   * dynamic_spec_query_tokens_per_req : the true ragged per-seq lengths.
        #   * num_spec_query_tokens (scalar) : graph CAPACITY selector q_eff, so
        #     C = bs*q_eff >= Σ (q_eff = ceil(Σ/bs) quantized up to a captured
        #     bucket). Graph replays a fixed C grid; tail [Σ:C] is -1-batch_id
        #     padding (kernels skip it). C tracks the SUM, not bs*max_len, so a
        #     long tail seq no longer inflates the whole batch (win over q-bucket).
        buckets = resolve_q_buckets(self.config.dspark.ragged_graph_sizes, full_q)
        if self.enforce_eager:
            # Eager: no graph → capacity == exact Σ (no bucket). Scalar = batch max
            # real len (positions/attn bound); layout is pure flat Σ.
            q_eff = int(new_len.max()) if scheduled_bs > 0 else full_q
        else:
            # Graph: pick the smallest bucket q_eff with bs*q_eff >= Σ.
            q_ceil = (total_new + scheduled_bs - 1) // max(scheduled_bs, 1)
            q_eff = quantize_to_bucket(q_ceil, buckets)
        batch.num_spec_query_tokens = int(q_eff)
        batch.dynamic_spec_query_tokens_per_req = new_len

        # (No flat scheduled_spec_decode_tokens is built here: the ragged
        # input_ids are assembled downstream in _ragged_fill_deferred_all_same
        # from prev_token_ids (anchor) + draft_token_ids, which never consults
        # scheduled_spec_decode_tokens.)

    def _dspark_sync_graph_shape_dp(self, batch: ScheduledBatch) -> None:
        """DP-attention: force the decode graph shape (bs, q) IDENTICAL on every
        DP rank via an all-reduce MAX of (q, decode_bs, total_tokens).

        DSpark scheduling picks q/bs per rank from the local batch, so they
        diverge; the decode graph's MoE all_gather pads to ``graph_bs *
        max_seqlen_q``, so divergent shapes -> mismatched collective rows ->
        RCCL deadlock. Adopting the DP-max only enlarges graph capacity (real
        tokens stay flat-packed, extra slots are skipped padding) -> lossless;
        per-request ``num_scheduled_tokens`` is untouched so attention stays
        ragged. Called on every rank each step (real + dummy) so the collective
        never deadlocks; no-op for single-DP or non-spec.

        The real hot-path collective is folded into ``sync_dp_metadata`` (see
        ``_dspark_local_shape`` / ``_apply_dspark_shape_max``); this standalone
        method is only the non-merged call path.
        """
        shape = self._dspark_local_shape(batch)
        if shape is None:
            return
        import torch.distributed as dist

        shape_t = torch.tensor(list(shape), device="cpu", dtype=torch.int64)
        dist.all_reduce(shape_t, op=dist.ReduceOp.MAX, group=get_dp_group().cpu_group)
        self._apply_dspark_shape_max(
            batch, (int(shape_t[0]), int(shape_t[1]), int(shape_t[2]))
        )

    def _dspark_local_shape(
        self, batch: ScheduledBatch
    ) -> Optional[tuple[int, int, int]]:
        """Local (q, decode_bs, total_tokens) for the DSpark DP graph-shape sync,
        or None when the sync does not apply (single-DP or non-DSpark).

        Symmetric across ranks every step regardless of prefill/decode: on a
        prefill step this returns (1, 0, 0) so the DP-MAX reduction still has a
        well-defined identity contribution and every rank participates in the
        same collective (matching the pre-merge all_reduce semantics)."""
        if self.config.parallel_config.data_parallel_size <= 1:
            return None
        drafter = getattr(self, "drafter", None)
        if drafter is None or not getattr(drafter, "use_dspark", False):
            return None
        local_q = int(getattr(batch, "num_spec_query_tokens", 1))
        local_bs = int(getattr(batch, "total_seqs_num_decode", 0))
        # Also DP-max the ragged decode token total (total_tokens_num_decode).
        # PIECEWISE 1D-ragged replays at a num_tokens bucket sized to this total;
        # to keep the MoE all_gather row count identical across ranks (else RCCL
        # deadlock) every rank must pick the SAME bucket, so bisect on the DP-max
        # total, not the local one.
        local_total_tokens = int(getattr(batch, "total_tokens_num_decode", 0))
        return local_q, local_bs, local_total_tokens

    def _apply_dspark_shape_max(
        self, batch: ScheduledBatch, shape_max: Optional[tuple[int, int, int]]
    ) -> None:
        """Adopt the DP-MAX (q, decode_bs, total_tokens). Raising q/bs/total only
        enlarges graph capacity (real tokens stay flat-packed in [0:total]), so
        it is always lossless; see _dspark_sync_graph_shape_dp docstring."""
        if shape_max is None:
            return
        batch.num_spec_query_tokens = int(shape_max[0])
        batch.dspark_dp_bs = int(shape_max[1])
        batch.dspark_dp_total_tokens = int(shape_max[2])

    def prepare_inputs(
        self,
        batch: ScheduledBatch,
        input_ids: torch.Tensor = None,
        preprocessed: Optional[tuple] = None,
    ):
        # NOTE: DSpark q-bucket shrink happens in prepare_model BEFORE
        # prepare_input_ids, so the batch is already reduced when we get here.
        # ``preprocessed``: when prepare_model already ran the merged DP collective
        # (to get the DSpark DP-max q before prepare_input_ids), it passes the
        # cached _preprocess tuple here so we DON'T issue a second all_gather.
        is_prefill = batch.total_tokens_num_prefill > 0
        bs = batch.total_seqs_num
        num_scheduled_tokens = np.asarray(batch.num_scheduled_tokens)
        cu_seqlens_q, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
        if preprocessed is None:
            preprocessed = self._preprocess(
                batch,
                num_scheduled_tokens=num_scheduled_tokens,
                dspark_shape=self._dspark_local_shape(batch),
            )
            self._apply_dspark_shape_max(batch, preprocessed[6])
        (
            num_input_tokens,
            num_tokens_across_dp,
            dp_uniform_decode,
            max_tokens,
            tbo_collective_active,
            ub_max_tokens_across_dp,
            _dspark_shape_max,
        ) = preprocessed

        if not tbo_collective_active:
            self._pcp_tbo_balanced_active = False

        self.forward_vars["cu_seqlens_q"].np[1 : bs + 1] = cu_seqlens_q

        # mtp_step = per-seq decode token count, used by ForwardMode.decide to
        # recover batch size as num_input_tokens // mtp_step. This is exactly
        # the batch's single-source-of-truth decode length (= mtp_k+1, or the
        # DSpark q-bucket when shrunk); num_input_tokens = scheduled_bs *
        # num_spec_query_tokens, so the division recovers bs correctly. Prefill
        # has no drafter / uses 1.
        decide_num_input_tokens = num_input_tokens
        dp_bs = batch.dspark_dp_bs
        is_ragged = (
            getattr(batch, "dynamic_spec_query_tokens_per_req", None) is not None
        )
        if not is_prefill and (dp_bs is not None or is_ragged):
            # DSpark: the real Σtokens is irregular, but we
            # replay the rectangular (bs, q_eff) graph whose capacity is bs*q_eff
            # (q_eff = num_spec_query_tokens, the quantized bucket). Feed
            # ForwardMode the GRAPH-CAPACITY token count so it recovers
            # padded_scheduled_bs = bs*q_eff // q_eff = bs and picks the matching
            # (bs, q_eff) graph; the real ragged tokens sit in [0:Σ], the tail is
            # -1 padding (CTAs bail).
            q_eff = int(batch.num_spec_query_tokens)
            eff_bs = dp_bs if dp_bs is not None else batch.total_seqs_num_decode
            mtp_step = q_eff
            decide_num_input_tokens = int(eff_bs) * q_eff
        elif not is_prefill and hasattr(self, "drafter"):
            mtp_step = batch.num_spec_query_tokens
        else:
            mtp_step = (self.drafter.mtp_k + 1) if hasattr(self, "drafter") else 1
        forward_mode = ForwardMode.decide(
            is_prefill=is_prefill,
            total_seqs_num=batch.total_seqs_num,
            scheduled_bs_decode=batch.total_seqs_num_decode,
            num_input_tokens=decide_num_input_tokens,
            dp_uniform_decode=dp_uniform_decode,
            enforce_eager=self.enforce_eager,
            graph_bs=self.graph_bs,
            mtp_step=mtp_step,
        )

        if not is_prefill:
            scheduled_bs = batch.total_seqs_num_decode
            bs = forward_mode.effective_bs  # single source of truth
            assert bs >= scheduled_bs, (
                f"effective_bs={bs} < scheduled_bs={scheduled_bs}; "
                f"ForwardMode.decide invariant violated"
            )
            # Only pad cu_seqlens_q out to the cudagraph capture size if we
            # actually grew bs. Eager (bs == scheduled_bs) leaves the slice
            # empty so no overwrite happens.
            if bs > scheduled_bs:
                self.forward_vars["cu_seqlens_q"].np[scheduled_bs + 1 : bs + 1] = (
                    self.forward_vars["cu_seqlens_q"].np[scheduled_bs]
                )
        attn_metadata, positions = self.attn_metadata_builder.build(batch=batch, bs=bs)
        context_bs = batch.total_seqs_num_prefill if is_prefill else scheduled_bs

        # MoE's pad_for_all_gather reads context.graph_bs to pad hidden_states
        # before a cross-DP all_gather, so it must be unified across DP ranks
        # under uniform decode (where pad path is taken). Use forward_mode's
        # moe_pad_bs, which equals effective_bs except in the uniform-eager
        # corner (enforce_eager / bs>graph_bs[-1]) where attention needs local
        # but MoE pad needs the DP-unified padded_scheduled_bs.
        graph_bs = num_input_tokens if is_prefill else forward_mode.moe_pad_bs
        drafter = getattr(self, "drafter", None)
        if not is_prefill and getattr(drafter, "use_dspark", False):
            graph_bs = self._dspark_ragged_moe_graph_bs(batch, graph_bs)
        context = Context(
            positions=positions,
            is_prefill=is_prefill,
            is_dummy_run=batch.is_dummy_run,
            batch_size=context_bs,
            graph_bs=graph_bs,
            dp_uniform_decode=dp_uniform_decode,
            forward_mode=forward_mode,
        )

        actual_num_tokens = batch.total_tokens_num

        spec_decode_metadata = None
        if not is_prefill and hasattr(self, "drafter") and not batch.is_dummy_run:
            scheduled_bs = batch.total_seqs_num_decode
            spec_decode_metadata = self.drafter.calc_spec_decode_metadata(
                num_scheduled_tokens[:scheduled_bs],
                cu_seqlens_q[:scheduled_bs],
                input_ids,
            )

        pcp_size = self.config.prefill_context_parallel_size
        _pcp_tbo_balanced = (
            is_prefill
            and pcp_size > 1
            and tbo_collective_active
            and not batch.is_dummy_run
            and getattr(self, "_pcp_tbo_balanced_active", False)
        )
        if _pcp_tbo_balanced:
            # Request-boundary split for PCP+TBO prefill (see
            # _build_pcp_balanced_slices). forward_vars stay GLOBAL here.
            ubatch_slices, self._pcp_bal_groups = self._build_pcp_balanced_slices(
                batch, num_scheduled_tokens, pcp_size
            )
        else:
            ubatch_slices = self._maybe_create_tbo_slices(
                batch,
                is_prefill,
                scheduled_bs if not is_prefill else 0,
                actual_num_tokens,
                num_scheduled_tokens,
                tbo_collective_active,
            )

        set_forward_context(
            attn_metadata=attn_metadata,
            atom_config=self.config,
            context=context,
            num_tokens=actual_num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            spec_decode_metadata=spec_decode_metadata,
            ubatch_slices=ubatch_slices,
            ub_max_tokens_across_dp=ub_max_tokens_across_dp,
        )
        return graph_bs

    def prepare_sample(
        self, batch: ScheduledBatch
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool, bool]:
        bs = batch.total_seqs_num

        # Check on CPU whether all requests are greedy (temperature=0)
        all_greedy = (batch.temperatures == 0).all()

        # Check on CPU whether any fan-out sibling needs per-row random noise.
        # Missing attribute (e.g. dummy runs, older callers) -> False.
        needs_independent_noise = bool(
            getattr(batch, "needs_independent_noise", np.zeros(0, dtype=bool)).any()
        )

        temp_buffer = self.forward_vars["temperatures"]
        # Clamp temperatures on CPU to avoid division by zero in sampler
        temp_buffer.np[:bs] = np.maximum(batch.temperatures, SAMPLER_EPS)
        temperatures = temp_buffer.copy_to_gpu(bs)

        # Check on CPU whether filtering is needed to avoid GPU sync in sampler.
        # If no filtering needed, return None to skip GPU copy entirely.
        needs_topk = (batch.top_ks != -1).any()
        needs_topp = (batch.top_ps < 1.0).any()

        if needs_topk:
            top_k_buffer = self.forward_vars["top_ks"]
            top_k_buffer.np[:bs] = batch.top_ks
            # If all values are the same, only copy one element to save bandwidth
            if bs > 1 and (batch.top_ks == batch.top_ks[0]).all():
                top_ks = top_k_buffer.copy_to_gpu(1)
            else:
                top_ks = top_k_buffer.copy_to_gpu(bs)
        else:
            top_ks = None

        if needs_topp:
            top_p_buffer = self.forward_vars["top_ps"]
            top_p_buffer.np[:bs] = batch.top_ps
            # If all values are the same, only copy one element to save bandwidth
            if bs > 1 and (batch.top_ps == batch.top_ps[0]).all():
                top_ps = top_p_buffer.copy_to_gpu(1)
            else:
                top_ps = top_p_buffer.copy_to_gpu(bs)
        else:
            top_ps = None

        return temperatures, top_ks, top_ps, all_greedy, needs_independent_noise

    def prepare_model(self, batch: ScheduledBatch):
        self._dspark_apply_q_bucket(batch)
        # DSpark-only early DP sync: only DSpark under DP needs the DP-max q
        # BEFORE prepare_input_ids (to size input_ids). Run the merged packed
        # all_gather (TBO + DSpark [q, bs, total_tokens]) ONCE here and reuse it in
        # prepare_inputs, so the step issues a single cross-DP collective.
        dspark_shape = self._dspark_local_shape(batch)
        preprocessed = None
        if dspark_shape is not None:
            preprocessed = self._preprocess(
                batch,
                num_scheduled_tokens=np.asarray(batch.num_scheduled_tokens),
                dspark_shape=dspark_shape,
            )
            self._apply_dspark_shape_max(batch, preprocessed[6])
        total_tokens_num = batch.total_tokens_num
        assert total_tokens_num > 0

        temperatures, top_ks, top_ps, all_greedy, needs_independent_noise = (
            self.prepare_sample(batch)
        )
        input_ids = self.tokenID_processor.prepare_input_ids(batch)
        self.prepare_inputs(batch, input_ids, preprocessed=preprocessed)
        return (
            input_ids,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            needs_independent_noise,
        )

    @staticmethod
    def _detailed_label_suffix(batch: Optional[ScheduledBatch]) -> str:
        """Detailed attention aggregates for the trace label, or ``""``.

        These fields are only populated by
        `Scheduler.compute_detailed_aggregates` when profiling is active
        and ``ATOM_ENABLE_DETAILED_ANNOTATION`` is set, so on the normal
        (unprofiled) path this returns an empty string without any extra work.
        Appending here keeps the annotation on the ``prefill[]``/``decode[]``
        ``record_function`` (a GPU-recognized layer) instead of nesting an
        extra span above ``run_model``.
        """
        if batch is None or batch.detailed_sqsq is None:
            return ""
        return (
            f" sqsq={batch.detailed_sqsq}"
            f" sqsk={batch.detailed_sqsk}"
            f" sk={batch.detailed_sk}"
        )

    def _build_pcp_balanced_slices(
        self,
        batch: ScheduledBatch,
        num_scheduled_tokens: np.ndarray,
        pcp_size: int,
    ) -> "tuple[list[UBatchSlice], list[PcpBalGroup]]":
        """Build request-boundary-split ubatch slices for PCP+TBO prefill.

        Split REQUESTS into two groups at a request boundary near the token
        midpoint. Each group is an independent "non-TBO PCP mini-batch": padded
        to a pcp multiple and round-robin striped as a whole, so every sequence
        stays intact in one group (root-fixes token-split R1/R2). forward_vars
        stay GLOBAL here; build_ubatch_prefill_metadata slices the FULL
        (un-reindexed) metadata per group and calls _apply_pcp_reindex on it.

        Returns (ubatch_slices, groups): token_slice is in the LOCAL concat
        space [g0_local | g1_local] that run_model produces (see
        _apply_pcp_balanced_stripe); groups are the PcpBalGroup descriptors
        consumed by run_model (per-group stripe) and
        build_ubatch_prefill_metadata (slice + reindex).
        """
        num_prefill_reqs = batch.total_seqs_num_prefill
        per_req = np.asarray(num_scheduled_tokens[:num_prefill_reqs], dtype=np.int64)
        total_tok = int(per_req.sum())
        cum = np.cumsum(per_req)  # cum[j] = sum of reqs [0..j]
        target = total_tok // 2
        # request boundary whose cumulative token count is closest to target
        split_idx = int(np.searchsorted(cum, target, side="left")) + 1
        split_idx = max(1, min(split_idx, num_prefill_reqs - 1))
        # global token count of group0 (reqs [0:split_idx])
        b0 = int(cum[split_idx - 1])
        h0 = pcp_pad_len(b0, pcp_size)
        h1 = pcp_pad_len(total_tok - b0, pcp_size)
        l0 = h0 // pcp_size
        l1 = h1 // pcp_size
        ubatch_slices = [
            UBatchSlice(
                request_slice=slice(0, split_idx),
                token_slice=slice(0, l0),
            ),
            UBatchSlice(
                request_slice=slice(split_idx, num_prefill_reqs),
                token_slice=slice(l0, l0 + l1),
            ),
        ]
        groups = [
            PcpBalGroup(0, split_idx, 0, b0, h0),
            PcpBalGroup(split_idx, num_prefill_reqs, b0, total_tok, h1),
        ]
        return ubatch_slices, groups

    def _apply_pcp_balanced_stripe(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        groups: "list[PcpBalGroup]",
        pcp_size: int,
        forward_context,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """PCP+TBO prefill per-group round-robin stripe, before UBatchWrapper.

        Each request group is padded to a pcp multiple and round-robin striped
        as a WHOLE (so sequences stay intact per group), then the two groups'
        1/pcp shards are concatenated into [g0_local | g1_local]. token_slice
        (built in prepare_inputs) indexes into this concat. Returns the striped
        (input_ids, positions).
        """
        g_ids, g_pos = [], []
        for grp in groups:
            seg_ids = input_ids[grp.tok_start : grp.tok_end]
            seg_pos = positions[grp.tok_start : grp.tok_end]
            pad = grp.pad_total - (grp.tok_end - grp.tok_start)
            if pad > 0:
                seg_ids = torch.cat([seg_ids, seg_ids.new_zeros(pad)])
                seg_pos = torch.cat([seg_pos, seg_pos.new_zeros(pad)])
            g_ids.append(pcp_round_robin_split(seg_ids, pcp_size))
            g_pos.append(pcp_round_robin_split(seg_pos, pcp_size))
        input_ids = torch.cat(g_ids)
        positions = torch.cat(g_pos)
        # context.positions = local per-group concat so _make_ubatch_context
        # slices each ubatch's forward positions correctly.
        forward_context.context.positions = positions
        # Hash MoE: local per-group-concat ids. Each ForCausalLM.forward
        # allgathers its ubatch's slice (g_i local, H_i/pcp) across pcp ranks →
        # H_i ids, matching moe_pcp_merge_forward's per-ubatch hidden allgather.
        if envs.ATOM_PCP_MOE_MERGE:
            forward_context.context.input_ids = input_ids
        return input_ids, positions

    def _restore_pcp_balanced_output(
        self,
        mo: torch.Tensor,
        groups: "list[PcpBalGroup]",
        pcp_size: int,
    ) -> torch.Tensor:
        """Restore PCP+TBO request-boundary-split output.

        UBatchWrapper concatenated the two groups' 1/pcp output shards
        [g0_local | g1_local]. Each group was striped independently, so restore
        per group: pcp_allgather_rerange its shard back to the group's global
        order, crop the per-group pad, then concat to the full global sequence.
        """
        outs = []
        off = 0
        for grp in groups:
            local_len = grp.pad_total // pcp_size  # group's 1/pcp token count
            seg = pcp_allgather_rerange(mo[off : off + local_len], pcp_size)
            outs.append(seg[: grp.tok_end - grp.tok_start])  # crop per-group pad
            off += local_len
        return torch.cat(outs)

    def run_model(
        self,
        input_ids: torch.Tensor,
        batch: Optional[ScheduledBatch] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_context = get_forward_context()
        context = forward_context.context
        bs = context.batch_size
        is_prefill = context.is_prefill
        positions = context.positions

        # Dispatch is owned by ForwardMode.decide() (called in prepare_inputs).
        # Every run_model caller MUST go through prepare_inputs first, so
        # forward_mode is always set here.
        forward_mode = context.forward_mode
        assert forward_mode is not None, (
            "context.forward_mode is None; run_model invoked without going "
            "through prepare_inputs. Add ForwardMode.decide() at the new "
            "entry point instead of re-deriving the 4-OR dispatch here."
        )

        # Single canonical shape check; contract owned by ForwardMode, which
        # internally short-circuits for prefill / cudagraph.
        forward_mode.assert_shape_contract(input_ids, forward_context.attn_metadata)

        # Profiler label. Kind (prefix) distinguishes real/dummy and
        # eager/cudagraph; `tbo=1` marks a step that ran TBO ubatches. See
        # `build_run_label`.
        label = build_run_label(
            is_prefill=is_prefill,
            use_cudagraph=forward_mode.use_cudagraph,
            is_dummy=context.is_dummy_run,
            tbo_on=forward_context.ubatch_slices is not None,
            bs=bs,
            # The CUDAGraph replays a padded batch (context.graph_bs); pass it so
            # the label shows bs=<real>/<graph> when they differ.
            graph_bs=context.graph_bs if forward_mode.use_cudagraph else None,
            batch=batch,
        )

        # Profiler label. Kind (prefix) distinguishes real/dummy and
        # eager/cudagraph; `tbo=1` marks a step that ran TBO ubatches. See
        # `build_run_label`.
        label = build_run_label(
            is_prefill=is_prefill,
            use_cudagraph=forward_mode.use_cudagraph,
            is_dummy=context.is_dummy_run,
            tbo_on=forward_context.ubatch_slices is not None,
            bs=bs,
            # The CUDAGraph replays a padded batch (context.graph_bs); pass it so
            # the label shows bs=<real>/<graph> when they differ.
            graph_bs=context.graph_bs if forward_mode.use_cudagraph else None,
            batch=batch,
            detailed_suffix=self._detailed_label_suffix(batch),
        )

        # PCP+TBO prefill: per-group round-robin stripe before UBatchWrapper (see
        # _apply_pcp_balanced_stripe). _pcp_tbo_balanced also gates the per-group
        # output restore further below.
        _pcp_size = self.config.prefill_context_parallel_size
        _pcp_bal_groups = getattr(self, "_pcp_bal_groups", None)
        _pcp_tbo_balanced = (
            _pcp_size > 1
            and isinstance(self.model, UBatchWrapper)
            and forward_context.ubatch_slices is not None
            and is_prefill
            and not forward_context.context.is_dummy_run
            and _pcp_bal_groups is not None
        )
        if _pcp_tbo_balanced:
            input_ids, positions = self._apply_pcp_balanced_stripe(
                input_ids, positions, _pcp_bal_groups, _pcp_size, forward_context
            )

        if not forward_mode.use_cudagraph:
            # prefill, or decode forced eager (enforce_eager / DP peer
            # prefill / bs above the largest captured graph).
            with record_function(label):
                # Handle multimodal prefill: compute vision embeddings and merge
                inputs_embeds = None
                if (
                    is_prefill
                    and hasattr(self.model, "get_vision_embeddings")
                    and batch is not None
                    and hasattr(batch, "multimodal_data")
                    and batch.multimodal_data
                ):
                    mm_data_values = list(batch.multimodal_data.values())
                    pixel_values = torch.cat(
                        [mm_data["pixel_values"] for mm_data in mm_data_values], dim=0
                    ).to(device=self.device, dtype=self.config.torch_dtype)
                    grid_thw = torch.cat(
                        [mm_data["image_grid_thw"] for mm_data in mm_data_values],
                        dim=0,
                    ).to(device=self.device)
                    vision_embeds = self.model.get_vision_embeddings(
                        pixel_values, grid_thw
                    )
                    text_embeds = self.model.embed_input_ids(input_ids)
                    inputs_embeds = self.model.merge_multimodal_embeddings(
                        input_ids, text_embeds, vision_embeds
                    )

                if inputs_embeds is None:
                    model_output = self.model(input_ids, positions)
                else:
                    model_output = self.model(
                        input_ids, positions, inputs_embeds=inputs_embeds
                    )
                # PCP+TBO prefill (request-boundary split): UBatchWrapper concatenated the two
                # groups' 1/pcp output shards [g0_local | g1_local]. Restore each
                # group independently: pcp_allgather_rerange its shard back to the
                # group's global order, crop off the per-group pad, then concat to
                # the full global sequence. Per-group (not single global) because
                # each group was striped independently.
                if _pcp_tbo_balanced:
                    if self.use_aux_hidden_state_outputs:
                        _h, _aux = model_output
                        model_output = (
                            self._restore_pcp_balanced_output(
                                _h, _pcp_bal_groups, _pcp_size
                            ),
                            _aux,
                        )
                    else:
                        model_output = self._restore_pcp_balanced_output(
                            model_output, _pcp_bal_groups, _pcp_size
                        )
                if self.use_aux_hidden_state_outputs:
                    hidden_states, self._aux_hidden_states = model_output
                else:
                    hidden_states = model_output
                    self._aux_hidden_states = None
                # DSpark captures aux hidden states via forward hooks (the model
                # itself returns only hidden_states); assemble them in order.
                self._collect_dspark_aux(hidden_states.shape[0])
                logits = self.model.compute_logits(hidden_states)
        else:
            # decode[bs=128 tok=128 d=128] / decode[... p=2 d=126 spec=3] /
            # dummy_decode[...] — see build_run_label.
            with record_function(label):
                graph_bs = context.graph_bs
                max_q_len = forward_context.attn_metadata.max_seqlen_q
                num_tokens = context.batch_size * max_q_len  # real (output slice)

                if self._piecewise_cg_active():
                    num_tokens_pad, real_tokens, _captured = (
                        self._piecewise_replay_shape(batch, graph_bs, max_q_len)
                    )
                    _is_dummy = batch is not None and batch.is_dummy_run
                    # Pad tail to a legal vocab id / position (builder fills to
                    # graph_cap >= num_tokens_pad, so a no-op safety net).
                    if num_tokens_pad > real_tokens:
                        self.forward_vars["input_ids"].gpu[
                            real_tokens:num_tokens_pad
                        ].zero_()
                        self.forward_vars["positions"].gpu[
                            real_tokens:num_tokens_pad
                        ].zero_()
                    _pos = (
                        self._mrope_positions_view(num_tokens_pad)
                        if self.use_mrope
                        else self.forward_vars["positions"].gpu[:num_tokens_pad]
                    )
                    forward_context.cudagraph_runtime_mode = (
                        CUDAGraphMode.PIECEWISE
                        if (not _is_dummy and _captured)
                        else CUDAGraphMode.NONE
                    )
                    forward_context.batch_descriptor = BatchDescriptor(
                        num_tokens=num_tokens_pad
                    )
                    model_output = self.model(
                        self.forward_vars["input_ids"].gpu[:num_tokens_pad], _pos
                    )
                    forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE
                    forward_context.batch_descriptor = None
                    if self.use_aux_hidden_state_outputs:
                        hidden_states, self._aux_hidden_states = model_output
                    else:
                        hidden_states = model_output
                        self._aux_hidden_states = None
                    # DSpark: forward hooks wrote per-layer aux hidden during the
                    # forward; assemble them. Spec keeps the padded [0:Σ] layout
                    # (postprocess/draft re-gather to bs via next_token_locs);
                    # non-spec slices to the real num_tokens so pad rows never leak
                    # into sampled_token_ids -> prev_token_ids -> next-step shape
                    # mismatch.
                    _is_spec = hasattr(self, "drafter")
                    _slice_len = num_tokens_pad if _is_spec else num_tokens
                    self._collect_dspark_aux(_slice_len)
                    hidden_states = hidden_states[:_slice_len]
                    logits = self.model.compute_logits(hidden_states)
                    return logits, hidden_states

                graph_key = (graph_bs, max_q_len)
                self.graphs[graph_key].replay()
                hidden_states = self.forward_vars["outputs"][:num_tokens]
                if graph_key in self.graph_aux_hidden:
                    self._aux_hidden_states = [
                        aux[:num_tokens] for aux in self.graph_aux_hidden[graph_key]
                    ]
                else:
                    self._aux_hidden_states = None
                # DSpark: hooks write aux hidden into fixed preallocated buffers
                # in-place (cudagraph-safe); slice to this step's token count.
                self._collect_dspark_aux(num_tokens)
                if self.logits_in_graph:
                    logits = self.graph_logits[graph_key][:num_tokens]
                else:
                    logits = self.model.compute_logits(hidden_states)

        return logits, hidden_states

    def postprocess(
        self,
        batch: ScheduledBatch,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor | None,
        top_ps: torch.Tensor | None,
        all_greedy: bool,
        # following for draft
        hidden_states: torch.Tensor,
        needs_independent_noise: bool = False,
    ) -> ScheduledBatchOutput:
        spec_decode_metadata = get_forward_context().spec_decode_metadata
        bs = batch.total_seqs_num
        if spec_decode_metadata is None:
            sampled_tokens = self.sampler(
                logits,
                temperatures,
                top_ks,
                top_ps,
                all_greedy,
                needs_independent_noise=needs_independent_noise,
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
                needs_independent_noise=needs_independent_noise,
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
            num_reject_tokens = self.drafter.mtp_k - num_bonus_tokens
            next_token_locs = num_bonus_tokens

        if get_tp_group().world_size > 1 and self.tokenID_processor.is_deferred_out:
            sampled_tokens = get_tp_group().broadcast(sampled_tokens, src=0)

        # Compute logprobs if any sequence requested them
        need_logprobs = any(batch.return_logprobs)
        sampled_logprobs = None
        if need_logprobs:
            logits_fp32 = logits.float()
            log_probs = torch.log_softmax(logits_fp32, dim=-1)
            sampled_logprobs = log_probs.gather(
                -1, sampled_tokens.to(torch.long).unsqueeze(-1)
            ).squeeze(-1)
            if get_tp_group().world_size > 1 and self.tokenID_processor.is_deferred_out:
                sampled_logprobs = get_tp_group().broadcast(sampled_logprobs, src=0)

        self.forward_done_event.record()
        # Capture before prepare_sampled_ids(), which advances self.prev_batch to current batch.
        prev_batch = self.tokenID_processor.prev_batch
        token_id_dict, logprobs_map = self.tokenID_processor.prepare_sampled_ids(
            batch, sampled_tokens, self.forward_done_event, sampled_logprobs
        )
        # Extract req_ids and token_ids from dict (key -1 is the is_deferred_out flag)
        req_ids_out = [k for k in token_id_dict if k != -1]
        token_ids_out = [token_id_dict[k] for k in req_ids_out]

        draft_token_ids: Optional[np.ndarray] = None
        if self.tokenID_processor.is_deferred_out:
            if hasattr(self, "drafter"):
                prev_rejected_num = self.tokenID_processor.prev_rejected_num
                prev_bonus_num = self.tokenID_processor.prev_bonus_num
                self.tokenID_processor.send_mtp_status_to_cpu_async(
                    num_reject_tokens, next_token_locs, self.forward_done_event
                )  # Async copy to CPU
                next_token_ids = torch.gather(
                    sampled_tokens.view(bs, -1), 1, next_token_locs.view(-1, 1)
                ).view(bs)
                self.tokenID_processor.prev_token_ids = next_token_ids
                # self.debug(f"{sampled_tokens=}")
                # self.debug(f"{next_token_locs=}")
                draft_token_ids = self.propose_draft_token_ids(
                    batch,
                    self.tokenID_processor.input_ids.gpu[
                        1 : batch.total_tokens_num + 1
                    ],
                    hidden_states,
                    next_token_ids,
                    num_reject_tokens,
                )
                # self.debug(f"{num_bonus_tokens=}")

            elif prev_batch is not None:
                prev_rejected_num = np.zeros(prev_batch.total_seqs_num, dtype=np.int32)
                prev_bonus_num = np.zeros(prev_batch.total_seqs_num, dtype=np.int32)
            else:
                # First forward pass: no deferred output yet, req_ids_out is empty
                prev_rejected_num = np.zeros(0, dtype=np.int32)
                prev_bonus_num = np.zeros(0, dtype=np.int32)
        else:
            prev_rejected_num = np.zeros(batch.total_seqs_num, dtype=np.int32)
            prev_bonus_num = np.zeros(batch.total_seqs_num, dtype=np.int32)

        # DSpark Phase 2: carry this step's per-request ell back to the scheduler
        # as a {req_id: ell} dict (req_id-keyed avoids any output/draft batch
        # ordering ambiguity). The worker already fired this map in propose() via
        # verify_scheduler.record_ell(batch.req_ids).
        dspark_ell = None
        drafter = getattr(self, "drafter", None)
        verify_scheduler = getattr(drafter, "verify_scheduler", None)
        if verify_scheduler is not None:
            dspark_ell = verify_scheduler.ell_nonblocking()

        return ScheduledBatchOutput(
            req_ids=req_ids_out,
            token_ids=token_ids_out,
            draft_token_ids=draft_token_ids,
            is_deferred_out=self.tokenID_processor.is_deferred_out,
            num_rejected=prev_rejected_num,
            num_bonus=prev_bonus_num,
            logprobs=logprobs_map,
            dspark_ell=dspark_ell,
        )

    @torch.inference_mode()
    @with_eplb_forward_monitor
    def forward(self, batch: ScheduledBatch) -> ScheduledBatchOutput:
        (
            input_ids,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            needs_independent_noise,
        ) = self.prepare_model(batch)
        logits, hidden_states = self.run_model(input_ids, batch)
        fwd_output = self.postprocess(
            batch,
            logits,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            hidden_states,
            needs_independent_noise=needs_independent_noise,
        )
        reset_forward_context()

        return fwd_output

    @torch.inference_mode()
    def process_kvconnector_output(self, connector_meta_output):
        """Dispatch KV connector metadata to initiate async KV loading."""
        if connector_meta_output is not None:
            connector = get_kvconnector()
            if connector is not None:
                connector.start_load_kv(connector_meta_output)

    @torch.inference_mode()
    def async_proc_aggregation(self) -> KVConnectorOutput:
        """Collect finished send/recv status from the KV connector."""
        connector = get_kvconnector()
        if connector is None:
            return KVConnectorOutput()

        finished = connector.get_finished()
        # New connectors may return the full KVConnectorOutput so they can
        # report richer states. LMCache offload uses failed_recving to wake a
        # request for local recompute, and finished_saving to release blocks
        # whose free was deferred while a background save read their KV.
        if isinstance(finished, KVConnectorOutput):
            return finished

        # Legacy P/D connectors still return the old
        # (done_sending, done_recving) tuple. Normalize it so EngineCore and
        # Scheduler only need to consume KVConnectorOutput.
        done_sending, done_recving = finished

        return KVConnectorOutput(
            finished_sending=done_sending, finished_recving=done_recving
        )

    def propose_draft_token_ids(
        self,
        batch: ScheduledBatch,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        num_reject_tokens: torch.Tensor,
    ):
        forward_context = get_forward_context()

        positions = forward_context.context.positions
        # Anchor (last verified target token) flat index = segment_start +
        # num_bonus. prepare_inputs counts back from each segment's END
        # (cu_seqlens_q[1:]), so offset = full_q - num_bonus = 1 + num_reject.
        last_token_offset = 1 + num_reject_tokens

        # DSpark q-shrink: segments are length q<full_q but the end-relative
        # offset is measured against full_q, over-counting by (full_q-q) -> OOB.
        # Subtract the shrink. No-op when q==full_q or on prefill/mixed steps.
        ragged_lens = getattr(batch, "dynamic_spec_query_tokens_per_req", None)
        if ragged_lens is not None and batch.total_tokens_num_prefill == 0:
            # RAGGED: each seg has its own len_i; anchor offset = len_i - num_bonus_i
            # (num_bonus_i = mtp_k - num_reject_i), applied to cu_seqlens_q ends.
            sbs = batch.total_seqs_num_decode
            lens_t = torch.as_tensor(
                np.asarray(ragged_lens)[:sbs],
                device=num_reject_tokens.device,
                dtype=num_reject_tokens.dtype,
            )
            num_bonus = self.drafter.mtp_k - num_reject_tokens[:sbs]
            last_token_offset = lens_t - num_bonus
        elif (
            hasattr(self, "drafter")
            and getattr(self.drafter, "dspark_confidence_schedule", False)
            and batch.total_tokens_num_prefill == 0
        ):
            full_q = self.drafter.mtp_k + 1
            q_actual = batch.num_spec_query_tokens
            if 1 <= q_actual < full_q:
                last_token_offset = last_token_offset - (full_q - q_actual)

        assert isinstance(self.drafter, EagleProposer)

        last_token_indices = self.drafter.prepare_inputs(
            batch.total_seqs_num, last_token_offset
        )

        draft_token = self.drafter.propose(
            target_token_ids=input_ids,
            target_positions=positions,
            target_hidden_states=hidden_states,
            num_reject_tokens=num_reject_tokens,
            next_token_ids=next_token_ids,
            last_token_indices=last_token_indices,
            aux_hidden_states=self._aux_hidden_states,
        )
        # DSpark Phase 2: stash this step's scheduler-chosen ell keyed by req_id,
        # so next step's calc_spec_decode_metadata can re-map it onto the (possibly
        # reordered) batch. Keying by req_id (not batch position) is required:
        # continuous batching reorders requests between steps.
        verify_scheduler = getattr(self.drafter, "verify_scheduler", None)
        if verify_scheduler is not None:
            verify_scheduler.record_ell(batch.req_ids[: batch.total_seqs_num])
        return self.tokenID_processor.prepare_draft_ids(batch, draft_token)

    def start_capture_profiler(self):
        """Set up the per-bs CUDA graph capture profiler (profiles in place).

        Profiles the capture phase as graphs are captured and writes one trace
        per batch size, per rank (``bs_<bs>_rank<rank>.json.gz``). Enabled on
        every rank when a torch profiler dir is set and mark-trace is on.
        """
        self._capture_profile_enabled = (
            self.profiler_dir is not None and self.mark_trace
        )
        if self._capture_profile_enabled:
            self._profile_bs_idx = 0
            self.capture_traces_dir = os.path.join(self.profiler_dir, "capture_traces")
            os.makedirs(self.capture_traces_dir, exist_ok=True)
            logger.info(f"{self.label}: Starting CUDA graph capture profiler...")

            def on_trace_ready(prof):
                # Invariant: exactly two prof.step() calls happen per captured
                # batch size (schedule wait=1 + active=1, repeat=0), so
                # on_trace_ready fires once per bs, in self.graph_bs order.
                # This is a profiling-only diagnostic; log-and-skip rather than
                # assert so a cadence mismatch can never abort CUDA-graph
                # capture at server startup (and isn't stripped under python -O).
                if self._profile_bs_idx >= len(self.graph_bs):
                    logger.warning(
                        "capture profiler fired %d times but only %d batch "
                        "sizes were captured; skipping extra trace. Check the "
                        "prof.step() cadence in capture_cudagraph.",
                        self._profile_bs_idx + 1,
                        len(self.graph_bs),
                    )
                    return
                bs = self.graph_bs[self._profile_bs_idx]
                trace_file = os.path.join(
                    self.capture_traces_dir, f"bs_{bs}_rank{self.rank}.json.gz"
                )
                prof.export_chrome_trace(trace_file)
                logger.info(f"Saved trace for bs={bs} to {trace_file}")
                self._profile_bs_idx += 1

            self.capture_profiler = torch_profiler.profile(
                activities=[
                    torch_profiler.ProfilerActivity.CUDA,
                    torch_profiler.ProfilerActivity.CPU,
                ],
                schedule=torch_profiler.schedule(wait=1, warmup=0, active=1, repeat=0),
                record_shapes=True,
                with_stack=True,
                profile_memory=False,
                on_trace_ready=on_trace_ready,
            )
        else:
            self.capture_profiler = nullcontext()

    @torch.inference_mode()
    def _piecewise_cg_active(self) -> bool:
        """True when the compiled model's dense pieces self-capture PIECEWISE
        cudagraphs (attention eager between them). In that mode the runner does
        NOT build the manual FULL whole-forward graphs — decode calls the model
        directly and the per-piece CUDAGraphWrapper handles capture/replay."""
        if self.enforce_eager:
            return False
        # Driven by --cudagraph-mode (default FULL -> manual capture, unchanged).
        # PIECEWISE / FULL_AND_PIECEWISE -> per-piece cudagraph path.
        mode = getattr(self.config.compilation_config, "cudagraph_mode", None)
        return mode is not None and mode.requires_piecewise_compilation()

    def _force_aiter_unreg_capture_for_piecewise(self):
        """PIECEWISE cudagraph + aiter custom all_gather/reduce_scatter: force the
        copy-in ('unreg') capture path instead of the direct-read ('registered')
        one.

        The registered path lets the collective kernel directly read each peer's
        ORIGINAL input pointer (cross-registered at register_graph_buffers). That
        is only safe under a single whole-forward FULL cudagraph, whose global
        read/overwrite ordering holds across all ranks. PIECEWISE splits the
        forward into many small graphs with eager sections between them, losing
        that ordering: a fast rank can overwrite its pool-recycled input via a
        later piece while a slow peer is still reading it -> stale cross-rank
        reads -> progressive hidden corruption -> repeated-token garbage
        (DP+PIECEWISE accuracy bug). The unreg path snapshots the input into a
        pre-registered pool before the collective, so it is order-independent.
        """
        seen = set()
        for getter in ("get_tp_group", "get_dp_group", "get_ep_group"):
            try:
                from aiter.dist import parallel_state as _ps

                group = getattr(_ps, getter)()
            except Exception:
                continue
            dc = getattr(group, "device_communicator", None)
            ca = getattr(dc, "ca_comm", None) if dc is not None else None
            if ca is None or id(ca) in seen:
                continue
            seen.add(id(ca))
            if getattr(ca, "enable_register_for_capturing", False):
                ca.enable_register_for_capturing = False
                logger.info(
                    "PIECEWISE: forced aiter ca_comm (%s) to unreg copy-in "
                    "capture path for cudagraph-safe DP collectives.",
                    getter,
                )

    def _piecewise_replay_shape(self, batch, graph_bs, max_q_len):
        """Pick the PIECEWISE replay token count for one decode step.

        Returns ``(num_tokens_pad, real_tokens, captured)``:
        - ``num_tokens_pad``: token count to forward (the captured bucket size).
        - ``real_tokens``: real tokens present in ``[0:real_tokens]``.
        - ``captured``: whether a matching cudagraph bucket exists (else eager).

        DSpark (has drafter, TP-only) replays at a flat num_tokens bucket sized
        to the REAL ragged token total (= total_tokens_num_decode) so MoE/linear
        shrink with it (dynamic verify length); attention is eager on the flat
        [0:total] tokens, the [total:pad] tail is masked. Non-spec (or DP) uses
        the rectangular bucket num_tokens == bs.
        """
        is_dummy = batch is not None and batch.is_dummy_run
        use_ragged_bucket = (
            batch is not None
            and not is_dummy
            and self._piecewise_sorted_tokens
            and hasattr(self, "drafter")
        )
        if use_ragged_bucket:
            dp_total_tokens = batch.dspark_dp_total_tokens
            real_tokens = (
                int(dp_total_tokens)
                if dp_total_tokens is not None
                else int(batch.total_tokens_num_decode)
            )
            buckets = self._piecewise_sorted_tokens
            idx = bisect.bisect_left(buckets, real_tokens)
            if idx < len(buckets):
                return buckets[idx], real_tokens, True
            # total tokens exceeds the largest captured bucket -> eager.
            return max(real_tokens, graph_bs * max_q_len), real_tokens, False

        num_tokens_pad = graph_bs * max_q_len
        captured = num_tokens_pad in self._piecewise_captured_tokens
        return num_tokens_pad, num_tokens_pad, captured

    def _dspark_ragged_moe_graph_bs(self, batch, default_graph_bs):
        """MoE all_gather pad row count for a DSpark ragged PIECEWISE decode step.

        ``context.graph_bs`` is what MoE's ``pad_for_all_gather`` pads
        hidden_states to before the cross-DP all_gather, so every DP rank must
        agree on it. But DSpark ragged does NOT replay at a rectangular bs*q
        grid: it replays at the flat num_tokens bucket ``_piecewise_replay_shape``
        picks from the DP-max total token count. Derive graph_bs from that SAME
        bucket (bucket // q) so the padded row count matches the tokens actually
        forwarded. Falls back to ``default_graph_bs`` when this isn't a DSpark
        ragged step or the bucket isn't an exact multiple of q.
        """
        dp_total_tokens = batch.dspark_dp_total_tokens
        if (
            not self._piecewise_cg_active()
            or dp_total_tokens is None
            or not self._piecewise_sorted_tokens
        ):
            return default_graph_bs
        q = int(batch.num_spec_query_tokens)
        buckets = self._piecewise_sorted_tokens
        idx = bisect.bisect_left(buckets, int(dp_total_tokens))
        if idx < len(buckets) and q > 0 and buckets[idx] % q == 0:
            return buckets[idx] // q
        return default_graph_bs

    def _dspark_capture_q_buckets(self, full_q: int) -> list[int]:
        """DSpark query-length buckets to capture graphs for (paper Phase 2).

        Confidence scheduling replays a SMALLER max_q_len than full_q, so we
        capture one rectangular graph set per bucket. RAGGED and the older
        q-bucket path use independent size sets. Defaults to ``[full_q]`` (the
        Phase-1 single-graph behavior) when confidence scheduling is off.
        """
        if not (
            hasattr(self, "drafter")
            and getattr(self.drafter, "dspark_confidence_schedule", False)
        ):
            return [full_q]
        from atom.spec_decode.dspark_scheduler import resolve_q_buckets

        dspark = self.config.dspark
        sizes = dspark.ragged_graph_sizes if dspark.ragged else dspark.q_buckets
        return resolve_q_buckets(sizes, full_q)

    def _piecewise_per_token_bytes(self) -> float:
        """Estimated GPU bytes a captured PIECEWISE graph retains per token.

        Derived from model geometry (hidden * dtype * layers * live-tensors/layer)
        so it holds for ANY model, not a magic per-token constant. Under DP the
        MoE all_gathers hidden to ~dp_size x local tokens, so each piece retains
        far more per local token than TP (measured DSV4: TP 2.32MB/tok vs DP
        7.7MB/tok, ~3.3x at dp=8). Attention doesn't amplify, so scale by a
        sub-linear dp**0.6 (8**0.6=3.48, just above the measured 3.3).
        """
        hf = self.config.hf_config
        dtype_bytes = torch.finfo(self.config.torch_dtype).bits // 8
        _LIVE_TENSORS_PER_LAYER = 2.8
        per_token = (
            int(hf.hidden_size)
            * dtype_bytes
            * int(hf.num_hidden_layers)
            * _LIVE_TENSORS_PER_LAYER
        )
        dp_size = self.config.parallel_config.data_parallel_size
        if dp_size > 1:
            per_token *= float(dp_size) ** 0.6
        return per_token

    def _piecewise_skip_capture(self, num_tokens: int) -> bool:
        """Whether to skip capturing a PIECEWISE bucket of ``num_tokens`` tokens.

        Two guards, both DP-safe (the decision must be identical on every rank,
        else capture loops desync and the next get_dp_padding all_reduce couples
        mismatched num_tokens -> "batch_id_per_token len < T"):

        1. DP+spec hard cap: big bs*q buckets never run under DP but bloat the
           pool and don't overlap comm, so cap at ATOM_PIECEWISE_DP_MAX_TOKENS.
        2. Memory guard: skip a bucket whose estimated capture footprint won't
           fit in free GPU memory (adapts to GPU size / config, no hardcoded
           cap). DP amplifies the retained per-token footprint (MoE all_gather
           ~dp_size x tokens); scale the slope by dp**0.6 to match
           _estimate_cudagraph_overhead. Free is min-reduced across DP so all
           ranks skip the same set.
        """
        dp_size = self.config.parallel_config.data_parallel_size
        if dp_size > 1 and hasattr(self, "drafter"):
            dp_cap = int(os.environ.get("ATOM_PIECEWISE_DP_MAX_TOKENS", "512"))
            if num_tokens > dp_cap:
                if self.rank == 0:
                    logger.info(
                        "PIECEWISE DP-cap skip num_tokens=%d "
                        "(> %d = ATOM_PIECEWISE_DP_MAX_TOKENS)",
                        num_tokens,
                        dp_cap,
                    )
                return True

        # Memory guard slope: capture footprint grows ~linearly with hidden size.
        # Empirically ~600 bytes/token per hidden-dim (0.004GB/token measured at
        # hidden=7168 -> 0.004*2**30/7168 = 599B, rounded).
        _GUARD_BYTES_PER_TOKEN_PER_DIM = 600
        slope = _GUARD_BYTES_PER_TOKEN_PER_DIM * self.config.hf_config.hidden_size
        if dp_size > 1:
            slope *= float(dp_size) ** 0.6
        free = torch.cuda.mem_get_info()[0]
        if dp_size > 1:
            import torch.distributed as dist
            from aiter.dist.parallel_state import get_dp_group

            free_t = torch.tensor([free], device="cpu", dtype=torch.int64)
            dist.all_reduce(
                free_t, op=dist.ReduceOp.MIN, group=get_dp_group().cpu_group
            )
            free = int(free_t.item())
        need = slope * num_tokens * 1.25 + (4 << 30)
        if (free >> 30) < (int(need) >> 30):
            if self.rank == 0:
                logger.info(
                    "PIECEWISE skip num_tokens=%d: free=%.1fGB < need=%.1fGB",
                    num_tokens,
                    free / 1e9,
                    need / 1e9,
                )
            return True
        return False

    def capture_cudagraph(self):
        _piecewise = self._piecewise_cg_active()
        if _piecewise:
            logger.info(
                "PIECEWISE cudagraph: capturing per-piece graphs (attention "
                "eager); manual FULL whole-forward capture disabled."
            )
            self._force_aiter_unreg_capture_for_piecewise()
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

        # Drop any capture size that exceeds max_num_seqs — those graphs would
        # never be replayed since the scheduler can't produce a batch larger
        # than max_num_seqs. Warn so the user notices a misconfig (default
        # cuda_graph_sizes=[512] vs e.g. max_num_seqs=16) without crashing.
        max_bs = self.config.max_num_seqs
        oversized = [s for s in self.graph_bs if s > max_bs]
        if oversized:
            self.graph_bs = [s for s in self.graph_bs if s <= max_bs]
            logger.warning(
                "cudagraph capture sizes %s exceed max_num_seqs=%d; dropping. "
                "Remaining: %s",
                oversized,
                max_bs,
                self.graph_bs,
            )
        assert self.graph_bs, (
            f"no cudagraph capture sizes left after filtering by "
            f"max_num_seqs={max_bs}; pass --cudagraph-capture-sizes or raise "
            f"--max-num-seqs."
        )

        # PIECEWISE: the set of num_tokens shapes whose dense pieces we captured
        # (reset here; initialized empty in __init__). run_model dispatches by
        # num_tokens; a shape NOT in here would force a runtime (uncoordinated)
        # capture that hangs on collectives, so run_model falls back to eager for
        # uncaptured shapes.
        self._piecewise_captured_tokens = set()

        self.forward_vars["kv_indptr"].gpu.zero_()
        if self.is_deepseek_v32 and "sparse_kv_indptr" in self.forward_vars:
            self.forward_vars["sparse_kv_indptr"].gpu.zero_()

        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = dict()
        self.graph_logits: dict[tuple[int, int], torch.Tensor] = dict()
        self.graph_aux_hidden: dict[tuple[int, int], list[torch.Tensor]] = dict()
        self.graph_pool = None
        is_tbo = self.config.enable_tbo and isinstance(self.model, UBatchWrapper)
        # TBO graphs don't capture compute_logits, so disable logits_in_graph.
        self.logits_in_graph = self.world_size == 1 and not is_tbo

        # start capture profiler
        self.start_capture_profiler()

        @contextmanager
        def pause_gc():
            # No GC during capture: a finalizer's hipModuleUnload aborts it (HIP 900).
            gc.collect()
            gc.disable()
            try:
                yield
            finally:
                gc.enable()
                gc.collect()

        _rsv_before_capture = torch.cuda.memory_reserved()
        _alloc_before_capture = torch.cuda.memory_allocated()

        input_ids = self.forward_vars["input_ids"].gpu
        positions = self.forward_vars["positions"].gpu
        outputs = self.forward_vars["outputs"]

        full_q_len = self.drafter.mtp_k + 1 if hasattr(self, "drafter") else 1
        # Capture one graph per (bs, query-length bucket). Buckets default to
        # [full_q_len] (single-graph, classic per-bs capture); DSpark confidence
        # scheduling expands to the smaller q-buckets a decode step may replay.
        q_buckets = self._dspark_capture_q_buckets(full_q_len)
        if q_buckets != [full_q_len]:
            logger.info("DSpark CUDA-graph query buckets: %s", q_buckets)

        # Whether this backend's capture builder supports a dynamic (per-bucket)
        build_capture = self.attn_metadata_builder.build_for_cudagraph_capture
        supports_dynamic_q_len = (
            "max_q_len" in inspect.signature(build_capture).parameters
        )

        with pause_gc(), graph_capture() as capture_ctx, self.capture_profiler as prof:
            for max_q_len in q_buckets:
                capture_range = (
                    tqdm.tqdm(self.graph_bs) if self.rank == 0 else self.graph_bs
                )
                for bs in capture_range:
                    if self.rank == 0:
                        capture_range.set_description(f"Capturing {bs=}, {max_q_len=}")

                    cu_seqlens_q = np.arange(
                        0, (bs + 1) * max_q_len, max_q_len, dtype=np.int32
                    )
                    self.forward_vars["cu_seqlens_q"].np[: bs + 1] = cu_seqlens_q
                    self.forward_vars["cu_seqlens_q"].copy_to_gpu(bs + 1)

                    num_tokens = bs * max_q_len
                    if _piecewise and self._piecewise_skip_capture(num_tokens):
                        continue
                    # Use a simple, safe position pattern for capture.
                    self.forward_vars["positions"].np[:num_tokens] = (
                        np.arange(num_tokens, dtype=np.int64) % max_q_len
                    )
                    if supports_dynamic_q_len:
                        attn_metadata, context = build_capture(
                            bs=bs, max_q_len=max_q_len
                        )
                    else:
                        attn_metadata, context = build_capture(bs=bs)
                    if self.use_mrope:
                        mrope_positions = self._mrope_positions_view(num_tokens)
                        mrope_positions.copy_(
                            positions[:num_tokens].unsqueeze(0).expand(3, -1)
                        )
                        context.positions = mrope_positions
                    num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
                    num_tokens += num_pad
                    # get_dp_padding built num_tokens_across_dp from the PRE-pad
                    # count, but we just padded num_tokens. Capture is symmetric
                    # (every DP rank captures the same bs), so the padded count is
                    # uniform across ranks. Rebuild the tensor at the padded size so
                    # DPMetadata.make's `across_dp[rank] == num_tokens` holds.
                    if num_tokens_across_dp is not None:
                        num_tokens_across_dp = torch.full_like(
                            num_tokens_across_dp, num_tokens
                        )
                    # Create ubatch slices for TBO capture (need > 2 requests)
                    ubatch_slices = None
                    if is_tbo and self.config.enable_tbo_decode and bs > 2:
                        ubatch_slices = maybe_create_ubatch_slices(
                            num_reqs=bs,
                            num_tokens=num_tokens,
                        )

                    set_forward_context(
                        attn_metadata=attn_metadata,
                        atom_config=self.config,
                        context=context,
                        num_tokens=num_tokens,
                        num_tokens_across_dp=num_tokens_across_dp,
                        ubatch_slices=ubatch_slices,
                        in_hipgraph=True,
                    )

                    # Warmup
                    model_positions = (
                        self._mrope_positions_view(num_tokens)
                        if self.use_mrope
                        else positions[:num_tokens]
                    )
                    model_output = self.model(input_ids[:num_tokens], model_positions)
                    if self.use_aux_hidden_state_outputs:
                        outputs[:num_tokens] = model_output[0]
                    else:
                        outputs[:num_tokens] = model_output
                    if self.logits_in_graph:
                        self.model.compute_logits(outputs[:num_tokens])
                    if prof is not None:
                        prof.step()

                    if _piecewise:
                        # PIECEWISE: no manual whole-forward graph; the compiled
                        # per-piece wrappers self-capture. Replay once to register.
                        fc = get_forward_context()
                        fc.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
                        fc.batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
                        self.model(input_ids[:num_tokens], model_positions)
                        fc.cudagraph_runtime_mode = CUDAGraphMode.NONE
                        fc.batch_descriptor = None
                        self._piecewise_captured_tokens.add(num_tokens)
                        continue

                    # Capture
                    with (
                        record_function(f"capture_graph_bs_{bs}")
                        if self.mark_trace
                        else nullcontext()
                    ):
                        if ubatch_slices is not None:
                            # TBO capture: threads + multi-stream captured in graph.
                            graph, graph_output = self.model.capture_tbo_graph(
                                input_ids[:num_tokens],
                                positions[:num_tokens],
                                self.graph_pool,
                                capture_ctx.stream,
                                output_buffer=outputs[:num_tokens],
                            )
                            graph_aux = (
                                graph_output[1]
                                if self.use_aux_hidden_state_outputs
                                else None
                            )
                        else:
                            # Standard single-stream capture
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(
                                graph, self.graph_pool, stream=capture_ctx.stream
                            ):
                                model_output = self.model(
                                    input_ids[:num_tokens], model_positions
                                )
                                if self.use_aux_hidden_state_outputs:
                                    outputs[:num_tokens] = model_output[0]
                                    graph_aux = model_output[1]
                                else:
                                    outputs[:num_tokens] = model_output
                                    graph_aux = None
                                if self.logits_in_graph:
                                    graph_logits = self.model.compute_logits(
                                        outputs[:num_tokens]
                                    )
                    if self.graph_pool is None:
                        self.graph_pool = graph.pool()
                    self.graphs[(bs, max_q_len)] = graph
                    if self.logits_in_graph and ubatch_slices is None:
                        self.graph_logits[(bs, max_q_len)] = graph_logits
                    if prof is not None:
                        prof.step()
                    if graph_aux is not None:
                        self.graph_aux_hidden[(bs, max_q_len)] = graph_aux
                    torch.cuda.synchronize()
        self.graph_bs.sort(reverse=False)

        # PIECEWISE: sorted 1D num_tokens buckets for run_model's round_up_1d(Σ)
        # dispatch (bisect_left over this to pick the tightest captured shape).
        self._piecewise_sorted_tokens = sorted(self._piecewise_captured_tokens)
        if _piecewise and self.rank == 0:
            logger.info(
                "PIECEWISE captured %d num_tokens buckets: %s",
                len(self._piecewise_sorted_tokens),
                self._piecewise_sorted_tokens,
            )

        # DSpark Phase 2: calibrate the SPS(B) throughput profile from the just-
        # captured target graphs (each is a B = bs*max_q_len token forward, i.e.
        # exactly one verification step at batch B). Cheap, GPU-only, one-shot.
        self._maybe_calibrate_dspark_sps(full_q_len)

        # How much GPU memory the CUDA graph capture consumed (pool = reserved
        # delta; the allocated delta is what the graphs pin live).
        _pool_bytes = max(torch.cuda.memory_reserved() - _rsv_before_capture, 0)
        _alloc_bytes = max(torch.cuda.memory_allocated() - _alloc_before_capture, 0)
        if self.rank == 0:
            logger.info(
                "CUDA graph capture memory: %d graphs | pool(reserved)=%.2fGB "
                "allocated=%.2fGB",
                len(self.graphs) + len(self._piecewise_captured_tokens),
                _pool_bytes / (1 << 30),
                _alloc_bytes / (1 << 30),
            )

        # Post-init memory validation
        free_after, total_after = torch.cuda.mem_get_info()
        actual_usage = total_after - free_after
        target_usage = int(total_after * self.config.gpu_memory_utilization)
        usage_ratio = actual_usage / total_after
        logger.info(
            f"Post-init memory: "
            f"actual={actual_usage / (1 << 30):.2f}GB ({usage_ratio:.1%}), "
            f"target={target_usage / (1 << 30):.2f}GB "
            f"({self.config.gpu_memory_utilization:.0%}), "
            f"reserved={torch.cuda.memory_reserved() / (1 << 30):.2f}GB, "
            f"allocated={torch.cuda.memory_allocated() / (1 << 30):.2f}GB"
        )
        if usage_ratio > self.config.gpu_memory_utilization + 0.02:
            logger.warning(
                f"Actual GPU memory usage ({usage_ratio:.1%}) exceeds target "
                f"({self.config.gpu_memory_utilization:.0%}) by "
                f"{(usage_ratio - self.config.gpu_memory_utilization):.1%}. "
                f"Consider reducing gpu_memory_utilization."
            )

        return time.time() - start_time, self.graph_bs, _pool_bytes

    @torch.inference_mode()
    def _maybe_calibrate_dspark_sps(self, max_q_len: int, n_iters: int = 20) -> None:
        """Profile SPS(B) by timing the captured target graphs, then hand a dense
        cost table to the DSpark drafter (paper §3.2.2, scheduler input).

        Each captured graph ``self.graphs[(bs, max_q_len)]`` is a forward over
        ``B = bs * max_q_len`` tokens — exactly one verification step at batch B.
        We replay each a few times, take the median step time, and densify the
        (B, steps/sec) samples into ``sps_table[B]``. No-op unless a DSpark
        drafter with confidence scheduling enabled is present.
        """
        drafter = getattr(self, "drafter", None)
        if drafter is None or not getattr(drafter, "use_dspark", False):
            return
        verify_scheduler = getattr(drafter, "verify_scheduler", None)
        if verify_scheduler is None:
            return
        if not getattr(self, "graphs", None):
            return
        if self.config.dspark.disable_sps_calib:
            logger.info("DSpark SPS calibration disabled; using synthetic stub.")
            return

        from atom.spec_decode.dspark_scheduler import build_sps_table

        # DSpark RAGGED graph: replay-based SPS calibration is UNSAFE here. Each
        # `graph.replay()` runs the FULL decode graph (incl. SWA/KV writes) with
        # synthetic data at real cache slots [0:bs], polluting the KV cache real
        # requests then read. The scheduler only needs a monotone SPS(B) shape,
        # so use a synthetic table instead (matches the proven DISABLE_SPS_CALIB
        # path). Timed ragged calibration is a follow-up (needs a scratch KV pool
        # + buffer save/restore around the replays).
        if self.config.dspark.ragged:
            logger.info(
                "DSpark SPS calibration skipped under RAGGED graph "
                "(replay would pollute KV cache); using synthetic stub."
            )
            return

        token_points: list[int] = []
        sps_points: list[float] = []
        for bs in self.graph_bs:
            graph = self.graphs.get((bs, max_q_len))
            if graph is None:
                continue
            B = bs * max_q_len
            # Warm replay, then timed replays (median for robustness to jitter).
            graph.replay()
            torch.cuda.synchronize()
            times_ms: list[float] = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(n_iters):
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times_ms.append(start.elapsed_time(end))
            times_ms.sort()
            median_ms = times_ms[len(times_ms) // 2]
            if median_ms <= 0:
                continue
            token_points.append(B)
            sps_points.append(1000.0 / median_ms)  # steps per second

        if not token_points:
            logger.warning("DSpark SPS calibration found no timeable graphs.")
            return

        max_b = self.config.max_num_seqs * max_q_len
        sps_table = build_sps_table(token_points, sps_points, max_b).to(self.device)
        verify_scheduler.sps_table = sps_table
        logger.info(
            "DSpark SPS calibrated over %d points (B=%d..%d), table size %d.",
            len(token_points),
            token_points[0],
            token_points[-1],
            sps_table.numel(),
        )
