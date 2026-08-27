# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import itertools
import logging
import time
from collections import Counter
from dataclasses import fields
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from atom.config import Config
from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager
from atom.model_engine.multimodal import get_mrope_input_positions
from atom.model_engine.sequence import Sequence
from atom.model_engine.stop_strings import (
    IncrementalDetokenizer,
    check_stop_strings,
)
from atom.sampling_params import SamplingParams
from atom.utils import envs

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
        data_parallel_master_port = kwargs.get("data_parallel_master_port", None)
        config = Config(model, **config_kwargs)
        self.config = config
        self.tokenizer = tokenizer or _load_tokenizer(
            config.model, config.trust_remote_code
        )
        config.bos_token_id = self.tokenizer.bos_token_id
        config.eos_token_id = self.tokenizer.eos_token_id
        stop_token_ids = set(config.stop_token_ids)
        # separate eos_token_id from stop_token_ids
        stop_token_ids.discard(config.eos_token_id)
        config.stop_token_ids = list(stop_token_ids)
        # Legacy path only: callers that pass DP topology as loose kwargs
        # instead of a ParallelConfig. When a parallel_config was supplied it
        # is already authoritative, and overwriting it here would reset a
        # multi-node topology back to a single local rank.
        if "parallel_config" not in config_kwargs:
            config.parallel_config.data_parallel_size = data_parallel_size
            if data_parallel_master_port is not None:
                config.parallel_config.data_parallel_master_port = (
                    data_parallel_master_port
                )
        self.data_parallel_size = config.parallel_config.data_parallel_size
        # PCP and DP-attention are not yet compatible: PCP stripe-splits
        # input_ids to 1/pcp_size in ForCausalLM.forward, but DP-attention's
        # `_gather_ids_for_dp` all-gathers using dp_metadata sizes computed on
        # the FULL (un-split) token count, so all_gatherv asserts
        # `1/pcp_size != full`.
        if config.enable_dp_attention and config.pipeline_parallel_size > 1:
            raise ValueError(
                "--enable-dp-attention and pipeline-parallel (pp>1) "
                "cannot be used together."
            )
        if config.prefill_context_parallel_size > 1 and config.enable_dp_attention:
            raise ValueError(
                "prefill_context_parallel_size > 1 (-pcp) combined with "
                "--enable-dp-attention is not supported yet (may be supported "
                "in a future release): PCP splits tokens to 1/pcp_size while "
                "DP-attention's id-gather expects the full token count, "
                "causing an all_gatherv size mismatch. For now, disable one of "
                "them (use -tp N -pcp M without DP-attention, or -dp N "
                "--enable-dp-attention without -pcp)."
            )
        # PCP + TBO prefill: supported via coordinated splitting.
        # PCP + TBO decode: not yet supported (pcp_all_reduce semantics under
        # per-request ubatch split are unverified).
        if config.prefill_context_parallel_size > 1 and config.enable_tbo:
            # TBO overlaps compute with the attn<->MoE PCP collectives, which
            # only exist in MoE merge mode (ATOM_PCP_MOE_MERGE=1). With
            # ATOM_PCP_MOE_MERGE=0, MoE runs on each rank's 1/W token shard
            # with NO extra comm between attn and MoE, so TBO has nothing
            # to overlap and only adds ubatch-splitting overhead. Force it off.
            if not envs.ATOM_PCP_MOE_MERGE:
                logger.warning(
                    "Disabling TBO because ATOM_PCP_MOE_MERGE=0: in this "
                    "situation it runs MoE on each rank's 1/W token shard with "
                    "no extra attn<->MoE communication, so TBO has nothing to "
                    "overlap and only adds overhead."
                )
                config.enable_tbo = False
                config.enable_tbo_decode = False
            elif config.tensor_parallel_size > 1:
                # Cross-communicator (PCP x TP) RCCL deadlock:
                # under TBO the PCP collectives (comm_stream) run concurrently
                # and UNORDERED with the TP-group all_reduces (compute_stream,
                # from attention wo_b / MoE RowParallelLinear). On the TPxPCP
                # rank grid with large collectives this forms a cross-rank
                # circular wait -> hang (reproduced MI355 TP4PCP2/TP2PCP4 merge
                # +TBO, 64k/c32). Serialized (non-TBO single stream) is fine, and
                # TP=1 has no TP communicator so it cannot form the cycle. Only
                # TP=1 + PCP keeps TBO; TP>1 falls back to non-TBO.
                logger.warning(
                    "Disabling TBO: prefill_context_parallel_size > 1 (-pcp) "
                    "with tensor_parallel_size > 1 (-tp) hangs under TBO due to "
                    "a PCP<->TP cross-communicator RCCL deadlock (concurrent "
                    "unordered collectives on comm/compute streams; see "
                    "PCP_TBO.md 14.4). TBO with PCP is only supported at -tp 1 "
                    "(e.g. -tp 1 -pcp 8)."
                )
                config.enable_tbo = False
                config.enable_tbo_decode = False
            elif config.enable_tbo_decode:
                raise ValueError(
                    "prefill_context_parallel_size > 1 (-pcp) combined with "
                    "--enable-tbo all (decode TBO) is not supported yet. "
                    "Use --enable-tbo (prefill only) with -pcp."
                )
            # Under PCP, TBO prefill uses a request-boundary split (the
            # non-default TBO split mode; never token-midpoint split), so
            # ATOM_TBO_PREFILL_TOKEN_SPLIT is ignored.
            if config.enable_tbo and envs.ATOM_TBO_PREFILL_TOKEN_SPLIT:
                logger.warning(
                    "ATOM_TBO_PREFILL_TOKEN_SPLIT is ignored under PCP: TBO "
                    "prefill uses request-boundary balanced grouping."
                )
        self.io_processor = InputOutputProcessor(
            config, self.tokenizer, config.kv_cache_block_size
        )
        if config.enable_rapidserve:
            self.core_mgr = DisaggCoreManager(config)
        else:
            self.core_mgr = CoreManager(config)
        # A stop string is decided in the frontend, so the frontend is what
        # has to tell the engine core to stop working on that request.
        self.io_processor.abort_request = self.core_mgr.abort_request
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
        prompt_or_tokens_list: list[str | list[int]],
        sampling_params_list: SamplingParams | list[SamplingParams],
        stream_callback=None,
        multimodal_data_list: list[dict] | None = None,
        request_ids: list[str] | None = None,
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

        # Handle multimodal data
        if multimodal_data_list is not None:
            if len(prompt_or_tokens_list) != len(multimodal_data_list):
                raise ValueError(
                    f"number of elements in prompt_or_tokens_list and multimodal_data_list is different: "
                    f"{len(prompt_or_tokens_list)=} vs {len(multimodal_data_list)=}"
                )
            mm_data_iter = multimodal_data_list
        else:
            mm_data_iter = itertools.repeat(None)

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
        for prompt, sampling_param, callback, mm_data, request_id in zip(
            prompt_or_tokens_list,
            sampling_params_iter,
            stream_callback_iter,
            mm_data_iter,
            request_id_iter,
        ):
            req = self.io_processor.preprocess(
                prompt,
                sampling_param,
                stream_callback=callback,
                multimodal_data=mm_data,
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
        prompts: list[str],
        sampling_params: SamplingParams | list[SamplingParams],
        request_ids: list[str] | None = None,
    ) -> list[str]:
        # Reset DP routing state (round-robin cursor + in-flight load) so a
        # fresh batch gets deterministic DP assignment and no leaked counts.
        self.core_mgr.reset_dp_router()

        # A no-op callback, purely so `_wrap_for_stop_strings` has something
        # to wrap: without one the offline path would see no per-step output
        # and a stop string could only be applied after the request had
        # already run to EOS or `max_tokens`.
        self.add_request(
            prompts,
            sampling_params,
            stream_callback=lambda _request_output: None,
            request_ids=request_ids,
        )
        outputs = {}
        while not self.is_finished() and (
            self.core_mgr.is_alive() or self.core_mgr.is_rest()
        ):
            seqs = self.step()
            outs = self.io_processor.postprocess(seqs)
            outputs.update(outs)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        return outputs

    def generate_multimodal(
        self,
        token_ids_list: list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        multimodal_data_list: list[dict],
    ) -> list[dict]:
        """Generate completions for multimodal inputs (token IDs + vision data)."""
        self.core_mgr.reset_dp_router()
        self.add_request(
            token_ids_list,
            sampling_params,
            multimodal_data_list=multimodal_data_list,
        )
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
        self.core_mgr.broadcast_utility_command_sync("start_profile")
        logger.info("Profiling started")

    def stop_profile(self) -> list[dict[str, Any]]:
        responses = self.core_mgr.broadcast_utility_command_sync(
            "stop_profile", timeout=envs.ATOM_PROFILER_TIMEOUT
        )
        return [resp.get("result", {}) for resp in responses]

    def print_mtp_statistics(self):
        self.core_mgr.send_utility_command("get_mtp_stats")

    def get_mtp_statistics(self, timeout: float = 30.0) -> dict[str, Any]:
        """Return aggregated speculative decoding statistics across DP ranks."""
        responses = self.core_mgr.broadcast_utility_command_sync(
            "get_mtp_statistics", timeout=timeout
        )
        rank_stats = [
            resp.get("result", resp)
            for resp in responses
            if resp.get("result", resp).get("enabled", False)
        ]

        distribution: Counter[int] = Counter()
        for stats in rank_stats:
            distribution.update(
                {
                    int(accepted): int(steps)
                    for accepted, steps in stats.get("distribution", {}).items()
                }
            )

        total_draft_tokens = sum(
            int(stats.get("total_draft_tokens", 0)) for stats in rank_stats
        )
        total_accepted_tokens = sum(
            int(stats.get("total_accepted_tokens", 0)) for stats in rank_stats
        )
        total_steps = sum(distribution.values())

        return {
            "enabled": bool(rank_stats),
            "total_draft_tokens": total_draft_tokens,
            "total_accepted_tokens": total_accepted_tokens,
            "acceptance_rate": (
                total_accepted_tokens / total_draft_tokens
                if total_draft_tokens
                else 0.0
            ),
            "average_tokens_per_forward": (
                1 + total_accepted_tokens / total_steps if total_steps else 0.0
            ),
            "distribution": dict(sorted(distribution.items())),
            "distribution_percent": {
                k: v / total_steps if total_steps else 0.0
                for k, v in sorted(distribution.items())
            },
        }

    def get_cache_statistics(self, timeout: float = 30.0) -> dict[str, Any]:
        """Return aggregated prefix-cache statistics across DP ranks.

        The four rates are the ones `[Cache Stats]` logs, recomputed from
        summed counters rather than averaged: ranks admit different numbers of
        tokens, so the mean of their rates is not the rate of their union.

        `cached <= wanted <= compressed <= full` by construction, which is what
        makes the differences below meaningful:
          hit                 reuse actually admitted
          compressed_hit      reuse the prefix index held, before the
                              per-request state classes had their say
          lost_to_checkpoint  declined only because no checkpoint existed at
                              that boundary — what a denser ladder recovers
          lost_unrecoverable  declined for a reason no checkpoint touches
        """
        responses = self.core_mgr.broadcast_utility_command_sync(
            "get_cache_statistics", timeout=timeout
        )
        rank_stats = [
            resp.get("result", resp)
            for resp in responses
            if resp.get("result", resp).get("enabled", False)
        ]
        totals = {
            key: sum(int(stats.get(key, 0)) for stats in rank_stats)
            for key in (
                "requests",
                "cached_tokens",
                "compressed_tokens",
                "wanted_tokens",
                "full_tokens",
                "checkpoints_kept",
                "checkpoints_dropped",
                "checkpoints_evicted",
                "demands_recorded",
                "demands_declined_no_room",
                "chunks_cut_for_demand",
            )
        }
        full = totals["full_tokens"]

        def rate(num: int) -> float:
            return num / full if full else 0.0

        return {
            "enabled": bool(rank_stats),
            **totals,
            "hit": rate(totals["cached_tokens"]),
            "compressed_hit": rate(totals["compressed_tokens"]),
            "lost_to_checkpoint": rate(
                totals["wanted_tokens"] - totals["cached_tokens"]
            ),
            "lost_unrecoverable": rate(
                totals["compressed_tokens"] - totals["wanted_tokens"]
            ),
        }

    def get_metrics_statistics(self) -> dict[str, Any]:
        """Return a DP-aggregated snapshot for the Prometheus exporter.

        Reads the snapshots each EngineCore pushes on its own clock, so this is
        a local dict lookup: no round trip, no deadline, and nothing that can
        fail just because the engine is busy.
        """
        rank_stats = [
            stats
            for stats in self.core_mgr.latest_metrics.values()
            if stats.get("enabled", False)
        ]

        def summed(key: str) -> int:
            return sum(int(stats.get(key, 0)) for stats in rank_stats)

        kv_total = summed("kv_blocks_total")
        kv_used = summed("kv_blocks_used")

        mtp_rank_stats = [
            stats.get("mtp", {})
            for stats in rank_stats
            if stats.get("mtp", {}).get("enabled", False)
        ]
        mtp_distribution: Counter[int] = Counter()
        for stats in mtp_rank_stats:
            mtp_distribution.update(
                {
                    int(accepted): int(steps)
                    for accepted, steps in stats.get("distribution", {}).items()
                }
            )
        mtp_draft = sum(
            int(stats.get("total_draft_tokens", 0)) for stats in mtp_rank_stats
        )
        mtp_accepted = sum(
            int(stats.get("total_accepted_tokens", 0)) for stats in mtp_rank_stats
        )
        mtp_steps = sum(mtp_distribution.values())

        cache_rank_stats = [
            stats.get("cache", {})
            for stats in rank_stats
            if stats.get("cache", {}).get("enabled", False)
        ]
        cache_keys = (
            "requests",
            "cached_tokens",
            "compressed_tokens",
            "wanted_tokens",
            "full_tokens",
            "checkpoints_kept",
            "checkpoints_dropped",
            "checkpoints_evicted",
            "checkpoints_orphaned",
            "demands_recorded",
            "demands_declined_no_room",
            "chunks_cut_for_demand",
        )
        cache_totals = {
            key: sum(int(stats.get(key, 0)) for stats in cache_rank_stats)
            for key in cache_keys
        }
        cache_full = cache_totals["full_tokens"]
        offload_rank_stats = [
            stats.get("offload", {}) for stats in rank_stats if stats.get("offload")
        ]
        offload_keys = (
            "load_requests",
            "loaded_tokens",
            "load_failures",
            "save_requests",
            "saved_tokens",
            "loads_pending",
            "saves_pending",
        )
        offload_totals = {
            key: sum(int(stats.get(key, 0)) for stats in offload_rank_stats)
            for key in offload_keys
        }

        return {
            "enabled": bool(rank_stats),
            "requests_running": summed("requests_running"),
            "requests_waiting": summed("requests_waiting"),
            "requests_parked_kv_load": summed("requests_parked_kv_load"),
            "requests_partial_prefill": summed("requests_partial_prefill"),
            "requests_finished": summed("requests_finished"),
            "prompt_tokens": summed("prompt_tokens"),
            "generation_tokens": summed("generation_tokens"),
            "preemptions": summed("preemptions"),
            "kv_blocks_used": kv_used,
            "kv_blocks_free": summed("kv_blocks_free"),
            "kv_blocks_total": kv_total,
            "kv_blocks_indexed": summed("kv_blocks_indexed"),
            "kv_cache_usage_ratio": kv_used / kv_total if kv_total else 0.0,
            "mtp": {
                "enabled": bool(mtp_rank_stats),
                "total_draft_tokens": mtp_draft,
                "total_accepted_tokens": mtp_accepted,
                "acceptance_rate": mtp_accepted / mtp_draft if mtp_draft else 0.0,
                "average_tokens_per_forward": (
                    1 + mtp_accepted / mtp_steps if mtp_steps else 0.0
                ),
                "distribution": dict(sorted(mtp_distribution.items())),
            },
            "cache": {
                "enabled": bool(cache_rank_stats),
                **cache_totals,
                "hit": (
                    cache_totals["cached_tokens"] / cache_full if cache_full else 0.0
                ),
                "compressed_hit": (
                    cache_totals["compressed_tokens"] / cache_full
                    if cache_full
                    else 0.0
                ),
                "lost_to_checkpoint": (
                    (cache_totals["wanted_tokens"] - cache_totals["cached_tokens"])
                    / cache_full
                    if cache_full
                    else 0.0
                ),
                "lost_unrecoverable": (
                    (cache_totals["compressed_tokens"] - cache_totals["wanted_tokens"])
                    / cache_full
                    if cache_full
                    else 0.0
                ),
            },
            "offload": offload_totals,
        }


class InputOutputProcessor:

    def __init__(self, config, tokenizer, block_size):
        self.config = config
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.requests = {}
        # req_id -> truncate_to, for requests a stop string ended. Written by
        # the wrapped stream callback, read once by `postprocess`, which
        # removes the entry with the request.
        self._stop_string_hits: dict[int, int] = {}
        # Set by LLMEngine once its CoreManager exists. A stop string is
        # decided here, in the frontend, so ending the request means telling
        # the engine core to stop working on it.
        self.abort_request = None
        # `has_per_req_cache` flags model architectures that need a
        # per-request stateful buffer outside the paged KV pool. Sequences
        # constructed for these models trigger BlockManager to reserve a
        # per-req cache slot. Currently: GDN-based models (Qwen3-Next /
        # Qwen3.5). Future stateful models (DeepseekV4, etc.) extend the set.
        self._external_to_internal: dict[str, int] = {}
        self._internal_to_external: dict[int, str] = {}
        self.has_per_req_cache = False
        self.num_speculative_tokens = 0
        if (
            hasattr(self.config, "speculative_config")
            and self.config.speculative_config is not None
        ):
            self.num_speculative_tokens = (
                self.config.speculative_config.num_speculative_tokens
            )
        if self.config.hf_config.model_type in self._per_req_cache_model_types():
            self.has_per_req_cache = True

    @staticmethod
    def _per_req_cache_model_types() -> frozenset[str]:
        """Single source of truth for which model_types use per-req cache.

        Read by Sequence-construction (here) AND by ModelRunner's startup
        sanity check, which asserts that any model whose attention builder
        declares a per-request state sub-pool has its model_type
        registered here. Adding a new stateful-attention model means
        adding its model_type to this set.
        """
        return frozenset(
            {
                "qwen3_next",
                "qwen3_5_text",
                "qwen3_5_moe_text",
                "kimi_linear",
                "deepseek_v4",
                # GLM-5.3-Flash: KDA linear-attention layers keep a per-request
                # recurrent state, same as kimi_linear.
                "glm5_next_text",
            }
        )

    def _wrap_for_stop_strings(self, sampling_params, callback):
        """Wrap a stream callback so a stop string ends the request.

        This is the frontend half of the split described in
        `atom.model_engine.stop_strings`: the engine core cannot see text, so
        the one stop condition defined over text is decided here, against the
        detokenized output the client will actually receive, and the request
        is then aborted.

        Requests without stop strings are handed back their own callback
        unchanged -- no detokenizer, no wrapper, no per-step work. That is the
        overwhelming majority of them.
        """
        stops = sampling_params.stop_strings
        if not stops:
            return callback

        detokenizer = IncrementalDetokenizer(self.tokenizer)
        include = sampling_params.include_stop_str_in_output
        # The abort is fire-and-forget and the engine keeps producing until it
        # lands, so more output can arrive for a request already stopped.
        # Everything after the first hit is dropped rather than re-checked.
        state = {"stopped": False}

        def on_output(request_output):
            if not state["stopped"]:
                delta = detokenizer.update(
                    request_output.output_tokens or (), request_output.finished
                )
                hit = check_stop_strings(detokenizer.text, len(delta), stops, include)
                if hit is not None:
                    # Only the cut is kept. Which stop string matched is not
                    # reported: `finish_reason` already says one did, and
                    # OpenAI's schema has no field for the identity.
                    _stop_string, truncate_to = hit
                    if truncate_to < 0:
                        # `check_stop_strings` says -1 when the match already
                        # ends the text, so there is nothing to cut *yet*. The
                        # abort is asynchronous, though, and whatever the
                        # engine emits before it lands still reaches
                        # `completion_token_ids`. Pinning the length here is
                        # what keeps that tail out of the final text.
                        truncate_to = len(detokenizer.text)
                    state["stopped"] = True
                    request_output.finished = True
                    request_output.finish_reason = "stop_sequence"
                    request_output.stop_truncate_to = truncate_to
                    self._stop_string_hits[request_output.request_id] = truncate_to
                    if self.abort_request is not None:
                        try:
                            self.abort_request(request_output.request_id)
                        except Exception:
                            # The request ends for the client either way; a
                            # failed abort only costs the tokens it would have
                            # saved, and raising here would lose the output.
                            logger.warning(
                                "Abort after stop string failed for request %s",
                                request_output.request_id,
                                exc_info=True,
                            )
            elif not request_output.finished:
                # Trailing output from before the abort landed.
                return

            if callback is not None:
                callback(request_output)

        return on_output

    def preprocess(
        self,
        prompt_or_tokens: str | list[int],
        sampling_params: SamplingParams,
        stream_callback=None,
        kv_transfer_params=None,
        multimodal_data=None,
        request_id: str | None = None,
        data_parallel_rank: int | None = None,
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
            multimodal_data=multimodal_data,
            parent_request_id=request_id,
            data_parallel_rank=data_parallel_rank,
        )
        return seqs[0]

    def preprocess_fanout(
        self,
        prompt_or_tokens: str | list[int],
        sampling_params: SamplingParams,
        stream_callback=None,
        stream_callbacks: list | None = None,
        kv_transfer_params=None,
        multimodal_data=None,
        parent_request_id: str | None = None,
        data_parallel_rank: int | None = None,
    ) -> list[Sequence]:
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
        mrope_positions = None
        mrope_position_delta = 0
        if multimodal_data is not None:
            mrope_positions, mrope_position_delta = get_mrope_input_positions(
                self.config,
                tokens,
                multimodal_data,
            )

        if stream_callbacks is not None and len(stream_callbacks) != n:
            raise ValueError(
                f"stream_callbacks length {len(stream_callbacks)} does not match n={n}"
            )

        seqs: list[Sequence] = []
        for i in range(n):
            cb = (
                stream_callbacks[i] if stream_callbacks is not None else stream_callback
            )
            seq = Sequence(
                tokens,
                self.block_size,
                sampling_params,
                stream_callback=self._wrap_for_stop_strings(sampling_params, cb),
                num_draft_tokens=self.num_speculative_tokens,
                has_per_req_cache=self.has_per_req_cache,
                kv_transfer_params=kv_transfer_params,
                multimodal_data=multimodal_data,
                mrope_positions=mrope_positions,
                mrope_position_delta=mrope_position_delta,
                needs_independent_noise=(n > 1),
                parent_request_id=parent_request_id,
                sibling_index=i,
                request_id=parent_request_id if n == 1 else None,
                data_parallel_rank=data_parallel_rank,
            )
            seq.arrive_time = time.time()
            self.requests[seq.id] = seq
            if seq.external_request_id is not None:
                self._external_to_internal[seq.external_request_id] = seq.id
                self._internal_to_external[seq.id] = seq.external_request_id
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

    def postprocess(self, reqs: list[Sequence]):
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
            # A stop string ends the request from here, so this is also where
            # its two effects land: the text is cut where the client asked,
            # and the reason reads `stop_sequence` rather than whatever the
            # engine core happened to report for the tokens that arrived
            # before the abort took hold.
            truncate_to = self._stop_string_hits.pop(req.id, None)
            if truncate_to is not None:
                if truncate_to >= 0:
                    output_str = output_str[:truncate_to]
                req.leave_reason = "stop_sequence"
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
            )
            outputs[req.id] = {
                "text": output_str,
                # `list`, not the `array("i")` slice: this is what `generate()`
                # hands a caller, and the storage type is ours to change.
                "token_ids": list(req.completion_token_ids),
                # `list` for the same reason as `token_ids` above.
                "logprobs": list(req.logprobs) if req.return_logprobs else None,
                "latency": req.leave_time - req.arrive_time,
                "finish_reason": req.leave_reason,
                "num_tokens_input": req.num_prompt_tokens,
                "num_tokens_output": req.num_completion_tokens,
                "ttft": ttft,  # Time to first token in seconds
                "tpot": tpot,  # Time per output token in seconds
            }
        return outputs

    def discard_stop_state(self, req_id) -> None:
        """Forget a request's stop-string hit without producing an output.

        `postprocess` is the ordinary consumer, but a request can leave
        without reaching it -- a disconnected client, an abort -- and the
        entry would then outlive the request.
        """
        self._stop_string_hits.pop(req_id, None)

    def has_pending_requests(self):
        return len(self.requests) > 0
