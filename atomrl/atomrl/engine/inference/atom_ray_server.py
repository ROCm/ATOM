# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Ray-colocated ATOM rollout server for AtomRL.

This mirrors the native Ray vLLM rollout path, but hosts ATOM's
``AsyncLLMEngine`` in each colocated Ray actor. The public RPC surface matches
``VLLMRayServer`` closely enough for ``VLLMHttpEngine`` to route token-in
generation and for the trainer to reuse the existing ZMQ CUDA-IPC weight sync.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import socket
import sys
import time
from multiprocessing import shared_memory
from typing import Any
from uuid import uuid4

import torch
import zmq

logger = logging.getLogger(__name__)
# A Ray actor configures no logging at all, so records fall through to
# `logging.lastResort`, which drops anything below WARNING. Everything
# this module says at INFO -- including the KV pool each engine actually
# got, the number that makes `gpu_memory_utilization` comparable between
# engines -- was being thrown away. Setting the level is not enough on its
# own; the records need somewhere to go. Ray relays actor stderr to the
# driver, so a stream handler is all it takes, and the guard keeps the
# driver process (where the root logger is already configured) from
# printing everything twice.
logger.setLevel(os.getenv("ATOMRL_LOGGING_LEVEL", "INFO"))
if not logger.handlers and not logging.getLogger().handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_h)


class ATOMRayServer:
    """Ray actor hosting one ATOM AsyncLLMEngine on a colocated GPU."""

    def __init__(
        self,
        model_name: str,
        engine_kwargs: dict[str, Any],
        replica_rank: int,
        base_seed: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.engine_kwargs = dict(engine_kwargs)
        self.replica_rank = int(replica_rank)
        self.base_seed = base_seed
        self._seed_warning_emitted = False
        self.engine = None
        # ATOM's preemption counter is cumulative; keep the last reading so
        # each batch can report its own.
        self._preemptions_seen = 0
        # Set by the first generate_group(); from then on every request goes
        # through it (see AtomGroupPump).
        self._pump: Any = None

    async def launch(self) -> bool:
        from atom.rollout.async_engine import AsyncLLMEngine

        kwargs = dict(self.engine_kwargs)
        kwargs.setdefault("model", self.model_name)
        kwargs.setdefault("master_addr", self._get_node_ip())
        kwargs.setdefault("port", self._get_free_port())
        self._pin_cudagraph_mode(kwargs)
        self.engine = AsyncLLMEngine(**kwargs)
        logger.info(
            "ATOMRayServer[%d]: AsyncLLMEngine ready (master=%s:%s online_quant=%s).",
            self.replica_rank,
            kwargs.get("master_addr"),
            kwargs.get("port"),
            kwargs.get("online_quant_config"),
        )
        return True

    @staticmethod
    def _is_no_eager(kwargs: dict[str, Any]) -> bool:
        """Will this rollout run torch.compile and capture CUDA graphs?

        `_pin_cudagraph_mode` applies exactly here and nowhere else: under
        `enforce_eager` there are no graphs for a cudagraph_mode to describe.
        """
        comp_cfg = kwargs.get("compilation_config") or {}
        level = int(comp_cfg.get("level", 0) or 0)
        return level > 0 or not bool(kwargs.get("enforce_eager", True))

    def _pin_cudagraph_mode(self, kwargs: dict[str, Any]) -> None:
        """Choose ATOM's CUDA-graph strategy for a no-eager rollout.

        Only applies once torch.compile is on (``enforce_eager=false`` or
        ``compilation_config.level>0``). ATOM leaves ``cudagraph_mode`` unset and
        then defaults it to PIECEWISE, which gives every compiled dense piece its
        own graph and asserts that the piece's inputs keep their capture-time
        addresses. The attention between two pieces runs eager and allocates its
        output afresh each call, so the very first replay aborts all rollout
        workers with "Input addresses for cudagraphs are different during
        replay". FULL captures the whole forward instead and has no such
        boundary.

        Override with ``atom_cfg.engine_kwargs.compilation_config.cudagraph_mode``.
        ATOM builds that pin the mode themselves still win — this only supplies a
        value.
        """
        if not self._is_no_eager(kwargs):
            return

        comp_cfg = dict(kwargs.get("compilation_config") or {})
        level = int(comp_cfg.get("level", 0) or 0)
        mode = comp_cfg.get("cudagraph_mode") or "FULL"
        if isinstance(mode, str):
            from atom.config import CUDAGraphMode

            try:
                mode = CUDAGraphMode[mode.upper()]
            except KeyError as exc:
                supported = ", ".join(m.name for m in CUDAGraphMode)
                raise ValueError(
                    f"unknown ATOM cudagraph_mode {mode!r}; supported: {supported}"
                ) from exc

        comp_cfg["cudagraph_mode"] = mode
        kwargs["compilation_config"] = comp_cfg
        logger.info(
            "ATOMRayServer[%d]: no-eager rollout with compilation level=%d, "
            "cudagraph_mode=%s",
            self.replica_rank,
            level,
            getattr(mode, "name", mode),
        )

    @staticmethod
    def _get_node_ip() -> str:
        try:
            import ray

            return ray.util.get_node_ip_address()
        except OSError:
            return socket.gethostbyname(socket.gethostname())

    @staticmethod
    def _get_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return int(sock.getsockname()[1])

    def ready(self) -> bool:
        return self.engine is not None

    def _build_sampling_params(self, params: dict[str, Any], prompt_length: int):
        from atom.sampling_params import SamplingParams

        params = dict(params)
        max_model_len = int(self.engine_kwargs.get("max_model_len") or 0)
        max_possible = None
        if max_model_len > 0:
            max_possible = max(1, max_model_len - int(prompt_length))

        max_tokens = params.pop("max_tokens", params.pop("max_new_tokens", 128))
        if max_possible is not None:
            max_tokens = min(int(max_tokens), max_possible)

        requested_seed = params.pop("seed", None)
        if (
            requested_seed is not None or self.base_seed is not None
        ) and not self._seed_warning_emitted:
            logger.warning(
                "ATOM SamplingParams has no per-request seed contract; "
                "ignoring requested/base seed rather than probing private fields"
            )
            self._seed_warning_emitted = True

        # ATOM accepts bool/int for logprobs. AtomRL passes vLLM-style
        # logprobs=0 when token logprobs are requested.
        logprobs = params.pop("logprobs", None)
        if logprobs is not None and not isinstance(logprobs, bool):
            logprobs = int(logprobs) >= 0

        if params.pop("do_sample", None) is False:
            params["temperature"] = 0.0
            params["top_k"] = -1
            params["top_p"] = 1.0

        temperature = float(params.pop("temperature", 1.0))
        top_p = float(params.pop("top_p", 1.0))
        top_k = int(params.pop("top_k", -1))
        ignore_eos = bool(params.pop("ignore_eos", False))
        stop_strings = params.pop("stop_strings", params.pop("stop", None))

        for key in ("repetition_penalty", "stop_token_ids", "min_tokens"):
            if key in params:
                logger.debug(
                    "ATOM rollout dropping unsupported sampling param %s=%r",
                    key,
                    params.pop(key),
                )
        if params:
            logger.debug(
                "ATOM rollout dropping unsupported sampling params: %s", sorted(params)
            )

        sp_kwargs = {
            "max_tokens": max(1, int(max_tokens)),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "logprobs": logprobs,
            "ignore_eos": ignore_eos,
            "stop_strings": stop_strings,
        }
        return SamplingParams(**sp_kwargs)

    @staticmethod
    def _completion(out: Any, prompt_ids: list[int]) -> dict[str, Any]:
        """One ATOM output as the result dict every generate path returns.

        ``routed_experts`` (R3, int16 ``[seq_len-1, layers, top_k]``) is kept
        when the engine returned it.
        """
        out = out if isinstance(out, dict) else {}
        logprobs = out.get("logprobs")
        result = {
            "text": out.get("text", ""),
            "prompt_token_ids": list(prompt_ids),
            "token_ids": list(out.get("token_ids", [])),
            "logprobs": [float(x) for x in logprobs] if logprobs is not None else None,
        }
        routed = out.get("routed_experts")
        if routed is not None:
            result["routed_experts"] = routed
        return result

    async def generate(
        self,
        prompt: list[int],
        sampling_params: dict[str, Any],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        if self.engine is None:
            raise RuntimeError(
                "ATOMRayServer.launch() must be called before generate()."
            )

        prompt_ids = list(prompt)
        rid = request_id or uuid4().hex
        if self._pump is not None:
            [result] = await self._generate_on_pump(prompt_ids, 1, sampling_params, rid)
            return result
        sp = self._build_sampling_params(sampling_params, prompt_length=len(prompt_ids))

        def _generate_blocking():
            return self.engine.generate([prompt_ids], sp, request_ids=[rid])[0]

        out = await asyncio.get_event_loop().run_in_executor(None, _generate_blocking)
        return self._completion(out, prompt_ids)

    async def generate_batch(
        self,
        prompts: list[list[int]],
        sampling_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self.engine is None:
            raise RuntimeError(
                "ATOMRayServer.launch() must be called before generate_batch()."
            )

        prompt_ids_list = [list(p) for p in prompts]
        grouped_prompts: list[list[int]] = []
        grouped_counts: list[int] = []
        for prompt_ids in prompt_ids_list:
            if grouped_prompts and prompt_ids == grouped_prompts[-1]:
                grouped_counts[-1] += 1
            else:
                grouped_prompts.append(prompt_ids)
                grouped_counts.append(1)

        if self._pump is not None:
            groups = await asyncio.gather(
                *(
                    self._generate_on_pump(p, n, sampling_params)
                    for p, n in zip(grouped_prompts, grouped_counts)
                )
            )
            return [result for group in groups for result in group]

        request_ids = [uuid4().hex for _ in grouped_prompts]
        params = []
        for prompt_ids, n in zip(grouped_prompts, grouped_counts):
            sp = self._build_sampling_params(
                sampling_params, prompt_length=len(prompt_ids)
            )
            sp.n = n
            params.append(sp)

        def _generate_blocking():
            return self.engine.generate(
                grouped_prompts, params, request_ids=request_ids
            )

        outs = await asyncio.get_event_loop().run_in_executor(None, _generate_blocking)
        self._log_kv_pressure()
        expanded_prompts = [
            p for p, n in zip(grouped_prompts, grouped_counts) for _ in range(n)
        ]
        if os.environ.get("ATOMRL_ATOM_CHECK_ALIGNMENT"):
            self._audit_alignment(expanded_prompts, outs)
        return [self._completion(out, p) for p, out in zip(expanded_prompts, outs)]

    async def generate_group(
        self,
        prompt: list[int],
        num_generations: int,
        sampling_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """``num_generations`` completions of one prompt, safe to run concurrently.

        ``engine.generate`` holds the engine until it is idle, so it serves one
        batch at a time; streaming groups goes through an AtomGroupPump instead,
        which every later request on this server shares.
        """
        if self.engine is None:
            raise RuntimeError(
                "ATOMRayServer.launch() must be called before generate_group()."
            )
        if self._pump is None:
            from atomrl.engine.inference.atom_group_pump import AtomGroupPump

            self._pump = AtomGroupPump(self.engine)
        return await self._generate_on_pump(
            list(prompt), num_generations, sampling_params
        )

    async def _generate_on_pump(
        self,
        prompt_ids: list[int],
        num_generations: int,
        sampling_params: dict[str, Any],
        request_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sp = self._build_sampling_params(sampling_params, prompt_length=len(prompt_ids))
        sp.n = num_generations
        outs = await self._pump.submit(prompt_ids, sp, request_id or uuid4().hex)
        return [self._completion(out, prompt_ids) for out in outs]

    def _log_kv_pressure(self) -> None:
        """Report how hard this batch leaned on the KV pool.

        `timing_s/agent_loop/num_preempted` in the trainer metrics is a vLLM
        agent-loop counter that nothing on the ATOM path ever writes, so it
        reads 0 no matter what ATOM did -- which is how thousands of preemptions
        per step went unnoticed while the rollout quality fell over. ATOM keeps
        the real number; this is the one place that asks for it.

        Preemption is recompute: the sequence gives back the KV for everything
        it has generated so far and forwards the whole context again. A pool
        that cannot hold `max_num_seqs` sequences at their real length pays that
        repeatedly, so this is also the first number to look at when generation
        is slower than it should be.
        """
        try:
            stats = self.engine.get_metrics_statistics()
        except Exception:
            logger.debug(
                "ATOM[%d]: KV pressure stats unavailable",
                self.replica_rank,
                exc_info=True,
            )
            return
        total = stats.get("kv_blocks_total") or 0
        preempt = int(stats.get("preemptions") or 0)
        delta = preempt - self._preemptions_seen
        self._preemptions_seen = preempt
        logger.info(
            "ATOM[%d] kv: blocks=%s used=%s (%.0f%%) preemptions=+%d (total %d)",
            self.replica_rank,
            total,
            stats.get("kv_blocks_used"),
            100.0 * float(stats.get("kv_cache_usage_ratio") or 0.0),
            delta,
            preempt,
        )

    def _audit_alignment(self, expanded_prompts: list[list[int]], outs: list) -> None:
        """Check that result i really was decoded against prompt i.

        ``generate_batch`` pairs ATOM's results with its own prompt list by
        position. ATOM now echoes the prompt each sequence actually used, so
        the pairing can be verified instead of assumed.

        Off unless ``ATOMRL_ATOM_CHECK_ALIGNMENT`` is set: it indexes every
        prompt in the batch, which is 768 per replica on the MoE rollout, and
        it has never once fired. Kept because the question it answers -- "did
        the engine hand back someone else's completion?" -- is the first one
        worth asking when generations look swapped, and it is the only way to
        separate that from the KV-level cross-talk that looks identical from
        the outside.
        """
        n_out = len(outs)
        n_want = len(expanded_prompts)
        # Where each distinct prompt sits in the list we sent.
        where: dict[tuple, list[int]] = {}
        for i, p_ids in enumerate(expanded_prompts):
            where.setdefault(tuple(p_ids), []).append(i)

        bad = 0
        shifts: dict[Any, int] = {}
        examples: list[tuple] = []
        for i, (p_ids, out) in enumerate(zip(expanded_prompts, outs)):
            got = out.get("prompt_token_ids") if isinstance(out, dict) else None
            if got is None:
                logger.warning(
                    "ATOM[%d] align-audit: engine did not echo prompts.",
                    self.replica_rank,
                )
                return
            if list(got) == list(p_ids):
                continue
            bad += 1
            # Which slot did this response actually belong to? A constant
            # offset means outputs are shifted (something went missing); a
            # scattered spread means they came back permuted.
            homes = where.get(tuple(got))
            delta = min((h - i for h in homes), key=abs) if homes else "unsent"
            shifts[delta] = shifts.get(delta, 0) + 1
            if len(examples) < 3:
                examples.append((i, delta))
        healthy = bad == 0 and n_out == n_want
        logger.log(
            logging.INFO if healthy else logging.WARNING,
            "ATOM[%d] align-audit: returned=%d expected=%d mismatched=%d "
            "offset_histogram=%s examples(idx,offset)=%s",
            self.replica_rank,
            n_out,
            n_want,
            bad,
            sorted(shifts.items(), key=lambda kv: -kv[1])[:8],
            examples,
        )

    async def update_weights_from_ipc(
        self, use_shm: bool = False, version: int | None = None
    ) -> bool:
        if self.engine is None:
            raise RuntimeError(
                "ATOMRayServer.launch() must be called before weight sync."
            )
        with self.engine.weight_sync_session():
            if use_shm:
                self._update_weights_from_shm_sync(version)
            else:
                self._update_weights_from_ipc_sync(version)
        return True

    def _get_zmq_handle(self) -> str:
        replica_rank = os.environ.get("LUMEN_REPLICA_RANK", "0")
        job_id = os.environ.get("LUMEN_RAY_JOB_ID", "0")
        return (
            f"ipc:///tmp/lumen-colocate-zmq-{job_id}-replica-{replica_rank}-rank-0.sock"
        )

    def _counts_are_exact(self) -> bool:
        """Whether ATOM's per-bucket ``updated`` can be compared for equality.

        Only on a BF16 rollout. With online quantization on, a fused parameter's
        shards accumulate in a staging buffer and are requantized when the last
        one arrives, so ATOM counts one update for the group and nothing for the
        shards ahead of it -- a bucket that ends mid-group reports fewer updates
        than it holds, with nothing wrong. See assert_bucket_fully_applied.
        """
        return not (self.engine_kwargs.get("online_quant_config") or {})

    @staticmethod
    def _bucket_meta(
        raw_bucket_meta: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], int]:
        bucket_meta: dict[str, dict[str, Any]] = {}
        used_bytes = 0
        for name, meta in raw_bucket_meta.items():
            shape = tuple(meta["shape"])
            dtype = meta["dtype"]
            offset = int(meta["offset"])
            nbytes = int(dtype.itemsize * torch.Size(shape).numel())
            bucket_meta[name] = {
                "shape": shape,
                "dtype": str(dtype),
                "offset": offset,
                "nbytes": nbytes,
            }
            used_bytes = max(used_bytes, offset + nbytes)
        return bucket_meta, used_bytes

    def _update_weights_from_ipc_sync(self, version: int | None = None) -> None:
        from atom.rollout.weight_sync import rebuild_ipc_handle
        from atomrl.engine.inference.bucketed_weight_transfer import (
            check_bucket_version,
        )

        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self._get_zmq_handle())

        per_gpu_buffers = None
        ipc_buffer = None
        try:
            comm_metadata = socket.recv_pyobj()
            socket.send(b"")
            ipc_buffer = rebuild_ipc_handle(comm_metadata, device_id=0)
            bucket_size = int(ipc_buffer.numel())
            num_gpus = int(
                self.engine_kwargs.get("tensor_parallel_size", 1) or 1
            ) * int(self.engine_kwargs.get("data_parallel_size", 1) or 1)
            per_gpu_buffers = {
                gpu_idx: torch.empty(
                    bucket_size, dtype=torch.uint8, device=f"cuda:{gpu_idx}"
                )
                for gpu_idx in range(num_gpus)
            }
            stats = {"buckets": 0, "weights": 0}
            first_metadata = socket.recv_pyobj()
            check_bucket_version(first_metadata, version)
            first_used_bytes = self._bucket_meta(first_metadata["bucket_meta"])[1]
            public_bucket_mb = max(
                1,
                (max(bucket_size, first_used_bytes) + (1 << 20) - 1) >> 20,
            )
            pending_ack = False
            last_bucket_seen = False
            shared_ipc_buffer = ipc_buffer

            def weights():
                nonlocal bucket_size, pending_ack, last_bucket_seen, per_gpu_buffers
                metadata = first_metadata
                while True:
                    if pending_ack:
                        socket.send(b"")
                        pending_ack = False
                    if metadata is None:
                        metadata = socket.recv_pyobj()
                    check_bucket_version(metadata, version)
                    raw_bucket_meta = metadata["bucket_meta"]
                    is_last = bool(metadata["is_last"])
                    bucket_meta, used_bytes = self._bucket_meta(raw_bucket_meta)

                    direct_tensors = {
                        name: rebuild_ipc_handle(meta["handle"], device_id=0)
                        for name, meta in raw_bucket_meta.items()
                        if meta.get("handle") is not None
                    }
                    if used_bytes > bucket_size:
                        if stats["buckets"] > 0:
                            raise RuntimeError(
                                f"bucket needs {used_bytes} B but the staging buffer is "
                                f"{bucket_size} B and ATOM's runner has already mapped it; "
                                "the sender must size its bucket to the largest tensor "
                                "(see AtomActorWorker.update_weights_ipc_send)"
                            )
                        del per_gpu_buffers
                        bucket_size = used_bytes
                        per_gpu_buffers = {
                            gpu_idx: torch.empty(
                                bucket_size,
                                dtype=torch.uint8,
                                device=f"cuda:{gpu_idx}",
                            )
                            for gpu_idx in range(num_gpus)
                        }
                    for gpu_idx, dst in per_gpu_buffers.items():
                        for name, tensor in direct_tensors.items():
                            meta = raw_bucket_meta[name]
                            nbytes = (
                                meta["dtype"].itemsize
                                * torch.Size(meta["shape"]).numel()
                            )
                            offset = int(meta["offset"])
                            dst[offset : offset + nbytes].copy_(
                                tensor.contiguous().view(-1).view(torch.uint8),
                                non_blocking=True,
                            )
                        if not direct_tensors:
                            dst[:used_bytes].copy_(
                                shared_ipc_buffer[:used_bytes], non_blocking=True
                            )
                        torch.cuda.synchronize(gpu_idx)
                    source = per_gpu_buffers[0]
                    for name, meta in raw_bucket_meta.items():
                        nbytes = (
                            meta["dtype"].itemsize * torch.Size(meta["shape"]).numel()
                        )
                        offset = int(meta["offset"])
                        yield (
                            name,
                            source[offset : offset + nbytes]
                            .view(meta["dtype"])
                            .reshape(tuple(meta["shape"])),
                        )
                    stats["buckets"] += 1
                    stats["weights"] += len(bucket_meta)
                    pending_ack = True
                    if is_last:
                        last_bucket_seen = True
                        return
                    metadata = None

            self.engine.load_weights(
                weights(),
                bucket_size_mb=public_bucket_mb,
                num_gpus=num_gpus,
                mode="ipc",
            )
            if pending_ack or last_bucket_seen:
                socket.send(b"")
            logger.info(
                "ATOM online weight reload: buckets=%d weights=%d",
                stats["buckets"],
                stats["weights"],
            )
        finally:
            socket.close()
            ctx.term()
            del per_gpu_buffers
            del ipc_buffer
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def _update_weights_from_shm_sync(self, version: int | None = None) -> None:
        from atomrl.engine.inference.bucketed_weight_transfer import (
            check_bucket_version,
        )

        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self._get_zmq_handle())
        shm = None
        try:
            comm_metadata = socket.recv_pyobj()
            socket.send(b"")
            shm = shared_memory.SharedMemory(name=comm_metadata["name"])

            def weights():
                pending_ack = False
                while True:
                    if pending_ack:
                        socket.send(b"")
                    metadata = socket.recv_pyobj()
                    check_bucket_version(metadata, version)
                    raw_bucket_meta = metadata["bucket_meta"]
                    for name, meta in raw_bucket_meta.items():
                        nbytes = int(
                            meta["dtype"].itemsize * torch.Size(meta["shape"]).numel()
                        )
                        tensor = torch.frombuffer(
                            shm.buf,
                            dtype=meta["dtype"],
                            count=nbytes // meta["dtype"].itemsize,
                            offset=int(meta["offset"]),
                        ).reshape(tuple(meta["shape"]))
                        yield name, tensor
                    pending_ack = True
                    if metadata["is_last"]:
                        return

            self.engine.load_weights(weights(), mode="shm")
        finally:
            if shm is not None:
                shm.close()
            socket.close()
            ctx.term()

    async def sleep(self, level: int = 2) -> bool:
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called first.")
        self.engine.sleep(level=level)
        return True

    async def wake_up(self, tags: list[str] | None = None) -> bool:
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called first.")
        self.engine.wake_up(tags=tags or ["weights", "kv_cache"])
        return True

    async def generation_pause_modes(self) -> list[str]:
        """Modes ``pause_generation`` takes here; empty on an ATOM that cannot pause.

        ATOM reports it as the ``pause_generation`` capability, which takes ``keep``.
        """
        if (await self.get_capabilities()).supports("pause_generation"):
            return ["keep"]
        return []

    async def pause_generation(
        self, mode: str = "keep", clear_cache: bool = True
    ) -> bool:
        """Freeze generation for a weight update, as ``VLLMRayServer`` does.

        What runs waits. The weight update preempts it and drops the KV the old
        weights computed, so ``clear_cache`` leaves ATOM nothing to do. After
        ``resume_generation`` the requests recompute their context on the new
        weights and go on.
        """
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called first.")
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.engine.pause_generation(mode=mode)
        )
        return True

    async def resume_generation(self) -> bool:
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called first.")
        await asyncio.get_event_loop().run_in_executor(
            None, self.engine.resume_generation
        )
        return True

    async def collective_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        barrier: bool = False,
        timeout: float = 600.0,
    ) -> list[Any]:
        """Invoke *method* on every ATOM model runner, DP-major then TP rank.

        Mirrors ``VLLMRayServer.collective_rpc`` so the same caller drives
        either backend. Returns one entry per rank; a rank that failed raises
        here rather than returning a half-answer, because every current caller
        treats the call as a barrier and would otherwise proceed on partial
        state.

        The executor hop is required: the engine call blocks on a queue, and
        running it on the actor's event loop would stall ``generate`` for the
        whole weight sync. ``_generate_blocking`` above uses the same pattern.
        """
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called first.")

        def _blocking():
            return self.engine.collective_rpc(
                method,
                timeout=timeout,
                args=tuple(args),
                kwargs=kwargs or {},
                barrier=barrier,
            )

        replies = await asyncio.get_event_loop().run_in_executor(None, _blocking)
        failed = [r for r in replies if not r.ok]
        if failed:
            raise RuntimeError(
                f"collective_rpc({method}) failed on {len(failed)}/{len(replies)} "
                "ranks: " + "; ".join(f"tp{r.tp_rank}: {r.error}" for r in failed[:8])
            )
        return [r.value for r in replies]

    async def get_capabilities(self) -> Any:
        """What this ATOM engine supports, negotiated rather than assumed.

        Lets the caller stop probing with ``hasattr`` and stop hardcoding
        behaviour per backend. Features are intersected across ranks upstream,
        so anything reported here is usable by a collective.
        """
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called first.")

        def _blocking():
            return self.engine.get_capabilities()

        return await asyncio.get_event_loop().run_in_executor(None, _blocking)

    def rdma_preflight(self, interface: str, hca: str) -> dict[str, Any]:
        """Fail now, loudly, if this container cannot do RDMA at all.

        Without this the group still forms and the transfer still "works" --
        over TCP, at a fraction of the bandwidth, with nothing in the logs
        saying so. Checked before the rendezvous because a missing device here
        would otherwise surface as all nine ranks hanging for the full timeout.
        """
        from pathlib import Path

        uverbs = sorted(Path("/dev/infiniband").glob("uverbs*"))
        if not uverbs or not (Path("/sys/class/infiniband") / hca).exists():
            raise RuntimeError(
                f"RDMA unavailable in ATOM rollout container: "
                f"uverbs={[p.name for p in uverbs]}, hca={hca!r}"
            )
        if os.environ.get("NCCL_IB_DISABLE", "0") == "1":
            raise RuntimeError("NCCL_IB_DISABLE=1 would force Socket transport")
        return {
            "replica": self.replica_rank,
            "interface": interface,
            "hca": hca,
            "uverbs": len(uverbs),
        }

    async def init_rdma_weight_group(
        self,
        master_addr: str,
        master_port: int,
        base_rank: int,
        world_size: int,
        group_name: str,
        timeout_s: int = 600,
    ) -> bool:
        """Join every rank of this replica to the trainer's broadcast group.

        Rank 0 is the trainer; this replica occupies ``base_rank`` upward. The
        per-rank offset is computed worker-side, since only the worker knows its
        own TP rank and local DP rank.
        """
        await self.collective_rpc(
            "init_rdma_weight_group",
            kwargs={
                "master_addr": master_addr,
                "master_port": int(master_port),
                "base_rank": int(base_rank),
                "world_size": int(world_size),
                "group_name": group_name,
                "timeout_s": int(timeout_s),
            },
            timeout=float(timeout_s),
        )
        return True

    async def receive_weights_rdma(
        self,
        group_name: str,
        version: int,
        verify_full_load: bool = True,
        prequantized_fp8: bool = False,
    ) -> Any:
        """Receive one weight version into every rank of this replica.

        ATOM checks that the stream covered its weights when the sync completes,
        so ``verify_full_load`` only matters to the vLLM receiver.
        """
        if prequantized_fp8:
            # Rejected rather than ignored: silently dropping it would leave the
            # trainer quantising and ATOM expecting BF16, which shows up as
            # garbage output rather than an error.
            raise NotImplementedError(
                "ATOM's RDMA receive path is BF16-only; "
                "weight_sync fp8 quantization on the trainer is not supported yet"
            )
        # A weight stream is tens of GB; the default RPC budget is far too
        # short for it.
        timeout = float(os.environ.get("ATOMRL_RDMA_RECV_TIMEOUT_S", "1800"))
        # A weight sync: the engine fences serving around it and drops the
        # KV the old weights computed.
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.engine.receive_weights_rdma(
                group_name, int(version), timeout=timeout
            ),
        )

    async def destroy_rdma_weight_group(self, group_name: str) -> bool:
        if self.engine is None:
            return True
        await self.collective_rpc(
            "destroy_rdma_weight_group", kwargs={"group_name": group_name}
        )
        return True

    async def reset_prefix_cache(self) -> bool:
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called first.")
        await asyncio.get_event_loop().run_in_executor(
            None, self.engine.reset_prefix_cache
        )
        return True

    async def wait_for_requests_to_drain(self, timeout_s: float = 60.0) -> bool:
        """Block until the engine reports no pending requests.

        Previously ``return True``, which made every caller's drain barrier a
        no-op: a weight sync could begin while responses were still in flight,
        producing rollouts from a half-updated model. ATOM exposes
        ``is_finished()``, so the barrier can be real.
        """
        if self.engine is None:
            raise RuntimeError("ATOMRayServer.launch() must be called first.")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if await asyncio.get_event_loop().run_in_executor(
                None, self.engine.is_finished
            ):
                return True
            await asyncio.sleep(0.05)
        logger.warning("ATOM requests did not drain within %.1fs", timeout_s)
        return False

    async def shutdown(self) -> bool:
        if self.engine is None:
            return True
        self.engine.shutdown()
        self.engine = None
        return True

    async def reload_weights_from_path(self, weight_dir: str) -> bool:
        """Reload weights from a safetensors directory (for multi-GPU TP replicas)."""
        if self.engine is None:
            raise RuntimeError(
                "ATOMRayServer.launch() must be called before reload_weights_from_path()."
            )

        import json

        from safetensors.torch import load_file

        def read_weight_metadata() -> tuple[list[str], int]:
            index_path = os.path.join(weight_dir, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as index_file:
                    index = json.load(index_file)
                filenames = sorted(set(index["weight_map"].values()))
            else:
                filenames = sorted(
                    name
                    for name in os.listdir(weight_dir)
                    if name.endswith(".safetensors")
                )

            largest_tensor = 0
            for filename in filenames:
                with open(os.path.join(weight_dir, filename), "rb") as weight_file:
                    header_size = int.from_bytes(weight_file.read(8), "little")
                    header = json.loads(weight_file.read(header_size))
                for key, entry in header.items():
                    if key != "__metadata__":
                        begin, end = entry["data_offsets"]
                        largest_tensor = max(largest_tensor, end - begin)
            return filenames, largest_tensor

        files, largest = await asyncio.to_thread(read_weight_metadata)

        def weight_iter():
            for fname in files:
                sd = load_file(os.path.join(weight_dir, fname))
                yield from sd.items()

        # A bucket must hold the largest tensor whole, and an FSDP2 export
        # carries FP32 master weights: Qwen3-8B's embedding alone is 2.3 GiB.
        # Sized from the safetensors headers, which give every tensor's bytes.
        bucket_size_mb = max(2048, -(-largest // (1 << 20)))
        self.engine.load_weights(
            weight_iter(),
            bucket_size_mb=bucket_size_mb,
            mode="shm",
        )
        logger.info(
            "ATOMRayServer[%d]: reloaded weights from %s", self.replica_rank, weight_dir
        )
        return True


class ATOMReplicaManager:
    """Driver-side controller for colocated ATOM rollout actors."""

    def __init__(
        self,
        actor_wg,
        model_name: str,
        engine_kwargs: dict[str, Any],
        *,
        max_concurrency: int = 64,
        base_seed: int | None = None,
    ) -> None:
        self.actor_wg = actor_wg
        self.model_name = model_name
        self.engine_kwargs = dict(engine_kwargs)
        self.max_concurrency = int(max_concurrency)
        self.base_seed = base_seed
        self.num_replicas = actor_wg.num_workers
        self.servers: list = []
        # Retained rather than recomputed in create(): the RDMA rank striding
        # needs both, and a replica running DP internally contributes
        # tp * dp ranks to the group, not tp.
        self.tensor_parallel_size = int(
            self.engine_kwargs.get("tensor_parallel_size", 1) or 1
        )
        self.data_parallel_size = int(
            self.engine_kwargs.get("data_parallel_size", 1) or 1
        )
        self.rdma_group_name: str | None = None

    def create(self) -> None:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        job_id = ray.get_runtime_context().get_job_id()
        infos = self.actor_wg.execute_all_sync("get_colocation_info")
        logger.info("ATOMReplicaManager: colocation infos = %s", infos)

        atom_tp = int(self.engine_kwargs.get("tensor_parallel_size", 1) or 1)
        num_workers = len(infos)
        if num_workers % atom_tp != 0:
            raise ValueError(
                f"num_workers ({num_workers}) must be divisible by "
                f"atom tensor_parallel_size ({atom_tp})"
            )
        num_replicas = max(1, num_workers // atom_tp)
        self.num_replicas = num_replicas

        true_vocab_size = self._get_true_vocab_size()
        # ATOM's no-eager compilation_config.level>0 rollout needs a live Dynamo, but the
        # FSDP2 training actors must stay on TORCHDYNAMO_DISABLE=1. Scope the opt-in to the
        # rollout actors here instead of letting the launcher export it process-tree wide.
        dynamo_required = self._torch_compile_enabled()
        remote_cls = ray.remote(ATOMRayServer)

        for r in range(num_replicas):
            group = infos[r * atom_tp : (r + 1) * atom_tp]
            node_id = group[0]["node_id"]
            all_gpu_ids: list[str] = []
            for info in group:
                all_gpu_ids.extend(str(g) for g in info["gpu_ids"])
            gpu_ids_str = ",".join(all_gpu_ids)

            nccl_port = 29500 + r
            disable_custom_ar = os.environ.get(
                "ATOMRL_DISABLE_CUSTOM_AR", "1" if atom_tp > 1 else "0"
            )
            env_vars = {
                "CUDA_VISIBLE_DEVICES": gpu_ids_str,
                "HIP_VISIBLE_DEVICES": gpu_ids_str,
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
                "MASTER_PORT": str(nccl_port),
                "LUMEN_REPLICA_RANK": str(r),
                "LUMEN_RAY_JOB_ID": str(job_id),
            }
            if dynamo_required:
                env_vars["TORCHDYNAMO_DISABLE"] = "0"
            for key in (
                "ATOMRL_ATOM_CHECK_ALIGNMENT",
                "ATOM_ISOLATE_TORCH_COMPILE_CACHE",
                "ATOM_LOG_LEVEL",
                "ATOM_USE_TORCH_RMSNORM",
                "ATOM_FORCE_ATTN_TRITON",
                "VERL_ATOM_AGENT_LOG",
                "VERL_MEMORY_AGENT_LOG",
            ):
                if key in os.environ:
                    env_vars[key] = os.environ[key]

            engine_kwargs = self._engine_kwargs_for_replica(r, job_id)
            if true_vocab_size is not None:
                engine_kwargs.setdefault("true_vocab_size", int(true_vocab_size))
            engine_kwargs.setdefault(
                "disable_custom_all_reduce",
                disable_custom_ar in ("1", "true", "True"),
            )

            _dbg = os.environ.get("ATOMRL_DEBUG", "0") in ("1", "true", "True")
            if _dbg:
                logger.info(
                    "[DBG] ATOMReplicaManager: replica %d — gpus=%s disable_ca=%s runner=%s",
                    r,
                    gpu_ids_str,
                    disable_custom_ar,
                    engine_kwargs.get("runner_qualname", "default"),
                )

            server = remote_cls.options(
                num_gpus=0,
                num_cpus=1,
                name=f"lumen-atom-replica-{r}",
                max_concurrency=self.max_concurrency,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
                runtime_env={"env_vars": env_vars},
            ).remote(
                model_name=self.model_name,
                engine_kwargs=engine_kwargs,
                replica_rank=r,
                base_seed=self.base_seed,
            )
            self.servers.append(server)

        logger.info(
            "ATOMReplicaManager: rollout-scoped TORCHDYNAMO_DISABLE=%s (driver keeps %s)",
            "0" if dynamo_required else "<inherited>",
            os.environ.get("TORCHDYNAMO_DISABLE", "<unset>"),
        )
        for i, s in enumerate(self.servers):
            ray.get(s.launch.remote())
            logger.info(
                "ATOMReplicaManager: replica %d/%d launched.", i + 1, num_replicas
            )
        logger.info(
            "ATOMReplicaManager: launched %d colocated rollout replicas (atom_tp=%d, workers=%d).",
            num_replicas,
            atom_tp,
            num_workers,
        )

    def _torch_compile_enabled(self) -> bool:
        """True when the ATOM engine will run torch.compile (no-eager or level>0)."""
        comp_cfg = self.engine_kwargs.get("compilation_config") or {}
        level = int(comp_cfg.get("level", 0) or 0)
        return level > 0 or not bool(self.engine_kwargs.get("enforce_eager", True))

    def _get_true_vocab_size(self) -> int | None:
        try:
            from transformers import AutoTokenizer

            vocab_size = len(
                AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            )
            logger.info(
                "ATOMReplicaManager: true tokenizer vocab size = %d", vocab_size
            )
            return int(vocab_size)
        except (ImportError, OSError, ValueError) as exc:
            logger.warning(
                "ATOMReplicaManager: failed to resolve tokenizer vocab size: %s", exc
            )
            return None

    def _engine_kwargs_for_replica(
        self, replica_rank: int, job_id: str
    ) -> dict[str, Any]:
        kwargs = dict(self.engine_kwargs)
        if os.getenv("ATOM_ISOLATE_TORCH_COMPILE_CACHE", "0") not in {
            "1",
            "true",
            "TRUE",
            "yes",
            "YES",
        }:
            return kwargs

        if not self._torch_compile_enabled():
            return kwargs

        comp_cfg = dict(kwargs.get("compilation_config") or {})
        cache_root = os.getenv(
            "ATOM_TORCH_COMPILE_CACHE_ROOT", "/tmp/atom_torch_compile_cache"
        )
        safe_job_id = "".join(
            ch if ch.isalnum() or ch in "-_." else "_" for ch in str(job_id)
        )
        comp_cfg["cache_dir"] = os.path.join(
            cache_root, safe_job_id, f"replica_{replica_rank}"
        )
        kwargs["compilation_config"] = comp_cfg
        logger.info(
            "ATOMReplicaManager: replica %d torch compile cache_dir=%s",
            replica_rank,
            comp_cfg["cache_dir"],
        )
        return kwargs

    def sleep_all(self, level: int = 2) -> None:
        import ray

        ray.get([s.sleep.remote(level) for s in self.servers])

    def wake_all(self, tags: list[str] | None = None) -> None:
        import ray

        ray.get([s.wake_up.remote(tags) for s in self.servers])

    def drain_all(self) -> None:
        import ray

        ray.get([s.wait_for_requests_to_drain.remote() for s in self.servers])

    def generation_pause_modes(self) -> set[str]:
        """Pause modes every replica takes."""
        import ray

        modes = ray.get([s.generation_pause_modes.remote() for s in self.servers])
        return set.intersection(*(set(m) for m in modes)) if modes else set()

    def pause_all(self, mode: str = "keep", clear_cache: bool = True) -> None:
        import ray

        ray.get([s.pause_generation.remote(mode, clear_cache) for s in self.servers])

    def resume_all(self) -> None:
        import ray

        ray.get([s.resume_generation.remote() for s in self.servers])

    def collective_rpc(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        barrier: bool = False,
        timeout: float = 600.0,
    ) -> list:
        """Run *method* on every rank of every replica.

        Returns one list per replica, replica-ordered, each holding that
        replica's per-rank values. Kept nested rather than flattened so a caller
        checking coverage can still tell which replica a value came from.
        """
        import ray

        return ray.get(
            [
                s.collective_rpc.remote(method, args, kwargs, barrier, timeout)
                for s in self.servers
            ]
        )

    def get_capabilities(self) -> list:
        """Per-replica capabilities.

        Returned per replica rather than merged: replicas are separate engines,
        and a caller that needs a single answer should decide for itself whether
        to intersect them or to treat a disagreement as a configuration error.
        """
        import ray

        return ray.get([s.get_capabilities.remote() for s in self.servers])

    # ── RDMA weight transfer ──────────────────────────────────────────────
    #
    # Mirrors VLLMReplicaManager's interface exactly, so the backend-agnostic
    # _sync_weights_rdma in the trainer drives either one unchanged.

    @property
    def _ranks_per_replica(self) -> int:
        """Ranks one replica contributes to the weight group.

        A replica running DP internally is several engines behind one actor
        handle, and each of their TP ranks joins separately, so this is tp * dp
        rather than tp. Getting it wrong shifts every later replica's base rank
        and the rendezvous hangs with no useful error.
        """
        return self.tensor_parallel_size * self.data_parallel_size

    def init_rdma_weight_group(
        self,
        actor_wg,
        *,
        interface: str,
        hca: str,
        require_rdma: bool,
        timeout_s: int,
        group_name: str,
    ) -> dict[str, Any]:
        """Build one persistent trainer + all-workers RCCL communicator."""
        import ray

        self._assert_workers_can_receive_rdma()

        if require_rdma:
            # Preflight everyone before anybody rendezvouses: a container
            # missing its verbs device would otherwise park all ranks until the
            # timeout, naming none of them.
            checks = [actor_wg.call_single_async(0, "rdma_preflight", interface, hca)]
            checks.extend(
                server.rdma_preflight.remote(interface, hca) for server in self.servers
            )
            logger.info("ATOM RDMA preflight: %s", ray.get(checks))

        rendezvous = actor_wg.execute_rank_zero_sync("get_rdma_rendezvous", interface)
        master_addr = str(rendezvous["address"])
        master_port = int(rendezvous["port"])
        world_size = 1 + self.num_replicas * self._ranks_per_replica

        # Trainer takes rank 0; replica r takes the block starting at
        # 1 + r * ranks_per_replica. Every rank must call in or the rendezvous
        # blocks, so these go out together and are joined as a set.
        refs = [
            actor_wg.call_single_async(
                0,
                "init_rdma_weight_group",
                master_addr,
                master_port,
                world_size,
                group_name,
                timeout_s,
            )
        ]
        for replica_rank, server in enumerate(self.servers):
            refs.append(
                server.init_rdma_weight_group.remote(
                    master_addr,
                    master_port,
                    1 + replica_rank * self._ranks_per_replica,
                    world_size,
                    group_name,
                    timeout_s,
                )
            )
        ray.get(refs)

        self.rdma_group_name = group_name
        logger.info(
            "ATOM RDMA weight group ready: %s master=%s:%d world=%d "
            "(%d replicas x %d ranks + trainer)",
            group_name,
            master_addr,
            master_port,
            world_size,
            self.num_replicas,
            self._ranks_per_replica,
        )
        return {
            "master_addr": master_addr,
            "master_port": master_port,
            "world_size": world_size,
        }

    def _assert_workers_can_receive_rdma(self) -> None:
        """Refuse to build a group the workers cannot serve.

        Uses ATOM's general capability negotiation rather than a bespoke
        handshake. Checked before the rendezvous for the same reason as
        preflight: afterwards the failure is a hang, not a message.
        """
        import ray

        for replica_rank, server in enumerate(self.servers):
            try:
                caps = ray.get(server.get_capabilities.remote())
            except Exception as exc:
                raise RuntimeError(
                    f"ATOM replica {replica_rank} could not report capabilities; "
                    "the engine predates capability discovery, so its RDMA "
                    "support cannot be confirmed"
                ) from exc
            if not caps.supports("rdma_weight_receive"):
                partial = caps.partial()
                hint = (
                    f" (present on some ranks but not all: {sorted(partial)})"
                    if "rdma_weight_receive" in partial
                    else ""
                )
                raise RuntimeError(
                    f"ATOM replica {replica_rank} does not support "
                    f"rdma_weight_receive{hint}; every rank needs the receiver "
                    "for the group to be usable"
                )

    def start_receive_weights_rdma(
        self,
        *,
        version: int,
        verify_full_load: bool,
        prequantized_fp8: bool = False,
        online_staging: str = "gpu",
    ) -> list:
        """Arm every replica's receiver and return the refs un-awaited.

        ``online_staging`` is taken so the trainer drives both managers alike;
        it tunes vLLM's online-FP8 reload, and ATOM's receive is BF16 only.

        Deliberately not awaited here: all receivers and the trainer's sender
        must sit inside the broadcast at once. Joining them in turn would park
        the first receiver waiting for a sender that is itself waiting for this
        call to return.
        """
        if not self.rdma_group_name:
            raise RuntimeError("ATOM RDMA weight group has not been initialized")
        return [
            server.receive_weights_rdma.remote(
                self.rdma_group_name,
                int(version),
                bool(verify_full_load),
                bool(prequantized_fp8),
            )
            for server in self.servers
        ]

    def destroy_rdma_weight_group(self, actor_wg=None) -> None:
        import ray

        if not self.rdma_group_name:
            return
        refs = [
            server.destroy_rdma_weight_group.remote(self.rdma_group_name)
            for server in self.servers
        ]
        if actor_wg is not None:
            refs.append(actor_wg.call_single_async(0, "destroy_rdma_weight_group"))
        try:
            ray.get(refs)
        # RDMA actor teardown must not hide the initiating shutdown failure.
        except Exception as exc:  # noqa: BLE001 -- boundary
            logger.warning("ATOM RDMA group teardown failed: %s", exc)
        self.rdma_group_name = None

    def reload_weights_from_path(self, weight_dir: str) -> None:
        import ray

        ray.get([s.reload_weights_from_path.remote(weight_dir) for s in self.servers])

    def shutdown(self) -> None:
        import ray

        try:
            ray.get([s.shutdown.remote() for s in self.servers])
        # Actor teardown can surface backend exceptions through Ray wrappers.
        except Exception:  # noqa: BLE001,S110 -- best-effort cleanup boundary
            pass
        for s in self.servers:
            try:
                ray.kill(s)
            except Exception:  # noqa: BLE001,S110 -- best-effort cleanup boundary
                pass
        self.servers = []
