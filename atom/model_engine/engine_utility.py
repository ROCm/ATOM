# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import queue
import uuid
from typing import ClassVar

from atom.model_engine.collective_rpc import COLLECTIVE_RPC_CMD, RpcPayload
from atom.model_engine.sequence import SequenceStatus
from atom.utils import envs

logger = logging.getLogger("atom")

# Commands whose senders never wait, so their handlers answer nothing and a
# synchronous caller could only wait out its timeout on them.
FIRE_AND_FORGET_UTILITY_CMDS = frozenset({"abort_request", "get_mtp_stats"})
WEIGHT_UPDATE_UTILITY_CMDS = frozenset(
    {"update_weights", "update_weights_shm", "update_weights_ipc"}
)

# For a direct weight update whose sender gives no deadline of its own;
# broadcast_utility_command_sync always does.
_DIRECT_UPDATE_TIMEOUT_S = 300.0


class _ReplyQueue:
    """The EngineCore output queue, as the utility handlers see it.

    Stamps every ``UTILITY_RESPONSE`` with the request id of the command it
    answers, so CoreManager matches replies to callers by id rather than by
    position, where a late or unasked-for reply became the next caller's.
    """

    def __init__(self, output_queue):
        self._queue = output_queue
        self.request_id = None

    def put_nowait(self, item) -> None:
        if self.request_id is not None and item[0] == "UTILITY_RESPONSE":
            item = ("UTILITY_RESPONSE", {**item[1], "request_id": self.request_id})
        self._queue.put_nowait(item)


class EngineUtilityHandler:
    """Centralised handler for all utility commands dispatched by EngineCore.

    Covers weight management, memory lifecycle, profiling, MTP statistics,
    and TorchSpec hidden-state extraction.  Every command is registered in
    ``_UTILITY_HANDLERS`` and executed in the main busy-loop thread so that
    ``runner_mgr.call_func`` calls are serialized.

    Parameters
    ----------
    runner_mgr : AsyncIOProcManager
        The model-runner process manager used to execute ``call_func``.
    output_queue : queue.Queue
        The EngineCore output queue for pushing ``UTILITY_RESPONSE`` messages
        back to ``CoreManager``.
    label : str, optional
        Label used in log messages (default ``"Engine Core"``).
    scheduler : Scheduler, optional
        The scheduler instance, needed by MTP statistics handlers.
    """

    # Utility command name  ->  handler method name
    _UTILITY_HANDLERS: ClassVar[dict[str, str]] = {
        "update_weights": "_handle_update_weights",
        "update_weights_shm": "_handle_update_weights_shm",
        "update_weights_ipc": "_handle_update_weights_ipc",
        "discard_failed_weight_sync": "_handle_discard_failed_weight_sync",
        "release_memory": "_handle_release_memory",
        "resume_memory": "_handle_resume_memory",
        "clear_kv_cache": "_handle_clear_kv_cache",
        "configure_hidden_states": "_handle_configure_hidden_states",
        "start_profile": "_handle_start_profile",
        "stop_profile": "_handle_stop_profile",
        "get_mtp_stats": "_handle_get_mtp_stats",
        "get_mtp_statistics": "_handle_get_mtp_statistics",
        "get_cache_statistics": "_handle_get_cache_statistics",
        "abort_request": "_handle_abort_request",
        COLLECTIVE_RPC_CMD: "_handle_collective_rpc",
    }

    def __init__(
        self, runner_mgr, output_queue, label: str = "Engine Core", scheduler=None
    ):
        self.runner_mgr = runner_mgr
        self.output_queue = _ReplyQueue(output_queue)
        self.label = label
        self.scheduler = scheduler

    def process_queue(self, utility_queue, engine):
        """Drain *utility_queue* and execute each command.

        When the queue is empty, ``engine._has_pending_utility`` is set to
        ``False`` so that the next busy-loop iteration can skip the check.

        Sleep/wake state is tracked on *engine._is_rl_weights_offloaded* so that the
        busy-loop can skip model execution while the weights are offloaded. A
        bucketed weight sync wakes the engine only if its last bucket was
        applied on every rank.
        """
        if not engine._has_pending_utility:
            return

        while True:
            try:
                cmd, args = utility_queue.get_nowait()
                reply = self._execute_utility_command(cmd, args)
                # Track sleep/wake transitions
                if cmd == "release_memory":
                    tags = args.get("tags", []) if isinstance(args, dict) else []
                    if "weights" in tags:
                        engine._is_rl_weights_offloaded = True
                        logger.info(f"{self.label}: engine entered sleep mode")
                elif cmd == "resume_memory":
                    tags = args.get("tags", []) if isinstance(args, dict) else []
                    if "weights" in tags:
                        if getattr(engine, "_rl_weights_inconsistent", False):
                            engine._is_rl_weights_offloaded = True
                            logger.error(
                                f"{self.label}: refusing to resume inconsistent "
                                f"weights; a complete weight sync must succeed first"
                            )
                        else:
                            engine._is_rl_weights_offloaded = False
                            logger.info(f"{self.label}: engine exited sleep mode")
                elif cmd in WEIGHT_UPDATE_UTILITY_CMDS:
                    failed = isinstance(reply, dict) and bool(reply.get("error"))
                    is_complete = cmd == "update_weights" or (
                        args.get("is_last", True) if isinstance(args, dict) else True
                    )
                    if failed:
                        engine._rl_weights_inconsistent = True
                        engine._is_rl_weights_offloaded = True
                        logger.error(
                            f"{self.label}: weight update failed; serving stays "
                            f"fenced until a complete sync succeeds: {reply['error']}"
                        )
                    elif is_complete:
                        engine._rl_weights_inconsistent = False
                        engine._is_rl_weights_offloaded = False
                        logger.info(
                            f"{self.label}: engine exited sleep mode (weights updated)"
                        )
            except queue.Empty:
                engine._has_pending_utility = False
                break

    def _execute_utility_command(self, cmd: str, args: dict) -> dict | None:
        """Run *cmd*'s handler and return what it returned: the weight
        updates hand back the reply they sent."""
        import time as _time

        log = logger.info
        log(f"{self.label}: executing utility command: {cmd}")
        t0 = _time.monotonic()

        reply = None
        handler_name = self._UTILITY_HANDLERS.get(cmd)
        self.output_queue.request_id = (
            args.get("request_id") if isinstance(args, dict) else None
        )
        try:
            if handler_name:
                handler = getattr(self, handler_name)
                try:
                    reply = handler(args)
                except Exception as exc:
                    # Still fatal to the engine, as before; but a synchronous
                    # caller now hears why instead of waiting out its timeout.
                    if cmd not in FIRE_AND_FORGET_UTILITY_CMDS:
                        self.output_queue.put_nowait(
                            ("UTILITY_RESPONSE", self._error_reply(cmd, exc))
                        )
                    raise
            else:
                # Answer, do not just log: a synchronous caller would otherwise
                # wait out its timeout and learn only that, not the misspelling.
                logger.warning(f"{self.label}: Unknown utility command: {cmd}")
                self.output_queue.put_nowait(
                    (
                        "UTILITY_RESPONSE",
                        {"cmd": cmd, "error": f"unknown utility command {cmd!r}"},
                    )
                )
        finally:
            self.output_queue.request_id = None

        elapsed = _time.monotonic() - t0
        log(f"{self.label}: utility command '{cmd}' finished in {elapsed:.2f}s")
        return reply

    @staticmethod
    def _error_reply(cmd: str, exc: Exception) -> dict:
        return {"cmd": cmd, "error": f"{type(exc).__name__}: {exc}"}

    def _handle_collective_rpc(self, args: dict):
        """Invoke an arbitrary ModelRunner method on every TP rank.

        Runs in the EngineCore busy loop, so this DP rank stops scheduling for
        the call's duration. That is wanted for a weight swap, but it does mean
        a caller passing a long timeout is deliberately stalling generation.
        """
        method = args.get("method")
        request_id = args.get("request_id")
        # Checked here, before the broadcast: every TP worker resolves the name
        # with getattr, which raises TypeError on anything but a string, and the
        # manager's output thread routes replies with the id as a dict key.
        if (
            not isinstance(method, str)
            or not method
            or not isinstance(request_id, str)
            or not request_id
        ):
            self.output_queue.put_nowait(
                (
                    "UTILITY_RESPONSE",
                    {
                        "cmd": COLLECTIVE_RPC_CMD,
                        "request_id": request_id,
                        "error": "collective_rpc needs a method name and a request "
                        "id, both non-empty strings",
                    },
                )
            )
            return

        try:
            payload = RpcPayload(
                request_id=request_id,
                args=tuple(args.get("args", ())),
                kwargs=dict(args.get("kwargs") or {}),
                barrier=bool(args.get("barrier", False)),
            )
            replies = self.runner_mgr.collective_rpc(
                method, payload, timeout=float(args.get("timeout", 300.0))
            )
        except Exception as exc:  # noqa: BLE001 - reported, never raised at the loop
            # Raising here would kill the EngineCore busy loop and take the
            # engine down with it; the caller gets the reason instead.
            self.output_queue.put_nowait(
                (
                    "UTILITY_RESPONSE",
                    {
                        "cmd": COLLECTIVE_RPC_CMD,
                        "request_id": request_id,
                        "method": method,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            )
            return

        failures = [r for r in replies if not r.ok]
        logger.info(
            f"{self.label}: collective_rpc {method} ranks={len(replies)} "
            f"failed={len(failures)}"
        )
        self.output_queue.put_nowait(
            (
                "UTILITY_RESPONSE",
                {
                    "cmd": COLLECTIVE_RPC_CMD,
                    "request_id": request_id,
                    "method": method,
                    "tp_world_size": self.runner_mgr.proc_num,
                    "results": [
                        {
                            "tp_rank": r.tp_rank,
                            "value": r.value,
                            "error": r.error,
                        }
                        for r in replies
                    ],
                },
            )
        )

    def _update_on_every_rank(
        self, cmd: str, method: str, args: dict, *call_args, barrier: bool = False
    ) -> dict:
        """Run a direct weight update on every TP rank; the reply for all.

        ``call_func`` returns rank 0's result alone, so a rank that rejected a
        tensor still reported success and the caller went on with a partly
        updated model -- if that rank's raise had not ended its worker. Through
        the generic path every rank answers on its own channel, a failure is
        caught where it happens, and success means every rank succeeded.

        Every rank is sent the same tensors, so every rank should update the
        same number of parameters: one that updated fewer skipped what its
        peers wrote, and the shards no longer belong to one model.

        *barrier* holds each rank at the worker barrier until every rank has
        finished, so none moves on while another still reads the caller's
        shared buffer.
        """
        payload = RpcPayload(
            request_id=f"{cmd}-{uuid.uuid4().hex}", args=call_args, barrier=barrier
        )
        timeout = args.get("timeout", _DIRECT_UPDATE_TIMEOUT_S)
        try:
            replies = self.runner_mgr.utility_rpc(method, payload, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - answered, never raised at the loop
            self._discard_failed_update_on_every_rank(cmd, timeout)
            return self._error_reply(cmd, exc)
        failed = [r for r in replies if not r.ok]
        if failed:
            error = "; ".join(f"TP rank {r.tp_rank}: {r.error}" for r in failed)
        elif any(r.value != replies[0].value for r in replies[1:]):
            error = "TP ranks updated different numbers of parameters: " + ", ".join(
                f"rank {r.tp_rank} updated {r.value}" for r in replies
            )
        else:
            result = replies[0].value
            logger.info(
                f"{self.label}: {cmd} completed on every TP rank, updated={result}"
            )
            return {"cmd": cmd, "result": result}
        self._discard_failed_update_on_every_rank(cmd, timeout)
        logger.error(f"{self.label}: {cmd} failed: {error}")
        return {"cmd": cmd, "error": error}

    def _discard_failed_update_on_every_rank(self, cmd: str, timeout: float) -> dict:
        """Keep no rank's scratch after this engine's update failed."""
        payload = RpcPayload(request_id=f"{cmd}-discard-{uuid.uuid4().hex}")
        try:
            replies = self.runner_mgr.utility_rpc(
                "discard_failed_weight_sync", payload, timeout=timeout
            )
            failed = [r for r in replies if not r.ok]
            if failed:
                error = (
                    f"{cmd} cleanup failed on {len(failed)}/{len(replies)} "
                    f"TP rank(s)"
                )
                logger.error(f"{self.label}: {error}")
                return {"cmd": "discard_failed_weight_sync", "error": error}
        except Exception as exc:
            logger.exception(f"{self.label}: {cmd} cleanup could not be broadcast")
            return self._error_reply("discard_failed_weight_sync", exc)
        return {"cmd": "discard_failed_weight_sync", "result": True}

    def _handle_discard_failed_weight_sync(self, args: dict) -> dict:
        """Clear abandoned update scratch on every TP rank of this engine."""
        reply = self._discard_failed_update_on_every_rank(
            args.get("failed_cmd", "weight update"),
            args.get("timeout", _DIRECT_UPDATE_TIMEOUT_S),
        )
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def _handle_update_weights(self, args: dict) -> dict:
        """Handle direct weight update command."""
        reply = self._update_on_every_rank(
            "update_weights",
            "update_weights",
            args,
            args.get("named_tensors", []),
            args.get("flush_cache", True),
        )
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def _handle_update_weights_shm(self, args: dict) -> dict:
        """Handle shared-memory weight update command.

        Only lightweight metadata (shm_name, bucket_meta) travels through the
        control path.  The actual tensor data resides in POSIX shared memory and
        is read directly by each ModelRunner process.

        After all ModelRunners finish, a UTILITY_RESPONSE is pushed onto the
        output_queue so that the caller (LLMEngine) can synchronise.

        After completion, a ``UTILITY_RESPONSE`` is pushed so the caller
        (LLMEngine) can synchronise.
        """
        reply = self._update_on_every_rank(
            "update_weights_shm",
            "update_weights_from_shm",
            args,
            args.get("shm_name", ""),
            args.get("bucket_meta", {}),
            args.get("is_last", True),
            barrier=True,
        )
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def _handle_update_weights_ipc(self, args: dict) -> dict:
        """Handle CUDA IPC weight update command.

        The caller (LLMEngine) sends a CUDA IPC handle pointing to a GPU
        buffer that already contains the weight data. Each ModelRunner
        sub-process uses ``rebuild_ipc_handle()`` to map the same GPU memory
        and reads weights directly — no CPU round-trip.

        When ``ipc_handles`` (per-GPU dict) is present, each ModelRunner
        opens only its own GPU's handle — always same-GPU IPC, safe on ROCm.
        """
        reply = self._update_on_every_rank(
            "update_weights_ipc",
            "update_weights_from_ipc",
            args,
            args.get("ipc_handle"),
            args.get("bucket_meta", {}),
            args.get("is_last", True),
            args.get("ipc_handles"),
            barrier=True,
        )
        self.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    def _handle_release_memory(self, args: dict):
        """Handle memory release command (sleep mode)."""
        tags = args.get("tags", ["weights", "kv_cache"])
        result = self.runner_mgr.call_func("release_memory", tags, wait_out=True)
        logger.info(f"{self.label}: release_memory completed, tags={tags}")
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "release_memory", "result": result})
        )

    def _handle_resume_memory(self, args: dict):
        """Handle memory resume command (wake up mode)."""
        tags = args.get("tags", ["weights", "kv_cache"])
        result = self.runner_mgr.call_func("resume_memory", tags, wait_out=True)
        logger.info(f"{self.label}: resume_memory completed, tags={tags}")
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "resume_memory", "result": result})
        )

    def _handle_clear_kv_cache(self, args: dict):
        """Handle KV cache clear command."""
        # Use wait_out=True to ensure the GPU zero_() kernel completes before
        # any subsequent release_memory call can modify memory mappings.
        result = self.runner_mgr.call_func("clear_kv_cache", wait_out=True)
        logger.info(f"{self.label}: KV cache cleared")
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "clear_kv_cache", "result": result})
        )

    def _handle_abort_request(self, args: dict):
        """Cancel queued work promptly; running forwards finish via postprocess."""
        req_id = args.get("req_id") if isinstance(args, dict) else None
        if req_id is None or self.scheduler is None:
            return
        abort = getattr(self.scheduler, "abort_request", None)
        if callable(abort):
            found = abort(req_id)
        else:
            found = False
            for seq in list(self.scheduler.running) + list(self.scheduler.waiting):
                if seq.id == req_id:
                    seq.status = SequenceStatus.ABORTED
                    found = True
        logger.info(f"{self.label}: abort_request req_id={req_id} found={found}")

    def _handle_configure_hidden_states(self, args: dict):
        """Configure hidden states extraction on all model runners (TorchSpec)."""
        aux_layer_ids = args.get("aux_layer_ids", [])
        mooncake_config = args.get("mooncake_config", {})
        result = self.runner_mgr.call_func(
            "configure_hidden_states", aux_layer_ids, mooncake_config, wait_out=True
        )
        logger.info(
            f"{self.label}: configure_hidden_states completed, "
            f"aux_layers={aux_layer_ids}"
        )
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "configure_hidden_states", "result": result})
        )

    # ------------------------------------------------------------------
    # Profiler
    # ------------------------------------------------------------------

    def _handle_start_profile(self, args: dict):
        result = self.runner_mgr.call_func("start_profiler", wait_out=True)
        # Flip the scheduler flag so per-iteration detailed aggregates
        # (compute_detailed_aggregates) are emitted while profiling is active.
        if self.scheduler is not None:
            self.scheduler.profile_active = True
        logger.info(f"{self.label}: profiler started")
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "start_profile", "result": result})
        )

    def _handle_stop_profile(self, args: dict):
        logger.info(f"{self.label}: stopping profiler...")
        result = self.runner_mgr.call_func("stop_profiler", wait_out=True)
        if self.scheduler is not None:
            self.scheduler.profile_active = False
        logger.info(f"{self.label}: profiler stopped, result={result}")
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "stop_profile", "result": result})
        )

    # ------------------------------------------------------------------
    # MTP statistics
    # ------------------------------------------------------------------

    def _handle_get_mtp_stats(self, args: dict):
        """Print MTP statistics to log (fire-and-forget)."""
        stats = None if self.scheduler is None else self.scheduler.engine_stats
        if stats is not None and stats.spec_enabled:
            stats.log_spec()
        else:
            logger.info(
                "\n[MTP Stats] No MTP statistics available "
                "(MTP not enabled or no tokens processed)\n"
            )

    def _handle_get_mtp_statistics(self, args: dict):
        """Return structured MTP statistics via UTILITY_RESPONSE."""
        stats = None if self.scheduler is None else self.scheduler.engine_stats
        if stats is None or not stats.spec_enabled:
            result = {"enabled": False}
        else:
            result = stats.spec_statistics()
            result["enabled"] = True
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "get_mtp_statistics", "result": result})
        )

    # ------------------------------------------------------------------
    # Prefix cache statistics
    # ------------------------------------------------------------------

    def _handle_get_cache_statistics(self, args: dict):
        """Return structured prefix-cache statistics via UTILITY_RESPONSE.

        Same counters the periodic `[Cache Stats]` log line reports, on demand
        instead of every hundredth request — a client measuring reuse over a
        handful of requests cannot wait for that interval, and reading it out
        of a log is not something a client can do at all.
        """
        stats = None if self.scheduler is None else self.scheduler.engine_stats
        if stats is None or not stats.cache_enabled:
            result = {"enabled": False}
        else:
            result = stats.cache_statistics()
            result["enabled"] = True
            # The cache section counts the reuse a request wanted and did not
            # get; the funnel is where it was lost.
            result |= self.scheduler.block_manager.checkpoint_funnel()
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "get_cache_statistics", "result": result})
        )

    def push_metrics(self, *, scheduler_metrics: bool = True) -> None:
        """Publish this rank's metrics snapshot on the output socket.

        Pushed on the engine's own clock rather than answered on demand. The
        pull version was a synchronous round trip with a 5s deadline fired every
        5s from the API server; whenever the engine was busy -- a long prefill,
        a GEMM autotune, a large batch -- it could not answer in time, so under
        load it failed on essentially every attempt, buried the server log in
        tracebacks, and left late replies in the response queue for the *next*
        caller to mistake for its own. Pushing removes the deadline, and with it
        the last off-loop writer on the control socket.
        """
        # Poll ready device events even after the final forward. Workers write
        # native metrics directly; this RPC has no response payload.
        if envs.ATOM_ENABLE_METRICS_DEVICE_TIMER and self.runner_mgr is not None:
            self.runner_mgr.call_func("poll_forward_metrics")
        if scheduler_metrics:
            self.output_queue.put_nowait(("METRICS", self.collect_metrics()))

    def collect_metrics(self) -> dict:
        """One rank's scheduler, KV, MTP, and cache metrics."""
        if self.scheduler is None:
            result = {"enabled": False}
        else:
            running, waiting = self.scheduler.get_request_counts()
            # None on the P/D prefill side, which owns no blocks — the decode
            # process does. Its snapshot then carries no kv_blocks_* keys at
            # all rather than a fabricated empty pool; the aggregator sums with
            # `.get(key, 0)`, so the decode rank's real figures come through
            # unchanged.
            block_manager = getattr(self.scheduler, "block_manager", None)
            kv_pool = None if block_manager is None else block_manager.kv
            kv_connector = getattr(self.scheduler, "kv_connector", None)

            engine_stats = self.scheduler.engine_stats
            if not engine_stats.spec_enabled:
                mtp = {"enabled": False}
            else:
                mtp = {"enabled": True, **engine_stats.spec_statistics()}

            if not engine_stats.cache_enabled:
                cache = {"enabled": False}
            else:
                cache = {
                    "enabled": True,
                    **engine_stats.cache_statistics(),
                    **self.scheduler.block_manager.checkpoint_funnel(),
                }

            offload = (
                kv_connector.get_statistics()
                if kv_connector is not None and hasattr(kv_connector, "get_statistics")
                else {}
            )
            result = {
                "enabled": True,
                # "prefill" / "decode" / "" — lets the aggregator recognise a
                # P/D pair, where one request is held by both ranks at once.
                "role": getattr(self.scheduler, "_METRICS_ROLE", ""),
                "requests_running": running,
                "requests_waiting": waiting,
                "requests_parked_kv_load": int(
                    getattr(self.scheduler, "_num_parked_remote_kv", 0)
                ),
                "requests_partial_prefill": int(
                    getattr(self.scheduler, "_partial_prefill_count", 0)
                ),
                "requests_finished": int(
                    getattr(self.scheduler, "total_finished_requests", 0)
                ),
                "prompt_tokens": int(getattr(self.scheduler, "total_prompt_tokens", 0)),
                "generation_tokens": int(
                    getattr(self.scheduler, "total_generation_tokens", 0)
                ),
                "preemptions": int(getattr(self.scheduler, "total_preemptions", 0)),
                "mtp": mtp,
                "cache": cache,
                "offload": offload,
            }
            if kv_pool is not None:
                reusable = kv_pool.num_reusable_free
                result |= {
                    "kv_blocks_used": kv_pool.num_used,
                    "kv_blocks_free": kv_pool.num_free,
                    "kv_blocks_total": kv_pool.num_blocks,
                    "kv_blocks_indexed": kv_pool.num_indexed,
                    "kv_blocks_evictable": reusable,
                    "kv_blocks_vacant": kv_pool.num_free - reusable,
                }

            metrics = getattr(self.scheduler, "metrics", None)
            if metrics is not None:
                parked = sum(
                    seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
                    for seq in self.scheduler.waiting
                )
                # Shared-cache disaggregation has a separate prefill queue,
                # whereas connector-based PD parks requests in `waiting`.
                external = parked + len(getattr(self.scheduler, "prefill_waiting", ()))
                result["scheduler_metrics"] = {
                    "running": running,
                    "waiting": max(0, waiting - external),
                    "waiting_kv": external,
                }

        return result
