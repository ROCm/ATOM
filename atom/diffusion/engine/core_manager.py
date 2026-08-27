# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Spawns and talks to a replica's GPU workers.

``CoreManager``'s role without the token-serving machinery -- no DP load
balancing, no streaming hook, no per-sequence callbacks. One collective group
running one job at a time: spawn N ranks, fan a request to all, drain one
output socket.

One PUSH socket per rank rather than a PUB, because PUB drops to slow joiners
and a rank that misses a job hangs the replica in an all-to-all. One shared PULL
for outputs; only rank 0 reports results, and errors carry their rank.
"""

import logging
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
from typing import Self

import zmq

from atom.diffusion.config import DiffusionConfig
from atom.diffusion.engine.engine_core import DiffusionEngineCore
from atom.diffusion.engine.protocol import (
    EngineOutput,
    EngineRequest,
    OutputType,
    RequestType,
)
from atom.utils import get_open_port, get_open_zmq_ipc_path, make_zmq_socket

logger = logging.getLogger(__name__)

DEFAULT_READY_TIMEOUT_S = 1800.0
"""Loading H3 is ~200 s for the DiT plus the encoder and VAEs, and a cold JIT
cache adds more. Generous on purpose: a timeout here kills a replica that was
merely still loading."""


class DiffusionCoreManager:
    """Owns the worker processes and the sockets to them."""

    def __init__(
        self,
        config: DiffusionConfig,
        *,
        ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S,
    ) -> None:
        self.config = config
        self.ready_timeout_s = ready_timeout_s
        self.outputs: queue.Queue[EngineOutput] = queue.Queue()

        self.ctx = zmq.Context(io_threads=2)
        self._output_address = get_open_zmq_ipc_path()
        self._input_addresses = [
            get_open_zmq_ipc_path() for _ in range(config.num_gpus)
        ]
        self._output_socket = make_zmq_socket(
            self.ctx, self._output_address, zmq.PULL, bind=True
        )
        self._input_sockets = [
            make_zmq_socket(self.ctx, address, zmq.PUSH, bind=True)
            for address in self._input_addresses
        ]

        self.processes: list[mp.process.BaseProcess] = []
        self._closed = False
        self._dead_ranks: dict[int, str] = {}
        self._output_thread = threading.Thread(
            target=self._drain_outputs, name="diffusion-outputs", daemon=True
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn every rank and block until all report READY."""
        # Workers inherit the environment as it is at spawn time, and a missing
        # MASTER_PORT surfaces as every rank hanging in init_process_group.
        env = {
            "MASTER_ADDR": os.environ.get("MASTER_ADDR", "127.0.0.1"),
            "MASTER_PORT": os.environ.get("MASTER_PORT") or str(get_open_port()),
        }
        os.environ.update(env)
        logger.info(
            "worker rendezvous at %s:%s", env["MASTER_ADDR"], env["MASTER_PORT"]
        )

        # spawn, never fork: forking a process that has touched CUDA
        # re-initialises the runtime in the child and crashes.
        context = mp.get_context("spawn")
        for rank in range(self.config.num_gpus):
            process = context.Process(
                target=DiffusionEngineCore.run_worker,
                args=(
                    self.config,
                    rank,
                    self._input_addresses[rank],
                    self._output_address,
                ),
                name=f"diffusion-rank{rank}",
                daemon=True,
            )
            process.start()
            self.processes.append(process)

        self._output_thread.start()
        self._await_ready()

    def _await_ready(self) -> None:
        pending = set(range(self.config.num_gpus))
        deadline = time.monotonic() + self.ready_timeout_s
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"ranks {sorted(pending)} did not become ready within "
                    f"{self.ready_timeout_s:.0f}s"
                )
            try:
                output = self.outputs.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                self._assert_workers_alive()
                continue
            if output.type is OutputType.READY:
                pending.discard(output.rank)
                logger.info(
                    "diffusion rank %d ready (%d/%d)",
                    output.rank,
                    self.config.num_gpus - len(pending),
                    self.config.num_gpus,
                )
            elif output.type is OutputType.DEAD:
                raise RuntimeError(
                    f"rank {output.rank} died during startup: {output.error}"
                )
            else:
                # A stray output before everyone is up is not fatal, but it
                # means a rank started work early -- worth seeing.
                logger.warning("unexpected %s before ready", output.type)

    def _assert_workers_alive(self) -> None:
        for rank, process in enumerate(self.processes):
            if not process.is_alive() and rank not in self._dead_ranks:
                raise RuntimeError(
                    f"rank {rank} exited with code {process.exitcode} before "
                    "reporting ready"
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.send(EngineRequest(type=RequestType.SHUTDOWN))
        except Exception as exc:  # noqa: BLE001 - shutdown is best-effort
            logger.debug("shutdown broadcast failed: %s", exc)

        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                # A worker parked in a collective will not see the shutdown
                # message; killing it is the only way out, and leaving it
                # alive holds ~100 GB of VRAM.
                logger.warning("killing unresponsive %s", process.name)
                process.kill()
                process.join(timeout=10)

        for socket in [*self._input_sockets, self._output_socket]:
            socket.close(linger=0)
        self.ctx.term()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------
    # messaging
    # ------------------------------------------------------------------

    def send(self, request: EngineRequest) -> None:
        """Fan one request out to every rank."""
        payload = pickle.dumps(request)
        for socket in self._input_sockets:
            socket.send(payload)

    def _drain_outputs(self) -> None:
        while True:
            try:
                message = self._output_socket.recv()
            except zmq.error.ContextTerminated:
                return
            except zmq.ZMQError:
                return
            output: EngineOutput = pickle.loads(message)
            if output.type is OutputType.DEAD:
                self._dead_ranks[output.rank] = output.error or "unknown"
            self.outputs.put(output)

    @property
    def dead_ranks(self) -> dict[int, str]:
        return dict(self._dead_ranks)
