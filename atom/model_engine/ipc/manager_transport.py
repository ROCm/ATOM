"""Preparation budgets and prompt leases for the Manager's existing sockets."""

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .buffer_types import BufferLease
from .control_codec import ControlCodec
from .prompt_arena import ArenaCapacityError, PromptArenaOwner
from .request_codec import RequestCodec
from .result_codec import ResultCodec

logger = logging.getLogger("atom")


@dataclass
class PreparedRequest:
    rank: int
    requests: list
    message: object
    pins: list = field(default_factory=list)
    consumers: list = field(default_factory=list)
    sent: bool = False
    engine_id: str = ""
    tracker: object = None

    def release(self):
        for lease in self.pins:
            lease.release()
        self.pins.clear()
        if not self.sent:
            for lease in self.consumers:
                lease.release()
        self.consumers.clear()
        self.message = None
        self.requests.clear()


class ManagerTransport:
    def __init__(self, config, *, enable_shm=True):
        self.config = config
        self.session_id = uuid.uuid4().hex
        self.peers = {}
        self.lock = threading.RLock()
        self.pending_jobs = self.pending_bytes = 0
        self.closed = False
        self._counter = 0
        self._sends = []
        self._consumers = {}
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="AtomIPC")
        options = {
            "max_header_bytes": config.ipc_max_header_bytes,
            "max_batch_size": config.ipc_max_batch_size,
            "max_batch_bytes": getattr(config, "ipc_max_batch_bytes", 512 << 20),
        }
        self.request = RequestCodec(**options)
        self.control = ControlCodec(**options)
        self.result = ResultCodec(**options)
        self.arena = None
        self.probe = None
        if enable_shm and config.ipc_transport != "raw":
            try:
                capacity = config.ipc_shm_capacity_bytes
                if capacity is None:
                    capacity = PromptArenaOwner.capacity_for_requests(
                        config.max_num_seqs
                        * config.parallel_config.data_parallel_size_local,
                        config.max_model_len,
                    )
                self.arena = PromptArenaOwner(config.ipc_shm_dir, capacity)
                self.probe = self.arena.create_probe()
                logger.info("Prompt SHM pool: %.1f MiB", capacity / (1024 * 1024))
            except OSError as exc:
                if self.arena is not None:
                    self.arena.close()
                    self.arena = None
                self.probe = None
                if config.ipc_transport == "shm":
                    raise
                logger.info("Prompt SHM unavailable; using raw frames: %s", exc)

    def next_id(self):
        with self.lock:
            self._counter += 1
            return self._counter

    def context(self, rank):
        return {
            "session_id": self.session_id,
            "engine_id": self.peers[rank]["engine_id"],
            "message_id": self.next_id(),
        }

    def probe_payload(self, rank):
        payload = {"probe_id": f"{self.session_id}:{rank}"}
        if self.arena is not None:
            payload.update(registration=self.arena.registration, buffer=self.probe[0])
        return payload

    def accept_probe(self, rank, payload):
        if payload["probe_id"] != f"{self.session_id}:{rank}":
            raise ValueError("unexpected shared memory probe reply")
        peer = self.peers[rank]
        if peer.get("ready"):
            raise ValueError("duplicate shared memory probe reply")
        modes = peer.get("buffer_modes", ("raw", "shm"))
        accessible = bool(
            self.probe is not None
            and payload.get("accessible")
            and payload.get("digest") == self.probe[1]
            and payload.get("arena_epoch") == self.arena.epoch
            and peer.get("same_host", False)
            and "shm" in modes
        )
        if self.config.ipc_transport == "shm" and not accessible:
            raise RuntimeError(f"rank {rank} cannot access the prompt arena")
        if not accessible and "raw" not in modes:
            raise RuntimeError(f"rank {rank} does not support raw prompt frames")
        self.peers[rank]["shm"] = accessible
        self.peers[rank]["ready"] = True

    @staticmethod
    def input_bytes(requests):
        seen = set()
        tensors = set()
        total = 0
        for request in requests:
            identity = request.prompt.buffer_identity
            if identity not in seen:
                seen.add(identity)
                total += request.prompt.nbytes
            for value in [
                *(request.multimodal_data or {}).values(),
                request.mrope_positions,
            ]:
                if id(value) in tensors:
                    continue
                tensors.add(id(value))
                if hasattr(value, "numel"):
                    total += value.numel() * value.element_size()
                elif hasattr(value, "nbytes"):
                    total += value.nbytes
        return total

    def reserve(self, nbytes):
        if nbytes < 0:
            raise ValueError("IPC preparation byte count must be nonnegative")
        if nbytes > self.config.ipc_max_pending_bytes:
            raise ValueError("admission exceeds the IPC preparation byte limit")
        self.reap()
        with self.lock:
            if self.closed:
                raise RuntimeError("IPC transport is closed")
            if (
                self.pending_jobs >= self.config.ipc_max_pending_jobs
                or self.pending_bytes + nbytes > self.config.ipc_max_pending_bytes
            ):
                return False
            self.pending_jobs += 1
            self.pending_bytes += nbytes
            return True

    def unreserve(self, nbytes):
        with self.lock:
            self.pending_jobs -= 1
            self.pending_bytes -= nbytes

    def prepare(self, routes):
        prepared = []
        sources = {}
        try:
            # Seal each shared source once, for all of its destinations.
            with self.lock:
                if self.closed:
                    raise RuntimeError("IPC transport is closed")
                peers = {rank: dict(self.peers[rank]) for rank, _ in routes}
            for rank, requests in routes:
                if not peers[rank].get("ready"):
                    raise RuntimeError(f"IPC peer {rank} is not ready")
                for request in requests:
                    handle = request.prompt
                    identity = handle.buffer_identity
                    if identity not in sources:
                        sources[identity] = [handle.acquire(), False]
                    sources[identity][1] |= peers[rank]["shm"]
            for handle, local in sources.values():
                use_shm = local and (
                    self.config.ipc_transport == "shm"
                    or handle.nbytes >= self.config.ipc_shm_threshold_bytes
                )
                if handle.sealed:
                    if self.config.ipc_transport == "shm" and handle.descriptor is None:
                        raise ValueError(
                            "forced SHM cannot use a prompt already sealed as raw"
                        )
                    continue
                try:
                    handle.seal(self.arena if use_shm else None)
                except ArenaCapacityError:
                    if self.config.ipc_transport != "auto":
                        raise
                    # Queued requests can outnumber the running request limit.
                    handle.seal()
            for rank, requests in routes:
                entry = PreparedRequest(
                    rank, list(requests), None, engine_id=peers[rank]["engine_id"]
                )
                prepared.append(entry)
                refs = {}
                seen = set()
                for request in requests:
                    handle = request.prompt
                    if handle.descriptor is not None and peers[rank]["shm"]:
                        refs[request.id] = handle.descriptor
                    if handle.buffer_identity not in seen:
                        seen.add(handle.buffer_identity)
                        if self.arena is not None:
                            entry.pins.append(self.arena.acquire_send_pin(handle))
                        else:
                            entry.pins.append(handle.acquire())
                entry.message = self.request.encode(
                    "ADD",
                    requests,
                    buffer_refs=refs,
                    session_id=self.session_id,
                    engine_id=entry.engine_id,
                    message_id=self.next_id(),
                )
            return prepared
        except BaseException:
            for entry in prepared:
                entry.release()
            raise
        finally:
            for handle, _ in sources.values():
                handle.release()

    def activate(self, entry):
        with self.lock:
            peer = self.peers.get(entry.rank, {})
            if (
                self.closed
                or not peer.get("ready")
                or peer.get("engine_id") != entry.engine_id
            ):
                raise RuntimeError("IPC peer changed while preparing the admission")
            if entry.consumers:
                return
            for request in entry.requests:
                handle = request.prompt
                if handle.descriptor is None or not peer["shm"]:
                    continue
                key = (
                    self.session_id,
                    self.peers[entry.rank]["engine_id"],
                    request.id,
                    "request",
                )
                lease = self.arena.acquire_consumer(handle, key)
                entry.consumers.append(lease)
                self._consumers[entry.rank, request.id] = (handle.descriptor, key)

    def release_consumer(self, rank, payload):
        with self.lock:
            key = (rank, payload["request_id"])
            record = self._consumers.get(key)
            if record is None:
                return
            descriptor, consumer = record
            if (
                descriptor.arena_id != payload["arena_id"]
                or descriptor.epoch != payload["arena_epoch"]
                or descriptor.allocation_id != payload["allocation_id"]
                or descriptor.generation != payload["allocation_generation"]
            ):
                return
            self._consumers.pop(key)
            self.arena.release_consumer(consumer, descriptor)

    def retain(self, tracker, message, pins=()):
        with self.lock:
            self._sends.append((tracker, message, list(pins)))
        self.reap()

    def defer_unreserve(self, nbytes, trackers):
        self.retain(
            tuple(trackers), None, [BufferLease(lambda: self.unreserve(nbytes))]
        )

    def reap(self):
        with self.lock:
            pending = []
            for tracker, message, pins in self._sends:
                done = (
                    all(item is None or item.done for item in tracker)
                    if isinstance(tracker, tuple)
                    else tracker is None or tracker.done
                )
                if not done:
                    pending.append((tracker, message, pins))
                else:
                    for pin in pins:
                        pin.release()
            self._sends = pending

    def forget_unsent(self, entry):
        with self.lock:
            for request in entry.requests:
                self._consumers.pop((entry.rank, request.id), None)
        entry.release()

    def close(self, dead_engines):
        self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.reap()
        if self.arena is not None:
            with self.lock:
                dead_engines = set(dead_engines)
                for engine_id in dead_engines:
                    self.arena.reclaim_dead_engine(engine_id)
                self._consumers = {
                    key: record
                    for key, record in self._consumers.items()
                    if record[1][1] not in dead_engines
                }
            try:
                self.arena.close()
            except (BufferError, RuntimeError):
                # A live peer or a queued send still owns this allocation.
                logger.warning("Prompt arena remains leased during shutdown")
