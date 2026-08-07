# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Stream online quantization module by module during checkpoint loading.

Eligible parameters start on meta, stage arrivals on the host, then move to the
device, quantize, and release their source storage once complete.
"""

import concurrent.futures
import logging
import threading

import torch
import torch.utils._python_dispatch
from torch import nn

from atom.utils import envs

logger = logging.getLogger("atom")

_HOST = torch.device("cpu")


class _CopyCounter(torch.utils._python_dispatch.TorchDispatchMode):
    """Count elements written by ``aten.copy_`` in the current thread."""

    def __init__(self):
        super().__init__()
        self.copied_numel = 0
        self.moe_arrival = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.ops.aten.copy_.default:
            assert args[0].numel() == args[1].numel()
            self.copied_numel += args[0].numel()
        return func(*args, **kwargs)


class OnlineQuantStreamer:
    """Per-load state for streaming online quantization."""

    @classmethod
    def maybe_create(cls, model: nn.Module, load_dummy: str | None):
        """Return a streamer when enabled and applicable."""
        if not envs.ATOM_ONLINE_QUANT_STREAMING or load_dummy:
            return None
        candidates = [
            (mod_name, m)
            for mod_name, m in model.named_modules()
            if getattr(m, "_stream_online_quant", False)
        ]
        if not candidates:
            return None
        return cls(candidates)

    def __init__(self, candidates: list[tuple[str, nn.Module]]):
        self.candidates = candidates
        self.param_to_module: dict[int, nn.Module] = {}
        # Claimed modules no longer accept loader arrivals.
        self.done_module_ids: set[int] = set()
        self.excessive_loads: list[str] = []

        self._host_staging = envs.ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING
        # Module claims are decided under each module's lock.
        self._done_lock = threading.Lock()
        self._params_dict: dict | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._futures: list = []
        # Bound the executor's otherwise-unbounded queue and its memory usage.
        self._slots: threading.Semaphore | None = None
        # Side streams avoid serializing workers on the default stream.
        self._worker_stream = threading.local()

        for mod_name, m in candidates:
            # Coverage comes from copy counts or explicit MoE regions.
            m._stream_loaded_numel = 0
            # Protects storage materialization and coverage accounting.
            m._stream_lock = threading.Lock()
            m._stream_claimed = False
            m._stream_on_host = self._host_staging
            # Used only when loader calls must be replayed from meta.
            m._stream_buffer_list = []
            # Published after materialization under `_stream_lock`.
            m._stream_materialized_param_ids = set()
            # Names of source parameters released by `_finalize`.
            m._stream_param_names = [
                (f"{mod_name}.{p_name}" if mod_name else p_name, p_name)
                for p_name, p in m.named_parameters(recurse=False)
                if p is not None
            ]
            # Per-parameter `(expert, shard)` coverage from ExpertStagingPool.
            m._stream_moe_arrivals = {}
            # Mixed semantic and generic accounting cannot trigger early.
            m._stream_moe_declined_param_ids = set()
            m._stream_tracking_invalid = False
            m._stream_expected_numel = sum(
                p.numel() for _, p in m.named_parameters(recurse=False) if p is not None
            )
            for _, p in m.named_parameters(recurse=False):
                if p is not None:
                    self.param_to_module[id(p)] = m

    # ── loading-loop wiring ───────────────────────────────────────────────

    def manages_param(self, param: nn.Parameter) -> bool:
        """Whether this parameter belongs to a streamed quantization module."""
        return id(param) in self.param_to_module

    def bind_params_dict(self, params_dict: dict) -> None:
        self._params_dict = params_dict

    def release_params_dict(self) -> None:
        """Drop the params_dict ref so its Parameters can be collected."""
        self._params_dict = None

    def resolve_num_threads(self, num_threads: int) -> int:
        """Use concurrent checkpoint loading only with host staging."""
        if self._host_staging:
            return num_threads
        if num_threads > 1:
            tail_threads = envs.ATOM_ONLINE_QUANT_STREAMING_THREADS
            logger.info(
                "Online-quant streaming enabled: the checkpoint walk runs "
                "single-threaded; per-module quantization %s.",
                (
                    f"is offloaded to {tail_threads} worker thread(s)"
                    if tail_threads > 0
                    else "runs inline on the walking thread"
                ),
            )
        return 1

    def setup_online_quant_pool(self) -> None:
        """Create the module-finalization pool.

        Worker threads must set their CUDA device and use independent streams.
        """
        num_workers = envs.ATOM_ONLINE_QUANT_STREAMING_THREADS
        if num_workers <= 0:
            return
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        worker_stream = self._worker_stream

        def _worker_init():
            if device is not None:
                torch.cuda.set_device(device)
                worker_stream.s = torch.cuda.Stream(device=device)

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="atom-stream-quant",
            initializer=_worker_init,
        )
        self._slots = threading.Semaphore(2 * num_workers)

    def drain(self) -> None:
        """Wait for all finalizers and surface worker exceptions."""
        for future in concurrent.futures.as_completed(self._futures):
            future.result()
        self._futures.clear()

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ── trigger ───────────────────────────────────────────────────────────

    def run(self, fn, args) -> None:
        """Run one loader call and record its physical or logical coverage."""
        param = args[0] if args else None
        module = self.param_to_module.get(id(param)) if param is not None else None
        if module is None:
            fn(*args)
            return
        if self._is_nonlocal_expert_arrival(module, fn, args):
            return
        if module._stream_claimed:
            # Never overwrite quantized storage with a late source arrival.
            self.excessive_loads.append(
                getattr(module, "prefix", type(module).__name__)
            )
            return
        counter = self._run_counted(module, param, fn, args)
        with module._stream_lock:
            moe_arrival = counter.moe_arrival
            if moe_arrival is not None:
                param_id, region, covered_numel = moe_arrival
                if param_id in module._stream_moe_declined_param_ids:
                    # A concurrent decline invalidated semantic accounting.
                    module._stream_tracking_invalid = True
                elif region is not None:
                    filled = module._stream_moe_arrivals.setdefault(param_id, set())
                    if region not in filled:
                        filled.add(region)
                        module._stream_loaded_numel += covered_numel
            else:
                # Repeated copies into one parameter must not claim it early.
                module._stream_loaded_numel += min(
                    counter.copied_numel,
                    param.numel(),
                )
            claimed = (
                not module._stream_tracking_invalid
                and module._stream_loaded_numel >= module._stream_expected_numel
                and not module._stream_claimed
            )
            if claimed:
                module._stream_claimed = True
        if claimed:
            self._submit_finalize(module)

    @staticmethod
    def _is_nonlocal_expert_arrival(module, fn, args) -> bool:
        """Whether a claimed EP module can ignore this checkpoint arrival."""
        return (
            len(args) >= 5
            and getattr(fn, "__self__", None) is module
            and hasattr(module, "_map_global_expert_id_to_local_expert_id")
            and module._map_global_expert_id_to_local_expert_id(args[4]) == -1
        )

    def _run_counted(self, module, param, fn, args) -> _CopyCounter:
        """Apply one loader call and return its logical or physical coverage."""
        deferred = self._ensure_storage_or_defer(module, param)
        # MoE fast path
        if not deferred:
            try:
                moe_arrival = self._try_stage_expert_arrival(module, param, fn, args)
            except NotImplementedError:
                if not module._stream_on_host:
                    raise
                # Retry on the load device when the CPU lacks a kernel.
                self._settle_on_device(module)
                moe_arrival = self._try_stage_expert_arrival(module, param, fn, args)
            if moe_arrival is not None:
                counter = _CopyCounter()
                counter.moe_arrival = moe_arrival
                return counter

        # Generic path
        counter = _CopyCounter()
        try:
            with counter:
                fn(*args)
        except NotImplementedError:
            # Preserve staged data and retry where the dtype has kernels.
            if not module._stream_on_host:
                raise
            self._settle_on_device(module)
            counter = _CopyCounter()
            with counter:
                fn(*args)
        if deferred:
            module._stream_buffer_list.append((fn, args))
        return counter

    def _try_stage_expert_arrival(self, module, param, fn, args):
        """Try the ExpertStagingPool semantic protocol on stream storage."""
        if (
            len(args) < 5
            or getattr(fn, "__self__", None) is not module
            or not hasattr(module, "stage_expert_weight")
            or not hasattr(module, "expected_batched_arrivals")
            or not hasattr(module, "_map_global_expert_id_to_local_expert_id")
            or not hasattr(module, "is_batched_expert_slot")
            or not hasattr(module, "batched_expert_region_numel")
        ):
            return None

        param_id = id(param)
        if param_id in module._stream_moe_declined_param_ids:
            return None

        _, loaded_weight, weight_name, shard_id, global_expert_id = args[:5]
        expected = module.expected_batched_arrivals(param)
        if not expected:
            return None

        local_expert_id = module._map_global_expert_id_to_local_expert_id(
            global_expert_id
        )
        if local_expert_id == -1:
            return (param_id, None, 0)
        if not module.is_batched_expert_slot(local_expert_id):
            return None

        staged = module.stage_expert_weight(
            param=param,
            staging=param.data,
            loaded_weight=loaded_weight,
            local_expert_id=local_expert_id,
            shard_id=shard_id,
            weight_name=weight_name,
        )
        if staged:
            covered_numel = module.batched_expert_region_numel(
                param,
                local_expert_id,
                shard_id,
            )
            return (
                param_id,
                (local_expert_id, shard_id),
                covered_numel,
            )

        # False means no bytes were written, so generic fallback is safe.
        with module._stream_lock:
            module._stream_moe_declined_param_ids.add(param_id)
            if module._stream_moe_arrivals.get(param_id):
                module._stream_tracking_invalid = True
        return None

    def materialize_fused_param(self, param: nn.Parameter) -> None:
        """Materialize a fused parameter whose loader bypasses ``run``."""
        param_id = id(param)
        module = self.param_to_module.get(param_id)
        if module is None:
            return
        with module._stream_lock:
            # Fused writes have no semantic coverage, so force post-load fallback.
            module._stream_moe_declined_param_ids.add(param_id)
            module._stream_tracking_invalid = True
            if param.data.is_meta:
                self._materialize_meta_to_host_or_device(param, module._load_device)
            elif param.data.device != module._load_device:
                self._swap_storage(param, param.data.to(module._load_device))
            module._stream_materialized_param_ids.add(param_id)

    # ── tail ──────────────────────────────────────────────────────────────

    @staticmethod
    def _swap_storage(param: nn.Parameter, buf: torch.Tensor) -> None:
        """Replace storage while preserving the Parameter object and attributes.

        Copy attributes before ``swap_tensors`` so concurrent dispatch never
        observes a missing ``weight_loader``.
        """
        replacement = nn.Parameter(buf, requires_grad=param.requires_grad)
        replacement.__dict__.update(param.__dict__)
        torch.utils.swap_tensors(param, replacement)

    @classmethod
    def _materialize_meta_to_host_or_device(cls, param: nn.Parameter, device) -> None:
        """Materialize zeroed storage, including unwritten padding bytes."""
        buf = torch.empty(tuple(param.shape), dtype=param.dtype, device=device)
        buf.view(torch.uint8).zero_()
        cls._swap_storage(param, buf)

    def _ensure_storage_or_defer(self, module: nn.Module, param: nn.Parameter) -> bool:
        """Materialize host storage or defer the loader call for replay.

        Storage inspection and swapping stay under the module lock so racing
        arrivals cannot discard each other's writes.
        """
        param_id = id(param)
        if param_id in module._stream_materialized_param_ids:
            return False
        with module._stream_lock:
            # Recheck after waiting before touching a possibly swapped TensorImpl.
            if param_id in module._stream_materialized_param_ids:
                return False
            if not param.data.is_meta:
                module._stream_materialized_param_ids.add(param_id)
                return False
            if not module._stream_on_host:
                return True
            self._materialize_meta_to_host_or_device(param, _HOST)
            module._stream_materialized_param_ids.add(param_id)
            return False

    def _settle_on_device(self, module: nn.Module) -> None:
        """Move staged parameters to the module's load device."""
        device = module._load_device
        with module._stream_lock:
            if not module._stream_on_host:
                return
            module._stream_on_host = False
            for _, p in module.named_parameters(recurse=False):
                if p is not None and not p.data.is_meta and p.data.device != device:
                    self._swap_storage(p, p.data.to(device))
                if p is not None and not p.data.is_meta:
                    module._stream_materialized_param_ids.add(id(p))

    def _materialize_and_replay(self, module: nn.Module) -> None:
        """Materialize remaining parameters and replay deferred loader calls."""
        for _, p in module.named_parameters(recurse=False):
            if p is not None and p.data.is_meta:
                self._materialize_meta_to_host_or_device(p, module._load_device)
            if p is not None:
                module._stream_materialized_param_ids.add(id(p))
        for fn, args in module._stream_buffer_list:
            fn(*args)
        module._stream_buffer_list.clear()

    def _settle_weights(self, module: nn.Module) -> None:
        """Give the module device storage holding everything that arrived."""
        self._settle_on_device(module)
        self._materialize_and_replay(module)

    def _finalize(self, module: nn.Module) -> None:
        """Move, quantize, and release one completed module."""
        self._settle_weights(module)
        module.process_weights_after_loading()
        # Keep stale objects alive for id-based lookup, but release their storage.
        params_dict = self._params_dict
        if params_dict is None:
            return
        for full_name, p_name in module._stream_param_names:
            stale = params_dict.get(full_name)
            if stale is not None and stale is not getattr(module, p_name, None):
                stale.data = torch.empty(0, dtype=stale.dtype, device=stale.device)

    def _submit_finalize(self, module: nn.Module) -> None:
        """Hand a completed module to the tail workers and return immediately."""
        # Mark claimed before the worker starts to prevent double post-processing.
        with self._done_lock:
            self.done_module_ids.add(id(module))
        if self._executor is None:
            self._finalize(module)
            return

        def _task():
            try:
                s = getattr(self._worker_stream, "s", None)
                if s is None:
                    self._finalize(module)
                else:
                    with torch.cuda.stream(s):
                        self._finalize(module)
                    s.synchronize()
            finally:
                # Always release the slot so worker failures cannot deadlock loading.
                self._slots.release()

        self._slots.acquire()
        self._futures.append(self._executor.submit(_task))

    # ── post-load report ──────────────────────────────────────────────────

    def replay_stragglers_and_report(self, is_rank0: bool) -> None:
        """Settle unclaimed modules and report coverage fallbacks."""
        stranded = []
        for mod_name, m in self.candidates:
            # Detect missing params before `_settle_weights` zero-fills them.
            targeted = {id(a[0]) for _, a in m._stream_buffer_list}
            for p_name, p in m.named_parameters(recurse=False):
                if p.data.is_meta and id(p) not in targeted:
                    stranded.append(f"{mod_name}.{p_name}")
            self._settle_weights(m)

        fell_back = [n for n, m in self.candidates if id(m) not in self.done_module_ids]
        if not is_rank0:
            return
        if stranded:
            logger.warning(
                "Online-quant streaming: %d parameter(s) were never loaded "
                "and stayed on the meta device; they have been zero-filled "
                "so post-processing can run, but the model is almost "
                "certainly wrong. First %d: %s",
                len(stranded),
                min(len(stranded), 20),
                stranded[:20],
            )
        if self.excessive_loads:
            unique = sorted(set(self.excessive_loads))
            logger.warning(
                "Online-quant streaming: dropped %d load(s) that arrived "
                "after their module was already quantized, across %d "
                "module(s). The checkpoint supplies more elements for these "
                "than _stream_expected_numel accounts for, so the surplus cannot "
                "be stored; verify the expected-size computation. First "
                "%d: %s",
                len(self.excessive_loads),
                len(unique),
                min(len(unique), 20),
                unique[:20],
            )
        log = logger.warning if fell_back else logger.info
        log(
            "Online-quant streaming: %d/%d eligible modules quantized during "
            "load, %d fell back to the post-load pass (no memory saving for "
            "those). First %d fallbacks: %s",
            len(self.candidates) - len(fell_back),
            len(self.candidates),
            len(fell_back),
            min(len(fell_back), 20),
            fell_back[:20],
        )
