# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The daemon's model runner (``ATOM_WEIGHT_CACHE=daemon``)."""

from __future__ import annotations

import contextlib
import gc
import os
import signal
import time

import torch
from torch import nn

from atom.model_engine.model_runner import ModelRunner
from atom.model_loader.loader import load_model
from atom.model_loader.weight_cache.export import (
    _skip_prepare_finalize,
    build_export,
    snapshot,
)
from atom.model_loader.weight_cache.key import (
    _CodeTracer,
    _record_jit,
    _SourceChanged,
    code_fingerprint,
    compute_key,
    exported_structure,
    structure,
)
from atom.model_loader.weight_cache.recorder import InputRecorder
from atom.model_loader.weight_cache.transport import (
    PROTOCOL_VERSION,
    _DaemonLock,
    _Server,
    gpu_identity,
    socket_path,
)
from atom.model_loader.weight_cache.values import _TAG, _entries, logger
from atom.utils import envs


class WeightCacheDaemonRunner(ModelRunner):
    """Loads its shard like a serving runner and serves it to clients over
    the socket; sets up nothing for serving requests (weights_only), so its
    EngineCore skips the KV cache, warmup and graphs and reports ready."""

    weights_only = True

    def __init__(self, *args, **kwargs):
        self._input_recorder = InputRecorder.start()
        super().__init__(*args, **kwargs)

    def _build_and_load_model(self, model_class):
        config = self.config
        t0 = time.perf_counter()
        label = f"rank {self.rank}"
        path = socket_path(self.device)
        self._wc_lock = _DaemonLock(
            path,
            rank=self.rank,
            model=config.model,
            gpu=gpu_identity(self.device),
            idle_timeout_s=envs.ATOM_PRELOAD_IDLE_TIMEOUT_S,
        )
        key = compute_key(config, self.rank, self.device)
        recorder, tracer, kernels = self._input_recorder, _CodeTracer(), {}
        recorder.enter_construct(config)
        try:
            self.model = model_class(config)
            fused_fn = getattr(self.model, "load_fused_expert_weights", None)
            torch.set_default_device(None)
            built = structure(self.model)
            before = snapshot(self.model)
            with tracer, _record_jit(kernels):
                recorder.enter_load(self.model)
                with _skip_prepare_finalize(self.model):
                    load_model(
                        self.model,
                        config.model,
                        config.hf_config,
                        config.load_dummy,
                        load_fused_expert_weights_fn=fused_fn,
                    )
        finally:
            recorder.finish_load(tracer.codes)  # before anything else reads the model
        reads = recorder.key_reads(config.model)
        after = snapshot(self.model)
        try:
            codes = code_fingerprint(tracer.codes, kernels.values())
        except _SourceChanged as e:
            raise RuntimeError(
                f"{_TAG} {label}: {e} changed after this daemon started, so the weights it "
                "loaded may not match the file; restart the daemon"
            ) from None
        export = build_export(self.model, before, after)
        views = export.pop("views")
        self._release_serving_state(export)
        payload = {
            "protocol": PROTOCOL_VERSION,
            "key": key,
            "reads": reads,
            "code": codes,
            "structure": exported_structure(built, export),
            "export": export,
            "summary": {
                "pid": os.getpid(),
                "daemon_pid": self._wc_lock.info.get("daemon_pid"),
                "bytes": export["bytes"],
            },
        }
        logger.info(
            f"{_TAG} {label}: load read {len(reads['env'])} env vars, {len(reads['config'])} config "
            f"fields, {len(reads['attrs'])} construction attributes, {len(reads['classattrs'])} class "
            f"attributes, {len(reads['globals'])} globals, {len(reads['files'])} files"
        )
        self._wc_server = _Server(
            path,
            payload,
            views,
            label,
            idle_timeout_s=self._wc_lock.info["idle_timeout_s"],
            on_idle=self._stop_daemon_launch,
        )
        self._wc_lock.set_state("serving", bytes=export["bytes"])
        entries = sum(
            len(i["params"]) + len(i["buffers"]) + len(i["attrs"])
            for i in export["plan"].values()
        )
        logger.info(
            f"{_TAG} {label}: serving {entries} entries ({export['bytes'] / (1 << 30):.2f} GiB, "
            f"{len(codes['entries'])} code entries) at {path} after {time.perf_counter() - t0:.1f}s"
        )
        if export["unshippable"]:
            logger.warning(
                f"{_TAG} {label}: loading created state that cannot be shared, so clients will "
                f"refuse this cache: {export['unshippable'][:10]}"
            )

    def _stop_daemon_launch(self):
        """Idle timeout: shut the whole daemon down through its main
        process, as SIGTERM to the launch would."""
        main_pid = self._wc_lock.info.get("daemon_pid")
        if main_pid:
            with contextlib.suppress(ProcessLookupError):
                os.kill(main_pid, signal.SIGTERM)
        else:
            self.exit()  # main process unknown: at least stop serving

    def exit(self):
        """Stop serving: called by the engine's shutdown (or on SIGTERM).
        Clients that mapped the weights keep them until they exit."""
        if not getattr(self, "still_running", False):
            return
        self.still_running = False
        server = getattr(self, "_wc_server", None)
        if server is not None:
            server.close()
        lock = getattr(self, "_wc_lock", None)
        if lock is not None:
            lock.release()
        logger.info(f"{_TAG} rank {self.rank}: stopped serving")

    def _release_serving_state(self, export: dict) -> None:
        """Everything but the weights is a client's KV cache lost: drop the
        communicators and runner buffers that only serving needs."""
        from aiter import destroy_dist_env

        # Tensors the clients build for themselves (unchanged buffers such
        # as RoPE caches) need no device memory here.
        shared = export["storages"]
        for _, obj, kind, k, v in _entries(self.model):
            if (
                kind != "param"
                and isinstance(v, torch.Tensor)
                and not isinstance(v, nn.Parameter)
                and v.is_cuda
                and v.untyped_storage().data_ptr() not in shared
            ):
                if kind == "buffer":
                    obj._buffers[k] = v.to("meta")
                else:
                    vars(obj)[k] = v.to("meta")
        with contextlib.suppress(Exception):
            destroy_dist_env()
        if torch.distributed.is_initialized():
            with contextlib.suppress(Exception):
                torch.distributed.destroy_process_group()
        for name in (
            "tokenID_processor",
            "sampler",
            "_dynamic_chunking",
            "vision_embeddings",
        ):
            if hasattr(self, name):
                setattr(self, name, None)
        gc.collect()
        torch.cuda.empty_cache()
        # Pinned staging buffers from loading stay cached in host memory.
        with contextlib.suppress(Exception):
            torch._C._host_emptyCache()
        free, total = torch.cuda.mem_get_info(self.device)
        gib, weights = 1 << 30, export["bytes"]
        logger.info(
            f"{_TAG} rank {self.rank}: after releasing serving state: device memory used "
            f"{(total - free) / gib:.2f} GiB, weights {weights / gib:.2f} GiB, other allocated "
            f"{(torch.cuda.memory_allocated() - weights) / gib:.2f} GiB, allocator reserved-but-free "
            f"{(torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / gib:.2f} GiB"
        )
