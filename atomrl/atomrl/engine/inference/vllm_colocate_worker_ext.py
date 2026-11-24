# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
#
# Modified by Advanced Micro Devices, Inc.
# Source: https://github.com/verl-project/verl/blob/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc/verl/workers/rollout/vllm_rollout/utils.py

"""vLLM worker extension for colocated ZMQ CUDA-IPC weight sync.

Vendored + trimmed from verl's ``vLLMColocateWorkerExtension``
(``verl/workers/rollout/vllm_rollout/utils.py``). vLLM instantiates the worker
with this class mixed in via ``worker_extension_cls`` so that, no matter the
underlying worker class (V0/V1), the rollout worker exposes
``update_weights_from_ipc``. LoRA / FP8 / QAT branches are dropped; AtomRL's
verl-aligned BF16 path only needs standard ``model.load_weights``.

The ZMQ socket path is keyed by ``{ray_job_id, replica_rank, local_rank}`` so it
matches the training-side ``BucketedWeightSender`` regardless of per-worker
``CUDA_VISIBLE_DEVICES`` differences and is unique across replicas / jobs.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import signal
import time
from types import MethodType

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("ATOMRL_LOGGING_LEVEL", "WARN"))


def _install_all_gather_lifecycle_diagnostics(group_cls=None) -> None:
    """Bracket one target-sized AITER AllGather with device synchronizations."""
    if os.environ.get("ATOMRL_DIAG_ALL_GATHER", "0") != "1":
        return
    if group_cls is None:
        from aiter.dist.parallel_state import GroupCoordinator

        group_cls = GroupCoordinator

    original = group_cls.all_gather
    if getattr(original, "_atomrl_lifecycle_diagnostic", False):
        return
    target_numel = int(os.environ.get("ATOMRL_DIAG_ALL_GATHER_NUMEL", "517120"))

    def all_gather(self, input_, use_custom=False, dim=-1):
        if input_.numel() != target_numel:
            return original(self, input_, use_custom=use_custom, dim=dim)

        storage = input_.untyped_storage()
        logger.warning(
            "[AG_LIFECYCLE] phase=before_presync rank=%s shape=%s dtype=%s "
            "device=%s contiguous=%s ptr=%#x storage_ptr=%#x storage_nbytes=%d "
            "storage_offset_bytes=%d dim=%d use_custom=%s",
            getattr(self, "rank_in_group", getattr(self, "rank", "?")),
            tuple(input_.shape),
            input_.dtype,
            input_.device,
            input_.is_contiguous(),
            input_.data_ptr(),
            storage.data_ptr(),
            storage.nbytes(),
            input_.data_ptr() - storage.data_ptr(),
            dim,
            use_custom,
        )
        torch.cuda.synchronize(input_.device)
        logger.warning("[AG_LIFECYCLE] phase=after_presync")
        output = original(self, input_, use_custom=use_custom, dim=dim)
        torch.cuda.synchronize(input_.device)
        logger.warning(
            "[AG_LIFECYCLE] phase=after_all_gather output_shape=%s output_ptr=%#x",
            tuple(output.shape),
            output.data_ptr(),
        )
        return output

    all_gather._atomrl_lifecycle_diagnostic = True
    group_cls.all_gather = all_gather


def _monkey_patch_compute_logits(model, vocab_size: int) -> None:
    """Mask out-of-vocab (padded) logits to -inf (verl monkey_patch_compute_logits).

    vLLM pads the vocab to a multiple of TP/64; without masking, the padded
    columns can be sampled — and after an online-FP8 weight reload the requantized
    lm_head can make those padded logits large, so the rollout emits garbage /
    never-EOS sequences that collapse training. verl applies this at engine init.
    """
    original_compute_logits = model.compute_logits

    def compute_logits(self, *args, **kwargs):
        logits = original_compute_logits(*args, **kwargs)
        if logits is not None:
            logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


def set_death_signal() -> None:
    """Kill this process when its parent (the server actor) dies (Linux only)."""
    if platform.system() != "Linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGKILL)
    except OSError:
        pass


class vLLMColocateWorkerExtension:
    """Mixed into the vLLM worker; receives IPC weight buckets and loads them."""

    def inspect_weight_integrity(self) -> dict[str, object]:
        """Scan resident model parameters and buffers for the first NaN/Inf."""
        from itertools import chain

        from atomrl.engine.inference.weight_integrity import scan_named_tensors

        model = self.model_runner.model
        report = scan_named_tensors(
            chain(model.named_parameters(), model.named_buffers()),
            stop_on_first_bad=True,
        )
        report["local_rank"] = int(self.local_rank)
        return report

    def inspect_fp8_scales(self) -> dict[str, object]:
        """Summarize resident block-scale validity and dynamic range."""
        from itertools import chain

        from atomrl.engine.inference.weight_integrity import scan_fp8_scales

        model = self.model_runner.model
        report = scan_fp8_scales(chain(model.named_parameters(), model.named_buffers()))
        report["local_rank"] = int(self.local_rank)
        return report

    def get_rdma_capabilities(self) -> dict[str, object]:
        """Return this worker's versioned RDMA reload capabilities."""
        from atomrl.engine.inference.rdma_protocol import (
            RDMA_PROTOCOL_VERSION,
            RDMACapability,
        )

        return RDMACapability(
            protocol_version=RDMA_PROTOCOL_VERSION,
            module_path=vLLMColocateWorkerExtension.__module__,
            online_quant_reload=True,
            prequantized_stream=True,
        ).to_dict()

    def __new__(cls, **kwargs):
        set_death_signal()
        # Online per-block FP8 (verl-aligned `fp8_per_block`): the BF16 actor
        # weights are re-quantized to per-128-block FP8 inside vLLM. On ROCm the
        # per-block weight post-processing needs a guard patch, applied here in
        # the worker subprocess before the model is built.
        try:
            vllm_config = kwargs.get("vllm_config")
            want_patch = os.environ.get("ATOMRL_FP8_PER_BLOCK", "0") == "1"
            if vllm_config is not None:
                from atomrl.engine.inference.vllm_fp8_utils import is_online_quant_model

                want_patch = want_patch or is_online_quant_model(vllm_config)
            if want_patch:
                from atomrl.engine.inference.vllm_fp8_utils import (
                    apply_vllm_fp8_per_block_patches,
                )

                apply_vllm_fp8_per_block_patches()
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.warning("fp8_per_block patch at worker __new__ failed: %s", exc)
        return super().__new__(cls)

    def _get_zmq_handle(self) -> str:
        replica_rank = os.environ.get("LUMEN_REPLICA_RANK", "0")
        job_id = os.environ.get("LUMEN_RAY_JOB_ID", "0")
        return f"ipc:///tmp/lumen-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{self.local_rank}.sock"

    def monkey_patch_model(self, vocab_size: int) -> None:
        """verl-aligned engine-init patch: mask OOV/padded logits to -inf."""
        _install_all_gather_lifecycle_diagnostics()
        model = self.model_runner.model
        _monkey_patch_compute_logits(model, vocab_size)
        logger.info(
            "monkey_patch_model: compute_logits OOV mask (vocab=%d)", vocab_size
        )

        from atomrl.moe.router_precision import enable_fp32_moe_router

        enable_fp32_moe_router(model)

    def init_rdma_weight_group(
        self,
        master_addr: str,
        master_port: int,
        base_rank: int,
        world_size: int,
        group_name: str,
        timeout_s: int = 600,
    ) -> bool:
        """Join an isolated RCCL weight-broadcast group."""
        from datetime import timedelta

        from atomrl.utils.independent_process_group import (
            init_independent_process_group,
        )

        groups = getattr(self, "_rdma_weight_groups", None)
        if groups is None:
            groups = {}
            self._rdma_weight_groups = groups
        if group_name in groups:
            return True
        rank = int(base_rank) + int(self.local_rank)
        groups[group_name] = init_independent_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            timeout=timedelta(seconds=int(timeout_s)),
            world_size=int(world_size),
            rank=rank,
            group_name=group_name,
        )
        logger.warning(
            "vLLM worker local_rank=%d joined RDMA weight group %s as rank=%d/%d",
            self.local_rank,
            group_name,
            rank,
            world_size,
        )
        return True

    def receive_weights_rdma(
        self,
        group_name: str,
        version: int,
        verify_full_load: bool = True,
        prequantized_fp8: bool = False,
        online_staging: str = "gpu",
    ) -> dict[str, object]:
        """Receive HF-named GPU buckets over RCCL and load them in place.

        For online FP8 quantized models (``fp8_per_block``), wraps the reload
        with vLLM's layerwise quantization lifecycle so per-block FP8 scales
        are rebuilt from the fresh BF16 weights after each update. That reload
        holds tensors until a module is complete, so each one is copied first,
        to ``online_staging`` (``gpu`` or ``cpu``). The other paths load views
        into the received bucket.
        """
        from vllm.platforms import current_platform

        groups = getattr(self, "_rdma_weight_groups", {})
        if group_name not in groups:
            raise RuntimeError(f"RDMA weight group is not initialized: {group_name}")
        if getattr(self, "device", None) is None:
            self.device = torch.device(
                f"{current_platform.device_type}:{self.local_rank}"
            )

        model = self.model_runner.model
        integrity_enabled = os.environ.get("ATOMRL_WEIGHT_SYNC_INTEGRITY", "0") == "1"
        integrity_reports: dict[str, object] = {}
        if integrity_enabled:
            integrity_reports["before_prepare"] = self.inspect_weight_integrity()

        from atomrl.engine.inference.rdma_weight_transfer import (
            receive_weight_stream,
        )

        # When actor sends pre-quantized FP8 weights (fp8_quantize=True),
        # the received tensors include .weight (fp8) + .weight_scale_inv (fp32).
        # vLLM's load_weights writes them directly into the model's FP8 params
        # — no online requant needed.
        #
        # When actor sends BF16 weights (fp8_quantize=False, default),
        # we need the online requant lifecycle: prepare → load BF16 → finalize.
        is_online_quant = False
        try:
            from atomrl.engine.inference.vllm_fp8_utils import is_online_quant_model

            is_online_quant = is_online_quant_model(self.model_runner.vllm_config)
        except (ImportError, AttributeError, RuntimeError):
            pass

        logger.warning(
            "RDMA receiver mode on local_rank=%d: "
            "online_quant=%s prequantized_fp8=%s",
            self.local_rank,
            is_online_quant,
            prequantized_fp8,
        )
        if prequantized_fp8 and not is_online_quant:
            raise RuntimeError(
                "prequantized FP8 RDMA stream requires a vLLM "
                "fp8_per_block online-quantized model"
            )
        if is_online_quant and not prequantized_fp8:
            from atomrl.engine.inference.vllm_fp8_utils import (
                ReloadFingerprintTracker,
                finalize_online_quantized_weights_loading,
                prepare_online_quantized_weights_for_loading,
            )

            # Snapshot resident BF16/FP8 state before prepare mutates reload state.
            fingerprints = ReloadFingerprintTracker(model)
            model_config = self.model_runner.vllm_config.model_config

            def log_static_changes(phase: str) -> None:
                try:
                    changed = fingerprints.checkpoint_static_changed_names()
                except RuntimeError as exc:
                    logger.error(
                        "RDMA checkpoint-static diagnostic failed on "
                        "local_rank=%d phase=%s: %s",
                        self.local_rank,
                        phase,
                        exc,
                    )
                    return
                if changed:
                    logger.error(
                        "RDMA checkpoint-static tensors changed on local_rank=%d "
                        "phase=%s: %s",
                        self.local_rank,
                        phase,
                        changed,
                    )

            reload_error: BaseException | None = None
            try:
                prepare_online_quantized_weights_for_loading(model)
                log_static_changes("after_prepare")
                if integrity_enabled:
                    integrity_reports["after_prepare"] = self.inspect_weight_integrity()
                stats = receive_weight_stream(
                    groups[group_name],
                    model,
                    device=self.device,
                    expected_version=int(version),
                    verify_full_load=bool(verify_full_load),
                    streamed_scales=False,
                    fingerprint_tracker=fingerprints,
                    finalize_fingerprints=False,
                    staging=online_staging,
                )
                log_static_changes("after_load")
                if integrity_enabled:
                    integrity_reports["after_load"] = self.inspect_weight_integrity()
            except BaseException as exc:
                reload_error = exc
                raise
            finally:
                finalize_error: BaseException | None = None
                try:
                    finalize_online_quantized_weights_loading(model, model_config)
                    log_static_changes("after_finalize")
                    if integrity_enabled:
                        integrity_reports["after_finalize"] = (
                            self.inspect_weight_integrity()
                        )
                # Finalization records every termination path so the primary reload failure is not lost.
                except BaseException as exc:  # noqa: BLE001 -- boundary
                    finalize_error = exc
                try:
                    fingerprints.restore_checkpoint_static_tensors()
                    log_static_changes("after_static_restore")
                except BaseException as exc:
                    if reload_error is None and finalize_error is None:
                        raise
                    logger.warning(
                        "secondary RDMA checkpoint-static restore failure "
                        "after reload/finalize error: %s",
                        exc,
                    )
                if finalize_error is not None:
                    if reload_error is None:
                        raise finalize_error
                    logger.warning(
                        "secondary RDMA online FP8 finalize failure after "
                        "reload error: %s",
                        finalize_error,
                    )
            stats["verification"] = fingerprints.finalize()
        elif prequantized_fp8:
            from atomrl.engine.inference.vllm_fp8_utils import (
                prepare_prequantized_fp8_weights_for_loading,
            )

            metadata = prepare_prequantized_fp8_weights_for_loading(model)
            logger.warning(
                "Restored prequantized FP8 loader metadata on local_rank=%d: %s",
                self.local_rank,
                metadata,
            )
            stats = receive_weight_stream(
                groups[group_name],
                model,
                device=self.device,
                expected_version=int(version),
                verify_full_load=bool(verify_full_load),
                streamed_scales=True,
                staging="none",
            )
        else:
            from atomrl.engine.inference.vllm_moe_weight_sync import (
                FusedMoEWeightRouter,
            )

            stats = receive_weight_stream(
                groups[group_name],
                model,
                device=self.device,
                expected_version=int(version),
                verify_full_load=bool(verify_full_load),
                streamed_scales=False,
                staging="none",
                moe_router=FusedMoEWeightRouter(model),
            )

        verification = stats.get("verification", {})
        if integrity_enabled:
            integrity_reports["after_reload"] = self.inspect_weight_integrity()
            stats["integrity"] = integrity_reports
        logger.warning(
            "RDMA reload checks complete on local_rank=%d "
            "(online_quant=%s, prequantized_fp8=%s): "
            "manifest=%s, static=checked, representatives=sampled, "
            "first_source_snapshot=%s, exact_name_mappings=%s, "
            "exact_name_change_checks=%s; summary=%s",
            self.local_rank,
            is_online_quant,
            prequantized_fp8,
            "checked" if verify_full_load else "disabled",
            verification.get("first_source_snapshot", "unknown"),
            verification.get("exact_name_mappings", 0),
            verification.get("exact_name_change_checks", 0),
            stats,
        )
        return stats

    def destroy_rdma_weight_group(self, group_name: str) -> bool:
        groups = getattr(self, "_rdma_weight_groups", {})
        group = groups.pop(group_name, None)
        if group is not None:
            import torch.distributed as dist

            dist.destroy_process_group(group)
        return True

    def update_weights_from_ipc(
        self,
        use_shm: bool = False,
        version: int | None = None,
        prequantized_fp8: bool = False,
    ) -> dict[str, object] | None:
        """Receive bucketed weights over ZMQ IPC and load them into the model.

        ``prequantized_fp8=True`` means the actor already quantized: the buckets
        carry ``.weight`` in FP8 plus ``.weight_scale_inv``, so the online requant
        lifecycle is skipped and the tensors are written straight into the model's
        FP8 parameters -- the colocated twin of ``receive_weights_rdma``'s
        prequantized branch.
        """
        from vllm.platforms import current_platform

        from atomrl.engine.inference.bucketed_weight_transfer import (
            BucketedWeightReceiver,
        )
        from atomrl.engine.inference.vllm_moe_weight_sync import (
            FusedMoEWeightRouter,
            assert_weight_sync_coverage,
            unstack_expert_weights,
        )

        if getattr(self, "device", None) is None:
            dev_type = current_platform.device_type
            self.device = torch.device(f"{dev_type}:{self.local_rank}")

        model = self.model_runner.model
        model_config = self.model_runner.vllm_config.model_config

        # Online per-block FP8 needs vLLM's layerwise-reload lifecycle around the
        # weight load so the per-block FP8 scales are rebuilt from the fresh BF16
        # weights (verl `is_online_quant_model` branch). Standard BF16 rollout
        # just loads + runs the non-idempotent post-load transforms once.
        is_online_quant = False
        try:
            from atomrl.engine.inference.vllm_fp8_utils import is_online_quant_model

            is_online_quant = is_online_quant_model(self.model_runner.vllm_config)
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.warning("online-quant detection failed: %s", exc)

        if prequantized_fp8:
            if not is_online_quant:
                raise RuntimeError(
                    "prequantized FP8 IPC stream requires a vLLM "
                    "fp8_per_block online-quantized model"
                )
            from atomrl.engine.inference.vllm_fp8_utils import (
                prepare_prequantized_fp8_weights_for_loading,
            )

            metadata = prepare_prequantized_fp8_weights_for_loading(model)
            logger.warning(
                "Restored prequantized FP8 loader metadata on local_rank=%d: %s",
                self.local_rank,
                metadata,
            )
            receiver = BucketedWeightReceiver(
                zmq_handle=self._get_zmq_handle(),
                device=self.device,
                use_shm=use_shm,
                expected_version=version,
            )
            stats = {"buckets": 0, "weights": 0}
            loaded: set[str] = set()

            def _load_prequantized(weights):
                # No clone here, unlike the online branch below. That one clones
                # because the layerwise reload DEFERS requant past the call and the
                # receiver's tensors are views into an IPC buffer the sender
                # overwrites. prepare_prequantized_fp8_weights_for_loading only
                # restores parameter classes and loaders -- it installs no deferred
                # reload -- so load_weights copies into the parameters during the
                # call, exactly as the BF16 path does.
                stats["buckets"] += 1
                stats["weights"] += len(weights)
                loaded.update(model.load_weights(weights) or ())

            receiver.receive_weights(on_bucket_received=_load_prequantized)
            # assert_weight_sync_coverage ignores *_scale_inv (a BF16 trainer never
            # sends those, so vLLM derives them locally). Here the trainer DOES send
            # them, and a silently dropped scale name leaves the rollout dequantizing
            # fresh FP8 weights with the scales it computed at startup -- wrong, but
            # only by a little, so it surfaces as drift rather than a crash. Verify
            # them explicitly, the way the RDMA path does via streamed_scales=True.
            from atomrl.engine.inference.vllm_fp8_utils import (
                TensorNameClass,
                classify_tensor_name,
            )

            missing_scales = sorted(
                name
                for name, _ in model.named_parameters()
                if name.endswith("weight_scale_inv")
                and classify_tensor_name(name, streamed_scales=True)
                is TensorNameClass.RELOADABLE
                and name not in loaded
            )
            if missing_scales:
                raise RuntimeError(
                    "prequantized FP8 IPC sync loaded "
                    f"{len(loaded)} names but left {len(missing_scales)} block-scale "
                    "parameters untouched, so the rollout would dequantize new "
                    "weights with stale scales: "
                    f"{missing_scales[:8]}"
                    f"{' ...' if len(missing_scales) > 8 else ''}"
                )
            assert_weight_sync_coverage(
                model, loaded, context="colocate-ipc-prequantized-fp8"
            )
            summary = {"prequantized_fp8": True, "online_quant": True, **stats}
            logger.warning(
                "IPC prequantized FP8 reload complete on local_rank=%d: %s",
                self.local_rank,
                summary,
            )
            return summary

        if is_online_quant:
            # verl-exact online per-block reload: initialize layerwise reload BEFORE
            # the first bucket, load each bucket as it arrives (the wrapped loaders
            # track per-layer completion across buckets), finalize AFTER the last.
            from atomrl.engine.inference.vllm_fp8_utils import (
                ReloadFingerprintTracker,
                finalize_online_quantized_weights_loading,
                prepare_online_quantized_weights_for_loading,
            )

            receiver = BucketedWeightReceiver(
                zmq_handle=self._get_zmq_handle(),
                device=self.device,
                use_shm=use_shm,
                expected_version=version,
            )
            _stats = {"buckets": 0, "weights": 0}
            fingerprints = ReloadFingerprintTracker(model)
            prepare_online_quantized_weights_for_loading(model)

            def _load_online(weights):
                # CRITICAL: the receiver hands out tensors that are VIEWS into the
                # shared IPC buffer, which the sender overwrites on the next bucket
                # (and _cleanup frees entirely before finalize runs). The online
                # layerwise reload DEFERS per-layer FP8 requant until a layer is
                # complete / until finalize, holding these views -> by then they
                # point at freed/overwritten memory -> corrupted weights (uniform
                # lm_head -> policy collapse after the first update). Clone so the
                # deferred reload owns valid storage. (The standard BF16 path copies
                # into params during the call, so it does not need this.)
                cloned = [(n, t.clone()) for (n, t) in unstack_expert_weights(weights)]
                fingerprints.observe_source(cloned)
                _stats["buckets"] += 1
                _stats["weights"] += len(cloned)
                model.load_weights(cloned)

            load_error: BaseException | None = None
            try:
                receiver.receive_weights(on_bucket_received=_load_online)
            except BaseException as exc:
                load_error = exc
                raise
            finally:
                try:
                    finalize_online_quantized_weights_loading(model, model_config)
                except BaseException as exc:
                    if load_error is None:
                        raise
                    logger.warning(
                        "secondary online FP8 finalize failure after reload "
                        "error: %s",
                        exc,
                    )
            fingerprint_summary = fingerprints.finalize()
            summary = {
                "online_quant": True,
                **_stats,
                "fingerprints": fingerprint_summary,
                "timing": dict(getattr(receiver, "stats", {})),
            }
            logger.warning(
                "IPC online reload checks complete: manifest=not aggregated, "
                "static=checked, representatives=sampled, "
                "first_source_snapshot=%s, exact_name_mappings=%s, "
                "exact_name_change_checks=%s; summary=%s",
                fingerprint_summary.get("first_source_snapshot", "unknown"),
                fingerprint_summary.get("exact_name_mappings", 0),
                fingerprint_summary.get("exact_name_change_checks", 0),
                summary,
            )
            return summary

        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
            expected_version=version,
        )
        # transformers 5.x sends MoE experts as fused 3D tensors, whose names
        # match none of vLLM's per-expert mappings; the router loads those and
        # everything else goes through vLLM's own load_weights.
        router = FusedMoEWeightRouter(model)
        loaded: set[str] = set()
        timing = {"route_s": 0.0, "load_weights_s": 0.0}

        def _load(weights):
            t0 = time.perf_counter()
            passthrough, moe_loaded = router.route(weights)
            t1 = time.perf_counter()
            loaded.update(moe_loaded)
            if passthrough:
                loaded.update(model.load_weights(passthrough) or ())
            timing["route_s"] += t1 - t0
            timing["load_weights_s"] += time.perf_counter() - t1

        t_total = time.perf_counter()
        receiver.receive_weights(on_bucket_received=_load)
        t_post = time.perf_counter()
        assert_weight_sync_coverage(model, loaded, context="colocate-ipc")

        # Some post-load transforms are non-idempotent; run once after all buckets.
        try:
            from vllm.model_executor.model_loader.utils import (
                process_weights_after_loading,
            )

            process_weights_after_loading(model, model_config, self.device)
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.warning("process_weights_after_loading skipped: %s", exc)
        now = time.perf_counter()
        return {
            **getattr(receiver, "stats", {}),
            **timing,
            "post_s": now - t_post,
            "total_s": now - t_total,
        }

    def reload_weights_from_safetensors(self, weight_dir: str) -> None:
        """Load weights from safetensors on shared storage (separation mode)."""
        import json
        import os

        from safetensors.torch import load_file
        from vllm.platforms import current_platform

        if getattr(self, "device", None) is None:
            dev_type = current_platform.device_type
            self.device = torch.device(f"{dev_type}:{self.local_rank}")

        index_path = os.path.join(weight_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            files = sorted(set(index["weight_map"].values()))
        else:
            files = sorted(
                f for f in os.listdir(weight_dir) if f.endswith(".safetensors")
            )

        model = self.model_runner.model
        loaded_names: set[str] = set()
        external_weight_count = 0
        for fname in files:
            sd = load_file(os.path.join(weight_dir, fname))
            external_weight_count += len(sd)
            loaded = model.load_weights(list(sd.items()))
            if loaded is None:
                raise RuntimeError(
                    "vLLM model.load_weights returned no load manifest; "
                    "cannot verify online weight synchronization."
                )
            loaded_names.update(loaded)

        expected_names = {name for name, _ in model.named_parameters()}
        missing_names = sorted(expected_names - loaded_names)
        if missing_names:
            raise RuntimeError(
                "Incomplete vLLM weight reload: "
                f"loaded {len(loaded_names)}/{len(expected_names)} internal "
                f"parameters from {external_weight_count} HF tensors; "
                f"missing={missing_names[:20]}"
            )

        # The model was already transformed/compiled during engine startup.
        # process_weights_after_loading() is not idempotent for all vLLM model
        # implementations; running it after every reload can re-wrap modules
        # and trigger a second multi-worker torch.compile.  Reload only replaces
        # parameter data, then wait for all H2D copies before acknowledging RPC.
        torch.cuda.synchronize(self.device)
        logger.warning(
            "reload_weights_from_safetensors: verified %d internal parameters "
            "from %d HF tensors at %s",
            len(loaded_names),
            external_weight_count,
            weight_dir,
        )
