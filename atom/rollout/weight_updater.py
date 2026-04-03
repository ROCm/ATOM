# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import Optional

import torch

logger = logging.getLogger("atom")


class WeightUpdaterMixin:
    """Mixin providing weight update capabilities for ModelRunner.

    Host class must provide:
      - self.model (nn.Module)
      - self.device (torch.device)
      - self.rank (int) — TP rank
      - self.world_size (int) — TP size
      - self.label (str)
      - self.clear_kv_cache() — method
    """

    def _get_param_to_module_mapping(self) -> dict[str, tuple]:
        """
        Get or build the parameter name to module mapping.

        This mapping is cached after the first call to avoid expensive
        rebuilding on every weight update.

        Returns:
            Dict mapping parameter full name to (module, param_name, param) tuple
        """
        if not hasattr(self, '_param_to_module') or self._param_to_module is None:
            self._param_to_module = {}
            for module_name, module in self.model.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    self._param_to_module[full_name] = (module, param_name, param)
            logger.debug(
                f"{self.label}: Built param_to_module mapping with "
                f"{len(self._param_to_module)} parameters"
            )
        return self._param_to_module

    def _get_packed_modules_mapping(self) -> dict:
        if not hasattr(self, '_cached_packed_mapping'):
            self._cached_packed_mapping = (
                getattr(self.model, 'packed_modules_mapping', None) or {}
            )
        return self._cached_packed_mapping

    def _get_packed_shard_order(self) -> dict[str, list]:
        """Build {target_suffix: [shard_id_0, shard_id_1, ...]} preserving declaration order."""
        if not hasattr(self, '_cached_packed_shard_order'):
            order: dict[str, list] = {}
            for _, (tgt, shard_id) in self._get_packed_modules_mapping().items():
                order.setdefault(tgt, []).append(shard_id)
            self._cached_packed_shard_order = order
        return self._cached_packed_shard_order

    def _resolve_packed_name(
        self, name: str, param_to_module: dict
    ) -> tuple[str, object, str] | None:
        """Try to resolve an HF name to an ATOM packed parameter.

        Returns (atom_full_name, shard_id, target_suffix) or None.
        """
        for src_suffix, (tgt_suffix, shard_id) in self._get_packed_modules_mapping().items():
            if src_suffix in name:
                atom_name = name.replace(src_suffix, tgt_suffix)
                if atom_name in param_to_module:
                    return atom_name, shard_id, tgt_suffix
        return None

    def _apply_packed_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        param_to_module: dict,
    ) -> str:
        """Handle a single incoming weight that belongs to a packed (fused) module.

        For FP8 params, shards are accumulated in a float32 buffer using the
        module's weight_loader (which handles GQA-aware TP sharding for QKV).
        Once all shards arrive, the buffer is requantized to FP8 in one shot.

        Returns:
            'updated'     – fused param fully updated (all shards received)
            'accumulated' – shard stored, waiting for remaining shards
            'skipped'     – not a packed param or lookup failed
        """
        resolved = self._resolve_packed_name(name, param_to_module)
        if resolved is None:
            return 'skipped'

        atom_name, shard_id, tgt_suffix = resolved
        module, param_name, param = param_to_module[atom_name]
        weight_loader = getattr(module, "weight_loader", None)
        if weight_loader is None:
            return 'skipped'

        if self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
            if not hasattr(self, '_packed_weight_accum'):
                self._packed_weight_accum = {}

            if atom_name not in self._packed_weight_accum:
                self._packed_weight_accum[atom_name] = {'shards': {}}

            self._packed_weight_accum[atom_name]['shards'][shard_id] = tensor.clone()

            expected = self._get_packed_shard_order().get(tgt_suffix, [])
            if set(self._packed_weight_accum[atom_name]['shards'].keys()) >= set(expected):
                buf = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=torch.float32, device=self.device),
                    requires_grad=False,
                )
                wlp = getattr(param, 'weight_loader_process', None)
                if wlp is not None:
                    buf.weight_loader_process = wlp

                for sid in expected:
                    shard_t = self._packed_weight_accum[atom_name]['shards'][sid]
                    shard_gpu = shard_t.to(device=self.device, dtype=torch.float32)
                    weight_loader(buf, shard_gpu, sid)

                self._requantize_fp8_weight(module, param_name, param, buf.data)
                del self._packed_weight_accum[atom_name]
                logger.debug(
                    f"{self.label}: FP8 packed weight updated: {atom_name} "
                    f"(composed from {len(expected)} shards)"
                )
                return 'updated'
            return 'accumulated'

        tensor_gpu = tensor.to(device=self.device)
        weight_loader(param, tensor_gpu, shard_id)
        return 'updated'

    def _try_shard_weight(
        self,
        param: torch.nn.Parameter,
        tensor: torch.Tensor,
        tp_rank: int,
        tp_size: int
    ) -> bool:

        param_shape = param.shape
        tensor_shape = tensor.shape

        if len(param_shape) != len(tensor_shape):
            return False

        # Find which dimension needs sharding
        shard_dim = None
        for dim in range(len(param_shape)):
            if tensor_shape[dim] == param_shape[dim] * tp_size:
                shard_dim = dim
                break
            elif tensor_shape[dim] != param_shape[dim]:
                # Dimension mismatch but not by tp_size factor
                return False

        if shard_dim is None:
            # No dimension needs sharding but shapes don't match
            return False

        # Shard the tensor along the identified dimension
        shard_size = param_shape[shard_dim]
        start_idx = tp_rank * shard_size

        tensor = tensor.to(device=self.device, dtype=param.dtype)
        sharded_tensor = tensor.narrow(shard_dim, start_idx, shard_size)
        param.data.copy_(sharded_tensor)

        return True

    @staticmethod
    def _is_fp8_param(module: torch.nn.Module, param: torch.nn.Parameter) -> bool:
        return (
            param.dtype.is_floating_point
            and param.element_size() < 2
            and getattr(module, "weight_scale", None) is not None
        )

    def _requantize_fp8_weight(
        self,
        module: torch.nn.Module,
        param_name: str,
        param: torch.nn.Parameter,
        tensor: torch.Tensor,
    ) -> None:
        """Requantize a full-precision weight to FP8 with updated weight_scale.

        Called when FSDP sends float32/bfloat16 trained weights to an FP8 model.
        Computes new per-block (or per-tensor/per-token) scale factors and writes
        both the FP8 weight and scale into the module in place.
        """
        weight_scale = module.weight_scale
        fp8_dtype = param.dtype
        fp8_max = torch.finfo(fp8_dtype).max

        tensor_gpu = tensor.to(device=self.device, dtype=torch.float32)

        tp_size = self.world_size
        if tp_size > 1 and tensor_gpu.shape != param.shape:
            for dim in range(len(param.shape)):
                if tensor_gpu.shape[dim] == param.shape[dim] * tp_size:
                    shard_size = param.shape[dim]
                    tensor_gpu = tensor_gpu.narrow(dim, self.rank * shard_size, shard_size)
                    break

        if tensor_gpu.shape != param.shape:
            logger.warning(
                f"{self.label}: Shape mismatch in FP8 requantize for {param_name}: "
                f"param={param.shape}, tensor={tensor_gpu.shape}"
            )
            return

        from aiter import QuantType as _QT
        quant_type = getattr(module, "quant_type", None)

        if quant_type is not None and quant_type.value == _QT.per_1x128.value:
            N, K = tensor_gpu.shape
            block_k = 128
            K_blocks = (K + block_k - 1) // block_k
            if K % block_k != 0:
                padded = torch.zeros(N, K_blocks * block_k, dtype=torch.float32, device=self.device)
                padded[:, :K] = tensor_gpu
            else:
                padded = tensor_gpu
            blocks = padded.reshape(N, K_blocks, block_k)
            block_amax = blocks.abs().amax(dim=-1)
            scale = (block_amax / fp8_max).clamp(min=1e-12)
            quantized = (blocks / scale.unsqueeze(-1)).to(fp8_dtype)
            quantized = quantized.reshape(N, K_blocks * block_k)[:, :K].contiguous()
            param.data.copy_(quantized)
            ws = weight_scale.data
            weight_scale.data.copy_(scale[:ws.shape[0], :ws.shape[1]].contiguous().to(ws.dtype))

        elif quant_type is not None and quant_type.value == _QT.per_Tensor.value:
            amax = tensor_gpu.abs().max()
            scale = (amax / fp8_max).clamp(min=1e-12)
            param.data.copy_((tensor_gpu / scale).to(fp8_dtype))
            weight_scale.data.fill_(scale.item())

        elif quant_type is not None and quant_type.value == _QT.per_Token.value:
            row_amax = tensor_gpu.abs().amax(dim=-1, keepdim=True)
            scale = (row_amax / fp8_max).clamp(min=1e-12)
            param.data.copy_((tensor_gpu / scale).to(fp8_dtype))
            weight_scale.data.copy_(scale.to(weight_scale.dtype))

        else:
            logger.warning(f"{self.label}: Unknown quant_type {quant_type} for FP8 requantize")
            return

        self._post_process_fp8_weight(module, param)
        logger.debug(
            f"{self.label}: FP8 requantized {param_name} on {type(module).__name__}, "
            f"quant_type={quant_type}, scale_shape={weight_scale.shape}"
        )

    def _post_process_fp8_weight(
        self,
        module: torch.nn.Module,
        param: torch.nn.Parameter,
    ) -> None:
        """Post-process an FP8 weight after update: normalization and shuffle.

        Must be called after any FP8 weight write (both requantize and direct copy)
        to ensure the weight layout matches what ATOM's GEMM kernels expect.
        """
        weight_scale = getattr(module, "weight_scale", None)

        if getattr(module, "need_normalize_e4m3fn_to_e4m3fnuz", False) and weight_scale is not None:
            from atom.model_ops.utils import normalize_e4m3fn_to_e4m3fnuz
            param.data, weight_scale.data, _ = normalize_e4m3fn_to_e4m3fnuz(
                param.data, weight_scale.data
            )

        quant_type = getattr(module, "quant_type", None)
        if quant_type is None:
            return

        from aiter import QuantType as _QT
        from atom.model_ops.utils import shuffle_weights

        needs_shuffle = False
        if quant_type.value == _QT.per_1x128.value:
            needs_shuffle = True
        elif quant_type.value == _QT.per_1x32.value:
            needs_shuffle = True
        elif quant_type.value == _QT.per_Token.value:
            try:
                from atom.model_ops import dtypes
                needs_shuffle = param.dtype == dtypes.fp8
            except ImportError:
                needs_shuffle = param.element_size() < 2

        if needs_shuffle:
            shuffle_weights(param)

    def update_weights(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        clear_kv_cache: bool = True
    ) -> int:
        """
        Update model weights from named tensors.

        Called by RLHF frameworks after each training step to
        synchronize weights from training engine to inference engine.

        Supports both direct parameter names and HuggingFace-style names that
        map to ATOM's fused parameters (qkv_proj, gate_up_proj) via the model's
        packed_modules_mapping.

        Args:
            named_tensors: List of (parameter_name, tensor) tuples.
                           Tensors should be full (unsharded) weights.
            clear_kv_cache: Whether to clear KV cache after update

        Returns:
            Number of parameters successfully updated
        """
        param_to_module = self._get_param_to_module_mapping()

        updated = 0
        skipped = 0
        ignored_scales = 0

        for name, tensor in named_tensors:
            if name not in param_to_module:
                result = self._apply_packed_weight(name, tensor, param_to_module)
                if result == 'updated':
                    updated += 1
                elif result == 'accumulated':
                    pass
                elif "weight_scale" in name or "input_scale" in name:
                    ignored_scales += 1
                else:
                    logger.debug(
                        f"{self.label}: Unmatched parameter: {name}"
                    )
                    skipped += 1
                continue

            module, param_name, param = param_to_module[name]
            weight_loader = getattr(module, "weight_loader", None)

            if self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
                self._requantize_fp8_weight(module, param_name, param, tensor)
                updated += 1
            elif self._is_fp8_param(module, param) and tensor.dtype == param.dtype:
                tensor = tensor.to(device=self.device)
                param.data.copy_(tensor)
                self._post_process_fp8_weight(module, param)
                updated += 1
            elif tensor.shape == param.shape:
                tensor = tensor.to(device=self.device, dtype=param.dtype)
                param.data.copy_(tensor)
                updated += 1
            elif weight_loader is not None and callable(weight_loader):
                try:
                    tensor = tensor.to(device=self.device)
                    weight_loader(param, tensor)
                    updated += 1
                except Exception as e:
                    logger.warning(f"{self.label}: weight_loader failed for {name}: {e}")
                    skipped += 1
            else:
                tp_size = self.world_size
                tp_rank = self.rank
                if tp_size > 1 and self._try_shard_weight(param, tensor, tp_rank, tp_size):
                    updated += 1
                else:
                    logger.warning(
                        f"{self.label}: Shape mismatch for {name}: "
                        f"expected {param.shape}, got {tensor.shape}"
                    )
                    skipped += 1

        if clear_kv_cache:
            self.clear_kv_cache()

        if hasattr(self, '_packed_weight_accum'):
            self._packed_weight_accum.clear()

        logger.info(
            f"{self.label}: Weight update complete - "
            f"updated={updated}, skipped={skipped}, "
            f"ignored_scales={ignored_scales}"
        )
        return updated

    def update_weights_from_shm(
        self,
        shm_name: str,
        bucket_meta: dict,
        is_last: bool = True,
    ) -> int:
        """
        Update model weights by reading tensor data from POSIX shared memory.

        Only lightweight metadata (shm_name, bucket_meta) is transmitted through
        the control path (EngineCore -> MessageQueue).  The heavy tensor payload
        resides in ``/dev/shm/<shm_name>`` and each ModelRunner maps it directly.

        Args:
            shm_name: Name of the POSIX shared-memory segment created by the
                       caller (LLMEngine).
            bucket_meta: ``{param_name: {"shape": tuple, "dtype": str,
                       "offset": int, "nbytes": int}}``.
            is_last: If ``True``, clear the KV cache after applying the weights
                     (last bucket in a multi-bucket transfer).

        Returns:
            Number of parameters successfully updated in this bucket.
        """
        from multiprocessing import shared_memory as _shm_mod
        from unittest.mock import patch

        # Open the existing shared-memory segment (do NOT unlink – caller owns it)
        with patch(
            "multiprocessing.resource_tracker.register",
            lambda *args, **kwargs: None,
        ):
            shm = _shm_mod.SharedMemory(name=shm_name)

        try:
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)
            param_to_module = self._get_param_to_module_mapping()

            updated = 0
            skipped = 0
            ignored_scales = 0

            for name, meta in bucket_meta.items():
                # Reconstruct a CPU tensor view from shared memory
                dtype_str = meta["dtype"].replace("torch.", "")
                dtype = getattr(torch, dtype_str)
                offset = meta["offset"]
                nbytes = meta["nbytes"]
                tensor = (
                    buffer[offset : offset + nbytes]
                    .view(dtype=dtype)
                    .view(meta["shape"])
                )

                if name not in param_to_module:
                    result = self._apply_packed_weight(name, tensor, param_to_module)
                    if result == 'updated':
                        updated += 1
                    elif result == 'accumulated':
                        pass
                    elif "weight_scale" in name or "input_scale" in name:
                        ignored_scales += 1
                    else:
                        logger.debug(
                            f"{self.label}: Unmatched parameter: {name}"
                        )
                        skipped += 1
                    continue

                module, param_name, param = param_to_module[name]
                weight_loader = getattr(module, "weight_loader", None)

                if self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
                    self._requantize_fp8_weight(module, param_name, param, tensor)
                    updated += 1
                elif self._is_fp8_param(module, param) and tensor.dtype == param.dtype:
                    tensor = tensor.to(device=self.device)
                    param.data.copy_(tensor)
                    self._post_process_fp8_weight(module, param)
                    updated += 1
                elif tensor.shape == param.shape:
                    tensor = tensor.to(device=self.device, dtype=param.dtype)
                    param.data.copy_(tensor)
                    updated += 1
                elif weight_loader is not None and callable(weight_loader):
                    try:
                        tensor = tensor.to(device=self.device)
                        weight_loader(param, tensor)
                        updated += 1
                    except Exception as e:
                        logger.warning(
                            f"{self.label}: weight_loader failed for {name}: {e}"
                        )
                        skipped += 1
                else:
                    tp_size = self.world_size
                    tp_rank = self.rank
                    if tp_size > 1 and self._try_shard_weight(
                        param, tensor, tp_rank, tp_size
                    ):
                        updated += 1
                    else:
                        logger.warning(
                            f"{self.label}: Shape mismatch for {name}: "
                            f"expected {param.shape}, got {tensor.shape}"
                        )
                        skipped += 1

            if is_last:
                self.clear_kv_cache()
                if hasattr(self, '_packed_weight_accum'):
                    if self._packed_weight_accum:
                        logger.warning(
                            f"{self.label}: Incomplete packed weight accumulators: "
                            f"{list(self._packed_weight_accum.keys())}"
                        )
                    self._packed_weight_accum.clear()

            logger.info(
                f"{self.label}: SHM weight update bucket done - "
                f"updated={updated}, skipped={skipped}, "
                f"ignored_scales={ignored_scales}, is_last={is_last}"
            )
            return updated
        finally:
            shm.close()

    def update_weights_from_ipc(
        self,
        ipc_handle,
        bucket_meta: dict,
        is_last: bool = True,
        ipc_handles: Optional[dict] = None,
    ) -> int:
        """Update model weights by reading tensor data from a CUDA IPC shared buffer.

        The sender (typically the RLHF training process) has allocated a GPU
        buffer, copied weight data into it, and obtained a CUDA IPC handle via
        ``reduce_tensor()``.

        When ``ipc_handles`` (per-GPU) is provided, each ModelRunner opens
        ONLY its own GPU's handle — always same-GPU IPC, no cross-GPU
        ``hipIpcOpenMemHandle``.  This avoids the ROCm/MI300X crash where
        opening an IPC handle from a different physical GPU causes a
        "Memory access fault".

        When ``ipc_handles`` is ``None``, falls back to the original
        ``ipc_handle`` (single handle) behavior.

        Args:
            ipc_handle: CUDA IPC handle from ``reduce_tensor(buffer)`` in
                the sender process.  Used as fallback when ``ipc_handles``
                is not provided.
            bucket_meta: ``{param_name: {"shape": tuple, "dtype": str,
                       "offset": int, "nbytes": int}}``.
            is_last: If ``True``, clear the KV cache after applying the weights
                     (last bucket in a multi-bucket transfer).
            ipc_handles: Per-GPU IPC handles dict ``{device_index: handle}``.
                When provided, each ModelRunner opens the handle for its own
                GPU (same-GPU IPC, safe on ROCm).

        Returns:
            Number of parameters successfully updated in this bucket.
        """
        # Cache the IPC buffer mapping: only open once per weight-update cycle.
        if not hasattr(self, '_ipc_buffer') or self._ipc_buffer is None:
            from atom.rollout.weight_sync import rebuild_ipc_handle
            # Global index for looking up the correct per-GPU IPC handle
            # (ATOMHttpServer keys handles by global device index: 0..num_gpus-1)
            global_device_idx = (self.config.parallel_config.data_parallel_rank_local or 0) * self.world_size + self.rank
            # Local index for rebuild_ipc_handle's device_id parameter
            # (DP subprocess has remapped CUDA_VISIBLE_DEVICES, so it sees cuda:0..tp-1)
            local_device_idx = self.device.index
            if ipc_handles is not None and global_device_idx in ipc_handles:
                # Per-GPU IPC: open the handle for THIS runner's GPU
                # (same-GPU IPC, safe on ROCm/MI300X).
                self._ipc_buffer = rebuild_ipc_handle(
                    ipc_handles[global_device_idx], device_id=local_device_idx
                )
                logger.info(
                    f"{self.label}: opened per-GPU IPC buffer mapping "
                    f"(size={self._ipc_buffer.numel()} bytes, "
                    f"global_device_idx={global_device_idx}, local_device_idx={local_device_idx}, "
                    f"buffer_device={self._ipc_buffer.device}, "
                    f"runner_device={self.device})"
                )
            else:
                # Fallback: single IPC handle (original behavior).
                # Do NOT override device_id: the buffer stays mapped on the
                # sender's GPU.  Cross-GPU runners access it via D2D copy
                # (.to(self.device)).  Overriding device_id to this runner's
                # GPU causes hipIpcOpenMemHandle to fail on ROCm when the
                # handle belongs to a different GPU.
                self._ipc_buffer = rebuild_ipc_handle(ipc_handle)
                logger.info(
                    f"{self.label}: opened IPC buffer mapping "
                    f"(size={self._ipc_buffer.numel()} bytes, "
                    f"buffer_device={self._ipc_buffer.device}, "
                    f"runner_device={self.device})"
                )
        buffer = self._ipc_buffer

        param_to_module = self._get_param_to_module_mapping()

        updated = 0
        skipped = 0
        ignored_scales = 0

        for name, meta in bucket_meta.items():
            dtype_str = meta["dtype"].replace("torch.", "")
            dtype = getattr(torch, dtype_str)
            offset = meta["offset"]
            nbytes = meta["nbytes"]

            # View into the IPC buffer (on sender's GPU), then copy to
            # this runner's device.  .to() always returns a new tensor
            # when the device differs; for same-device case we need an
            # explicit copy so the sender can safely overwrite the buffer.
            src = buffer[offset : offset + nbytes].view(dtype=dtype).view(meta["shape"])
            if src.device == self.device:
                tensor = src.clone()
            else:
                tensor = src.to(device=self.device)

            if name not in param_to_module:
                result = self._apply_packed_weight(name, tensor, param_to_module)
                if result == 'updated':
                    updated += 1
                elif result == 'accumulated':
                    pass
                elif "weight_scale" in name or "input_scale" in name:
                    ignored_scales += 1
                else:
                    logger.debug(
                        f"{self.label}: Unmatched parameter: {name}"
                    )
                    skipped += 1
                continue

            module, param_name, param = param_to_module[name]
            weight_loader = getattr(module, "weight_loader", None)

            if self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
                self._requantize_fp8_weight(module, param_name, param, tensor)
                updated += 1
            elif self._is_fp8_param(module, param) and tensor.dtype == param.dtype:
                param.data.copy_(tensor)
                self._post_process_fp8_weight(module, param)
                updated += 1
            elif tensor.shape == param.shape:
                if tensor.dtype != param.dtype:
                    tensor = tensor.to(dtype=param.dtype)
                param.data.copy_(tensor)
                updated += 1
            elif weight_loader is not None and callable(weight_loader):
                try:
                    weight_loader(param, tensor)
                    updated += 1
                except Exception as e:
                    logger.warning(
                        f"{self.label}: weight_loader failed for {name}: {e}"
                    )
                    skipped += 1
            else:
                tp_size = self.world_size
                tp_rank = self.rank
                if tp_size > 1 and self._try_shard_weight(
                    param, tensor, tp_rank, tp_size
                ):
                    updated += 1
                else:
                    logger.warning(
                        f"{self.label}: Shape mismatch for {name}: "
                        f"expected {param.shape}, got {tensor.shape}"
                    )
                    skipped += 1

        # Only release the IPC buffer mapping on the last bucket
        if is_last:
            self._ipc_buffer = None
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass  # ipc_collect may not be available on all platforms

            self.clear_kv_cache()
            if hasattr(self, '_packed_weight_accum'):
                if self._packed_weight_accum:
                    logger.warning(
                        f"{self.label}: Incomplete packed weight accumulators: "
                        f"{list(self._packed_weight_accum.keys())}"
                    )
                self._packed_weight_accum.clear()

        logger.info(
            f"{self.label}: IPC weight update bucket done - "
            f"updated={updated}, skipped={skipped}, "
            f"ignored_scales={ignored_scales}, is_last={is_last}"
        )
        return updated
