# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Speculative Decoding Draft Model Distillation trainer.

Implements the off-policy distillation loop for training
Eagle3 / DFlash draft models:

1. Sample prompts from the dataset.
2. Teacher forward (no grad) -- produce hidden states on dataset sequences.
3. Draft model forward (with grad) -- predict next tokens from teacher hiddens.
4. Loss: cross-entropy / forward KL with position-dependent decay weighting.

Unlike OPD, there is **no student rollout**: the teacher processes dataset
sequences directly, and the draft model learns to match/predict from the
teacher's internal representations.
"""

from __future__ import annotations

import gc
import logging
import math
import mmap
import os
import queue
import threading
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from atomrl.core.config import AtomRLConfig
from atomrl.trainer.callbacks import Callback, LoggingCallback

logger = logging.getLogger(__name__)


_PREFETCH_SENTINEL = object()


class _TeacherPrefetcher:
    """Prefetch teacher inference in a single background worker thread.

    All Mooncake receives happen on **CPU** to avoid concurrent GPU access
    from the prefetch thread and the training thread (which causes
    VM_L2_PROTECTION_FAULT on ROCm/MI350).  The main thread moves tensors
    to GPU when it calls ``get()``.
    """

    def __init__(self, trainer: SpecDistillTrainer) -> None:
        self._trainer = trainer
        self._device = trainer._device
        self._req_queue: queue.Queue[int | object] = queue.Queue()
        self._res_queue: queue.Queue[tuple | BaseException] = queue.Queue(maxsize=2)
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _loop(self) -> None:
        while True:
            step = self._req_queue.get()
            if step is _PREFETCH_SENTINEL:
                break
            try:
                ids, mask, loss_mask = self._trainer._get_batch_sequences(step)
                data = self._trainer._teacher_inference_rank0(
                    ids,
                    mask,
                    recv_device=torch.device("cpu"),
                )
                self._res_queue.put((data, ids, mask, loss_mask))
            # Prefetch thread transports arbitrary teacher failures to its consumer.
            except Exception as e:  # noqa: BLE001 -- boundary
                self._res_queue.put(e)

    def prefetch(self, step: int) -> None:
        """Submit a step for background processing (non-blocking)."""
        self._req_queue.put(step)

    def get(self) -> tuple:
        """Block until the next result is ready. Keep rank0_data on CPU for streaming."""
        item = self._res_queue.get()
        if isinstance(item, BaseException):
            raise item
        data, ids, mask, loss_mask = item
        return data, ids, mask, loss_mask

    def stop(self) -> None:
        self._req_queue.put(_PREFETCH_SENTINEL)
        self._worker.join(timeout=5)


_SHM_SLOTS = 3
_SHM_POLL_MS = 0.001
_SHM_TIMEOUT = float(os.environ.get("ATOMRL_SHM_TIMEOUT", "1800"))


class _ShmWriterThread:
    """Rank-0 only. Consumes from _TeacherPrefetcher, writes teacher tensors
    + input_ids + attention_mask to a double-buffered SHM slot, then creates
    a sentinel file to signal other ranks.

    Pure CPU/file I/O — no CUDA or NCCL calls — safe on ROCm.
    """

    def __init__(
        self, prefetcher: _TeacherPrefetcher, teacher_keys: tuple[str, ...]
    ) -> None:
        self._prefetcher = prefetcher
        self._teacher_keys = teacher_keys
        self._slot_free = [threading.Event() for _ in range(_SHM_SLOTS)]
        for ev in self._slot_free:
            ev.set()
        self._mmap_cache: list[dict] = [{} for _ in range(_SHM_SLOTS)]
        self._work: queue.Queue = queue.Queue()
        self._rank0_buffer: queue.Queue = queue.Queue(maxsize=_SHM_SLOTS)
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="shm-writer"
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            item = self._work.get()
            if item is _PREFETCH_SENTINEL:
                break
            step, slot = item
            try:
                logger.info(
                    "[shm-writer] step=%d slot=%d: waiting prefetcher.get()...",
                    step,
                    slot,
                )
                rank0_data, ids, mask, loss_mask = self._prefetcher.get()
                logger.info(
                    "[shm-writer] step=%d slot=%d: got prefetch data, enqueue rank0_buffer...",
                    step,
                    slot,
                )
                teacher_ids = rank0_data.get("input_ids", ids)
                buf_entry = {k: rank0_data[k] for k in self._teacher_keys}
                buf_entry["input_ids"] = teacher_ids
                buf_entry["attention_mask"] = mask
                buf_entry["loss_mask"] = loss_mask
                buf_entry["_slot"] = slot
                self._rank0_buffer.put(buf_entry)
                if not self._slot_free[slot].wait(timeout=_SHM_TIMEOUT):
                    logger.error(
                        "[shm-writer] slot %d not free after %ds, step=%d",
                        slot,
                        _SHM_TIMEOUT,
                        step,
                    )
                    del rank0_data
                    continue
                self._slot_free[slot].clear()
                self._publish(slot, step, rank0_data, teacher_ids, mask, loss_mask)
                logger.info("[shm-writer] step=%d slot=%d: published", step, slot)
                del rank0_data
            except Exception:
                logger.exception("[shm-writer] step=%d slot=%d failed", step, slot)

    def _get_mmap_buf(self, slot: int, key: str, path: str, nbytes: int) -> np.ndarray:
        cache = self._mmap_cache[slot]
        if key in cache:
            fd, mm, arr, cached_size = cache[key]
            if cached_size >= nbytes:
                return arr[:nbytes]
            del arr
            del cache[key]
            mm.close()
            os.close(fd)
        alloc = nbytes + nbytes // 8  # 12.5% headroom
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd, alloc)
        mm = mmap.mmap(fd, alloc)
        arr = np.frombuffer(mm, dtype=np.uint8)
        cache[key] = (fd, mm, arr, alloc)
        return arr[:nbytes]

    def _publish(
        self,
        slot: int,
        step: int,
        rank0_data: dict,
        input_ids,
        attention_mask,
        loss_mask,
    ) -> None:
        slot_dir = f"/dev/shm/_teacher_{slot}"
        os.makedirs(slot_dir, exist_ok=True)
        for fname in os.listdir(slot_dir):
            if fname.startswith("READY_"):
                try:
                    os.remove(os.path.join(slot_dir, fname))
                except FileNotFoundError:
                    pass
        meta: dict = {}
        for key in self._teacher_keys:
            t = rank0_data[key].contiguous()
            raw = t.view(torch.uint8).numpy().ravel()
            dst = self._get_mmap_buf(slot, key, f"{slot_dir}/{key}.bin", raw.nbytes)
            np.copyto(dst, raw)
            meta[key] = list(t.shape)
        for name, tensor in (
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
            ("loss_mask", loss_mask),
        ):
            c = tensor.contiguous()
            c.numpy().tofile(f"{slot_dir}/{name}.bin")
            meta[name] = list(c.shape)
        torch.save(meta, f"{slot_dir}/meta.pt")
        open(f"{slot_dir}/READY_{step}", "w").close()

    def submit(self, step: int, slot: int) -> None:
        self._work.put((step, slot))

    def get_rank0(self) -> dict:
        return self._rank0_buffer.get()

    def release_slot(self, slot: int) -> None:
        self._slot_free[slot].set()

    def stop(self) -> None:
        for ev in self._slot_free:
            ev.set()
        self._work.put(_PREFETCH_SENTINEL)
        self._thread.join(timeout=10)
        for slot_cache in self._mmap_cache:
            for fd, mm, arr, sz in slot_cache.values():
                try:
                    mm.close()
                    os.close(fd)
                except (BufferError, OSError):
                    pass
            slot_cache.clear()


class _ShmLoaderThread:
    """All ranks. Polls for sentinel file, mmap-loads teacher data from SHM slot.

    Pure CPU/file I/O — no CUDA or NCCL calls — safe on ROCm.
    """

    def __init__(self, rank: int, teacher_keys: tuple[str, ...]) -> None:
        self._rank = rank
        self._teacher_keys = teacher_keys
        self._work: queue.Queue = queue.Queue()
        self._ready: queue.Queue = queue.Queue(maxsize=_SHM_SLOTS)
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"shm-loader-r{rank}"
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            item = self._work.get()
            if item is _PREFETCH_SENTINEL:
                break
            step, slot = item
            try:
                self._wait_sentinel(slot, step)
                result = self._load(slot)
                result["_slot"] = slot
                self._ready.put(result)
            # Loader thread transports arbitrary deserialization failures to its consumer.
            except Exception as exc:  # noqa: BLE001 -- boundary
                self._ready.put(exc)

    def _wait_sentinel(self, slot: int, step: int) -> None:
        path = f"/dev/shm/_teacher_{slot}/READY_{step}"
        deadline = time.time() + _SHM_TIMEOUT
        while not os.path.exists(path):
            if time.time() > deadline:
                raise TimeoutError(f"[rank {self._rank}] shm-loader timeout: {path}")
            time.sleep(_SHM_POLL_MS)

    def _load(self, slot: int) -> dict:
        slot_dir = f"/dev/shm/_teacher_{slot}"
        meta = torch.load(f"{slot_dir}/meta.pt", weights_only=True)
        result: dict = {}
        for key in self._teacher_keys:
            shape = meta[key]
            nbytes = 2
            for d in shape:
                nbytes *= d
            path = f"{slot_dir}/{key}.bin"
            fd = os.open(path, os.O_RDONLY)
            mm = mmap.mmap(fd, nbytes, access=mmap.ACCESS_READ)
            t = torch.empty(nbytes, dtype=torch.uint8)
            t.numpy()[:] = np.frombuffer(mm, dtype=np.uint8, count=nbytes)
            mm.close()
            os.close(fd)
            result[key] = t.view(torch.bfloat16).reshape(shape)
        for name in ("input_ids", "attention_mask", "loss_mask"):
            if name not in meta:
                continue
            shape = meta[name]
            t = torch.empty(shape, dtype=torch.int64)
            with open(f"{slot_dir}/{name}.bin", "rb") as f:
                f.readinto(t.numpy())
            result[name] = t
        if "loss_mask" not in result:
            result["loss_mask"] = result["attention_mask"].clone()
        return result

    def submit(self, step: int, slot: int) -> None:
        self._work.put((step, slot))

    def get(self) -> dict:
        item = self._ready.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def stop(self) -> None:
        self._work.put(_PREFETCH_SENTINEL)
        self._thread.join(timeout=10)


class SpecDistillTrainer:
    """Off-policy draft model distillation trainer.

    Lifecycle: ``setup() -> train() -> cleanup()``.
    """

    def __init__(self, config: AtomRLConfig) -> None:
        self.config = config
        self.global_step: int = 0
        self.last_metrics: dict[str, float] = {}
        self.callbacks: list[Callback] = []

        self._teacher_model: torch.nn.Module | None = None
        self._teacher_engine: Any | None = (
            None  # AtomTeacherEngine or VllmTeacherEngine
        )
        self._mooncake_master: Any | None = None
        self._draft_model: torch.nn.Module | None = None
        self._lm_head_weight: torch.Tensor | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._tokenizer: Any = None
        self._dataset: Any = None

        self._is_distributed: bool = torch.distributed.is_initialized()
        self._rank: int = torch.distributed.get_rank() if self._is_distributed else 0
        self._world_size: int = (
            torch.distributed.get_world_size() if self._is_distributed else 1
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------
    # Distributed wrapping check
    # ------------------------------------------------------------------

    _SKIP_FSDP_MAX_GPU_GB = 80.0  # model BF16 + FP32 master + Adam states

    def _can_skip_distributed_wrapping(self, model: torch.nn.Module) -> bool:
        """Decide whether FSDP2/replicate can be safely skipped.

        Safe to skip when ALL of:
          1. training_backend is explicitly "none", OR
          2. All three auto-detect conditions hold:
             a. Model memory footprint (BF16 + FP32 master + optimizer)
                fits on a single GPU (< 80GB)
             b. No Dropout/DropPath modules — forward is deterministic
             c. All ranks receive identical data (broadcast from rank 0)
                which is always true in spec_distill training
        """
        backend = getattr(self.config.policy, "training_backend", "fsdp2")
        if backend == "none":
            return True

        num_params = sum(p.numel() for p in model.parameters())
        # BF16 model (2B/param) + FP32 master (4B) + Adam m+v (8B) = 14B/param
        est_gpu_gb = num_params * 14.0 / (1024**3)
        if est_gpu_gb >= self._SKIP_FSDP_MAX_GPU_GB:
            return False

        has_dropout = any(
            isinstance(
                m,
                (
                    torch.nn.Dropout,
                    torch.nn.Dropout1d,
                    torch.nn.Dropout2d,
                    torch.nn.Dropout3d,
                ),
            )
            for m in model.modules()
        )
        return not has_dropout

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize teacher, draft model, optimizer, and dataset."""
        teacher_cfg = self.config.algorithm.teacher
        spec_cfg = self.config.algorithm.spec_distill
        draft_cfg = self.config.algorithm.draft

        teacher_name = teacher_cfg.model_name or self.config.policy.model_name
        if not teacher_name:
            raise ValueError(
                "Teacher model name required via algorithm.teacher.model_name "
                "or policy.model_name."
            )

        # ---- Teacher model (frozen) ----
        if teacher_cfg.inference_backend == "vllm":
            self._setup_teacher_vllm(teacher_cfg, teacher_name)
        elif teacher_cfg.inference_backend == "atom":
            self._setup_teacher_atom(teacher_cfg, teacher_name)
        elif teacher_cfg.inference_backend == "hf":
            self._setup_teacher_hf(teacher_name)
        else:
            raise ValueError(
                "Unsupported teacher inference backend "
                f"{teacher_cfg.inference_backend!r}; choose 'hf', 'atom', or 'vllm'."
            )

        # Determine hidden dim from teacher
        teacher_hidden_dim = self._lm_head_weight.shape[1]

        # ---- Draft model ----
        draft_type = spec_cfg.draft_type.lower()
        logger.info(
            "[rank %d] Building draft model: type=%s, hidden_dim=%d",
            self._rank,
            draft_type,
            teacher_hidden_dim,
        )

        if draft_type == "eagle3":
            from atomrl.models.eagle3 import Eagle3Model

            num_layers = draft_cfg.num_layers or 1
            spec_length = self.config.algorithm.spec_distill.spec_length
            num_heads = getattr(draft_cfg, "num_heads", None) or max(
                1, teacher_hidden_dim // 128
            )
            ffn_dim = getattr(draft_cfg, "ffn_dim", None)
            head_dim = getattr(draft_cfg, "head_dim", None) or 128
            rms_norm_eps = getattr(draft_cfg, "rms_norm_eps", 1e-6)
            rope_theta = getattr(draft_cfg, "rope_theta", 1000000.0)
            num_kv_heads = getattr(draft_cfg, "num_kv_heads", None)

            rope_scaling = None
            rope_scaling_type = getattr(draft_cfg, "rope_scaling_type", None)
            if rope_scaling_type:
                rope_scaling = {
                    "type": rope_scaling_type,
                    "rope_type": rope_scaling_type,
                    "factor": getattr(draft_cfg, "rope_scaling_factor", 64.0),
                    "original_max_position_embeddings": getattr(
                        draft_cfg, "rope_original_max_pos", 4096
                    ),
                    "beta_fast": getattr(draft_cfg, "rope_beta_fast", 32.0),
                    "beta_slow": getattr(draft_cfg, "rope_beta_slow", 1.0),
                    "mscale": getattr(draft_cfg, "rope_mscale", 1.0),
                    "mscale_all_dim": getattr(draft_cfg, "rope_mscale_all_dim", 1.0),
                    "low_freq_factor": getattr(draft_cfg, "rope_low_freq_factor", 1.0),
                    "high_freq_factor": getattr(
                        draft_cfg, "rope_high_freq_factor", 4.0
                    ),
                }

            teacher_vocab_size = self._lm_head_weight.shape[0]
            self._draft_model = Eagle3Model(
                hidden_dim=teacher_hidden_dim,
                vocab_size=teacher_vocab_size,
                num_heads=num_heads,
                num_layers=num_layers,
                length=spec_length,
                ffn_dim=ffn_dim,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                rope_theta=rope_theta,
                num_kv_heads=num_kv_heads,
                rope_scaling=rope_scaling,
            )
        elif draft_type == "dflash":
            from atomrl.models.dflash import DFlashModel

            num_layers = draft_cfg.num_layers or 2
            self._draft_model = DFlashModel(
                hidden_dim=teacher_hidden_dim,
                num_target_layers=spec_cfg.num_target_layers,
                num_heads=max(1, teacher_hidden_dim // 128),
                num_layers=num_layers,
                block_size=8,
            )
        else:
            raise ValueError(
                f"Unknown draft_type: {draft_type!r}. Use 'eagle3' or 'dflash'."
            )

        resume_from = getattr(draft_cfg, "resume_from", None)
        if resume_from and not getattr(draft_cfg, "from_scratch", False):
            import glob as _glob

            ckpt_files = sorted(
                _glob.glob(os.path.join(resume_from, "checkpoint_*.pt"))
            )
            safetensor_files = sorted(
                _glob.glob(os.path.join(resume_from, "*.safetensors"))
            )
            if ckpt_files:
                ckpt_path = ckpt_files[-1]
                logger.info(
                    "[rank %d] Loading draft weights from %s", self._rank, ckpt_path
                )
                payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                # CheckpointManager wraps state inside {step, state_dict: {model_state_dict, ...}}.
                # Without this unwrap, payload.get("model_state_dict") returns None and
                # the strict=False load silently keeps the random init.
                if "state_dict" in payload and isinstance(payload["state_dict"], dict):
                    payload = payload["state_dict"]
                sd = payload.get("model_state_dict", payload)
                info = self._draft_model.load_state_dict(sd, strict=False)
                if info.missing_keys or info.unexpected_keys:
                    logger.warning(
                        "[rank %d] resume_from load_state_dict mismatch: missing=%d, unexpected=%d (first missing: %s)",
                        self._rank,
                        len(info.missing_keys),
                        len(info.unexpected_keys),
                        info.missing_keys[:3],
                    )
                else:
                    logger.info(
                        "[rank %d] resume_from: loaded %d tensors cleanly",
                        self._rank,
                        len(sd),
                    )
            elif safetensor_files:
                from safetensors.torch import load_file as _load_safetensors

                logger.info(
                    "[rank %d] Loading draft weights from HF safetensors: %s",
                    self._rank,
                    resume_from,
                )
                sd = {}
                for sf in safetensor_files:
                    sd.update(_load_safetensors(sf, device="cpu"))
                mapped = {}
                for k, v in sd.items():
                    if k in ("embed_tokens.weight", "lm_head.weight"):
                        continue
                    nk = k.replace("midlayer.", "layers.0.")
                    if nk == "norm.weight":
                        nk = "out_norm.weight"
                    mapped[nk] = v
                missing, unexpected = self._draft_model.load_state_dict(
                    mapped, strict=False
                )
                if missing:
                    logger.info(
                        "[rank %d] HF resume missing keys: %s", self._rank, missing
                    )
                if unexpected:
                    logger.info(
                        "[rank %d] HF resume unexpected keys: %s",
                        self._rank,
                        unexpected,
                    )
            else:
                logger.warning(
                    "[rank %d] resume_from=%s but no checkpoints found",
                    self._rank,
                    resume_from,
                )

        draft_dtype_str = getattr(draft_cfg, "dtype", "bfloat16")
        _dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        draft_dtype = _dtype_map.get(draft_dtype_str, torch.bfloat16)
        self._draft_model.to(device=self._device, dtype=draft_dtype)
        self._grad_scaler = None

        if draft_type == "eagle3":
            trainable = sum(
                p.numel() for p in self._draft_model.parameters() if p.requires_grad
            )
            logger.info(
                "[rank %d] Eagle3 draft model: %d trainable params "
                "(lm_head removed, uses teacher lm_head at forward time)",
                self._rank,
                trainable,
            )

        # Apply Lumen AITER kernel optimizations before FSDP2 sharding
        training_quant = {}
        if hasattr(self.config, "quantization") and hasattr(
            self.config.quantization, "training"
        ):
            tq = self.config.quantization.training
            if tq is not None:
                training_quant = (
                    tq
                    if isinstance(tq, dict)
                    else {
                        k: getattr(tq, k, None)
                        for k in [
                            "fp8",
                            "lumen_norm",
                            "lumen_linear",
                            "hf_attn_patch",
                            "fp8_param_manager",
                        ]
                    }
                )
        lumen_norm = training_quant.get("lumen_norm", False)
        lumen_linear = training_quant.get("lumen_linear", False)
        hf_attn_patch = training_quant.get("hf_attn_patch", False)
        if lumen_norm or lumen_linear or hf_attn_patch:
            try:
                import inspect as _inspect

                from atomrl.lumen.config import LumenConfig

                _lumen_params = _inspect.signature(LumenConfig.__init__).parameters
                lumen_kwargs: dict[str, object] = {"scaling": "none"}
                if "lumen_norm" in _lumen_params:
                    lumen_kwargs["lumen_norm"] = bool(lumen_norm)
                if "lumen_linear" in _lumen_params:
                    lumen_kwargs["lumen_linear"] = bool(lumen_linear)
                if "hf_attn_patch" in _lumen_params:
                    lumen_kwargs["hf_attn_patch"] = bool(hf_attn_patch)
                spec_cfg = getattr(self.config.algorithm, "spec_distill", None)
                if (
                    spec_cfg is not None
                    and getattr(spec_cfg, "loss_type", "") == "forward_kl"
                ) and "fused_kl_loss" in _lumen_params:
                    lumen_kwargs["fused_kl_loss"] = True
                lumen_cfg = LumenConfig(**lumen_kwargs)
                _manager, self._draft_model = lumen_cfg.enable(self._draft_model)
            except ImportError:
                logger.warning("Lumen not available; using default kernels")

        if self._is_distributed:
            skip_fsdp = self._can_skip_distributed_wrapping(self._draft_model)
            if skip_fsdp:
                self._draft_model.to(device=self._device)
                if self._world_size > 1:
                    from torch.distributed._composable.replicate import replicate

                    replicate(self._draft_model)
                    logger.info(
                        "[rank %d] Using composable replicate for gradient sync (%d params)",
                        self._rank,
                        sum(p.numel() for p in self._draft_model.parameters()),
                    )
                else:
                    logger.info(
                        "[rank %d] Single GPU — no distributed wrapping needed",
                        self._rank,
                    )
            else:
                from atomrl.engine.training.fsdp_backend import FSDP2Backend

                self._draft_model = FSDP2Backend.apply_fsdp2(
                    self._draft_model, {"enabled": True, "strategy": "replicate"}
                )

        from atomrl.lumen.quantize.optimizer_manager import (
            FP32MasterWeightOptimizer,
            build_training_scheduler,
        )

        policy = self.config.policy
        self._optimizer = FP32MasterWeightOptimizer(
            self._draft_model.parameters(),
            torch.optim.AdamW,
            lr=policy.learning_rate,
            weight_decay=policy.weight_decay,
        )
        self._scheduler = build_training_scheduler(
            self._optimizer,
            policy,
            total_steps=int(self.config.num_training_steps),
        )
        self._optimizer.scheduler = self._scheduler

        # ---- Tokenizer ----
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            teacher_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        # ---- Resume from checkpoint ----
        ckpt_cfg = self.config.checkpointing
        if ckpt_cfg.resume and ckpt_cfg.checkpoint_dir:
            self._resume_from_checkpoint(ckpt_cfg.checkpoint_dir)

        # ---- Dataset ----
        self._load_dataset()

        # ---- Eval cache ----
        eval_cfg = self.config.eval
        if eval_cfg.enabled:
            self._build_eval_cache(num_samples=eval_cfg.num_samples)

        if not self.callbacks:
            self.callbacks.append(
                LoggingCallback(interval=max(1, self.config.logger.log_interval))
            )

        if self._is_distributed:
            torch.distributed.barrier()

        logger.info(
            "[rank %d] SpecDistillTrainer.setup complete: teacher=%s, draft=%s",
            self._rank,
            teacher_name,
            draft_type,
        )

    def _setup_teacher_hf(self, teacher_name: str) -> None:
        """Load teacher model via HuggingFace (original path)."""
        from transformers import AutoModelForCausalLM

        logger.info(
            "[rank %d] Loading teacher model (HF): %s", self._rank, teacher_name
        )
        self._teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self._teacher_model.eval()
        for p in self._teacher_model.parameters():
            p.requires_grad_(False)
        self._teacher_model.to(self._device)

        lm_head_weight = None
        for name, param in self._teacher_model.named_parameters():
            if "lm_head" in name and "weight" in name:
                lm_head_weight = param.detach().clone()
                break
        if lm_head_weight is None:
            raise RuntimeError("Could not find lm_head.weight in teacher model.")
        self._lm_head_weight = lm_head_weight
        # The Eagle3 step embeds the shifted input ids with the teacher's table.
        self._embed_weight = (
            self._teacher_model.get_input_embeddings().weight.detach().clone()
        )
        # The HF forward returns no last hidden state, so nothing is normalized.
        self._norm_weight = None
        self._norm_eps = 1e-6

    def _setup_teacher_atom(self, teacher_cfg: Any, teacher_name: str) -> None:
        """Load teacher model via ATOM engine + Mooncake TCP.

        Hidden states (3 aux layers + last hidden + token embeds) are
        transferred via Mooncake TCP, same as VllmTeacherEngine.
        """
        from atomrl.engine.inference.atom_teacher_engine import AtomTeacherEngine

        gpu_ids = teacher_cfg.gpu_ids or list(range(teacher_cfg.tensor_parallel_size))
        mc_config = getattr(self.config, "mooncake", None)
        transport = getattr(teacher_cfg, "transport", "mooncake")

        logger.info(
            "[rank %d] Loading teacher model (ATOM): %s (tp=%d, gpus=%s, transport=%s)",
            self._rank,
            teacher_name,
            teacher_cfg.tensor_parallel_size,
            gpu_ids,
            transport,
        )

        # Launch Mooncake master if needed
        if (
            self._rank == 0
            and mc_config is not None
            and transport == "mooncake"
            and not mc_config.master_server_address
        ):
            from atomrl.transfer.mooncake_master import launch_mooncake_master

            self._mooncake_master = launch_mooncake_master(mc_config)
            logger.info("Mooncake master started: %s", mc_config.master_server_address)

        # Only rank 0 starts the ATOM subprocess
        if self._rank == 0:
            max_bs = max(1, int(self.config.policy.train_global_batch_size))
            max_seq = int(self.config.policy.max_total_sequence_length)
            # Build ATOM-specific config dict from teacher.atom sub-section
            atom_cfg_raw = getattr(teacher_cfg, "atom", None)
            atom_config = {}
            if atom_cfg_raw is not None:
                if isinstance(atom_cfg_raw, dict):
                    atom_config = dict(atom_cfg_raw)
                else:
                    for k in dir(atom_cfg_raw):
                        if not k.startswith("_"):
                            atom_config[k] = getattr(atom_cfg_raw, k)
                # Merge extra_args into top-level
                extra = atom_config.pop("extra_args", None)
                if extra and isinstance(extra, dict):
                    atom_config.update(extra)

            self._teacher_engine = AtomTeacherEngine(
                model_name=teacher_name,
                tensor_parallel_size=teacher_cfg.tensor_parallel_size,
                gpu_ids=gpu_ids,
                mooncake_config=mc_config,
                transport=transport,
                quantization=getattr(teacher_cfg, "quantization", ""),
                atom_config=atom_config,
                max_batch_size=max_bs,
                max_seq_len=max_seq,
                local_device=self._device,
            )
            self._teacher_engine.start()
            self._lm_head_weight = self._teacher_engine.get_lm_head_weight()
            self._embed_weight = self._teacher_engine.get_embed_weight()
            self._norm_weight, self._norm_eps = self._teacher_engine.get_norm_weight()
        else:
            self._teacher_engine = None
            self._lm_head_weight = None
            self._embed_weight = None
            self._norm_weight = None
            self._norm_eps = 1e-6

        # Broadcast weights to all ranks (same pattern as _setup_teacher_vllm)
        if self._is_distributed:
            for attr in ("_lm_head_weight", "_embed_weight", "_norm_weight"):
                w = getattr(self, attr)
                if self._rank == 0:
                    shape = torch.tensor(
                        list(w.shape),
                        dtype=torch.long,
                        device=self._device,
                    )
                else:
                    shape = torch.zeros(2, dtype=torch.long, device=self._device)
                ndim_t = torch.tensor(
                    [len(w.shape) if w is not None else 0],
                    dtype=torch.long,
                    device=self._device,
                )
                torch.distributed.broadcast(ndim_t, src=0)
                ndim = ndim_t.item()
                if self._rank != 0:
                    shape = torch.zeros(ndim, dtype=torch.long, device=self._device)
                else:
                    shape = torch.tensor(
                        list(w.shape),
                        dtype=torch.long,
                        device=self._device,
                    )
                torch.distributed.broadcast(shape, src=0)
                dims = [shape[i].item() for i in range(ndim)]
                if self._rank == 0:
                    weight = w.to(self._device)
                else:
                    weight = torch.zeros(
                        *dims,
                        dtype=torch.bfloat16,
                        device=self._device,
                    )
                torch.distributed.broadcast(weight, src=0)
                setattr(self, attr, weight.cpu())

            eps_t = torch.tensor(
                [self._norm_eps], dtype=torch.float64, device=self._device
            )
            torch.distributed.broadcast(eps_t, src=0)
            self._norm_eps = eps_t.item()

    def _setup_teacher_vllm(self, teacher_cfg: Any, teacher_name: str) -> None:
        """Load teacher via vLLM + ATOM plugin + Mooncake."""
        from atomrl.engine.inference.vllm_teacher_engine import VllmTeacherEngine

        gpu_ids = teacher_cfg.gpu_ids or list(range(teacher_cfg.tensor_parallel_size))
        logger.info(
            "[rank %d] Loading teacher (vLLM+ATOM): %s (tp=%d, gpus=%s)",
            self._rank,
            teacher_name,
            teacher_cfg.tensor_parallel_size,
            gpu_ids,
        )

        mc_config = getattr(self.config, "mooncake", None)

        if (
            self._rank == 0
            and mc_config is not None
            and not mc_config.master_server_address
        ):
            from atomrl.transfer.mooncake_master import launch_mooncake_master

            self._mooncake_master = launch_mooncake_master(mc_config)
            logger.info("Mooncake master started: %s", mc_config.master_server_address)

        if self._rank == 0:
            # If algorithm.spec_distill.aux_hidden_state_layer_ids is set the
            # draft model was built for those exact teacher layers; the teacher
            # must capture from the same list, not the auto-picked default.
            spec_cfg = getattr(self.config.algorithm, "spec_distill", None)
            aux_override = (
                getattr(spec_cfg, "aux_hidden_state_layer_ids", None)
                if spec_cfg
                else None
            )
            self._teacher_engine = VllmTeacherEngine(
                model_name=teacher_name,
                gpu_ids=gpu_ids,
                mooncake_config=mc_config,
                tensor_parallel_size=teacher_cfg.tensor_parallel_size,
                quantization=getattr(teacher_cfg, "quantization", ""),
                max_batch_size=max(1, int(self.config.policy.train_global_batch_size)),
                max_seq_len=int(self.config.policy.max_total_sequence_length),
                local_device=self._device,
                aux_layer_ids=list(aux_override) if aux_override else None,
            )
            self._teacher_engine.start()
            self._lm_head_weight = self._teacher_engine.get_lm_head_weight()
            self._embed_weight = self._teacher_engine.get_embed_weight()
            self._norm_weight, self._norm_eps = self._teacher_engine.get_norm_weight()
        else:
            self._teacher_engine = None
            self._lm_head_weight = None
            self._embed_weight = None
            self._norm_weight = None
            self._norm_eps = 1e-6

        if self._is_distributed:
            for attr in ("_lm_head_weight", "_embed_weight", "_norm_weight"):
                w = getattr(self, attr)
                if self._rank == 0:
                    shape = torch.tensor(
                        list(w.shape),
                        dtype=torch.long,
                        device=self._device,
                    )
                else:
                    shape = torch.zeros(2, dtype=torch.long, device=self._device)
                ndim_t = torch.tensor(
                    [len(w.shape) if w is not None else 0],
                    dtype=torch.long,
                    device=self._device,
                )
                torch.distributed.broadcast(ndim_t, src=0)
                ndim = ndim_t.item()
                if self._rank != 0:
                    shape = torch.zeros(ndim, dtype=torch.long, device=self._device)
                else:
                    shape = torch.tensor(
                        list(w.shape),
                        dtype=torch.long,
                        device=self._device,
                    )
                torch.distributed.broadcast(shape, src=0)
                dims = [shape[i].item() for i in range(ndim)]
                if self._rank == 0:
                    weight = w.to(self._device)
                else:
                    weight = torch.zeros(
                        *dims,
                        dtype=torch.bfloat16,
                        device=self._device,
                    )
                torch.distributed.broadcast(weight, src=0)
                setattr(self, attr, weight.cpu())

            eps_t = torch.tensor(
                [self._norm_eps], dtype=torch.float64, device=self._device
            )
            torch.distributed.broadcast(eps_t, src=0)
            self._norm_eps = eps_t.item()

    def _teacher_inference_rank0(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recv_device: torch.device | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Run teacher inference on rank 0 only (no NCCL). Thread-safe."""
        if self._rank != 0:
            return None
        return self._teacher_engine.extract_hidden_states(
            input_ids,
            attention_mask,
            recv_device=recv_device,
        )

    def _teacher_broadcast(
        self,
        rank0_data: dict[str, torch.Tensor] | None,
        input_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """NCCL broadcast teacher data from rank 0 to all ranks."""
        if self._rank == 0:
            hidden_states = rank0_data["hidden_states"]
            token_embeds = rank0_data["token_embeds"]
            last_hidden_states = rank0_data["last_hidden_states"]
        else:
            hidden_states = None
            token_embeds = None
            last_hidden_states = None

        if self._is_distributed:
            if self._rank == 0:
                shape = torch.tensor(
                    [
                        hidden_states.shape[0],
                        hidden_states.shape[1],
                        hidden_states.shape[2],
                        token_embeds.shape[2],
                        last_hidden_states.shape[2],
                    ],
                    dtype=torch.long,
                    device=self._device,
                )
            else:
                shape = torch.zeros(5, dtype=torch.long, device=self._device)
            torch.distributed.broadcast(shape, src=0)
            torch.cuda.synchronize()
            logger.warning("[rank %d] BCAST shape OK", self._rank)

            B, T = shape[0].item(), shape[1].item()
            D_hidden, D_embed, D_last = (
                shape[2].item(),
                shape[3].item(),
                shape[4].item(),
            )
            if self._rank == 0:
                h = hidden_states.to(self._device)
                e = token_embeds.to(self._device)
                lhs = last_hidden_states.to(self._device)
            else:
                h = torch.zeros(
                    B, T, D_hidden, dtype=torch.bfloat16, device=self._device
                )
                e = torch.zeros(
                    B, T, D_embed, dtype=torch.bfloat16, device=self._device
                )
                lhs = torch.zeros(
                    B, T, D_last, dtype=torch.bfloat16, device=self._device
                )
            torch.distributed.broadcast(h, src=0)
            torch.cuda.synchronize()
            logger.warning(
                "[rank %d] BCAST hidden_states OK (B=%d T=%d D=%d, %.1fG)",
                self._rank,
                B,
                T,
                D_hidden,
                h.element_size() * h.nelement() / 1e9,
            )
            torch.distributed.broadcast(e, src=0)
            torch.cuda.synchronize()
            logger.warning("[rank %d] BCAST token_embeds OK", self._rank)
            torch.distributed.broadcast(lhs, src=0)
            torch.cuda.synchronize()
            logger.warning("[rank %d] BCAST last_hidden_states OK", self._rank)
            hidden_states = h
            token_embeds = e
            last_hidden_states = lhs

        return {
            "hidden_states": hidden_states,
            "token_embeds": token_embeds,
            "input_ids": input_ids,
            "last_hidden_states": last_hidden_states,
        }

    def _teacher_forward_vllm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher forward via vLLM+ATOM engine (rank 0 only, then broadcast)."""
        rank0_data = self._teacher_inference_rank0(input_ids, attention_mask)
        return self._teacher_broadcast(rank0_data, input_ids)

    def _broadcast_batch_info(
        self,
        input_ids: torch.Tensor | None,
    ) -> tuple[int, int]:
        """Broadcast (B, T) from rank 0 so all ranks agree on batch shape."""
        if self._is_distributed:
            if self._rank == 0:
                info = torch.tensor(
                    [input_ids.shape[0], input_ids.shape[1]],
                    dtype=torch.long,
                    device=self._device,
                )
            else:
                info = torch.zeros(2, dtype=torch.long, device=self._device)
            torch.distributed.broadcast(info, src=0)
            return info[0].item(), info[1].item()
        return input_ids.shape[0], input_ids.shape[1]

    _TEACHER_KEYS = ("hidden_states", "token_embeds", "last_hidden_states")
    _SHM_DIR = "/dev/shm/_teacher"

    def _publish_to_shm(self, rank0_data: dict[str, torch.Tensor]) -> None:
        """Rank 0: write teacher data to shared memory. Other ranks wait."""
        import os as _os

        _os.makedirs(self._SHM_DIR, exist_ok=True)
        meta = {}
        for key in self._TEACHER_KEYS:
            t = rank0_data[key].contiguous()
            t.view(torch.uint8).numpy().tofile(f"{self._SHM_DIR}/{key}.bin")
            meta[key] = list(t.shape)
        torch.save(meta, f"{self._SHM_DIR}/meta.pt")

    def _load_from_shm(self) -> dict[str, torch.Tensor]:
        """All ranks: mmap teacher data from shared memory (zero-copy read)."""
        meta = torch.load(f"{self._SHM_DIR}/meta.pt", weights_only=True)
        result = {}
        for key in self._TEACHER_KEYS:
            shape = meta[key]
            nbytes = 2
            for d in shape:
                nbytes *= d
            storage = torch.UntypedStorage.from_file(
                f"{self._SHM_DIR}/{key}.bin",
                shared=True,
                nbytes=nbytes,
            )
            result[key] = (
                torch.empty(0, dtype=torch.bfloat16).set_(storage).reshape(shape)
            )
        return result

    def _cleanup_shm(self) -> None:
        import os as _os

        for f in (
            "hidden_states.bin",
            "token_embeds.bin",
            "last_hidden_states.bin",
            "meta.pt",
        ):
            try:
                _os.remove(f"{self._SHM_DIR}/{f}")
            except FileNotFoundError:
                pass

    def _load_dataset(self) -> None:
        """Load training dataset."""
        dataset_path = self.config.reward.dataset
        if not dataset_path:
            logger.warning("No dataset configured; using synthetic prompts.")
            self._dataset = None
            self._preprocessed = None
            return

        ds_cfg = self.config.dataset
        if ds_cfg.chat_template:
            from atomrl.data.dataset import load_and_preprocess_dataset

            tokenizer_path = self.config.policy.model_name
            if not os.path.isdir(tokenizer_path) and hasattr(
                self._tokenizer, "name_or_path"
            ):
                tokenizer_path = self._tokenizer.name_or_path
            if not os.path.isdir(tokenizer_path):
                teacher_path = self.config.algorithm.teacher.model_name
                if os.path.isdir(teacher_path):
                    tokenizer_path = teacher_path
            self._preprocessed = load_and_preprocess_dataset(
                dataset_path=dataset_path,
                tokenizer_path=tokenizer_path,
                max_length=self.config.policy.max_total_sequence_length,
                chat_template=ds_cfg.chat_template,
                seed=self.config.seed,
                last_turn_loss_only=ds_cfg.last_turn_loss_only,
                min_loss_tokens=ds_cfg.min_loss_tokens,
                num_workers=ds_cfg.num_preprocess_workers,
                cache_dir=ds_cfg.cache_dir,
            )
            self._dataset = None
            logger.info(
                "Preprocessed dataset: %d samples with chat_template=%s from %s",
                len(self._preprocessed),
                ds_cfg.chat_template,
                dataset_path,
            )
            return

        self._preprocessed = None
        from datasets import load_dataset

        dataset_split = getattr(self.config.reward, "dataset_split", "train")

        if os.path.isfile(dataset_path) or os.path.isdir(dataset_path):
            if dataset_path.endswith(".parquet"):
                self._dataset = load_dataset(
                    "parquet", data_files=dataset_path, split=dataset_split
                )
            elif dataset_path.endswith((".jsonl", ".json")):
                self._dataset = load_dataset(
                    "json", data_files=dataset_path, split=dataset_split
                )
            else:
                self._dataset = load_dataset(dataset_path, split=dataset_split)
        else:
            self._dataset = load_dataset(dataset_path, split=dataset_split)

        def _has_nonempty_prompt(s: dict) -> bool:
            convs = s.get("conversations")
            if convs and isinstance(convs, list) and isinstance(convs[0], dict):
                return any(
                    isinstance(m, dict)
                    and str(m.get("content") or m.get("value") or "").strip()
                    for m in convs
                )
            for key in ("prompt", "question", "input", "problem"):
                raw = s.get(key)
                if isinstance(raw, str) and raw.strip():
                    return True
                if isinstance(raw, list) and any(
                    isinstance(m, dict) and str(m.get("content") or "").strip()
                    for m in raw
                ):
                    return True
            return False

        n_before = len(self._dataset)
        self._dataset = self._dataset.filter(_has_nonempty_prompt)
        n_after = len(self._dataset)
        if n_after < n_before:
            logger.warning(
                "Dropped %d/%d samples with empty prompt", n_before - n_after, n_before
            )

        self._dataset = self._dataset.shuffle(seed=self.config.seed)
        logger.info(
            "Loaded dataset: %d samples (shuffled, seed=%d) from %s",
            len(self._dataset),
            self.config.seed,
            dataset_path,
        )

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------

    def _resume_from_checkpoint(self, checkpoint_dir: str) -> None:
        """Resume training from the latest checkpoint in ``checkpoint_dir``."""
        import glob as _glob

        ckpt_files = _glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
        if not ckpt_files:
            logger.info(
                "[rank %d] No checkpoints found in %s; starting from scratch.",
                self._rank,
                checkpoint_dir,
            )
            return

        import re as _re

        def _ckpt_step(p: str) -> int:
            m = _re.search(r"checkpoint_(\d+)\.pt$", p)
            return int(m.group(1)) if m else -1

        ckpt_path = max(ckpt_files, key=_ckpt_step)
        logger.info("[rank %d] Resuming from %s", self._rank, ckpt_path)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # CheckpointManager wraps state inside {"step": ..., "state_dict": {...}}
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            inner = payload["state_dict"]
            resumed_step = inner.get("step", payload.get("step", 0))
            payload = inner
        else:
            resumed_step = payload.get("step", 0)

        self.global_step = resumed_step
        logger.info(
            "[rank %d] Checkpoint payload keys: %s", self._rank, list(payload.keys())
        )

        sd = payload.get("model_state_dict")
        if sd and self._draft_model is not None:
            info = self._draft_model.load_state_dict(sd, strict=False)
            logger.info(
                "[rank %d] Loaded model weights from step %d (%d keys, missing=%s, unexpected=%s)",
                self._rank,
                resumed_step,
                len(sd),
                info.missing_keys,
                info.unexpected_keys,
            )

        opt_sd = payload.get("optimizer_state_dict")
        restored_optimizer_masters = bool(
            isinstance(opt_sd, dict) and opt_sd.get("_fp32_masters")
        )
        if opt_sd and self._optimizer is not None:
            if payload.get("optimizer_state_format") == "fp32_master_dcp" and hasattr(
                self._optimizer, "load_distributed_state_dict"
            ):
                from torch.distributed.checkpoint.state_dict import StateDictOptions

                self._optimizer.load_distributed_state_dict(
                    self._draft_model,
                    opt_sd,
                    StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
            else:
                self._optimizer.load_state_dict(opt_sd)
            logger.info(
                "[rank %d] Loaded optimizer state from step %d",
                self._rank,
                resumed_step,
            )

        fp32_params = payload.get("fp32_params")
        if fp32_params and hasattr(self._optimizer, "fp32_params"):
            for dst, src in zip(self._optimizer.fp32_params, fp32_params):
                dst.data.copy_(src.data.to(dst.device))
            logger.info(
                "[rank %d] Loaded %d FP32 master params", self._rank, len(fp32_params)
            )
        elif (
            sd
            and not restored_optimizer_masters
            and hasattr(self._optimizer, "sync_master_params_from_model")
        ):
            self._optimizer.sync_master_params_from_model()
            logger.info(
                "[rank %d] Synced FP32 params from loaded model weights", self._rank
            )

        scheduler_state = payload.get("scheduler_state_dict")
        sched_epoch = payload.get("scheduler_last_epoch")
        if scheduler_state is not None and hasattr(self._optimizer, "scheduler"):
            self._optimizer.scheduler.load_state_dict(scheduler_state)
            logger.info(
                "[rank %d] Restored scheduler to epoch %d (lr=%.2e)",
                self._rank,
                self._optimizer.scheduler.last_epoch,
                self._optimizer.get_learning_rate(),
            )
        elif sched_epoch is not None and hasattr(self._optimizer, "scheduler"):
            # Legacy checkpoints stored only the epoch; optimizer_state_dict
            # already carries the LR for that epoch, so do not step again.
            self._optimizer.scheduler.last_epoch = sched_epoch
        elif hasattr(self._optimizer, "scheduler") and resumed_step > 0:
            for _ in range(resumed_step):
                self._optimizer.scheduler.step()
            logger.info(
                "[rank %d] LR scheduler advanced to step %d (lr=%.2e)",
                self._rank,
                resumed_step,
                self._optimizer.get_learning_rate(),
            )

        logger.info("[rank %d] Resumed training from step %d", self._rank, resumed_step)

    # ------------------------------------------------------------------
    # Eval cache
    # ------------------------------------------------------------------

    def _build_eval_cache(self, num_samples: int = 256) -> None:
        """Cache the last ``num_samples`` from the shuffled dataset for eval.

        Stored on CPU as list of (input_ids, attention_mask, loss_mask) tuples.
        Non-overlapping with early training batches (takes from the tail).
        """
        if self._preprocessed is not None:
            ds_len = len(self._preprocessed)
            num_samples = min(num_samples, ds_len)
            start_idx = ds_len - num_samples
            max_len = self.config.policy.max_total_sequence_length

            cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for idx in range(start_idx, ds_len):
                item = self._preprocessed[idx]
                ids = item["input_ids"]
                if isinstance(ids, list):
                    ids = torch.tensor(ids, dtype=torch.long)
                ids = ids[: max_len - 1]
                lm = torch.tensor(item["loss_mask"], dtype=torch.long)
                lm = lm[: len(ids)]
                if len(lm) < len(ids):
                    lm = torch.cat(
                        [lm, torch.zeros(len(ids) - len(lm), dtype=torch.long)]
                    )
                attn = torch.ones(len(ids), dtype=torch.long)
                cache.append((ids.unsqueeze(0), attn.unsqueeze(0), lm.unsqueeze(0)))

            self._eval_cache = cache
            logger.info(
                "[rank %d] Eval cache built (preprocessed): %d samples (indices %d..%d)",
                self._rank,
                len(cache),
                start_idx,
                ds_len - 1,
            )
            return

        if self._dataset is None:
            logger.warning("No dataset; eval cache not built.")
            self._eval_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            return

        import json as _json

        ds_len = len(self._dataset)
        num_samples = min(num_samples, ds_len)
        start_idx = ds_len - num_samples
        max_len = self.config.policy.max_total_sequence_length
        self._tokenizer.padding_side = "right"

        cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for idx in range(start_idx, ds_len):
            s = self._dataset[idx]
            convs = s.get("conversations")
            if convs and isinstance(convs, list) and isinstance(convs[0], dict):
                if hasattr(self._tokenizer, "apply_chat_template"):
                    ROLE_MAP = {"human": "user", "gpt": "assistant"}
                    normalized = []
                    for m in convs:
                        if "role" in m and "content" in m:
                            normalized.append(m)
                        elif "from" in m and "value" in m:
                            normalized.append(
                                {
                                    "role": ROLE_MAP.get(m["from"], m["from"]),
                                    "content": m["value"],
                                }
                            )
                        else:
                            normalized.append(m)
                    text = self._tokenizer.apply_chat_template(
                        normalized, tokenize=False, add_generation_prompt=False
                    )
                else:
                    text = "\n".join(
                        m.get("content", "") for m in convs if isinstance(m, dict)
                    )
            else:
                raw = (
                    s.get("prompt")
                    or s.get("question")
                    or s.get("input")
                    or s.get("problem")
                    or ""
                )
                answer = (
                    s.get("answer")
                    or s.get("solution")
                    or s.get("generated_solution")
                    or s.get("target")
                    or s.get("output")
                    or ""
                )
                if isinstance(raw, list):
                    text = "\n".join(
                        m.get("content", "") for m in raw if isinstance(m, dict)
                    )
                elif isinstance(raw, str) and raw.startswith("["):
                    try:
                        msgs = _json.loads(raw)
                        text = "\n".join(
                            m.get("content", "") for m in msgs if isinstance(m, dict)
                        )
                    except (_json.JSONDecodeError, TypeError):
                        text = raw
                else:
                    text = str(raw)
                if answer:
                    text = text + " " + str(answer)

            enc = self._tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=max_len - 1,
                return_tensors="pt",
            )
            cache.append(
                (enc["input_ids"], enc["attention_mask"], enc["attention_mask"].clone())
            )

        self._eval_cache = cache
        logger.info(
            "[rank %d] Eval cache built: %d samples (indices %d..%d)",
            self._rank,
            len(cache),
            start_idx,
            ds_len - 1,
        )

    def run_validation(self) -> dict[str, float]:
        """Run eval on cached samples. Returns eval/* metrics.

        Must run on all ranks (teacher forward + FSDP need collective ops).
        """
        if not hasattr(self, "_eval_cache") or not self._eval_cache:
            return {}

        spec_cfg = self.config.algorithm.spec_distill
        draft_dtype = next(self._draft_model.parameters()).dtype
        eval_cfg = self.config.eval
        mb_size = eval_cfg.micro_batch_size

        self._draft_model.eval()

        all_losses: list[list[float]] = []
        all_accs: list[list[float]] = []

        # Batch eval samples
        all_ids = [c[0] for c in self._eval_cache]
        all_masks = [c[1] for c in self._eval_cache]
        all_lm = [c[2] for c in self._eval_cache]

        for mb_start in range(0, len(all_ids), mb_size):
            mb_end = min(mb_start + mb_size, len(all_ids))
            mb_ids_list = all_ids[mb_start:mb_end]
            mb_masks_list = all_masks[mb_start:mb_end]
            mb_lm_list = all_lm[mb_start:mb_end]

            max_len = max(ids.shape[1] for ids in mb_ids_list)
            padded_ids = []
            padded_masks = []
            padded_lm = []
            for ids, mask, lm in zip(mb_ids_list, mb_masks_list, mb_lm_list):
                pad_len = max_len - ids.shape[1]
                if pad_len > 0:
                    ids = F.pad(ids, (0, pad_len), value=self._tokenizer.pad_token_id)
                    mask = F.pad(mask, (0, pad_len), value=0)
                    lm = F.pad(lm, (0, pad_len), value=0)
                padded_ids.append(ids)
                padded_masks.append(mask)
                padded_lm.append(lm)

            input_ids = torch.cat(padded_ids, dim=0).to(self._device)
            attention_mask = torch.cat(padded_masks, dim=0).to(self._device)
            eval_loss_mask = torch.cat(padded_lm, dim=0).to(self._device)

            teacher_data = self._teacher_forward(input_ids, attention_mask)

            with torch.no_grad():
                aux_hidden = teacher_data["hidden_states"].to(
                    device=self._device, dtype=draft_dtype
                )
                t_ids = teacher_data["input_ids"].to(self._device)
                t_ids = torch.cat([t_ids[:, 1:], torch.zeros_like(t_ids[:, :1])], dim=1)
                lm_head_w = self._lm_head_weight.to(
                    device=self._device, dtype=draft_dtype
                )
                embed_w = self._embed_weight.to(device=self._device, dtype=draft_dtype)
                T_teacher = aux_hidden.shape[1]
                token_embeds = F.embedding(t_ids, embed_w)
                if token_embeds.shape[1] != T_teacher:
                    token_embeds = token_embeds[:, :T_teacher]
                if t_ids.shape[1] != T_teacher:
                    t_ids = t_ids[:, :T_teacher]
                loss_mask = eval_loss_mask
                if loss_mask.shape[1] != T_teacher:
                    loss_mask = loss_mask[:, :T_teacher]

                target_hs = teacher_data.get("last_hidden_states")
                if target_hs is not None:
                    target_hs = target_hs.to(device=self._device, dtype=draft_dtype)
                    target_hs = torch.cat(
                        [target_hs[:, 1:], torch.zeros_like(target_hs[:, :1])], dim=1
                    )
                    nan_mask = torch.isnan(target_hs).any(dim=-1)
                    if nan_mask.any():
                        loss_mask = loss_mask.clone()
                        loss_mask[:, : nan_mask.shape[1]][nan_mask] = 0
                        target_hs = target_hs.nan_to_num_(0.0)
                    if self._norm_weight is not None:
                        nw = self._norm_weight.to(
                            device=self._device, dtype=draft_dtype
                        )
                        ths_f32 = target_hs.float()
                        variance = ths_f32.pow(2).mean(-1, keepdim=True)
                        target_hs = (
                            ths_f32 * torch.rsqrt(variance + self._norm_eps)
                        ).to(draft_dtype) * nw

                eval_attn_mask = attention_mask.to(self._device)
                if eval_attn_mask.shape[1] != T_teacher:
                    eval_attn_mask = eval_attn_mask[:, :T_teacher]
                result = self._draft_model(
                    token_embeds=token_embeds,
                    aux_hidden_states=aux_hidden,
                    teacher_lm_head_weight=lm_head_w,
                    embed_weight=embed_w,
                    loss_mask=loss_mask,
                    target_ids=t_ids,
                    loss_type=spec_cfg.loss_type,
                    target_hidden_states=target_hs,
                    attention_mask=eval_attn_mask,
                )

            step_losses = [float(l.detach()) for l in result["losses"]]
            step_accs = [float(a.detach()) for a in result["accuracies"]]
            all_losses.append(step_losses)
            all_accs.append(step_accs)

            del teacher_data, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._draft_model.train()

        num_positions = max(len(row) for row in all_losses) if all_losses else 0
        avg_loss_per_pos = []
        avg_acc_per_pos = []
        for i in range(num_positions):
            vals = [row[i] for row in all_losses if i < len(row)]
            avg_loss_per_pos.append(sum(vals) / len(vals) if vals else 0.0)
            vals = [row[i] for row in all_accs if i < len(row)]
            avg_acc_per_pos.append(sum(vals) / len(vals) if vals else 0.0)

        # All-reduce across ranks
        if self._is_distributed:
            for i in range(num_positions):
                for arr in (avg_loss_per_pos, avg_acc_per_pos):
                    t = torch.tensor(arr[i], dtype=torch.float64, device=self._device)
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
                    arr[i] = float(t.item())

        # Compute weighted loss and simulated accept length
        total_loss = 0.0
        total_weight = 0.0
        for i, l in enumerate(avg_loss_per_pos):
            w = spec_cfg.position_decay**i
            total_loss += w * l
            total_weight += w

        cum_prod = 1.0
        simulated_acc_len = 0.0
        for acc in avg_acc_per_pos:
            cum_prod *= acc
            simulated_acc_len += cum_prod

        metrics: dict[str, float] = {
            "eval/loss": total_loss / total_weight if total_weight > 0 else 0.0,
            "eval/simulated_acc_len": simulated_acc_len,
        }
        for i, (l, a) in enumerate(zip(avg_loss_per_pos, avg_acc_per_pos)):
            metrics[f"eval/step_{i}_loss"] = l
            metrics[f"eval/step_{i}_acc"] = a

        return metrics

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _get_batch_sequences(
        self, step: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (input_ids, attention_mask, loss_mask) for a training step.

        When preprocessed data is available, loss_mask is the assistant-only mask.
        Otherwise, loss_mask = attention_mask (legacy behavior).
        """
        import json as _json

        bs = max(1, self.config.policy.train_global_batch_size)

        if self._preprocessed is not None:
            ds_len = len(self._preprocessed)
            start = (step * bs) % ds_len
            indices = [(start + i) % ds_len for i in range(bs)]
            max_len = self.config.policy.max_total_sequence_length

            batch_ids = []
            batch_loss_masks = []
            for idx in indices:
                item = self._preprocessed[idx]
                ids = item["input_ids"]
                if isinstance(ids, list):
                    ids = torch.tensor(ids, dtype=torch.long)
                lm = torch.tensor(item["loss_mask"], dtype=torch.long)
                if len(lm) > len(ids):
                    lm = lm[: len(ids)]
                elif len(lm) < len(ids):
                    lm = torch.cat(
                        [lm, torch.zeros(len(ids) - len(lm), dtype=torch.long)]
                    )
                if lm.sum() < 1:
                    replacement = (idx + bs) % ds_len
                    for _ in range(ds_len):
                        r_item = self._preprocessed[replacement]
                        r_ids = r_item["input_ids"]
                        if isinstance(r_ids, list):
                            r_ids = torch.tensor(r_ids, dtype=torch.long)
                        r_lm = torch.tensor(r_item["loss_mask"], dtype=torch.long)
                        if len(r_lm) > len(r_ids):
                            r_lm = r_lm[: len(r_ids)]
                        elif len(r_lm) < len(r_ids):
                            r_lm = torch.cat(
                                [
                                    r_lm,
                                    torch.zeros(
                                        len(r_ids) - len(r_lm), dtype=torch.long
                                    ),
                                ]
                            )
                        if r_lm.sum() >= 1:
                            ids, lm = r_ids, r_lm
                            break
                        replacement = (replacement + 1) % ds_len
                batch_ids.append(ids)
                batch_loss_masks.append(lm)

            max_seq = min(max(len(t) for t in batch_ids), max_len - 1)
            padded_ids = []
            padded_masks = []
            padded_loss_masks = []
            pad_id = self._tokenizer.pad_token_id or 0
            for ids, lm in zip(batch_ids, batch_loss_masks):
                ids = ids[:max_seq]
                lm = lm[:max_seq]
                pad_len = max_seq - len(ids)
                if pad_len > 0:
                    ids = torch.cat(
                        [ids, torch.full((pad_len,), pad_id, dtype=torch.long)]
                    )
                    lm = torch.cat([lm, torch.zeros(pad_len, dtype=torch.long)])
                attn = (ids != pad_id).long()
                padded_ids.append(ids.unsqueeze(0))
                padded_masks.append(attn.unsqueeze(0))
                padded_loss_masks.append(lm.unsqueeze(0))

            return (
                torch.cat(padded_ids, dim=0),
                torch.cat(padded_masks, dim=0),
                torch.cat(padded_loss_masks, dim=0),
            )

        if self._dataset is None:
            texts = [
                f"What is {i + step * bs} + {i + step * bs + 1}? The answer is {2 * (i + step * bs) + 1}."
                for i in range(bs)
            ]
        else:
            dataset_len = len(self._dataset)
            start = (step * bs) % dataset_len
            indices = [(start + i) % dataset_len for i in range(bs)]
            samples = [self._dataset[idx] for idx in indices]
            texts = []
            for s in samples:
                convs = s.get("conversations")
                if convs and isinstance(convs, list) and isinstance(convs[0], dict):
                    if hasattr(self._tokenizer, "apply_chat_template"):
                        ROLE_MAP = {"human": "user", "gpt": "assistant"}
                        normalized = []
                        for m in convs:
                            if "role" in m and "content" in m:
                                normalized.append(m)
                            elif "from" in m and "value" in m:
                                normalized.append(
                                    {
                                        "role": ROLE_MAP.get(m["from"], m["from"]),
                                        "content": m["value"],
                                    }
                                )
                            else:
                                normalized.append(m)
                        text = self._tokenizer.apply_chat_template(
                            normalized, tokenize=False, add_generation_prompt=False
                        )
                    else:
                        text = "\n".join(
                            m.get("content", "") for m in convs if isinstance(m, dict)
                        )
                    texts.append(text)
                    continue
                raw = (
                    s.get("prompt")
                    or s.get("question")
                    or s.get("input")
                    or s.get("problem")
                    or ""
                )
                answer = (
                    s.get("answer")
                    or s.get("solution")
                    or s.get("generated_solution")
                    or s.get("target")
                    or s.get("output")
                    or ""
                )
                if isinstance(raw, list):
                    text = "\n".join(
                        m.get("content", "") for m in raw if isinstance(m, dict)
                    )
                elif isinstance(raw, str) and raw.startswith("["):
                    try:
                        msgs = _json.loads(raw)
                        text = "\n".join(
                            m.get("content", "") for m in msgs if isinstance(m, dict)
                        )
                    except (_json.JSONDecodeError, TypeError):
                        text = raw
                else:
                    text = str(raw)
                if answer:
                    text = text + " " + str(answer)
                texts.append(text)

        max_len = self.config.policy.max_total_sequence_length
        self._tokenizer.padding_side = "right"
        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len - 1,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        return input_ids, attention_mask, attention_mask.clone()

    # ------------------------------------------------------------------
    # Teacher forward
    # ------------------------------------------------------------------

    def _teacher_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run teacher on dataset sequences, return hidden states.

        Returns a dict with:
        - ``"hidden_states"``: 3 concatenated aux hidden states ``[B, T, 3*D]``
          from layers [1, N//2-1, N-4] (Eagle3 convention)
        - ``"token_embeds"``: input embeddings ``[B, T, D]``
        - ``"input_ids"``: the original input ids
        """
        backend = self.config.algorithm.teacher.inference_backend
        if backend == "vllm":
            return self._teacher_forward_vllm(input_ids, attention_mask)
        if backend == "atom":
            return self._teacher_forward_atom(input_ids, attention_mask)
        return self._teacher_forward_hf(input_ids, attention_mask)

    def _teacher_forward_atom(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher forward via ATOM engine + Mooncake TCP (rank 0, then broadcast).

        Returns 3 aux hidden states [B,T,3*D], token_embeds [B,T,D],
        last_hidden_states [B,T,D] — same format as vLLM path.
        """
        rank0_data = self._teacher_inference_rank0(input_ids, attention_mask)
        return self._teacher_broadcast(rank0_data, input_ids)

    def _teacher_forward_hf(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher forward via HuggingFace (original path).

        Extracts 3 aux hidden states from layers [1, N//2-1, N-4] and
        concatenates them to [B, T, 3*H].  HF ``output_hidden_states``
        returns index 0 = embedding, index i+1 = layer i output, so the
        HF indices are [2, N//2, N-3].
        """
        if self._teacher_model is None:
            raise RuntimeError("Teacher model not loaded.")

        micro_bs = max(1, self.config.algorithm.opd.teacher_micro_batch_size)

        all_hidden: list[torch.Tensor] = []
        all_embeds: list[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, input_ids.shape[0], micro_bs):
                end = min(start + micro_bs, input_ids.shape[0])
                ids_chunk = input_ids[start:end].to(self._device)
                mask_chunk = attention_mask[start:end].to(self._device)

                outputs = self._teacher_model(
                    input_ids=ids_chunk,
                    attention_mask=mask_chunk,
                    output_hidden_states=True,
                )

                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    all_embeds.append(outputs.hidden_states[0].cpu())
                    # HF hidden_states: [0]=embed, [i+1]=layer_i output
                    # Aux layers: [1, N//2-1, N-4]
                    # HF indices: [2, N//2, N-3]
                    num_layers = len(outputs.hidden_states) - 1
                    hf_aux_ids = [2, num_layers // 2, num_layers - 3]
                    aux_concat = torch.cat(
                        [outputs.hidden_states[i].cpu() for i in hf_aux_ids],
                        dim=-1,
                    )  # [micro_B, T, 3*H]
                    all_hidden.append(aux_concat)
                else:
                    raise RuntimeError("Teacher model did not return hidden_states.")

                del outputs

        return {
            "hidden_states": torch.cat(all_hidden, dim=0),
            "token_embeds": torch.cat(all_embeds, dim=0),
            "input_ids": input_ids,
        }

    # ------------------------------------------------------------------
    # Draft model train step
    # ------------------------------------------------------------------

    def _debug_sync(self, tag: str, step: int) -> None:
        """Sync GPU and log checkpoint for crash debugging."""
        torch.cuda.synchronize()
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_res = torch.cuda.memory_reserved() / 1e9
        logger.warning(
            "[rank %d] DEBUG step=%d checkpoint: %s OK (%.1fG/%.1fG)",
            self._rank,
            step,
            tag,
            mem_alloc,
            mem_res,
        )

    def _install_backward_hooks(self) -> None:
        """Register backward hooks on each submodule of the draft model for crash debugging."""
        if getattr(self, "_bwd_hooks_installed", False):
            return
        self._bwd_hooks_installed = True
        rank = self._rank

        def _make_hook(name):
            def hook(module, grad_input, grad_output):
                torch.cuda.synchronize()
                mem = torch.cuda.memory_allocated() / 1e9
                logger.warning("[rank %d] BWD hook: %s OK (%.1fG)", rank, name, mem)

            return hook

        model = self._draft_model
        for name, mod in model.named_modules():
            if name == "":
                continue
            mod.register_full_backward_hook(_make_hook(name))
        logger.warning("[rank %d] Installed backward debug hooks on draft model", rank)

    def _train_step_eagle3(
        self,
        teacher_data: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Eagle3 draft model training step (single micro-batch)."""
        step = self.global_step
        # Probe: first few steps log fine-grained progress + peak mem to localize hangs/OOM.
        _probe = step < 3

        def _mem(tag: str) -> None:
            if not _probe:
                return
            try:
                alloc = torch.cuda.memory_allocated(self._device) / (1024**3)
                peak = torch.cuda.max_memory_allocated(self._device) / (1024**3)
                logger.info(
                    "[rank %d] step=%d eagle3-probe %s: alloc=%.2fGiB peak=%.2fGiB",
                    self._rank,
                    step,
                    tag,
                    alloc,
                    peak,
                )
            except RuntimeError:
                logger.info("[rank %d] step=%d eagle3-probe %s", self._rank, step, tag)

        _mem("enter")
        draft_dtype = next(self._draft_model.parameters()).dtype
        aux_hidden = teacher_data["hidden_states"].to(
            device=self._device, dtype=draft_dtype
        )  # [B, T, 3*H]
        input_ids = teacher_data["input_ids"].to(self._device)
        input_ids = torch.cat(
            [input_ids[:, 1:], torch.zeros_like(input_ids[:, :1])], dim=1
        )
        _mem(
            f"inputs-on-gpu (aux={tuple(aux_hidden.shape)} ids={tuple(input_ids.shape)})"
        )

        if not hasattr(self, "_lm_head_w_gpu") or self._lm_head_w_gpu is None:
            _mem("before-lazy-lm_head-materialize")
            self._lm_head_w_gpu = self._lm_head_weight.to(
                device=self._device, dtype=draft_dtype
            )
            self._embed_w_gpu = self._embed_weight.to(
                device=self._device, dtype=draft_dtype
            )
            _mem("after-lazy-lm_head-materialize")
        lm_head_w = self._lm_head_w_gpu
        embed_w = self._embed_w_gpu
        token_embeds = torch.nn.functional.embedding(input_ids, embed_w)
        _mem("after-embed")

        if loss_mask is not None:
            loss_mask = loss_mask.to(self._device)
        else:
            loss_mask = attention_mask.to(self._device)
        spec_cfg = self.config.algorithm.spec_distill
        target_hs = teacher_data.get("last_hidden_states")

        aux_nan = torch.isnan(aux_hidden).any(dim=-1)  # [B, T]
        target_nan = torch.zeros_like(aux_nan)
        if target_hs is not None:
            target_hs = target_hs.to(device=self._device, dtype=draft_dtype)
            target_hs = torch.cat(
                [target_hs[:, 1:], torch.zeros_like(target_hs[:, :1])], dim=1
            )
            target_nan = torch.isnan(target_hs).any(dim=-1)

        nan_mask = aux_nan | target_nan
        if nan_mask.any():
            nan_count = nan_mask.sum().item()
            aux_only = (aux_nan & ~target_nan).sum().item()
            tgt_only = (target_nan & ~aux_nan).sum().item()
            both = (aux_nan & target_nan).sum().item()
            loss_mask = loss_mask.clone()
            loss_mask[:, : nan_mask.shape[1]][nan_mask] = 0
            aux_hidden = aux_hidden.nan_to_num_(0.0)
            if target_hs is not None:
                target_hs = target_hs.nan_to_num_(0.0)
            logger.warning(
                "Masked %d teacher NaN positions (aux_only=%d tgt_only=%d both=%d)",
                nan_count,
                aux_only,
                tgt_only,
                both,
            )

        if target_hs is not None and self._norm_weight is not None:
            nw = self._norm_weight.to(device=self._device, dtype=draft_dtype)
            ths_f32 = target_hs.float()
            variance = ths_f32.pow(2).mean(-1, keepdim=True)
            target_hs = (ths_f32 * torch.rsqrt(variance + self._norm_eps)).to(
                draft_dtype
            ) * nw

        _mem("before-forward")
        result = self._draft_model(
            token_embeds=token_embeds,
            aux_hidden_states=aux_hidden,
            teacher_lm_head_weight=lm_head_w,
            embed_weight=embed_w,
            loss_mask=loss_mask,
            target_ids=input_ids,
            loss_type=spec_cfg.loss_type,
            target_hidden_states=target_hs,
            attention_mask=attention_mask.to(self._device),
        )
        _mem("after-forward")

        total_loss = torch.tensor(0.0, device=self._device)
        metrics: dict[str, float] = {}

        for i, (step_loss, step_acc) in enumerate(
            zip(result["losses"], result["accuracies"])
        ):
            weight = spec_cfg.position_decay**i
            total_loss = total_loss + weight * step_loss
            metrics[f"step_{i}_loss"] = float(step_loss.detach())
            metrics[f"step_{i}_acc"] = float(step_acc.detach())

        num_accum = getattr(self, "_grad_accum_steps", 1)
        scaled_loss = total_loss / num_accum

        _mem("before-backward")
        scaled_loss.backward()
        _mem("after-backward")

        metrics["loss"] = float(total_loss.detach())
        return metrics

    def _train_step_dflash(
        self,
        teacher_data: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """DFlash draft model training step (single micro-batch)."""
        teacher_hidden = teacher_data["hidden_states"].to(self._device)
        input_ids = teacher_data["input_ids"].to(self._device)
        lm_head_w = self._lm_head_weight.to(self._device)
        if loss_mask is not None:
            loss_mask = loss_mask.to(self._device)
        else:
            loss_mask = attention_mask.to(self._device)

        spec_cfg = self.config.algorithm.spec_distill

        result = self._draft_model(
            teacher_hidden_states=teacher_hidden,
            lm_head_weight=lm_head_w,
            loss_mask=loss_mask,
            target_ids=input_ids,
            loss_decay_gamma=spec_cfg.loss_decay_gamma,
        )

        loss = result.get("loss")
        if loss is None:
            return {"loss": 0.0}

        num_accum = getattr(self, "_grad_accum_steps", 1)
        (loss / num_accum).backward()

        metrics: dict[str, float] = {"loss": float(loss.detach())}
        if "accuracy" in result:
            metrics["accuracy"] = float(result["accuracy"].detach())
        return metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Spec Distill loop: teacher forward on dataset -> draft model train."""
        if (
            self._draft_model is None
            and self._teacher_model is None
            and self._teacher_engine is None
        ):
            raise RuntimeError("Call setup() before train().")

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_begin(self)

        total_steps = int(self.config.num_training_steps)
        start_step = self.global_step
        if start_step > 0:
            logger.info(
                "[rank %d] Resuming from step %d / %d",
                self._rank,
                start_step,
                total_steps,
            )
        logger.info(
            "[rank %d] SpecDistill train loop: total_steps=%d", self._rank, total_steps
        )
        spec_cfg = self.config.algorithm.spec_distill
        draft_type = spec_cfg.draft_type.lower()

        use_vllm = self.config.algorithm.teacher.inference_backend in ("vllm", "atom")
        prefetcher = (
            _TeacherPrefetcher(self) if (use_vllm and self._rank == 0) else None
        )
        shm_writer = (
            _ShmWriterThread(prefetcher, self._TEACHER_KEYS)
            if (use_vllm and self._rank == 0)
            else None
        )
        shm_loader = (
            _ShmLoaderThread(self._rank, self._TEACHER_KEYS) if use_vllm else None
        )

        if use_vllm:
            if self._rank == 0:
                for _slot in range(_SHM_SLOTS):
                    _sd = f"/dev/shm/_teacher_{_slot}"
                    if os.path.exists(_sd):
                        for _fn in os.listdir(_sd):
                            if _fn.startswith("READY_"):
                                try:
                                    os.remove(os.path.join(_sd, _fn))
                                except FileNotFoundError:
                                    pass
            if self._is_distributed:
                torch.distributed.barrier()

            for s in range(start_step, min(start_step + _SHM_SLOTS, total_steps)):
                if prefetcher is not None:
                    prefetcher.prefetch(s)
                _slot = s % _SHM_SLOTS
                if shm_writer is not None:
                    shm_writer.submit(s, _slot)
                if shm_loader is not None:
                    shm_loader.submit(s, _slot)

            if shm_writer is not None:
                _pipeline_data = shm_writer.get_rank0()
            elif shm_loader is not None:
                _pipeline_data = shm_loader.get()
            else:
                _pipeline_data = None
            logger.info("[rank %d] Pipeline primed — first step data ready", self._rank)
            from concurrent.futures import ThreadPoolExecutor

            _loader_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="shm-async"
            )
            _loader_future = None
        elif prefetcher is not None:
            prefetcher.prefetch(start_step)
            if start_step + 1 < total_steps:
                prefetcher.prefetch(start_step + 1)

        for step in range(start_step, total_steps):
            step_start = time.time()
            self.global_step = step
            if self._rank == 0:
                for cb in self.callbacks:
                    cb.on_step_begin(self, step)

            t0 = time.time()

            if use_vllm and shm_loader is not None:
                if step == start_step:
                    loaded = _pipeline_data
                elif shm_writer is not None:
                    loaded = _loader_future.result()
                else:
                    loaded = _loader_future.result()
                cpu_data = {k: loaded[k] for k in self._TEACHER_KEYS}
                input_ids = loaded["input_ids"]
                attention_mask = loaded["attention_mask"]
                loss_mask = loaded.get("loss_mask", attention_mask.clone())
                _consumed_slot = loaded["_slot"]
                teacher_time = time.time() - t0
                # Ensure every rank has finished copying this slot out of SHM
                # before rank 0 frees it; otherwise the writer can overwrite
                # the slot (deleting READY_{step}) before slower ranks' loader
                # observes the sentinel, leaving them stuck in _wait_sentinel.
                if self._is_distributed:
                    torch.distributed.barrier()
                if shm_writer is not None:
                    shm_writer.release_slot(_consumed_slot)
                _next_step = step + _SHM_SLOTS
                if _next_step < total_steps:
                    if prefetcher is not None:
                        prefetcher.prefetch(_next_step)
                    if shm_writer is not None:
                        shm_writer.submit(_next_step, _consumed_slot)
                    shm_loader.submit(_next_step, _consumed_slot)
                if step + 1 < total_steps:
                    if shm_writer is not None:
                        _loader_future = _loader_executor.submit(shm_writer.get_rank0)
                    else:
                        _loader_future = _loader_executor.submit(shm_loader.get)
            else:
                input_ids, attention_mask, loss_mask = self._get_batch_sequences(step)
                if prefetcher is not None:
                    rank0_data, _, _, _ = prefetcher.get()
                elif use_vllm:
                    rank0_data = self._teacher_inference_rank0(
                        input_ids,
                        attention_mask,
                        recv_device=torch.device("cpu"),
                    )
                else:
                    teacher_data = self._teacher_forward(input_ids, attention_mask)
                if use_vllm and rank0_data is not None and "input_ids" in rank0_data:
                    input_ids = rank0_data["input_ids"]
                teacher_time = time.time() - t0
                if prefetcher is not None and step + 2 < total_steps:
                    prefetcher.prefetch(step + 2)

            # --- Micro-batch streaming loop (double-buffer + async broadcast) ---
            _dbg = step <= start_step + 3
            t1 = time.time()
            B, T = input_ids.shape[0], input_ids.shape[1]
            configured_mb = getattr(self.config.policy, "train_micro_batch_size", 0)
            if configured_mb > 0:
                mb = min(configured_mb, B)
            else:
                adaptive_mb = max(1, 16384 // max(T, 1))
                mb = max(1, min(adaptive_mb, B))
            world_size = self._world_size if self._is_distributed else 1
            num_accum = (B + mb - 1) // mb
            local_accum = (num_accum + world_size - 1) // world_size
            self._grad_accum_steps = local_accum
            self._optimizer.zero_grad(set_to_none=True)

            metrics_accum: dict[str, float] = {}
            accum_count = 0
            self._draft_model.train()

            if use_vllm:
                # This rank's micro-batch indices (round-robin)
                my_mbs = [i for i in range(num_accum) if i % world_size == self._rank]

                # Discover dimensions from shared data
                D_hidden = cpu_data["hidden_states"].shape[2]
                D_embed = cpu_data["token_embeds"].shape[2]
                D_last = cpu_data["last_hidden_states"].shape[2]

                # Ping-pong GPU buffers
                def _alloc_gpu_buf(
                    D_embed=D_embed, D_hidden=D_hidden, D_last=D_last, T=T, mb=mb
                ):
                    return {
                        "hidden_states": torch.empty(
                            mb, T, D_hidden, dtype=torch.bfloat16, device=self._device
                        ),
                        "token_embeds": torch.empty(
                            mb, T, D_embed, dtype=torch.bfloat16, device=self._device
                        ),
                        "last_hidden_states": torch.empty(
                            mb, T, D_last, dtype=torch.bfloat16, device=self._device
                        ),
                    }

                bufs = [_alloc_gpu_buf(), _alloc_gpu_buf()]
                xfer_stream = torch.cuda.Stream(device=self._device)

                # Pre-fill buf[0] with first micro-batch (sync, default stream)
                if my_mbs:
                    mb0 = my_mbs[0]
                    s0 = mb0 * mb
                    e0 = min(s0 + mb, B)
                    for key in self._TEACHER_KEYS:
                        bufs[0][key][: e0 - s0].copy_(
                            cpu_data[key][s0:e0].to(self._device)
                        )

                _use_grad_sync = (
                    self._is_distributed
                    and self._world_size > 1
                    and len(my_mbs) > 1
                    and hasattr(self._draft_model, "set_requires_gradient_sync")
                )
                cur = 0
                if _dbg:
                    logger.info(
                        "[rank %d] step=%d micro-batch loop: B=%d T=%d mb=%d num_accum=%d my_mbs=%d",
                        self._rank,
                        step,
                        B,
                        T,
                        mb,
                        num_accum,
                        len(my_mbs),
                    )
                for local_idx, mb_idx in enumerate(my_mbs):
                    if _use_grad_sync:
                        self._draft_model.set_requires_gradient_sync(
                            local_idx == len(my_mbs) - 1
                        )
                    mb_start = mb_idx * mb
                    mb_end = min(mb_start + mb, B)
                    actual_mb = mb_end - mb_start
                    nxt = 1 - cur

                    # Async H2D next micro-batch on xfer_stream (no NCCL)
                    if local_idx + 1 < len(my_mbs):
                        next_mb = my_mbs[local_idx + 1]
                        ns = next_mb * mb
                        ne = min(ns + mb, B)
                        with torch.cuda.stream(xfer_stream):
                            for key in self._TEACHER_KEYS:
                                bufs[nxt][key][: ne - ns].copy_(
                                    cpu_data[key][ns:ne].to(self._device),
                                    non_blocking=True,
                                )

                    # Compute on buf[cur] (default stream)
                    mb_ids = input_ids[mb_start:mb_end]
                    mb_mask = attention_mask[mb_start:mb_end]
                    mb_lm = loss_mask[mb_start:mb_end]
                    mb_teacher = {
                        "hidden_states": bufs[cur]["hidden_states"][:actual_mb],
                        "token_embeds": bufs[cur]["token_embeds"][:actual_mb],
                        "input_ids": mb_ids,
                        "last_hidden_states": bufs[cur]["last_hidden_states"][
                            :actual_mb
                        ],
                    }

                    # Clamp into [1, T] defensively. The raw value should always
                    # already satisfy 0 <= sum.max() <= T for a 0/1 mask, but
                    # torch occasionally fires "Truncating the start/stop/step
                    # of slice" UserWarnings here, suggesting a stale traced
                    # shape sees mb_actual_len as out of bounds. The clamp
                    # prevents the trim from ever producing a malformed slice
                    # and silences the symbolic-shape warning regardless.
                    mb_actual_len = int(mb_mask.sum(dim=-1).max().item())
                    mb_actual_len = max(1, min(mb_actual_len, mb_mask.shape[1]))
                    if mb_actual_len < mb_mask.shape[1]:
                        mb_mask = mb_mask[:, :mb_actual_len].contiguous()
                        mb_lm = mb_lm[:, :mb_actual_len].contiguous()
                        trimmed = {}
                        for k, v in mb_teacher.items():
                            if v.dim() >= 2 and v.shape[1] >= mb_actual_len:
                                trimmed[k] = v[:, :mb_actual_len].contiguous()
                            else:
                                trimmed[k] = v
                        mb_teacher = trimmed

                    if _dbg and (local_idx == 0 or local_idx == len(my_mbs) - 1):
                        logger.info(
                            "[rank %d] step=%d mb[%d/%d]: start train_step (T_trim=%d, grad_sync=%s)",
                            self._rank,
                            step,
                            local_idx,
                            len(my_mbs),
                            mb_actual_len,
                            local_idx == len(my_mbs) - 1,
                        )

                    if draft_type == "eagle3":
                        m = self._train_step_eagle3(mb_teacher, mb_mask, mb_lm)
                    else:
                        m = self._train_step_dflash(mb_teacher, mb_mask, mb_lm)

                    if _dbg and (local_idx == 0 or local_idx == len(my_mbs) - 1):
                        logger.info(
                            "[rank %d] step=%d mb[%d/%d]: done",
                            self._rank,
                            step,
                            local_idx,
                            len(my_mbs),
                        )

                    for k, v in m.items():
                        if isinstance(v, float) and not math.isnan(v):
                            metrics_accum[k] = metrics_accum.get(k, 0.0) + v
                    accum_count += 1
                    del mb_teacher, mb_mask, mb_lm

                    if local_idx + 1 < len(my_mbs):
                        torch.cuda.current_stream().wait_stream(xfer_stream)
                    cur = nxt

                del bufs, cpu_data
            else:
                _num_mbs_novm = (B + mb - 1) // mb
                _use_grad_sync_novm = (
                    self._is_distributed
                    and self._world_size > 1
                    and _num_mbs_novm > 1
                    and hasattr(self._draft_model, "set_requires_gradient_sync")
                )
                for mb_idx, mb_start in enumerate(range(0, B, mb)):
                    if _use_grad_sync_novm:
                        self._draft_model.set_requires_gradient_sync(
                            mb_idx == _num_mbs_novm - 1
                        )
                    mb_end = min(mb_start + mb, B)
                    mb_teacher = {
                        k: v[mb_start:mb_end] for k, v in teacher_data.items()
                    }
                    mb_mask = attention_mask[mb_start:mb_end]
                    mb_lm = loss_mask[mb_start:mb_end]
                    mb_actual_len = int(mb_mask.sum(dim=-1).max().item())
                    mb_actual_len = max(1, min(mb_actual_len, mb_mask.shape[1]))
                    if mb_actual_len < mb_mask.shape[1]:
                        mb_mask = mb_mask[:, :mb_actual_len].contiguous()
                        mb_lm = mb_lm[:, :mb_actual_len].contiguous()
                        trimmed = {}
                        for k, v in mb_teacher.items():
                            if v.dim() >= 2 and v.shape[1] >= mb_actual_len:
                                trimmed[k] = v[:, :mb_actual_len].contiguous()
                            else:
                                trimmed[k] = v
                        mb_teacher = trimmed
                    if draft_type == "eagle3":
                        m = self._train_step_eagle3(mb_teacher, mb_mask, mb_lm)
                    else:
                        m = self._train_step_dflash(mb_teacher, mb_mask, mb_lm)
                    for k, v in m.items():
                        if isinstance(v, float) and not math.isnan(v):
                            metrics_accum[k] = metrics_accum.get(k, 0.0) + v
                    accum_count += 1
                    del mb_teacher, mb_mask, mb_lm

            metrics = {k: v / max(accum_count, 1) for k, v in metrics_accum.items()}

            grad_norm = self._optimizer.step_with_grad_norm(
                self.config.policy.max_grad_norm
            )
            self._scheduler.step()

            train_time = time.time() - t1

            metrics["grad_norm"] = float(grad_norm)
            metrics["lr"] = self._optimizer.get_learning_rate()
            metrics["timing/step_s"] = time.time() - step_start
            metrics["timing/teacher_s"] = teacher_time
            metrics["timing/train_s"] = train_time
            metrics["seq/max_len"] = int(input_ids.shape[1])

            if self._is_distributed:

                for k in list(metrics.keys()):
                    t = torch.tensor(
                        metrics[k], dtype=torch.float64, device=self._device
                    )
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
                    metrics[k] = float(t.item())

            self.last_metrics = metrics

            for cb in self.callbacks:
                cb.on_step_end(self, step, metrics)

            if use_vllm and shm_loader is not None:
                del loaded
                logger.info(
                    "[rank %d] step=%d end: slot=%d done",
                    self._rank,
                    step,
                    _consumed_slot,
                )
            elif use_vllm:
                del rank0_data
            else:
                del teacher_data
            del input_ids, attention_mask, loss_mask

        if use_vllm and shm_loader is not None:
            _loader_executor.shutdown(wait=False)
        if shm_writer is not None:
            shm_writer.stop()
        if shm_loader is not None:
            shm_loader.stop()
        if prefetcher is not None:
            prefetcher.stop()

        if self._rank == 0:
            for cb in self.callbacks:
                cb.on_train_end(self)

        logger.info(
            "[rank %d] SpecDistillTrainer.train finished after %d steps.",
            self._rank,
            total_steps,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release all resources.

        Teacher engine must shut down BEFORE the mooncake master so the
        vLLM-side Mooncake producer client disconnects while its master
        is still up; otherwise its background Ping/FetchTasks threads
        enter an infinite RPC_FAIL retry loop and prevent process exit.
        """
        if self._teacher_engine is not None:
            try:
                self._teacher_engine.shutdown()
            # Third-party teacher backends can raise backend-specific errors during
            # best-effort teardown; Mooncake cleanup must still run afterwards.
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[rank %d] teacher_engine.shutdown failed: %s",
                    self._rank,
                    e,
                )
            self._teacher_engine = None
        if self._mooncake_master is not None:
            try:
                self._mooncake_master.shutdown()
            # Shutdown must not mask training completion when the transfer service
            # has already stopped or raises a backend-specific RPC exception.
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[rank %d] mooncake_master.shutdown failed: %s",
                    self._rank,
                    e,
                )
            self._mooncake_master = None
        del self._teacher_model, self._draft_model
        self._teacher_model = None
        self._draft_model = None
        self._lm_head_weight = None
        self._optimizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("[rank %d] SpecDistillTrainer.cleanup complete.", self._rank)


__all__ = ["SpecDistillTrainer"]
