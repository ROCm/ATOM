"""AITER MK1 persistent decoder over ATOM's native KV cache.

ATOM owns request admission, cache allocation, page reservation, block maps,
and ordinary fallback. AITER owns compiled-checkpoint loading, native binaries,
weight lifetime, cache binding, and quantum execution.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger("atom")

PERSISTENT_LAYER_COUNT = 36
PERSISTENT_PHYSICAL_PAGE_SIZE = 16
PERSISTENT_KV_HEADS = 8
PERSISTENT_HEAD_DIM = 64
PERSISTENT_PLANE_BLOCK_BYTES = 16384
PERSISTENT_MAX_QUANTUM_TOKENS = 8
SAMPLING_EPS = 1.0e-5


@dataclass(frozen=True)
class PersistentDecodePlan:
    """Frozen scheduler command consumed by the runner worker."""

    request_id: int
    pending_token: int
    committed_kv_length: int
    max_sequence_length: int
    max_tokens: int
    eos_token_id: int
    ignore_eos: bool
    block_table: tuple[int, ...]


@dataclass(frozen=True)
class PersistentDecodeDecision:
    plan: PersistentDecodePlan | None
    rejection_reason: str | None

    @property
    def selected(self) -> bool:
        return self.plan is not None


def decide_persistent_decode(
    *,
    mode: str,
    seqs: Iterable[Any],
    num_scheduled_tokens: Iterable[int],
    has_queued_work: bool,
    use_spec: bool,
    max_model_len: int,
    eos_token_id: int,
    extra_eos_token_ids: Iterable[int] = (),
) -> PersistentDecodeDecision:
    """Apply the batch-one persistent-decode admission contract."""

    if mode == "off":
        return PersistentDecodeDecision(None, "disabled")
    seqs = tuple(seqs)
    scheduled = tuple(int(value) for value in num_scheduled_tokens)
    if use_spec:
        return PersistentDecodeDecision(None, "speculative_decode")
    if has_queued_work:
        return PersistentDecodeDecision(None, "queued_work")
    if len(seqs) != 1 or len(scheduled) != 1:
        return PersistentDecodeDecision(None, "unsupported_decode_batch")
    if scheduled[0] != 1:
        return PersistentDecodeDecision(None, "tokens_per_query")

    seq = seqs[0]
    if float(seq.temperature) > SAMPLING_EPS:
        return PersistentDecodeDecision(None, "non_greedy")
    if bool(seq.return_logprobs):
        return PersistentDecodeDecision(None, "logprobs")
    if getattr(seq, "stop_strings", None):
        return PersistentDecodeDecision(None, "stop_strings")
    if getattr(seq, "stop_token_sequences", None):
        return PersistentDecodeDecision(None, "stop_token_sequences")
    if getattr(seq, "kv_transfer_params", None):
        return PersistentDecodeDecision(None, "kv_transfer")

    committed = int(seq.num_tokens) - 1
    if committed < 0 or committed >= int(max_model_len) - 1:
        return PersistentDecodeDecision(None, "max_sequence_length")
    remaining = int(seq.max_tokens) - len(seq.output_tokens)
    if remaining < 1:
        return PersistentDecodeDecision(None, "no_output_remaining")

    eos_ids = {int(eos_token_id), *(int(token) for token in extra_eos_token_ids)}
    eos_ids.discard(-1)
    quantum_cap = PERSISTENT_MAX_QUANTUM_TOKENS
    if len(eos_ids) > 1 and not seq.ignore_eos:
        quantum_cap = 1
    quantum = min(
        quantum_cap,
        remaining,
        int(max_model_len) - committed - 1,
    )
    if quantum < 1:
        return PersistentDecodeDecision(None, "no_output_remaining")

    return PersistentDecodeDecision(
        PersistentDecodePlan(
            request_id=int(seq.id),
            pending_token=int(seq.last_token),
            committed_kv_length=committed,
            max_sequence_length=int(max_model_len),
            max_tokens=quantum,
            eos_token_id=int(eos_token_id),
            ignore_eos=bool(seq.ignore_eos),
            block_table=tuple(int(block) for block in seq.block_table),
        ),
        None,
    )


class AtomPersistentDecoder:
    """Translate ATOM scheduler/cache state into the AITER MK1 provider ABI."""

    def __init__(self, runner) -> None:
        from aiter.MK1 import (
            BackendLaunchError,
            KVCacheBinding,
            MK1Config,
            PersistentDecoder as AiterPersistentDecoder,
            PrelaunchError,
            QuantumRequest,
        )

        self._backend_error = BackendLaunchError
        self._prelaunch_error = PrelaunchError
        self._quantum_request = QuantumRequest
        self.runner = runner
        self.config = runner.config
        self.healthy = False

        mk1_config = MK1Config(
            device=torch.cuda.current_device(),
            max_sequence_length=int(self.config.max_model_len),
            cache_scalar="bfloat16",
            batch_size=1,
            mode=self.config.persistent_decoder,
        )
        logger.info(
            "Loading ATOM persistent checkpoint through AITER provider: %s",
            self.config.persistent_decoder_checkpoint,
        )
        self.native = AiterPersistentDecoder.from_checkpoint(
            mk1_config,
            self.config.persistent_decoder_checkpoint,
        )

        expected_plane_shape = (
            int(runner.num_physical_kvcache_blocks),
            PERSISTENT_PHYSICAL_PAGE_SIZE,
            PERSISTENT_KV_HEADS,
            PERSISTENT_HEAD_DIM,
        )
        key_planes = tuple(runner.kv_cache[0, index] for index in range(36))
        value_planes = tuple(runner.kv_cache[1, index] for index in range(36))
        for index, (key, value) in enumerate(zip(key_planes, value_planes)):
            if not key.is_contiguous() or not value.is_contiguous():
                raise RuntimeError(
                    f"ATOM cache layer {index} must have contiguous K/V planes"
                )
            if (
                tuple(key.shape) != expected_plane_shape
                or tuple(value.shape) != expected_plane_shape
            ):
                raise RuntimeError(
                    f"ATOM cache layer {index} has incompatible geometry "
                    f"K={tuple(key.shape)} V={tuple(value.shape)}"
                )
        key_bytes = tuple(plane.view(torch.uint8).flatten() for plane in key_planes)
        value_bytes = tuple(plane.view(torch.uint8).flatten() for plane in value_planes)
        expected_plane_bytes = (
            int(runner.num_physical_kvcache_blocks) * PERSISTENT_PLANE_BLOCK_BYTES
        )
        if any(
            plane.numel() != expected_plane_bytes for plane in key_bytes + value_bytes
        ):
            raise RuntimeError("ATOM cache layer has incompatible plane byte length")

        self.native.bind_cache(
            KVCacheBinding(
                key_planes=key_bytes,
                value_planes=value_bytes,
                block_counts=(int(runner.num_physical_kvcache_blocks),) * 36,
                block_strides=(PERSISTENT_PLANE_BLOCK_BYTES,) * 36,
                pools=(1,) * 36,
            )
        )
        self.rope_cosine, self.rope_sine = self._rope_tables()
        self.healthy = True
        checkpoint = self.native.checkpoint_info() or {}
        logger.info(
            "ATOM persistent decoder ready: checkpoint_bytes=%s, cache_blocks=%d",
            checkpoint.get("persistent_bytes", "unknown"),
            runner.num_physical_kvcache_blocks,
        )

    def _rope_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        rotary = self.runner.model.model.layers[0].self_attn.rotary_emb
        cosine_source = rotary.cos_cache.squeeze()
        sine_source = rotary.sin_cache.squeeze()
        if cosine_source.ndim != 2 or cosine_source.shape[1] != 32:
            raise RuntimeError(
                f"unsupported GPT-OSS RoPE cache shape {tuple(cosine_source.shape)}"
            )
        length = min(int(self.config.max_model_len), cosine_source.shape[0])
        cosine = torch.zeros(
            (length, 64), dtype=torch.bfloat16, device=self.runner.device
        )
        sine = torch.zeros_like(cosine)
        cosine[:, :32] = cosine_source[:length].to(torch.bfloat16)
        sine[:, :32] = sine_source[:length].to(torch.bfloat16)
        return cosine.contiguous(), sine.contiguous()

    def run(self, plan: PersistentDecodePlan):
        ratio = int(self.runner.block_size // self.runner.physical_block_size)
        physical_blocks = tuple(
            int(block) * ratio + offset
            for block in plan.block_table
            for offset in range(ratio)
        )
        block_map = torch.tensor(
            physical_blocks, dtype=torch.int32, device=self.runner.device
        ).contiguous()
        request = self._quantum_request(
            pending_token=plan.pending_token,
            committed_kv_length=plan.committed_kv_length,
            max_sequence_length=plan.max_sequence_length,
            max_tokens=plan.max_tokens,
            eos_token_id=plan.eos_token_id,
            ignore_eos=plan.ignore_eos,
            full_block_map=block_map,
            sliding_block_map=None,
            cancellation_flag=None,
            rope_cosine=self.rope_cosine,
            rope_sine=self.rope_sine,
        )
        try:
            result = self.native.run_quantum(request)
        except self._prelaunch_error:
            raise
        except self._backend_error:
            self.healthy = False
            raise

        emitted = tuple(int(token) for token in result.emitted_tokens)
        if not emitted or len(emitted) > plan.max_tokens:
            self.healthy = False
            raise RuntimeError("persistent decoder returned an invalid token count")
        committed = int(result.committed_kv_length)
        if committed != plan.committed_kv_length + len(emitted):
            self.healthy = False
            raise RuntimeError("persistent decoder returned an invalid cache frontier")
        if int(result.pending_token) != emitted[-1]:
            self.healthy = False
            raise RuntimeError("persistent decoder returned an invalid pending token")

        return emitted

    def close(self) -> None:
        native = getattr(self, "native", None)
        if native is not None:
            native.close()
