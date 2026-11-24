# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""KV Connector that writes hidden states via MORI-IO P2P GPU-to-GPU transfer.

Drop-in replacement for MooncakeHiddenStatesConnector. Uses MoriIOStore
instead of EagleMooncakeStore for direct GPU RDMA transfer without CPU staging.

vLLM discovers this connector via ``kv_connector_module_path`` in the
``kv_transfer_config`` dict.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)

HIDDEN_STATES_DTYPE_STR = "bfloat16"


def _sanitize_key(key: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
    if sanitized and sanitized[0].isdigit():
        sanitized = "k" + sanitized
    return sanitized


def _extract_from_kv_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    padded_kv = kv_cache.flatten(0, 1)[slot_mapping]
    return padded_kv[:num_tokens]


def _slot_mapping_from_block_ids(
    block_ids: list[int],
    page_size: int,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    block_ids_gpu = torch.tensor(block_ids, dtype=torch.int64, device=device)
    offsets = torch.arange(page_size, dtype=torch.int64, device=device)
    return (block_ids_gpu.unsqueeze(1) * page_size + offsets).flatten()[:num_tokens]


@dataclass
class _ReqMeta:
    req_id: str
    token_ids: torch.Tensor
    block_ids: list[int] = field(default_factory=list)


@dataclass
class MoriConnectorMetadata(KVConnectorMetadata):
    requests: list[_ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        req_id: str,
        token_ids: list[int],
        block_ids: list[int],
    ) -> None:
        self.requests.append(
            _ReqMeta(
                req_id=req_id,
                token_ids=torch.tensor(token_ids),
                block_ids=list(block_ids),
            )
        )


class MoriHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """KV Connector that stores hidden states via MORI-IO P2P RDMA.

    Same interface as MooncakeHiddenStatesConnector but uses MoriIOStore
    for direct GPU-to-GPU transfer without CPU staging.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self.cache_layers: list[str] = []
        self._cache_layer_group_id: int = self._find_cache_layer_group_id(
            kv_cache_config
        )

        assert (
            self._vllm_config.speculative_config is not None
        ), "MoriHiddenStatesConnector requires 'extract_hidden_states' speculative method"
        spec_config = self._vllm_config.speculative_config.draft_model_config.hf_config
        self._layer_ids = list(
            getattr(spec_config, "eagle_aux_hidden_state_layer_ids", [])
        )
        self.num_hidden_states = len(self._layer_ids)
        self._hidden_size = vllm_config.model_config.get_hidden_size()

        self._num_training_layers = max(self.num_hidden_states - 1, 1)

        self._active_requests: dict[str, Any] = {}
        self._req_blocks: dict[str, list[int]] = {}
        self._req_metadata: dict[str, dict[str, Any]] = {}

        self._mori_store = None
        self._mori_setup_done = False
        self._tp_rank: int | None = None

    @staticmethod
    def _find_cache_layer_group_id(kv_cache_config) -> int:
        if kv_cache_config is None:
            return None  # type: ignore[return-value]
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                if "cache_only_layers" in name:
                    logger.info(
                        f"Cache-only layer found in KV group {gid}: {name} "
                        f"(total groups={len(kv_cache_config.kv_cache_groups)})"
                    )
                    return gid
        logger.warning(
            f"Cache-only layer NOT found in KV cache groups "
            f"(groups={[g.layer_names for g in kv_cache_config.kv_cache_groups]})"
        )
        return None  # type: ignore[return-value]

    def _get_tp_rank(self) -> int:
        if self._tp_rank is None:
            try:
                from vllm.distributed import get_tensor_model_parallel_rank

                self._tp_rank = get_tensor_model_parallel_rank()
            except (OSError, RuntimeError):
                self._tp_rank = 0
        return self._tp_rank

    def _ensure_mori_store(self) -> bool:
        if self._mori_setup_done:
            return self._mori_store is not None

        if self._get_tp_rank() != 0:
            self._mori_setup_done = True
            return False

        src_gpu = int(os.environ.get("MORI_IO_SRC_GPU", "0"))
        dst_gpu = int(os.environ.get("MORI_IO_DST_GPU", "0"))

        if src_gpu == dst_gpu and not os.environ.get("MORI_IO_FORCE"):
            logger.warning(
                "MoriHiddenStatesConnector: MORI_IO_SRC_GPU == MORI_IO_DST_GPU (%d); "
                "hidden states will NOT be stored.",
                src_gpu,
            )
            self._mori_setup_done = True
            return False

        try:
            from atomrl.transfer.mori_io_store import MoriIOStore

            self._mori_store = MoriIOStore(
                role="producer",
                src_gpu=src_gpu,
                dst_gpu=dst_gpu,
                max_seq_len=int(os.environ.get("MORI_IO_MAX_SEQ_LEN", "8192")),
                hidden_dim=self._hidden_size,
                num_aux_layers=self.num_hidden_states,
            )

            device: torch.device | None = None
            if torch.cuda.is_initialized():
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self._mori_store.setup(device=device)
            self._mori_setup_done = True
            logger.info(
                "MoriHiddenStatesConnector: MORI-IO store initialized "
                f"(src_gpu={src_gpu}, dst_gpu={dst_gpu})"
            )
            return True
        except Exception:
            logger.exception("MoriHiddenStatesConnector: failed to init MORI-IO store")
            self._mori_setup_done = True
            return False

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def wait_for_save(self):
        if self._mori_store is not None:
            self._mori_store.flush()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )

        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        self.cache_layers = list(layers.keys())
        assert (
            len(self.cache_layers) == 1
        ), f"Expected 1 CacheOnlyAttentionLayer, got {len(self.cache_layers)}"

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if layer_name not in self.cache_layers:
            return

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionMetadata,
        )

        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata)

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, MoriConnectorMetadata)

        if not self._ensure_mori_store():
            if self._get_tp_rank() == 0:
                logger.warning("save_kv_layer: MORI-IO store not available, skipping")
            return

        page_size = kv_layer.shape[1]

        for request in connector_metadata.requests:
            num_tokens = request.token_ids.shape[0]
            mori_key = _sanitize_key(request.req_id)

            slot_mapping = _slot_mapping_from_block_ids(
                request.block_ids,
                page_size,
                num_tokens,
                device=kv_layer.device,
            )
            num_slots = slot_mapping.shape[0]

            if num_slots < num_tokens:
                continue

            hidden_states_3d = _extract_from_kv_cache(
                kv_layer, slot_mapping, num_tokens
            )

            all_hidden = hidden_states_3d.reshape(num_tokens, -1)

            split_at = self._num_training_layers * self._hidden_size
            hidden_states = all_hidden[:, :split_at]
            last_hidden_states = all_hidden[:, -self._hidden_size :]

            input_ids = request.token_ids.to(hidden_states.device)

            try:
                self._mori_store.put(
                    key=mori_key,
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    last_hidden_states=last_hidden_states,
                    target=None,
                )
            except Exception:
                logger.exception(
                    f"save_kv_layer: failed to store via MORI-IO for {request.req_id}"
                )

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        assert num_external_tokens == 0, "This connector is store-only"

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MoriConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            group_sizes = [len(g) for g in new_req.block_ids]
            gid = self._cache_layer_group_id
            if gid is None:
                gid = max(
                    range(len(new_req.block_ids)),
                    key=lambda i: len(new_req.block_ids[i]),
                )
                self._cache_layer_group_id = gid
                logger.warning(
                    f"Resolved cache-only KV group id={gid} (group_sizes={group_sizes})"
                )
            meta.add_request(
                new_req.req_id,
                token_ids=token_ids,
                block_ids=new_req.block_ids[gid],
            )
            logger.debug(
                "build_connector_meta: req_id=%s key=%s num_tokens=%d gid=%d "
                "group_sizes=%s chosen_blocks=%d",
                new_req.req_id,
                _sanitize_key(new_req.req_id),
                len(token_ids),
                gid,
                group_sizes,
                len(new_req.block_ids[gid]),
            )
            self._active_requests[new_req.req_id] = new_req
            self._req_blocks[new_req.req_id] = list(new_req.block_ids[gid])

            seq_len = len(token_ids)
            training_hidden_size = self._num_training_layers * self._hidden_size
            mori_key = _sanitize_key(new_req.req_id)
            self._req_metadata[new_req.req_id] = {
                "mooncake_key": mori_key,
                "tensor_shapes": {
                    "hidden_states": (seq_len, training_hidden_size),
                    "input_ids": (seq_len,),
                    "last_hidden_states": (seq_len, self._hidden_size),
                },
                "tensor_dtypes": {
                    "hidden_states": HIDDEN_STATES_DTYPE_STR,
                    "input_ids": "int64",
                    "last_hidden_states": HIDDEN_STATES_DTYPE_STR,
                },
                "num_layers": self.num_hidden_states,
                "input_ids_list": token_ids,
            }

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._active_requests:
                continue

            new_block_ids = cached_reqs.new_block_ids[i]
            if new_block_ids is None:
                continue

            cached_req = self._active_requests[req_id]
            req_block_ids = self._req_blocks[req_id]

            block_ids = new_block_ids[self._cache_layer_group_id]
            req_block_ids.extend(block_ids)

            meta.add_request(
                req_id=req_id,
                token_ids=cached_req.prompt_token_ids or [],
                block_ids=req_block_ids,
            )

        return meta

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        _ = self._active_requests.pop(req_id, None)
        _ = self._req_blocks.pop(req_id, None)

        mori_meta = self._req_metadata.pop(req_id, None)
        return False, mori_meta

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        group0_ids = block_ids[0] if block_ids else []
        return self.request_finished(request, group0_ids)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        return "NHD"
