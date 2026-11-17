# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A V4.1 deployment as the mono decodes' selection reads it (``selection``): plain values from ATOM's ``Config``, the
drafter, the metadata builder, the environment and the device. Building one never raises: a field the objects do not
carry, or cannot read, takes its default.

The backends' ``serves`` still ask their own refusal functions of the raw objects (``atom_config``, ``drafter_obj``,
``metadata_builder``), so their reasons, exceptions and patch points are theirs; the typed fields name the deployment.
(Imported under an alias: ``atom.mono.runtime.deployment`` is the shared refusals' module.)
"""

from __future__ import annotations

import dataclasses
import importlib.util
from typing import Any

import torch

from atom.mono.runtime import deployment as runtime_deployment
from atom.utils import envs

# the model's shape as a deployment names it (``hi_gate.GEOMETRY`` holds the hi path's full set)
GEOMETRY = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "n_routed_experts",
    "hc_mult",
    "vocab_size",
    "sliding_window",
    "num_nextn_predict_layers",
)
INDEXER = (
    "index_topk",
    "candidate_topk_blocks",
    "candidate_block_size",
    "index_n_heads",
    "index_head_dim",
)


def _read(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - an unreadable value is the field's default
        return default


def _device() -> tuple[str | None, int | None]:
    """(gfx arch, CUs) of the current device, (None, None) without one."""
    if not torch.cuda.is_available():
        return None, None
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return p.gcnArchName.split(":")[0], int(p.multi_processor_count)


@dataclasses.dataclass(frozen=True)
class Deployment:
    # the environment and the device
    mono_enabled: bool = False
    side_streams: int | None = None
    triton_moe: bool = False
    device_arch: str | None = None
    device_cus: int | None = None
    flydsl: bool = False
    # parallelism
    tp: int = 1
    dp_attention_ranks: int = 0
    data_parallel_size: int = 1
    expert_parallel: bool = False
    pp: int = 1
    decode_cp: int = 1
    multinode_tp: bool = False
    tbo: bool = False
    compilation_level: int | None = None
    enforce_eager: bool = False
    # caches and scheduling
    kv_cache_dtype: str | None = None
    index_cache_dtype: str | None = None
    index_cache_layout: str | None = None
    kv_block_size: int | None = None
    prefix_caching: bool = True
    max_model_len: int | None = None
    kv_transfer: bool = False
    max_num_seqs: int | None = None
    # speculative decoding
    spec_method: str | None = None
    num_speculative_tokens: int | None = None
    dspark_block_size: int | None = None
    dspark_confidence_schedule: bool | None = None
    dspark_ragged: bool | None = None
    drafter: bool = False
    block_drafter: bool = False
    # the model
    attention_keys: int | None = None
    indexer_shape: tuple | None = None
    geometry: tuple = ()
    # the metadata builder
    builder: bool = False
    decode_round_builder: bool = False
    engram: bool = False
    engram_uva: bool = False
    engram_overlap: bool = False
    ckpt: str | None = None
    # the objects the backends read
    atom_config: Any = dataclasses.field(default=None, compare=False, repr=False)
    drafter_obj: Any = dataclasses.field(default=None, compare=False, repr=False)
    metadata_builder: Any = dataclasses.field(default=None, compare=False, repr=False)

    @property
    def band(self) -> str | None:
        """The concurrency band of ``max_num_seqs``: "C1", "C2-8" or ">C8"."""
        n = self.max_num_seqs
        if n is None:
            return None
        return "C1" if n <= 1 else "C2-8" if n <= 8 else ">C8"

    @classmethod
    def of(cls, atom_config, drafter=None, metadata_builder=None) -> Deployment:
        cfg = atom_config
        hf = getattr(cfg, "hf_config", None)
        spec = getattr(cfg, "speculative_config", None)
        dspark = getattr(cfg, "dspark", None)
        host = getattr(getattr(metadata_builder, "engram", None), "host", None)
        arch, cus = _read(_device, (None, None))
        kv = getattr(cfg, "kv_transfer_config", None)
        level = getattr(getattr(cfg, "compilation_config", None), "level", None)
        return cls(
            mono_enabled=bool(_read(lambda: envs.ATOM_MONO_ENABLE, False)),
            side_streams=_read(lambda: envs.ATOM_DSV41_SIDE_STREAMS),
            triton_moe=bool(_read(lambda: envs.ATOM_USE_TRITON_MOE, False)),
            device_arch=arch,
            device_cus=cus,
            flydsl=bool(_read(lambda: importlib.util.find_spec("flydsl"), False)),
            tp=int(getattr(cfg, "tensor_parallel_size", 1) or 1),
            dp_attention_ranks=_read(
                lambda: runtime_deployment.dp_attention_ranks(cfg), 0
            ),
            data_parallel_size=getattr(
                getattr(cfg, "parallel_config", None), "data_parallel_size", 1
            ),
            expert_parallel=bool(getattr(cfg, "enable_expert_parallel", False)),
            pp=getattr(cfg, "pipeline_parallel_size", 1),
            decode_cp=getattr(cfg, "decode_context_parallel_size", 1),
            multinode_tp=bool(getattr(cfg, "is_multinode_tp", False)),
            tbo=bool(
                getattr(cfg, "enable_tbo", False)
                or getattr(cfg, "enable_tbo_decode", False)
            ),
            compilation_level=_read(lambda: int(level)) if level is not None else None,
            enforce_eager=bool(getattr(cfg, "enforce_eager", False)),
            kv_cache_dtype=getattr(cfg, "kv_cache_dtype", None),
            index_cache_dtype=getattr(cfg, "index_cache_dtype", None),
            index_cache_layout=getattr(cfg, "index_cache_layout", None),
            kv_block_size=getattr(cfg, "kv_cache_block_size", None),
            prefix_caching=bool(getattr(cfg, "enable_prefix_caching", True)),
            max_model_len=getattr(cfg, "max_model_len", None),
            kv_transfer=bool(kv),
            max_num_seqs=getattr(cfg, "max_num_seqs", None),
            spec_method=getattr(spec, "method", None),
            num_speculative_tokens=getattr(spec, "num_speculative_tokens", None),
            dspark_block_size=getattr(hf, "dspark_block_size", None),
            dspark_confidence_schedule=getattr(dspark, "confidence_schedule", None),
            dspark_ragged=getattr(dspark, "ragged", None),
            drafter=drafter is not None,
            block_drafter=hasattr(getattr(drafter, "model", None), "block_backbone"),
            attention_keys=_read(lambda: hf.sliding_window + hf.index_topk),
            indexer_shape=_read(lambda: tuple(getattr(hf, k) for k in INDEXER)),
            geometry=tuple((k, getattr(hf, k, None)) for k in GEOMETRY),
            builder=metadata_builder is not None,
            decode_round_builder=hasattr(metadata_builder, "use_decode_round"),
            engram=host is not None,
            engram_uva=bool(getattr(host, "uva", False)),
            engram_overlap=getattr(host, "overlap", None) is not None,
            ckpt=getattr(cfg, "model", None),
            atom_config=cfg,
            drafter_obj=drafter,
            metadata_builder=metadata_builder,
        )
