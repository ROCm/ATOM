# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The cache key: the checkpoint and the drafter's, the model's HF config, the
shard a rank holds, the quantization config, and the launch options and
environment variables listed here as changing what loading produces. Nothing
else is looked at: after a change to loading code, or to a setting that is
not listed, stop the daemon by hand
(``python -m atom.model_loader.weight_cache stop``)."""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import os

from torch import nn

from atom.model_loader.weight_cache.values import _fp, _json
from atom.model_loader.weight_iterator import _MAX_SAFETENSORS_HEADER_BYTES
from atom.model_loader.weight_utils import local_model_dir

# Launch options (Config fields) that change the loaded weights, or decide
# whether loading changes an object. The cache does not find these by itself:
# a server whose setting is missing here gets another launch's weights without
# a word, so add one when loading starts to depend on it. What only picks a
# kernel or sizes a cache does not belong, unless an object loading creates
# reads it: a server gets that object as the daemon built it.
_CONFIG_FIELDS = (
    "load_dummy",
    # Which MoE implementation, and whether the shared expert is fused in.
    "moe_backend",
    "moe_all2all_backend",
    "moe_ep_flatten_tp_across_dp",
    # Whether the MoE fuses its communication, which fixes the experts' layout.
    "enable_tbo",
    # How wide experts and attention projections are sharded.
    "dp_logical_size",
    "decode_context_parallel_size",
    "dcp_config.enable_query_replication",
    "dcp_config.indexer_dcp_only",
    # Whether loading keeps a DCP-wide copy of the MLA value projection.
    "dcp_config.enable_project_before_merge",
)
# The environment variables that do, under the same rule.
_ENV_VARS = (
    # Whether the shared expert is fused into the routed experts.
    "ATOM_DISABLE_MORI_EP",
    "ATOM_FUSE_SHARED_EXPERT",
    "ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4",
    # The layout post-processing leaves weights and scales in.
    "ATOM_FP4_MXSCALE_USE_32X4_SCALE_SHUFFLE",
    "ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE",
    "ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE",
    "ATOM_GROUP32_WEIGHT_PRESHUFFLE",
    # Whether the MiniMax-M3 qkv linear keeps a chain-layout weight scale after loading.
    "ATOM_M3_QKV_CHAIN",
    "ATOM_MOE_GU_ITLV",
    "ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM",
    "ATOM_USE_TRITON_GEMM",
    "ATOM_USE_TRITON_MOE",
    "ATOM_USE_TRITON_MOE_DECODE",
    "AITER_DISABLE_COMM_FUSED_MOE",
    # What attention makes of its weights after loading.
    "ATOM_FORCE_ATTN_TRITON",
    "ATOM_USE_UNIFIED_ATTN",
    "ATOM_DSV41_MHC_BACKEND",
    "ATOM_ENABLE_GLM_FUSED_INDEXER",
    "ATOM_MHC_USE_BF16",
    "ATOM_MLA_ABSORB_BF16",
    "ATOM_USE_TRITON_MXFP4_BMM",
    # Whether each rank holds the whole vocabulary embedding.
    "ATOM_REPLICATE_VOCAB_EMBED",
    # Whether loading realigns a fused norm's quantization to the layer after.
    "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION",
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT",
    # Read by the MoE method online quantization swaps in while loading.
    "AITER_FLYDSL_USE_HERD",
    "ATOM_MORI_FP4_DISPATCH",
    "ATOM_MORI_V2",
    "ATOM_USE_TRITON_MOE_A4W4",
)

_SMALL_FILE = 1 << 20


def file_identity(path: str) -> str:
    """A file as the key sees it: a small one (a config) by content, a large
    one (a weight shard) by size, mtime and inode, as rewriting it changes."""
    try:
        st = os.stat(path)
        if st.st_size > _SMALL_FILE:
            return f"{st.st_size}:{st.st_mtime_ns}:{st.st_ino}"
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError as e:
        return f"<unreadable {type(e).__name__}>"


# The files loading reads from a checkpoint directory (its top level, as the
# weight iterator does). Others, such as what a load writes there, do not count.
_LOADED_FILES = ("*.safetensors", "*.safetensors.index.json", "config.json")


def _checkpoint_digest(model_path: str) -> str:
    """The files loading reads by name and file_identity, with the
    safetensors headers (as bytes: parsing them takes ~1 s on V4-Pro)."""
    root = os.path.realpath(model_path)
    if not os.path.isdir(root):
        return f"path:{root}"
    h, weights = hashlib.sha256(), False
    for name in sorted(os.listdir(root)):
        if not any(fnmatch.fnmatch(name, p) for p in _LOADED_FILES):
            continue
        path = os.path.join(root, name)
        h.update(f"{name}\0{file_identity(path)}\n".encode())
        if name.endswith(".safetensors"):
            weights = True
            with contextlib.suppress(OSError), open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                h.update(f.read(min(n, _MAX_SAFETENSORS_HEADER_BYTES)))
    # No weights there yet (a hub model before its download): the other files
    # do not tell two repos or revisions apart, the snapshot's path does.
    return h.hexdigest() if weights else f"incomplete:{root}:{h.hexdigest()}"


def _setting(config, name: str) -> str:
    for part in name.split("."):
        config = getattr(config, part)
    return str(config)


def compute_key(config, rank: int) -> dict:
    """Which checkpoint, drafter and shard, and the listed settings. Taken
    before a load, by the daemon and by a server alike: files that change
    under the daemon's load leave it a key no server's files give. (So does
    the first load of a hub id, which downloads them: that launch or the next
    one loads from disk or replaces the daemon once, depending on timing.)"""
    hf = config.hf_config
    try:
        hf_dict = hf.to_dict()
    except Exception:  # noqa: BLE001
        hf_dict = vars(hf)
    pc = config.parallel_config
    spec = config.speculative_config
    quant = config.quant_config
    return {
        "checkpoint": _checkpoint_digest(local_model_dir(config.model)),
        # The daemon holds this drafter's weights too.
        "draft": (
            {
                "method": spec.method,
                "checkpoint": spec.model
                and _checkpoint_digest(local_model_dir(spec.model)),
            }
            if spec
            else None
        ),
        "hf_config": hashlib.sha256(_json(hf_dict).encode()).hexdigest(),
        "rank": rank,
        # The shard a rank holds depends on the whole layout (EP spans DP).
        "parallel": {
            "tp": config.tensor_parallel_size,
            "tp_world": config.tp_world_size,
            "pcp": config.pcp_group_size,
            "dp": pc.data_parallel_size,
            "dp_rank": pc.data_parallel_rank,
            "dp_attention": bool(config.enable_dp_attention),
            "ep": bool(config.enable_expert_parallel),
        },
        # Online quantization included.
        "quant": quant.compute_hash() if quant else None,
        "options": {k: _setting(config, k) for k in _CONFIG_FIELDS},
        "env": {k: os.environ.get(k) for k in _ENV_VARS},
    }


def _diff_key(daemon: dict, client: dict, prefix: str = "") -> list[str]:
    """Differences between two nested dicts (key, structure)."""
    diffs = []
    for k in sorted(set(daemon) | set(client)):
        a, b = daemon.get(k, "<missing>"), client.get(k, "<missing>")
        if isinstance(a, dict) and isinstance(b, dict):
            diffs += _diff_key(a, b, f"{prefix}{k}.")
        elif a != b:
            diffs.append(f"{prefix}{k}: daemon={str(a)[:120]} client={str(b)[:120]}")
    return diffs


def structure(model: nn.Module) -> dict:
    """The module tree, module types, parameter shapes/dtypes and buffer
    shapes/dtypes, comparable between a real and a meta-built model."""
    return {
        path: {
            "type": f"{type(mod).__module__}.{type(mod).__qualname__}",
            "params": {
                k: _fp(p, content=False)
                for k, p in mod._parameters.items()
                if p is not None
            },
            "buffers": {
                k: _fp(b, content=False)
                for k, b in mod._buffers.items()
                if b is not None
            },
        }
        for path, mod in model.named_modules(remove_duplicate=False)
    }


def exported_structure(built: dict, export: dict) -> dict:
    """``built`` with only the buffers the daemon exports or deletes: the
    client builds the others itself, e.g. caches sized by --max-model-len.
    Both sides filter with the daemon's export plan before comparing."""
    out = {}
    for path, entry in built.items():
        item = export["plan"].get(path, {})
        keep = set(item.get("buffers", {})) | {
            k for kind, k in item.get("delete", []) if kind == "buffer"
        }
        out[path] = {
            **entry,
            "buffers": {k: v for k, v in entry["buffers"].items() if k in keep},
        }
    return out
