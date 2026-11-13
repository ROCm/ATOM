# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Find checkpoints on disk, read what they are, and size them against GPUs."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from atom.models.registry import support_model_arch_dict

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")
# Fraction of a GPU the server may use (ATOM's --gpu-memory-utilization
# default) and the share of that weights may take so KV cache, activations
# and graphs still fit.
GPU_MEM_UTIL = 0.9
WEIGHT_SHARE = 0.75


def hf_cache_dirs() -> list[Path]:
    dirs = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(env):
            dirs.append(Path(os.environ[env]))
    if os.environ.get("HF_HOME"):
        dirs.append(Path(os.environ["HF_HOME"]) / "hub")
    dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    return [d for d in dict.fromkeys(dirs) if d.is_dir()]


def _hf_snapshot(cache: Path, repo_id: str) -> Path | None:
    repo_dir = cache / ("models--" + repo_id.replace("/", "--"))
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return None
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        snap = snaps / ref.read_text().strip()
        if (snap / "config.json").is_file():
            return snap
    candidates = [s for s in snaps.iterdir() if (s / "config.json").is_file()]
    return max(candidates, key=lambda s: s.stat().st_mtime) if candidates else None


def resolve_model(ref: str, search_roots: tuple[str, ...] = ()) -> Path | None:
    """A local path or Hugging Face repo id -> directory holding config.json."""
    path = Path(os.path.expanduser(ref))
    if (path / "config.json").is_file():
        return path
    if re.fullmatch(r"[\w.\-]+/[\w.\-]+", ref):
        for cache in hf_cache_dirs():
            if snap := _hf_snapshot(cache, ref):
                return snap
    if search_roots and "/" not in ref.strip("/").split("/", 1)[-1]:
        # A bare or org/name reference: look for it on the usual disks.
        for m in find_models(None, roots=tuple(search_roots))["models"]:
            if m["name"].lower() == ref.lower() or m["name"].lower().endswith(
                "/" + ref.lower().split("/")[-1]
            ):
                return Path(m["path"])
    return None


def weight_bytes(model_dir: Path) -> int:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        try:
            total = json.loads(index.read_text()).get("metadata", {}).get("total_size")
            if total:
                return int(total)
        except (OSError, ValueError):
            pass
    total = 0
    for f in model_dir.iterdir():
        if f.suffix in WEIGHT_SUFFIXES:
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _text_config(cfg: dict) -> dict:
    for key in ("text_config", "language_config", "llm_config"):
        if isinstance(cfg.get(key), dict):
            return {**cfg, **cfg[key]}
    return cfg


def _first(cfg: dict, *keys, default=None):
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k]
    return default


def describe_config(cfg: dict) -> dict:
    t = _text_config(cfg)
    arch = (cfg.get("architectures") or t.get("architectures") or [None])[0]
    quant = cfg.get("quantization_config") or t.get("quantization_config") or {}
    heads = _first(t, "num_attention_heads")
    hidden = _first(t, "hidden_size")
    head_dim = _first(t, "head_dim", "v_head_dim") or (
        hidden // heads if hidden and heads else None
    )
    experts = _first(t, "n_routed_experts", "num_experts", "num_local_experts")
    return {
        "architecture": arch,
        "model_type": cfg.get("model_type"),
        "atom_supported": arch in support_model_arch_dict,
        "layers": _first(t, "num_hidden_layers", "n_layers"),
        "hidden_size": hidden,
        "attention_heads": heads,
        "kv_heads": _first(t, "num_key_value_heads", default=heads),
        "head_dim": head_dim,
        "mla_kv_lora_rank": t.get("kv_lora_rank"),
        "vocab_size": t.get("vocab_size"),
        "max_context": _first(
            t, "max_position_embeddings", "max_sequence_length", "seq_length"
        ),
        "dtype": _first(cfg, "torch_dtype", "dtype") or t.get("torch_dtype"),
        "quantization": (
            {
                k: quant[k]
                for k in ("quant_method", "fmt", "weight_block_size", "bits")
                if k in quant
            }
            or None
        ),
        "moe_experts": experts,
        "experts_per_token": t.get("num_experts_per_tok") if experts else None,
        "needs_trust_remote_code": bool(cfg.get("auto_map")),
    }


# Model types whose KV cache is plain per-layer K/V (or MLA latent), so a
# per-token size is meaningful. Sparse / compressed / linear-attention models
# (DeepSeek-V3.2/V4/V4.1, GLM-5 DSA, Kimi-K3, Qwen3-Next, MiniMax-M3) keep far
# less per token than any simple formula says, so they get no number at all.
STANDARD_KV_MODEL_TYPES = {
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "glm4_moe",
    "gpt_oss",
    "ministral",
    "deepseek_v3",
    "kimi_k2",
    "mimo_v2",
}


def kv_bytes_per_token(desc: dict, kv_dtype_bytes: int = 1) -> int | None:
    """Rough full-attention KV size; None when the layout isn't standard."""
    layers = desc.get("layers")
    if not layers or desc.get("model_type") not in STANDARD_KV_MODEL_TYPES:
        return None
    if desc.get("mla_kv_lora_rank"):
        # MLA caches the latent plus a 64-wide rope part per layer.
        return layers * (desc["mla_kv_lora_rank"] + 64) * kv_dtype_bytes
    if desc.get("kv_heads") and desc.get("head_dim"):
        return 2 * layers * desc["kv_heads"] * desc["head_dim"] * kv_dtype_bytes
    return None


def size_against_gpus(
    total_weight_bytes: int, gpu_vram_mib: int, desc: dict | None = None
) -> dict:
    """Smallest power-of-two TP whose weight share fits, plus KV headroom."""
    if not total_weight_bytes or not gpu_vram_mib:
        return {"note": "unknown weight size or GPU memory"}
    vram = gpu_vram_mib * 2**20
    budget = vram * GPU_MEM_UTIL
    out: dict = {"weights_gib": round(total_weight_bytes / 2**30, 1), "per_tp": {}}
    min_tp = None
    for tp in (1, 2, 4, 8):
        per_gpu = total_weight_bytes / tp
        fits = per_gpu <= budget * WEIGHT_SHARE
        kv_room = max(budget - per_gpu * 1.08, 0)  # +8% runtime buffers
        entry = {
            "weights_per_gpu_gib": round(per_gpu / 2**30, 1),
            "fits": fits,
            "kv_room_per_gpu_gib": round(kv_room / 2**30, 1),
        }
        kv_tok = kv_bytes_per_token(desc or {})
        if kv_tok and fits:
            # TP shards heads (MLA latent is replicated), so capacity is per GPU.
            shard = 1 if (desc or {}).get("mla_kv_lora_rank") else tp
            entry["approx_kv_tokens"] = int(kv_room * shard / kv_tok)
        out["per_tp"][tp] = entry
        if fits and min_tp is None:
            min_tp = tp
    out["min_tp"] = min_tp
    out["note"] = (
        "Conservative estimate: all checkpoint bytes split evenly across TP "
        f"ranks, fp8 KV, gpu-memory-utilization {GPU_MEM_UTIL}. Weights kept "
        "off-GPU (e.g. DeepSeek-V4.1 Engram tables live in host memory) make "
        "the real footprint smaller -- a recipe's TP choice wins over this."
    )
    return out


def inspect_model(
    ref: str, gpu_vram_mib: int | None = None, search_roots: tuple[str, ...] = ()
) -> dict:
    model_dir = resolve_model(ref, search_roots)
    if model_dir is None:
        return {
            "model": ref,
            "found": False,
            "hint": (
                "Not found locally. Use find_models to search the disks, or "
                "download it (`hf download <repo> --local-dir <dir>`) first."
            ),
        }
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
    except (OSError, ValueError) as e:
        return {"model": ref, "found": True, "path": str(model_dir), "error": str(e)}
    desc = describe_config(cfg)
    wbytes = weight_bytes(model_dir)
    report = {
        "model": ref,
        "found": True,
        "path": str(model_dir),
        **desc,
        "weights_gib": round(wbytes / 2**30, 1) if wbytes else None,
        "has_chat_template": (model_dir / "chat_template.jinja").is_file()
        or "chat_template"
        in (
            (model_dir / "tokenizer_config.json").read_text()
            if (model_dir / "tokenizer_config.json").is_file()
            else ""
        ),
    }
    if not desc["atom_supported"]:
        report["warning"] = (
            f"Architecture {desc['architecture']!r} is not in ATOM's model "
            "registry (atom/models/registry.py); ATOM cannot serve it natively. "
            "The vLLM/SGLang plugin backends may (see recipes/atom_vllm)."
        )
    if gpu_vram_mib:
        report["sizing"] = size_against_gpus(wbytes, gpu_vram_mib, desc)
    return report


_FIND_CACHE: dict[tuple, tuple[float, dict]] = {}
FIND_CACHE_TTL_S = 300


def find_models(
    query: str | None = None,
    roots: tuple[str, ...] = (),
    max_depth: int = 3,
    time_budget_s: float = 8.0,
) -> dict:
    """Directories with a config.json and weights under the given roots.

    Scanning shared /mnt trees takes seconds, and a conversation asks about
    one model after another, so results are reused for a few minutes.
    """
    key = (query, tuple(roots), max_depth)
    hit = _FIND_CACHE.get(key)
    if hit and time.monotonic() - hit[0] < FIND_CACHE_TTL_S:
        return hit[1]
    result = _scan_models(query, roots, max_depth, time_budget_s)
    _FIND_CACHE[key] = (time.monotonic(), result)
    return result


def _scan_models(query, roots, max_depth, time_budget_s) -> dict:
    deadline = time.monotonic() + time_budget_s
    terms = [t for t in re.split(r"[\s/_\-]+", (query or "").lower()) if t]
    found: dict[str, dict] = {}
    skip = {"proc", "sys", "dev", "usr", "lib", "bin", "etc", ".git", "site-packages"}

    def consider(d: Path, name: str) -> None:
        if terms and not all(t in name.lower() for t in terms):
            return
        try:
            cfg = json.loads((d / "config.json").read_text())
        except (OSError, ValueError):
            return
        if "architectures" not in cfg and "text_config" not in cfg:
            return
        try:
            if not any(f.suffix in WEIGHT_SUFFIXES for f in d.iterdir()):
                return
        except OSError:
            return
        real = str(d.resolve())
        if real in found:
            return
        arch = (cfg.get("architectures") or [None])[0]
        found[real] = {
            "name": name,
            "path": str(d),
            "architecture": arch,
            "atom_supported": arch in support_model_arch_dict,
            "weights_gib": round(weight_bytes(d) / 2**30, 1),
        }

    def walk(d: Path, depth: int) -> None:
        if time.monotonic() > deadline or depth > max_depth:
            return
        # Shared boxes are full of directories we may not read; Path.is_dir()
        # raises PermissionError on them (3.12), so guard the whole visit.
        try:
            if (d / "config.json").is_file():
                consider(d, d.name)
                return
            entries = [e for e in d.iterdir() if e.is_dir() and e.name not in skip]
        except OSError:
            return
        for e in entries:
            if e.name.startswith("models--"):
                hf_name = e.name[len("models--") :].replace("--", "/")
                try:
                    for s in (e / "snapshots").iterdir():
                        if (s / "config.json").is_file():
                            consider(s, hf_name)
                except OSError:
                    pass
                continue
            if not e.name.startswith("."):
                walk(e, depth + 1)

    for root in list(roots) + [str(d) for d in hf_cache_dirs()]:
        try:
            if Path(root).is_dir():
                walk(Path(root), 0)
        except OSError:
            continue
    # Servable models first: on shared boxes the caches are mostly embedding
    # models and other engines' test checkpoints.
    models = sorted(
        found.values(), key=lambda m: (not m["atom_supported"], m["name"].lower())
    )
    return {
        "query": query,
        "searched": list(roots) + [str(d) for d in hf_cache_dirs()],
        "truncated": time.monotonic() > deadline,
        "models": models[:60],
    }
