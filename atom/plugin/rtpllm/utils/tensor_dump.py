from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import torch

logger = logging.getLogger("atom.plugin.rtpllm.tensor_dump")

_COUNTERS: dict[str, int] = defaultdict(int)
_DECODE_SEEN: dict[tuple[str, int | None], bool] = {}


def _enabled() -> bool:
    return os.getenv("ATOM_RTP_DUMP_ENABLE", "0") == "1"


def _dump_root() -> str:
    return os.getenv("ATOM_RTP_DUMP_ROOT", "/mnt/raid0/zhaoan12/cache/atom_rtp")


def _dump_full_tensor() -> bool:
    return os.getenv("ATOM_RTP_DUMP_FULL", "0") == "1"


def _max_dump_elems() -> int:
    return int(os.getenv("ATOM_RTP_DUMP_MAX_ELEMS", "8192"))


def _dump_interval() -> int:
    return max(int(os.getenv("ATOM_RTP_DUMP_INTERVAL", "1")), 1)


def _prefill_and_first_decode_only() -> bool:
    return os.getenv("ATOM_RTP_DUMP_PREFILL_FIRST_DECODE_ONLY", "1") == "1"


def _layer_filter() -> set[int] | None:
    raw = os.getenv("ATOM_RTP_DUMP_LAYERS", "").strip()
    if not raw:
        return None
    layers: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            layers.add(int(item))
        except ValueError:
            logger.warning("Ignore invalid ATOM_RTP_DUMP_LAYERS item: %s", item)
    return layers if layers else None


def _rank() -> int:
    for key in ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"):
        val = os.getenv(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue
    return 0


def dump_tensor(
    *,
    tag: str,
    tensor: torch.Tensor | None,
    layer: int | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    if not _enabled() or tensor is None:
        return

    if _prefill_and_first_decode_only():
        is_prefill = None
        if meta is not None and "is_prefill" in meta:
            is_prefill = bool(meta.get("is_prefill"))
        # Keep dump scope tight to avoid decode-step explosion.
        if is_prefill is None:
            return
        if not is_prefill:
            key = (tag, layer)
            if _DECODE_SEEN.get(key, False):
                return
            _DECODE_SEEN[key] = True

    layers = _layer_filter()
    if layers is not None and layer is not None and layer not in layers:
        return

    idx = _COUNTERS[tag]
    _COUNTERS[tag] += 1
    if idx % _dump_interval() != 0:
        return

    base_dir = os.path.join(_dump_root(), "atom", f"rank{_rank()}_pid{os.getpid()}")
    os.makedirs(base_dir, exist_ok=True)
    safe_tag = tag.replace("/", "_").replace(" ", "_")
    prefix = os.path.join(base_dir, f"{idx:06d}_{safe_tag}")

    t = tensor.detach()
    t_cpu = t.to("cpu")
    save_tensor = t_cpu
    if (not _dump_full_tensor()) and t_cpu.numel() > _max_dump_elems():
        save_tensor = t_cpu.reshape(-1)[: _max_dump_elems()].clone()

    try:
        torch.save(save_tensor, f"{prefix}.pt")
        if t_cpu.numel() > 0:
            t_stat = t_cpu.float() if t_cpu.is_floating_point() else t_cpu.to(torch.float32)
            stat = {
                "min": float(t_stat.min().item()),
                "max": float(t_stat.max().item()),
                "mean": float(t_stat.mean().item()),
                "std": float(t_stat.std(unbiased=False).item()),
            }
            finite_ratio = None
            if t_cpu.is_floating_point():
                finite_ratio = float(torch.isfinite(t_cpu).float().mean().item())
        else:
            stat = {"min": None, "max": None, "mean": None, "std": None}
            finite_ratio = None

        info = {
            "time": datetime.utcnow().isoformat() + "Z",
            "tag": tag,
            "layer": layer,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "numel": int(t.numel()),
            "dump_full_tensor": _dump_full_tensor(),
            "dump_tensor_numel": int(save_tensor.numel()),
            "finite_ratio": finite_ratio,
            "stat": stat,
            "meta": meta or {},
        }
        with open(f"{prefix}.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=True, indent=2)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to dump tensor for tag=%s: %s", tag, e)
