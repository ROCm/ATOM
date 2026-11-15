"""Stopgap when this GPU's arch is missing from the image's sgl_kernel.

``sgl_kernel.apply_token_bitmask_inplace_cuda`` works on an arch it was
built for. The SGLang release image is built on MI355X, so
``common_ops.so`` contains only gfx950 code objects. On MI308X (gfx942)
HIP reports no compatible code object and every ``sgl_kernel`` GPU op
SIGSEGVs, including the xgrammar vocab-mask apply used by
``response_format={"type":"json_object"}``.

Replace that binding with a Torch GPU ``masked_fill_`` only when the
current device arch is absent from ``common_ops*.so``. A gfx950 device
keeps the native kernel. Other ``sgl_kernel`` ops (for example MTP
``verify_tree_greedy``) have the same image gap and are not covered here.

TODO: remove this patch once the image ships a gfx942 ``sgl_kernel``.

Disable with ``ATOM_SGLANG_XGRAMMAR_HIP_TORCH_BITMASK=0``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch

from atom.utils import envs

logger = logging.getLogger("atom.plugin.sglang.xgrammar_hip_bitmask")

_BITS_PER_BLOCK = 32
_GFX_ARCH_RE = re.compile(rb"amdgcn-amd-amdhsa--(gfx[0-9a-z]+)")


def _env_enabled() -> bool:
    return bool(envs.ATOM_SGLANG_XGRAMMAR_HIP_TORCH_BITMASK)


def _gfx_arch_token(gcn_arch_name: str) -> str:
    return gcn_arch_name.split(":")[0].strip()


def _embedded_gfx_archs_from_blob(blob: bytes) -> set[str]:
    return {match.group(1).decode() for match in _GFX_ARCH_RE.finditer(blob)}


def _should_replace_native_bitmask(device_arch: str, embedded_archs: set[str]) -> bool:
    """True only when the device arch is known and absent from sgl_kernel."""

    arch = _gfx_arch_token(device_arch)
    if not arch or not embedded_archs:
        return False
    return arch not in embedded_archs


def _current_device_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
    return _gfx_arch_token(str(name))


def _sgl_kernel_embedded_archs() -> set[str]:
    try:
        import sgl_kernel
    except Exception:  # noqa: BLE001 - CPU-only or outlines-only installs
        logger.info("sgl_kernel not importable; keep native xgrammar bitmask")
        return set()

    root = Path(sgl_kernel.__file__).resolve().parent
    blobs = sorted(root.glob("common_ops*.so"))
    if not blobs:
        logger.info("no sgl_kernel common_ops*.so under %s", root)
        return set()

    archs: set[str] = set()
    for path in blobs:
        archs.update(_embedded_gfx_archs_from_blob(path.read_bytes()))
    return archs


def _device_lacks_sgl_kernel_code_object() -> bool:
    device_arch = _current_device_arch()
    embedded = _sgl_kernel_embedded_archs()
    replace = _should_replace_native_bitmask(device_arch, embedded)
    if replace:
        logger.info(
            "sgl_kernel code objects %s do not include %s",
            sorted(embedded),
            device_arch,
        )
    else:
        logger.info(
            "keep native xgrammar bitmask; device=%r embedded=%s",
            device_arch,
            sorted(embedded),
        )
    return replace


def _apply_bitmask_torch_gpu(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: list[int] | torch.Tensor | None = None,
) -> None:
    """Apply packed int32 token bitmask on GPU via Torch."""

    if bitmask.dtype != torch.int32:
        raise TypeError(f"bitmask must be int32, got {bitmask.dtype}")

    logits_2d = logits.unsqueeze(0) if logits.ndim == 1 else logits
    bitmask_2d = bitmask.unsqueeze(0) if bitmask.ndim == 1 else bitmask

    if indices is not None:
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long, device=logits.device)
        else:
            indices = indices.to(device=logits.device, dtype=torch.long)
        logits_rows = logits_2d.index_select(0, indices)
        bitmask_rows = bitmask_2d.index_select(0, indices)
    else:
        if logits_2d.shape[0] != bitmask_2d.shape[0]:
            raise ValueError(
                f"batch size mismatch: logits {logits_2d.shape[0]} vs "
                f"bitmask {bitmask_2d.shape[0]}"
            )
        logits_rows = logits_2d
        bitmask_rows = bitmask_2d
        indices = None

    covered = min(logits_rows.shape[-1], bitmask_rows.shape[-1] * _BITS_PER_BLOCK)
    bit_ids = torch.arange(_BITS_PER_BLOCK, device=logits.device, dtype=torch.int32)
    # bitmask bit==0 → masked (set to -inf), same as upstream Triton/xgrammar.
    expanded = ((bitmask_rows.unsqueeze(-1) >> bit_ids) & 1) == 0
    mask = expanded.reshape(bitmask_rows.shape[0], -1)[:, :covered]
    logits_rows[:, :covered].masked_fill_(mask, float("-inf"))
    # Padded logits beyond the bitmask must stay unsampleable.
    if covered < logits_rows.shape[-1]:
        logits_rows[:, covered:].fill_(float("-inf"))

    if indices is not None:
        logits_2d.index_copy_(0, indices, logits_rows)


def _apply_bitmask_hip_safe(logits, vocab_mask, indices=None) -> None:
    _apply_bitmask_torch_gpu(logits, vocab_mask, indices=indices)


def apply_xgrammar_hip_bitmask_patch() -> None:
    """Replace the HIP bitmask binding only when this arch has no code object."""

    if not _env_enabled():
        logger.info("Skip xgrammar HIP bitmask patch (env disabled)")
        return

    try:
        from sglang.srt.utils import is_hip
    except Exception:  # noqa: BLE001
        return
    if not is_hip():
        return

    try:
        # Importing the module binds apply_token_bitmask_inplace_cuda into
        # its globals; methods look that name up at call time.
        from sglang.srt.constrained import xgrammar_backend as xg
    except Exception:  # noqa: BLE001 - outlines-only installs
        logger.info("xgrammar backend not importable; skip HIP bitmask patch")
        return

    if getattr(xg, "_atom_hip_bitmask_patched", False):
        return

    if not hasattr(xg, "apply_token_bitmask_inplace_cuda"):
        logger.info("xgrammar backend has no apply_token_bitmask_inplace_cuda; skip")
        return

    if not _device_lacks_sgl_kernel_code_object():
        return

    xg.apply_token_bitmask_inplace_cuda = _apply_bitmask_hip_safe
    xg._atom_hip_bitmask_patched = True
    logger.info(
        "Patched xgrammar HIP vocab-mask apply to Torch GPU; "
        "sgl_kernel has no code object for this device"
    )
