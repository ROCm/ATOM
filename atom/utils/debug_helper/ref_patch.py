# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Monkey-patch helpers for instrumenting read-only reference implementations.

When the reference is in ``/data/<model>/inference/model.py`` (read-only,
shipped with checkpoint), you can't add dump statements directly. This
module provides a context-manager-style API that wraps the reference's
forward methods to capture intermediate tensors using the same dump
infrastructure as ``atom.utils.debug_helper.dump``.

Pattern
-------
    from atom.utils.debug_helper import patch_block_forward

    with patch_block_forward(ref_model, layer_attr="layer_id"):
        ref_model.forward(...)
        # dumps written to ATOM_FWD_DUMP_DIR via debug_helper.dump machinery

The patcher uses the same ``ATOM_FWD_DUMP_*`` env vars as
``install_block_forward_hooks`` so both sides emit comparable filenames.
The reference filename prefix is configurable via ``side_prefix`` so you
can write ``ref_layer{LL}_*.pt`` next to ``layer{LL}_*.pt`` from ATOM.

Why monkey-patch instead of register_forward_hook?
- Reference forwards often consume non-tensor arguments (start_pos, etc.)
  that PyTorch hooks can't easily intercept.
- Reference may have multiple internal sub-stages between module entry
  and exit (q_norm, RoPE, scatter, sparse_attn) that need named dumps.
- Reference's ``forward`` may not be a registered ``nn.Module.forward``
  (e.g. monkey-patched at module level).
"""

from __future__ import annotations

import contextlib
import os
import re
from typing import Callable, Iterator, Optional

import torch

from atom.utils import envs


def _get_rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


def _make_dump_fn(
    layer_id: int,
    cls_name: str,
    side_prefix: str = "ref",
) -> Optional[Callable[[str, torch.Tensor], None]]:
    """Return a ``dump(stage_name, tensor)`` closure or None when disabled.

    Filename format: ``{side_prefix}_layer{LL}_{cls_name}__{stage_name}_rank{R}.pt``

    Multiple stages within one forward call are written as separate files
    so they don't collide with the per-call counter scheme used by
    ``install_block_forward_hooks``.
    """
    dump_dir = envs.ATOM_FWD_DUMP_DIR
    if not dump_dir:
        return None
    layers_env = envs.ATOM_FWD_DUMP_LAYERS
    if layers_env:
        wanted = {int(x) for x in layers_env.split(",") if x}
        if layer_id not in wanted:
            return None
    rank = _get_rank()

    def _dump(stage_name: str, t: torch.Tensor) -> None:
        if not isinstance(t, torch.Tensor):
            return
        safe = re.sub(r"[^A-Za-z0-9_]", "_", stage_name)
        fname = os.path.join(
            dump_dir,
            f"{side_prefix}_layer{layer_id:02d}_{cls_name}__{safe}_rank{rank}.pt",
        )
        torch.save({"hidden": t.detach().cpu(), "shape": tuple(t.shape)}, fname)

    return _dump


@contextlib.contextmanager
def patch_method(
    target_class: type,
    method_name: str,
    wrapper_factory: Callable[[Callable], Callable],
) -> Iterator[None]:
    """Context manager: wrap ``target_class.method_name`` with ``wrapper_factory``.

    ``wrapper_factory(orig_method) -> new_method`` builds the replacement
    bound function. On exit the original is restored.

    Example::

        def wrap(orig):
            def patched(self, x, start_pos):
                dump = _make_dump_fn(self.layer_id, "Attn")
                if dump: dump("x_in", x)
                out = orig(self, x, start_pos)
                if dump: dump("out", out)
                return out
            return patched

        with patch_method(Attention, "forward", wrap):
            model.forward(...)
    """
    orig = getattr(target_class, method_name)
    new = wrapper_factory(orig)
    setattr(target_class, method_name, new)
    try:
        yield
    finally:
        setattr(target_class, method_name, orig)


@contextlib.contextmanager
def patch_block_forward(
    block_class: type,
    layer_attr: str = "layer_id",
    side_prefix: str = "ref",
    extra_stages: Optional[dict[str, Callable]] = None,
) -> Iterator[None]:
    """Patch ``block_class.forward`` to dump hidden_in / hidden_out per layer.

    Minimal: dumps just the input ``x`` and the return value, named
    ``hidden_in`` and ``hidden_out``. The wrapper assumes the first positional
    arg after ``self`` is the hidden state and the return is a tensor (or a
    tuple whose [0] is the tensor).

    For richer sub-stage capture, write a custom wrapper using
    ``patch_method`` directly — see the docstring there.

    ``side_prefix`` differentiates ATOM vs reference output files
    (default "ref"). Set to "atom" if patching ATOM's own model.
    """

    def wrap(orig: Callable) -> Callable:
        def patched(self, *args, **kwargs):
            layer_id = getattr(self, layer_attr, None)
            cls_name = type(self).__name__
            dump = (
                _make_dump_fn(int(layer_id), cls_name, side_prefix)
                if layer_id is not None
                else None
            )
            if dump and args:
                dump("hidden_in", args[0])
            out = orig(self, *args, **kwargs)
            if dump:
                t = out[0] if isinstance(out, tuple) else out
                if isinstance(t, torch.Tensor):
                    dump("hidden_out", t)
            return out

        return patched

    with patch_method(block_class, "forward", wrap):
        yield


@contextlib.contextmanager
def patch_module_dump(
    target_class: type,
    method_name: str = "forward",
    cls_name_override: Optional[str] = None,
    side_prefix: str = "ref",
    parent_layer_attr: str = "layer_id",
) -> Iterator[None]:
    """Patch any module's ``forward`` to dump return value tensor.

    Use for sub-modules that don't carry ``layer_id`` themselves (Compressor,
    Indexer, RMSNorm). The wrapper walks the call stack for an enclosing
    block whose ``layer_attr`` it can read; if not found, falls back to
    layer_id=99 (= "unknown layer").

    For multi-call-per-forward modules (called inside per-seq dispatch loop),
    pair this with ``ATOM_FWD_DUMP_ONE_SHOT=0`` so each call gets a unique
    file. Otherwise only the first call is captured.
    """
    cls_name = cls_name_override or target_class.__name__

    def wrap(orig: Callable) -> Callable:
        def patched(self, *args, **kwargs):
            # Try to find an enclosing block layer_id via known sub-module
            # attributes set by the parent (e.g. self.layer_id forwarded).
            layer_id = getattr(self, parent_layer_attr, None) or 99
            dump = _make_dump_fn(int(layer_id), cls_name, side_prefix)
            out = orig(self, *args, **kwargs)
            if dump:
                t = out[0] if isinstance(out, tuple) else out
                if isinstance(t, torch.Tensor):
                    dump("out", t)
            return out

        return patched

    with patch_method(target_class, method_name, wrap):
        yield
