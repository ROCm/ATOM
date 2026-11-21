"""aiter's FlyDSL implicit-GEMM convolution for the VAE patches (``lumen_vae_conv.py``).

:func:`conv` has ``F.conv2d`` / ``F.conv3d`` semantics. It runs the kernel where
the kernel applies -- BF16, on the GPU, no gradient needed (it is forward-only)
-- and torch everywhere else, so an FP32 VAE or a caller that needs a backward
still gets the right answer.

The kernel is ``aiter.ops.flydsl.flydsl_conv_implicit`` (ROCm/aiter#5370). It
selects launch tiles from per-model tuned tables
(``aiter/configs/model_configs/*_bf16_tuned_conv3d.csv``), including the 96- and
192-wide N tiles that VAE channel counts of 96/192/384 fill exactly and a
power-of-two tile ladder leaves a quarter empty.

Two things an aiter install older than that PR gets wrong once its files are
added, and which this module handles so the installed files can stay as
shipped. With an aiter that includes the PR, both steps find what they need and
do nothing:

* **The tuned table path.** The PR adds ``AITER_CONFIGS.AITER_CONFIG_CONV3D_BF16_FILE``
  in ``aiter/jit/core.py``. Without it the kernel's table loader catches the
  ``AttributeError`` and returns an empty table, so every call silently takes
  the heuristic tile. :func:`_install_config_property` supplies the same
  property, resolved the same way (``AITER_CONFIG_CONV3D_BF16`` if set, else the
  canonical CSV merged with every ``model_configs/*bf16_tuned_conv3d*.csv``).
* **The package import.** An older ``aiter/ops/flydsl/__init__.py`` eagerly
  imports GEMM/MoE kernels written against a flydsl that still has
  ``flydsl.compiler.protocol.fly_values``, which the flydsl this kernel needs
  (>= 0.3, for ``flydsl.expr.struct``) no longer has. The package then cannot
  be imported at all, although nothing the convolution uses is involved.
  :func:`import_flydsl_module` tries the normal import first, and only if that
  fails loads the convolution modules underneath a bare package object, which
  it removes again afterwards: any other ``import aiter.ops.flydsl`` fails
  exactly as it did before, so the rest of aiter is left as the process had it.

:func:`tuned_lookup_stats` counts how each call was resolved -- exact tuned row,
a row borrowed from the same layer at a nearby resolution, or the heuristic --
because a table that is present but never matched looks identical to one that
works everywhere except in the step time.
"""

import functools
import importlib
import importlib.util
import logging
import os
import sys
import types

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_PKG = "aiter.ops.flydsl"

# "unkeyed" is a call the table cannot describe at all (asymmetric or non-zero
# padding mode), so it never reaches a lookup.
_stats = {"exact": 0, "borrowed": 0, "heuristic": 0, "unkeyed": 0}
_shapes = {}


def _install_config_property():
    """Give ``AITER_CONFIGS`` the conv3d tuned-table path, if this aiter lacks it."""
    core = importlib.import_module("aiter.jit.core")
    cls = type(core.AITER_CONFIGS)
    if hasattr(cls, "AITER_CONFIG_CONV3D_BF16_FILE"):
        return False
    default = os.getenv(
        "AITER_CONFIG_CONV3D_BF16",
        f"{core.AITER_ROOT_DIR}/aiter/configs/bf16_tuned_conv3d.csv",
    )

    def _conv3d_bf16_file(self):
        return self.get_config_file(
            "AITER_CONFIG_CONV3D_BF16", default, "bf16_tuned_conv3d"
        )

    cls.AITER_CONFIG_CONV3D_BF16_FILE = property(_conv3d_bf16_file)
    return True


def import_flydsl_module(name):
    """``aiter.ops.flydsl.<name>``, even where that package's ``__init__`` cannot import.

    See the module docstring. The package is only ever bare for the duration of
    this call, so each submodule a caller needs (the kernel, or the tuner's
    ``conv3d_policy``) is loaded through here rather than imported afterwards.
    """
    target = f"{_PKG}.{name}"
    try:
        return importlib.import_module(target)
    except ImportError as exc:
        # The package imported and the submodule itself is what failed: not
        # the eager-__init__ case, so there is nothing to work around.
        if _PKG in sys.modules:
            raise
        first = exc

    spec = importlib.util.find_spec(_PKG)
    if spec is None or not spec.submodule_search_locations:
        raise first
    bare = types.ModuleType(_PKG)
    bare.__path__ = list(spec.submodule_search_locations)
    bare.__spec__ = spec
    bare.__package__ = _PKG
    sys.modules[_PKG] = bare
    try:
        mod = importlib.import_module(target)
    finally:
        if sys.modules.get(_PKG) is bare:
            del sys.modules[_PKG]
        parent = sys.modules.get(_PKG.rpartition(".")[0])
        if getattr(parent, "flydsl", None) is bare:
            delattr(parent, "flydsl")
    logger.info(
        "aiter conv: %s imported without running %s/__init__.py (%s)",
        target,
        _PKG,
        first,
    )
    return mod


def _count_lookups(mod):
    """Wrap the kernel's tuned-row lookup so every call's resolution is counted."""
    lookup = mod._lookup_tuned_tile
    if getattr(lookup, "_lumen_counted", False):
        return

    @functools.lru_cache(maxsize=1)
    def _device():
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx

        return (get_gfx(), get_cu_num())

    def counted(key, device):
        hit = lookup(key, device)
        if key is None:
            _stats["unkeyed"] += 1
            return hit
        if hit is None:
            how = "heuristic"
        elif (*_device(), *key) in mod._load_tuned_table():
            how = "exact"
        else:
            how = "borrowed"
        _stats[how] += 1
        _shapes[key] = how
        return hit

    counted._lumen_counted = True
    mod._lookup_tuned_tile = counted


@functools.lru_cache(maxsize=1)
def load():
    """``flydsl_conv_implicit`` from aiter, or None if this install cannot provide it."""
    try:
        _install_config_property()
        mod = import_flydsl_module("conv_kernels")
        fn = mod.flydsl_conv_implicit
    except (ImportError, OSError, AttributeError) as exc:
        logger.info("aiter conv: unavailable (%s: %s)", type(exc).__name__, exc)
        return None
    _count_lookups(mod)
    rows = len(mod._load_tuned_table())
    if not rows:
        logger.warning(
            "aiter conv: tuned table is empty; every convolution will take the heuristic tile"
        )
    logger.info("aiter conv: %s.conv_kernels with %d tuned rows", _PKG, rows)
    return fn


def tuned_lookup_stats():
    """Calls and distinct shapes resolved by exact tuned row, borrowed row, or heuristic."""
    distinct = {k: 0 for k in _stats if k != "unkeyed"}
    for how in _shapes.values():
        distinct[how] += 1
    return {"calls": dict(_stats), "shapes": distinct}


def tuned_misses():
    """Shape keys (in the tuned table's column order) that fell back to the heuristic."""
    return sorted(k for k, how in _shapes.items() if how == "heuristic")


_calls = {"flydsl": 0, "torch": 0}


def conv(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """``F.conv2d`` / ``F.conv3d`` by filter rank: aiter's kernel where it applies, torch otherwise."""
    fn = load()
    needs_grad = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
    if (
        fn is None
        or needs_grad
        or not x.is_cuda
        or x.dtype is not torch.bfloat16
        or weight.dtype is not torch.bfloat16
    ):
        _calls["torch"] += 1
        op = F.conv3d if weight.dim() == 5 else F.conv2d
        return op(
            x,
            weight,
            None if bias is None else bias.to(x.dtype),
            stride,
            padding,
            dilation,
            groups,
        )
    _calls["flydsl"] += 1
    return fn(x, weight, bias, stride, padding, dilation, groups=groups)


def conv_calls():
    """How many :func:`conv` calls ran on the FlyDSL kernel and how many on torch."""
    return dict(_calls)


__all__ = [
    "conv",
    "conv_calls",
    "import_flydsl_module",
    "load",
    "tuned_lookup_stats",
    "tuned_misses",
]
