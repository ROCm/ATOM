# SPDX-License-Identifier: MIT
"""ATOM_HOSTFAST_TRITON: launch hot decode-step Triton kernels without
re-running Triton's per-call binder / specialization / cache-key work.

``JITFunction.run`` binds every argument, calls back into Python for every
tensor's specialization (``get_tensor_specialization`` / ``is_within_2gb`` on
AMD), stringifies a cache key and looks the kernel up, on every launch. py-spy of
the c64 decode loop puts that at ~295 us a step across ~15 launches, against
~31 us for the launches themselves.

`FastLaunch` wraps one JITFunction. Its key is a *refinement* of Triton's own
specialization, built from the same per-parameter flags Triton uses
(constexpr / do_not_specialize / do_not_specialize_on_alignment):

* constexpr parameter          -> (its value, its type)
* tensor (specialized)          -> (dtype, ptr % 16 == 0 if aligned, storage <= 2 GiB if buffer ops)
* tensor (do_not_specialize)    -> dtype
* int (specialized)             -> (v == 1, i32/i64/u64 class, v % 16 == 0 if aligned)
* int (do_not_specialize)       -> i32/i64/u64 class
* bool                          -> value (finer than Triton's i1)
* float                         -> "fp32" (Triton never value-specializes a non-constexpr float)
* None                          -> None
* launch options (num_warps...) -> value

Two calls with equal keys therefore get the same Triton specialization, i.e.
the same compiled kernel, so the cached ``CompiledKernel`` is exactly what
Triton would have picked: numerics are unchanged by construction. Any argument
of another type (numpy scalar, tuple, descriptor, ...) takes Triton's ordinary
path for that call. A kernel with pre-run hooks, or a process with launch
enter/exit hooks installed (proton), always takes the ordinary path.

A miss is an ordinary Triton launch (which still hits Triton's own compile
cache); its CompiledKernel is remembered under the key.

Assumes the global Triton knobs that feed the compile options (debug,
instrumentation mode, the stage-inspection hook) do not change after a kernel's
first launch, as in serving.
"""

import importlib
import os

import torch
from triton import knobs
from triton.runtime import driver
from triton.runtime.jit import JITFunction

_MISSING = object()
_FLOAT = "fp32"
_I32_MIN, _I32_MAX = -(2**31), 2**31 - 1
_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1
_MAX_INT32 = 2**31 - 1
_MAX_ENTRIES = 4096

_Tensor = torch.Tensor

# Modes, one a parameter.
_CONSTEXPR, _SPEC_ALIGN, _SPEC_NOALIGN, _NOSPEC = 0, 1, 2, 3

stats = {"hit": 0, "miss": 0, "bypass": 0}
_LOG_EVERY = int(os.environ.get("ATOM_HOSTFAST_TRITON_LOG", "0") or 0)
# ATOM_HOSTFAST_TRITON_VERIFY=1: on every hit also run Triton's own binder and
# cache lookup and fail loudly if it would have picked a different kernel. Costs
# more than it saves; for a correctness run before the timed A/B.
_VERIFY = os.environ.get("ATOM_HOSTFAST_TRITON_VERIFY", "0") == "1"


def _triton_choice(kernel, device, args, kwargs):
    """The CompiledKernel ``JITFunction.run`` (Triton 3.9) selects for this
    call, or None if it would compile."""
    from triton.runtime.jit import compute_cache_key

    kw = dict(kwargs)
    kw["debug"] = kw.get("debug", kernel.debug) or knobs.runtime.debug
    kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
    kw["fpsan_homomorphic_casts"] = knobs.compilation.fpsan_homomorphic_casts
    kernel_cache, kernel_key_cache, _target, _backend, binder = kernel.device_caches[
        device
    ]
    _bound, specialization, options = binder(*args, **kw)
    if knobs.runtime.add_stages_inspection_hook is not None:
        _k, digest = knobs.runtime.add_stages_inspection_hook()
        specialization.append(f'("custom_pipeline", {digest})')
    return kernel_cache.get(
        compute_cache_key(kernel_key_cache, specialization, options)
    )


def _hooks_active() -> bool:
    rt = knobs.runtime
    enter, leave = rt.launch_enter_hook, rt.launch_exit_hook
    return bool(getattr(enter, "calls", enter)) or bool(getattr(leave, "calls", leave))


def _int_class(v: int) -> int:
    if _I32_MIN <= v <= _I32_MAX:
        return 0
    if _I64_MIN <= v <= _I64_MAX:
        return 1
    return 2


class FastLaunch:
    """``kernel[grid](*args, **kwargs)`` with a cached CompiledKernel. Every
    other attribute (``warmup``, ``run``, ``fn``, ...) is the wrapped kernel's."""

    def __init__(self, kernel: JITFunction):
        self._kernel = kernel
        params = kernel.params
        self._names = tuple(p.name for p in params)
        self._index = {n: i for i, n in enumerate(self._names)}
        modes = []
        for p in params:
            if p.is_constexpr:
                modes.append(_CONSTEXPR)
            elif p.do_not_specialize:
                modes.append(_NOSPEC)
            elif p.do_not_specialize_on_alignment:
                modes.append(_SPEC_NOALIGN)
            else:
                modes.append(_SPEC_ALIGN)
        self._modes = tuple(modes)
        self._defaults = tuple(p.default if p.has_default else _MISSING for p in params)
        self._cache: dict = {}

    def __getattr__(self, name):
        if name == "_kernel":
            raise AttributeError(name)
        return getattr(self._kernel, name)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return self._call(grid, args, kwargs)

        return launch

    def _ordinary(self, grid, args, kwargs):
        stats["bypass"] += 1
        return self._kernel[grid](*args, **kwargs)

    def _call(self, grid, args, kwargs):
        kernel = self._kernel
        if kernel.pre_run_hooks or _hooks_active():
            return self._ordinary(grid, args, kwargs)
        names = self._names
        nargs = len(args)
        if nargs > len(names):
            return self._ordinary(grid, args, kwargs)
        vals = list(args)
        if nargs < len(names):
            vals.extend(self._defaults[nargs:])
        opts = ()
        if kwargs:
            index = self._index
            extra = None
            for k, v in kwargs.items():
                i = index.get(k)
                if i is None:
                    if extra is None:
                        extra = []
                    extra.append((k, v))
                elif i < nargs:
                    return self._ordinary(grid, args, kwargs)  # Triton raises
                else:
                    vals[i] = v
            if extra:
                opts = tuple(extra)

        buffer_ops = knobs.amd.use_buffer_ops
        modes = self._modes
        key = [opts]
        append = key.append
        for i, v in enumerate(vals):
            m = modes[i]
            if m == _CONSTEXPR:
                if v is _MISSING:
                    return self._ordinary(grid, args, kwargs)
                append(v)
                append(type(v))  # True / 1 / 1.0 are distinct constexprs to Triton
                continue
            tv = type(v)
            if tv is _Tensor or isinstance(v, _Tensor):
                if m == _NOSPEC:
                    append(v.dtype)
                else:
                    append(
                        (
                            v.dtype,
                            m == _SPEC_ALIGN and (v.data_ptr() & 15) == 0,
                            buffer_ops and v.untyped_storage().size() <= _MAX_INT32,
                        )
                    )
            elif tv is int:
                if m == _NOSPEC:
                    append(_int_class(v))
                else:
                    append((v == 1, _int_class(v), m == _SPEC_ALIGN and (v & 15) == 0))
            elif tv is bool:
                append(("b", v))
            elif tv is float:
                append(_FLOAT)
            elif v is None:
                append(None)
            else:
                return self._ordinary(grid, args, kwargs)
        device = driver.active.get_current_device()
        append(device)
        key = tuple(key)
        try:
            compiled = self._cache.get(key)
        except TypeError:  # an unhashable constexpr / option
            return self._ordinary(grid, args, kwargs)
        if compiled is None:
            stats["miss"] += 1
            compiled = kernel[grid](*args, **kwargs)
            if compiled is not None and hasattr(compiled, "packed_metadata"):
                if len(self._cache) >= _MAX_ENTRIES:
                    self._cache.clear()
                self._cache[key] = compiled
            return compiled

        stats["hit"] += 1
        if _VERIFY:
            ref = _triton_choice(kernel, device, args, kwargs)
            if ref is not compiled:
                raise RuntimeError(
                    f"ATOM_HOSTFAST_TRITON: cached kernel for {kernel.fn.__name__} differs "
                    f"from Triton's choice (key {key})"
                )
        if _LOG_EVERY and stats["hit"] % _LOG_EVERY == 0:
            print(f"[ATOM hostfast triton] pid {os.getpid()} {stats}", flush=True)
        if callable(grid):
            grid = grid(dict(zip(names, vals)))
        n = len(grid)
        # Not hoisted: graph capture redirects the current stream.
        stream = driver.active.get_current_stream(device)
        compiled.run(
            grid[0],
            grid[1] if n > 1 else 1,
            grid[2] if n > 2 else 1,
            stream,
            compiled.function,
            compiled.packed_metadata,
            None,  # launch_metadata: only read by launch hooks (none installed)
            None,
            None,
            *vals,
        )
        return compiled


def fast_launch(kernel):
    """``kernel`` wrapped, or ``kernel`` itself when it is not a plain
    JITFunction (autotuner, heuristics, already wrapped)."""
    if isinstance(kernel, FastLaunch) or not isinstance(kernel, JITFunction):
        return kernel
    import inspect

    for p in kernel.params:
        if p._param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return kernel
    return FastLaunch(kernel)


# The decode step's host-heavy launches (py-spy, c64 MTP-3): module, global name.
# Every one is launched only as ``name[grid](...)`` through its module global and
# never called from another kernel, so replacing the global is enough.
TARGETS = (
    ("atom.spec_decode.eagle_fused_glue", "_advance_draft_pass_kernel"),
    ("atom.spec_decode.eagle_fused_glue", "_last_draft_pass_kernel"),
    ("atom.spec_decode.eagle_fused_glue", "_stage_verify_step0_kernel"),
    ("atom.spec_decode.eagle_fused_glue", "_token_split_tiles_kernel"),
    ("atom.model_ops.verify_epilogue", "_verify_epilogue_kernel"),
    ("atom.model_ops.decode_input_ids", "_fill_deferred_decode_ids_kernel"),
    ("atom.utils.block_convert", "kv_indices_generate_kernel"),
    ("atom.model_ops.presence_penalty", "_apply_presence_penalty_kernel"),
    ("atom.model_ops.presence_penalty", "_mark_tokens_kernel"),
    ("atom.model_ops.spec_rejection", "_rs_verify_kernel"),
    ("atom.model_ops.spec_rejection", "_draft_uniform_kernel"),
    ("aiter.ops.triton.fusions.topk_topp_sample", "_topk_topp_sample_stats_kernel"),
    ("aiter.ops.triton.fusions.topk_topp_sample", "_topk_topp_sample_select_kernel"),
    (
        "aiter.ops.triton._gluon_kernels.gfx1250.fusions.topk_topp_sample",
        "_topk_topp_sample_stats_gluon_kernel",
    ),
    (
        "aiter.ops.triton._gluon_kernels.gfx1250.fusions.topk_topp_sample",
        "_topk_topp_sample_select_gluon_kernel",
    ),
)

_installed = False


def install_from_env() -> list:
    """ATOM_HOSTFAST_TRITON=1: wrap `TARGETS` (plus ATOM_HOSTFAST_TRITON_EXTRA,
    ``module:name`` comma-separated). Returns the names wrapped."""
    global _installed
    if os.environ.get("ATOM_HOSTFAST_TRITON", "0") != "1" or _installed:
        return []
    _installed = True
    targets = list(TARGETS)
    for item in os.environ.get("ATOM_HOSTFAST_TRITON_EXTRA", "").split(","):
        if ":" in item:
            mod, _, name = item.strip().partition(":")
            targets.append((mod, name))
    done = []
    for mod_name, attr in targets:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001, S112
            continue  # an optional backend not importable here
        kernel = getattr(mod, attr, None)
        if kernel is None:
            continue
        wrapped = fast_launch(kernel)
        if wrapped is not kernel:
            setattr(mod, attr, wrapped)
            done.append(f"{mod_name}.{attr}")
    print(f"[ATOM hostfast triton] wrapped {len(done)} kernels: {done}", flush=True)
    return done
