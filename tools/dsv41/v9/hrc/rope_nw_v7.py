"""DEC-ROPE-NW v7: launch-config patch + observer for atom.model_ops.deepseek_v41.rope_window._rope_window_kernel.

All cells install the same proxy (post-import hook on the rope_window module). rope_quant_window looks up the
module global `_rope_window_kernel` at call time, so every caller (cache.py decode/prefill, dspark drafter,
mono check) goes through the proxy. DSV41_ROPE_NW=1 -> launch with num_warps=1 (algorithm and constexprs
unchanged; microbench rope_mb/ byte-equal vs default num_warps=4 at T in {1024,1536,8192}, H in {16,64},
ring/no-ring). Unset -> pass-through (no num_warps kwarg), observer only.
Counts are Python-call-time keyed (grid0, grid1, num_warps, capturing): captured graphs count once per capture.
Dumps <run>/ropenw/counts_<pid>.json on every new key and every 4096 calls."""

import importlib.abc, json, os, sys, threading  # noqa: I001
from pathlib import Path

TARGET = "atom.model_ops.deepseek_v41.rope_window"
OUT = Path(os.environ["DSV41_SERVICE_RUN"]) / "ropenw"
NW = int(os.environ["DSV41_ROPE_NW"]) if os.environ.get("DSV41_ROPE_NW") else None
assert NW in (None, 1), NW
COUNTS = {}
STATE = {"calls": 0, "mode": None, "nw": NW}
LOCK = threading.Lock()


def _dump():
    OUT.mkdir(exist_ok=True)
    rows = [
        {"grid0": k[0], "grid1": k[1], "num_warps": k[2], "capturing": k[3], "count": v}
        for k, v in sorted(COUNTS.items(), key=str)
    ]
    tmp = OUT / f".counts_{os.getpid()}.tmp"
    tmp.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "calls": STATE["calls"],
                "mode": STATE["mode"],
                "nw_env": NW,
                "rows": rows,
            }
        )
        + "\n"
    )
    os.replace(tmp, OUT / f"counts_{os.getpid()}.json")


class _Proxy:
    _dsv41_ropenw = True

    def __init__(self, orig):
        self._orig = orig

    def __getattr__(self, name):
        return getattr(self._orig, name)

    def __getitem__(self, grid):
        launch = self._orig[grid]

        def run(*args, **kwargs):
            import torch

            g = tuple(int(x) for x in grid) + (1,) * (2 - len(grid))
            key = (g[0], g[1], NW, bool(torch.cuda.is_current_stream_capturing()))
            with LOCK:
                new = key not in COUNTS
                COUNTS[key] = COUNTS.get(key, 0) + 1
                STATE["calls"] += 1
                if new or STATE["calls"] % 4096 == 0:
                    _dump()
            if NW is None:
                return launch(*args, **kwargs)
            assert "num_warps" not in kwargs
            return launch(*args, num_warps=NW, **kwargs)

        return run


def _wrap(mod):
    if getattr(mod._rope_window_kernel, "_dsv41_ropenw", False):
        return
    mod._rope_window_kernel = _Proxy(mod._rope_window_kernel)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _wrap(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = (
                finder.find_spec(fullname, path, target)
                if hasattr(finder, "find_spec")
                else None
            )
            if spec is not None:
                spec.loader = _Loader(spec.loader)
                return spec
        return None


def install():
    if TARGET in sys.modules:
        STATE["mode"] = "already_imported_patch"
        _wrap(sys.modules[TARGET])
    else:
        STATE["mode"] = "post_import_hook"
        sys.meta_path.insert(0, _Finder())
    _dump()
    return {"mode": STATE["mode"], "nw": NW}
