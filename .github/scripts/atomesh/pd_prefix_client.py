"""Run the existing AIPerf CLI only after verifying client hook activation."""

import runpy

import sitecustomize

if not sitecustomize.ACTIVE:
    raise RuntimeError("prefix READ client import overlay is inactive")

# Resolve now, before any benchmark service starts. Every spawned client
# process inherits the same PYTHONPATH and validates the exact base on import.
from aiperf.timing import phase_orchestrator

if "/prefix-read-fixed/client/" not in phase_orchestrator.__file__:
    raise RuntimeError("unpatched AIPerf phase orchestrator")

# Spawned AIPerf children re-import this file as __mp_main__.
if __name__ == "__main__":
    runpy.run_module("aiperf", run_name="__main__")
