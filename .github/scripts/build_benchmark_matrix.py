#!/usr/bin/env python3
"""Compute the benchmark cell matrix for the ATOM Benchmark workflow.

Reads the GitHub event name and workflow_dispatch inputs from the environment
and emits the first-level matrix configs (variant × scenario, each carrying a
concurrency list; see ``catalog.build_cell_configs``) to ``$GITHUB_OUTPUT`` as
``configs_json`` plus a ``has_cells`` flag.

Behaviour by event:
- ``schedule``      -> all models, catalog ``default_scenarios`` (nightly grid).
- ``workflow_dispatch`` -> only models whose checkbox is ticked, workload from
  the ``param_lists`` input. Also validates that the dispatch model checkboxes
  stay in sync with the catalog prefixes (fails fast on drift).

This replaces the former inline Python in the ``parse-param-lists`` and
``load-models`` jobs so the logic is testable (see tests/ci/).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import (  # noqa: E402
    build_cell_configs,
    load_variants,
    validate_dispatch_inputs,
)

CATALOG = ".github/benchmark/models.json"
DEFAULT_PARAM_LISTS = "1024,1024,128,0.8"

# workflow_dispatch inputs that are NOT model toggles.
RESERVED_INPUTS = {
    "extra_args",
    "image",
    "runner",
    "enable_profiler",
    "enable_rtl",
    "param_lists",
    "atom_commit",
}


def _emit(configs: list[dict]) -> None:
    # One entry per first-level matrix config (variant × scenario); each carries
    # a JSON `concurrency` list the reusable template fans out over. Grouping
    # keeps both matrix levels far under GitHub's 256-job-per-matrix limit that a
    # flat per-cell matrix would overflow.
    payload = json.dumps(configs)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"configs_json={payload}\n")
            f.write(f"has_cells={'true' if configs else 'false'}\n")
    else:
        print(payload)


def main() -> int:
    event = os.environ.get("EVENT_NAME", "")
    inputs = json.loads(os.environ.get("INPUTS_JSON") or "{}")

    if event == "schedule":
        model_filter = None
        param_lists = None
    else:
        model_keys = {k for k in inputs if k not in RESERVED_INPUTS}
        problems = validate_dispatch_inputs(CATALOG, model_keys)
        if problems:
            for p in problems:
                print(f"ERROR: {p}", file=sys.stderr)
            print(
                "workflow_dispatch model checkboxes are out of sync with "
                f"{CATALOG}; update one to match the other.",
                file=sys.stderr,
            )
            return 1
        model_filter = {k for k in model_keys if inputs.get(k)}
        param_lists = inputs.get("param_lists") or DEFAULT_PARAM_LISTS

    configs = build_cell_configs(
        CATALOG, param_lists=param_lists, model_filter=model_filter
    )
    _emit(configs)

    n_cells = sum(len(json.loads(c["concurrency"])) for c in configs)
    n_models = len({c["prefix"] for c in configs})
    n_total = len(load_variants(CATALOG))
    print(
        f"Event={event}: {n_cells} cells across {n_models} models "
        f"-> {len(configs)} matrix configs ({n_total} variants in catalog)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
