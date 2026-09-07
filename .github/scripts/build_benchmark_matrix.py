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
from catalog import (
    build_cell_configs,
    build_cells,
    load_variants,
    validate_dispatch_inputs,
)

CATALOG = ".github/benchmark/models.json"
DEFAULT_PARAM_LISTS = "1024,1024,128,0.8"

# Each dedicated agentic toggle replays ONE model's ~1h/cell agentic curve,
# independent of the random model grid. Keys map to catalog prefixes; the map
# must cover exactly the catalog's agentic prefixes (asserted in the tests).
AGENTIC_MODEL_INPUTS = {
    "agentic_deepseek_v4pro": "deepseek-v4-pro",
    "agentic_m3_mxfp4": "m3-mxfp4-agentic",
}

# workflow_dispatch inputs that are NOT random model-grid toggles (the agentic
# model toggles included -- they select agentic replays, not the random sweep).
RESERVED_INPUTS = {
    *AGENTIC_MODEL_INPUTS,
    "agentic_duration",
    "agentic_concurrency",
    "extra_args",
    "image",
    "runner",
    "enable_profiler",
    "enable_rtl",
    "param_lists",
    "atom_commit",
    "publish_to_dashboard",
}


def parse_conc_filter(raw: str | None) -> set[int] | None:
    """Parse the `agentic_concurrency` dispatch box into a `conc_filter`.

    Free text rather than a fixed option list: the box takes any subset of the
    curve ("48,64"), and nothing has to be kept in sync with the catalog's
    concurrency lists. Empty or "all" (any case) means no filter.

    Raises ValueError on anything else rather than falling back to "all" -- a
    typo that silently ran the whole 9-cell sweep would cost most of a day of
    8-GPU time, which is exactly what this box exists to avoid.
    """
    picked = (raw or "").strip()
    if not picked or picked.lower() == "all":
        return None
    out: set[int] = set()
    for part in picked.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            raise ValueError(
                f"agentic_concurrency: {part!r} is not a number -- expected "
                f"'all', or a comma-separated list such as '48,64'"
            ) from None
    # A box holding only separators ("," / " , ") carries no selection.
    return out or None


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

    # The nightly must never pick up the agentic variants -- each is ~1h of
    # 8-GPU time. `BENCH_KIND_FILTER` is the explicit override
    # (atom-agentic-benchmark.yaml pins it to `aiperf_agentic` and replays every
    # agentic model on a schedule); otherwise a dispatch opts in per model via a
    # dedicated agentic checkbox and a schedule never does.
    agentic_models = {
        prefix for key, prefix in AGENTIC_MODEL_INPUTS.items() if inputs.get(key)
    }
    env_filter = os.environ.get("BENCH_KIND_FILTER", "")
    if env_filter:
        bench_kinds = {k for k in env_filter.split(",") if k}
    elif event != "schedule" and agentic_models:
        # EXCLUSIVE over bench_kind: only agentic variants run, never the random
        # sweep in the same dispatch (mixing a few random cells into an agentic
        # run only makes the curve harder to read). The dedicated toggles select
        # WHICH models directly, so the run is exactly the ticked agentic models
        # -- no interaction with the random grid checkboxes above.
        bench_kinds = {"aiperf_agentic"}
        model_filter = agentic_models
        param_lists = None
    else:
        bench_kinds = {"random"}

    # `agentic_concurrency` narrows the replay to the picked point(s) of the
    # curve. Agentic-only on purpose: a cell is ~1h of 8-GPU time, so re-running
    # one point instead of the 9-cell set is the difference between an hour and
    # most of a day. The random sweep is cheap per cell and already has
    # `param_lists` for that, so the filter never touches it.
    conc_filter = None
    if "aiperf_agentic" in bench_kinds:
        try:
            conc_filter = parse_conc_filter(inputs.get("agentic_concurrency"))
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    configs = build_cell_configs(
        CATALOG,
        param_lists=param_lists,
        model_filter=model_filter,
        bench_kind_filter=bench_kinds,
        conc_filter=conc_filter,
    )
    if conc_filter and not configs:
        # The box is free text, so this is usually a typo rather than catalog
        # drift. Fail loudly and name what IS runnable: `has_cells=false` would
        # just skip the run in silence and look like a passing dispatch.
        available = sorted(
            {c["conc"] for c in build_cells(CATALOG, bench_kind_filter=bench_kinds)}
        )
        print(
            f"ERROR: no agentic cells at concurrency {sorted(conc_filter)}. "
            f"Available in {CATALOG}: "
            f"{','.join(str(c) for c in available)} (or 'all').",
            file=sys.stderr,
        )
        return 1
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
