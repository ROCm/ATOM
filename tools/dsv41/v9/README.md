# DSV4.1 V9 owner stack (D)

Opt-in service stack on top of the FP8 owner execution
(`atom/models/deepseek_v41/owner`, `tools/dsv41/owner_bootstrap`). Ordinary
ATOM startup is unchanged. `hrc/` and `pfx/` are the next layers; each chains
the previous one.

What it adds, through `dsv41_v9.guarded_service_v2` (`DSV41_COMBINED_GUARDED=1`):

- Owner 384-row steps: Mega-MHC supply (`new_supply`), the minimal-wait
  stage-1 G1 (`variants`, `flydsl_minimal`) and the ordered BF16
  reduce-scatter (`ordered_adapter`, HIP `csrc/ordered_rs.cu`).
- Prefill FFN on layers 2-18 with 4096-16384 rows: explicit route → top6 /
  shared join → push RS (`explicit_ffn`, `explicit_routes`, HIP
  `csrc/push_join.cu`).
- Anything else falls through to the native implementation.

The ordered RS summation order comes from `formula.make_codes`; at startup each
rank checks it bitwise against the native RS and refuses to serve otherwise
(`$DSV41_COMBINED_REPORT/rank*_startup.json`). The FlyDSL change is applied as
small checked text edits to the installed AITER builder, loaded as a sibling
module, so AITER itself is not edited.

## Run

Through the dashboard runner, which sets the owner and V9 environment, the run
directories and the kernel cache:

```bash
python tools/dsv41/run_dashboard.py serve --execution v9 \
  --aiter-source <aiter> --output <fresh run dir>
python tools/dsv41/run_dashboard.py benchmark --execution v9 \
  --aiter-source <aiter> --output <fresh run dir>/client
```

`--execution hrc` / `pfx` select the later layers. The switches live in
`owner_stacks` of `tools/dsv41/dashboard_profile.json`; `plan` prints the
resolved environment without starting anything.

The HIP kernels build on first use into `$DSV41_V9_KERNEL_DIR` (default
`~/.cache/atom/dsv41_v9_kernels`), against the AITER custom all-reduce header.
To build them ahead of serving:

```bash
python tools/dsv41/v9/build_kernels.py --cache-dir <cache>/v9_kernels
```

A failed install exits the server process with 78. Service counters go to
`$P0_COUNTER_DIR`, owner graph coverage next to `$DSV41_OWNER_GRAPH_PHASE_FILE`.

## Status

The 2026-10-31 C256 A/B/A/B was measured on an older ATOM/AITER pair with the
earlier, separately calibrated version of this stack. That result does not
transfer: this tree needs its own service A/B and task-accuracy run.
