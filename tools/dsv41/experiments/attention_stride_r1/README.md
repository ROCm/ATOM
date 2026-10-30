# Attention stride A0: source snapshot, GPU unrun

This directory publishes the prepared experiment from
`validation/dashboard_whole_20261029/attention_stride_r1/` outside the ATOM
repository. The 14 files in [source/](source/) are byte-for-byte copies;
[SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) records original paths and SHA256.
The original experiment remains in place. Nothing here installs a runtime
adapter or changes the dashboard default.

The adjacent `ruff.toml` keeps the original external-package classification
of `atom` for import sorting while inheriting the repository's checks. This
allows linting without rewriting the archived source bytes.

The candidate retains four native FP8 H16 calls, native split selection and
projection arithmetic. It passes strided Q/scale views and slices of a single
caller-owned BF16 output, removing four input copies and the final concat.
The fixture loads real layer8/layer9 attention and mHC weights; its hidden
states and historical caches are constructed. It covers FULL/REUSE selection,
full/ragged rows, cache writes, numerical/self comparisons and graph lifetime.
No FFN between the two attention windows is included.

CPU syntax, formatting and static layout checks were performed on the original
preparation. There has been **no GPU launch, weight load, numerical GPU result,
timing result or model accuracy acceptance** for A0. The recorded static check
is preparation evidence at original runtime commit
`96405d8609abdb0e9be594b47dcd4a492e81c6d4`, not validation of the rebased runtime.

Start with [candidate.py](source/candidate.py), [binding.py](source/binding.py),
[probe.py](source/probe.py) and the [original README](source/README.md) to review
the design. The original README/RESOURCE job state is a dated observation.

## Historical execution dependencies

This is an archival source package, not a standalone runnable benchmark.
Scripts derive paths from the original directory depth. Do not run the copied
`launch.py`, `driver.py`, `static_check.py` or `review.py` in this directory.
The launcher also checks the original runtime HEAD and exact allocation.
Rebase changes that contract; do not bypass the check by editing only its SHA.

The original harness depends on these external artifacts:

- `dashboard_whole_20261029/r8/common.py`, its cache/pool helpers and cache archive;
- the frozen `aiter_fp8` tree and `FP8_DEPENDENCY.json`;
- `owner_window_cost_r4/{fixture.py,device_clock.py,SOURCE.json.gz}` and
  `owner_window_cost_r3/output/window_clock.so`;
- `validation/structure_ceiling_20261024/serialize_comgr.so`;
- the pinned ROCm image, real checkpoint, four-GPU runtime and verified allocation.

The snapshot includes neither these binaries/dependency trees nor weights.
See the [dependency record](../../../../docs/dsv41_results_20261030/FP8_DEPENDENCY.json).
For a future run, prepare a new revision against the chosen runtime, inspect
native API changes and qualify pointers/layout/nonzero writes/graph retirement
before loading weights. Reuse only verified dependencies. Preserve this source
snapshot and the old receipts.

The complete optimization context and result boundaries are in the
[project summary](../../../../docs/dsv41_optimization_summary.md).
