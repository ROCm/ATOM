# Service-tested DSV4.1 V9 snapshot

This opt-in source snapshot preserves the owner/prefill combination used in the
2026-10-31 current-main C256 A/B/A/B. It is an experimental, dependency-pinned
deployment, with task accuracy still unqualified. Ordinary ATOM startup remains
unchanged. The model-side P1 implementation is in `owner_ffn_supply.py`.

`LOCK.json` records the measured model/owner source, AITER revision, runtime ASTs,
external binary/calibration hashes, qualification scope, deployment profile and
performance result. Runtime code covers owner single AGV, ordered BF16 RS,
limited G1 wait0, and explicit prefill route → top6/shared join → push RS.
The bootstrap retains the original formal observer so reproduction does not
silently change the measurement instrumentation.

The FlyDSL changes are stored as line edits against a hash-locked AITER builder,
including the native-fresh/full-wait references needed by the original runtime
module. Materialization reconstructs the **exact measured builder bytes** without
editing AITER or copying thousands of unchanged upstream lines into ATOM.
The ordered HIP header is reconstructed in the same way. The two `.cu` files are
the measured backend sources; original SPDX notices remain in upstream inputs.

On the original server, from the workspace containing `ATOM/` and `aiter/`:

```bash
python3 ATOM/tools/dsv41/v9/prepare.py \
  --reference-run validation/mainline_abab_20261031/main43_v9p2_b1_r1 \
  --aiter aiter \
  --output validation/v9_reproduction/new_run
```

This performs no GPU work and refuses an existing output directory. It verifies
the dependency, model/owner source, runtime ASTs and original calibration/binaries,
then materializes runtime code, bootstrap, qualification, environment and client
commands outside the checkout. Original qualification evidence remains external
and must still be present. This is intentionally a reproduction of the measured
server, not a claim that another machine has the same floating-point reduction
order. `numeric_r2/rank*_orders.pt` is required runtime calibration, not weights.
It is validated again against native RS at startup on every rank.

Use the generated `environment.list` with the server/client commands in
`manifest.json`, following independent stop/start/ready/client/drain/stop. The
formal client uses 512 regular warmup requests plus common C8/C12 supplemental
warmup, then 2560 formal requests at C256, random 8192/1024, range0.8, seed0.
Register each experiment's frozen planned record before GPU execution, and its
completed record afterward, including failures. Do not mix results from another
warmup protocol. Raw logs and experiment tooling remain outside the repository.

To rebuild the HIP backends, run inside the locked ROCm environment:

```bash
python3 ATOM/tools/dsv41/v9/build_kernels.py \
  --aiter aiter --output validation/v9_reproduction/new_build
```

Rebuilding checks header hashes and writes source/build/binaries into the external
directory. New binaries require numerical/graph and service qualification;
`prepare.py` deliberately reuses only the hash-locked measured binaries.
Dependency migration, expanding shapes/layers, enabling FP4 activation, or
removing instrumentation also require separate qualification. No unqualified
projection Quant/Sum prototypes are included.

Measured results and remaining accuracy/measurement qualifications are kept
with the external run records referenced by `LOCK.json`.
