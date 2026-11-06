# DSV4.1 HRC opt-in layer (D + G10 + H + R + C)

Best measured C256 combination as of 2026-11-04; superseded as the best arm by
`../pfx` (Q = this layer with R off + PF-CLR/PF-SHS), which chains it. Everything
is opt-in; ordinary ATOM startup is unchanged.

| Part | What | Switch |
|------|------|--------|
| D | V9 owner stack (`../`) | `--execution v9` |
| G10 | TP-sharded greedy argmax, no full-vocab gather (`model_runner.py`, `embed_head.py`) | `ATOM_GREEDY_SHARDED_ARGMAX=1` |
| H | moe2 layout row in `dsv41_fp8_tuned_fmoe.csv` + scoped decode FP8 wait0 G1 (`decode_fp8_wait0_v5.py`) | CSV in `AITER_CONFIG_FMOE` in place of AITER's `dsv41_fp8_tuned_fmoe.csv`, `DSV41_DECODE_FP8_WAIT0=1` |
| R | rope window single-wave launch (`rope_nw_v7.py`) | `DSV41_ROPE_OBS=1 DSV41_ROPE_NW=1` |
| C | prefill comm-fused MoE (MXFP8 reduce-scatter payload) for rows in (16384, bucket] (`commfused_v13.py`) | `DSV41_PRE_COMMFUSED=1 ATOM_DSV41_PREFILL_FFN_COMMFUSED=1 ATOM_DSV41_PREFILL_FFN_COMMFUSED_TUNER_DIR=<aiter>/op_tests/tuners` |

## Use

`python tools/dsv41/run_dashboard.py serve --execution hrc --aiter-source <aiter> --output <run>`
sets everything above (`owner_stacks.hrc` in `tools/dsv41/dashboard_profile.json`):
this directory first on `PYTHONPATH` (its `sitecustomize.py` runs the V9
bootstrap, then H, R, C in that order) and `AITER_CONFIG_FMOE` as AITER's
default merge with this CSV in place of `dsv41_fp8_tuned_fmoe.csv`.
`rope_nw_v7` writes counters under `$P0_COUNTER_DIR/ropenw`, C its hits under
`$ATOM_DSV41_PREFILL_FFN_COMMFUSED_HIT_DIR`. A failed install exits the server
process with 80 (H), 83 (R) or 86 (C).

## Historical evidence (ATOM 9a821a52d + AITER c141da6bd, 4×MI355X TP4; not re-measured on this tree)

- Service, C256 random 8192/1024, 2560 requests, same-suite ABBA:
  HR ≈ 303.8 s → HRC ≈ 290.9 s, **−12.94 s (−4.26 %)**. The gain of C is
  conditional on HR (C|HR); there is no independent C cell. Historical gains
  are not multiplied in.
- GSM8K (3-shot, 1319, c256), two runs per arm, paired by question:
  HRC − HR = −0.0057 flex / −0.0042 strict, one-sided 95 % lower bound
  −0.0136 / −0.0121 ≥ −0.015 → non-inferior. C path: 1532/1536 fused, 0 errors.
- Acceptance: full service win (repeat-qualified, conditional); service
  numerics lossy-pass (conditional); task accuracy non-inferior on GSM8K only.
  Short service, other tasks, long context and token-level parity: not measured.

## Scope

The numbers above are for the tree named in their heading. Running HRC on this
branch (newer ATOM main and AITER) needs its own service ABBA and GSM8K.
