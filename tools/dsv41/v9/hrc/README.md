# DSV4.1 HRC opt-in layer (D + G10 + H + R + C)

Best measured C256 combination as of 2026-11-04; superseded as the best arm by
`../pfx` (Q = this layer with R off + PF-CLR/PF-SHS), which chains it. Everything
is opt-in; ordinary ATOM startup is unchanged.

| Part | What | Switch |
|------|------|--------|
| D | V9 owner service (`../`, prepared by `../prepare.py`) | V9 `environment.list` |
| G10 | TP-sharded greedy argmax, no full-vocab gather (`model_runner.py`, `embed_head.py`) | `ATOM_GREEDY_SHARDED_ARGMAX=1` |
| H | moe2 layout row in `dsv41_fp8_tuned_fmoe.csv` + scoped decode FP8 wait0 G1 (`decode_fp8_wait0_v5.py`) | CSV in `AITER_CONFIG_FMOE` in place of AITER's `dsv41_fp8_tuned_fmoe.csv`, `DSV41_DECODE_FP8_WAIT0=1` |
| R | rope window single-wave launch (`rope_nw_v7.py`) | `DSV41_ROPE_OBS=1 DSV41_ROPE_NW=1` |
| C | prefill comm-fused MoE (MXFP8 reduce-scatter payload) for rows in (16384, bucket] (`commfused_v13.py`) | `DSV41_PRE_COMMFUSED=1 ATOM_DSV41_PREFILL_FFN_COMMFUSED=1 ATOM_DSV41_PREFILL_FFN_COMMFUSED_TUNER_DIR=<aiter>/op_tests/tuners` |

## Use

1. Prepare D exactly as in `../README.md` (pinned AITER, pinned model source).
2. Use the generated `environment.list`, prepend `ATOM/tools/dsv41/v9/hrc` to
   `PYTHONPATH` (this `sitecustomize.py` runs the V9 bootstrap first, then H, R,
   C in the measured order), and add the switches above.
3. `rope_nw_v7` writes counters under `$DSV41_SERVICE_RUN/ropenw`. A failed
   install exits the server process with 80 (H), 83 (R) or 86 (C).

`LOCK.json` ties each module to its frozen file in the measured run: only
black/noqa formatting differs (normalized AST equal), and the CSV is byte equal.

## Evidence (measured on ATOM 9a821a52d + AITER c141da6bd, 4×MI355X TP4)

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

This branch is rebased on ATOM main, which needs a newer AITER than the pinned
c141da6bd, and main changed `atom/models/deepseek_v41/model.py`. `../prepare.py`
therefore refuses this tree by design. Running D/HRC on the rebased tree is a
dependency migration and needs its own qualification (service ABBA and GSM8K);
the numbers above do not transfer automatically.
