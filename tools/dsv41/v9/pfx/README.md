# DSV4.1 PFX opt-in layer (Q = D + G10 + H + HB1 + C + PFX)

Best measured C256 arm as of 2026-11-05. It chains the HRC layer (`../hrc`)
and adds three prefill comm-fused changes. Everything is opt-in; ordinary
ATOM startup is unchanged, and neither the V9 runtime nor the HRC modules
are modified (this layer rebinds two functions after they load).

| Part | What | Switch |
|------|------|--------|
| D | V9 owner stack (`../`) | `--execution v9` |
| G10 | TP-sharded greedy argmax | `ATOM_GREEDY_SHARDED_ARGMAX=1` |
| H | tuned FMoE CSV + decode FP8 wait0 (`../hrc`) | `AITER_CONFIG_FMOE` with `../hrc/dsv41_fp8_tuned_fmoe.csv`, `DSV41_DECODE_FP8_WAIT0=1` |
| HB1 | upstream head-block RoPE (R off) | `DSV41_ROPE_OBS=1`, `DSV41_ROPE_NW` unset |
| C | prefill comm-fused MoE (`../hrc/commfused_v13.py`) | `DSV41_PRE_COMMFUSED=1 ATOM_DSV41_PREFILL_FFN_COMMFUSED=1 ATOM_DSV41_PREFILL_FFN_COMMFUSED_TUNER_DIR=<aiter>/op_tests/tuners` |
| nl shim | drop ATOM main's device `num_local_tokens` before `commfused_v13` (padded rows are already masked there) | always with C |
| PF-CLR v3 | `clear_routes.py`: one thread per route row for `compile_clear_missing_routes` | `DSV41_PFX=1` |
| PF-SHS v3 | `local_shared.py`: Triton shared-expert combine replacing `commfused_v13._local_shared` | `DSV41_PFX=1` |

Scheduler profile of the measured arm: `ATOM_PREFILL_DECODE_INTERVAL=1`,
`--max-num-batched-tokens 32768`, owner side streams 0, owner FFN supply `mhc`.

## Use

1. `python tools/dsv41/run_dashboard.py serve --execution pfx --aiter-source <aiter> --output <run>`
   applies all of the above (`owner_stacks.pfx` extends `hrc`: R off, PFX on,
   interval 1, 32768 batched tokens). This directory goes first on
   `PYTHONPATH`; its `sitecustomize.py` runs the HRC bootstrap first, then the
   nl shim, then PFX.
2. The server process prints `PFX_INSTALL_OK pid=<pid> pfx=1`; `HITS` in the
   module counts the nl drops and both rebinds. A failed install exits with 89.
3. Optional: call `compile_clear_missing_routes(196608, 1152, 1024)` once before
   serving so the FlyDSL compile is not inside the first prefill.

## Historical evidence (ATOM b5e03d5f8 + AITER 7cb3db1b7, 4×MI355X TP4, C256 random 8192/1024, 2560 requests)

- Same-session N/D/Q: N (origin/main native) 434.85 s (n=4), D 322.05 s (n=3),
  **Q 291.59 s (n=4)**. Q−D −9.46 % (t −7.7), Q−N −32.95 % (1.49× throughput).
  TTFT regresses vs N (2173 vs 1421 ms) by the interval=1 scheduling choice.
- PFX 2×2 on P = HB1R0C1 (SHS v2 then): PFX11 − PFX00 −1.94 % (t 3.0). CLR and
  SHS share one prefill window; their split is conditional.
- SHS v3 vs v2 (ABBA, 3+3): −0.94 %, not significant; kept because it is the
  simpler kernel with equal TPOT. Hits per pid: clr 20480, shs 10240; no JIT in
  window.
- Primitives: both kernels bitwise equal to the originals (10 cases).
- GSM8K (3-shot, 1319, c256, PQQP): Q 0.9185 vs P 0.9223 flexible (−0.38 pp,
  within same-arm repeat spread), every cell ≥ 0.91; task accuracy PASS.

## Scope

- The numbers are for the tree above. This branch is rebased again on a newer
  main; Q has not been re-measured on it.
- `clear_routes.py` belongs upstream in AITER
  `aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/producer.py`; once there,
  that part of this layer goes away.
