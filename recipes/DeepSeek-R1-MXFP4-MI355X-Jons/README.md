# AMD DSR1-MXFP4 Inference Optimization — Submission (Team Jons)

## Overview
Optimization of `DeepSeek-R1-0528-MXFP4` inference on 8× MI355X (gfx950) for AMD's competition benchmark at ISL=8192 / OSL=1024 across concurrency 4, 32, 128.

## Final leaderboard standings (Team Jons)

| Conc | Throughput per GPU | Score (out of 1000) | Target | % of target | Event ID |
|-----:|-------------------:|-------------------:|-------:|------------:|----------|
|    4 |             **757.12** | T#3 / I#2 — 840 |   1500 | 50.5% | `d2eb2378c2d540248005d9e1882a11b1` |
|   32 |            **2351.06** | T#1 / I#1 — 1000 |   3900 | 60.3% | `474be027ba7c4ec992371ff5f50508f2` |
|  128 |            **3537.19** | T#1 / I#1 — 1000 |   6000 | 58.9% | (May 8 submission) |
| **Total** | — | **2840 / 3000** | — | — | — |

## Key technical contribution
**Discovered that `DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` model gives faster inference at conc=4 than the standard `DeepSeek-R1-0528-MXFP4` model.** Same architecture but with the MoE separately FP4-quantized. Lower Mean TPOT (5.64-5.82ms vs 5.95-6.40ms) → higher throughput per GPU peak: 757.12 (vs 742 with standard model).

This pushed our c4 leaderboard from 742 → 757 (about +2% but enough to climb in T rank).

## Files

- `TECHNICAL_APPROACH.md` — what we changed and why
- `PERFORMANCE_METRICS.md` — throughput numbers + raw JSON
- `launchers/`
  - `launch_atom_c4_level3.sh` — conc=4 with standard model
  - `launch_atom_c4_level3_mtp_moefp4.sh` — **conc=4 with MoEFP4 model (BEST)**
  - `launch_atom_tp8_spec3_bigbatch.sh` — conc=128 (TOP-1)
  - `submit_c4_moefp4.sh` — submission script for c4 MoEFP4
  - `run_dsr1_c4only_moefp4.sh` — c4 perf-test driver for MoEFP4
- `results/`
  - `peak_c4_757_moefp4.json` — the 757.12 submission JSON
  - `submit_c32_bb_level3_*.json.json` — 2351 c32 submission JSON
  - `submit_bigbatch_c128_*.json.json` — 3537 c128 submission JSON
  - `submit_tp8_fp8_level3_c4_*.json.json` — prior 711 baseline (superseded by 757)
- `prototypes/`
  - `triton_mla_fp8_multi.py` — bonus: custom Triton fp8 MLA kernel (functionally correct, perf needs work)
  - `TRITON_FP8_MLA_HANDOFF.md` — handoff doc
  - `sglang_patches/deepseek_weight_loader.py` — partial SGLang MTP loader fixes
