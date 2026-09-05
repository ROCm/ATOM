# Performance Metrics — Submitted Results (Team Jons)

All numbers from AMD's `dsr1_benchmark` harness. GSM8K validation passed (`gsm8k_metric ≥ 0.93`).

## Final leaderboard submissions

### conc=4 — 757.12 tok/s/GPU (Event `d2eb2378c2d540248005d9e1882a11b1`)
| Metric | Value |
|--------|------:|
| **Throughput per GPU** | **757.12 tok/s** |
| Total Token throughput | 6056.94 tok/s |
| Mean TPOT (ms) | 5.64 |
| Median TPOT (ms) | 6.07 |
| P99 TPOT (ms) | 7.22 |
| Mean TTFT (ms) | 267.59 |
| Median E2E (ms) | 6477.40 |
| Interactivity (tok/s/user) | 162.8 |
| GSM8K | 0.9356 ✓ |
| **Config** | TP=8 fp8 spec=3 level=3 cudagraph=[1,2,4,8] **DSR1-MXFP4-MTP-MoEFP4 model** |
| Baseline target | 1500 tok/s |

### conc=32 — 2351.06 tok/s/GPU (Event `474be027ba7c4ec992371ff5f50508f2`)
| Metric | Value |
|--------|------:|
| **Throughput per GPU** | **2351.06 tok/s** |
| Total Token throughput | 18808.52 tok/s |
| Mean TPOT (ms) | ~14.7 |
| Interactivity | 65.5 tok/s/user |
| GSM8K | 0.9393 ✓ |
| **Config** | TP=8 fp8 spec=3 + bigbatch + level=3 + wide cudagraph **(DSR1-MXFP4)** |
| Baseline target | 3900 tok/s |

### conc=128 — 3537.19 tok/s/GPU (May 8 submission)
| Metric | Value |
|--------|------:|
| **Throughput per GPU** | **3537.19 tok/s** |
| Total Token throughput | 28297.49 tok/s |
| Mean TPOT (ms) | 38.32 |
| Interactivity | 24.07 tok/s/user |
| GSM8K | 0.9348 ✓ |
| **Config** | TP=8 fp8 spec=3 + `max-num-batched-tokens=131072` + `max-num-seqs=256` **(DSR1-MXFP4)** |
| Baseline target | 6000 tok/s |

## Key finding: model matters at conc=4

We tested both available model variants on identical config:

| Model | Conc=4 peak | Median TPOT |
|-------|------------:|------------:|
| `DeepSeek-R1-0528-MXFP4` (376GB, 82 shards) | 736.80 | 6.40 ms |
| `DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` (350GB, 76 shards) | **757.12** | **6.07 ms** |

At conc=32 and conc=128, the standard model was faster — the MoEFP4 variant only helps at conc=4 (small batches benefit from the FP4 MoE quant).

## Hardware
- 8× AMD MI355X (gfx950) per node
- ROCm 7 (in `rocm/atom` image)
- Inference engine: ATOM v0.1.2 (`rocm/atom:rocm7.2.1-ubuntu24.04-pytorch2.9.1-atom0.1.2`)
