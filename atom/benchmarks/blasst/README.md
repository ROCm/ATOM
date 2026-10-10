# BLASST benchmark

Measures BLASST block skipping in the Triton `unified_attention` kernel: speedup
against the same kernel at `block_skip_threshold=0`, how far the output moved,
and how much skipping actually happened.

## Reproduce

Needs one MI355X (gfx950).

Two dependencies are not in any released package, and the numbers below cannot
be reproduced without both:

- **AITER** carrying `block_skip_threshold` on `unified_attention`, from
  **ROCm/aiter#5868**, which is **not merged**. Without it the call raises
  `TypeError` on an unexpected keyword -- the benchmark does not run at all.
- **Triton** at `71d121b069` plus a patch forcing the chain-dot transform
  across an `scf.if`. Block skipping puts the `P@V` dot inside such an `if`,
  where Triton's region-local chain-dot detection stops firing and the dot
  takes a different warp layout from the dense path. The patch is **not
  upstream**, and the speedups below were measured with it active.

A container pinning and building both, with setup instructions and the sweep
scripts, is on the **`blasst-v2-repro`** branch -- see
`atom/benchmarks/blasst/_repro/README.md` there. It is kept off this branch so
the review diff stays on the feature.

The rest of this file documents the benchmark itself, and assumes that
environment is in place.

**1. Random inputs.** No model, no dataset -- run this first.

```bash
python -m atom.benchmarks.blasst.bench_unified_attention_blasst \
    --mode random --seqlens 16384 --shapes 32x8 \
    --thresholds 1e-9,0.02878,0.05756,1.0,2.0,4.0
```

MI355X, bf16, 32q/8kv, seqlen 16384:

| threshold | speedup | tile_elide |
|---|---|---|
| 1e-9 | 0.93x | 0.0% |
| 0.02878 | 0.93x | 0.0% |
| 0.05756 | 0.95x | 0.0% |
| 1 | 1.08x | 21.4% |
| 2 | 1.50x | 70.9% |
| 4 | 1.75x | 90.2% |

The first three rows are the overhead floor: nothing elides, so the ~7% extra
time is the fixed price of the skip check. **Both calibrated thresholds elide
0.0% here** -- random scores almost never produce a tile where every row agrees
to skip, where real activations at the same threshold elide 45.0%. Compare the
two input modes at matched `tile_elide`, never at matched threshold.

**2. Real activations.** Needs `transformers`, a model, and RULER data laid out
as `<ruler_root>/ctx_32768/<task>/validation.jsonl`. The repro branch has a
generator; sequence length is counted in the target model's tokens, so data
built for one model is not valid for another.

```bash
python -m atom.benchmarks.blasst.bench_unified_attention_blasst \
    --mode ruler --input-file <ruler_root>/ctx_32768/niah_single_1/validation.jsonl \
    --model <Qwen3-8B dir> --ruler-mode full --num-prompts 50 \
    --thresholds 0.02878 --csv niah_single_1.csv
```

**3. The headline.** All 13 tasks x 50 prompts x 36 layers = 23,400
measurements, ~2.5 h on one MI355X, timed at `BLASST_BENCH_WARMUP=3` and
`BLASST_BENCH_REPEAT=10`. The repro branch scripts the loop over all 13 tasks
and the reduction; the headline is a ratio of **summed** times, not a mean of
per-row speedups, which would weight a cheap layer the same as an expensive one.

Expect `1.2315x` at `45.0%` mean elision, by task from 1.1303x at 28.9%
(`niah_multikey_2`) to 1.3199x at 56.9% (`qa_2`). How much a task elides is a
property of its attention pattern, not of the threshold.

Elision also varies strongly by layer at one threshold and a single threshold serves every layer. 

## Flags

| flag | default | notes |
|---|---|---|
| `--mode` | `random` | `random` or `ruler` |
| `--shapes` | `64x4` | `NQxNKV` head configs, random mode |
| `--seqlens` | `16384,32768,65536` | random mode |
| `--thresholds` | *(built-in sweep)* | comma-separated |
| `--block-size` | `16` | KV page size |
| `--head-dim` | `128` | |
| `--ruler-mode` | `quick` | `quick` replays layers {0,7,18,35}; `full` replays all |
| `--layers` | *(none)* | explicit indices; overrides `--ruler-mode` |
| `--num-prompts` | `2` | ruler mode; prompts taken from the head of the file |
| `--skip-sparsity` | off | skip the elision measurement (one launch per threshold) |
| `--csv` | *(none)* | write rows to this path |

`--ruler-mode quick` is a fast check, not a substitute for `full`: per-layer
speedup varies widely enough that it can land several percent either side of the
full aggregate.

Timing is the mean of `REPEAT` launches after `WARMUP`, defaulting to 20 and 5.
Override with `BLASST_BENCH_REPEAT` / `BLASST_BENCH_WARMUP`; the headline sweep
above uses 10 and 3.

## Output

| column | meaning |
|---|---|
| `speedup` | dense ms / BLASST ms, same kernel both sides |
| `rel_diff` | mean relative difference from the `threshold=0` output |
| `tile_elide` | fraction of visited tiles the kernel actually skipped |

`tile_elide` is what predicts speedup. The V load and `P@V` are elided per tile,
so work is saved only when **every** row in the block votes to skip; partial
agreement saves nothing.

`rel_diff` is `mean|blasst − dense| / mean|dense|`. It measures how far the
approximation moved the answer, not correctness -- both sides are the same
kernel. Correctness is covered by the kernel's tests in AITER. Being a ratio of
means it cannot be read as "every element is within X%", and it grows large at
high thresholds where the kernel is discarding nearly everything.

## How elision is measured

From the **kernel's own counter**, not a replay. `unified_attention` takes an
optional `skip_counter` buffer and accumulates `[tiles visited, tiles elided]`
in it, computed by the kernel that actually ran over every tile it visited.

The counter costs one **untimed** extra launch per threshold  and does not perturb the timed runs.

`--skip-sparsity` turns that launch off and prints `-`. The timings should not
move; if they do, the counter has leaked into the measurement.

## Scope: the 2D kernel only

`unified_attention` dispatches to a 2D kernel (full prefill) or a 3D kernel plus
`reduce_segments` (few queries over long KV). **Block skipping exists only in the
2D kernel.** 
Every case therefore proves it reached the 2D kernel before measuring anything.
A shape that routes elsewhere prints `SKIPPED`.
