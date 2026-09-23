# V4.1 Flash agentic sweep

Branch: `lirzhang/v41-agentic-tp4-noep-main-ci`.
CI recipe: `lirzhang/agentic-benchmark-ci` at `a86e0849`.
Branch created directly from ATOM main at `cdbd5f2005da78b6e9ef533a395293810587155a`, including
V4.1 level-3/FULL support. The older local runtime rejected level 3.

Run **ATOM Benchmark** on this branch with `agentic_deepseek_v41flash=true`,
`agentic_concurrency=all`, `agentic_duration=3600`, and the other agentic
toggles false. Leave `atom_commit` empty to use this branch. The random-model
checkboxes are ignored when the V4.1 agentic toggle is selected.
The branch also contains a manual-only **ATOM Agentic Benchmark** workflow
that selects only V4.1.

The matrix runs concurrency 1, 2, 8, 16, 32, 64. Every cell uses TP4 on GPUs
0-3, no EP or DPA, level 3, FULL graphs, BF16 KV, FP8 index, DSpark5, synthetic
acceptance length 3.51, prefix caching, max_num_seqs=128, 16k batched tokens,
and an 8192-token checkpoint interval. Each cell profiles for 3600 seconds;
startup, warmup and drain are additional. Warmup uses 5 requests per lane.
The c32 cell retains the previous dense capture list (1-32,48,64,128); other
cells use 1-8,16,32,48,64,128.

The runtime includes the same explicit `ATOM_DSV41_BENCHMARK_SYNTHETIC=1`
admission opt-in used by the previous local benchmark. Default admission
still rejects synthetic acceptance. This is a performance benchmark and does
not measure generation quality.

Default image: `rocm/atom-dev:latest`. Default runner label:
`atom-mi355-8gpu.predownload`. Each cell reserves a runner through the existing
CI template and runs the model on four GPUs. Artifacts are attached per cell.
