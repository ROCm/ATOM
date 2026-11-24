# Reproducing MiniMax-M3.1 TP4 performance: step-by-step

Two repos, side by side:

```bash
git clone --branch p1-c17-reproducible https://github.com/ganyi1996ppo/ATOM.git
git clone --branch p1-integration https://github.com/ganyi1996ppo/aiter.git
cd aiter && git submodule update --init --recursive && cd ..
```

**Important**: a shallow `git clone` of aiter does NOT fetch its `3rdparty/composable_kernel`
submodule; several JIT kernel builds hard-fail without it. Always run
`git submodule update --init --recursive` right after cloning aiter.

The ATOM checkout already contains everything else needed: `amd_kernels/` (vendored,
`Inferact/amd-kernels` @ `fd233ef`), `p1_sls_shared/`, `df_draft_ar/`, `l4c_y/`, `p1l2/`,
`p1k2/` (vendored PYTHONPATH-overlay packages the real recipe needs), and `recipe/`
(the full, adapted env-arm chain that produces the production C17 configuration).

## Start the server

Run this inside a container with ROCm/torch/aiter's build deps already set up (the project's
own `rocm/atom-dev` image works), with 4 GPUs available:

```bash
cd ATOM/recipe
export PYPATH_PRE=/path/to/your/aiter/clone     # the ONE thing you must set yourself
source ./common.sh
source ./env_armC17.sh      # the real, full C17 production recipe chain (B -> int1 -> C -> C2..C16 -> C17)
export TAG=my_repro_run CONC=1
bash /path/to/gpurun-equivalent/lease-4-gpus-then-run.sh ./serve.sh
# or, with no GPU-lease wrapper, simply:
#   HIP_VISIBLE_DEVICES=0,1,2,3 TAG=my_repro_run CONC=1 bash ./serve.sh
```

`env_armC17.sh` sets `REPO` (auto-detected from the recipe script's own location --
no hardcoded cluster path), every `ATOM_*`/`AITER_*` flag the C17 arm uses, and points
`AITER_CONFIG_GEMM_BF16`/`AITER_CONFIG_GEMM_A8W8_...` at the tuned-GEMM CSVs vendored in
`recipe/configs/`. `serve.sh` builds `PYTHONPATH=$PYPATH_PRE:$REPO` and launches
`python3 -m atom.entrypoints.openai_server` on port 8890 (+6 for the metrics port),
TP4, `--kv_cache_dtype nvfp4 --index-cache-dtype nvfp4`, DSpark k=7, prefix caching on.

Logs land at `${P1_OUT:-$HOME/p1-repro-out}/logs/server_$TAG.log` (override `P1_OUT` if
you want them elsewhere). Watch for `Server started successfully and ready to accept requests`.

**First run is slow**: there is no warmed FlyDSL JIT cache on a fresh machine, so every
custom kernel gets compiled from scratch the first time it is hit. In this verification
pass that took ~7 minutes end to end (weight load + full JIT warm-up) before the server
reported ready, with a lot of expected `warning: inline asm clobber list contains
reserved registers: m0` noise in the log -- harmless, not an error.

## Send a test request

```bash
curl -s http://127.0.0.1:8890/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m3.1","messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

This recipe runs with DSpark's **synthetic** acceptance length (`AL=4.04`, the golden
value for k=7) by default -- the response content will literally be repeated
`"synthetic"` tokens, by design. That is the normal, expected output for this
performance-benchmarking configuration; it exercises the real forward/decode/attention/
MoE/allreduce kernel path on real weights with the real recipe, it's just not meant to
produce coherent text. Pass `AL=real` to `serve.sh` if you want actual model answers
(needed for accuracy/GSM8K-style checks, not for performance reproduction).

## What this branch has verified end to end (this pass, 2026-11-24)

On node 035 (MI355X, TP4), from completely fresh clones of both branches above (no reuse
of any internal cluster worktree), with `PYTHONPATH` pointed at nothing but these two
checkouts:
- `import aiter` / `import atom` / `import amd_kernels` all resolve to the fresh clones,
  confirmed via each running server process's own `/proc/<pid>/environ`.
- The server reached "ready to accept requests" (DSpark k=7, prefix caching, NVFP4 KV)
  from a cold JIT cache.
- **20 real concurrent chat-completion requests, zero errors, zero timeouts**, every
  response a valid, complete (`finish_reason` = `stop` or `length`) chat completion.
- Clean shutdown: GPU memory returned to 0% on every leased GPU within 20s, no orphaned
  KFD processes.

One earlier attempt on a different GPU set (0-3, same node) hit a genuine hardware fault
(`Memory access fault ... Reason: DRAM ECC failure` / `SRAM ECC failure`) unrelated to
this branch -- a real transient GPU ECC error, not a code or recipe problem; it reproduced
cleanly on a second GPU set (4-7) right after. If you hit something that looks like this,
it's almost certainly your hardware, not this branch -- retry on different GPUs.

**Not re-verified this pass** (carried over from the prior HANDOFF.md write-up, still true):
a full 3600s-class official benchmark run from these exact checkouts. The previously-
measured C17 performance numbers come from the internal cluster tree these branches were
exported from; this pass confirms the exported code runs correctly end-to-end on fresh
hardware with real traffic, not that it reproduces the exact throughput/latency figures at
official duration/concurrency. See `HANDOFF.md` for the full honest status of every other
track (mono-G2, amd_kernels elimination) and what is/isn't claimed.
