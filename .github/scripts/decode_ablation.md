# TP4/C1 focused decode attribution

Run `decode_ablation.py prepare`, then `run` for `official37`, `current38`,
`current37`, `current38_repeat` in that order, and always `cleanup`.
The existing `image_ab.py` owns image digests, server flags, model mount,
container identity, GPU guards and the exact baseline client lock.

This is a diagnostic synthetic-prompt experiment, not an AgentX replay or an
SA score. It preserves TP4, C1, DSpark5, expected acceptance length 3.51,
FULL graphs, BF16 KV and FP4 index cache. Public deterministic text is tokenized
once; the exact 64K, 256K and 512K token arrays are reused in every phase.
Each context gets two warmups and six 4096-output-token measurements with
balanced context order. Prefix caching is enabled. The client reads actual
MTP distribution deltas before and after each request, alongside raw SSE
arrival times, exact usage and output text hashes.

`current37` changes only `triton` and `triton-kernels` from the current image
to the official 3.7 build. All other installed versions must remain unchanged;
ATOM source SHA, AITER source identity and all versions are archived. The two
3.8 phases use fresh containers and repeat the same request order to bound
within-job variability. Every phase starts only after two samples verify the
selected GPUs are free; no unrelated process or cache is removed.

After the unprofiled measurements, a separate 128-output-token request at
256K is traced using the existing profiler endpoints on all four TP ranks.
Profiled timings are excluded from performance estimates. No engine code is
modified. Inspect per-kernel timing, GPU gaps, source identity and acceptance
step counts before attributing a difference to the compiler or one kernel.
Passing this diagnostic does not establish a bound for the full SA workload.
