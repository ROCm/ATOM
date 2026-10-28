# MiniMax M3.1 Flash TP4 P/D with Mooncake Store and DSpark

The current configuration and CI inputs are documented in
[MiniMax-M3.1-Flash-PD.md](../MiniMax-M3.1-Flash-PD.md).

Use that recipe for the NVFP4 / FP4-index Gluon target, DSpark K7 / AL4.04,
1 TiB Mooncake Store, and C32 / C48 / C64 / C88 benchmarks. Each benchmark runs
3600 seconds of formal profiling after warmup; torch tracing is disabled.
The separate C32 trace case captures 30 seconds after Profiling +1800 seconds.

The former FP8 / DSpark3 recipe and its dense-gather workspace option have been
superseded. The workspace feature is reverted from this integration branch.
