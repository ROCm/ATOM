# Scoped AiTer tuning overrides

`minimax_m3_fp4_reduce_tuned_fmoe.csv` is an unmodified copy of
`aiter/configs/model_configs/minimax_m3_fp4_tuned_fmoe.csv` from
[ROCm/aiter@b1cdc19ebf3a52b101221fee5b24a3ef7753c244](https://github.com/ROCm/aiter/blob/b1cdc19ebf3a52b101221fee5b24a3ef7753c244/aiter/configs/model_configs/minimax_m3_fp4_tuned_fmoe.csv).

This is the parent of [AiTer #5812](https://github.com/ROCm/aiter/commit/7be37e87a56ac63d7c9eda4fcc1425d1ba65e775),
which switched the M3 16K/32K GEMM2 tuning rows to the scatter epilog.
[ATOM run 37867633847](https://github.com/ROCm/ATOM/actions/runs/37867633847/job/113618310487)
failed in `moe_scatter_reduce::scatter_reduce_kernel_impl<6144, 5, 8, true>`
during 32768-token warmup on both P and D.

The M3 C40 AgentX case sets `AITER_CONFIG_FMOE` to this table; C48 and the C40
GSM8K case inherit it. The override keeps the requested 32768-token budget,
TP4/TP4 placement, and concurrency. Both launcher backends mount the repository
at `/workspace/ATOM`, including when serving with `atom_source=image`.

The table retains all 32 upstream rows (TP2/TP4, token buckets 1 through 32768).
An explicit `AITER_CONFIG_FMOE` replaces the default merged tuning tables,
so a file with only the large-token rows would discard decode tuning.
For TP4 at 16K/32K, GEMM2 uses
`flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64`, paired with
the original GEMM1 and `block_m=64`. Do not change the GEMM2 name without
also preserving its sorting and GEMM1 layout.

This is a targeted workaround, not a proven repair of the invalid address.
Validate startup, GSM8K and AgentX on MI355X before accepting it. Rebaseline
performance because the GEMM kernels changed. Remove the override after an
AiTer scatter fix passes the same workload. See the
[investigation](../../../docs/analysis/minimax_m3_pd_c40_startup_failure_20261012_zh.md).
