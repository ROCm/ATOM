# Qwen3.5-35B-A3B fused ("mono") decode for gfx1250

One persistent FlyDSL kernel per decoder layer for Qwen3.5-35B-A3B-MXFP4 decode at TP4 on
gfx1250: GDN layers (in_proj + conv, the gated delta-rule recurrence, gated norm + out_proj) and
full-attention layers (qkvg, q/k norm + RoPE, paged attention, gated o_proj), each followed by the
MoE block (router, shared expert, 256 MXFP4 experts) and the all-reduces between ranks, with no
kernel launches inside a layer.

- `kernels.py`: decode, 1-4 tokens a step (one token a request).
- `kernels_mtp.py`: MTP verify steps, R requests x 4 tokens (anchor + 3 drafts), up to 16
  requests (64 rows), in groups of 16 tokens.
- `runner.py`: binds ATOM's weights, caches and metadata to the kernels; `dispatch.py` decides per
  forward whether a step takes the mono path; `layout.py` / `config.py`: buffers and shapes.

Enable with `ATOM_QWEN35_MONO=1`. MTP verify steps also need `ATOM_ENABLE_REPLAYSSM=0` (the kernels
write the per-draft state slots). Requirements (else ATOM's own path runs): TP4, bf16 KV, block 16,
`max_model_len <= 16384`, no prefill in the step. FlyDSL 0.4.0 (`/opt/flydsl-0.4.0` first on
`PYTHONPATH` in the gfx1250 images). `ATOM_QWEN35_MONO_CHECK=1` runs every layer both ways and logs
the difference (eager only); `ATOM_QWEN35_MONO_TIMING=1` logs a per-stage timeline.

Measured (image rocm/fw-bringup-mlperf:gfx1250-atom-20261026-dsv41-qwen-e15-08, 8k/1k, MTP3,
acceptance length 3.4), TPOT ms, native ATOM / mono: c1 1.25 / 0.84, c4 1.42 / 1.01, c8 1.70 /
1.38, c16 2.12 / 2.02. Verify forward at c16: 40 layers in ~3.7 ms.

## TP1, MTP verify on ReplaySSM (`*_tp1.py`)

The whole model on one GPU (TP1; 4 replicas for 4xTP1), MTP verify steps of R requests x S tokens
(S = 5: the anchor + 4 drafts), up to 32 tokens (two groups of 16), GDN state on ReplaySSM (the
checkpoint + committed records: each wave rebuilds its slice, runs the window's tokens, writes their
u records; k / g records once every CTA is done; a flush when the records would overflow).

- `kernels_tp1.py`: the GDN and attention layer kernels, 256 CTAs; projections stream 16-row tiles,
  out_proj / o_proj hold the whole K in registers with B straight from the L2-resident input rows;
  attention splits each request's context over CTAs in proportion to its length (any length).
- `runner_tp1.py` / `layout_tp1.py` / `config_tp1.py`: weights, caches, metadata, buffers, shapes.

Enable with `ATOM_QWEN35_MONO=1` on a `-tp 1` server with MTP 4 (`--num-speculative-tokens 4`) and
ReplaySSM on (the default with MTP). Also needs bf16 KV, block 16, no prefill in the step; the
context bound is `ATOM_QWEN35_MONO_MAX_CTX` (default 262144). `ATOM_QWEN35_MONO_CHECK=1` (eager)
runs every layer both ways from ATOM's state (`ATOM_QWEN35_MONO_CHECK_ISOLATED=1`, the default) and
logs per-layer errors and the cache / ReplaySSM state written: GDN residual <= 5.4e-4, attention
3.2e-3 relative; the routed MoE differs from ATOM's by ~15-20%, which is ATOM's FlyDSL MoE's own
deviation from the checkpoint's math (the mono MoE is within 3.5% of it).

Measured on gfx1250 (ATOM 5948fef20, aiter a0df93d2), decode steps at batch 1:
- ~78k-token context, one request: step 4.51 ms vs 5.01 ms native.
- MLPerf Endpoints c=4 (one 4xTP1 replica) trace: step 3.91 ms vs 4.40 ms native (two streams) /
  4.74 ms (one stream).
- MLPerf Endpoints c=4, 10-minute points: TPOT median 1.47 ms vs 1.39 ms native, TTFT median 141
  vs 131 ms; the end-to-end gap despite the faster step is still being investigated.
