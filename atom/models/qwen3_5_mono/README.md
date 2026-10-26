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
