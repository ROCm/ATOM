# Qwen3.5 / 3.6 MTP drafter: fused pass kernels for gfx1250 (off by default)

Persistent FlyDSL kernels for the MTP drafter's pass on TP1 / DP ranks, for passes of at most 16
rows (larger passes take ATOM's own path). 256 CTAs x 8 waves; every loop is a runtime loop (no
unrolled K loops), weights stream through per-wave LDS buffers filled by async copies.

- `ATOM_MTP_MONO=1` (`post.py`): after the attention, one launch runs o_proj + residual, the
  post-attention norm, the router with softmax top-8 (renormalised), the selected experts
  (MXFP4 1x32 weights, MXFP8 1x32 activations), the shared expert, the final norm, an MXFP8 1x32
  LM head and the argmax. Needs `ATOM_MTP_MONO_HEAD=<checkpoint dir>` whose `outside.safetensors`
  holds `lm_head.weight` (F8_E4M3) and `lm_head.weight_scale`; the other weights are read from the
  served checkpoint's `mtp.safetensors` (experts quantized with aiter's `per_1x32_f4_quant`).
- `ATOM_MTP_MONO_PRE=1` (`pre.py`): before the attention, one launch runs the embedding rows, both
  pre-fc norms, fc, the input norm and the q/k/v projection on the drafter's own bf16 weights.
- `ATOM_MTP_MONO_CHECK=1`: eager passes run ATOM's path too (it serves) and log how the kernel's
  hidden rows and token ids compare.

The attention, its q/k norm + rotary + cache write, the segment reduce and the draft-state kernels
stay ATOM's. `host.py` binds weights and buffers; `device.py` holds the gfx1250 helpers (as the
DeepSeek-V4.1 step kernel's); `head8.py` is the MXFP8 head with the argmax in its epilogue.

Measured on MI455X, Qwen3.6-35B-A3B MXFP4, DP4, MLPerf Endpoints agentic c64 (3 requests a rank in
the trace): the post-attention kernel takes 83-89 us against ~95 us for the ten kernels it replaces
(the gain is the MXFP8 head: ~35 us against 67 us for the bf16 head + argmax); the pre-attention
kernel takes ~16.5 us against ~13 us for the four it replaces. Draft acceptance is unchanged
(66.1% vs 66.2% at MTP-3, 59.3% vs 59.1% at MTP-4, 10-minute runs). End to end the difference is
within the ~2% run-to-run spread of 10-minute runs, hence off by default.
