# TP1 width optimization after the 2026-10-16 rebase

The TP1 changes are based on remote `0617ed48b`. They preserve the upstream
GDN convolution BF16 product rounding and BF16 sigmoid-beta rounding, and use
the upstream ATOM-owned GDR emitter. Production TP/width gates and resource
limits remain unchanged. S2/S4 remain experimental, not default-enabled.

Changes include packed multi-token GDN projections, contiguous HC inject
logits, runner-owned HC rows up to four, HC completion summaries, staged
routed G1 inputs, independent S4 tail queue banks, selected G2/recurrence
pointers, and S4 QSA output projection columns. Temporal weight loads apply
only to paired TP1 finish contexts. Expert-owned TP8 G2 retains its original
per-token dispatch and peer publication context.

Validation in the existing `zufa_qwen38_mono` container:

- `pytest tests/test_qwen4_exp_mono.py tests/test_mono_runtime.py -q`:
  135 passed. Scratch layout expectations include the additional 256-byte
  queue bank; finish scratch is 57,680 bytes, layer scratch 144,896 bytes.
- GDN projections compared with remote `0617ed48b`: S2/S4 independent random
  inputs, output and convolution state bitwise equal after one execution and
  1,000 CUDA Graph replays.
- Integrated G2 selected contexts: same, different and mixed expert sets,
  including zero route weights; bitwise equal after one execution and 1,000
  CUDA Graph replays.
- Whole-model TP1, GPU2, BF16 KV, FULL CUDA Graph, fixed 19-token prompt,
  128 output tokens, greedy, prefix caching disabled, five measured rounds:
  S1/S2/S4 Mono builds succeeded; all request token hashes match the historical
  Native reference. SGPR spill ceilings were relaxed for this audit only;
  private scratch and VGPR spills were not permitted.

| Width | Historical Native TPOT (ms) | Integrated Mono TPOT (ms) |
| --- | ---: | ---: |
| S1 | 7.7952 | 6.2627 |
| S2 | 8.2295 | 7.9965 |
| S4 | 9.4991 | 10.1724 |

The Native column comes from an earlier same-GPU run; it is not a paired
post-rebase measurement. These figures are direct-engine fixed-batch TPOT,
not HTTP concurrency measurements. The first integration attempt failed
FlyDSL compilation and fell back to Native; its timings are excluded.
The successful rerun followed a fix moving cache-policy lookup outside a
dynamic G2 branch. Full accuracy evaluation, diverse-input whole-model
validation, TP8 GPU revalidation and independent A/B/A/B performance
acceptance remain outstanding. No 1.21x S2/S4 speedup is claimed.
