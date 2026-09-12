# DeepSeek-V4.1 numerical reference

The test reference is pinned to `deepseek-ai/DeepSeek-V4.1-Flash` revision
`dba1be0a40aa45a94ad051997016db3960a90277`. `fixtures/reference_manifest.json`
records source SHA256 values; the configuration and token fixtures come from
that snapshot. No model weights or tokenizer payloads are vendored.

`reference.load_reference` executes the unmodified official model math in a
private module namespace. Only its local imports are redirected, replacing
TileLang kernels with the small PyTorch definitions in `oracle_kernels.py`.
This allows module comparisons on the existing ROCm PyTorch environment.
It does not establish TileLang/GPU bitwise parity or model-level accuracy.

The oracles preserve FP8 activation QAT, E2M1 packing and tie-to-even rounding,
E4M3 versus E8M0 scale rules, 20-iteration Sinkhorn, and the attention kernel's
64-row online softmax with BF16 probability rounding. They deliberately
materialize/dequantize small tensors and must not enter a serving path.

```bash
python -m pytest -q tests/models/deepseek_v41
ATOM_DSV41_REFERENCE=/path/to/DeepSeek-V4.1-Flash \
  python -m pytest -q tests/models/deepseek_v41
```

Without the reference path, only the local-checkpoint and official-model tests
skip. The checkpoint check validates all 48 headers, the complete 96085-tensor
index, contiguous offsets, exact file sizes, readable final pages, and available
download revision metadata. It does not checksum the 475 GiB tensor payload.
Use `lm_eval` for model accuracy once the ATOM model is executable.


P03 compares Single-Pass mHC, router, weighted SwiGLU, and Engram residual math
with the pinned upstream methods. Tests cover native GPU W4A8 experts and FP8
Engram projection, actual layer-1 table rows/projection weights, and official
tokenizer hashes across chunks, image boundaries and accepted prefix lengths.
Prefetch and fallback row IDs both match that official history oracle. Request
snapshot identity, ragged staging, padding, and cancellation are covered in
`tests/model_ops/test_engram.py`. This remains module-level validation.
