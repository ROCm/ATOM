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

The harness cache regression and paired-statistics tests use optional evaluation
packages. The recorded complete run uses lm-eval 0.4.11, sqlitedict 2.1.0 and
SciPy 1.16.3. Tests needing those packages skip when unavailable; pure cache
identity checks still run. Install the evaluation packages for the full suite:

```bash
python -m pip install 'lm-eval==0.4.11' 'scipy==1.16.3'
python -m pytest -q tests/models/deepseek_v41
ATOM_DSV41_REFERENCE=/path/to/DeepSeek-V4.1-Flash \
  python -m pytest -q tests/models/deepseek_v41
```

With evaluation dependencies installed but no reference path, the
local-checkpoint and official-model tests skip. The checkpoint check validates
all 48 headers, the complete 96085-tensor index, contiguous offsets, exact file
sizes, readable final pages, and available
download revision metadata. It does not checksum the 475 GiB tensor payload.
Use `lm_eval_checkpoint.py` for paired real-checkpoint quality evaluation;
commands and results are in `docs/deepseek_v41_validation.md`.

P03 compares Single-Pass mHC, router, weighted SwiGLU, and Engram residual math
with the pinned upstream methods. Tests cover native GPU W4A8 experts and FP8
Engram projection, actual layer-1 table rows/projection weights, and official
tokenizer hashes across chunks, image boundaries and accepted prefix lengths.
Prefetch and fallback row IDs both match that official history oracle. Request
snapshot identity, ragged staging, padding, and cancellation are covered in
`tests/model_ops/test_engram.py`. This remains module-level validation.

P04 index selection tests live in `test_indexer.py`. They cover compact candidate
blocks, candidate-only Reindex, causal visibility, short/empty prefixes, and both
`small_position` and `large_position` score-tie policies on CPU and ROCm.
Selection is stable across key tile boundaries; higher scores always win, the
newest visible block is retained, and returned position IDs remain ascending.
The pinned upstream top-k does not define a deterministic position tie rule, so
exact ties use explicit position-based expectations in addition to dense
reference checks on untied scores. `test_indexer_long.py` also exercises the
published 512-token / 2,048-block limits with 32 heads, D=128 and 32,771 keys.
Indexer checks alone do not establish model accuracy or throughput.
