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

## Running

```bash
python -m pytest -q tests/models/deepseek_v41
ATOM_DSV41_REFERENCE=/path/to/DeepSeek-V4.1-Flash \
  python -m pytest -q tests/models/deepseek_v41
```

Without a reference path, the local-checkpoint and official-model tests skip.
The checkpoint check validates all 48 headers, the complete 96,085-tensor index,
contiguous offsets, exact file sizes, readable final pages, and available
download revision metadata; it does not checksum the 475 GiB tensor payload.

## What is covered here

`test_math.py` compares Single-Pass mHC, the router, weighted SwiGLU and the
Engram residual against the pinned upstream methods, including native GPU W4A8
experts, FP8 Engram projection, and actual layer-1 table rows and projection
weights. Official tokenizer hashes are checked across chunks, image boundaries
and accepted prefix lengths, and prefetch and fallback row IDs both match that
history oracle. Request snapshot identity, ragged staging, padding and
cancellation live in `tests/model_ops/engram/test_host_path.py`. This is module-level
validation, not model accuracy.

`test_indexer.py` covers compact candidate blocks, candidate-only Reindex,
causal visibility, short and empty prefixes, and the smaller-position score-tie
rule on both CPU and ROCm. Selection is stable across key tile boundaries,
higher scores always win, the newest visible block is retained, and returned
position IDs stay ascending. NaN scores are ignored for candidate ranking and
positive infinity is capped below the newest block's exclusive infinity pin;
the compacted context must remain inside its allocation even for nonfinite
scores. Candidate-table tests also bypass selection to cover a missing newest
block, partially filled candidate lists, empty rows, and changed visibility
during graph replay. The last kept block contributes at most one block of
visible rows. Because the pinned upstream top-k defines no
deterministic position tie rule, exact ties are asserted against explicit
position-based expectations in addition to dense reference checks on untied
scores. Indexer checks alone establish neither model accuracy nor throughput.

## Manual GPU gates

`validate_dspark_lifecycle.py` here, and `validate_runtime.py` under
`tests/attentions/deepseek_v41/`, are `torchrun` entry points rather than pytest
modules — they need four GPUs and the real checkpoint, so pytest never collects
them. Their invocations are in
[the DSpark guide](../../../docs/deepseek_v41_dspark.md#validation) and
[the runtime guide](../../../docs/deepseek_v41_runtime.md#scheduler-acceptance).
