# DeepSeek V4.1 A8W4 attention

`ATOM_DSV41_A8W4_ATTN=auto` enables direct attention over the native NVFP4 CSA
and MXFP8 SWA cache planes when AITER provides a supported backend. Set it to
`off` to keep the existing gather/dequantize-to-BF16 adapter. This setting does
not change the cache allocation, stored bytes, indexer format, or SWA quantizer.

| Backend | Device | Local query heads | Head dimension |
| --- | --- | --- | --- |
| OPUS | gfx950 | AITER's small/large head paths | 512 |
| ASM | gfx1250 B0+ | 64 | 512 |

The required AITER interfaces are the OPUS A8W4 addition in
[ROCm/aiter#7124](https://github.com/ROCm/aiter/pull/7124) and the ASM addition in
[ROCm/aiter#7407](https://github.com/ROCm/aiter/pull/7407). Older AITER builds keep
the BF16 adapter. The public V4.1 function alone is not sufficient: older versions
of it only support A8W8, so ATOM also checks each backend's A8W4 symbol.

The backend is chosen before Q preparation. ASM additionally requires at most
65536 query rows and the FP32 value of `1/sqrt(512)` as the softmax scale. An
unsupported shape uses the original BF16 Q path; ATOM does not catch a failed
attention launch and retry after producing Q8.

## Inputs and cache lifetime

Both backends receive the same arguments:

- Q: MXFP8 `[N,H,512]` and group32 E8M0 scales `[N,H,16]`, emitted by the
  existing RoPE/quantization producer.
- CSA: contiguous NVFP4 data `[Pc,256]` and group16 E4M3 scales `[Pc,32]`.
- SWA: contiguous MXFP8 data `[Ps,512]` and group32 E8M0 scales `[Ps,16]`.
- Two int32 CSR lists, each containing row IDs into its own source.
- A caller-owned BF16 output buffer. The existing inverse RoPE, output
  quantization, and grouped output projection remain after attention.

Data and scale views for a source share its allocation. CSA rows are addressed
directly through the page table; no selected-row BF16 scratch is needed.

Decode reads the SWA ring directly. Prefill stages only the current batch's
rings, followed by space for current KV, using the same `prefill_window` helper
as A8W8. The producer writes current KV directly into the tail views. Historical
rows use `batch * ring_slots + position % ring_slots`; current rows use
`batch_capacity * ring_slots + token_index`. Persistent ring writes remain after
prefill attention, so the chunk cannot overwrite history it still needs.

The CSR builder uses its existing split-source mode to emit untagged CSA/SWA
row IDs. A8W4 indptrs are separate from the BF16 adapter's indptrs and owned by
the step. Group results distinguish the two representations; bounded queries
rebuild their own indptrs after masking non-query rows. This permits different
query shapes in a step to select different attention backends without reusing
an incompatible list.

Native attention and the vLLM attention override share this preparation and
dispatch behavior. Mono continues to use its own kernels; it does not dispatch
through this wrapper.

## Splits and verification

AITER retains ownership of automatic split selection and partial workspaces.
The OPUS and ASM limits and workspace layouts differ, so the wrapper forwards
`num_kv_splits` without sharing a workspace ABI between them. With automatic
selection, AITER sees the CSR allocation capacity rather than the live nnz.
Performance measurements should include Q preparation, prefill SWA staging,
CSR generation, attention, and output processing.

Relevant tests:

```sh
pytest tests/test_deepseek_v41_a8w4_attention.py
pytest tests/attentions/deepseek_v41/test_a8w4_attention.py
pytest tests/attentions/deepseek_v41/test_contiguous_indices.py
pytest tests/attentions/deepseek_v41/test_contiguous_cache.py
pytest tests/test_vllm_deepseek_v41_attention_copy.py
```

The host contract test isolates the functions from device-discovery imports.
Device tests compare against attention on the dequantized Q/CSA/SWA values,
exercise explicit and automatic splits, and replay a graph while the number of
live requests changes. Backend availability is checked before running device
cases; skipped ASM cases are not evidence of gfx1250 correctness.
