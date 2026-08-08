# Triton fp8 MLA decode kernel — handoff (May 12, 2026)

## Why this exists

The precompiled AITER asm MLA decode kernels in `rocm/atom:rocm7.2.1-ubuntu24.04-pytorch2.9.1-atom0.1.2`
only support fp8 with `qo_len ∈ {1, 2, 4}`. Hard C++ assert in
`/app/aiter-test/csrc/py_itfs_cu/asm_mla.cu:304`:

```cpp
}else if (q_type == "fp8"){
    ...
    }else if (max_seqlen_q > 4){
        AITER_CHECK(false, __func__, ":only support fp8 mla decoding for qo_len <= 4");
    }
}
```

The corresponding precompiled .co kernel binaries in `/app/aiter-test/hsa/gfx950/mla/`
top out at `qseqlen4` for fp8 (`mla_a8w8_qh16_qseqlen{1,2,4}_gqaratio16.co`).
There is **no asm source** in the tree for higher qSeqLens — only the binaries.

This blocks MTP spec ≥ 4 in fp8, which is the only viable path to higher tok/s/GPU
at conc=4 on this build (bf16 spec=7 was empirically validated and gave 553
tok/s/GPU — worse than fp8 spec=3 at 736 due to KV bandwidth cost).

## What's in this directory

- `triton_mla_fp8_multi.py` — Triton implementation of MLA stage1 decode for
  fp8 + arbitrary qo_len up to 8. Replaces `aiter.mla_decode_stage1_asm_fwd`.
- `aiter_mla_triton.py` — patched `aiter/mla.py` that routes fp8 + qo_len > 4
  to the Triton kernel. Mountable into the container.
- `test_triton_mla.py` — standalone correctness test comparing Triton output
  to the asm output at qo_len=4 baseline.

## Status

- ✅ Kernel compiles and runs without crash at qo_len=4 and qo_len=8
- ✅ Output is finite (no NaN/inf)
- ❌ Output diverges from asm baseline at qo_len=4 — `max_abs_diff=1.0`,
  `max_rel_diff~5800` on synthetic random inputs
- ❌ Not tested in end-to-end ATOM bench

LSE differs by ~ln(15) (asm=4.12 vs triton=1.41 on the synthetic test).
Likely causes (untested hypotheses):

1. **fp8 dtype semantics on AMD**: asm kernel may interpret fp8 bits as `e4m3fnuz`
   (AMD-native) while torch stores as `e4m3fn` (and Triton's `tl.float8e4nv` is
   `e4m3fn`). If the asm decodes with fnuz semantics, my torch+Triton dequant
   produces different float values.
2. **q_scale / kv_scale convention**: I treat them as scalar fp32 multipliers
   on the dequantized value. If asm applies them differently (e.g., as bit-shift
   on exponent), the magnitudes would diverge.
3. **Causal mask**: I apply causal mask among qo_len tokens
   (`KV[j]` valid for `Q[m]` iff `j <= seq_len - qo_len + m`). The asm kernel
   for qSeqLen=4 may not apply causal mask (`causal=0` in `mla_asm.csv`), or it
   may handle it via the kernel's internal qh64 expansion. Need to verify.
4. **kv_per_split assignment**: my Triton computes `split_start = cdiv(seq_len, NUM_KV_SPLITS) * split_id`. Asm uses `num_kv_splits_indptr` (passed externally) — may use a non-uniform split assignment.

## To make this work

Estimated 2-3 more hours of focused debugging:

1. **Validate fp8 dtype**: write a tiny test that runs `gemm_a8w8` (existing
   AITER fp8 GEMM) on a known input and compares against torch reference.
   That tells us the fp8 decode convention.
2. **Verify scale semantics**: same approach with `mla_decode_stage1_asm_fwd`
   at qo_len=1 and walk through outputs by hand.
3. **Fix the divergence**: once root cause identified, fix Triton kernel.
4. **Validate at qo_len=4**: max_abs_diff should be ≤ ~1e-2 vs asm.
5. **Validate at qo_len=8** (no asm reference) by running ATOM end-to-end
   and checking GSM8K ≥ 0.93.

## Performance projection (if it works)

Even with a correct Triton kernel, projected throughput at spec=7:
- ASM step time at spec=3 (qo_len=4): ~5.4ms (gives 736 tok/s/GPU at conc=4)
- Triton step time at spec=7 (qo_len=8): expected 1.5-2× slower than ASM per
  operation (Triton overhead vs hand-tuned ASM)
- Throughput projection: 736 × (8/4) × (1/1.5) = **~980 tok/s/GPU**
- Best case (Triton parity with ASM): 736 × 2 = **1472 tok/s/GPU**
- Target: **1500 tok/s/GPU** — within reach IF Triton matches ASM and
  acceptance rate stays at ~67% at spec=7

This is the ONLY remaining path to ≥1000 tok/s/GPU on this build short of
getting an AITER rebuild with proper fp8 qSeqLen>4 kernels from AMD.

## Integration plan (when kernel is correct)

To deploy:
1. Place `triton_mla_fp8_multi.py` and `aiter_mla_triton.py` in `atom_patches/`
2. Mount in `run_dsr1_c4only_patched.sh` like:
   ```
   -v /projects/teamK/supreme-leader/atom_patches/aiter_mla_triton.py:/app/aiter-test/aiter/mla.py:ro
   ```
3. Use launcher `launch_atom_c4_patched_spec5.sh` (or variants with spec=4..7)
4. Set `--kv_cache_dtype fp8` (NOT bf16 — Triton path requires fp8 Q+KV)
5. Set `--num-speculative-tokens 5` (or 6, 7)
6. Run with `run_dsr1_c4only_patched.sh c4_patched_spec5`
