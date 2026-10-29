# MiniMax-M3 SP MoE FP4 prequant gate: shipped-aiter drift

Status: known limitation on the rebased branch (`c645d60f` + tests-only patch); no
product code change. Recorded after the GPU check `tests/distributed/check_ulysses_mxfp4.py`
failed on the shipped pairing.

## What drifted

`atom/model_ops/moe.py::Mxfp4MoEMethod._sp_input_can_prequantize` ends with

```python
return config.prequant and not config.run_1stage and config.ksplit <= 1
```

where `config = aiter.fused_moe.get_2stage_cfgs(...)`. In the shipped aiter tree
(`aiter-ship`, content `a8559963`/`9d0a52f2`),
`aiter/fused_moe.py::_make_mxfp4_metadata` now sets

```python
prequant = p1["a_dtype"] == "fp8" and not p1["inline_quant"]
```

i.e. `MoEConfig.prequant` describes **FP8-activation** prequant only. The FP4
activation-prequant capability moved to a runtime metadata transform inside
`fused_moe()` (it replaces the metadata with `prequant=True` for the FP4 a4w4 stage-1
kernel on the Opus auxiliary path), and `fused_moe()` accepts caller
`prequantized_a1` / `prequantized_a1_scale` for `QuantType.per_1x32`.

## Consequence

On the shipped pairing ATOM's predicate is always `False` for MXFP4, so
`gather_sp_input` takes `return sp_moe_gather(x), None`: BF16 gather followed by
`fused_moe`. The output is correct, but the intended row-local-FP4-across-gather
optimisation is not used. It is not a correctness problem: the fallback path is
byte-identical to the gather+quant reference, verified on 4 ranks (gfx950) by
`tests/distributed/check_ulysses_mxfp4.py` at tokens 16/8192/32768. It also means the
`gather_sp_input` prequant branch and the `sp_topk` prequantized-routing path are not
exercised on GPU with the shipped tree.

## Evidence

- `tests/distributed/check_ulysses_mxfp4.py` (4 ranks, gfx950): BF16-gather fallback
  PASS at tokens 16/8192/32768; the prequant branch is skipped because
  `_sp_input_can_prequantize` is false, and that is reported, not silently passed.
- CPU pins: `tests/model_ops/test_sp_moe_dispatch.py` covers both the prequant path
  (mocked eligibility) and the unsupported-prequantization fallback.
- PR #2643's verification-scope note records this as a coverage gap (the
  model-level e2e that would have exercised the in-engine path needs the
  MiniMax-M3-MXFP4 checkpoint, which was not available on the verification node).

## Follow-up (separate item, not this merge)

Either aiter exposes a public "FP4 prequant capability" predicate (the capability is
currently decided inside `fused_moe`), or ATOM mirrors that runtime transform in
`_sp_input_can_prequantize`; then re-run the GPU check to validate the byte-exact FP4
path and enable the optimisation.
