# EAGLE 3 / EAGLE 3.1 Speculative Decoding

ATOM supports the EAGLE-family speculative decoding methods through the
`--method eagle3` CLI flag. This guide covers both the original EAGLE 3 path
(in production since the K2.5 release) and the EAGLE 3.1 extensions added
in May 2026 (`fc_norm`, post-norm hidden state feedback, MLA draft variant).

For the framework that hosts both methods, see
[`serving_benchmarking_guide.md` § 6](serving_benchmarking_guide.md).

---

## 1. EAGLE 3 vs EAGLE 3.1

EAGLE 3 is a self-distilled drafter that runs one transformer decoder layer
per speculative step over the concatenation of an embedding tensor and a
target-side hidden state. The drafter shares the target's tokenizer and
emits its own auxiliary projection (`fc`) over a few selected target hidden
layers (`eagle_aux_hidden_state_layer_ids`).

EAGLE 3.1 keeps the same `--method eagle3` code path and adds two
backward-compatible toggles in the drafter HF config. They are detected via
`getattr(...)` with the EAGLE 3 default, so loading an existing EAGLE 3
checkpoint (no new fields) takes the original code path byte-for-byte.

| Field on draft `config.json`  | Type / default              | Behavior when set                                                                                                                                                                                              |
|-------------------------------|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `target_hidden_size`          | `int`, default `hidden_size`| Sizes the `fc` input fan-in per aux slot. Lets the draft project from a target whose hidden dim differs from the draft's.                                                                                      |
| `num_aux_hidden_states`       | `int`, default `len(eagle_aux_hidden_state_layer_ids)` (else `3`) | Number of target hidden-state slots concatenated into the `fc` input. Multiplies the `fc` fan-in.                                                              |
| `fc_norm`                     | `bool`, default `False`     | When `True`, each aux slot is RMSNormed (one `RMSNorm` per slot) before concatenation and `fc`. When `False`, the legacy raw-concat path runs.                                                                 |
| `norm_output`                 | `bool`, default `False`     | When `True`, the drafter feeds its **post-norm** hidden state to its own next speculative step instead of the pre-norm residual. The logits path always applies the final norm once.                           |

Together these target *attention drift*: under long contexts or out-of-distribution
prompts, the drafter's attention shifts away from the sink token as speculative
depth grows, and acceptance length collapses. EAGLE 3.1 reports up to **2×
acceptance length on long-context production workloads** and up to **2.03×
per-user output throughput at concurrency 1** on Kimi-K2.6-NVFP4 + vLLM TP=4
(GB200). See the [vLLM blog post](https://vllm.ai/blog/2026-05-26-eagle-3-1)
and the original integration in [vLLM PR #42764](https://github.com/vllm-project/vllm/pull/42764).

---

## 2. Drafter architectures

ATOM ships two EAGLE 3 drafter classes, both routed through
`atom.spec_decode.eagle.support_eagle_model_arch_dict` based on the draft
HF config's `architectures[0]` after `Config.hf_config_override` rewrites it
to the internal name:

| Target stack            | Draft HF arch (in `config.json`)        | Internal class (in `support_eagle_model_arch_dict`) | Source file                              | KV cache                                                                 |
|-------------------------|-----------------------------------------|-----------------------------------------------------|------------------------------------------|--------------------------------------------------------------------------|
| MHA / GQA (Llama-style) | `LlamaForCausalLMEagle3`                | `Eagle3LlamaModel`                                  | `atom/models/eagle3_llama.py`            | Independent draft pool via `Eagle3DraftBuilder`                          |
| MLA (DeepSeek V3 / K2.6)| `Eagle3DeepseekV2ForCausalLM`           | `Eagle3DeepseekMLAModel`                            | `atom/models/eagle3_deepseek_mla.py`     | Piggybacks the target's MLA KV pool at `layer_id = num_hidden_layers`    |

Both classes implement the same set of EAGLE 3.1 toggles. The MLA variant
reuses the MLA attention block from `atom/models/deepseek_v2.py` rather than
re-implementing it. Because its KV shape is identical to the target's, no
parallel `Eagle3MLADraftBuilder` is needed — the draft slots into the target
pool as one additional layer (accounted for by `num_nextn_predict_layers`).

---

## 3. Drafter forward contract

To support post-norm feedback, `forward()` on EAGLE 3.1 drafters returns a
tuple:

```python
hidden_for_logits, hidden_for_next_step = model(input_ids, positions, hidden_states)
```

- `hidden_for_logits`: pre-norm hidden state. The final `norm` + `lm_head`
  in `compute_logits()` applies on top of this, exactly as in EAGLE 3.
- `hidden_for_next_step`: post-norm hidden state when `norm_output=True`,
  otherwise identical to `hidden_for_logits`. This is what the propose loop
  feeds back as `hidden_states` on the next speculative step.

`EagleProposer.propose()` (`atom/spec_decode/eagle.py`) unpacks the tuple via
`isinstance(model_out, tuple)`, so legacy drafters returning a single tensor
continue to work without changes.

---

## 4. Usage

### 4.1 K2.6 + EAGLE 3.1 MLA draft

```bash
python -m atom.entrypoints.openai_server \
    --model /path/to/Kimi-K2.6-MXFP4 \
    --trust-remote-code \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --method eagle3 \
    --num-speculative-tokens 3 \
    --draft-model lightseekorg/kimi-k2.6-eagle3.1-mla
```

The draft `config.json` should expose `fc_norm: true`, `norm_output: true`,
`kv_lora_rank` (signals MLA), and an
`eagle_config.eagle_aux_hidden_state_layer_ids` list. These drive the routing
and toggles automatically — no extra CLI flags.

### 4.2 K2.5 + legacy EAGLE 3 draft (backward-compat)

```bash
python -m atom.entrypoints.openai_server \
    --model /path/to/Kimi-K2.5-MXFP4 \
    --trust-remote-code \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --method eagle3 \
    --num-speculative-tokens 3 \
    --draft-model /path/to/kimi-k2.5-eagle3
```

The legacy draft's `config.json` has none of the new fields, so all `getattr`
toggles fall through to their EAGLE 3 defaults. Behavior is byte-equivalent
to pre-EAGLE-3.1 ATOM.

---

## 5. Runtime acceptance stats

ATOM's scheduler already emits per-window acceptance via the `SpecStats`
class (`atom/model_engine/scheduler.py`). With
`num_speculative_tokens=3`, a `[MTP Stats]` line appears every 1000 decode
steps in the server log:

```
[MTP Stats] acceptance: 59.62%  avg toks/fwd: 2.79  dist 0/1/2/3: 21.69/19.53/16.97/41.80%
```

No instrumentation needed — read these directly from the server's stdout log
(`docker logs <container>` or the `tee` target you launched it under).

---

## 6. Reference: 8× MI355X validation

Measured 2026-05-27 on the `rocm/atom-dev:latest` container, TP=8, MXFP4
targets, gsm8k 5-shot, 1319 samples.

| Setup                                         | gsm8k 5-shot       | Acceptance | Avg toks/fwd | Dist 0/1/2/3              |
|-----------------------------------------------|--------------------|------------|--------------|---------------------------|
| K2.5 MXFP4 + `kimi-k2.5-eagle3` (legacy path) | 0.9356 (±0.0068)   | 67.96%     | 3.04         | 14.95 / 16.30 / 18.68 / 50.07% |
| K2.6 MXFP4 + `kimi-k2.6-eagle3.1-mla`         | 0.9393 (±0.0066)   | 59.62%     | 2.79         | 21.69 / 19.53 / 16.97 / 41.80% |

The K2.5 legacy run is the backward-compat probe — it must match the
pre-EAGLE-3.1 K2.5 baseline. The K2.6 run shows EAGLE 3.1 functional on the
new MLA draft without regressing correctness vs. the legacy path.

The lower acceptance rate on K2.6 vs K2.5 is **not a regression** — the two
spec pairs are not directly comparable (different targets, different draft
training maturity), and gsm8k is a short-context structured-reasoning task
rather than the long-context production workload EAGLE 3.1 was designed for.
A direct K2.6-target ablation requires an additional K2.5-style EAGLE 3
draft sized to K2.6, which is not yet publicly available.

---

## 7. Implementation notes for contributors

- **Adding a new MLA target's EAGLE 3.1 draft**: register an HF arch rewrite
  in `Config.hf_config_override` (`atom/config.py`), add the entry to
  `support_eagle_model_arch_dict` (`atom/spec_decode/eagle.py`), and reuse
  `Eagle3DeepseekMLAModel` if the target's MLA head dims match. Sizing per
  draft is automatic via the four `getattr` toggles in § 1.
- **`atom_config` copy semantics in `EagleProposer.__init__`**: the draft
  config is constructed by shallow-copying `atom_config` and then isolating
  `hf_config` + `compilation_config`. Avoid `copy.deepcopy(atom_config)` —
  models like K2.6 attach `cuda.Stream` objects to the config tree from
  custom HF modeling code, and `Stream` cannot be pickled.
- **`mtp_start_layer_idx` dispatch** in `atom/model_engine/model_runner.py`
  branches on `speculative_config.method == "eagle3"`, not on the presence
  of `eagle3_draft_builder`. MLA drafts piggyback the target pool and so
  have no draft builder, but they still need the layer-index offset.
- **Do not modify `@support_torch_compile`-decorated model files** for
  EAGLE-related changes — instrument at call sites (`EagleProposer.propose`,
  `ModelRunner.run_model`) instead. See the project `CLAUDE.md`.

---

## 8. References

- vLLM integration PR (merged main, ships v0.22.0): https://github.com/vllm-project/vllm/pull/42764
- TorchSpec training PR: https://github.com/lightseekorg/TorchSpec/pull/97
- vLLM blog: https://vllm.ai/blog/2026-05-26-eagle-3-1
- Kimi K2.6 EAGLE 3.1 draft model: https://huggingface.co/lightseekorg/kimi-k2.6-eagle3.1-mla
- Original EAGLE repository: https://github.com/SafeAILab/EAGLE
