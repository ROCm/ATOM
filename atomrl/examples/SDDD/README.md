# SDDD: speculative decoding draft distillation

Speculative decoding drafts tokens with a small model and verifies them with the large
one in a single forward pass; the speedup depends on how often the draft is right. SDDD
trains the draft on the target's own hidden states: the teacher runs on inference GPUs,
its hidden states stream to the trainer, and an Eagle3 head learns to predict the
teacher's next tokens. See [the SDDD guide](../../atomrl-docs/docs/source/advance/sddd.md).

## Examples

| `run.sh` id | What it runs | Hardware | Status |
|---|---|---|---|
| `sddd-eagle3-smoke-hf` | one-layer Eagle3 head on a Qwen3-8B teacher (HF forward), two steps | 1 GPU, release image | validated |
| `sddd-kimi-k25-eagle3-phase1` | Kimi K2.5 teacher on ATOM (MXFP4, TP=4), FSDP2 draft, Mooncake transfer | 8× MI350X, example image | long run |
| `sddd-kimi-k25-eagle3-phase2` | phase 2 on mixed-domain data, resumed from phase 1 | 8× MI350X, example image | long run |

## Results

The smoke validates the draft model, the loss and the optimizer without vLLM, ATOM or
Mooncake. A freshly initialised draft starts each of the 4 speculative positions near
ln(vocab) = 11.9 nats:

| Metric | Four runs | Accepted range |
|---|---|---|
| Loss at speculative position 0 | 12.2–15.0 | 10–20 |
| Total loss (0.8-decayed sum over 4 positions) | 36.0–44.2 | 29–60 |

## Run

```bash
bash examples/run.sh prepare sddd-eagle3-smoke-hf      # Qwen3-8B
bash examples/run.sh sddd-eagle3-smoke-hf              # about 1 min
```

The Kimi K2.5 recipe, its image, data and benchmarks are in
[Kimi_K25_SDDD_MI350_ATOM/](Kimi_K25_SDDD_MI350_ATOM/README.md).
