# Kimi K2.5 Eagle3 draft distillation on 8× MI350X

Trains an Eagle3 speculative-decoding draft for Kimi K2.5 (1T MoE) from the teacher's
own hidden states, on one node of 8× MI350X:

| GPUs | Role |
|---|---|
| 0–3 | AtomRL FSDP2: Eagle3 draft training (one Transformer layer, BF16) |
| 4–7 | Kimi K2.5 teacher, TP=4: ATOM with MXFP4 and AITER gfx950 kernels, or vLLM |

```
Training GPUs (0-3)                    Inference GPUs (4-7)
AtomRL FSDP2                  <---   ATOM / vLLM teacher (TP=4)
  Eagle3 draft model, BF16              AITER gfx950 kernels
       ^                                      |
  Mooncake TCP  <--------------------  hidden states via
  EagleMooncakeStore                  MooncakeHiddenStatesConnector
```

The draft takes three auxiliary hidden states from teacher layers 1, N/2−1 and N−4 and is
trained with a position-decayed forward KL over 4 speculative positions,
L = Σᵢ 0.8ⁱ · KL(teacherᵢ ‖ draftᵢ), on assistant tokens only. Tokenization uses
the Hugging Face tokenizer's own chat template; AtomRL marks assistant turns
for the loss mask.

## Recipe

Two phases after [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3),
on [lightseekorg/kimi-mtp-dataset](https://huggingface.co/datasets/lightseekorg/kimi-mtp-dataset), batch size 8:

| Phase | Config | Teacher | Data | Steps |
|---|---|---|---|---|
| 1 | `phase1_foundation.yaml` | ATOM, MXFP4 | perfectblend, ~296K samples | 37,005 (1 epoch), from scratch |
| 2 | `phase2_mixed.yaml` | vLLM, Kimi K2.5 INT4 | mixed domain (VL, Chinese, tool calls, agents, writing), ~181K samples | 67,826 (3 epochs), from phase 1 |
| 2, ATOM | `phase2_atom.yaml` | ATOM, MXFP4 | the same | 22,609 (1 epoch), from phase 1 |

## Quick start

### 1. Models and data in `/dev/shm`

```bash
huggingface-cli download amd/Kimi-K2.5-MXFP4 --local-dir /dev/shm/Kimi-K2.5-MXFP4    # ATOM teacher
huggingface-cli download moonshotai/Kimi-K2.5 --local-dir /dev/shm/Kimi-K2.5-BF16    # vLLM teacher, phase 2
huggingface-cli download lightseekorg/kimi-mtp-dataset --repo-type dataset --local-dir /dev/shm/kimi-mtp-dataset
```

### 2. Image

```bash
bash examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/docker/build.sh
```

The image bundles vLLM 0.19.1 (patched for `extract_hidden_states`), ATOM, AITER, AtomRL,
Lumen, Mooncake (built for HIP) and MORI (MoE all-to-all).

### 3. Train

```bash
bash examples/run.sh sddd-kimi-k25-eagle3-phase1
bash examples/run.sh sddd-kimi-k25-eagle3-phase2
```

or directly:

```bash
bash examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/run_full_training_docker.sh                 # phase 1
bash examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/run_full_training_docker.sh --phase2-only   # phase 2
```

Checkpoints go to `/dev/shm/checkpoints/kimi_k25_eagle3_v2_phase{1,2}`;
`tools/export_eagle3_hf_*.py` exports them to Hugging Face safetensors.

### 4. Benchmark

```bash
bash examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/run_benchmark_vllm_docker.sh
```

`tools/bench_*.py` measure accept length and SpeedBench throughput on ATOM and vLLM.

## Smoke test

`bash examples/run.sh sddd-eagle3-smoke-hf` trains the draft on a Qwen3-8B teacher through
plain HF forward on one GPU in the release image, without vLLM, ATOM or Mooncake; see
[the SDDD overview](../README.md).

## References

- [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3): training recipe
- [Kimi K2.5 technical report](https://arxiv.org/abs/2505.01759)
