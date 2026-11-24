# SDDD Training

This guide walks through training Eagle3 speculative decoding draft models using AtomRL's SDDD pipeline.

## Kimi K2.5 Eagle3 (ATOM + Mooncake)

Train an Eagle3 draft model for Kimi K2.5 on 8 GPUs with ATOM inference and Mooncake RDMA transfer.

### GPU Layout

- **GPUs 0-3**: FSDP2 Eagle3 draft model training (BF16)
- **GPUs 4-7**: ATOM — Kimi K2.5 teacher (TP=4, BF16 to MXFP4)

### Quick Start

```bash
# Smoke test
CUDA_VISIBLE_DEVICES=0 python -m atomrl.trainer.main \
  --config examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/configs/smoke_test_hf.yaml

# Full two-phase training (Docker)
bash examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/run_full_training_docker.sh

# Phase 2 only
bash examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/run_full_training_docker.sh --phase2-only
```

### Models

| Role | Model | Notes |
|------|-------|-------|
| **Teacher** | [moonshotai/Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) | 1T MoE, BF16, online MXFP4 via ATOM/AITER |
| **Draft** | Eagle3 (from scratch) | hidden=7168, 1 layer, 64 heads |

## MI350-Specific Examples

AtomRL includes MI350-optimized SDDD configurations:

- `examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/` — Kimi K2.5 on MI350 with ATOM

## Monitoring

```bash
# Watch training logs
tail -f /datasets/checkpoints/kimi_k25_eagle3_phase1/phase1.log
```

## References

- [lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3) — Training recipe
- [KimiV2.5 Technical Report](https://arxiv.org/abs/2505.01759)

For architecture details and configuration options, see {doc}`/advance/sddd`.
