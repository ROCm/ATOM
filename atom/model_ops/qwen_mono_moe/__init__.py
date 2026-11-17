"""ATOM_QWEN_MONO_MOE: the TP4 MoE half of Qwen3.5-35B-A3B MXFP4 decode in one FlyDSL kernel on gfx1250 (router,
top-k, MXFP4 experts, shared expert, gated add, 4-rank all-reduce, residual add and the next RMSNorm). See
integration.py."""
