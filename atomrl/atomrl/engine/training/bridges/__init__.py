"""Megatron <-> Hugging Face weight bridges, one module per model family.

``core`` holds the streaming rule converter and the primitives every family
shares; ``gpt`` is the standard GPTModel layout (dense and MoE, validated on
Qwen3), and ``dsv3`` / ``dsv4`` hold what those families change: dims, extra
export rules and the HF -> Megatron load.
"""
