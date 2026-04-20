#!/usr/bin/env python3
"""Compare ATOM native vision encoder vs HuggingFace reference."""

import torch
import numpy as np

def test_rotary_embeddings():
    """Compare rotary embedding computation between ATOM and HF."""
    from atom.models.qwen3_5_vl import Qwen3VisionTransformer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeVisionRotaryEmbedding,
    )

    # Simulate vision config
    head_dim = 64  # typical for hidden_size=1152, num_heads=18
    spatial_merge_size = 2

    # HF reference
    hf_rotary = Qwen3_5MoeVisionRotaryEmbedding(head_dim // 2)  # dim = 32

    # Test grid
    grid_thw = [[1, 28, 28]]  # single image, 28x28 patches
    grid_thw_tensor = torch.tensor(grid_thw)

    # HF computation
    merge_size = spatial_merge_size
    max_hw = max(max(h, w) for _, h, w in grid_thw)
    freq_table = hf_rotary(max_hw)  # (max_hw, dim // 2) = (28, 16)

    # Build pos_ids HF style
    t, h, w = grid_thw[0]
    merged_h, merged_w = h // merge_size, w // merge_size
    block_rows = torch.arange(merged_h)
    block_cols = torch.arange(merged_w)
    intra_row = torch.arange(merge_size)
    intra_col = torch.arange(merge_size)
    row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
    col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
    row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
    col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
    hf_pos_ids = torch.stack((row_idx, col_idx), dim=-1)

    hf_embeddings = freq_table[hf_pos_ids]  # (seq, 2, dim//2)
    hf_embeddings = hf_embeddings.flatten(1)  # (seq, dim)
    hf_emb = torch.cat((hf_embeddings, hf_embeddings), dim=-1)  # (seq, 2*dim) = (seq, head_dim)
    hf_cos = hf_emb.cos()
    hf_sin = hf_emb.sin()

    # ATOM computation — using rot_pos_ids
    atom_pos_ids = Qwen3VisionTransformer.rot_pos_ids(h, w, spatial_merge_size)

    # Check pos_ids match
    print(f"HF pos_ids shape: {hf_pos_ids.shape}, ATOM pos_ids shape: {atom_pos_ids.shape}")
    pos_ids_match = torch.equal(hf_pos_ids, atom_pos_ids.long())
    print(f"Position IDs match: {pos_ids_match}")

    if not pos_ids_match:
        diff_mask = (hf_pos_ids != atom_pos_ids.long())
        diff_indices = diff_mask.any(dim=-1).nonzero().flatten()
        print(f"First 5 mismatches at indices: {diff_indices[:5]}")
        for i in diff_indices[:5]:
            print(f"  idx {i}: HF={hf_pos_ids[i].tolist()}, ATOM={atom_pos_ids[i].tolist()}")

    # ATOM rotary computation
    rotary_dim = head_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    positions = torch.arange(max_hw, dtype=torch.float32)
    atom_freq_table = torch.outer(positions, inv_freq)

    # Check freq table match
    freq_match = torch.allclose(freq_table, atom_freq_table, atol=1e-6)
    print(f"Frequency table match: {freq_match}")
    if not freq_match:
        print(f"  HF freq shape: {freq_table.shape}, ATOM freq shape: {atom_freq_table.shape}")
        print(f"  Max diff: {(freq_table - atom_freq_table).abs().max():.6e}")

    # Final cos/sin
    atom_embeddings = atom_freq_table[atom_pos_ids.long()]
    atom_embeddings = atom_embeddings.flatten(1)
    atom_emb = torch.cat((atom_embeddings, atom_embeddings), dim=-1)
    atom_cos = atom_emb.cos()
    atom_sin = atom_emb.sin()

    cos_match = torch.allclose(hf_cos, atom_cos, atol=1e-5)
    sin_match = torch.allclose(hf_sin, atom_sin, atol=1e-5)
    print(f"Cos match: {cos_match}, Sin match: {sin_match}")
    if cos_match and sin_match:
        print("✓ Rotary embeddings MATCH!")
    else:
        print(f"✗ Max cos diff: {(hf_cos - atom_cos).abs().max():.6e}")
        print(f"✗ Max sin diff: {(hf_sin - atom_sin).abs().max():.6e}")

    return cos_match and sin_match


def test_rotary_application():
    """Test that rotate_half style matches."""
    head_dim = 64
    seq_len = 100
    num_heads = 18

    # Create random inputs
    x = torch.randn(seq_len, num_heads, head_dim)
    cos = torch.randn(seq_len, head_dim)
    sin = torch.randn(seq_len, head_dim)

    # HF style: rotate_half
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def hf_apply(x, cos, sin):
        orig_dtype = x.dtype
        x = x.float()
        cos = cos.unsqueeze(-2).float()
        sin = sin.unsqueeze(-2).float()
        out = (x * cos) + (rotate_half(x) * sin)
        return out.to(orig_dtype)

    # ATOM style
    from atom.models.qwen3_5_vl import Qwen3VisionAttention
    atom_out = Qwen3VisionAttention._apply_rotary_emb(x, cos, sin)
    hf_out = hf_apply(x, cos, sin)

    match = torch.allclose(atom_out, hf_out, atol=1e-5)
    print(f"\nRotary application match: {match}")
    if not match:
        print(f"Max diff: {(atom_out - hf_out).abs().max():.6e}")
    else:
        print("✓ Rotary application MATCHES!")

    return match


def test_position_embeddings():
    """Compare position embedding interpolation."""
    from atom.models.qwen3_5_vl import Qwen3VisionTransformer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeVisionModel,
    )
    from transformers import AutoConfig

    # Load config
    config = AutoConfig.from_pretrained("/data/models/Qwen3.5-35B-A3B-FP8")
    vision_config = config.vision_config

    # Create both models (CPU, no weights needed for pos_embed test)
    # HF model
    hf_model = Qwen3_5MoeVisionModel(vision_config)
    hf_model.eval()

    # ATOM model
    atom_model = Qwen3VisionTransformer(vision_config)
    atom_model.eval()

    # Copy pos_embed weights
    atom_model.pos_embed.weight.data.copy_(hf_model.pos_embed.weight.data)

    # Test grid
    grid_thw = torch.tensor([[1, 28, 28]])

    # HF pos embed (on CPU)
    with torch.no_grad():
        hf_pos = hf_model.fast_pos_embed_interpolate(grid_thw)

    # ATOM pos embed (need to set device property)
    with torch.no_grad():
        atom_pos = atom_model._compute_pos_embed(grid_thw.tolist())

    print(f"\nHF pos embed shape: {hf_pos.shape}, ATOM pos embed shape: {atom_pos.shape}")
    if hf_pos.shape == atom_pos.shape:
        match = torch.allclose(hf_pos.float(), atom_pos.float(), atol=1e-4)
        print(f"Position embeddings match: {match}")
        if not match:
            print(f"Max diff: {(hf_pos.float() - atom_pos.float()).abs().max():.6e}")
            # Find first mismatch location
            diff = (hf_pos.float() - atom_pos.float()).abs()
            max_idx = diff.argmax()
            row = max_idx // diff.shape[-1]
            col = max_idx % diff.shape[-1]
            print(f"Max diff at [{row}, {col}]: HF={hf_pos[row, col]:.6f}, ATOM={atom_pos[row, col]:.6f}")
        else:
            print("✓ Position embeddings MATCH!")
    else:
        print(f"✗ Shape mismatch!")

    return hf_pos.shape == atom_pos.shape


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Rotary Embedding Computation")
    print("=" * 60)
    test_rotary_embeddings()

    print("\n" + "=" * 60)
    print("Test 2: Rotary Application")
    print("=" * 60)
    test_rotary_application()

    print("\n" + "=" * 60)
    print("Test 3: Position Embeddings")
    print("=" * 60)
    test_position_embeddings()
