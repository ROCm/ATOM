"""
TurboQuant Bit-Packing — Pack 2/3-bit indices into dense byte arrays.

Without packing: 128 coords × 1 byte = 128 bytes per vector
With 2-bit packing: 128 coords × 2 bits = 32 bytes per vector (4×)
With 3-bit packing: 128 coords × 3 bits = 48 bytes per vector (2.67×)

Combined with QJL signs (1-bit × 128 = 16 bytes) and norms (2 bytes):

  Keys (2-bit + 1-bit QJL):  32 + 16 + 2 + 2 = 52 bytes/vector
  Values (3-bit):            48 + 2 = 50 bytes/vector
  Total:                     102 bytes/vector

  vs FP16 uncompressed:      128 × 2 × 2 = 512 bytes/vector
  Compression ratio:         512 / 102 = 5.02×
"""

import torch


def pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack uint8 indices (values 0-3) into 2-bit packed bytes.

    Input:  (seq_len, head_dim) uint8, values in [0, 3]
    Output: (seq_len, head_dim // 4) uint8, 4 values per byte

    Bit layout per byte: [val3:2 | val2:2 | val1:2 | val0:2] (LSB first)
    """
    seq_len, d = indices.shape
    assert d % 4 == 0, f"head_dim must be divisible by 4, got {d}"
    idx = indices.to(torch.uint8).view(seq_len, d // 4, 4)
    packed = (
        (idx[:, :, 0])
        | (idx[:, :, 1] << 2)
        | (idx[:, :, 2] << 4)
        | (idx[:, :, 3] << 6)
    )
    return packed  # (seq_len, d // 4) uint8


def unpack_2bit(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    """
    Unpack 2-bit packed bytes back to uint8 indices.

    Input:  (seq_len, head_dim // 4) uint8
    Output: (seq_len, head_dim) uint8, values in [0, 3]
    """
    seq_len = packed.shape[0]
    vals = torch.stack([
        packed & 0x03,
        (packed >> 2) & 0x03,
        (packed >> 4) & 0x03,
        (packed >> 6) & 0x03,
    ], dim=-1)  # (seq_len, d//4, 4)
    return vals.reshape(seq_len, head_dim)


def pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack uint8 indices (values 0-7) into 3-bit packed representation.

    Uses 3 bytes per 8 values (24 bits = 8 × 3 bits).

    Input:  (seq_len, head_dim) uint8, values in [0, 7]
    Output: (seq_len, head_dim * 3 // 8) uint8

    Bit layout per 3-byte group (8 values → 24 bits):
      byte0: [v2:1 | v1:3 | v0:3]     bits [7:0]
      byte1: [v5:0 | v4:3 | v3:3 | v2:2]  bits [15:8]
      byte2: [v7:3 | v6:3 | v5:2]     bits [23:16]

    Simplified: pack as 8 values × 3 bits into 3 uint8 bytes.
    """
    seq_len, d = indices.shape
    assert d % 8 == 0, f"head_dim must be divisible by 8, got {d}"
    idx = indices.to(torch.int32).view(seq_len, d // 8, 8)

    # Pack 8 × 3-bit values into 3 bytes (24 bits)
    # Use a 32-bit intermediate to hold 24 bits
    bits = torch.zeros(seq_len, d // 8, dtype=torch.int32, device=indices.device)
    for i in range(8):
        bits |= (idx[:, :, i] & 0x07) << (i * 3)

    # Extract 3 bytes from the 24-bit value
    byte0 = (bits & 0xFF).to(torch.uint8)
    byte1 = ((bits >> 8) & 0xFF).to(torch.uint8)
    byte2 = ((bits >> 16) & 0xFF).to(torch.uint8)

    packed = torch.stack([byte0, byte1, byte2], dim=-1)  # (seq_len, d//8, 3)
    return packed.reshape(seq_len, d * 3 // 8)


def unpack_3bit(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    """
    Unpack 3-bit packed bytes back to uint8 indices.

    Input:  (seq_len, head_dim * 3 // 8) uint8
    Output: (seq_len, head_dim) uint8, values in [0, 7]
    """
    seq_len = packed.shape[0]
    n_groups = head_dim // 8
    packed = packed.reshape(seq_len, n_groups, 3)

    # Reconstruct 24-bit values from 3 bytes
    bits = (
        packed[:, :, 0].to(torch.int32)
        | (packed[:, :, 1].to(torch.int32) << 8)
        | (packed[:, :, 2].to(torch.int32) << 16)
    )

    # Extract 8 × 3-bit values
    vals = torch.stack(
        [(bits >> (i * 3)) & 0x07 for i in range(8)], dim=-1
    ).to(torch.uint8)

    return vals.reshape(seq_len, head_dim)


def pack_1bit(signs: torch.Tensor) -> torch.Tensor:
    """
    Pack int8 signs (-1/+1) into 1-bit packed bytes.

    Input:  (seq_len, head_dim) int8, values in {-1, 1}
    Output: (seq_len, head_dim // 8) uint8, 8 signs per byte

    Convention: sign=1 → bit=1, sign=-1 → bit=0
    """
    seq_len, d = signs.shape
    assert d % 8 == 0
    # Convert -1/+1 to 0/1
    bits = ((signs.to(torch.int32) + 1) // 2).view(seq_len, d // 8, 8)
    packed = torch.zeros(seq_len, d // 8, dtype=torch.uint8, device=signs.device)
    for i in range(8):
        packed |= (bits[:, :, i].to(torch.uint8) << i)
    return packed


def unpack_1bit(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    """
    Unpack 1-bit packed bytes to int8 signs.

    Input:  (seq_len, head_dim // 8) uint8
    Output: (seq_len, head_dim) int8, values in {-1, 1}
    """
    seq_len = packed.shape[0]
    bits = torch.stack(
        [(packed >> i) & 1 for i in range(8)], dim=-1
    )  # (seq_len, d//8, 8)
    signs = (bits.to(torch.int8) * 2 - 1)  # 0→-1, 1→+1
    return signs.reshape(seq_len, head_dim)


def packed_memory_footprint(seq_len: int, head_dim: int = 128) -> dict:
    """Calculate bit-packed memory footprint."""
    # Keys: 2-bit indices (packed) + 1-bit signs (packed) + norms
    key_idx_bytes = seq_len * head_dim * 2 // 8   # 2-bit packed
    key_sign_bytes = seq_len * head_dim // 8       # 1-bit packed
    key_norm_bytes = seq_len * 2                    # fp16 vec_norms
    key_rnorm_bytes = seq_len * 2                   # fp16 residual_norms
    key_total = key_idx_bytes + key_sign_bytes + key_norm_bytes + key_rnorm_bytes

    # Values: 3-bit indices (packed) + norms
    val_idx_bytes = seq_len * head_dim * 3 // 8    # 3-bit packed
    val_norm_bytes = seq_len * 2                    # fp16 vec_norms
    val_total = val_idx_bytes + val_norm_bytes

    # Uncompressed FP16
    uncomp = seq_len * head_dim * 2 * 2  # K + V in fp16

    return {
        "key_bytes": key_total,
        "val_bytes": val_total,
        "total_bytes": key_total + val_total,
        "uncompressed_bytes": uncomp,
        "compression_ratio": uncomp / (key_total + val_total),
        "key_bytes_per_token": key_total / seq_len,
        "val_bytes_per_token": val_total / seq_len,
        "total_bytes_per_token": (key_total + val_total) / seq_len,
    }
