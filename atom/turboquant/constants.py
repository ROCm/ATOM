"""TurboQuant constants — block sizes and architecture parameters."""

HEAD_DIM = 128              # Attention head dimension
BLOCK_Q = 16                # Query block size for tiled attention
BLOCK_KV = 64               # Key/Value block size for tiled processing
BLOCK_S = 64                # Sequence block size for compression kernels
SUPPORTED_BITS = {1, 2, 3, 4}
DEFAULT_TOTAL_BITS = 3      # 2-bit keys + 1-bit QJL = 3 total
DEFAULT_SEED = 42
