# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-safe cache geometry shared by M3 writers and PAGE transfer metadata."""

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128

# Page-16 SHUFFLE layout for the AITER ASM / gluon paged-attention path. The KV
# cache is allocated with physical page size 16 (the ASM kernel page), and each
# logical sparse block (128 tokens) spans PAGES_PER_SPARSE_BLOCK contiguous
# physical 16-pages. Used by the fused SHUFFLE KV-insert and the sparse
# block-table builders.
ASM_PAGE_SIZE = 16
PAGES_PER_SPARSE_BLOCK = SPARSE_BLOCK_SIZE // ASM_PAGE_SIZE  # 8


def to_page16_shuffle(k_cache, v_cache, k_scale, v_scale):
    """Reinterpret the standard page-128 SHUFFLE KV/scale views as page-16
    SHUFFLE for the MiniMax-M3 ASM/gluon kernels. Zero-copy (128 == 8*16):

        K:     [N, nkv, hd//x, 128, x] -> [N*8, nkv, hd//x, 16, x]
        V:     [N, nkv, 128//x, hd, x] -> [N*8, nkv, 16//x, hd, x]
        scale: [N, nkv, 128]           -> [N*8, nkv, 16]   (fp8 only)

    Scales are re-viewed only when present (fp8); bf16 passes them through
    (None).
    """
    n_blocks, nkv = k_cache.shape[0], k_cache.shape[1]
    x = k_cache.shape[-1]
    head_dim = k_cache.shape[2] * x
    num_phys16 = n_blocks * PAGES_PER_SPARSE_BLOCK

    k16 = k_cache.view(num_phys16, nkv, head_dim // x, ASM_PAGE_SIZE, x)
    v16 = v_cache.view(num_phys16, nkv, ASM_PAGE_SIZE // x, head_dim, x)
    if k_scale is not None and v_scale is not None:
        k_scale = k_scale.view(num_phys16, nkv, ASM_PAGE_SIZE)
        v_scale = v_scale.view(num_phys16, nkv, ASM_PAGE_SIZE)
    return k16, v16, k_scale, v_scale
