from .paged_attention import PagedAttention
from .radix_attention import RadixAttention

# This global class is used to construct the attention op in model,
# it can be assigned to different attention ops.
# By default, PagedAttention is used.
# For sglang, RadixAttention will be assigned to ATTN_CLS
ATTN_CLS = PagedAttention
