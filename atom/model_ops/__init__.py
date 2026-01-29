from .paged_attention import PagedAttention
from .radix_attention import RadixAttention

# this global class is used to construct the attention op in model
# it can be assigned to different attention op
# default PagedAttention is used as ATOM for now supports PagedAttention
# for sglang, RadixAttention will be assigned to ATTN_CLS
# TODO: add env flag or argument to swicth the attention class
ATTN_CLS = PagedAttention
