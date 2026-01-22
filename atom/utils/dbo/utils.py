from typing import TYPE_CHECKING, Any
from atom.utils.forward_context import AttentionMetaData, UBatchSlice, UBatchSlices
import torch
import os
import threading

prev_set_stream = torch.cuda.set_stream

_current_stream_tls = threading.local()

_graph_pool_id = None

prev_set_stream = torch.cuda.set_stream


_current_platform = None

class Platform:
    _global_graph_pool: Any | None = torch.cuda

    def get_global_graph_pool(self) -> Any:
        """
        Return the global graph pool for this platform.
        """
        cls = self.__class__
        if cls._global_graph_pool is None:
            cls._global_graph_pool = self.graph_pool_handle()
        return cls._global_graph_pool



def _patched_set_stream(stream: torch.cuda.Stream) -> None:
    _current_stream_tls.value = stream
    prev_set_stream(stream)


torch.cuda.set_stream = _patched_set_stream


# if TYPE_CHECKING:
current_platform = Platform()


def current_stream() -> torch.cuda.Stream:
    if not hasattr(_current_stream_tls, "value") or _current_stream_tls.value is None:
        torch.cuda.set_stream(torch.cuda.Stream())
    
    return _current_stream_tls.value

def set_graph_pool_id(graph_pool_id):
    global _graph_pool_id
    _graph_pool_id = graph_pool_id


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    """
    Creates a new query_start_loc that corresponds to the requests in
    request_slice.

    Note: This function creates a new tensor to hold the new query_start_locs.
    This will break cudagraph compatibility.
    """
    return (
        query_start_loc[request_slice.start : request_slice.stop + 1]
        - query_start_loc[request_slice.start]
    )


def _make_metadata_with_slice(
    ubatch_slice: UBatchSlice, 
    attn_metadata: AttentionMetaData,
    is_prefill: bool = False,
) -> AttentionMetaData:
    """
    This function creates a new AttentionMetaData that corresponds to
    the requests included in ubatch_slice
    """

    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    # cu_seqlens_q equals to query_start_loc
    cu_seqlens_q = attn_metadata.cu_seqlens_q
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert cu_seqlens_q[first_req] <= first_tok < cu_seqlens_q[first_req + 1], (
        "Token slice start outside of first request"
    )
    # NOTE: last token can be outside of the last request if we have CG padding.

    # If the "middle" request has tokens in both ubatches, we have to split it.
    # If ubatch_slice is the first ubatch then we will be splitting the last
    # request. If it's the second microbatch, then we will be splitting the
    # first request
    splits_first_request = first_tok > cu_seqlens_q[first_req]
    splits_last_request = last_tok < cu_seqlens_q[last_req + 1] - 1

    query_cu_seqlens_q = slice_query_start_locs(cu_seqlens_q, request_slice)

    assert len(query_cu_seqlens_q) >= 2, (
        f"cu_seqlens_q must have at least 2 elements, got {len(query_cu_seqlens_q)}"
    )

    if splits_first_request:
        tokens_skipped = first_tok - cu_seqlens_q[first_req]
        query_cu_seqlens_q[1:] -= tokens_skipped

    # context_lens equals to seq_lens
    seq_lens = attn_metadata.context_lens[request_slice]

    if splits_last_request:
        tokens_skipped = query_cu_seqlens_q[-1] - token_slice.stop
        query_cu_seqlens_q[-1] -= tokens_skipped

        # Make sure we don't modify the seq_lens tensors
        #  (not cudagraph compatible)
        seq_lens = seq_lens.clone()
        seq_lens[-1] -= tokens_skipped

    max_seqlen_k = int(seq_lens.max())

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    
    # Calculate max_seqlen_q from the sliced cu_seqlens_q
    query_lens = query_cu_seqlens_q[1:] - query_cu_seqlens_q[:-1]
    max_seqlen_q = int(torch.max(torch.abs(query_lens)).item())

    # This is to account for the case where we are in a dummy
    # run and cu_seqlens_q is full of 0s
    if max_seqlen_q == 0:
        max_seqlen_q = attn_metadata.max_seqlen_q

    min_seqlen_q = 0

    block_tables = attn_metadata.block_tables[request_slice] if attn_metadata.block_tables is not None else None
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    if is_prefill:
        # Prefill phase: need cu_seqlens_k for flash attention
        cu_seqlens_k = None
        if attn_metadata.cu_seqlens_k is not None:
            cu_seqlens_k = slice_query_start_locs(
                attn_metadata.cu_seqlens_k, request_slice
            )
            if splits_first_request:
                # Adjust cu_seqlens_k similar to cu_seqlens_q
                tokens_skipped = first_tok - cu_seqlens_q[first_req]
                cu_seqlens_k[1:] -= tokens_skipped
            if splits_last_request:
                tokens_skipped = cu_seqlens_k[-1] - token_slice.stop
                cu_seqlens_k[-1] -= tokens_skipped
        print("This is prefill attention metadata cu_seqlens_k", cu_seqlens_k)
        return AttentionMetaData(
            cu_seqlens_q=query_cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            slot_mapping=slot_mapping,
            context_lens=seq_lens,
            block_tables=block_tables,
            dropout_p=attn_metadata.dropout_p,
        )
    else:
        # Decode phase: primarily uses block_tables for paged attention
        return AttentionMetaData(
            cu_seqlens_q=query_cu_seqlens_q,
            # cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=seq_lens,
            block_tables=block_tables,
            kv_indptr=attn_metadata.kv_indptr[request_slice.start : request_slice.stop + 1] - attn_metadata.kv_indptr[request_slice.start] if attn_metadata.kv_indptr is not None else None,
            kv_indices=attn_metadata.kv_indices if attn_metadata.kv_indices is not None else None,
            kv_last_page_lens=attn_metadata.kv_last_page_lens[request_slice] if attn_metadata.kv_last_page_lens is not None else None,
        )



def split_attn_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: AttentionMetaData,
    is_prefill: bool = False,
) -> list[AttentionMetaData]:
    """
    Creates a new AttentionMetaData instance that corresponds to the
    requests for each UBatchSlice in ubatch_slices.

    Args:
        ubatch_slices: List of slices defining the micro-batches.
        common_attn_metadata: The original attention metadata to split.
        is_prefill: Whether this is a prefill phase (True) or decode phase (False).

    Note: This function does not modify common_attn_metadata
    """
    results = []
    for ubatch_slice in ubatch_slices:
        results.append(
            _make_metadata_with_slice(ubatch_slice, common_attn_metadata, is_prefill)
        )

    return results

# This just pads the second ubatch slice out to the total number of tokens
# (num_tokens + padding) since we do `create_ubatch_slices` before applying DP padding.
def _pad_out_ubatch_slices(
    ubatch_slices: UBatchSlices, num_total_tokens: int, num_reqs_padded: int
) -> UBatchSlices:
    # TODO(lucas): handle empty second ubatch
    padded_second_request_slice = slice(
        ubatch_slices[1].request_slice.start, num_reqs_padded
    )
    padded_second_token_slice = slice(
        ubatch_slices[1].token_slice.start, num_total_tokens
    )
    return [
        ubatch_slices[0],
        UBatchSlice(padded_second_request_slice, padded_second_token_slice),
    ]

