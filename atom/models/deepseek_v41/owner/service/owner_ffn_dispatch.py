"""Owner FFN boundary: routed TP partials or a complete TP all-reduce."""


def routed_forward(moe, packed, shared, *, wait_shared=None):
    experts = moe.experts
    backend = experts._comm_fused_moe
    if backend is not None and backend.supports(packed.shape[0]):
        if (
            backend.use_dp_reduce_scatter
            or experts.tp_size != 4
            or experts.ep_size != 1
            or experts.dp_size != 1
        ):
            raise ValueError("Owner comm-fused FFN requires a TP4 all-reduce output")
        # The ordinary custom-op entry consumes shared immediately; it does
        # not forward before_shared_add. Join before entering it. A later
        # overlap candidate must preserve this producer/consumer dependency.
        if wait_shared is not None:
            wait_shared()
    return experts.forward_maybe_comm_fused(packed, moe.router_logits(packed), shared)


def finish_output(routed, shared, complete, rank, capacity, reduce_scatter):
    """Consume a complete AR by slicing; combine TP partials before one RS."""
    expected = (4 * capacity, 5120)
    if routed.shape != expected or shared.shape != expected or not 0 <= rank < 4:
        raise ValueError("Owner FFN requires four rank-major shards of hidden 5120")
    if complete:
        # Includes shared and every TP rank already. Another add or RS would
        # count the same contribution twice (or four times).
        output = routed.narrow(0, rank * capacity, capacity)
    else:
        output = reduce_scatter(routed + shared)
    return output.view(1, capacity, 5120)
