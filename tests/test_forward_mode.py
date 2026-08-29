from atom.utils.forward_context import ForwardMode


def _decide(**overrides):
    args = {
        "is_prefill": False,
        "total_seqs_num": 12,
        "scheduled_bs_decode": 12,
        "num_input_tokens": 12,
        "dp_uniform_decode": True,
        "enforce_eager": False,
        "graph_bs": [1, 2, 4, 8, 16],
        "mtp_step": 1,
    }
    args.update(overrides)
    return ForwardMode.decide(**args)


def test_decode_uses_eager_when_query_width_graph_was_not_captured():
    mode = _decide(graph_shapes={(16, 4)})

    assert mode.use_cudagraph is False
    assert mode.effective_bs == 12
    assert mode.moe_pad_bs == 12


def test_decode_uses_matching_batch_and_query_width_graph():
    mode = _decide(graph_shapes={(16, 1)})

    assert mode.use_cudagraph is True
    assert mode.effective_bs == 16
