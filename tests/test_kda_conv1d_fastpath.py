# SPDX-License-Identifier: MIT
"""GPU regressions for the guarded KDA Conv1D update fast path."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.mamba_ops import causal_conv1d as conv
from atom.model_ops.mamba_ops import kda_conv1d_optimized as fast


def _same_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )


@pytest.fixture
def run_update(monkeypatch):
    """Observe dispatch while forwarding unchanged to the real GPU kernels."""
    launches = []

    class RecordingKernel:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def run(*args, **kwargs):
                launches.append((self.name, kwargs))
                return launch(*args, **kwargs)

            return run

    monkeypatch.setattr(
        conv,
        "_causal_conv1d_update_kernel",
        RecordingKernel("generic", conv._causal_conv1d_update_kernel),
    )
    monkeypatch.setattr(
        fast,
        "kda_conv1d_reuse_kernel",
        RecordingKernel("fast", fast.kda_conv1d_reuse_kernel),
    )

    def run(call, enabled, expected):
        monkeypatch.setattr(conv, "_KDA_CONV1D_FASTPATH_ENABLED", enabled)
        launches.clear()
        result = conv.causal_conv1d_update(**call)
        assert len(launches) == 1
        selected, options = launches[0]
        assert selected == expected
        if selected == "generic":
            assert options["BLOCK_N"] == 256
            assert "num_warps" not in options
            assert "enable_fp_fusion" not in options
        return result

    return run


def _case(spec=False, lengths=None, alias="inplace", width=4, window=5):
    """The last request is padding; every pool also has unused slots/tail."""
    torch.manual_seed(37)
    lengths = lengths if lengths is not None else [window if spec else 1] * 7
    batch, dim = len(lengths), 4608
    slots = 2 * batch + 3
    state_len = width - 1 + (window - 1 if spec else 0)
    state = torch.randn(
        slots, state_len + 2, dim, device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    fused = torch.randn(sum(lengths), 6284, device="cuda", dtype=torch.bfloat16)
    reads = list(reversed(range(batch - 1)))
    writes = reads.copy()
    if alias != "inplace":
        writes = list(range(batch, 2 * batch - 1))
        if alias == "shared_read":
            reads = [0] * (batch - 1)
    reads.append(-1)
    writes.append(-1)

    def indices(values):
        if not spec:
            return torch.tensor(values, dtype=torch.int32, device="cuda")
        # Production speculative metadata passes a strided first-column view.
        table = torch.zeros(batch, window, dtype=torch.int32, device="cuda")
        table[:, 0] = torch.tensor(values, dtype=torch.int32, device="cuda")
        return table[:, 0]

    call = {
        "x": fused[:, :dim],
        "conv_state": state,
        "weight": torch.randn(dim, width, device="cuda", dtype=torch.float32) * 0.5,
        "k_dim_size": 1536,
        "v_dim_size": 1536,
        "activation": "silu",
        "conv_state_indices": indices(writes),
        "conv_state_indices_in": indices(reads),
    }
    if spec:
        starts = [0]
        for length in lengths:
            starts.append(starts[-1] + length)
        call.update(
            query_start_loc=torch.tensor(starts, dtype=torch.int32, device="cuda"),
            max_query_len=window,
            num_accepted_tokens=(
                torch.arange(batch, dtype=torch.int32, device="cuda") % window + 1
            ),
        )
    live = torch.tensor(
        [i < batch - 1 for i, length in enumerate(lengths) for _ in range(length)],
        dtype=torch.bool,
        device="cuda",
    )
    return call, live, lengths, writes


def _compare(run_update, case, expected="fast", steps=1):
    call, live, lengths, writes = case
    original = call["conv_state"]
    reference = dict(call, conv_state=original.clone())
    actual = dict(call, conv_state=original.clone())
    written = {slot for slot, length in zip(writes, lengths) if slot >= 0 and length}
    untouched = [slot for slot in range(original.shape[0]) if slot not in written]
    for step in range(steps):
        if step:
            call["x"].normal_()
            if "num_accepted_tokens" in call:
                accepted = call["num_accepted_tokens"]
                accepted.copy_(accepted % call["max_query_len"] + 1)
        ref_outputs = run_update(reference, False, "generic")
        outputs = run_update(actual, True, expected)
        for output, ref_output in zip(outputs, ref_outputs):
            # Padding rows are torch.empty allocations and have no defined value.
            _same_bytes(output[live], ref_output[live])
        _same_bytes(actual["conv_state"], reference["conv_state"])
        _same_bytes(actual["conv_state"][untouched], original[untouched])
        for slot, length in zip(writes, lengths):
            if slot < 0 or length == 0:
                continue
            prefix = call["weight"].shape[1] - 1
            if "num_accepted_tokens" in call:
                prefix += length - 1
            _same_bytes(
                actual["conv_state"][slot, :, prefix:], original[slot, :, prefix:]
            )


@pytest.mark.parametrize(
    "spec,lengths,alias",
    [
        (False, [1, 1], "inplace"),
        (False, None, "inplace"),
        (False, None, "fork"),
        (True, None, "inplace"),
        (True, [0, 1, 2, 3, 4, 5, 5], "inplace"),
        (True, [5, 4, 3, 2, 1, 0, 5], "fork"),
        (True, None, "shared_read"),
        (True, [0] * 7, "inplace"),
    ],
    ids=[
        "decode_single",
        "decode",
        "decode_fork",
        "spec_all_acceptance_offsets",
        "spec_ragged",
        "spec_fork",
        "spec_shared_read",
        "spec_empty",
    ],
)
def test_fastpath_matches_generic(run_update, spec, lengths, alias):
    case = _case(spec, lengths, alias)
    if alias == "inplace":
        # Exercise the wrapper's default read-side indices as well.
        case[0]["conv_state_indices_in"] = None
    _compare(run_update, case, steps=3)


@pytest.mark.parametrize(
    "guard",
    [
        "bias",
        "activation",
        "width",
        "float32",
        "input_cast",
        "weight_dtype",
        "state_layout",
        "input_layout",
        "spec_window",
        "varlen_without_accepted",
        "apc",
    ],
)
def test_unsupported_modes_keep_generic_dispatch(run_update, guard):
    spec = guard in ("spec_window", "varlen_without_accepted")
    case = _case(
        spec,
        width=3 if guard == "width" else 4,
        window=4 if guard == "spec_window" else 5,
    )
    call = case[0]
    if guard == "bias":
        call["bias"] = torch.randn(4608, device="cuda", dtype=torch.float32)
    elif guard == "activation":
        call["activation"] = None
    elif guard == "float32":
        call["x"] = call["x"].float()
        call["conv_state"] = call["conv_state"].float()
    elif guard == "input_cast":
        call["x"] = call["x"].float()
    elif guard == "weight_dtype":
        call["weight"] = call["weight"].to(torch.bfloat16)
    elif guard == "state_layout":
        call["conv_state"] = call["conv_state"].contiguous()
    elif guard == "input_layout":
        call["x"] = call["x"].T.contiguous().T
    elif guard == "varlen_without_accepted":
        del call["num_accepted_tokens"]
    elif guard == "apc":
        for name in ("conv_state_indices", "conv_state_indices_in"):
            call[name] = call[name][:, None]
        batch = len(case[2])
        call["initial_state_idx"] = torch.zeros(batch, dtype=torch.int32, device="cuda")
        call["block_idx_last_scheduled_token"] = torch.zeros_like(
            call["initial_state_idx"]
        )
    _compare(run_update, case, expected="generic")
