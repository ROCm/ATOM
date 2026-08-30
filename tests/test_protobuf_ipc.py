import numpy as np
import pytest
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from atom.model_engine.ipc_utils import EngineCoreIpcCodec
from atom.model_engine.request import RequestOutput
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType
from atom.sampling_params import SamplingParams


def test_sequence_protobuf_roundtrip_preserves_engine_state():
    sequence = Sequence(
        [1, 2, 3],
        16,
        SamplingParams(temperature=0.4, top_k=12, max_tokens=7, logprobs=3),
        stop_token_sequences=[[7, 8]],
        id=42,
        request_id="request-42",
        kv_transfer_params={"connector": "memory"},
        mrope_positions=np.array([[1, 2]], dtype=np.int64),
    )
    sequence.status = SequenceStatus.RUNNING
    sequence.type = SequenceType.DECODE
    sequence.append_token(9)
    sequence.block_table.extend([3, 4])
    sequence.logprobs.extend([-.25])
    sequence.spec_token_ids = np.array([5, 6], dtype=np.int32)
    sequence.dspark_next_ell = 2
    sequence.parent_request_id = "parent-42"
    sequence.num_rejected = 1
    sequence.num_bonus_tokens = 2

    decoded = EngineCoreIpcCodec.decode_sequence(
        EngineCoreIpcCodec.encode_sequence(sequence)
    )

    assert decoded.id == sequence.id
    assert decoded.status == sequence.status
    assert decoded.type == sequence.type
    assert list(decoded.token_ids) == list(sequence.token_ids)
    assert list(decoded.output_tokens) == list(sequence.output_tokens)
    assert list(decoded.block_table) == list(sequence.block_table)
    assert list(decoded.logprobs) == pytest.approx(list(sequence.logprobs))
    assert np.array_equal(decoded.spec_token_ids, sequence.spec_token_ids)
    assert np.array_equal(decoded.mrope_positions, sequence.mrope_positions)
    assert decoded.dspark_next_ell == sequence.dspark_next_ell
    assert decoded.num_bonus_tokens == sequence.num_bonus_tokens


def test_engine_core_envelope_rejects_unknown_wire_version():
    frame = EngineCoreIpcCodec.encode_shutdown()
    assert (
        EngineCoreIpcCodec.decode_engine_core_envelope(frame).WhichOneof("payload")
        == "shutdown"
    )

    from atom.proto.engine import engine_core_proto as engine_core_pb2

    invalid = engine_core_pb2.EngineCoreEnvelope(wire_version=99)
    with pytest.raises(ValueError, match="unsupported"):
        EngineCoreIpcCodec.decode_engine_core_envelope(invalid.SerializeToString())


def test_add_and_stream_envelopes_roundtrip():
    sequence = Sequence([1, 2], 16, SamplingParams(max_tokens=4), id=7)
    add = EngineCoreIpcCodec.decode_engine_core_envelope(
        EngineCoreIpcCodec.encode_add_request([sequence])
    )
    decoded_sequences = EngineCoreIpcCodec.decode_add_request(add.add_request)
    assert [seq.id for seq in decoded_sequences] == [7]
    assert list(decoded_sequences[0].token_ids) == [1, 2]

    stream = EngineCoreIpcCodec.decode_engine_core_envelope(
        EngineCoreIpcCodec.encode_stream(
            [
                (
                    7,
                    RequestOutput(
                        request_id=7,
                        output_tokens=[3],
                        finished=True,
                        finish_reason="stop",
                        num_cached_tokens=2,
                    ),
                )
            ]
        )
    )
    [(sequence_id, output)] = EngineCoreIpcCodec.decode_stream(stream.stream)
    assert sequence_id == 7
    assert output.output_tokens == [3]
    assert output.finished
    assert output.finish_reason == "stop"


def test_utility_envelope_preserves_cuda_ipc_metadata_types():
    rebuild = rebuild_cuda_tensor
    payload = {
        "cmd": "update_weights_ipc",
        "ipc_handle": (
            rebuild,
            (
                torch.Tensor,
                torch.Size([4]),
                (1,),
                0,
                torch.UntypedStorage,
                torch.uint8,
                torch.device("cuda:0"),
                b"\x00\xff",
                2**60,
            ),
        ),
        "ipc_handles": {0: (rebuild, (b"rank-0",)), 3: (rebuild, (b"rank-3",))},
    }
    envelope = EngineCoreIpcCodec.decode_engine_core_envelope(
        EngineCoreIpcCodec.encode_utility_command(payload)
    )
    decoded = EngineCoreIpcCodec.decode_utility_command(envelope.utility_command)

    assert decoded == payload
    assert set(decoded["ipc_handles"]) == {0, 3}
    assert decoded["ipc_handle"][0] is rebuild


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bfloat16])
def test_sequence_multimodal_tensors_roundtrip(dtype):
    pixel_values = torch.arange(12).reshape(3, 4).to(dtype)
    image_grid_thw = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    sequence = Sequence(
        [1, 2],
        16,
        SamplingParams(max_tokens=4),
        id=8,
        multimodal_data={
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        },
    )

    decoded = EngineCoreIpcCodec.decode_sequence(
        EngineCoreIpcCodec.encode_sequence(sequence)
    )

    assert decoded.multimodal_data is not None
    assert torch.equal(decoded.multimodal_data["pixel_values"], pixel_values)
    assert torch.equal(decoded.multimodal_data["image_grid_thw"], image_grid_thw)

