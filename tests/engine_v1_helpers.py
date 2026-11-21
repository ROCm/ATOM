"""Hand engine_v1 frames to a decoder as a receiving socket does."""

import zmq


def received(frames):
    """What `recv_multipart(copy=False)` returns for `frames`: a `zmq.Frame` each."""
    return [zmq.Frame(bytes(frame)) for frame in frames]
