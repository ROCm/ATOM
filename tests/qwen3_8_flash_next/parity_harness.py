"""Single-rank distributed bring-up so ATOM modules can be built for parity tests.

The upstream vLLM implementation under /app/wef is the reference: several of
its files (hyperconnection.py, parts of ple_layer.py) import only torch, so
they can be loaded standalone and diffed against the ATOM port numerically.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import torch

WEF = "/app/wef/vllm/models/qwen3_8_flash_next"


def requires_reference() -> None:
    """Skip when the upstream checkout is not present.

    `/app/wef` is a read-only sibling clone, not a dependency this repo
    installs, so it can simply be absent. Tests that diff against it then have
    nothing to compare to -- which is a missing input, not a failure, and must
    not be reported as one. The self-contained tests (QSA against dense
    attention, the kernel contracts) still cover the port without it.
    """
    import pytest

    if not os.path.isdir(WEF):
        pytest.skip(f"reference implementation not found at {WEF}")


def requires_real_aiter() -> None:
    """Skip when a sibling test module has stubbed out aiter.

    Several ATOM tests install partial `aiter` stubs into sys.modules at
    import time, and those leak across the whole session. These parity tests
    need the genuine kernels and a GPU, so probe for the real entry points and
    skip rather than fail. Run them in their own process:
    `pytest tests/qwen3_8_flash_next/`.
    """
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("parity tests need a GPU")
    try:
        from aiter.dist import parallel_state as ps
    except ImportError:  # pragma: no cover - stubbed import
        pytest.skip("aiter.dist.parallel_state unavailable")
    for name in ("init_distributed_environment", "initialize_model_parallel"):
        function = getattr(ps, name, None)
        if function is None or isinstance(function, MagicMock):
            pytest.skip(f"aiter.dist.parallel_state.{name} is stubbed by another test")


def init_single_rank(backend: str = "nccl") -> None:
    """Bring up a 1-rank TP/PP group so ATOM's Linear layers can construct."""
    requires_real_aiter()
    from aiter.dist import parallel_state as ps

    if getattr(ps, "_TP", None) is not None:
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.cuda.set_device(0)
    ps.init_distributed_environment(world_size=1, rank=0, local_rank=0, backend=backend)
    ps.initialize_model_parallel(tensor_model_parallel_size=1)


def load_reference(relpath: str, name: str):
    """Import a standalone reference module from the wef checkout by path."""
    requires_reference()
    path = os.path.join(WEF, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_ngram_reference():
    """Exec the reference NGramEmbedding source in isolation.

    `ple_layer.py` cannot be imported (it pulls in vllm), but the n-gram
    embedding and its hash helpers depend only on torch plus a vocab
    embedding. Exec'ing that source slice keeps the reference honest -- it is
    the upstream text, not a paraphrase -- while stubbing the lookup so no
    102 GB table is needed.
    """
    requires_reference()
    import math
    from collections.abc import Iterable

    import torch.nn.functional as F
    from torch import nn

    with open(os.path.join(WEF, "nvidia/ple_layer.py")) as handle:
        source = handle.read()
    body = source[
        source.index("_MASK64") : source.index("class Qwen3_8FlashNextPLELayer")
    ]

    class _RecordingEmbedding(nn.Module):
        """Stands in for VocabParallelEmbedding; records the ids it is given."""

        def __init__(self, num_embeddings, embedding_dim, padding_size=1, prefix=""):
            super().__init__()
            self.org_vocab_size = num_embeddings
            self.embedding_dim = embedding_dim
            self.last_ids = None

        def forward(self, ids):
            self.last_ids = ids
            return torch.zeros(*ids.shape, self.embedding_dim)

    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "Iterable": Iterable,
        "VocabParallelEmbedding": _RecordingEmbedding,
        "AutoWeightsLoader": None,
        "copy_ple_embedding_shard_": None,
        "Qwen3_8FlashNextTextConfig": object,
    }
    exec(  # noqa: S102 - deliberately running upstream source as the reference
        compile(body, "ple_layer_reference", "exec"), namespace
    )
    return namespace


def load_ple_reference():
    """Exec the reference PLE layer with the vllm surface stubbed out.

    Slices the source before `direct_register_custom_op` (module-level, needs
    real vllm) and supplies the handful of objects the two classes touch. The
    convolution path used here is the reference's own `_short_conv_fallback`,
    reached by handing it a forward context with no attention metadata.
    """
    requires_reference()
    import math
    from collections.abc import Iterable, Sequence
    from types import SimpleNamespace

    import torch.nn.functional as F
    from torch import nn

    with open(os.path.join(WEF, "nvidia/ple_layer.py")) as handle:
        source = handle.read()
    body = source[
        source.index("_MASK64") : source.index("def qwen3_8_flash_next_ple_short_conv(")
    ]

    class _Embedding(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, padding_size=1, prefix=""):
            super().__init__()
            self.org_vocab_size = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))

        def forward(self, ids):
            return F.embedding(ids, self.weight)

    class _Linear(nn.Module):
        def __init__(
            self, input_size, output_size, bias=False, quant_config=None, prefix=""
        ):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(output_size, input_size))

        def forward(self, x):
            return F.linear(x, self.weight), None

    contexts: dict = {}

    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "Iterable": Iterable,
        "Sequence": Sequence,
        "VocabParallelEmbedding": _Embedding,
        "ReplicatedLinear": _Linear,
        "AutoWeightsLoader": None,
        "copy_ple_embedding_shard_": None,
        "Qwen3_8FlashNextTextConfig": object,
        "MambaBase": object,
        "MambaStateDtypeCalculator": SimpleNamespace(
            short_conv_state_dtype=lambda *a: (torch.float32,)
        ),
        "MambaStateShapeCalculator": SimpleNamespace(
            short_conv_state_shape=lambda **kw: ((1, 1),)
        ),
        "is_conv_state_dim_first": lambda: True,
        "get_forward_context": lambda: SimpleNamespace(
            attn_metadata=None, no_compile_layers=contexts
        ),
        "get_current_vllm_config": lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
        "MambaAttentionBackendEnum": SimpleNamespace(SHORT_CONV="short_conv"),
        "PleShortConvAttentionBackend": object,
        "PleShortConvAttentionMetadata": object,
        "NULL_BLOCK_ID": -1,
        "CacheConfig": object,
        "ModelConfig": object,
        "VllmConfig": object,
    }
    exec(  # noqa: S102 - deliberately running upstream source as the reference
        compile(body, "ple_layer_reference", "exec"), namespace
    )
    _register_reference_short_conv_op(contexts)
    namespace["_forward_contexts"] = contexts
    return namespace


_SHORT_CONV_OP_REGISTERED: list = []


def _register_reference_short_conv_op(contexts: dict) -> None:
    """Register `vllm::qwen3_8_flash_next_ple_short_conv` for the reference.

    The reference layer dispatches its convolution through this custom op, so
    it has to exist for the forward to run. The body mirrors the upstream one.
    """
    if _SHORT_CONV_OP_REGISTERED:
        _SHORT_CONV_OP_REGISTERED[0].update(contexts)
        contexts.update(_SHORT_CONV_OP_REGISTERED[0])
        return
    library = torch.library.Library("vllm", "FRAGMENT")
    library.define(
        "qwen3_8_flash_next_ple_short_conv("
        "Tensor inputs, Tensor(a!) output, str layer_name) -> ()"
    )

    registry = contexts

    def implementation(inputs, output, layer_name):
        result = registry[layer_name]._short_conv(inputs)
        output[: result.shape[0]].copy_(result)

    library.impl(
        "qwen3_8_flash_next_ple_short_conv",
        implementation,
        "CompositeExplicitAutograd",
    )
    _SHORT_CONV_OP_REGISTERED.append(registry)
    _SHORT_CONV_OP_REGISTERED.append(library)
