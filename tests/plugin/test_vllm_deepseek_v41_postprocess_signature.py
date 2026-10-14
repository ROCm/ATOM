# SPDX-License-Identifier: MIT
"""The V4.1 commit wrapper keeps `postprocess_sampled`'s calling convention.

This patch is installed globally, on every runner class, for every ATOM model
the plugin serves -- so its signature is an interface, not an internal detail.
A non-last pipeline rank calls it as

    self.postprocess_sampled(**self.pp_handler.get_prev_sampled_outputs(...))

whose dict is `sampled_tokens`, `num_sampled`, `num_rejected`, `idx_mapping`
and carries no `query_start_loc`. A wrapper that renamed the first or dropped
the last's default failed at argument binding:

    TypeError: postprocess_sampled() missing 2 required positional arguments:
    'sampled_token_ids' and 'query_start_loc'

which happens before any V4.1 code runs, so pipeline parallelism broke for
plugin models that never speculate.

Binding is checked without calling: these tests need no GPU, no engine and no
sampler, and binding is the whole failure.

`follow_wrapped=False` is load-bearing. `functools.wraps` sets `__wrapped__`,
and `inspect.signature` follows it by default -- so without this the checks
below read *upstream's* signature and pass no matter what the wrapper
declares. Caught by reinstating the broken signature and watching them stay
green.
"""

import inspect

import pytest

pytest.importorskip("vllm")

from atom.plugin.vllm.v41_speculative import patch_v41_speculative_commit
from atom.plugin.vllm.vllm_runtime import gpu_model_runner_classes

# What `pp_utils.get_prev_sampled_outputs` returns, by name.
PP_PAYLOAD = {
    "sampled_tokens": object(),
    "num_sampled": object(),
    "num_rejected": object(),
    "idx_mapping": object(),
}


@pytest.fixture
def patched_runners(monkeypatch):
    runners = [
        cls
        for cls in gpu_model_runner_classes()
        if getattr(cls, "postprocess_sampled", None) is not None
    ]
    if not runners:
        pytest.skip("this vLLM has no runner with postprocess_sampled")
    for cls in runners:
        monkeypatch.setattr(cls, "postprocess_sampled", cls.postprocess_sampled)
    patch_v41_speculative_commit()
    return runners


def test_the_pipeline_parallel_payload_still_binds(patched_runners):
    for cls in patched_runners:
        sig = inspect.signature(cls.postprocess_sampled, follow_wrapped=False)
        # `self` is unbound here, so supply a placeholder for it.
        sig.bind(object(), **PP_PAYLOAD)


def test_the_signature_matches_the_one_vllm_documents(patched_runners):
    """Names and defaults both: a positional-by-name caller is the one that
    breaks, and it breaks silently until a pipeline-parallel deployment."""
    for cls in patched_runners:
        params = inspect.signature(
            cls.postprocess_sampled, follow_wrapped=False
        ).parameters
        assert "sampled_tokens" in params, sorted(params)
        assert "sampled_token_ids" not in params, "upstream's name was renamed"
        assert params["query_start_loc"].default is None, (
            "query_start_loc lost its default; the pipeline-parallel caller "
            "does not pass it"
        )
