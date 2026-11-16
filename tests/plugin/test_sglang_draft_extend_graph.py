"""Which draft-extend CUDA graph runner a drafter keeps after graph init.

``atom.plugin.register`` patches SGLang's ``EagleDraftWorker.init_cuda_graphs``.
The SGLang classes it touches are replaced by stand-ins whose
``_capture_cuda_graphs`` follows SGLang 0.5.22: the draft-extend runner is
captured when graphs are on, the draft-extend backend is visible, and the HIP
gate admits the backend (``sglang_gate``; it admits ``QwenSparseAttnBackend``,
aiter multi-step and DSA, not ATOM's DeepSeek-V4 backend).
"""

import itertools
import sys
from types import ModuleType, SimpleNamespace

import pytest

from atom.plugin import register

ENABLE_EXTEND = "ATOM_SGLANG_V4_ENABLE_DRAFT_EXTEND_CG"
DISABLE_EXTEND = "ATOM_SGLANG_V4_DISABLE_DRAFT_EXTEND_CG"
DISABLE_DRAFT = "ATOM_SGLANG_V4_DISABLE_DRAFT_CG"
FLAGS = (ENABLE_EXTEND, DISABLE_EXTEND, DISABLE_DRAFT)

DRAFTERS = {
    "flash": {"arch": "Qwen4ExpForCausalLMNextN"},
    "dsv4-flash": {
        "arch": "DeepseekV4ForCausalLMNextN",
        "model_path": "/models/DeepSeek-V4-Flash",
    },
    "dsv4-pro": {
        "arch": "DeepseekV4ForCausalLMNextN",
        "model_path": "/models/DeepSeek-V4-Pro",
    },
    "glm52": {"arch": "GlmMoeDsaForCausalLMNextN", "model_type": "glm_moe_dsa"},
    "eagle3": {"arch": "LlamaForCausalLMEagle3"},
}


@pytest.fixture
def sglang(monkeypatch):
    """Fresh SGLang stand-ins with the plugin's spec-graph patch installed."""

    class DecodeCudaGraphRunner:
        def can_run_graph(self, forward_batch):
            return True

    class EAGLEDraftCudaGraphRunner:
        def execute(self, forward_batch):
            return "draft decode replay"

        def _execute(self, forward_batch):
            return "draft decode replay"

    class EAGLEDraftExtendCudaGraphRunner(DecodeCudaGraphRunner):
        def __init__(self, eagle_worker):
            assert eagle_worker.draft_extend_attn_backend is not None
            self.model_runner = eagle_worker.draft_runner
            self.buffers = SimpleNamespace(mrope_positions="mrope buffer")
            self.captured_by_sglang = eagle_worker.in_sglang_capture
            eagle_worker.captures.append(self)

        def can_run_graph(self, forward_batch):
            return True

        def execute(self, forward_batch, select_index):
            return ("draft extend replay", select_index)

    class EagleDraftWorker:
        def init_cuda_graphs(self):
            self._capture_cuda_graphs()

        def _capture_cuda_graphs(self):
            self.cuda_graph_runner = None
            self.cuda_graph_runner_for_draft_extend = None
            if self.server_args.disable_cuda_graph:
                return
            self.cuda_graph_runner = "draft decode runner"
            if self.draft_extend_attn_backend is not None and self.sglang_gate:
                self.in_sglang_capture = True
                try:
                    self.cuda_graph_runner_for_draft_extend = (
                        EAGLEDraftExtendCudaGraphRunner(self)
                    )
                finally:
                    self.in_sglang_capture = False

        def draft_forward(self, forward_batch):
            return None

        def _draft_extend_for_decode(self, batch, batch_result, *, kv_loc_plan=None):
            return None

    modules = {
        "sglang.srt.model_executor.runner": DecodeCudaGraphRunner,
        "sglang.srt.speculative.eagle_draft_cuda_graph_runner": EAGLEDraftCudaGraphRunner,
        "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner": EAGLEDraftExtendCudaGraphRunner,
        "sglang.srt.speculative.eagle_worker_v2": EagleDraftWorker,
    }
    for name, cls in modules.items():
        module = ModuleType(name)
        setattr(module, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, name, module)
    glm52_bridge = ModuleType("atom.plugin.sglang.glm52_dsa_bridge")
    glm52_bridge.GLM52_GRAPH_SEQ_LEN_CAPACITY = 10240
    monkeypatch.setitem(sys.modules, glm52_bridge.__name__, glm52_bridge)
    for flag in FLAGS:
        monkeypatch.delenv(flag, raising=False)

    swallowed = []
    monkeypatch.setattr(
        register.logger, "warning", lambda *args, **kwargs: swallowed.append(args)
    )
    register._patch_sglang_dsv4_spec_cuda_graph()
    assert EagleDraftWorker._atom_dsv4_init_cuda_graphs_patched
    yield SimpleNamespace(EagleDraftWorker=EagleDraftWorker)
    assert swallowed == []


def _worker(sglang, *, sglang_gate=True, disable_cuda_graph=False, **drafter):
    hf_config = SimpleNamespace(
        architectures=[drafter["arch"]], model_type=drafter.get("model_type", "")
    )
    model_path = drafter.get("model_path", "")
    backend = SimpleNamespace(name="draft extend backend")
    worker = sglang.EagleDraftWorker()
    worker.draft_runner = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, path=model_path),
        server_args=SimpleNamespace(model_path=model_path),
        attn_backend=backend,
    )
    worker.server_args = SimpleNamespace(
        disable_cuda_graph=disable_cuda_graph, speculative_num_draft_tokens=3
    )
    worker.draft_attn_backend = SimpleNamespace(attn_backends=[SimpleNamespace()])
    worker.draft_extend_attn_backend = backend
    worker.sglang_gate = sglang_gate
    worker.in_sglang_capture = False
    worker.captures = []
    return worker


def test_flash_drafter_keeps_the_runner_sglang_captured(sglang):
    worker = _worker(sglang, **DRAFTERS["flash"])
    backend = worker.draft_extend_attn_backend

    worker.init_cuda_graphs()

    runner = worker.cuda_graph_runner_for_draft_extend
    assert runner is not None and runner.captured_by_sglang
    assert worker.captures == [runner]
    assert worker.draft_extend_attn_backend is backend
    assert worker.cuda_graph_runner == "draft decode runner"


def test_flash_drafter_builds_the_runner_sglang_did_not_capture(sglang):
    worker = _worker(sglang, sglang_gate=False, **DRAFTERS["flash"])

    worker.init_cuda_graphs()

    runner = worker.cuda_graph_runner_for_draft_extend
    assert runner is not None and not runner.captured_by_sglang
    assert worker.captures == [runner]


@pytest.mark.parametrize("sglang_gate", [True, False])
@pytest.mark.parametrize("graphs_off_by", ["--disable-cuda-graph", DISABLE_EXTEND])
def test_flash_drafter_has_no_draft_extend_runner_with_graphs_off(
    sglang, monkeypatch, graphs_off_by, sglang_gate
):
    disable_cuda_graph = graphs_off_by == "--disable-cuda-graph"
    if not disable_cuda_graph:
        monkeypatch.setenv(graphs_off_by, "1")
    worker = _worker(
        sglang,
        sglang_gate=sglang_gate,
        disable_cuda_graph=disable_cuda_graph,
        **DRAFTERS["flash"],
    )

    worker.init_cuda_graphs()

    assert worker.cuda_graph_runner_for_draft_extend is None
    assert worker.captures == []


@pytest.mark.parametrize(
    "flags",
    [
        flags
        for n in range(len(FLAGS) + 1)
        for flags in itertools.combinations(FLAGS, n)
    ],
    ids=lambda flags: "+".join(flags) or "no-flags",
)
@pytest.mark.parametrize("sglang_gate", [True, False])
@pytest.mark.parametrize("disable_cuda_graph", [False, True])
def test_flash_draft_extend_runner_is_captured_once_and_never_dropped(
    sglang, monkeypatch, flags, sglang_gate, disable_cuda_graph
):
    for flag in flags:
        monkeypatch.setenv(flag, "1")
    worker = _worker(
        sglang,
        sglang_gate=sglang_gate,
        disable_cuda_graph=disable_cuda_graph,
        **DRAFTERS["flash"],
    )

    worker.init_cuda_graphs()

    runner = worker.cuda_graph_runner_for_draft_extend
    if not disable_cuda_graph and DISABLE_EXTEND not in flags:
        assert worker.captures == [runner]
    else:
        assert runner is None and worker.captures == []


def test_kept_sglang_runner_stages_the_flash_mrope_on_replay(sglang, monkeypatch):
    staged = []
    bridge = ModuleType("atom.plugin.sglang.qwen4_exp_bridge")
    bridge.stage_draft_graph_mrope = lambda buffer, batch: staged.append(
        (buffer, batch)
    )
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    worker = _worker(sglang, **DRAFTERS["flash"])
    worker.init_cuda_graphs()
    runner = worker.cuda_graph_runner_for_draft_extend
    batch = SimpleNamespace()

    assert runner.can_run_graph(batch)
    assert runner.execute(batch, "select") == ("draft extend replay", "select")
    assert staged == [("mrope buffer", batch)]


@pytest.mark.parametrize(
    "drafter, flags, sglang_gate, expect_runner, expect_captures, expect_decode",
    [
        ("dsv4-flash", (), False, "plugin", 1, True),
        ("dsv4-flash", (DISABLE_EXTEND,), False, None, 0, True),
        ("dsv4-flash", (DISABLE_DRAFT,), False, "plugin", 1, False),
        ("dsv4-pro", (), False, None, 0, True),
        ("dsv4-pro", (ENABLE_EXTEND,), False, "plugin", 1, True),
        ("dsv4-pro", (DISABLE_DRAFT,), False, None, 0, False),
        ("glm52", (), True, None, 0, True),
        ("glm52", (ENABLE_EXTEND,), True, None, 1, True),
        ("glm52", (ENABLE_EXTEND,), False, "plugin", 1, True),
        ("glm52", (ENABLE_EXTEND, DISABLE_DRAFT), True, None, 1, False),
        ("eagle3", (), True, None, 0, True),
        ("eagle3", (ENABLE_EXTEND,), True, "sglang", 1, True),
    ],
)
def test_draft_extend_runner_of_other_drafters(
    sglang,
    monkeypatch,
    drafter,
    flags,
    sglang_gate,
    expect_runner,
    expect_captures,
    expect_decode,
):
    for flag in flags:
        monkeypatch.setenv(flag, "1")
    worker = _worker(sglang, sglang_gate=sglang_gate, **DRAFTERS[drafter])

    worker.init_cuda_graphs()

    runner = worker.cuda_graph_runner_for_draft_extend
    if expect_runner is None:
        assert runner is None
    else:
        assert runner in worker.captures
        assert runner.captured_by_sglang == (expect_runner == "sglang")
    assert len(worker.captures) == expect_captures
    assert (worker.cuda_graph_runner is not None) == expect_decode
