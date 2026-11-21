from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ATOM_CONFIG = REPO_ROOT / "atom" / "config.py"
ATOM_SETUP = REPO_ROOT / "atom" / "model_engine" / "model_setup.py"
ATOM_PP_COMM = REPO_ROOT / "atom" / "distributed" / "pp_comm.py"
ATOMRL_SERVER = (
    REPO_ROOT / "atomrl" / "atomrl" / "engine" / "inference" / "atom_ray_server.py"
)
ATOM_ENGINE = (
    REPO_ROOT / "atomrl" / "atomrl" / "engine" / "inference" / "atom_engine.py"
)
ATOM_LLM_ENGINE = REPO_ROOT / "atom" / "model_engine" / "llm_engine.py"
ATOM_ASYNC_ENGINE = REPO_ROOT / "atom" / "rollout" / "async_engine.py"
ROOT_SETUP = REPO_ROOT / "setup.py"
ATOMRL_SETUP = REPO_ROOT / "atomrl" / "setup.py"
RL_TRAINER = REPO_ROOT / "atomrl" / "atomrl" / "trainer" / "rl_trainer.py"
ASYNC_TRAINER = REPO_ROOT / "atomrl" / "atomrl" / "trainer" / "async_trainer.py"
ROLLOUT_WORKER = REPO_ROOT / "atomrl" / "atomrl" / "workers" / "rollout_worker.py"
RAY_WORKER_GROUP = (
    REPO_ROOT / "atomrl" / "atomrl" / "controller" / "ray_worker_group.py"
)
VLLM_HTTP_ENGINE = (
    REPO_ROOT / "atomrl" / "atomrl" / "engine" / "inference" / "vllm_http_engine.py"
)
RDMA_TRANSFER = (
    REPO_ROOT / "atomrl" / "atomrl" / "engine" / "inference" / "rdma_weight_transfer.py"
)
ATOM_CAPABILITIES = REPO_ROOT / "atom" / "model_engine" / "capabilities.py"
RDMA_DOC = REPO_ROOT / "atomrl" / "examples" / "docs" / "07-disaggregated-rdma.md"
RDMA_DOC_CN = REPO_ROOT / "atomrl" / "examples" / "docs" / "07-disaggregated-rdma_cn.md"
OLD_PATCH = (
    REPO_ROOT
    / "atomrl"
    / "atomrl"
    / "engine"
    / "inference"
    / "model_runner_nocustomar.py"
)


def test_atom_exposes_custom_all_reduce_as_a_configured_public_option():
    config_text = ATOM_CONFIG.read_text(encoding="utf-8")
    setup_text = ATOM_SETUP.read_text(encoding="utf-8")
    pp_text = ATOM_PP_COMM.read_text(encoding="utf-8")

    assert "disable_custom_all_reduce: bool = False" in config_text
    assert "config.disable_custom_all_reduce" in setup_text
    assert "init_pp_aware_dist_env(" in setup_text
    assert "set_custom_all_reduce(not disable_custom_all_reduce)" in pp_text
    assert pp_text.index("set_custom_all_reduce(") < pp_text.index(
        "init_distributed_environment("
    )


def test_atomrl_passes_the_public_option_instead_of_a_runner_monkey_patch():
    server_text = ATOMRL_SERVER.read_text(encoding="utf-8")

    assert "disable_custom_all_reduce" in server_text
    assert "model_runner_nocustomar.NoCustomARModelRunner" not in server_text
    assert "_ENABLE_CUSTOM_ALL_REDUCE" not in server_text
    assert "_all_gather_out_place" not in server_text
    assert not OLD_PATCH.exists()


def test_atomrl_uses_public_engine_and_worker_rpc_seams():
    server_text = ATOMRL_SERVER.read_text(encoding="utf-8")
    engine_text = ATOM_ENGINE.read_text(encoding="utf-8")
    llm_engine_text = ATOM_LLM_ENGINE.read_text(encoding="utf-8")
    trainer_text = RL_TRAINER.read_text(encoding="utf-8")
    async_trainer_text = ASYNC_TRAINER.read_text(encoding="utf-8")

    assert "core_mgr" not in server_text
    assert "update_weights_from_ipc" in server_text
    assert "update_weights_from_shm" in server_text
    assert "engine.collective_rpc" in server_text
    assert "def is_sleeping" in engine_text
    assert "def set_weight_path" in engine_text
    assert "def wake_up" in engine_text
    assert "def reset_prefix_cache" in llm_engine_text
    assert "engine.reset_prefix_cache" in server_text
    assert "_atom_engine._sleeping" not in trainer_text
    assert "_atom_engine._weight_dir" not in trainer_text
    assert "_atom_engine._send_cmd" not in trainer_text
    assert "_atom_engine._weight_dir" not in async_trainer_text


def test_atom_weight_sync_session_discards_sender_failures():
    text = ATOM_ASYNC_ENGINE.read_text(encoding="utf-8")
    session = text.split("def weight_sync_session", 1)[1].split("\n    def ", 1)[0]
    assert "except BaseException as exc:" in session
    assert "DISCARD_WEIGHT_SYNC_CMD" in session
    assert "broadcast_utility_command_sync" in session


def test_atomrl_does_not_probe_private_engine_or_dispatch_state():
    server_text = ATOMRL_SERVER.read_text(encoding="utf-8")
    rollout_text = ROLLOUT_WORKER.read_text(encoding="utf-8")
    trainer_text = RL_TRAINER.read_text(encoding="utf-8")
    worker_group_text = RAY_WORKER_GROUP.read_text(encoding="utf-8")
    http_engine_text = VLLM_HTTP_ENGINE.read_text(encoding="utf-8")

    assert 'getattr(self._engine, "_model"' not in rollout_text
    assert 'getattr(self._engine, "_tp_size"' not in trainer_text
    assert "self._actor_wg._dp_rank_mapping" not in trainer_text
    assert "def dp_rank_mapping" in worker_group_text
    assert "def is_sleeping" in http_engine_text
    assert "__dataclass_fields__" not in server_text
    assert 'hasattr(sp, "n")' not in server_text
    assert "two ATOM generations" not in server_text


def test_capability_docs_describe_the_public_collective_contract():
    text = ATOM_CAPABILITIES.read_text(encoding="utf-8")
    assert "AtomRL probes ATOM with" not in text
    assert "collective-safe" in text.lower()


def test_rdma_docs_describe_configured_custom_all_reduce():
    for path in (RDMA_DOC, RDMA_DOC_CN):
        text = path.read_text(encoding="utf-8")
        assert "NoCustomARModelRunner" not in text
        assert "disable_custom_all_reduce" in text


def test_atomrl_imports_the_public_atom_rdma_wire_contract():
    text = RDMA_TRANSFER.read_text(encoding="utf-8")
    assert "from atom.rollout import rdma_protocol" in text
    assert "_CMD_END = 0" not in text
    assert "_CMD_BUCKET = 1" not in text
    assert "_HEADER_WORDS = 4" not in text


def test_lumen_extensions_are_owned_by_atomrl_package():
    root_setup = ROOT_SETUP.read_text(encoding="utf-8")
    atomrl_setup = ATOMRL_SETUP.read_text(encoding="utf-8")

    assert "lumen_hip_extensions" not in root_setup
    assert "LUMEN_BUILD_EXTENSIONS" in atomrl_setup
    assert "atomrl.lumen.csrc._fused_quant_transpose" in atomrl_setup
