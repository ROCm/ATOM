import pytest

from atom.plugin import prepare as plugin_prepare


@pytest.fixture(autouse=True)
def _reset_framework_state():
    # Autouse fixture: pytest runs this before/after every test.
    plugin_prepare._set_framework_backbone("atom")
    yield
    plugin_prepare._set_framework_backbone("atom")


def test_default_mode_is_server_mode():
    assert plugin_prepare.is_plugin_mode() is False
    assert plugin_prepare.is_vllm() is False
    assert plugin_prepare.is_sglang() is False


def test_set_framework_to_vllm():
    plugin_prepare._set_framework_backbone("vllm")
    assert plugin_prepare.is_plugin_mode() is True
    assert plugin_prepare.is_vllm() is True
    assert plugin_prepare.is_sglang() is False


def test_set_framework_to_sgl_alias():
    plugin_prepare._set_framework_backbone("sgl")
    assert plugin_prepare.is_plugin_mode() is True
    assert plugin_prepare.is_vllm() is False
    assert plugin_prepare.is_sglang() is True


def test_set_framework_unsupported_raises():
    with pytest.raises(ValueError, match="Unsupported framework"):
        plugin_prepare._set_framework_backbone("tensorflow")
