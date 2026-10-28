"""Complete TP4 request-owner serving with native expert and proposal seams."""

import os


def install():
    from atom.model_engine.model_runner import ModelRunner

    if os.environ.get("DSV41_OWNER_SERVING") != "1":
        return
    if getattr(ModelRunner, "_request_owner_service_installed", False):
        return
    from .owner_prefill_accounting import install as install_accounting
    from .owner_serving import install as install_execution

    install_execution()
    install_accounting()
    ModelRunner._request_owner_service_installed = True
