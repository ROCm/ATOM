"""Keep the service EP selector out of request-client Python startup."""

import functools


def bind(serving):
    original = serving.client_environment
    if getattr(original, "_step12_ep_client", False):
        raise RuntimeError("Client environment repair already installed")

    @functools.wraps(original)
    def client_environment():
        environment = original()
        if environment["DSV41_OWNER_SERVING"] != "0":
            raise RuntimeError("Request client must not install owner serving")
        if environment.get("DSV41_REBASE_SERVICE_SERVER") == "1":
            raise RuntimeError("Request client must not install server hooks")
        environment["DSV41_STEP12_OWNER_EP4"] = "0"
        return environment

    client_environment._step12_ep_client = True
    serving.client_environment = client_environment


def install_client_environment():
    import run_owner_serving as serving

    bind(serving)
