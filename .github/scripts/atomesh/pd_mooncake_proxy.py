#!/usr/bin/env python3
"""Run the pinned vLLM Mooncake example with harness health/model endpoints."""

import importlib.util
from pathlib import Path

import uvicorn
from fastapi import HTTPException


def main():
    path = Path(
        "/tmp/atomesh-native-vllm/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py"
    )
    spec = importlib.util.spec_from_file_location("mooncake_proxy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.global_args = module.parse_args()
    app = module.app

    @app.get("/liveness")
    async def liveness():
        return {"status": "alive"}

    @app.get("/health")
    async def health():
        if not app.state.ready.is_set():
            raise HTTPException(503, "Mooncake bootstrap is not ready")
        for client in app.state.prefill_clients + app.state.decode_clients:
            response = await client["client"].get("/health", timeout=10)
            if response.status_code != 200:
                raise HTTPException(503, "P/D service is not ready")
        return {"status": "ready"}

    @app.get("/v1/models")
    async def models():
        response = await app.state.decode_clients[0]["client"].get("/v1/models")
        response.raise_for_status()
        return response.json()

    uvicorn.run(app, host=module.global_args.host, port=module.global_args.port)


if __name__ == "__main__":
    main()
