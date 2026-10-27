#!/usr/bin/env python3
"""Readiness/metadata routes around the unchanged pinned Mooncake proxy."""
from fastapi.responses import JSONResponse, Response
from vendor import mooncake_connector_proxy as upstream
import uvicorn


@upstream.app.get("/health")
async def health():
    return Response(status_code=200 if upstream.app.state.ready.is_set() else 503)


@upstream.app.get("/v1/models")
async def models():
    response = await upstream.app.state.decode_clients[0]["client"].get("/v1/models")
    return JSONResponse(response.json(), status_code=response.status_code)


if __name__ == "__main__":
    upstream.global_args = upstream.parse_args()
    uvicorn.run(upstream.app, host=upstream.global_args.host, port=upstream.global_args.port)
