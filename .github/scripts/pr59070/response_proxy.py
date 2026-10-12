"""Local response archive for HTTP failures, token caps and matched prompts."""

import os
import hashlib
import json
from pathlib import Path
import time
from aiohttp import ClientSession, ClientTimeout, web

root = Path(os.environ["RUN_DIR"])


async def forward(request):
    mode = request.match_info["mode"]
    if mode not in ("fallback", "fused", "probe"):
        raise web.HTTPBadRequest()
    body = await request.json()
    started = time.time()
    async with request.app["client"].post(
        "http://127.0.0.1:2594/v1/completions", json=body
    ) as response:
        data = await response.read()
        record = dict(
            started=started,
            ended=time.time(),
            status=response.status,
            prompt_sha256=hashlib.sha256(
                json.dumps(body.get("prompt"), ensure_ascii=False).encode()
            ).hexdigest(),
            request_settings={k: v for k, v in body.items() if k != "prompt"},
        )
        try:
            record["response"] = json.loads(data)
        except ValueError:
            record["response_text"] = data.decode(errors="replace")
        with (root / f"{mode}-responses.jsonl").open("a") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
        return web.Response(
            body=data, status=response.status, content_type="application/json"
        )


async def lifetime(app):
    async with ClientSession(timeout=ClientTimeout(total=1900)) as client:
        app["client"] = client
        yield


app = web.Application()
app.cleanup_ctx.append(lifetime)
app.router.add_post("/{mode}/v1/completions", forward)
web.run_app(app, host="127.0.0.1", port=2595, access_log=None)
