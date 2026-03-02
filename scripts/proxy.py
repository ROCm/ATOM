# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Disaggregated P/D Proxy Service.

This service acts as a lightweight routing proxy between clients and the
disaggregated prefill/decode engine instances.  It handles:

1. **Service discovery**: Prefill and decode instances register themselves
   via ZMQ heartbeats.
2. **Request routing**: Incoming client requests are split into a prefill
   phase and a decode phase, routed to appropriate instances.
3. **Transfer coordination**: In "read" mode, the proxy waits for the
   prefill response (containing block metadata) before forwarding to decode.

Usage::

    python scripts/proxy.py
"""

import asyncio
import copy
import logging
import os
import re
import socket
import threading
import uuid

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

prefill_instances: list[dict] = []
decode_instances: list[dict] = []
request_nums: int = 0

app = Quart(__name__)

IP_PORT_PATTERN = re.compile(r"//(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)")


TRANSFER_TYPE: str | None = None


def _append_whole_dict_unique(target_list: list[dict], data_dict: dict) -> bool:
    """Append ``data_dict`` to ``target_list`` only if not already present.

    Comparison ignores the ``index`` field to avoid false negatives from
    heartbeat sequence numbers.
    """
    new_filtered = {k: v for k, v in data_dict.items() if k != "index"}
    for existed in target_list:
        existed_filtered = {k: v for k, v in existed.items() if k != "index"}
        if existed_filtered == new_filtered:
            return False
    logger.info("Registering new instance: %s", data_dict)
    target_list.append(data_dict)
    transfer_mode = data_dict.get("transfer_mode", "unknown")
    global TRANSFER_TYPE

    if TRANSFER_TYPE is None:
        TRANSFER_TYPE = transfer_mode
        logger.info("SET TRANSFER TYPE TO %s", TRANSFER_TYPE)
    elif transfer_mode != TRANSFER_TYPE:
        raise ValueError(f"mismatched transfer mode {TRANSFER_TYPE} vs {transfer_mode}")

    return True


_list_lock = threading.RLock()


def _listen_for_register(hostname: str, port: int) -> None:
    """Background loop that receives heartbeat / registration messages.

    Prefill (role ``"P"``) and decode (role ``"D"``) instances announce
    themselves over a ZMQ ROUTER socket.  De-duplication is handled by
    :func:`_append_whole_dict_unique`.
    """
    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")
    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)
    global prefill_instances
    global decode_instances

    while True:
        socks = dict(poller.poll())
        if router_socket not in socks:
            continue
        _remote_addr, msg = router_socket.recv_multipart()
        data = msgpack.loads(msg)

        if data["type"] == "HELLO":
            continue

        if data["type"] != "register":
            logger.warning("Unknown message type: %s", data["type"])
            continue

        with _list_lock:
            if data["role"] == "P":
                _append_whole_dict_unique(prefill_instances, data)
            elif data["role"] == "D":
                _append_whole_dict_unique(decode_instances, data)
            else:
                logger.warning("Unknown role in registration: %s", data["role"])


def start_service_discovery(hostname: str, port: int) -> threading.Thread:
    """Spawn a daemon thread that listens for instance registrations.

    Args:
        hostname: Network interface to bind on (e.g. ``"0.0.0.0"``).
        port: ZMQ ROUTER port for registration messages.

    Returns:
        The started listener thread.
    """
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    listener_thread = threading.Thread(
        target=_listen_for_register, args=(hostname, port), daemon=True
    )
    listener_thread.start()
    return listener_thread


async def send_request_to_prefill(
    endpoint: str,
    req_data: dict,
    request_id: str,
    d_endpoint: dict,
    dip: str,
    dport: str,
    selected_prefill_dp_rank: int | None,
) -> dict:
    """Forward a request to a prefill instance and return its JSON response.

    The request is modified to disable streaming and limit output to a
    single token (the prefill phase only computes KV caches, not tokens).
    """
    req_data["kv_transfer_params"].update(
        {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_handshake_port": d_endpoint["handshake_port"],
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": dip,
            "remote_port": dport,
        }
    )
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    req_data.pop("stream_options", None)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=6 * 6000 * 6000)
    ) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        if selected_prefill_dp_rank is not None:
            headers["X-data-parallel-rank"] = str(selected_prefill_dp_rank)
        async with session.post(
            url=endpoint, json=req_data, headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            raise RuntimeError(
                f"Prefill request failed with status {response.status}"
            )


async def start_decode_request(
    endpoint: str, req_data: dict, request_id: str
) -> tuple[aiohttp.ClientSession, aiohttp.ClientResponse]:
    """Initiate a streaming decode request, returning the open session/response.

    The caller is responsible for closing the session after consuming the
    response (see :func:`stream_decode_response`).
    """
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=6 * 6000 * 6000)
    )
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    response = await session.post(url=endpoint, json=req_data, headers=headers)
    return session, response


async def stream_decode_response(
    session: aiohttp.ClientSession,
    response: aiohttp.ClientResponse,
    request_id: str,
):
    """Yield response chunks from a decode instance, then close the session."""
    try:
        if response.status != 200:
            raise RuntimeError(
                f"Decode request {request_id} failed with status {response.status}"
            )
        async for chunk_bytes in response.content.iter_chunked(1024):
            yield chunk_bytes
    finally:
        await session.close()


def example_round_robin_dp_loader(request_number: int, dp_size: int) -> int:
    """Simple round-robin data-parallel rank selector."""
    return request_number % dp_size


def _extract_ip_port(url: str) -> tuple[str, str]:
    """Extract ``(ip, port)`` from a URL like ``http://1.2.3.4:8000/...``."""
    match = IP_PORT_PATTERN.search(url)
    if not match:
        raise ValueError(f"Cannot extract ip:port from URL: {url}")
    return match.groups()


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    """Route an incoming OpenAI-compatible request through the P/D pipeline.

    1. Select a prefill and a decode instance (round-robin).
    2. Forward the request to prefill (single-token, non-streaming).
    3. In *read* transfer mode, wait for the prefill response to obtain
       block metadata before issuing the decode request.
    4. Stream the decode response back to the client.
    """
    try:
        global request_nums
        request_nums += 1

        req_data = await request.get_json()
        request_id = str(uuid.uuid4())

        if not prefill_instances or not decode_instances:
            return await make_response((
                "Service Unavailable: no prefill or decode instances registered.",
                503,
            ))

        # Round-robin instance selection
        pid = request_nums % len(prefill_instances)
        did = request_nums % len(decode_instances)
        prefill_ep = prefill_instances[pid]
        decode_ep = decode_instances[did]

        # Optional DP rank selection within a prefill instance
        selected_prefill_dp_rank = None
        if prefill_ep["dp_size"] > 1:
            selected_prefill_dp_rank = example_round_robin_dp_loader(
                request_nums // len(prefill_instances),
                prefill_ep["dp_size"],
            )

        dip, dport = _extract_ip_port(decode_ep["request_address"])
        pip, pport = _extract_ip_port(prefill_ep["request_address"])

        # --- Prefill request ---
        req_data_to_prefill = copy.deepcopy(req_data)
        req_data_to_prefill["kv_transfer_params"] = {
            "remote_dp_size": decode_ep["dp_size"],
            "remote_tp_size": decode_ep["tp_size"],
        }

        send_prefill_task = asyncio.create_task(
            send_request_to_prefill(
                prefill_ep["request_address"],
                req_data_to_prefill,
                request_id,
                decode_ep,
                dip,
                dport,
                selected_prefill_dp_rank,
            )
        )

        # --- Decode request ---
        req_data["max_tokens"] -= 1
        req_data["kv_transfer_params"] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_handshake_port": prefill_ep["handshake_port"],
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": pip,
            "remote_port": pport,
        }

        logger.info("Transfer type: %s", TRANSFER_TYPE)

        if TRANSFER_TYPE == "read":
            prefill_response = await send_prefill_task
            logger.info("Prefill response received for request %s", request_id)
            prefill_kv = prefill_response["kv_transfer_params"]
            req_data["kv_transfer_params"]["remote_engine_id"] = prefill_kv[
                "remote_engine_id"
            ]
            req_data["kv_transfer_params"]["remote_block_ids"] = prefill_kv[
                "remote_block_ids"
            ]
            req_data["kv_transfer_params"]["transfer_id"] = prefill_kv["transfer_id"]

        req_data["kv_transfer_params"]["remote_dp_size"] = prefill_ep["dp_size"]
        req_data["kv_transfer_params"]["remote_tp_size"] = prefill_ep["tp_size"]

        if selected_prefill_dp_rank is not None:
            req_data["kv_transfer_params"]["remote_dp_rank"] = selected_prefill_dp_rank

        session, decode_response = await start_decode_request(
            decode_ep["request_address"], req_data, request_id
        )
        stream_gen = stream_decode_response(session, decode_response, request_id)
        response = await make_response(stream_gen)
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    except Exception as e:
        logger.exception("Error handling request: %s", e)
        resp = await make_response((f"Internal Server Error: {e!s}", 500))
        resp.headers["Content-Type"] = "application/json; charset=utf-8"
        return resp


if __name__ == "__main__":
    discovery_thread = start_service_discovery("0.0.0.0", 36367)

    app.debug = True
    app.config["BODY_TIMEOUT"] = 360000
    app.config["RESPONSE_TIMEOUT"] = 360000

    logger.info("Starting proxy on 0.0.0.0:10001")
    app.run(host="0.0.0.0", port=10001)
    discovery_thread.join()