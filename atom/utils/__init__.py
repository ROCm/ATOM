import contextlib
import ipaddress
import logging
import multiprocessing
import os
import signal
import socket
import tempfile
import time
from functools import lru_cache
from multiprocessing.context import ForkContext, SpawnContext
from multiprocessing.process import BaseProcess
from typing import Any, Iterator, Optional, Sequence, Union
from urllib.parse import urlparse
from uuid import uuid4

import psutil
import zmq
import zmq.asyncio

logger = logging.getLogger("atom")


def get_mp_context() -> Union[ForkContext, SpawnContext]:
    """Get a multiprocessing context with 'spawn' start method."""
    return multiprocessing.get_context("spawn")


def shutdown_all_processes(procs: list[BaseProcess]):
    # Shutdown the process.
    for proc in procs:
        if proc.is_alive():
            proc.terminate()

    # Allow 5 seconds for remaining procs to terminate.
    deadline = time.monotonic() + 5
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    for proc in procs:
        if proc.is_alive() and (pid := proc.pid) is not None:
            kill_process_tree(pid)


def kill_process_tree(pid: int):
    """
    Kills all descendant processes of the given pid by sending SIGKILL.

    Args:
        pid (int): Process ID of the parent process
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def split_host_port(host_port: str) -> tuple[str, int]:
    # ipv6
    if host_port.startswith("["):
        host, port = host_port.rsplit("]", 1)
        host = host[1:]
        port = port.split(":")[1]
        return host, int(port)
    else:
        host, port = host_port.split(":")
        return host, int(port)


def join_host_port(host: str, port: int) -> str:
    if is_valid_ipv6_address(host):
        return f"[{host}]:{port}"
    else:
        return f"{host}:{port}"


def get_distributed_init_method(ip: str, port: int) -> str:
    return get_tcp_uri(ip, port)


def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"


def get_open_zmq_inproc_path() -> str:
    return f"inproc://{uuid4()}"


def get_open_port() -> int:
    """
    Get an open port for the vLLM process to listen on.
    An edge case to handle, is when we run data parallel,
    we need to avoid ports that are potentially used by
    the data parallel master process.
    Right now we reserve 10 ports for the data parallel master
    process. Currently it uses 2 ports.
    """
    return _get_open_port()


def get_open_ports_list(count: int = 5) -> list[int]:
    """Get a list of open ports."""
    ports = set()
    while len(ports) < count:
        ports.add(get_open_port())
    return list(ports)


def _get_open_port() -> int:
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


@lru_cache()
def get_zmq_base_path() -> str:
    return tempfile.gettempdir()


def get_open_zmq_ipc_path() -> str:
    base_rpc_path = get_zmq_base_path()
    return f"ipc://{base_rpc_path}/{uuid4()}"


def get_engine_client_zmq_addr(local_only: bool, host: str, port: int = 0) -> str:
    """Assign a new ZMQ socket address.

    If local_only is True, participants are colocated and so a unique IPC
    address will be returned.

    Otherwise, the provided host and port will be used to construct a TCP
    address (port == 0 means assign an available port)."""

    return (
        get_open_zmq_ipc_path()
        if local_only
        else (get_tcp_uri(host, port or get_open_port()))
    )


def close_sockets(sockets: Sequence[Union[zmq.Socket, zmq.asyncio.Socket]]):
    for sock in sockets:
        if sock is not None:
            sock.close(linger=0)


def split_zmq_path(path: str) -> tuple[str, str, str]:
    """Split a zmq path into its parts."""
    parsed = urlparse(path)
    if not parsed.scheme:
        raise ValueError(f"Invalid zmq path: {path}")

    scheme = parsed.scheme
    host = parsed.hostname or ""
    port = str(parsed.port or "")

    if scheme == "tcp" and not all((host, port)):
        # The host and port fields are required for tcp
        raise ValueError(f"Invalid zmq path: {path}")

    if scheme != "tcp" and port:
        # port only makes sense with tcp
        raise ValueError(f"Invalid zmq path: {path}")

    return scheme, host, port


def make_zmq_path(scheme: str, host: str, port: Optional[int] = None) -> str:
    """Make a ZMQ path from its parts.

    Args:
        scheme: The ZMQ transport scheme (e.g. tcp, ipc, inproc).
        host: The host - can be an IPv4 address, IPv6 address, or hostname.
        port: Optional port number, only used for TCP sockets.

    Returns:
        A properly formatted ZMQ path string.
    """
    if port is None:
        return f"{scheme}://{host}"
    if is_valid_ipv6_address(host):
        return f"{scheme}://[{host}]:{port}"
    return f"{scheme}://{host}:{port}"


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L783 # noqa: E501
def make_zmq_socket(
    ctx: Union[zmq.asyncio.Context, zmq.Context],  # type: ignore[name-defined]
    path: str,
    socket_type: Any,
    bind: Optional[bool] = None,
    identity: Optional[bytes] = None,
    linger: Optional[int] = None,
) -> Union[zmq.Socket, zmq.asyncio.Socket]:  # type: ignore[name-defined]
    """Make a ZMQ socket with the proper bind/connect semantics."""

    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Calculate buffer size based on system memory
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)  # 0.5GB in bytes
    else:
        buf_size = -1  # Use system default buffer size

    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    if socket_type == zmq.XPUB:
        socket.setsockopt(zmq.XPUB_VERBOSE, True)

    # Determine if the path is a TCP socket with an IPv6 address.
    # Enable IPv6 on the zmq socket if so.
    scheme, host, _ = split_zmq_path(path)
    if scheme == "tcp" and is_valid_ipv6_address(host):
        socket.setsockopt(zmq.IPV6, 1)

    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket


@contextlib.contextmanager
def zmq_socket_ctx(
    path: str,
    socket_type: Any,
    bind: Optional[bool] = None,
    linger: int = 0,
    identity: Optional[bytes] = None,
) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    ctx = zmq.Context()  # type: ignore[attr-defined]
    try:
        yield make_zmq_socket(ctx, path, socket_type, bind=bind, identity=identity)
    except KeyboardInterrupt:
        logger.debug("Got Keyboard Interrupt.")

    finally:
        ctx.destroy(linger=linger)
