# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Image input from API clients: what may be sent, and loading it.

What a client may send follows DeepSeek's vision contract, which agrees with
OpenAI's where the two overlap (docs/api_image_input.md). The offline engine
keeps reading whatever it is given, server paths included: these rules are
the API's, where the client is not the operator.
"""

import functools
import http.client
import io
import ipaddress
import socket
import threading
import time
import urllib.request
from base64 import b64decode
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from atom.utils import envs

MAX_URL_CHARS = 8192
MAX_IMAGE_BYTES = 32 << 20
MAX_REQUEST_IMAGE_BYTES = 64 << 20
# This server's own bound, beyond DeepSeek's: images that compress well fit
# the byte limits and still decode to tens of GiB.
MAX_REQUEST_PIXELS = 256 * 10**6
MAX_IMAGES = 600
MAX_SIDE = 8192
# From this many images a request on, a side may be only half as long.
MANY_IMAGES = 15
MAX_SIDE_MANY = 4096
DOWNLOAD_SECONDS = 60
LOW_DETAIL_SIDE = 512
# PIL's format names; a phone camera's multi-picture JPEG opens as "MPO".
FORMATS = frozenset({"JPEG", "MPO", "PNG", "GIF", "WEBP"})
DETAILS = frozenset({"low", "high", "original", "auto"})
NO_FILES_API = "file references are not supported: this server has no Files API"
_IMAGE_PARTS = frozenset({"image_url", "image", "file"})


def check_chat_images(
    messages: Sequence[Any], *, tool_results: bool = False
) -> list[Any]:
    """``messages`` with each image part in its ``image_url`` form, checked
    before anything is fetched.

    A ``file`` part carries an image as base64 ``file_data``; the
    processor-style ``{"type": "image", "image": source}`` part is a source
    like any other, so it cannot name a server path either. Images belong to
    ``user`` messages, and with ``tool_results`` to ``tool`` messages too: the
    Responses and Messages contracts take them there, chat's does not.
    """
    roles = ("user", "tool") if tool_results else ("user",)
    where = "user messages and tool results" if tool_results else "user messages"
    checked, count = [], 0
    for message in messages:
        content = message.content
        if isinstance(content, list) and any(map(_is_image, content)):
            if message.role not in roles:
                raise ValueError(
                    f"images are allowed in {where} only, "
                    f"not in {message.role} messages"
                )
            content = [_image_url_part(p) if _is_image(p) else p for p in content]
            count += sum(map(_is_image, content))
            message = message.model_copy(update={"content": content})
        checked.append(message)
    if count > MAX_IMAGES:
        raise ValueError(f"a request may carry at most {MAX_IMAGES} images")
    return checked


def load_chat_images(messages: list[dict]) -> list[dict]:
    """``messages`` with each image part's source checked, fetched and
    decoded into an ``image`` part: the model's processor never reads a
    client's source itself. Blocking: run off the event loop."""
    count = sum(
        _is_image(part)
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )
    side = MAX_SIDE if count < MANY_IMAGES else MAX_SIDE_MANY
    total = pixels = 0

    def admit(image: Any) -> None:
        # Every decoded image lives until the processor runs: what they take
        # together is bounded from the headers, before any is decoded.
        nonlocal pixels
        pixels += image.size[0] * image.size[1]
        if pixels > MAX_REQUEST_PIXELS:
            raise ValueError(
                f"a request's images are limited to "
                f"{MAX_REQUEST_PIXELS // 10**6} megapixels together"
            )

    def load(source: Mapping) -> Any:
        nonlocal total
        data = _read(source["url"])
        if len(data) > MAX_IMAGE_BYTES:  # a URL stops at it; base64 arrives whole
            raise ValueError(f"an image is at most {MAX_IMAGE_BYTES >> 20} MiB")
        total += len(data)
        if total > MAX_REQUEST_IMAGE_BYTES:
            raise ValueError(
                f"a request's images are limited to "
                f"{MAX_REQUEST_IMAGE_BYTES >> 20} MiB together"
            )
        return _decode(data, source.get("detail"), side, admit)

    loaded = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = [
                (
                    {"type": "image", "image": load(_image_url_part(part)["image_url"])}
                    if _is_image(part)
                    else part
                )
                for part in content
            ]
            message = {**message, "content": content}
        loaded.append(message)
    return loaded


def _is_image(part: Any) -> bool:
    # A client may send any JSON as `type`: a list is unhashable.
    kind = part.get("type") if isinstance(part, Mapping) else None
    return isinstance(kind, str) and kind in _IMAGE_PARTS


def _image_url_part(part: Mapping) -> dict:
    kind = part["type"]
    if kind == "image_url":
        source = part.get("image_url")
        if not isinstance(source, Mapping):
            raise ValueError("image_url must include image_url.url")
        url, detail = source.get("url"), source.get("detail")
    elif kind == "image":
        url, detail = part.get("image"), part.get("detail")
    else:  # file: OpenAI nests it under "file", DeepSeek writes it on the part
        source = part["file"] if isinstance(part.get("file"), Mapping) else part
        if source.get("file_id") is not None:
            raise ValueError(NO_FILES_API)
        url, detail = source.get("file_data"), None
        if not isinstance(url, str) or not url.startswith("data:"):
            raise ValueError(
                "a file part carries an image as a base64 data URL in file_data"
            )
    if not isinstance(url, str) or not url:
        raise ValueError("an image needs a URL or a base64 data URL")
    if url.startswith("data:"):
        # Whatever type it declares: the bytes say what the image is.
        if not url.partition(",")[0].endswith(";base64"):
            raise ValueError("an image data URL must be base64-encoded")
    elif not url.lower().startswith(("http://", "https://")):
        raise ValueError(
            "an image is a base64 data URL or a public http(s) URL, "
            "never a server path"
        )
    elif len(url) > MAX_URL_CHARS:
        raise ValueError(f"image URLs are limited to {MAX_URL_CHARS} characters")
    if detail is not None and (not isinstance(detail, str) or detail not in DETAILS):
        raise ValueError(f"image detail must be one of {', '.join(sorted(DETAILS))}")
    image_url = {"url": url} if detail is None else {"url": url, "detail": detail}
    return {"type": "image_url", "image_url": image_url}


def _read(url: str) -> bytes:
    if url.startswith("data:"):
        try:
            return b64decode(url.partition(",")[2], validate=True)
        except ValueError as exc:  # binascii.Error included
            raise ValueError("invalid base64 image data") from exc
    return _download(url)


def _download(url: str) -> bytes:
    opener = (
        _trusting_opener() if envs.ATOM_IMAGE_URL_ALLOW_PRIVATE else _public_opener()
    )
    data = bytearray()
    with _Deadline(DOWNLOAD_SECONDS) as deadline:
        try:
            with opener.open(url, timeout=DOWNLOAD_SECONDS) as response:
                while chunk := response.read1(1 << 16):
                    data += chunk
                    if len(data) > MAX_IMAGE_BYTES:
                        raise ValueError(
                            f"an image is at most " f"{MAX_IMAGE_BYTES >> 20} MiB"
                        )
        # URLError and timeouts are OSErrors; http.client raises HTTPException
        # for a URL it cannot use (a non-numeric port) and a malformed reply.
        except (OSError, http.client.HTTPException) as exc:
            # A socket's own timeout may beat the timer to the deadline.
            if not deadline.expired and time.monotonic() < deadline.end:
                raise ValueError(f"could not fetch image {url[:200]}: {exc}") from exc
            deadline.expired = True
    if deadline.expired:  # the reply was cut short, or never came
        raise ValueError(f"an image URL must download within {DOWNLOAD_SECONDS} s")
    return bytes(data)


_downloading = threading.local()


class _Deadline:
    """One download's time, from connecting through the headers, every
    redirect and the last byte: when it runs out, the sockets the download
    opened are shut, which ends whatever is waiting on them."""

    def __init__(self, seconds: float):
        self.end = time.monotonic() + seconds
        self.expired = False
        self._sockets: list[socket.socket] = []
        self._lock = threading.Lock()
        self._timer = threading.Timer(seconds, self._expire)
        self._timer.daemon = True

    def __enter__(self):
        _downloading.deadline = self
        self._timer.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._timer.cancel()
        _downloading.deadline = None

    def remaining(self) -> float:
        return max(self.end - time.monotonic(), 0.001)

    def track(self, sock: socket.socket) -> None:
        with self._lock:
            self._sockets.append(sock)
            expired = self.expired
        if expired:
            _shut(sock)

    def _expire(self) -> None:
        with self._lock:
            self.expired = True
            sockets = list(self._sockets)
        for sock in sockets:
            _shut(sock)


def _shut(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass  # closed already


def _decode(data: bytes, detail: str | None, side: int, admit: Callable) -> Any:
    """The image in its own mode, as the media loader returns one: color
    conversion is the processor's, which may use the alpha channel."""
    from PIL import Image

    many = f" from {MANY_IMAGES} images a request on" if side < MAX_SIDE else ""
    # Pillow fails on bad data with more than OSError (SyntaxError, IndexError,
    # struct.error, ...): any of it is the client's data, not a server error.
    try:
        image = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError as exc:  # too large to even open
        raise ValueError(f"images are limited to {side} px a side{many}") from exc
    except Exception as exc:
        raise ValueError("invalid image data") from exc
    if image.format not in FORMATS:
        raise ValueError(
            f"unsupported image format {image.format}; "
            "supported: JPEG, PNG, GIF, WebP"
        )
    if max(image.size) > side:
        raise ValueError(
            f"images are limited to {side} px a side{many}, "
            f"got {image.size[0]}x{image.size[1]}"
        )
    admit(image)
    try:
        # Decode now, so bad data fails here rather than in the processor. An
        # animation stays on its first frame, the one a processor reads.
        image.load()
    except Exception as exc:
        raise ValueError("invalid image data") from exc
    if detail == "low":
        if image.mode.startswith("I;16"):
            image = image.convert("I")  # Pillow resizes no 16-bit mode
        image.thumbnail((LOW_DETAIL_SIDE, LOW_DETAIL_SIDE))
    return image


@functools.cache
def _public_opener() -> urllib.request.OpenerDirector:
    """Every hop, redirects included, connects to a public address. No proxy:
    it would connect on the client's behalf."""
    return _opener(_HTTPHandler(public=True), _HTTPSHandler(public=True))


@functools.cache
def _trusting_opener() -> urllib.request.OpenerDirector:
    """Any address, through the system proxy if one is set, for deployments
    that trust their clients with the internal network."""
    return _opener(
        urllib.request.ProxyHandler(),
        _HTTPHandler(public=False),
        _HTTPSHandler(public=False),
    )


def _opener(*handlers: urllib.request.BaseHandler) -> urllib.request.OpenerDirector:
    """http(s) only: no file, ftp or data handler, so a redirect cannot leave
    http either."""
    opener = urllib.request.OpenerDirector()
    for handler in (
        *handlers,
        _RedirectHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
        urllib.request.UnknownHandler(),
    ):
        opener.add_handler(handler)
    return opener


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib reads a redirect's whole body before it follows, with no
        # bound; nothing in it is needed.
        fp.close()
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _connection(base: type, public: bool) -> type:
    """``base`` within the download's deadline, and, if ``public``, reaching
    public addresses only."""

    class Connection(base):
        def connect(self):
            deadline = getattr(_downloading, "deadline", None)
            if deadline is not None:
                self.timeout = deadline.remaining()

            def create_connection(address, timeout, source_address=None):
                sock = _connect(address, timeout, source_address, public=public)
                if deadline is not None:
                    # A proxy tunnel and the TLS handshake still to come: in
                    # the time left, and cut short with the rest.
                    sock.settimeout(deadline.remaining())
                    deadline.track(sock)
                return sock

            self._create_connection = create_connection
            super().connect()
            if deadline is not None:
                # For https, the TLS socket: wrapping detached the plain one.
                deadline.track(self.sock)

        def _tunnel(self):
            super()._tunnel()
            deadline = getattr(_downloading, "deadline", None)
            if deadline is not None:
                # The handshake through the proxy: in what the CONNECT left.
                self.sock.settimeout(deadline.remaining())

    return Connection


class _HTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, *, public: bool):
        super().__init__()
        self._connection = _connection(http.client.HTTPConnection, public)

    def http_open(self, req):
        return self.do_open(self._connection, req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, *, public: bool):
        super().__init__()
        self._connection = _connection(http.client.HTTPSConnection, public)

    def https_open(self, req):
        return self.do_open(self._connection, req, context=self._context)


def _connect(address, timeout, source_address=None, *, public) -> socket.socket:
    """``socket.create_connection`` in the download's time left, each address
    tried in what the others left; and, if ``public``, to public addresses
    only.

    Then every address the host resolves to must be public, and the socket
    connects to one of those addresses rather than resolving the name again,
    so DNS cannot answer differently the second time. TLS still verifies the
    host name.
    """
    host, port = address
    resolved = list(
        dict.fromkeys(
            info[4][0]
            for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        )
    )
    if public and (not resolved or not all(map(_is_public, resolved))):
        raise ValueError("image URLs must resolve to public addresses")
    deadline = getattr(_downloading, "deadline", None)
    error = None
    for ip in resolved:
        try:
            return socket.create_connection(
                (ip, port),
                deadline.remaining() if deadline is not None else timeout,
                source_address,
            )
        except OSError as exc:
            error = exc
    raise error


# IPv6 addresses that stand for an IPv4 one in their last 32 bits: NAT64's
# well-known prefix, IPv4-compatible and SIIT. A host behind NAT64 reaches that
# IPv4 address, so it is the one to judge.
_EMBEDDED_IPV4 = tuple(
    ipaddress.ip_network(n) for n in ("64:ff9b::/96", "::/96", "::ffff:0:0:0/96")
)
# NAT64's local-use prefix: its IPv4 part can sit anywhere, so it is refused.
_LOCAL_NAT64 = ipaddress.ip_network("64:ff9b:1::/48")


def _is_public(ip: str) -> bool:
    address = ipaddress.ip_address(ip)
    if address.version == 6:
        # Deprecated site-local addresses still number some internal networks,
        # and `is_global` takes them for public.
        if address in _LOCAL_NAT64 or address.is_site_local:
            return False
        embedded = address.ipv4_mapped or address.sixtofour
        if embedded is None and any(address in n for n in _EMBEDDED_IPV4):
            embedded = ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
        address = embedded or address
    return address.is_global and not address.is_multicast
