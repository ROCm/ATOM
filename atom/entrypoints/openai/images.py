# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Image input for the serving APIs: translation, validation and loading.

Chat, Responses and Anthropic Messages each spell an image block their own
way. Every endpoint translates its blocks into the Chat ``image_url`` part
here, so one multimodal path serves all three and an image is never dropped
on the way: a block that cannot be served is a client error.

What a client may send follows DeepSeek's vision contract: JPEG, PNG, GIF
(first frame) or WebP, recognised from the bytes; a base64 data URL or a
public http(s) URL, never a server path; at most 32 MiB and 8192 px a side.
There is no Files API, so file references are refused.

"Public" is enforced: a URL is fetched only from hosts that resolve to
global addresses, redirects included, so a client cannot make the server
read its own network. ATOM_IMAGE_URL_ALLOW_PRIVATE=1 lifts this for
trusted clients.
"""

import functools
import http.client
import io
import ipaddress
import socket
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from typing import Any

from atom.multimodal.media.connector import MediaConnector
from atom.utils import envs

MAX_URL_LENGTH = 8192
MAX_IMAGE_BYTES = 32 << 20
MAX_IMAGE_SIDE = 8192
MAX_IMAGES = 600
DOWNLOAD_TIMEOUT_S = 60
LOW_DETAIL_SIDE = 512
# PIL's names; a phone camera's multi-picture JPEG decodes as "MPO".
IMAGE_FORMATS = frozenset({"JPEG", "MPO", "PNG", "GIF", "WEBP"})
DETAILS = frozenset({"low", "high", "original", "auto"})
ANTHROPIC_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)
NO_FILES_API = "file references are not supported: this server has no Files API"


def image_url_part(url: Any, detail: Any = None) -> dict:
    """A Chat ``image_url`` part, its source and ``detail`` checked."""
    if not isinstance(url, str) or not url:
        raise ValueError("an image needs a URL or base64 data URL")
    if len(url) > MAX_URL_LENGTH and not url.startswith("data:"):
        raise ValueError(f"image URLs are limited to {MAX_URL_LENGTH} characters")
    if not url.startswith(("data:image/", "http://", "https://")):
        raise ValueError("an image must be a base64 data:image URL or an http(s) URL")
    if url.startswith("data:") and not url.split(",", 1)[0].endswith(";base64"):
        raise ValueError("an image data URL must be base64-encoded")
    if detail is not None and detail not in DETAILS:
        raise ValueError(f"image detail must be one of {sorted(DETAILS)}")
    image_url = {"url": url} if detail is None else {"url": url, "detail": detail}
    return {"type": "image_url", "image_url": image_url}


def chat_image_parts(content: Any) -> Any:
    """Chat content with every image part in its ``image_url`` form.

    That is ``file`` parts, and the processor-style
    ``{"type": "image", "image": source}`` part some clients send, whose
    source the media loader would otherwise read as it came: a server path
    or any URL, past the checks every other image goes through.
    """
    if not isinstance(content, list):
        return content
    parts = []
    for part in content:
        if isinstance(part, Mapping) and part.get("type") == "image_url":
            source = part.get("image_url")
            url = source.get("url") if isinstance(source, Mapping) else source
            detail = source.get("detail") if isinstance(source, Mapping) else None
            part = image_url_part(url, detail)
        elif isinstance(part, Mapping) and part.get("type") == "file":
            source = part.get("file") if isinstance(part.get("file"), Mapping) else part
            if source.get("file_id") is not None:
                raise ValueError(NO_FILES_API)
            part = image_url_part(source.get("file_data"))
        elif isinstance(part, Mapping) and part.get("type") == "image":
            part = image_url_part(part.get("image"), part.get("detail"))
        parts.append(part)
    return parts


def responses_content(content: Any) -> Any:
    """Responses message content as Chat content: text, or text and images."""
    if not isinstance(content, list) or not any(
        isinstance(p, Mapping) and p.get("type") == "input_image" for p in content
    ):
        return None
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
        elif not isinstance(part, Mapping):
            continue
        elif part.get("type") == "input_image":
            if part.get("file_id") is not None:
                raise ValueError(NO_FILES_API)
            url = part.get("image_url")
            if isinstance(url, Mapping):
                url = url.get("url")
            parts.append(image_url_part(url, part.get("detail")))
        elif isinstance(part.get("text"), str):
            parts.append({"type": "text", "text": part["text"]})
    return parts


def anthropic_image_part(block: Mapping) -> dict:
    """An Anthropic ``image`` block as a Chat ``image_url`` part."""
    source = block.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("an image block needs a source")  # noqa: TRY004
    kind = source.get("type")
    if (kind == "base64" and "url" in source) or (kind == "url" and "data" in source):
        raise ValueError("an image source is base64 data or a URL, not both")
    if kind == "base64":
        media_type = source.get("media_type")
        if media_type not in ANTHROPIC_MEDIA_TYPES:
            raise ValueError(
                f"image media_type must be one of {sorted(ANTHROPIC_MEDIA_TYPES)}"
            )
        data = source.get("data")
        if not isinstance(data, str) or not data:
            raise ValueError("a base64 image source needs data")
        return image_url_part(f"data:{media_type};base64,{data}")
    if kind == "url":
        return image_url_part(source.get("url"))
    if kind == "file":
        raise ValueError(NO_FILES_API)
    raise ValueError("image source type must be base64, url or file")


def has_image(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(p, Mapping) and p.get("type") == "image_url" for p in content
    )


def check_image_roles(messages: Iterable[Mapping], roles: frozenset[str]) -> None:
    """Images belong only to ``roles``; anywhere else is a client error."""
    for message in messages:
        if message.get("role") not in roles and has_image(message.get("content")):
            raise ValueError(
                f"images are not allowed in {message.get('role')} messages; "
                f"allowed in: {', '.join(sorted(roles))}"
            )


def load_images(messages: list[dict]) -> list[dict]:
    """Messages with every ``image_url`` part fetched, checked and decoded.

    Each becomes an ``image`` part holding the decoded image, so the model's
    processor never reads a client-supplied source itself. Blocking: run off
    the event loop.
    """
    count = sum(
        1
        for m in messages
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if isinstance(p, Mapping) and p.get("type") == "image_url"
    )
    if count > MAX_IMAGES:
        raise ValueError(f"a request may carry at most {MAX_IMAGES} images")
    loaded = []
    for message in messages:
        content = message.get("content")
        if has_image(content):
            content = [
                (
                    {"type": "image", "image": _load(p["image_url"])}
                    if isinstance(p, Mapping) and p.get("type") == "image_url"
                    else p
                )
                for p in content
            ]
            message = {**message, "content": content}
        loaded.append(message)
    return loaded


def _load(source: Mapping):
    from PIL import Image, UnidentifiedImageError

    data = _read(source["url"])
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in IMAGE_FORMATS:
                raise ValueError(
                    f"unsupported image format {image.format}; "
                    "supported: JPEG, PNG, GIF, WebP"
                )
            if max(image.size) > MAX_IMAGE_SIDE:
                raise ValueError(
                    f"images are limited to {MAX_IMAGE_SIDE} px a side, "
                    f"got {image.size[0]}x{image.size[1]}"
                )
            # The first frame, for an animated GIF or WebP.
            image = image.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError("invalid image data") from exc
    if source.get("detail") == "low":
        image.thumbnail((LOW_DETAIL_SIDE, LOW_DETAIL_SIDE))
    return image


def _read(url: str) -> bytes:
    if url.startswith("data:"):
        data = MediaConnector().read(url)
    else:
        open_url = (
            urllib.request.urlopen
            if envs.ATOM_IMAGE_URL_ALLOW_PRIVATE
            else _public_opener().open
        )
        try:
            with open_url(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
                data = response.read(MAX_IMAGE_BYTES + 1)
        # http.client raises HTTPException, not OSError, for a URL it cannot
        # use (a non-numeric port) and for a malformed or truncated reply.
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            OSError,
            ValueError,
        ) as exc:
            raise ValueError(f"could not fetch image {url[:200]}: {exc}") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"images are limited to {MAX_IMAGE_BYTES >> 20} MiB")
    return data


@functools.cache
def _public_opener() -> urllib.request.OpenerDirector:
    """http(s) only, and every hop -- redirects included -- to a public address.

    No proxy handler (a proxy would connect on the client's behalf) and no
    ftp/file handlers (a redirect could otherwise leave http).
    """
    opener = urllib.request.OpenerDirector()
    for handler in (
        _PublicHTTPHandler(),
        _PublicHTTPSHandler(),
        urllib.request.HTTPRedirectHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
        urllib.request.UnknownHandler(),
    ):
        opener.add_handler(handler)
    return opener


class _PublicHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_public(http.client.HTTPConnection), req)


class _PublicHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(
            _public(http.client.HTTPSConnection), req, context=self._context
        )


def _public(connection_class):
    """``connection_class`` whose socket reaches only public addresses."""

    def connection(host, **kwargs):
        conn = connection_class(host, **kwargs)
        conn._create_connection = _connect_public
        return conn

    return connection


def _connect_public(address, *args, **kwargs) -> socket.socket:
    """``socket.create_connection``, but only to public addresses.

    Every address the host resolves to must be public, and the socket
    connects to one of those addresses instead of resolving the name again,
    so DNS cannot answer differently the second time.
    """
    host, port = address
    resolved = list(
        dict.fromkeys(
            info[4][0]
            for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        )
    )
    if not resolved or not all(_is_public(ip) for ip in resolved):
        raise ValueError("image URLs must resolve to public addresses")
    error = None
    for ip in resolved:
        try:
            return socket.create_connection((ip, port), *args, **kwargs)
        except OSError as exc:
            error = exc
    raise error


def _is_public(ip: str) -> bool:
    address = ipaddress.ip_address(ip)
    if getattr(address, "ipv4_mapped", None):
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast
