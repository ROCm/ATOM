# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chat image input: what a client may send, and how the server loads it."""

import asyncio
import base64
import datetime
import http.server
import io
import random
import shutil
import socket
import ssl
import struct
import subprocess
import threading
import time
import zlib

import pytest
from fastapi import HTTPException

from atom.entrypoints.openai import api_server, images
from atom.entrypoints.openai.protocol import ChatCompletionRequest, ChatMessage

Image = pytest.importorskip("PIL.Image")


def _encoded(fmt="PNG", size=(4, 3), mode="RGB"):
    out = io.BytesIO()
    Image.new(mode, size, 1 if mode == "1" else "red").save(out, format=fmt)
    return out.getvalue()


def _data_url(data, mime="image/png"):
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


PNG = _data_url(_encoded())
QUESTION = {"type": "text", "text": "What is this?"}


def _image(url, **source):
    return {"type": "image_url", "image_url": {"url": url, **source}}


def _user(*parts):
    return ChatMessage(role="user", content=[*parts, QUESTION])


def _load(*parts):
    [message] = images.load_chat_images([{"role": "user", "content": list(parts)}])
    return [part["image"] for part in message["content"]]


# What a client may send: checked before anything is fetched.


def test_image_parts_take_one_form():
    """`file` data and the processor-style `image` part become `image_url`
    parts; text, and messages without images, stay as they were."""
    system = ChatMessage(role="system", content="Be brief.")
    kept, checked = images.check_chat_images(
        [
            system,
            _user(
                _image(PNG, detail="low"),
                {"type": "file", "file": {"file_data": PNG, "filename": "a.png"}},
                {"type": "file", "file_data": PNG},
                {"type": "image", "image": PNG},
            ),
        ]
    )
    assert kept is system
    assert checked.content == [
        _image(PNG, detail="low"),
        *[_image(PNG)] * 3,
        QUESTION,
    ]


@pytest.mark.parametrize(
    "part, message",
    [
        (_image("/etc/passwd"), "never a server path"),
        (_image("file:///etc/passwd"), "never a server path"),
        (_image("ftp://example.com/a.png"), "never a server path"),
        ({"type": "image", "image": "/etc/passwd"}, "never a server path"),
        ({"type": "image", "image": {"url": PNG}}, "needs a URL"),
        ({"type": "image_url", "image_url": {}}, "needs a URL"),
        ({"type": "image_url", "image_url": PNG}, "image_url.url"),
        (_image("data:image/png,raw"), "base64-encoded"),
        (_image("https://example.com/" + "a" * 8192), "8192 characters"),
        (_image(PNG, detail="medium"), "detail must be one of"),
        ({"type": "file", "file": {"file_id": "file-1"}}, "no Files API"),
        ({"type": "file", "file_id": "file-1", "file_data": PNG}, "no Files API"),
        ({"type": "file", "file": {}}, "file_data"),
        ({"type": "file", "file_data": "https://example.com/a.png"}, "file_data"),
    ],
)
def test_what_a_client_may_not_send_is_refused(part, message):
    with pytest.raises(ValueError, match=message):
        images.check_chat_images([_user(part)])


def test_a_non_string_type_or_detail_is_the_clients_error():
    """A client may send any JSON there, and a list is unhashable: a 400, not
    a TypeError and a 500."""
    images.check_chat_images([_user({"type": ["image_url"], "image_url": {}})])
    for detail in (["low"], {"low": True}):
        with pytest.raises(ValueError, match="detail must be one of"):
            images.check_chat_images([_user(_image(PNG, detail=detail))])


def test_a_url_is_at_most_8192_characters():
    longest = "https://example.com/" + "a" * (8192 - len("https://example.com/"))
    images.check_chat_images([_user(_image(longest))])
    with pytest.raises(ValueError, match="8192 characters"):
        images.check_chat_images([_user(_image(longest + "a"))])


@pytest.mark.parametrize("role", ["system", "assistant", "tool"])
def test_images_belong_to_user_messages(role):
    with pytest.raises(ValueError, match="user messages only"):
        images.check_chat_images([ChatMessage(role=role, content=[_image(PNG)])])


def test_a_request_carries_at_most_600_images():
    images.check_chat_images([_user(*[_image(PNG)] * 600)])
    with pytest.raises(ValueError, match="at most 600 images"):
        images.check_chat_images(
            [_user(*[_image(PNG)] * 300), _user(*[_image(PNG)] * 301)]
        )


# Loading: decoded, in the four formats, within the limits.


@pytest.mark.parametrize(
    "fmt, mime",
    [
        ("JPEG", "image/jpeg"),
        ("PNG", "image/png"),
        ("GIF", "image/gif"),
        ("WEBP", "image/webp"),
    ],
)
def test_the_four_formats_load(fmt, mime):
    [image] = _load(_image(_data_url(_encoded(fmt), mime)))
    assert (image.format, image.size) == (fmt, (4, 3))


def test_the_format_is_read_from_the_bytes():
    """Not from the type a data URL declares: an uploader that cannot tell
    sends `application/octet-stream`."""
    [image] = _load(_image(_data_url(_encoded("JPEG"), "image/png")))
    assert image.format == "JPEG"
    [image] = _load(_image(_data_url(_encoded(), "application/octet-stream")))
    assert image.format == "PNG"
    pdf = {"type": "file", "file_data": _data_url(b"%PDF-1.4", "application/pdf")}
    with pytest.raises(ValueError, match="invalid image data"):
        _load(pdf)
    with pytest.raises(ValueError, match="unsupported image format BMP"):
        _load(_image(_data_url(_encoded("BMP"))))
    with pytest.raises(ValueError, match="invalid image data"):
        _load(_image(_data_url(b"not an image")))
    with pytest.raises(ValueError, match="invalid base64"):
        _load(_image("data:image/png;base64,***"))


def test_an_image_keeps_its_mode():
    """Color conversion is the processor's, which may use the alpha channel."""
    [image] = _load(_image(_data_url(_encoded(mode="RGBA"))))
    assert image.mode == "RGBA"


def _animation(*colors):
    frames = [Image.new("RGB", (4, 4), color) for color in colors]
    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:])
    return out.getvalue()


def test_an_animation_is_its_first_frame():
    [image] = _load(_image(_data_url(_animation("red", "blue"), "image/gif")))
    assert image.tell() == 0
    assert image.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_an_animation_cut_short_still_has_its_first_frame():
    """As the media loader reads one on main: later frames are not read."""
    gif = _animation("red", "blue", "green")
    [image] = _load(_image(_data_url(gif[: len(gif) * 5 // 6], "image/gif")))
    assert image.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_bad_image_data_is_the_clients_error():
    """Pillow fails on a broken chunk after the first IDAT with SyntaxError,
    not OSError, and only once it decodes."""
    noise = random.Random(0).randbytes(256 * 256 * 3)  # several IDAT chunks
    out = io.BytesIO()
    Image.frombytes("RGB", (256, 256), noise).save(out, format="PNG")
    png = out.getvalue()
    second = png.index(b"IDAT", png.index(b"IDAT") + 4)
    broken = png[:second] + b"\x01\x02\x03\x04" + png[second + 4 :]
    with pytest.raises(ValueError, match="invalid image data"):
        _load(_image(_data_url(broken)))


def _png_header(width, height):
    """A PNG that says it is `width` x `height` and holds no pixels."""

    def chunk(kind, body):
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


def test_a_huge_image_is_refused_from_its_header():
    """Pillow refuses a 20000x20000 header on opening."""
    with pytest.raises(ValueError, match="8192 px a side"):
        _load(_image(_data_url(_png_header(20000, 20000))))


def test_a_requests_decoded_pixels_are_limited_before_decoding():
    """Images that compress well fit the byte limits and decode to tens of
    GiB: the headers are added up first. A header with no pixels behind it
    is refused for its size, not found invalid by decoding."""
    big = _image(_data_url(_encoded(size=(4096, 4096), mode="1")))  # 16.8 MP
    _load(*[big] * 15)
    with pytest.raises(ValueError, match="megapixels together"):
        _load(*[big] * 15, _image(_data_url(_png_header(4096, 4096))))


def test_an_image_is_at_most_32_mib_however_sent(monkeypatch):
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", len(_encoded()) - 1)
    with pytest.raises(ValueError, match="an image is at most"):
        _load(_image(PNG))


def test_a_side_is_at_most_8192_px_or_4096_from_15_images_on():
    def line(width):
        return _image(_data_url(_encoded(size=(width, 1), mode="1")))

    _load(line(8192))
    with pytest.raises(ValueError, match="8192 px a side"):
        _load(line(8193))
    _load(line(4097), *[_image(PNG)] * 13)
    with pytest.raises(ValueError, match="4096 px a side from 15 images"):
        _load(line(4097), *[_image(PNG)] * 14)
    # Counted across the request, not per message.
    split = [
        {"role": "user", "content": [line(4097), *[_image(PNG)] * 6]},
        {"role": "user", "content": [_image(PNG)] * 8},
    ]
    with pytest.raises(ValueError, match="4096 px a side from 15 images"):
        images.load_chat_images(split)


def test_a_requests_images_are_limited_together(monkeypatch):
    monkeypatch.setattr(images, "MAX_REQUEST_IMAGE_BYTES", 2 * len(_encoded()))
    _load(_image(PNG), _image(PNG))
    with pytest.raises(ValueError, match="together"):
        _load(_image(PNG), _image(PNG), _image(PNG))


def test_low_detail_fits_512():
    wide = _data_url(_encoded(size=(2048, 1024)))
    details = ("low", "high", "original", "auto")
    loaded = _load(*(_image(wide, detail=detail) for detail in details))
    assert [image.size for image in loaded] == [(512, 256)] + [(2048, 1024)] * 3


def test_a_phones_multi_picture_jpeg_loads():
    """Pillow names a JPEG with an MPF segment, as phones write them, MPO."""
    out = io.BytesIO()
    frames = [Image.new("RGB", (4, 3), color) for color in ("red", "blue")]
    frames[0].save(out, format="MPO", save_all=True, append_images=frames[1:])
    [image] = _load(_image(_data_url(out.getvalue(), "image/jpeg")))
    assert image.format == "MPO"


def test_low_detail_takes_a_16_bit_image():
    """A depth map or a scan: Pillow resizes no 16-bit mode as it stands."""
    out = io.BytesIO()
    Image.new("I;16", (2048, 1024), 1000).save(out, format="PNG")
    [low] = _load(_image(_data_url(out.getvalue()), detail="low"))
    assert low.size == (512, 256)


# URLs: public addresses only, every redirect included.


@pytest.mark.parametrize(
    "ip, public",
    [
        ("8.8.8.8", True),
        ("2001:4860:4860::8888", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("172.16.0.1", False),
        ("192.168.1.1", False),
        ("169.254.169.254", False),
        ("100.64.0.1", False),
        ("0.0.0.0", False),
        ("224.0.0.1", False),
        ("::1", False),
        ("fe80::1", False),
        ("fc00::1", False),
        ("::ffff:127.0.0.1", False),
        # IPv6 standing for an IPv4 address: NAT64, IPv4-compatible, SIIT, 6to4.
        ("64:ff9b::a00:1", False),
        ("64:ff9b::a9fe:a9fe", False),
        ("64:ff9b::808:808", True),
        ("::a00:1", False),
        ("::ffff:0:a00:1", False),
        ("2002:a00:1::1", False),
        ("2002:808:808::1", True),
        ("64:ff9b:1::808:808", False),
        # Deprecated site-local, which `is_global` takes for public.
        ("fec0::1", False),
        ("fec0:0:0:ffff::1", False),
    ],
)
def test_what_counts_as_public(ip, public):
    assert images._is_public(ip) is public


@pytest.fixture
def server():
    """A local HTTP server: `routes` maps a path to what its handler writes."""
    routes = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            routes[self.path](self)

        def do_CONNECT(self):  # as a proxy
            routes["CONNECT"](self)

        def log_message(self, *_args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield routes, httpd.server_address[1]
    httpd.shutdown()


def _serve(body):
    def write(handler):
        handler.send_response(200)
        handler.end_headers()
        handler.wfile.write(body)

    return write


def _redirect(location):
    def write(handler):
        handler.send_response(302)
        handler.send_header("Location", location)
        handler.end_headers()

    return write


def _self_signed(cert_pem, key_pem):
    """A certificate for `public.test`, made for the test: with the openssl
    command, or `cryptography` where there is no such command."""
    openssl = shutil.which("openssl")
    if openssl:
        subprocess.run(
            [openssl, "req", "-x509", "-nodes", "-days", "1"]
            + ["-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1"]
            + ["-subj", "/CN=public.test", "-addext", "subjectAltName=DNS:public.test"]
            + ["-keyout", str(key_pem), "-out", str(cert_pem)],
            check=True,
            capture_output=True,
        )
        return
    x509 = pytest.importorskip("cryptography.x509", reason="no openssl to make a cert")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "public.test")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("public.test")]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    cert_pem.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_pem.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


@pytest.fixture
def tls_server(tmp_path, monkeypatch):
    """`server` over TLS, with a certificate for `public.test` the client is
    made to trust."""
    cert_pem, key_pem = tmp_path / "cert.pem", tmp_path / "key.pem"
    _self_signed(cert_pem, key_pem)
    trusting = ssl.create_default_context(cafile=str(cert_pem))
    monkeypatch.setattr(ssl, "_create_default_https_context", lambda *_a: trusting)
    # The https handler takes its context when the opener is built.
    images._public_opener.cache_clear()
    serving = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    serving.load_cert_chain(cert_pem, key_pem)
    routes = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            routes[self.path](self)

        def log_message(self, *_args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.socket = serving.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield routes, httpd.server_address[1]
    httpd.shutdown()
    images._public_opener.cache_clear()  # not to keep trusting the test CA


@pytest.fixture
def names(monkeypatch):
    """`public.test` is the local server, standing for a public host;
    `internal.test` is a private address; `rebind.test` resolves to both;
    `flip.test` answers public once, then private."""
    hosts = {
        "public.test": ["127.0.0.1"],
        "internal.test": ["10.0.0.1"],
        "rebind.test": ["127.0.0.1", "10.0.0.1"],
    }
    flips = []
    resolve = socket.getaddrinfo  # for the addresses the socket connects to

    def getaddrinfo(host, port, *args, **kwargs):
        if host == "flip.test":
            flips.append(host)
            ips = ["127.0.0.1"] if len(flips) == 1 else ["10.0.0.1"]
        elif host in hosts:
            ips = hosts[host]
        else:
            return resolve(host, port, *args, **kwargs)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]

    monkeypatch.setattr(images.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(images, "_is_public", lambda ip: ip == "127.0.0.1")


def test_a_public_url_loads(server, names):
    routes, port = server
    routes["/a.png"] = _serve(_encoded())
    [image] = _load(_image(f"http://public.test:{port}/a.png"))
    assert image.size == (4, 3)


@pytest.mark.parametrize(
    "path, message",
    [
        ("http://internal.test:{port}/a.png", "public addresses"),
        ("http://rebind.test:{port}/a.png", "public addresses"),
        ("https://internal.test:{port}/a.png", "public addresses"),
        ("http://public.test:{port}/to-internal", "public addresses"),
        ("http://public.test:{port}/to-file", "could not fetch"),
        ("http://public.test:{port}/to-ftp", "unknown url type: ftp"),
        ("http://public.test:{port}/missing", "could not fetch"),
        ("http://public.test:abc/a.png", "could not fetch"),
    ],
)
def test_a_url_or_redirect_off_the_public_network_is_refused(
    server, names, monkeypatch, path, message
):
    """https is refused before any TLS, so no certificate is needed. urllib
    would follow an ftp redirect, past the address check, if the opener had
    an ftp handler."""
    monkeypatch.setattr(images, "DOWNLOAD_SECONDS", 2)  # if it ever connects
    routes, port = server
    routes["/a.png"] = _serve(_encoded())
    routes["/to-internal"] = _redirect(f"http://internal.test:{port}/a.png")
    routes["/to-file"] = _redirect("file:///etc/passwd")
    routes["/to-ftp"] = _redirect("ftp://public.test/a.png")

    def missing(handler):
        handler.send_error(404)

    routes["/missing"] = missing
    with pytest.raises(ValueError, match=message):
        _load(_image(path.format(port=port)))


def test_the_download_connects_to_the_address_it_checked(server, names, monkeypatch):
    """A second lookup of `flip.test` would answer private."""
    routes, port = server
    routes["/a.png"] = _serve(_encoded())
    monkeypatch.setattr(images, "DOWNLOAD_SECONDS", 2)
    [image] = _load(_image(f"http://flip.test:{port}/a.png"))
    assert image.size == (4, 3)


def test_no_proxy_is_used(server, names, monkeypatch):
    """A proxy would connect on the client's behalf, past the address check."""
    routes, port = server
    routes["/a.png"] = _serve(_encoded())
    monkeypatch.setenv("http_proxy", f"http://internal.test:{port}")
    images._public_opener.cache_clear()
    try:
        [image] = _load(_image(f"http://public.test:{port}/a.png"))
    finally:
        images._public_opener.cache_clear()
    assert image.size == (4, 3)


def test_a_url_stops_at_32_mib_while_it_streams(server, names, monkeypatch):
    """Not once it is all in memory: here a body that never ends."""
    routes, port = server

    def endless(handler):
        try:
            handler.wfile.write(b"HTTP/1.0 200 OK\r\n\r\n")
            for _ in range(200):
                handler.wfile.write(b"x" * 65536)
                handler.wfile.flush()
                time.sleep(0.02)
        except OSError:
            pass  # the client gave up, as it should

    routes["/big.png"] = endless
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 1 << 20)
    monkeypatch.setattr(images, "DOWNLOAD_SECONDS", 2)
    started = time.monotonic()
    with pytest.raises(ValueError, match="an image is at most"):
        _load(_image(f"http://public.test:{port}/big.png"))
    assert time.monotonic() - started < 1.5


@pytest.mark.parametrize("phase", ["headers", "chunk size", "body"])
def test_one_deadline_holds_the_whole_download(server, names, monkeypatch, phase):
    """A server dripping bytes holds no phase past it: each byte would reset
    a socket's own timeout."""
    routes, port = server
    head = {
        "headers": b"HTTP/1.1 200 OK\r\n",
        "chunk size": b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
        "body": b"HTTP/1.0 200 OK\r\n\r\n",
    }[phase]
    drop = (
        b"X-Slow: 1\r\n"
        if phase == "headers"
        else b"0" if phase == "chunk size" else b"x"
    )

    def drip(handler):
        try:
            handler.wfile.write(head)
            for _ in range(20):
                handler.wfile.write(drop)
                handler.wfile.flush()
                time.sleep(0.2)
        except OSError:
            pass  # the client gave up, as it should

    routes["/slow.png"] = drip
    monkeypatch.setattr(images, "DOWNLOAD_SECONDS", 1)
    started = time.monotonic()
    with pytest.raises(ValueError, match="within 1 s"):
        _load(_image(f"http://public.test:{port}/slow.png"))
    assert time.monotonic() - started < 3


def test_the_deadline_holds_over_tls(tls_server, names, monkeypatch):
    """The timer shuts the TLS socket itself: wrapping detached the plain one,
    and each byte would reset the socket's own timeout."""
    routes, port = tls_server

    def drip(handler):
        try:
            handler.wfile.write(b"HTTP/1.0 200 OK\r\n\r\n")
            for _ in range(20):
                handler.wfile.write(b"x")
                handler.wfile.flush()
                time.sleep(0.2)
        except OSError:
            pass  # the client gave up, as it should

    routes["/slow.png"] = drip
    routes["/a.png"] = _serve(_encoded())
    [image] = _load(_image(f"https://public.test:{port}/a.png"))  # TLS works
    assert image.size == (4, 3)
    monkeypatch.setattr(images, "DOWNLOAD_SECONDS", 1)
    started = time.monotonic()
    with pytest.raises(ValueError, match="within 1 s"):
        _load(_image(f"https://public.test:{port}/slow.png"))
    assert time.monotonic() - started < 3


def test_a_redirects_body_is_not_read(server, names, monkeypatch):
    """urllib would read a redirect's whole body, unbounded, before following
    it: here one that never ends."""
    routes, port = server
    routes["/a.png"] = _serve(_encoded())

    def endless(handler):
        try:
            handler.wfile.write(b"HTTP/1.0 302 Found\r\nLocation: /a.png\r\n\r\n")
            for _ in range(200):
                handler.wfile.write(b"x" * 65536)
                handler.wfile.flush()
                time.sleep(0.02)
        except OSError:
            pass  # the client moved on, as it should

    routes["/endless"] = endless
    monkeypatch.setattr(images, "DOWNLOAD_SECONDS", 2)
    started = time.monotonic()
    [image] = _load(_image(f"http://public.test:{port}/endless"))
    assert image.size == (4, 3) and time.monotonic() - started < 1.5


def test_a_slow_connect_leaves_the_tls_handshake_only_what_is_left(names, monkeypatch):
    """The handshake runs on the connected socket's timeout: it must be the
    time left, not the whole deadline again."""
    silent = socket.create_server(("127.0.0.1", 0))  # accepts, never answers
    port = silent.getsockname()[1]
    connect = socket.create_connection

    def slow_connect(*args, **kwargs):
        time.sleep(1.5)
        return connect(*args, **kwargs)

    monkeypatch.setattr(images.socket, "create_connection", slow_connect)
    monkeypatch.setattr(images, "DOWNLOAD_SECONDS", 2)
    started = time.monotonic()
    try:
        with pytest.raises(ValueError, match="within 2 s"):
            _load(_image(f"https://public.test:{port}/a.png"))
    finally:
        silent.close()
    assert time.monotonic() - started < 2.8


def test_private_urls_by_default_and_when_allowed(server, monkeypatch):
    """No stand-ins: the local server is refused until the operator allows
    private addresses."""
    routes, port = server
    routes["/a.png"] = _serve(_encoded())
    url = f"http://127.0.0.1:{port}/a.png"
    with pytest.raises(ValueError, match="public addresses"):
        _load(_image(url))
    for proxy in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(proxy, raising=False)
        monkeypatch.delenv(proxy.upper(), raising=False)
    monkeypatch.setenv("ATOM_IMAGE_URL_ALLOW_PRIVATE", "1")
    images._trusting_opener.cache_clear()
    try:
        [image] = _load(_image(url))
    finally:
        images._trusting_opener.cache_clear()
    assert image.size == (4, 3)


def test_each_address_gets_only_the_time_left(names, monkeypatch):
    """Private URLs allowed or not: a host with many silent addresses holds
    the download no longer than one."""
    tried = []

    def silent(address, timeout, *_args):
        tried.append(timeout)
        raise TimeoutError

    monkeypatch.setattr(images.socket, "create_connection", silent)
    monkeypatch.setattr(images, "_is_public", lambda ip: True)
    with images._Deadline(1):
        for public in (True, False):
            with pytest.raises(TimeoutError):
                images._connect(("rebind.test", 80), 60, public=public)
    assert len(tried) == 4 and max(tried) <= 1


def test_a_proxy_tunnel_leaves_the_handshake_only_what_is_left(server, monkeypatch):
    """A CONNECT the proxy answers late, then a server that never speaks
    TLS: the handshake gets the time the CONNECT left, not the deadline
    again."""
    routes, port = server

    def slow_tunnel(handler):
        time.sleep(1.5)
        handler.send_response(200)
        handler.end_headers()
        time.sleep(3)  # silent

    routes["CONNECT"] = slow_tunnel
    for proxy in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(proxy, raising=False)
        monkeypatch.delenv(proxy.upper(), raising=False)
    monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("ATOM_IMAGE_URL_ALLOW_PRIVATE", "1")
    monkeypatch.setattr(images, "DOWNLOAD_SECONDS", 2)
    images._trusting_opener.cache_clear()
    started = time.monotonic()
    try:
        with pytest.raises(ValueError, match="within 2 s"):
            _load(_image("https://images.invalid/a.png"))
    finally:
        images._trusting_opener.cache_clear()
    assert time.monotonic() - started < 2.8


def test_allowed_urls_go_through_the_system_proxy(server, monkeypatch):
    """The local server stands in for the proxy; the host does not resolve."""
    routes, port = server
    routes["http://images.invalid/a.png"] = _serve(_encoded())  # as proxied
    for proxy in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(proxy, raising=False)
        monkeypatch.delenv(proxy.upper(), raising=False)
    monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("ATOM_IMAGE_URL_ALLOW_PRIVATE", "1")
    images._trusting_opener.cache_clear()
    try:
        [image] = _load(_image("http://images.invalid/a.png"))
    finally:
        images._trusting_opener.cache_clear()
    assert image.size == (4, 3)


# The Chat API: checked on arrival, decoded before the processor.


def test_a_file_part_reaches_the_image_path(monkeypatch):
    """The route goes on with the checked messages: the request's own would
    hold a `file` part, which the text path takes."""
    seen = []

    def prepare(messages, *_args):
        seen.extend(part["type"] for part in messages[0].content)
        raise ValueError("reached the image path")

    def text_path(*_args, **_kwargs):
        raise AssertionError("the text path ran")

    monkeypatch.setattr(api_server, "model_name", "m")
    monkeypatch.setattr(api_server, "_request_logger", None)
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", {})
    monkeypatch.setattr(api_server, "custom_message_encoder", None)
    monkeypatch.setattr(api_server, "_get_multimodal_processor", lambda: object())
    monkeypatch.setattr(api_server, "_prepare_multimodal_inputs", prepare)
    monkeypatch.setattr(api_server, "apply_chat_template", text_path)
    file_part = {"type": "file", "file": {"file_data": PNG, "filename": "a.png"}}
    request = ChatCompletionRequest(model="m", messages=[_user(file_part)])
    with pytest.raises(HTTPException) as reached:
        asyncio.run(api_server.chat_completions(request, None))
    assert reached.value.detail == "reached the image path"
    assert seen == ["image_url", "text"]


def test_the_chat_api_refuses_a_server_path_before_any_work(monkeypatch):
    def prepare(*_args):
        raise AssertionError("the image path ran")

    monkeypatch.setattr(api_server, "model_name", "m")
    monkeypatch.setattr(api_server, "_get_multimodal_processor", lambda: object())
    monkeypatch.setattr(api_server, "_prepare_multimodal_inputs", prepare)
    request = ChatCompletionRequest(
        model="m", messages=[_user({"type": "image", "image": "/etc/passwd"})]
    )
    with pytest.raises(HTTPException) as refused:
        asyncio.run(api_server.chat_completions(request, None))
    assert refused.value.status_code == 400
    assert "never a server path" in refused.value.detail


def test_the_processor_gets_decoded_images(monkeypatch):
    """Its media loader would read any source it is given, server paths
    included."""
    seen = {}

    def prepare(_config, _processor, _conversation, media, *_args, **_kwargs):
        seen.update(media)
        return [1], {}

    monkeypatch.setattr(api_server, "_get_engine_config", lambda: None)
    monkeypatch.setattr(api_server, "_get_multimodal_processor", lambda: None)
    monkeypatch.setattr(api_server, "prepare_multimodal_inputs", prepare)
    api_server._prepare_multimodal_inputs([_user(_image(PNG))], {})
    [image] = seen["image"]
    assert isinstance(image, Image.Image) and image.size == (4, 3)
