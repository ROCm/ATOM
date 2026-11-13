# API image input

Chat Completions (`/v1/chat/completions` and `/beta/chat/completions`) takes
images as content parts of `user` messages and loads them itself, before the
model's processor sees them (`atom/entrypoints/openai/images.py`). What a
client may send follows DeepSeek's
[vision contract](https://api-docs.deepseek.com/guides/vision/), which agrees
with OpenAI's where the two overlap. An image part below that breaks these
rules is HTTP 400 before generation.

| Part | Source |
|---|---|
| `{"type": "image_url", "image_url": {"url": ..., "detail": ...}}` | a base64 data URL, or an http(s) URL |
| `{"type": "file", "file": {"file_data": ...}}`, or `file_data` on the part | a base64 data URL |
| `{"type": "image", "image": ...}`, as some processors spell it | as for `image_url` |

- **Sources.** A base64 data URL, or a public http(s) URL of at most 8192
  characters, downloaded within 60 s; never a server path or `file://`. There
  is no Files API, so a `file_id` is refused.
- **Formats.** JPEG, PNG, GIF (its first frame) and WebP, recognised from the
  bytes rather than the type a data URL declares or a server sends. The image
  keeps its own mode; color conversion is the processor's.
- **Size.** An image is at most 32 MiB, base64 or fetched, and a request's
  images at most 64 MiB together. A side is at most 8192 px, or 4096 px from
  15 images a request on, and a request carries at most 600 images.
  Beyond DeepSeek's limits, a request's images hold at most 256 megapixels
  together once decoded, as their headers say before any is decoded: images
  that compress well would otherwise fit the byte limits and decode to tens
  of GiB.
- **`detail`.** `low` downscales the image to fit 512x512; `high`, `original`
  and `auto` keep it as it is.
- **Roles.** Images belong to `user` messages; in any other message they are
  refused.

Whether a deployment serves images at all is the model's to say:
DeepSeek-V4.1 refuses them under DSpark and DP attention.

## Responses and Messages

`/v1/responses` and `/v1/messages` take images where DeepSeek's contract has
them, translate them to the parts above, and check and load them the same
way.

| Endpoint | Image | Where |
|---|---|---|
| `/v1/responses` | `{"type": "input_image", "image_url": ..., "detail": ...}` | `user` and `developer` messages, and the `output` of `function_call_output` and `custom_tool_call_output` |
| `/v1/messages` | `{"type": "image", "source": {"type": "base64", "media_type": ..., "data": ...}}` or `{"source": {"type": "url", "url": ...}}` | `user` messages, and the `content` of `tool_result` |

A `file_id` and a `file` source are refused, as there is no Files API, and so
is an image anywhere else (`system`, `assistant`), or with `kv_transfer_params`:
a PD prefill cannot hand an image request off. A request without images
renders as before.

## Public URLs

A URL is fetched only from a host that resolves to global addresses, and so
is every redirect. The download connects to the address that was checked, so
DNS cannot answer differently the second time. Loopback, private, link-local
and other internal addresses are refused, and no HTTP proxy is used: a proxy
would connect on the client's behalf. Redirects stay on http(s).

Set `ATOM_IMAGE_URL_ALLOW_PRIVATE=1` where clients are trusted with the
internal network, for an intranet image store or egress only through a
proxy. Downloads then go to any address, through the system proxy if one is
set.

These rules are the server's. The offline engine (`LLM.generate`) still reads
what it is given, server paths included.
