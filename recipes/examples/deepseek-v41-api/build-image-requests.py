"""Build Chat, Responses and Messages image requests; no extra packages needed."""

import argparse
import base64
import copy
import json
from pathlib import Path


def read_image(path):
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        media_type = "image/jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        media_type = "image/gif"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        media_type = "image/webp"
    else:
        raise ValueError(f"Unsupported image format: {path}")
    if len(data) > 32 * 1024 * 1024:
        raise ValueError(f"Image exceeds 32 MiB: {path}")
    encoded = base64.b64encode(data).decode("ascii")
    return media_type, encoded


def build_requests(images, prompt, model):
    chat_parts = [{"type": "text", "text": prompt}]
    responses_parts = [{"type": "input_text", "text": prompt}]
    messages_parts = [{"type": "text", "text": prompt}]
    for media_type, encoded in images:
        url = f"data:{media_type};base64,{encoded}"
        chat_parts.append(
            {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
        )
        responses_parts.append(
            {"type": "input_image", "image_url": url, "detail": "auto"}
        )
        messages_parts.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": encoded,
                },
            }
        )
    return {
        "chat": {
            "model": model,
            "messages": [{"role": "user", "content": chat_parts}],
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 256,
        },
        "responses": {
            "model": model,
            "input": [{"role": "user", "content": responses_parts}],
            "reasoning": {"effort": "none"},
            "temperature": 0,
            "max_output_tokens": 256,
        },
        "messages": {
            "model": model,
            "messages": [{"role": "user", "content": messages_parts}],
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 256,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    paths = [args.image] + ([args.second] if args.second else [])
    images = [read_image(path) for path in paths]
    prompt = args.prompt or (
        "Compare the first and second images. Describe each in that order."
        if args.second
        else "Describe this image briefly."
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for endpoint, body in build_requests(images, prompt, args.model).items():
        for stream in (False, True):
            request = copy.deepcopy(body)
            suffix = "-stream" if stream else ""
            if stream:
                request["stream"] = True
                if endpoint == "chat":
                    request["stream_options"] = {"include_usage": True}
            target = args.output_dir / f"{endpoint}-image{suffix}.json"
            target.write_text(json.dumps(request, indent=2) + "\n", encoding="utf8")
            print(target)


if __name__ == "__main__":
    main()
