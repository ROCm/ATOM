"""Read-only weight visibility check; never allocates GPUs or downloads weights."""

import argparse
import hashlib
import json
from pathlib import Path


def check_weights(model, model_root=None):
    report = {"model_path": str(model), "status": "BLOCKED_ENV"}
    if model_root is not None and not model_root.is_dir():
        return {
            **report,
            "status": "UNKNOWN",
            "error": "Model root not mounted on runner",
        }
    try:
        config = model / "config.json"
        metadata = json.loads(config.read_text())
        report["architectures"] = metadata.get("architectures", [])
        report["quantization_config"] = metadata.get("text_config", {}).get(
            "quantization_config"
        ) or metadata.get("quantization_config")
        report["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        index = model / "model.safetensors.index.json"
        names = (
            sorted(set(json.loads(index.read_text())["weight_map"].values()))
            if index.exists()
            else ["model.safetensors"]
        )
        report["weights"] = {name: (model / name).stat().st_size for name in names}
        assert names and all(
            report["weights"].values()
        ), "Empty checkpoint index or shard"
        for name in names:
            with (model / name).open("rb") as weight:
                assert weight.read(8), name
        assert (
            model / "tokenizer_config.json"
        ).is_file(), "Missing tokenizer configuration"
        tokenizer_config = json.loads((model / "tokenizer_config.json").read_text())
        tokenizer_names = [
            name
            for name in ("tokenizer.json", "tokenizer.model")
            if (model / name).is_file()
        ]
        if not tokenizer_names:
            auto = tokenizer_config.get("auto_map", {}).get("AutoTokenizer", [])
            assert "tokenization_kimi.TikTokenTokenizer" in auto, "Missing tokenizer"
            # Verified K3 local bundle; inspect bytes only, never execute remote code.
            tokenizer_names = [
                "tiktoken.model",
                "tokenization_kimi.py",
                "encoding_k3.py",
            ]
        for name in tokenizer_names:
            with (model / name).open("rb") as tokenizer_file:
                assert tokenizer_file.read(8), f"Empty tokenizer file: {name}"
        report["tokenizer_files"] = tokenizer_names
        report["status"] = "FILES_VISIBLE"
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        report["error"] = repr(exc)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args()
    report = check_weights(args.model, args.model_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["status"] == "FILES_VISIBLE" else 2)
