"""Read-only weight visibility check; never allocates GPUs or downloads weights."""

import argparse
import hashlib
import json
from pathlib import Path


def check_weights(model, model_root=None, manifest=None):
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
        if manifest is not None:
            report["identity_manifest"] = str(manifest)
            expected = json.loads(manifest.read_text())
            report["index_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
            for key in ("config_sha256", "index_sha256"):
                if report[key] != expected[key]:
                    raise ValueError(f"Checkpoint {key} mismatch")
            if report["weights"] != expected["shards"]:
                raise ValueError("Checkpoint shard names/sizes mismatch")
            report["checkpoint_identity"] = "STRUCTURE_MATCH_NOT_FULL_WEIGHT_HASH"
        report["status"] = "FILES_VISIBLE"
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        report["error"] = repr(exc)
    return report


def check_draft_weights(model):
    """Inspect local draft metadata/structure, never load a model or tokenizer."""
    report = {
        "model_path": str(model),
        "status": "BLOCKED_ENV",
        # Frozen speculative.py uses the target tokenizer for non-heterogeneous drafts.
        "tokenizer": "TARGET_TOKENIZER_REUSED",
    }
    try:
        config_bytes = (model / "config.json").read_bytes()
        metadata = json.loads(config_bytes)
        architectures = metadata["architectures"]
        if (
            not isinstance(architectures, list)
            or not architectures
            or not all(isinstance(arch, str) and arch.strip() for arch in architectures)
        ):
            raise ValueError("Missing or invalid draft architectures")
        report["architectures"] = architectures
        report["quantization_config"] = metadata.get("text_config", {}).get(
            "quantization_config"
        ) or metadata.get("quantization_config")
        report["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        index = model / "model.safetensors.index.json"
        if index.exists():
            index_bytes = index.read_bytes()
            weight_map = json.loads(index_bytes)["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("Empty or invalid draft checkpoint index")
            names = list(weight_map.values())
            report["index_sha256"] = hashlib.sha256(index_bytes).hexdigest()
        else:
            names = ["model.safetensors"]
        for name in names:
            if (
                not isinstance(name, str)
                or not name.endswith(".safetensors")
                or Path(name).is_absolute()
                or ".." in Path(name).parts
            ):
                raise ValueError(f"Invalid draft shard name: {name!r}")
        report["weights"] = {}
        for name in sorted(set(names)):
            shard = model / name
            size = shard.stat().st_size
            report["weights"][name] = size
            with shard.open("rb") as weight:
                prefix = weight.read(8)
                header_size = int.from_bytes(prefix, "little")
                if len(prefix) != 8 or not 0 < header_size <= min(
                    size - 8, 100_000_000
                ):
                    raise ValueError(f"Empty or corrupt draft shard: {name}")
                header = json.loads(weight.read(header_size))
                if not isinstance(header, dict):
                    raise TypeError(f"Invalid safetensors header: {name}")
                offsets = sorted(
                    tensor["data_offsets"]
                    for key, tensor in header.items()
                    if key != "__metadata__"
                )
                end = 0
                for start, stop in offsets:
                    if (
                        type(start) is not int
                        or type(stop) is not int
                        or start != end
                        or stop < start
                    ):
                        raise ValueError(f"Invalid safetensors offsets: {name}")
                    end = stop
                if not offsets or end == 0 or end != size - 8 - header_size:
                    raise ValueError(f"Empty or truncated draft tensor data: {name}")
                weight.seek(-1, 2)
                if not weight.read(1):
                    raise ValueError(f"Unreadable draft shard: {name}")
        report["checkpoint_identity"] = "STRUCTURE_INSPECTED_NOT_IDENTITY_VERIFIED"
        report["status"] = "FILES_VISIBLE"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        report["error"] = repr(exc)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Require exact metadata/index hashes and shard sizes",
    )
    parser.add_argument(
        "--draft-model",
        type=Path,
        help="Local draft checkpoint; reuses target tokenizer",
    )
    args = parser.parse_args()
    report = check_weights(args.model, args.model_root, args.manifest)
    if args.draft_model is not None:
        report["draft"] = check_draft_weights(args.draft_model)
        if (
            report["status"] == "FILES_VISIBLE"
            and report["draft"]["status"] != "FILES_VISIBLE"
        ):
            report["status"] = "BLOCKED_ENV"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["status"] == "FILES_VISIBLE" else 2)
