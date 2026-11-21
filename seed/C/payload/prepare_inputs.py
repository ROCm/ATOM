"""CPU-only tokenizer/data preparation; no model weights, GPU or evaluator run."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

from evidence import read, sha, write


def load_protocol(source):
    spec = importlib.util.spec_from_file_location("native_protocol", source / "atom/entrypoints/openai/protocol.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert all(os.environ.get(k) == "" for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"))
    assert all(os.environ.get(k) == "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"))
    pins = read(Path(__file__).with_name("pins.json"))
    composition = Path(__file__).with_name("composition-manifest.json")
    assert sha(composition) == pins["model_manifest"]
    target = next(v for v in read(composition)["views"] if v["role"] == "target")
    for item in target["files"]:
        if item.get("actual_sha256"):
            assert sha(args.model / item["name"]) == item["actual_sha256"], item["name"]
    assert sha(args.source / "atom/entrypoints/openai/protocol.py") == pins["source_files"]["candidate"]["atom/entrypoints/openai/protocol.py"]
    for item in (pins["data"]["train"], pins["data"]["test"]):
        assert sha(args.data / item["file"]) == item["sha256"]
    import pyarrow.parquet as pq
    train = pq.read_table(args.data / "train.parquet").to_pylist()
    test = pq.read_table(args.data / "test.parquet").to_pylist()
    assert len(train) == 7473 and len(test) == 1319
    assert all(set(row) == {"question", "answer"} and all(isinstance(v, str) for v in row.values())
               for row in train + test)
    args.out.mkdir()  # new directory only; preparation failures are never silently reused
    write(args.out / "population.json", [{"doc_id": i, "doc": row} for i, row in enumerate(test)])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    tokens = tokenizer.encode("Explain how to verify a numerical calculation. " * 4096,
                              add_special_tokens=False)
    assert len(tokens) >= 8193
    protocol = load_protocol(args.source)
    fields = protocol.CompletionRequest.model_fields
    for n in (300, 512, 513, 8192, 8193):
        for output in ((1, 32) if n <= 513 else (1,)):
            request = {"model": "/input/model-target", "prompt_token_ids": tokens[:n],
                       "max_tokens": output, "temperature": 0, "ignore_eos": True,
                       "return_token_ids": True}
            assert set(request) <= set(fields), "a request field would be silently ignored"
            parsed = protocol.CompletionRequest.model_validate(request)
            assert parsed.get_prompt_or_tokens() == tokens[:n] and parsed.get_max_tokens() == output
            write(args.out / f"{n}-{output}.json", request)
    write(args.out / "manifest.json", {"asset_manifest_sha256": pins["model_manifest"],
          "generator_sha256": sha(__file__), "task_data_revision": pins["data"]["revision"],
          "population": {"train": 7473, "test": 1319},
          "files": {p.name: sha(p) for p in sorted(args.out.iterdir()) if p.is_file()},
          "scope": "exact input IDs and dataset rows; generated-token IDs are not exposed by the native completion response"})


if __name__ == "__main__":
    main()
