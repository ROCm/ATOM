#!/usr/bin/env python3
"""Refresh pinned protocol sources; --check verifies the vendored files offline.

An intentional version update changes REVISIONS and runs --update-manifest.
Review the resulting protocol and checksum changes together.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import ClassVar


class ProtoVendor:
    REVISIONS: ClassVar[dict] = {
        "envoy": "6d9bb7d9a85d616b220d1f8fe67b61f82bbdb8d3",  # v1.37.0
        "xds": "8bfbf64dc13ee1a570be4fbdcfccbdd8532463f0",
        "pgv": "4eb9011f3e6d551d067d87c89f082261164fac31",  # v1.3.0
        "grpc": "5e6ba94242b92e363220bc2163d55ce3554d4ecc",  # v1.78.0
    }
    REPOSITORIES: ClassVar[dict] = {
        "envoy": "envoyproxy/envoy",
        "xds": "cncf/xds",
        "pgv": "bufbuild/protoc-gen-validate",
        "grpc": "grpc/grpc",
    }
    MAX_FILE_BYTES = 4 * 1024 * 1024
    MAX_FILES = 256
    MANIFEST = "manifest.json"

    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).parent).resolve()
        self.downloaded = {}

    @staticmethod
    def validate_name(name):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", name):
            raise ValueError("Invalid dependency path: " + str(name))
        if any(part in ("", ".", "..") for part in name.split("/")):
            raise ValueError("Invalid dependency path: " + name)
        return name

    def target(self, root, name):
        self.validate_name(name)
        root = Path(root).resolve()
        target = root / name
        # Resolve existing symlinks before either reading or installing files.
        try:
            target.resolve().relative_to(root)
        except ValueError:
            raise ValueError("Dependency escapes output directory: " + name)
        return target

    def url(self, source, path):
        self.validate_name(path)
        return (
            f"https://raw.githubusercontent.com/"
            f"{self.REPOSITORIES[source]}/{self.REVISIONS[source]}/{path}"
        )

    def download(self, url):
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(self.MAX_FILE_BYTES + 1)
        if len(data) > self.MAX_FILE_BYTES:
            raise ValueError("Protocol file exceeds download limit")
        return data

    def fetch(self, name, source, path):
        self.target(self.root, name)
        if len(self.downloaded) >= self.MAX_FILES:
            raise ValueError("Protocol dependency count exceeds limit")
        data = self.download(self.url(source, path))
        self.downloaded[name] = data
        return data.decode("utf-8")

    def proto(self, name):
        self.validate_name(name)  # Validate even skipped built-in imports.
        if name in self.downloaded or name.startswith("google/protobuf/"):
            return
        if name.startswith("envoy/"):
            source, path = "envoy", "api/" + name
        elif name.startswith(("xds/", "udpa/")):
            source, path = "xds", name
        elif name.startswith("validate/"):
            source, path = "pgv", name
        else:
            raise ValueError("Unknown proto dependency: " + name)
        content = self.fetch(name, source, path)
        for dependency in re.findall(
            r'^\s*import\s+(?:(?:public|weak)\s+)?"([^"]+)"\s*;', content, re.MULTILINE
        ):
            self.proto(dependency)

    def manifest(self):
        manifest = json.loads(self.target(self.root, self.MANIFEST).read_text())
        if manifest["revisions"] != self.REVISIONS:
            raise ValueError("Pinned revisions differ from manifest")
        for name, digest in manifest["files"].items():
            self.target(self.root, name)
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid SHA-256 checksum: " + name)
        return manifest

    def check(self):
        for name, expected in self.manifest()["files"].items():
            actual = hashlib.sha256(
                self.target(self.root, name).read_bytes()
            ).hexdigest()
            if actual != expected:
                raise ValueError("Checksum mismatch: " + name)

    def install(self, update_manifest=False):
        hashes = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in self.downloaded.items()
        }
        if not update_manifest and hashes != self.manifest()["files"]:
            raise ValueError("Downloaded sources differ from checksum manifest")
        files = dict(self.downloaded)
        if update_manifest:
            files[self.MANIFEST] = (
                json.dumps(
                    {"revisions": self.REVISIONS, "files": hashes},
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode()
        # Finish all downloads, dependency traversal and hash checks before touching sources.
        with tempfile.TemporaryDirectory(
            prefix="ext-proc-proto-", dir=self.root.parent
        ) as temporary:
            temporary = Path(temporary)
            originals = {}
            installed = []
            for name, data in files.items():
                target = self.target(self.root, name)
                originals[name] = target.read_bytes() if target.exists() else None
                staged = self.target(temporary, name)
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(data)
            try:
                for name in files:
                    target = self.target(self.root, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(str(self.target(temporary, name)), str(target))
                    installed.append(name)
            except Exception:
                for name in reversed(installed):
                    target = self.target(self.root, name)
                    if originals[name] is None:
                        target.unlink()
                    else:
                        target.write_bytes(originals[name])
                raise

    def run(self, update_manifest=False):
        if not update_manifest:
            self.manifest()  # Reject an invalid manifest before network access.
        self.proto("envoy/service/ext_proc/v3/external_processor.proto")
        self.fetch(
            "grpc/health/v1/health.proto",
            "grpc",
            "src/proto/grpc/health/v1/health.proto",
        )
        for source in self.REVISIONS:
            self.fetch("licenses/" + source + ".txt", source, "LICENSE")
        self.install(update_manifest)
        print("Vendored", len(self.downloaded), "files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--update-manifest", action="store_true")
    args = parser.parse_args()
    vendor = ProtoVendor()
    if args.check:
        vendor.check()
    else:
        vendor.run(args.update_manifest)
