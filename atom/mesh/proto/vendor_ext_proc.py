#!/usr/bin/env python3
"""Vendor the official Envoy ext-proc import closure at fixed upstream revisions."""

from pathlib import Path
import re
import urllib.request


class ProtoVendor:
    ENVOY = "https://raw.githubusercontent.com/envoyproxy/envoy/v1.37.0/"
    XDS = "https://raw.githubusercontent.com/cncf/xds/8bfbf64dc13ee1a570be4fbdcfccbdd8532463f0/"
    PGV = "https://raw.githubusercontent.com/bufbuild/protoc-gen-validate/v1.3.0/"
    GRPC = "https://raw.githubusercontent.com/grpc/grpc/v1.78.0/"

    def __init__(self):
        self.root = Path(__file__).resolve().parent
        self.downloaded = set()

    def fetch(self, name, url):
        data = urllib.request.urlopen(url, timeout=30).read()
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.downloaded.add(name)
        return data.decode()

    def proto(self, name):
        if name in self.downloaded or name.startswith("google/protobuf/"):
            return
        if name.startswith("envoy/"):
            base = self.ENVOY + "api/"
        elif name.startswith(("xds/", "udpa/")):
            base = self.XDS
        elif name.startswith("validate/"):
            base = self.PGV
        else:
            raise ValueError("Unknown proto dependency: " + name)
        content = self.fetch(name, base + name)
        for dependency in re.findall(r'^import\s+(?:public\s+)?"([^"]+)";', content, re.M):
            self.proto(dependency)

    def run(self):
        self.proto("envoy/service/ext_proc/v3/external_processor.proto")
        self.fetch("grpc/health/v1/health.proto", self.GRPC + "src/proto/grpc/health/v1/health.proto")
        self.fetch("licenses/grpc.txt", self.GRPC + "LICENSE")
        for name, base in (("envoy", self.ENVOY), ("xds", self.XDS), ("pgv", self.PGV)):
            self.fetch("licenses/" + name + ".txt", base + "LICENSE")
        print("Vendored", len(self.downloaded), "files")


if __name__ == "__main__":
    ProtoVendor().run()
