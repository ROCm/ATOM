//! Regression coverage for the existing protocol maintenance tool. No network needed.
#[test]
fn vendoring_rejects_unsafe_downloads_and_preserves_sources_on_failure() {
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(r#"
import hashlib, io, json, pathlib, runpy, shutil, sys, tempfile
from unittest.mock import patch
module = runpy.run_path(sys.argv[1])
Vendor = module['ProtoVendor']
Vendor().check()
def rejected(call):
    try:
        call()
    except (ValueError, OSError):
        return
    raise AssertionError('unsafe operation accepted')
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory) / 'proto'
    root.mkdir()
    vendor = Vendor(root)
    for name in ['../out', '/absolute', 'envoy/../../out', 'google/protobuf/../out', 'envoy//x', 'envoy/%2e%2e/x', 'envoy/a\\b']:
        rejected(lambda: vendor.proto(name))
    outside = pathlib.Path(directory) / 'outside'
    outside.mkdir()
    (root / 'envoy').symlink_to(outside, target_is_directory=True)
    rejected(lambda: vendor.target(root, 'envoy/out.proto'))
    (root / 'envoy').unlink()
    with patch('urllib.request.urlopen', return_value=io.BytesIO(b'x' * (vendor.MAX_FILE_BYTES + 1))):
        rejected(lambda: vendor.download('https://unused'))
    # Optional import modifiers still participate in recursive validation.
    vendor.download = lambda url: b'import public "envoy/../../out";'
    rejected(lambda: vendor.proto('envoy/a.proto'))
    vendor = Vendor(root)
    files = {'a.proto': b'original-a', 'b.proto': b'original-b'}
    for name, content in files.items():
        (root / name).write_bytes(content)
    manifest = {'revisions': Vendor.REVISIONS, 'files': {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    vendor.downloaded = {'a.proto': b'untrusted'}
    rejected(vendor.install)
    assert (root / 'a.proto').read_bytes() == files['a.proto']
    # A later install failure restores already replaced files.
    vendor.downloaded = {'a.proto': b'new-a', 'b.proto': b'new-b'}
    replace = module['os'].replace
    def fail_second(src, dst):
        if str(dst).endswith('b.proto'):
            raise OSError('injected install failure')
        replace(src, dst)
    with patch('os.replace', side_effect=fail_second):
        rejected(lambda: vendor.install(update_manifest=True))
    for name, content in files.items():
        assert (root / name).read_bytes() == content
    assert json.loads((root / 'manifest.json').read_text()) == manifest
    vendor = Vendor(root)
    def failed_download(url):
        raise OSError('injected network failure')
    vendor.download = failed_download
    rejected(vendor.run)
    vendor.check()
"#)
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/proto/vendor_ext_proc.py"))
        .output().expect("python3 is required for the protocol maintenance regression");
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
