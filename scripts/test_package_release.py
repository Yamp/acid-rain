"""Release archives must be usable, complete and verifiable after download."""
import hashlib
import json
import plistlib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).with_name("package_release.py")


class ReleasePackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="acid rain release ")
        self.root = Path(self.temp.name)
        self.repo = self.root / "source"
        self.bin = self.root / "binaries"
        self.out = self.root / "artifacts"
        (self.repo / "desktop").mkdir(parents=True)
        self.bin.mkdir()
        for file in ["README.md", "LICENSE", "desktop/README.md"]:
            (self.repo / file).write_text("Documentation " + file)
        for file in ["vendor/wgpu-core/LICENSE.MIT", "vendor/wgpu-core/LICENSE.APACHE",
                     "vendor/README.md", "vendor/wgpu-core-backport.patch",
                     "vendor/wgpu-core-provenance.json"]:
            destination = self.repo / file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SCRIPT.parent.parent / file, destination)
        for file in ["acid-rain", "acid-rain-desktop", "acid-rain.exe", "acid-rain-desktop.exe"]:
            (self.bin / file).write_bytes(b"portable program " + file.encode())
        dxc_files = ["dxcompiler.dll", "licenses/dxc/LICENCE-MIT.txt",
                     "licenses/dxc/LICENSE-LLVM.txt", "licenses/dxc/LICENSE-MS.txt"]
        for file in dxc_files:
            destination = self.bin / file
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"DXC packaging test fixture: " + file.encode())
        (self.bin / "dxc-provenance.json").write_text(json.dumps({
            "architecture": "x64", "files": {file: hashlib.sha256((self.bin / file).read_bytes()).hexdigest()
                                                for file in dxc_files}}))

    def tearDown(self):
        self.temp.cleanup()

    def run_script(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    def package(self, target, version="v0.2.0"):
        return self.run_script("package", "--target", target, "--version", version,
                               "--repo", str(self.repo), "--bin-dir", str(self.bin), "--output", str(self.out))

    def test_linux_archive_contains_executables_docs_and_valid_hashes(self):
        result = self.package("x86_64-unknown-linux-gnu")
        self.assertEqual(result.returncode, 0, result.stderr)
        archive = next(self.out.glob("*.tar.gz"))
        extracted = self.root / "extracted"
        with tarfile.open(archive) as tar:
            permissions = {Path(member.name).name: member.mode for member in tar.getmembers()}
            tar.extractall(extracted, filter="data")
        package = next(extracted.iterdir())
        for name in ["acid-rain", "acid-rain-desktop", "start-desktop.sh", "start-terminal.sh"]:
            self.assertTrue(permissions[name] & 0o111, name)
        for name in ["README.md", "LICENSE", "DESKTOP.md", "START-HERE.txt", "BUILD-INFO.json"]:
            self.assertTrue((package / name).is_file(), name)
        for line in (package / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            self.assertEqual(hashlib.sha256((package / name).read_bytes()).hexdigest(), digest)
        digest = archive.with_name(archive.name + ".sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), digest)

    def test_windows_archive_has_double_click_launchers_and_both_executables(self):
        result = self.package("x86_64-pc-windows-msvc")
        self.assertEqual(result.returncode, 0, result.stderr)
        with zipfile.ZipFile(next(self.out.glob("*.zip"))) as archive:
            contents = {Path(name).name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
        for name in ["acid-rain.exe", "acid-rain-desktop.exe", "Start Desktop.cmd", "Start Terminal.cmd", "LICENSE"]:
            self.assertIn(name, contents)
        self.assertIn(b'%~dp0', contents["Start Desktop.cmd"])

    def test_windows_dxc_is_required_and_its_provenance_hashes_are_verified(self):
        compiler = self.bin / "dxcompiler.dll"
        original = compiler.read_bytes()
        compiler.unlink()
        result = self.package("x86_64-pc-windows-msvc")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dxcompiler.dll", result.stderr)
        compiler.write_bytes(original + b"corruption")
        result = self.package("x86_64-pc-windows-msvc")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum", result.stderr)
        self.assertFalse(list(self.out.glob("*.zip")))
        compiler.write_bytes(original)
        result = self.package("x86_64-pc-windows-msvc")
        self.assertEqual(result.returncode, 0, result.stderr)
        with zipfile.ZipFile(next(self.out.glob("*.zip"))) as archive:
            prefix = "acid-rain-0.2.0-windows-x86_64/"
            manifest = dict(reversed(line.split("  ", 1)) for line in
                            archive.read(prefix + "SHA256SUMS").decode().splitlines())
            for name in ["dxcompiler.dll", "licenses/dxc/LICENCE-MIT.txt", "licenses/dxc/LICENSE-LLVM.txt",
                         "licenses/dxc/LICENSE-MS.txt", "dxc-provenance.json"]:
                content = archive.read(prefix + name)
                self.assertEqual(content, (self.bin / name).read_bytes())
                self.assertEqual(hashlib.sha256(content).hexdigest(), manifest[name])

    def test_packaged_readme_resolves_source_links_at_release_tag(self):
        (self.repo / "docs/images").mkdir(parents=True)
        (self.repo / "native").mkdir()
        (self.repo / "native/README.md").write_text("Native guide")
        (self.repo / "docs/images/desktop.png").write_bytes(b"image fixture")
        source = ('[Desktop](desktop/README.md) [Metal](native/README.md#launch)\n'
                  '![GPU image](docs/images/desktop.png)\n'
                  '[External](https://example.com/guide) [Here](#controls)\n'
                  '```md\n[example](not-a-real-file.md)\n```\n')
        (self.repo / "README.md").write_text(source)
        result = self.package("x86_64-pc-windows-msvc")
        self.assertEqual(result.returncode, 0, result.stderr)
        with zipfile.ZipFile(next(self.out.glob("*.zip"))) as archive:
            name = next(name for name in archive.namelist() if name.endswith("/README.md"))
            readme = archive.read(name).decode()
        self.assertIn('[Desktop](https://github.com/Yamp/acid-rain/blob/v0.2.0/desktop/README.md)', readme)
        self.assertIn('[Metal](https://github.com/Yamp/acid-rain/blob/v0.2.0/native/README.md#launch)', readme)
        self.assertIn('![GPU image](https://raw.githubusercontent.com/Yamp/acid-rain/v0.2.0/docs/images/desktop.png)', readme)
        self.assertIn('[External](https://example.com/guide) [Here](#controls)', readme)
        self.assertIn('```md\n[example](not-a-real-file.md)\n```', readme)
        self.assertEqual((self.repo / "README.md").read_text(), source)

    def test_missing_binary_and_unsafe_version_fail_without_archive(self):
        (self.bin / "acid-rain-desktop").unlink()
        self.assertNotEqual(self.package("x86_64-unknown-linux-gnu").returncode, 0)
        self.assertNotEqual(self.package("x86_64-pc-windows-msvc", "../../escape").returncode, 0)
        self.assertFalse(list(self.out.glob("*.zip")) + list(self.out.glob("*.tar.gz")))

    def test_aggregate_manifest_rejects_corrupt_archive(self):
        self.assertEqual(self.package("x86_64-unknown-linux-gnu").returncode, 0)
        result = self.run_script("manifest", "--directory", str(self.out))
        self.assertEqual(result.returncode, 0, result.stderr)
        archive = next(self.out.glob("*.tar.gz"))
        archive.write_bytes(archive.read_bytes() + b"corrupted download")
        self.assertNotEqual(self.run_script("manifest", "--directory", str(self.out)).returncode, 0)

    def test_vendor_notices_are_required_preserved_and_checksummed(self):
        readme = self.repo / "vendor/README.md"
        readme.write_bytes(readme.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        license = self.repo / "vendor/wgpu-core/LICENSE.MIT"
        license_bytes = license.read_bytes()
        license.unlink()
        result = self.package("x86_64-pc-windows-msvc")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LICENSE.MIT", result.stderr)
        self.assertFalse(list(self.out.glob("*.zip")))
        license.write_bytes(license_bytes)
        result = self.package("x86_64-pc-windows-msvc")
        self.assertEqual(result.returncode, 0, result.stderr)
        with zipfile.ZipFile(next(self.out.glob("*.zip"))) as archive:
            prefix = "acid-rain-0.2.0-windows-x86_64/"
            manifest = dict(reversed(line.split("  ", 1)) for line in
                            archive.read(prefix + "SHA256SUMS").decode().splitlines())
            for source, name in [("vendor/wgpu-core/LICENSE.MIT", "LICENSE.MIT"),
                                 ("vendor/wgpu-core/LICENSE.APACHE", "LICENSE.APACHE"),
                                 ("vendor/README.md", "README.md"),
                                 ("vendor/wgpu-core-backport.patch", "wgpu-core-backport.patch"),
                                 ("vendor/wgpu-core-provenance.json", "wgpu-core-provenance.json")]:
                relative = "third-party-notices/wgpu-core/" + name
                content = archive.read(prefix + relative)
                if name == "README.md":
                    self.assertTrue(content.endswith((self.repo / source).read_bytes()))
                    self.assertIn(b'https://github.com/Yamp/acid-rain/tree/v0.2.0/vendor/wgpu-core', content)
                else:
                    self.assertEqual(content, (self.repo / source).read_bytes())
                self.assertEqual(hashlib.sha256(content).hexdigest(), manifest[relative])

    @unittest.skipUnless(sys.platform == "darwin", "macOS bundle signing needs codesign")
    def test_macos_bundle_signature_survives_archiving(self):
        # Use a real harmless Mach-O fixture so codesign validates executable
        # layout; it is never presented as an Acid Rain application or executed.
        shutil.copyfile("/usr/bin/true", self.bin / "acid-rain-desktop")
        result = self.package("aarch64-apple-darwin")
        self.assertEqual(result.returncode, 0, result.stderr)
        extracted = self.root / "mac-extracted"
        with tarfile.open(next(self.out.glob("*.tar.gz"))) as archive:
            archive.extractall(extracted, filter="data")
        app = next(extracted.iterdir()) / "Acid Rain Desktop.app"
        with (app / "Contents/Info.plist").open("rb") as file:
            info = plistlib.load(file)
        self.assertEqual(info["CFBundleExecutable"], "acid-rain-desktop")
        checked = subprocess.run(["codesign", "--verify", "--strict", str(app)], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == "__main__":
    unittest.main()
