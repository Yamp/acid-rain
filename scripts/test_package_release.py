"""Release archives must be usable, complete and verifiable after download."""
import hashlib
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
        for file in ["acid-rain", "acid-rain-desktop", "acid-rain.exe", "acid-rain-desktop.exe"]:
            (self.bin / file).write_bytes(b"portable program " + file.encode())

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
