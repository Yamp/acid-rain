"""Pinned DXC preparation must validate before touching release inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

SCRIPT = Path(__file__).with_name('prepare_dxc.py')
LICENSES = ('LICENCE-MIT.txt', 'LICENSE-LLVM.txt', 'LICENSE-MS.txt')


def fixture_dll(machine=0x8664):
    data = bytearray(512)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 0x3c, 0x80)
    data[0x80:0x84] = b'PE\0\0'
    struct.pack_into('<H', data, 0x84, machine)
    struct.pack_into('<H', data, 0x96, 0x2000)
    struct.pack_into('<H', data, 0x98, 0x20b)
    return bytes(data)


class PrepareDxcTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), 'Pinned DXC preparation script is missing')
        spec = importlib.util.spec_from_file_location('prepare_dxc', SCRIPT)
        self.prep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.prep)
        self.temp = tempfile.TemporaryDirectory(prefix='acid rain dxc ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'release bin'
        self.archive = self.root / 'compiler.zip'

    def write_archive(self, *, omit=(), extras=(), machine=0x8664):
        entries = [('bin\\x64\\dxcompiler.dll', fixture_dll(machine)),
                   ('bin\\x64\\dxil.dll', b'optional validator not distributed'),
                   ('bin\\x64\\dxc.exe', b'never extract or execute'),
                   ('bin\\arm64\\dxcompiler.dll', b'wrong architecture'),
                   *((name, ('original ' + name).encode()) for name in LICENSES)]
        with zipfile.ZipFile(self.archive, 'w') as archive:
            for name, data in [*entries, *extras]:
                if name not in omit:
                    archive.writestr(name, data)
        return hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def prepare_fixture(self, checksum):
        with mock.patch.object(self.prep, 'ARCHIVE_SHA256', checksum):
            return self.prep.prepare(self.archive, self.output)

    def test_official_release_and_hash_are_pinned_without_cli_override(self):
        self.assertEqual(self.prep.ARCHIVE_SHA256,
                         'ad31b1fc8443175d204f77a611fdb3ef2ec42759bdc2f1167368de24a4a7e7f1')
        self.assertEqual(self.prep.ARCHIVE_URL,
                         'https://github.com/microsoft/DirectXShaderCompiler/releases/download/'
                         'v1.9.2609/dxc_2026_09_29.zip')
        result = subprocess.run([sys.executable, str(SCRIPT), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('--sha256', result.stdout)
        self.assertNotIn('--url', result.stdout)

    def test_installs_only_x64_compiler_notices_and_checksummed_provenance(self):
        checksum = self.write_archive()
        self.prepare_fixture(checksum)
        expected = {'dxcompiler.dll', 'dxc-provenance.json',
                    *(f'licenses/dxc/{name}' for name in LICENSES)}
        self.assertEqual({p.relative_to(self.output).as_posix()
                          for p in self.output.rglob('*') if p.is_file()}, expected)
        self.assertEqual((self.output / 'dxcompiler.dll').read_bytes(), fixture_dll())
        provenance = json.loads((self.output / 'dxc-provenance.json').read_text())
        self.assertEqual(provenance['archive_sha256'], checksum)
        self.assertEqual(provenance['archive_url'], self.prep.ARCHIVE_URL)
        self.assertEqual(provenance['architecture'], 'x64')
        for name in LICENSES:
            self.assertEqual((self.output / 'licenses/dxc' / name).read_bytes(),
                             ('original ' + name).encode())
        self.assertEqual(set(provenance['files']), expected - {'dxc-provenance.json'})
        for name, digest in provenance['files'].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), digest)

    def test_bad_hash_fails_before_overwriting_existing_output(self):
        checksum = self.write_archive()
        self.archive.write_bytes(self.archive.read_bytes() + b'damaged download')
        self.output.mkdir()
        (self.output / 'dxcompiler.dll').write_bytes(b'existing compiler')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.prepare_fixture(checksum)
        self.assertEqual((self.output / 'dxcompiler.dll').read_bytes(), b'existing compiler')
        self.assertFalse((self.output / 'dxc-provenance.json').exists())

    def test_missing_compiler_or_notice_fails_before_creating_output(self):
        for missing in ('bin\\x64\\dxcompiler.dll', *LICENSES):
            with self.subTest(missing=missing):
                checksum = self.write_archive(omit=(missing,))
                with self.assertRaisesRegex(ValueError, 'Missing'):
                    self.prepare_fixture(checksum)
                self.assertFalse(self.output.exists())

    def test_wrong_architecture_fails_before_creating_output(self):
        checksum = self.write_archive(machine=0xaa64)
        with self.assertRaisesRegex(ValueError, 'x64'):
            self.prepare_fixture(checksum)
        self.assertFalse(self.output.exists())

    def test_unsafe_zip_paths_cannot_escape_output(self):
        for path in ('../escape.dll', '/absolute.dll', 'C:\\escape.dll'):
            with self.subTest(path=path):
                checksum = self.write_archive(extras=((path, b'invalid member'),))
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    self.prepare_fixture(checksum)
                self.assertFalse(self.output.exists())
                self.assertFalse((self.root / 'escape.dll').exists())

    def test_duplicate_normalized_member_is_rejected(self):
        checksum = self.write_archive(extras=(('bin/x64/dxcompiler.dll', fixture_dll()),))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.prepare_fixture(checksum)
        self.assertFalse(self.output.exists())

    def test_cached_cli_rejects_nonofficial_hash_without_network(self):
        self.write_archive()
        result = subprocess.run([sys.executable, str(SCRIPT), '--archive', str(self.archive),
                                 '--output-dir', str(self.output)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA-256', result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
