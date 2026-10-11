#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Verify and prepare the official Windows x64 DXC runtime; never execute it."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import struct
import sys
import tempfile
from urllib.error import URLError
from urllib.request import urlopen
import zipfile

VERSION = 'v1.9.2609'
ARCHIVE_NAME = 'dxc_2026_09_29.zip'
ARCHIVE_URL = (f'https://github.com/microsoft/DirectXShaderCompiler/releases/download/'
               f'{VERSION}/{ARCHIVE_NAME}')
ARCHIVE_SHA256 = 'ad31b1fc8443175d204f77a611fdb3ef2ec42759bdc2f1167368de24a4a7e7f1'
LICENSES = ('LICENCE-MIT.txt', 'LICENSE-LLVM.txt', 'LICENSE-MS.txt')
MEMBERS = {'bin/x64/dxcompiler.dll': 'dxcompiler.dll',
           **{name: f'licenses/dxc/{name}' for name in LICENSES}}


def check_x64_dll(data: bytes) -> None:
    """Inspect fixed PE headers without loading or executing Windows code."""
    valid = False
    if len(data) >= 64 and data[:2] == b'MZ':
        offset = struct.unpack_from('<I', data, 0x3c)[0]
        if offset + 26 <= len(data) and data[offset:offset + 4] == b'PE\0\0':
            machine = struct.unpack_from('<H', data, offset + 4)[0]
            flags = struct.unpack_from('<H', data, offset + 22)[0]
            magic = struct.unpack_from('<H', data, offset + 24)[0]
            valid = machine == 0x8664 and bool(flags & 0x2000) and magic == 0x20b
    if not valid:
        raise ValueError('DXC member is not a valid x64 PE32+ DLL')


def prepare(archive: Path, output: Path) -> dict:
    """Validate the complete input before publishing whitelisted runtime files."""
    with archive.open('rb') as source:
        checksum = hashlib.file_digest(source, 'sha256').hexdigest()
        if checksum != ARCHIVE_SHA256:
            raise ValueError(f'DXC archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {checksum}')
        source.seek(0)
        with zipfile.ZipFile(source) as bundle:
            index = {}
            seen = set()
            for member in bundle.infolist():
                normalized = member.filename.replace('\\', '/')
                path = PurePosixPath(normalized)
                if (path.is_absolute() or PureWindowsPath(normalized).drive or '..' in path.parts):
                    raise ValueError(f'Unsafe DXC archive path: {member.filename}')
                name = path.as_posix()
                key = name.casefold()
                if key in seen:
                    raise ValueError(f'Duplicate DXC archive member: {name}')
                seen.add(key)
                index[name] = member
            for name in MEMBERS:
                if name not in index or index[name].is_dir():
                    raise ValueError(f'Missing required DXC archive member: {name}')
            # ZIP reads also verify CRC for every selected file.
            payload = {destination: bundle.read(index[name]) for name, destination in MEMBERS.items()}
    check_x64_dll(payload['dxcompiler.dll'])
    for name in LICENSES:
        if not payload[f'licenses/dxc/{name}'].strip():
            raise ValueError(f'Missing license text: {name}')
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    provenance = {
        'schema_version': 1,
        'version': VERSION,
        'architecture': 'x64',
        'archive_name': ARCHIVE_NAME,
        'archive_url': ARCHIVE_URL,
        'archive_sha256': checksum,
        'release_api': f'https://api.github.com/repos/microsoft/DirectXShaderCompiler/releases/tags/{VERSION}',
        'dll_sha256': hashes['dxcompiler.dll'],
        'licenses': [f'licenses/dxc/{name}' for name in LICENSES],
        'license_reference': f'https://github.com/microsoft/DirectXShaderCompiler/blob/{VERSION}/docs/ReleaseNotes.md#licenses',
        'files': hashes,
        'executed': False,
    }
    payload['dxc-provenance.json'] = (json.dumps(provenance, indent=2) + '\n').encode('utf-8')
    output = output.resolve()
    # Do not follow a pre-existing release subdirectory symlink outside output.
    for name in payload:
        if not (output / name).parent.resolve().is_relative_to(output):
            raise ValueError(f'Unsafe DXC output path: {name}')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dxc-stage-', dir=output) as temporary:
        stage = Path(temporary)
        for name, data in payload.items():
            staged = stage / name
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(data)
        # Provenance is published last; each replacement is atomic on this volume.
        for name in payload:
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            (stage / name).replace(destination)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Release executable directory receiving dxcompiler.dll and notices')
    parser.add_argument('--archive', type=Path,
                        help='Use a cached official ZIP (the pinned SHA-256 is still required)')
    args = parser.parse_args()
    try:
        if args.archive:
            provenance = prepare(args.archive, args.output_dir)
        else:
            with tempfile.TemporaryDirectory(prefix='acid-rain-dxc-') as temporary:
                archive = Path(temporary) / ARCHIVE_NAME
                with urlopen(ARCHIVE_URL, timeout=60) as response, archive.open('wb') as target:
                    shutil.copyfileobj(response, target, length=1024 * 1024)
                provenance = prepare(archive, args.output_dir)
    except (OSError, ValueError, URLError, zipfile.BadZipFile) as error:
        print(f'DXC preparation failed: {error}', file=sys.stderr)
        return 1
    print(f"Verified DXC {VERSION} x64: {provenance['dll_sha256']}")
    print(f"Prepared {(args.output_dir.resolve() / 'dxcompiler.dll')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
