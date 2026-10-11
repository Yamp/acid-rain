#!/usr/bin/env python3
"""Build portable release archives and verify their SHA-256 manifests (Python 3.14)."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from urllib.parse import quote, unquote, urlsplit, urlunsplit

TARGETS = {
    "x86_64-unknown-linux-gnu": "linux-x86_64",
    "x86_64-pc-windows-msvc": "windows-x86_64",
    "aarch64-apple-darwin": "macos-arm64",
    "x86_64-apple-darwin": "macos-x86_64",
}

VENDOR_NOTICES = {
    "vendor/wgpu-core/LICENSE.MIT": "LICENSE.MIT",
    "vendor/wgpu-core/LICENSE.APACHE": "LICENSE.APACHE",
    "vendor/README.md": "README.md",
    "vendor/wgpu-core-backport.patch": "wgpu-core-backport.patch",
    "vendor/wgpu-core-provenance.json": "wgpu-core-provenance.json",
}

DXC_FILES = ["dxcompiler.dll", "licenses/dxc/LICENCE-MIT.txt", "licenses/dxc/LICENSE-LLVM.txt",
             "licenses/dxc/LICENSE-MS.txt"]


def digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def text(path: Path, content: str, executable: bool = False) -> None:
    path.write_text(content, encoding="utf-8", newline="")
    if executable:
        path.chmod(0o755)


def copy_executable(source: Path, destination: Path) -> None:
    shutil.copyfile(source, destination)
    destination.chmod(0o755)


def release_markdown(repo: Path, source: Path, version: str) -> str:
    """Keep archive documentation self-contained without copying source trees.

    Source-relative links point at the matching Git tag; images use raw URLs.
    Absolute links, local anchors, inline code and fenced examples are unchanged.
    """
    links = re.compile(r'(?P<label>!?\[[^\]\n]*\])\((?P<target><[^>\n]+>|[^\s()]+)(?P<title>\s+"[^"\n]*")?\)')
    def replace(match: re.Match) -> str:
        original = match["target"]
        target = original.removeprefix("<").removesuffix(">")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match[0]
        resolved = (source.parent / unquote(parsed.path)).resolve()
        if not resolved.is_relative_to(repo) or not resolved.exists():
            raise ValueError(f"Documentation link leaves the repository or is missing: {source}: {target}")
        path = quote(resolved.relative_to(repo).as_posix(), safe="/")
        if match["label"].startswith("!"):
            absolute = f"https://raw.githubusercontent.com/Yamp/acid-rain/v{version}/{path}"
        else:
            kind = "tree" if resolved.is_dir() else "blob"
            absolute = f"https://github.com/Yamp/acid-rain/{kind}/v{version}/{path}"
        destination = urlunsplit((*urlsplit(absolute)[:3], parsed.query, parsed.fragment))
        return f'{match["label"]}({destination}{match["title"] or ""})'
    output, fence = [], None
    for line in source.read_text(encoding="utf-8").splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            output.append(line)
        elif fence is not None:
            output.append(line)
        else:
            parts = re.split(r"(`+[^`]*`+)", line)
            output.append("".join(part if index % 2 else links.sub(replace, part)
                                  for index, part in enumerate(parts)))
    return "".join(output)


def package(args: argparse.Namespace) -> Path:
    version = args.version.removeprefix("v")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError("Version must be a semantic version, such as v0.2.0")
    repo, binaries, output = args.repo.resolve(), args.bin_dir.resolve(), args.output.resolve()
    windows, macos = "windows" in args.target, "apple" in args.target
    suffix = ".exe" if windows else ""
    for file in [binaries / ("acid-rain" + suffix), binaries / ("acid-rain-desktop" + suffix),
                 repo / "LICENSE", repo / "README.md", repo / "desktop/README.md",
                 *(repo / source for source in VENDOR_NOTICES)]:
        if not file.is_file():
            raise ValueError(f"Missing required release file: {file}")
    if windows:
        for relative in [*DXC_FILES, "dxc-provenance.json"]:
            if not (binaries / relative).is_file():
                raise ValueError(f"Missing required Windows compiler file: {relative}; run scripts/prepare_dxc.py first")
        provenance = json.loads((binaries / "dxc-provenance.json").read_text(encoding="utf-8"))
        if provenance.get("architecture") != "x64":
            raise ValueError("Windows release requires the x64 DXC compiler")
        for relative in DXC_FILES:
            if provenance.get("files", {}).get(relative) != digest(binaries / relative):
                raise ValueError(f"DXC provenance checksum mismatch: {relative}")
    if args.native_app and (not macos or not (args.native_app / "Contents/MacOS/AcidRain").is_file()):
        raise ValueError("--native-app requires a built macOS Acid Rain.app")
    output.mkdir(parents=True, exist_ok=True)
    name = f"acid-rain-{version}-{TARGETS[args.target]}"
    with tempfile.TemporaryDirectory(prefix="acid-rain-package-") as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        for binary in ["acid-rain", "acid-rain-desktop"]:
            copy_executable(binaries / (binary + suffix), stage / (binary + suffix))
        shutil.copyfile(repo / "LICENSE", stage / "LICENSE")
        notices = stage / "third-party-notices/wgpu-core"
        notices.mkdir(parents=True)
        for source, destination in VENDOR_NOTICES.items():
            if destination == "README.md":
                preface = ("This archive includes notices, provenance and the patch. The complete patched source is "
                           f"[available with this release](https://github.com/Yamp/acid-rain/tree/v{version}/vendor/wgpu-core). "
                           "The original source-package documentation follows.\n\n")
                (notices / destination).write_bytes(preface.encode("utf-8") + (repo / source).read_bytes())
            else:
                shutil.copyfile(repo / source, notices / destination)
        for source, destination in [("README.md", "README.md"), ("desktop/README.md", "DESKTOP.md")]:
            text(stage / destination, release_markdown(repo, repo / source, version))
        if windows:
            for relative in [*DXC_FILES, "dxc-provenance.json"]:
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(binaries / relative, destination)
            for label, binary in [("Desktop", "acid-rain-desktop"), ("Terminal", "acid-rain")]:
                launch = 'start "" ' if label == "Desktop" else ""
                text(stage / f"Start {label}.cmd", f'@echo off\r\ncd /d "%~dp0"\r\n{launch}"%~dp0{binary}.exe" %*\r\n')
        else:
            for label, binary in [("desktop", "acid-rain-desktop"), ("terminal", "acid-rain")]:
                text(stage / f"start-{label}.sh", '#!/bin/sh\nset -eu\ncd -- "$(dirname -- "$0")"\n'
                     f'exec "./{binary}" "$@"\n', executable=True)
        if macos:
            app = stage / "Acid Rain Desktop.app"
            executable = app / "Contents/MacOS"
            executable.mkdir(parents=True)
            copy_executable(binaries / "acid-rain-desktop", executable / "acid-rain-desktop")
            with (app / "Contents/Info.plist").open("wb") as file:
                plistlib.dump({"CFBundleName": "Acid Rain Desktop", "CFBundleDisplayName": "Acid Rain Desktop",
                               "CFBundleExecutable": "acid-rain-desktop", "CFBundleIdentifier": "art.j0j0.acidrain.desktop",
                               "CFBundlePackageType": "APPL", "CFBundleShortVersionString": version,
                               "CFBundleVersion": version, "LSMinimumSystemVersion": "14.0",
                               "NSHighResolutionCapable": True}, file)
            if sys.platform != "darwin":
                raise ValueError("Package macOS releases on macOS so the .app can be ad-hoc signed")
            subprocess.run(["codesign", "--force", "--sign", "-", "--timestamp=none", str(app)], check=True)
            subprocess.run(["codesign", "--verify", "--strict", str(app)], check=True)
            if args.native_app:
                shutil.copytree(args.native_app, stage / "Acid Rain Metal.app", symlinks=True)
                subprocess.run(["codesign", "--verify", "--strict", str(stage / "Acid Rain Metal.app")], check=True)
        text(stage / "START-HERE.txt", f"""Acid Rain {version} — {TARGETS[args.target]}

Graphical water: acid-rain-desktop{suffix}
Terminal edition: acid-rain{suffix}
Windows: double-click acid-rain-desktop.exe. Linux: run ./start-desktop.sh.
macOS: open Acid Rain Desktop.app; the optional Acid Rain Metal.app is the original native edition.

Press h in the graphical app for controls. Command-line options: acid-rain-desktop{suffix} --help
Windows: DirectX 12 (Vulkan can be selected with --backend vulkan).
Keep the bundled dxcompiler.dll beside the Windows desktop executable for DirectX 12.
Linux: Vulkan driver; Ubuntu 22.04 / glibc 2.35 or newer; both X11 and Wayland are supported.
macOS: Metal, macOS 14 or newer. Select the arm64 archive for Apple Silicon or x86_64 for Intel.
No Rust toolchain is required to run these binaries. GPU shaders are embedded.

See DESKTOP.md and README.md. LICENSE contains the Apache-2.0 license.
Vendored wgpu-core licenses and backport provenance are in third-party-notices/wgpu-core.
SHA256SUMS checks every regular file in this directory. The separately published
SHA256SUMS checks the downloaded archives. Release binaries are not notarized;
macOS apps are ad-hoc signed. They do not have a paid developer signature.
""")
        text(stage / "BUILD-INFO.json", json.dumps({"version": version, "target": args.target,
             "commit": os.environ.get("GITHUB_SHA"), "native_metal_included": bool(args.native_app),
             "portable_build_requested": os.environ.get("ACID_RAIN_PORTABLE") == "1"}, indent=2) + "\n")
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        for path in files:
            if not path.resolve().is_relative_to(stage.resolve()):
                raise ValueError(f"Release contains a symlink escaping its directory: {path}")
        text(stage / "SHA256SUMS", "".join(f"{digest(path)}  {path.relative_to(stage).as_posix()}\n" for path in files))
        archive = output / (name + (".zip" if windows else ".tar.gz"))
        if windows:
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as file:
                for path in sorted(stage.rglob("*")):
                    if path.is_file():
                        file.write(path, path.relative_to(stage.parent).as_posix())
        else:
            epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
            def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
                info.mtime = epoch; info.uid = 0; info.gid = 0; info.uname = ""; info.gname = ""
                # Windows does not implement Unix executable chmod bits. Keep
                # archive permissions explicit rather than inheriting the host.
                relative = info.name.removeprefix(name + "/")
                if info.isfile() and relative in {"acid-rain", "acid-rain-desktop", "start-desktop.sh", "start-terminal.sh"}:
                    info.mode = 0o755
                return info
            with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    tar.add(stage, arcname=name, filter=normalize)
        text(archive.with_name(archive.name + ".sha256"), f"{digest(archive)}  {archive.name}\n")
    print(f"Packaged {archive.name} ({archive.stat().st_size:,} bytes)")
    return archive


def manifest(directory: Path, expected: int | None = None) -> None:
    archives = sorted([*directory.glob("*.tar.gz"), *directory.glob("*.zip")])
    if not archives or (expected is not None and len(archives) != expected):
        raise ValueError(f"Expected {expected or 'at least one'} archives, found {len(archives)}")
    lines = []
    for archive in archives:
        checksum = archive.with_name(archive.name + ".sha256")
        actual = digest(archive)
        if checksum.read_text(encoding="utf-8") != f"{actual}  {archive.name}\n":
            raise ValueError(f"Archive checksum verification failed: {archive.name}")
        lines.append(f"{actual}  {archive.name}\n")
    text(directory / "SHA256SUMS", "".join(lines))
    print(f"Verified {len(archives)} archives; wrote {directory / 'SHA256SUMS'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("package")
    build.add_argument("--target", choices=TARGETS, required=True)
    build.add_argument("--version", required=True)
    build.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    build.add_argument("--bin-dir", type=Path, required=True)
    build.add_argument("--output", type=Path, default=Path("dist"))
    build.add_argument("--native-app", type=Path)
    verify = commands.add_parser("manifest")
    verify.add_argument("--directory", type=Path, default=Path("dist"))
    verify.add_argument("--expected", type=int)
    args = parser.parse_args()
    try:
        if args.command == "package":
            package(args)
        else:
            manifest(args.directory, args.expected)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"release packaging: {error}\n")


if __name__ == "__main__":
    main()
