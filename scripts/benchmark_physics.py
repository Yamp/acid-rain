#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Compare the archived and current physical-water native binaries sequentially."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent.parent


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--title", default="Physical water CPU comparison")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/physics-after")
    parser.add_argument("--before", type=Path, default=ROOT / "benchmarks/physics-before/acid-rain")
    parser.add_argument("--before-source", type=Path, default=ROOT / "benchmarks/physics-before/source.tar.gz")
    parser.add_argument("--after", type=Path, default=ROOT / "target/release/acid-rain")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    with tarfile.open(args.before_source) as archive:
        before_hashes = {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                         for member in archive.getmembers() if member.isfile() and member.name.endswith(".rs")}
        before_motion = archive.extractfile("src/motion.rs").read().decode()
    after_motion = (ROOT / "src/motion.rs").read_text()
    pattern = r"pub const (DEFAULT_SPEED|SURFACE_TIME_SCALE): f32 = ([0-9.]+);"
    speeds = dict(re.findall(pattern, after_motion))
    assert {k: float(v) for k, v in re.findall(pattern, before_motion)} == {k: float(v) for k, v in speeds.items()}, "match the clocks before comparing"
    binaries = {"before": args.before.resolve(), "after": args.after.resolve()}
    rows, commands = [], []
    args.output.mkdir(parents=True, exist_ok=True)
    for quality in ("balanced", "high", "low"):
        for width, height in ((120, 40), (200, 60), (320, 100)):
            for version in ("before", "after"):
                folder = args.output / version
                folder.mkdir(exist_ok=True)
                command = [str(binaries[version]), "--benchmark", "--quality", quality,
                           "--size", f"{width}x{height}", "--frames", str(args.frames),
                           "--report", str(folder / f"{quality}-{width}x{height}.csv")]
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                row = {"version": version, **next(csv.DictReader(io.StringIO(result.stdout)))}
                row.setdefault("flow", "classic")
                rows.append(row)
                commands.append({"argv": command, "stdout": result.stdout, "stderr": result.stderr})
                print(f"{version} {quality} {width}x{height}: {row['mean_ms']} ms", flush=True)
    with (args.output / "paired.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {args.title}", "",
             f"Sequential native release runs, 30 warmup + {args.frames} measured frames each. Both use {float(speeds['DEFAULT_SPEED']):g}× scene time and {float(speeds['SURFACE_TIME_SCALE']):g}× surface time at 60 Hz. Rain and aged-drop fixtures match; rendered pixels differ. Terminal emulation and transport are excluded.", "",
             "| Cells | Quality | Before mean ms | After mean ms | After p95 ms |",
             "| --- | --- | ---: | ---: | ---: |"]
    for quality in ("balanced", "high", "low"):
        for width, height in ((120, 40), (200, 60), (320, 100)):
            pair = {r["version"]: r for r in rows if r["quality"] == quality and int(r["width"]) == width and int(r["height"]) == height}
            before, after = pair["before"], pair["after"]
            lines.append(f"| {width}×{height} | {quality} | {float(before['mean_ms']):.3f} | {float(after['mean_ms']):.3f} | {float(after['p95_ms']):.3f} |")
    reproduce = shlex.join(["uv", "run", "--script", "scripts/benchmark_physics.py",
                            "--before", str(args.before), "--before-source", str(args.before_source),
                            "--after", str(args.after), "--output", str(args.output),
                            "--frames", str(args.frames), "--title", args.title])
    lines += ["", "Interactive simulation caps its grid at 240×120; these headless measurements use the entire requested grid. Raw reports/samples are in before/ and after/.", "",
              f"Reproduce after cargo check and a native release build: `{reproduce}` with no competing builds, previews or PTY tests.", ""]
    (args.output / "comparison.md").write_text("\n".join(lines))
    metadata = {"utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                "binary_sha256": {version: fingerprint(path) for version, path in binaries.items()},
                "before_source_sha256": before_hashes,
                "after_source_sha256": {str(p.relative_to(ROOT)): fingerprint(p) for p in sorted((ROOT / "src").glob("*.rs"))},
                "commands": commands, "rustc": subprocess.check_output(["rustc", "-vV"], text=True)}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
