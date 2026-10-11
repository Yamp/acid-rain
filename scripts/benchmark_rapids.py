#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Measure archived classic, current classic and current rapids sequentially."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import shlex
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parent.parent
VERSIONS = ("before-classic", "after-classic", "after-rapids")
QUALITIES = ("balanced", "high", "low")
SIZES = ((120, 40), (200, 60), (320, 100))


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): fingerprint(path)
            for path in sorted((ROOT / "src").glob("*.rs"))}


def clock_constants(source: str) -> dict[str, float]:
    pattern = r"pub const (DEFAULT_SPEED|SURFACE_TIME_SCALE): f32 = ([0-9.]+);"
    values = {key: float(value) for key, value in re.findall(pattern, source)}
    assert set(values) == {"DEFAULT_SPEED", "SURFACE_TIME_SCALE"}, "missing simulation clock constants"
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/rapids-after")
    parser.add_argument("--before", type=Path, default=ROOT / "benchmarks/rapids-before/acid-rain")
    parser.add_argument("--before-source", type=Path, default=ROOT / "benchmarks/rapids-before/source.tar.gz")
    parser.add_argument("--after", type=Path, default=ROOT / "target/release/acid-rain")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    with tarfile.open(args.before_source) as archive:
        before_hashes = {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                         for member in archive.getmembers() if member.isfile() and member.name.endswith(".rs")}
        before_motion = archive.extractfile("src/motion.rs").read().decode()
    clocks = clock_constants((ROOT / "src/motion.rs").read_text())
    assert clock_constants(before_motion) == clocks, "match scene and surface clocks before comparing"
    binaries = {"before-classic": args.before.resolve(), "after-classic": args.after.resolve(),
                "after-rapids": args.after.resolve()}
    binary_hashes = {version: fingerprint(path) for version, path in binaries.items()}
    after_hashes = source_hashes()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "utc_started": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "binary_sha256": binary_hashes, "before_source_sha256": before_hashes,
        "after_source_sha256": after_hashes, "clock_constants": clocks,
        "frames": args.frames, "warmup_frames": 30,
        "fixtures": {
            "classic": "10-second advance, then ten deterministic centers inserted with ages 0..9 seconds",
            "rapids": "ten chronological one-second advances with deterministic add_drop calls, indices 9..0",
            "rain": "one deterministic impact every 0.125 physical seconds during warmup/measurement",
        },
        "rustc": subprocess.check_output(["rustc", "-vV"], text=True),
        "commands": [],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    for quality in QUALITIES:
        for width, height in SIZES:
            for version in VERSIONS:
                folder = args.output / version
                folder.mkdir(exist_ok=True)
                report = folder / f"{quality}-{width}x{height}.csv"
                command = [str(binaries[version]), "--benchmark", "--quality", quality,
                           "--size", f"{width}x{height}", "--frames", str(args.frames), "--report", str(report)]
                if version == "after-rapids":
                    command.append("--rapids")
                result = subprocess.run(command, capture_output=True, text=True)
                metadata["commands"].append({"argv": command, "status": result.returncode,
                                             "stdout": result.stdout, "stderr": result.stderr})
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                result.check_returncode()
                parsed = list(csv.DictReader(io.StringIO(result.stdout)))
                assert len(parsed) == 1, f"expected one benchmark report: {version}"
                row = {"version": version, **parsed[0]}
                row.setdefault("flow", "classic")  # Archived reports predate this column.
                expected_flow = "rapids" if version == "after-rapids" else "classic"
                assert row["flow"] == expected_flow and row["quality"] == quality
                assert (int(row["width"]), int(row["height"])) == (width, height)
                assert int(row["measured_frames"]) == args.frames and int(row["warmup_frames"]) == 30
                assert all(math.isfinite(float(row[key])) and float(row[key]) >= 0
                           for key in ("mean_ms", "p50_ms", "p95_ms", "bytes_per_frame"))
                with report.with_suffix(".samples.csv").open() as samples_file:
                    samples = list(csv.DictReader(samples_file))
                assert len(samples) == args.frames
                assert [int(sample["frame"]) for sample in samples] == list(range(args.frames))
                assert all(math.isfinite(float(sample["work_ms"])) for sample in samples)
                rows.append(row)
                print(f"{version} {quality} {width}x{height}: {row['mean_ms']} ms; p95 {row['p95_ms']} ms", flush=True)
    assert binary_hashes == {version: fingerprint(path) for version, path in binaries.items()}, "binary changed during measurement"
    assert after_hashes == source_hashes(), "sources changed during measurement"
    with (args.output / "paired.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# River rapids CPU comparison", "",
             f"Sequential native release runs: 30 warmup + {args.frames} measured frames, 60 Hz fixed wall time, "
             f"{clocks['DEFAULT_SPEED']:g}× scene speed and {clocks['SURFACE_TIME_SCALE']:g}× surface time. "
             "Columns compare the previous classic renderer, the current default classic mode and optional river rapids. "
             "Terminal emulation and transport are excluded.", "",
             "| Cells | Quality | Before mean ms | Before p95 ms | Default mean ms | Default p95 ms | Rapids mean ms | Rapids p95 ms |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for quality in QUALITIES:
        for width, height in SIZES:
            cases = {row["version"]: row for row in rows if row["quality"] == quality
                     and (int(row["width"]), int(row["height"])) == (width, height)}
            values = " | ".join(f"{float(cases[version][key]):.3f}" for version in VERSIONS
                                for key in ("mean_ms", "p95_ms"))
            lines.append(f"| {width}×{height} | {quality} | {values} |")
    reproduce = shlex.join(["uv", "run", "--script", "scripts/benchmark_rapids.py",
                            "--before", str(args.before), "--before-source", str(args.before_source),
                            "--after", str(args.after), "--output", str(args.output), "--frames", str(args.frames)])
    lines += ["", "All modes reach ten physical seconds before warmup. Classic inserts ten aged analytic impulses; "
              "rapids replays those ten impacts chronologically through its evolving numerical flow. Fresh impacts use "
              "the same deterministic schedule. The pictures and physical workloads differ.", "",
              "Interactive surface sampling is capped at 240×120. These headless measurements use every requested "
              "surface cell; the rapids solver has its own fixed internal grid. Initialization/replay costs precede "
              "the measured frames. Each version directory contains the raw report and per-frame samples.", "",
              f"Reproduce after cargo check and a native release build: `{reproduce}`. Run without competing builds, "
              "captures or PTY tests.", ""]
    (args.output / "comparison.md").write_text("\n".join(lines))
    metadata["utc_completed"] = datetime.now(timezone.utc).isoformat()
    metadata["cases"] = len(rows)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
