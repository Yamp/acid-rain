#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Compare archived 1x, current 2x and current 4x trip rendering sequentially."""
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
DOSES = {"before-1x": 1.0, "after-2x": 2.0, "after-4x": 4.0}
FLOWS = ("classic", "rapids")
QUALITIES = ("balanced", "high", "low")
SIZE = (200, 60)
WARMUP = 30


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
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/trip-after/cpu")
    parser.add_argument("--before", type=Path, default=ROOT / "benchmarks/trip-before/acid-rain")
    parser.add_argument("--before-source", type=Path, default=ROOT / "benchmarks/trip-before/source.tar.gz")
    parser.add_argument("--after", type=Path, default=ROOT / "target/release/acid-rain")
    args = parser.parse_args()
    if not 1 <= args.frames <= 100_000:
        parser.error("--frames must be between 1 and 100000")
    args.output = args.output.resolve()
    args.before_source = args.before_source.resolve()
    archive_sha256 = fingerprint(args.before_source)
    with tarfile.open(args.before_source) as archive:
        before_hashes = {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                         for member in archive.getmembers() if member.isfile() and member.name.endswith(".rs")}
        before_motion = archive.extractfile("src/motion.rs").read().decode()
    clocks = clock_constants((ROOT / "src/motion.rs").read_text())
    assert clock_constants(before_motion) == clocks, "scene and surface clocks must match"
    binaries = {version: (args.before if version.startswith("before-") else args.after).resolve()
                for version in DOSES}
    binary_hashes = {version: fingerprint(path) for version, path in binaries.items()}
    after_hashes = source_hashes()
    after_cargo_hashes = {name: fingerprint(ROOT / name) for name in ("Cargo.toml", "Cargo.lock")}
    provenance_path = args.before.resolve().parent / "provenance.json"
    before_provenance = json.loads(provenance_path.read_text())
    assert before_provenance["rebuilt_native_binary_sha256"] == binary_hashes["before-1x"], \
        "baseline executable does not match its explicit native rebuild"
    assert before_provenance["archive_sha256"] == archive_sha256, "baseline archive changed"
    assert before_provenance["source_sha256"] == before_hashes, "baseline source hashes changed"
    assert before_provenance["rustflags"] == "-C target-cpu=native", "baseline must use native release flags"

    def check_inputs() -> None:
        assert binary_hashes == {version: fingerprint(path) for version, path in binaries.items()}, \
            "executable changed during measurement"
        assert after_hashes == source_hashes(), "sources changed during measurement"
        assert archive_sha256 == fingerprint(args.before_source), "baseline archive changed during measurement"
        assert after_cargo_hashes == {name: fingerprint(ROOT / name) for name in after_cargo_hashes}, \
            "Cargo files changed during measurement"

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "binary_sha256": binary_hashes,
        "before_source_sha256": before_hashes,
        "before_source_archive": str(args.before_source),
        "before_source_archive_sha256": archive_sha256,
        "before_build_provenance": before_provenance,
        "after_source_sha256": after_hashes, "after_cargo_sha256": after_cargo_hashes,
        "clock_constants": clocks, "rustc": subprocess.check_output(["rustc", "-vV"], text=True),
        "binaries": {version: str(path) for version, path in binaries.items()},
        "version_doses": DOSES, "flows": FLOWS, "qualities": QUALITIES, "size": SIZE,
        "frames": args.frames, "warmup_frames": WARMUP, "expected_cases": 18,
        "dose_reporting": "Every command passes --intensity explicitly. requested_intensity records the CLI request; the native benchmark CSV does not echo intensity.",
        "after_build_requirement": "Caller must cargo check and build the current executable with native release flags before running this script. This script fingerprints but does not rebuild it.",
        "fixtures": {
            "initialization": "Both binaries replay ten chronological one-second advances and deterministic add_drop_sized calls, indices 9..0, including pigment history.",
            "rain": "One deterministic impact every 0.125 physical seconds during warmup and measurement.",
            "drop_size": "Explicit --drop-size 1 for every command.",
            "scene": "Explicit --scene journey for every command.",
            "timing": "Headless work uses a fixed 60 Hz wall-time step and matched scene/surface clock constants.",
            "scope": "Water and pigment fixture schedules match. Optical output intentionally differs at requested doses 1, 2 and 4. Terminal emulation and transport are excluded.",
        },
        "commands": [],
    }
    metadata_path = args.output / "metadata.json"

    def save_metadata() -> None:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    save_metadata()
    rows = []
    reference_elapsed = None
    width, height = SIZE
    for quality in QUALITIES:
        for flow in FLOWS:
            for version, dose in DOSES.items():
                check_inputs()
                folder = args.output / version
                folder.mkdir(exist_ok=True)
                report = folder / f"{flow}-{quality}-{width}x{height}.csv"
                command = [str(binaries[version]), "--benchmark", "--scene", "journey",
                           "--quality", quality, "--size", f"{width}x{height}",
                           "--frames", str(args.frames), "--intensity", f"{dose:g}",
                           "--drop-size", "1", "--report", str(report)]
                if flow == "rapids":
                    command.append("--rapids")
                result = subprocess.run(command, capture_output=True, text=True)
                metadata["commands"].append({
                    "version": version, "flow": flow, "quality": quality,
                    "requested_intensity": dose, "argv": command,
                    "status": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                })
                save_metadata()
                result.check_returncode()
                parsed = list(csv.DictReader(io.StringIO(result.stdout)))
                assert len(parsed) == 1, f"expected one report for {version}/{flow}/{quality}"
                row = {"version": version, "requested_intensity": dose, **parsed[0]}
                assert row["flow"] == flow and row["quality"] == quality, f"wrong actual mode: {row}"
                assert row["scene"] == "journey", f"unexpected scene: {row}"
                assert (int(row["width"]), int(row["height"])) == SIZE
                assert int(row["measured_frames"]) == args.frames and int(row["warmup_frames"]) == WARMUP
                assert all(math.isfinite(float(row[key])) and float(row[key]) >= 0
                           for key in ("mean_ms", "p50_ms", "p95_ms", "bytes_per_frame"))
                if "intensity" in row:
                    assert float(row["intensity"]) == dose, f"reported intensity differs: {row}"
                with report.with_suffix(".samples.csv").open() as handle:
                    samples = list(csv.DictReader(handle))
                assert len(samples) == args.frames
                assert [int(sample["frame"]) for sample in samples] == list(range(args.frames))
                assert all(math.isfinite(float(sample["work_ms"])) and float(sample["work_ms"]) >= 0
                           for sample in samples)
                assert all(math.isfinite(float(sample["elapsed_seconds"])) for sample in samples)
                elapsed = [float(sample["elapsed_seconds"]) for sample in samples]
                assert all(a < b for a, b in zip(elapsed, elapsed[1:])), "simulation clock must advance"
                if reference_elapsed is None:
                    reference_elapsed = elapsed
                else:
                    assert elapsed == reference_elapsed, "cases sampled different scene times"
                assert report.read_text() == result.stdout, "saved report differs from stdout"
                check_inputs()
                rows.append(row)
                with (args.output / "paired.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"{version} ({dose:g}x) {flow} {quality} {width}x{height}: "
                      f"{row['mean_ms']} ms; p95 {row['p95_ms']} ms", flush=True)
    assert len(rows) == metadata["expected_cases"]
    lines = ["# Trip rendering CPU comparison", "",
             f"Sequential native release runs at {width}×{height}: {WARMUP} warmup + {args.frames} measured frames, "
             f"60 Hz fixed wall-time steps, {clocks['DEFAULT_SPEED']:g}× scene speed and "
             f"{clocks['SURFACE_TIME_SCALE']:g}× surface time. Every dose is supplied explicitly with --intensity. "
             "Measured work includes simulation, rendering and ANSI encoding; terminal emulation and transport "
             "are excluded. These timings do not establish interactive terminal FPS.", "",
             "| Flow | Quality | Before 1× mean ms | Before 1× p95 ms | After 2× mean ms | After 2× p95 ms | After 4× mean ms | After 4× p95 ms |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for flow in FLOWS:
        for quality in QUALITIES:
            cases = {row["version"]: row for row in rows if row["quality"] == quality and row["flow"] == flow}
            values = " | ".join(f"{float(cases[version][key]):.3f}" for version in DOSES
                                for key in ("mean_ms", "p95_ms"))
            lines.append(f"| {flow} | {quality} | {values} |")
    reproduce = shlex.join(["uv", "run", "--script", "scripts/benchmark_trip.py",
                            "--before", str(args.before.resolve()), "--before-source", str(args.before_source),
                            "--after", str(args.after.resolve()), "--output", str(args.output), "--frames", str(args.frames)])
    lines += ["", "Both binaries replay the same chronological water and pigment sources for ten physical seconds "
              "before warmup, then use the same eight-impact-per-physical-second schedule with drop size 1×. "
              "The optical images intentionally differ. This compares whole requested modes, not an isolated "
              "optimization on identical images. Initialization and source replay precede measured frames.", "",
              "The CSV currently does not echo intensity: requested_intensity in paired.csv and every saved command "
              "record the explicit CLI dose. Raw reports and per-frame samples are retained for all 18 cases. "
              "Source and executable hashes are checked before and after each case.", "",
              f"Reproduce after cargo check and a native release build: `{reproduce}`. Run without competing builds, "
              "captures or PTY tests.", ""]
    (args.output / "comparison.md").write_text("\n".join(lines))
    metadata["utc_completed"] = datetime.now(timezone.utc).isoformat()
    metadata["cases"] = len(rows)
    metadata["measured_scene_time_range"] = [reference_elapsed[0], reference_elapsed[-1]]
    save_metadata()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
