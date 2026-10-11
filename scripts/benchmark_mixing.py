#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Compare matched-speed dye mixing and the faster new default sequentially."""
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
DOSES = {"before-2x": 2.0, "after-2x": 2.0, "after-default": 2.0}
SPEEDS = {"before-2x": 1.1, "after-2x": 1.1, "after-default": 3.0}
BLEACH = {"before-2x": None, "after-2x": 1.0, "after-default": 1.0}
FLOWS = ("classic", "rapids")
QUALITIES = ("low", "balanced", "high")
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
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/mixing-after/cpu")
    parser.add_argument("--before", type=Path, default=ROOT / "benchmarks/mixing-before/acid-rain")
    parser.add_argument("--before-source", type=Path, default=ROOT / "benchmarks/mixing-before/source.tar.gz")
    parser.add_argument("--after", type=Path, default=ROOT / "benchmarks/mixing-after/acid-rain")
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
    before_clocks = clock_constants(before_motion)
    assert before_clocks["SURFACE_TIME_SCALE"] == clocks["SURFACE_TIME_SCALE"] == 0.25, \
        "physical surface time scale must stay matched"
    assert before_clocks["DEFAULT_SPEED"] == 1.1, "unexpected archived default speed"
    assert clocks["DEFAULT_SPEED"] == 3.0, "new default speed must be 3.0"
    binaries = {version: (args.before if version.startswith("before-") else args.after).resolve()
                for version in DOSES}
    binary_hashes = {version: fingerprint(path) for version, path in binaries.items()}
    after_hashes = source_hashes()
    after_cargo_hashes = {name: fingerprint(ROOT / name) for name in ("Cargo.toml", "Cargo.lock")}
    provenance_path = args.before.resolve().parent / "provenance.json"
    before_provenance = json.loads(provenance_path.read_text())
    assert before_provenance["rebuilt_native_binary_sha256"] == binary_hashes["before-2x"], \
        "baseline executable does not match its explicit native rebuild"
    assert before_provenance["archive_sha256"] == archive_sha256, "baseline archive changed"
    assert before_provenance["source_sha256"] == before_hashes, "baseline source hashes changed"
    assert before_provenance["rustflags"] == "-C target-cpu=native", "baseline must use native release flags"
    after_provenance_path = args.after.resolve().parent / "provenance.json"
    after_provenance = json.loads(after_provenance_path.read_text())
    assert after_provenance["rebuilt_native_binary_sha256"] == binary_hashes["after-2x"], \
        "after executable does not match its protected native build"
    assert after_provenance["source_sha256"] == after_hashes, "after executable was built from different sources"
    assert after_provenance["cargo_files_sha256"] == after_cargo_hashes, "after Cargo files differ from the build"
    assert after_provenance["rustflags"] == "-C target-cpu=native", "after must use native release flags"

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
        "after_build_provenance": after_provenance,
        "after_source_sha256": after_hashes, "after_cargo_sha256": after_cargo_hashes,
        "clock_constants": clocks, "before_clock_constants": before_clocks,
        "rustc": subprocess.check_output(["rustc", "-vV"], text=True),
        "binaries": {version: str(path) for version, path in binaries.items()},
        "version_doses": DOSES, "version_speeds": SPEEDS, "version_bleach": BLEACH,
        "flows": FLOWS, "qualities": QUALITIES, "size": SIZE,
        "frames": args.frames, "warmup_frames": WARMUP, "expected_cases": 18,
        "dose_reporting": "Every command passes --intensity explicitly. requested_intensity records the CLI request; the native benchmark CSV does not echo intensity.",
        "after_build_requirement": "Protected after executable must have sibling provenance.json proving the current Rust/Cargo source hashes and native release flags. This script validates provenance but does not rebuild it.",
        "fixtures": {
            "initialization": "Both binaries replay ten chronological one-second advances and deterministic add_drop_sized calls, indices 9..0, including pigment history.",
            "rain": "One deterministic impact every 0.125 physical seconds during warmup and measurement.",
            "drop_size": "Explicit --drop-size 1 for every command.",
            "scene": "Explicit --scene journey for every command.",
            "timing": "Fixed 60 Hz wall-time steps. before-2x and after-2x explicitly request speed 1.1 and must have identical sampled scene times; after-default explicitly requests speed 3.0 and advances physical time faster. Surface time scale stays 0.25.",
            "before_speed_behavior": "The archived headless benchmark ignores its speed option and uses DEFAULT_SPEED=1.1 internally. It still receives explicit --speed 1.1; recorded scene timestamps verify the effective rate. The after benchmark honors the supplied speed.",
            "bleach": "Only after variants receive --bleach 1. The archived binary does not implement bleaching or accept this option.",
            "scope": "Source schedules per physical second match. Intensity is explicitly 2 in every case. Dye transport, bleaching and optical images intentionally differ. The speed-3 default is a separate higher-physics-work case, not a matched-speed optimization comparison. Terminal emulation and transport are excluded.",
        },
        "commands": [],
    }
    metadata_path = args.output / "metadata.json"

    def save_metadata() -> None:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    save_metadata()
    rows = []
    reference_elapsed_by_speed = {}
    measured_scene_time_ranges = {}
    width, height = SIZE
    for quality in QUALITIES:
        for flow in FLOWS:
            for version, dose in DOSES.items():
                speed = SPEEDS[version]
                bleach = BLEACH[version]
                check_inputs()
                folder = args.output / version
                folder.mkdir(exist_ok=True)
                report = folder / f"{flow}-{quality}-{width}x{height}.csv"
                command = [str(binaries[version]), "--benchmark", "--scene", "journey",
                           "--quality", quality, "--size", f"{width}x{height}",
                           "--frames", str(args.frames), "--intensity", f"{dose:g}",
                           "--speed", f"{speed:g}",
                           "--drop-size", "1", "--report", str(report)]
                if bleach is not None:
                    command.extend(["--bleach", f"{bleach:g}"])
                if flow == "rapids":
                    command.append("--rapids")
                result = subprocess.run(command, capture_output=True, text=True)
                metadata["commands"].append({
                    "version": version, "flow": flow, "quality": quality,
                    "requested_intensity": dose, "requested_speed": speed,
                    "requested_bleach": bleach, "argv": command,
                    "status": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                })
                save_metadata()
                result.check_returncode()
                parsed = list(csv.DictReader(io.StringIO(result.stdout)))
                assert len(parsed) == 1, f"expected one report for {version}/{flow}/{quality}"
                row = {"version": version, "requested_intensity": dose,
                       "requested_speed": speed, "requested_bleach": bleach, **parsed[0]}
                assert row["flow"] == flow and row["quality"] == quality, f"wrong actual mode: {row}"
                assert row["scene"] == "journey", f"unexpected scene: {row}"
                assert (int(row["width"]), int(row["height"])) == SIZE
                assert int(row["measured_frames"]) == args.frames and int(row["warmup_frames"]) == WARMUP
                assert all(math.isfinite(float(row[key])) and float(row[key]) >= 0
                           for key in ("mean_ms", "p50_ms", "p95_ms", "bytes_per_frame"))
                if "intensity" in row:
                    assert float(row["intensity"]) == dose, f"reported intensity differs: {row}"
                if "speed" in row:
                    assert float(row["speed"]) == speed, f"reported speed differs: {row}"
                if "bleach" in row and bleach is not None:
                    assert float(row["bleach"]) == bleach, f"reported bleaching differs: {row}"
                with report.with_suffix(".samples.csv").open() as handle:
                    samples = list(csv.DictReader(handle))
                assert len(samples) == args.frames
                assert [int(sample["frame"]) for sample in samples] == list(range(args.frames))
                assert all(math.isfinite(float(sample["work_ms"])) and float(sample["work_ms"]) >= 0
                           for sample in samples)
                assert all(math.isfinite(float(sample["elapsed_seconds"])) for sample in samples)
                elapsed = [float(sample["elapsed_seconds"]) for sample in samples]
                assert all(a < b for a, b in zip(elapsed, elapsed[1:])), "simulation clock must advance"
                for frame, actual in enumerate(elapsed):
                    expected = 10.0 + (WARMUP + frame + 1) * speed / 60.0
                    assert math.isclose(actual, expected, rel_tol=2e-6, abs_tol=5e-6), \
                        f"effective scene speed differs for {version}: frame={frame}, time={actual}, expected={expected}"
                if speed not in reference_elapsed_by_speed:
                    reference_elapsed_by_speed[speed] = elapsed
                else:
                    assert elapsed == reference_elapsed_by_speed[speed], \
                        f"cases at speed {speed} sampled different scene times"
                measured_scene_time_ranges[version] = [elapsed[0], elapsed[-1]]
                assert report.read_text() == result.stdout, "saved report differs from stdout"
                check_inputs()
                rows.append(row)
                with (args.output / "paired.csv").open("w", newline="") as handle:
                    fieldnames = list(dict.fromkeys(key for saved_row in rows for key in saved_row))
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"{version} (intensity {dose:g}, speed {speed:g}) {flow} {quality} {width}x{height}: "
                      f"{row['mean_ms']} ms; p95 {row['p95_ms']} ms", flush=True)
    assert len(rows) == metadata["expected_cases"]
    lines = ["# Dye mixing CPU comparison", "",
             f"Sequential native release runs at {width}×{height}: {WARMUP} warmup + {args.frames} measured frames, "
             f"60 Hz fixed wall-time steps and {clocks['SURFACE_TIME_SCALE']:g}× surface time. "
             "Intensity is explicitly 2× in every case; speed is explicitly 1.1× for the matched pair and 3× "
             "for the new default. Only after variants receive --bleach 1. "
             "Measured work includes simulation, rendering and ANSI encoding; terminal emulation and transport "
             "are excluded. These timings do not establish interactive terminal FPS.", "",
             "Matched scene speed 1.1×; this compares the cost of the mixing and bleaching changes.", "",
             "| Flow | Quality | Before, speed 1.1 mean ms | Before, speed 1.1 p95 ms | After, speed 1.1 mean ms | After, speed 1.1 p95 ms |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for flow in FLOWS:
        for quality in QUALITIES:
            cases = {row["version"]: row for row in rows if row["quality"] == quality and row["flow"] == flow}
            values = " | ".join(f"{float(cases[version][key]):.3f}" for version in ("before-2x", "after-2x")
                                for key in ("mean_ms", "p95_ms"))
            lines.append(f"| {flow} | {quality} | {values} |")
    lines += ["", "New default scene speed 3×. This advances physical simulation time "
              f"{3.0 / 1.1:.3f}× faster per frame than the matched-speed pair, so this includes extra "
              "simulation and source work. It is reported separately from the matched-speed comparison.", "",
              "| Flow | Quality | After default, speed 3 mean ms | After default, speed 3 p95 ms |",
              "| --- | --- | ---: | ---: |"]
    for flow in FLOWS:
        for quality in QUALITIES:
            row = next(row for row in rows if row["flow"] == flow and row["quality"] == quality
                       and row["version"] == "after-default")
            lines.append(f"| {flow} | {quality} | {float(row['mean_ms']):.3f} | {float(row['p95_ms']):.3f} |")
    reproduce = shlex.join(["uv", "run", "--script", "scripts/benchmark_mixing.py",
                            "--before", str(args.before.resolve()), "--before-source", str(args.before_source),
                            "--after", str(args.after.resolve()), "--output", str(args.output), "--frames", str(args.frames)])
    lines += ["", "Both binaries replay the same chronological water and pigment sources for ten physical seconds "
              "before warmup, then use the same eight-impact-per-physical-second schedule with drop size 1×. "
              "The before benchmark is internally fixed at scene speed 1.1; the after benchmark honors --speed. "
              "Actual recorded timestamps are checked against each requested rate and matched between cases "
              "at the same speed. Dye dynamics, bleaching and optical images intentionally differ. This compares whole modes, not an isolated "
              "optimization on identical images. Initialization and source replay precede measured frames.", "",
              "requested_intensity, requested_speed and requested_bleach in paired.csv and every saved command "
              "record the explicit CLI parameters; an empty before bleach field means unsupported, not zero bleaching. "
              "Raw reports and per-frame samples are retained for all 18 cases. "
              "Source and executable hashes are checked before and after each case.", "",
              f"Reproduce after cargo check and a native release build: `{reproduce}`. Run without competing builds, "
              "captures or PTY tests.", ""]
    (args.output / "comparison.md").write_text("\n".join(lines))
    metadata["utc_completed"] = datetime.now(timezone.utc).isoformat()
    metadata["cases"] = len(rows)
    metadata["measured_scene_time_ranges"] = measured_scene_time_ranges
    save_metadata()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
