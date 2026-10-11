#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Compare protected water-mode builds sequentially, with Drain reported separately."""
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
FLOWS = {"before": ("classic", "rapids"), "after": ("classic", "rapids", "drain")}
QUALITIES = ("low", "balanced", "high")
SIZE = (200, 60)
WARMUP = 30
SPEED = 3.0
INTENSITY = 2.0
BLEACH = 1.0
DROP_SIZE = 1.0


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): fingerprint(path)
            for path in sorted((ROOT / "src").glob("*.rs"))}


def archive_contents(path: Path) -> dict[str, bytes]:
    with tarfile.open(path) as archive:
        return {member.name: archive.extractfile(member).read()
                for member in archive.getmembers() if member.isfile()}


def clock_constants(source: str) -> dict[str, float]:
    pattern = r"pub const (DEFAULT_SPEED|SURFACE_TIME_SCALE): f32 = ([0-9.]+);"
    return {key: float(value) for key, value in re.findall(pattern, source)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/whirlpool-after/cpu")
    parser.add_argument("--before", type=Path, default=ROOT / "benchmarks/whirlpool-before/acid-rain")
    parser.add_argument("--before-source", type=Path, default=ROOT / "benchmarks/whirlpool-before/source.tar.gz")
    parser.add_argument("--after", type=Path, default=ROOT / "benchmarks/whirlpool-after/acid-rain")
    parser.add_argument("--after-source", type=Path, default=ROOT / "benchmarks/whirlpool-after/source.tar.gz")
    args = parser.parse_args()
    if not 1 <= args.frames <= 100_000:
        parser.error("--frames must be between 1 and 100000")
    args.output = args.output.resolve()
    binaries = {"before": args.before.resolve(), "after": args.after.resolve()}
    archives = {"before": args.before_source.resolve(), "after": args.after_source.resolve()}
    binary_hashes = {version: fingerprint(path) for version, path in binaries.items()}
    archive_hashes = {version: fingerprint(path) for version, path in archives.items()}
    archived = {version: archive_contents(path) for version, path in archives.items()}
    sources = {version: {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
                         if name.endswith(".rs")} for version, contents in archived.items()}
    cargo_files = {version: {name: hashlib.sha256(contents[name]).hexdigest()
                             for name in ("Cargo.toml", "Cargo.lock")}
                   for version, contents in archived.items()}
    clocks = {version: clock_constants(contents["src/motion.rs"].decode())
              for version, contents in archived.items()}
    assert all(clock == {"DEFAULT_SPEED": SPEED, "SURFACE_TIME_SCALE": 0.25}
               for clock in clocks.values()), "both simulation clock constants must match"
    rustc = subprocess.check_output(["rustc", "-vV"], text=True)
    provenance = {}
    build_logs = {}
    for version, binary in binaries.items():
        proof = json.loads((binary.parent / "provenance.json").read_text())
        assert proof["rebuilt_native_binary_sha256"] == binary_hashes[version], f"{version} binary changed"
        assert proof["archive_sha256"] == archive_hashes[version], f"{version} source archive changed"
        assert proof["source_sha256"] == sources[version], f"{version} archived Rust sources differ"
        assert proof["cargo_files_sha256"] == cargo_files[version], f"{version} archived Cargo files differ"
        assert proof["rustflags"] == "-C target-cpu=native", f"{version} requires native release flags"
        assert proof["rustc"] == rustc, f"{version} requires the same compiler"
        log = json.loads((binary.parent / "native-build.json").read_text())
        assert [item["argv"] for item in log] == [
            ["cargo", "check", "--locked"], ["cargo", "build", "--release", "--locked"]
        ], f"{version} must record a check before its native build"
        assert all(item["returncode"] == 0 for item in log), f"{version} build did not pass"
        assert log[1]["env"]["RUSTFLAGS"] == proof["rustflags"]
        provenance[version] = proof
        build_logs[version] = log

    def check_inputs() -> None:
        assert binary_hashes == {version: fingerprint(path) for version, path in binaries.items()}, \
            "executable changed during measurement"
        assert archive_hashes == {version: fingerprint(path) for version, path in archives.items()}, \
            "source archive changed during measurement"
        assert sources["after"] == source_hashes(), "current sources differ from the protected after build"
        assert cargo_files["after"] == {name: fingerprint(ROOT / name) for name in cargo_files["after"]}, \
            "current Cargo files differ from the protected after build"

    check_inputs()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "utc_started": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "binaries": {version: str(path) for version, path in binaries.items()},
        "binary_sha256": binary_hashes,
        "source_archives": {version: str(path) for version, path in archives.items()},
        "source_archive_sha256": archive_hashes, "source_sha256": sources,
        "cargo_files_sha256": cargo_files, "build_provenance": provenance, "native_build_logs": build_logs,
        "rustc": rustc, "clock_constants": clocks, "flows": FLOWS, "qualities": QUALITIES,
        "size": SIZE, "frames": args.frames, "warmup_frames": WARMUP, "expected_cases": 15,
        "requested_speed": SPEED, "requested_intensity": INTENSITY,
        "requested_bleach": BLEACH, "requested_drop_size": DROP_SIZE,
        "fixtures": {
            "initialization": "Ten chronological one-second water advances and deterministic add_drop_sized calls, indices 9..0, including pigment history.",
            "rain": "One deterministic impact every 0.125 physical seconds during warmup and measurement.",
            "scene": "Explicit --scene journey; fixed 60 Hz wall-time steps; both benchmarks honor explicit --speed 3.",
            "controls": "Every case explicitly passes --speed 3 --intensity 2 --bleach 1 --drop-size 1.",
            "water_selection": "Before: Classic uses its archived default, Rapids uses --rapids. After: every mode uses explicit --water classic|rapids|drain. Native CSV flow is verified in every case.",
            "comparison": "Classic and Rapids are paired across versions. The new default Drain has no baseline equivalent and is reported separately at the same speed and settings.",
            "scope": "Measures simulation, rendering and ANSI encoding. Terminal transport and emulation are excluded; this is not interactive terminal FPS. Images and physics intentionally differ, so this compares whole modes rather than an isolated identical-image optimization.",
        },
        "commands": [],
    }
    metadata_path = args.output / "metadata.json"

    def save_metadata() -> None:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    save_metadata()
    rows = []
    reference_elapsed = None
    measured_scene_time_ranges = {}
    width, height = SIZE
    for quality in QUALITIES:
        for flow in FLOWS["after"]:
            for version in FLOWS:
                if flow not in FLOWS[version]:
                    continue
                check_inputs()
                folder = args.output / version
                folder.mkdir(exist_ok=True)
                report = folder / f"{flow}-{quality}-{width}x{height}.csv"
                command = [str(binaries[version]), "--benchmark", "--scene", "journey",
                           "--quality", quality, "--size", f"{width}x{height}", "--frames", str(args.frames),
                           "--intensity", f"{INTENSITY:g}", "--speed", f"{SPEED:g}",
                           "--bleach", f"{BLEACH:g}", "--drop-size", f"{DROP_SIZE:g}", "--report", str(report)]
                if version == "after":
                    command.extend(["--water", flow])
                elif flow == "rapids":
                    command.append("--rapids")
                result = subprocess.run(command, capture_output=True, text=True)
                requested = {"requested_intensity": INTENSITY, "requested_speed": SPEED,
                             "requested_bleach": BLEACH, "requested_drop_size": DROP_SIZE,
                             "requested_water": flow}
                metadata["commands"].append({
                    "version": version, "flow": flow, "quality": quality, **requested,
                    "argv": command, "status": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                })
                save_metadata()
                result.check_returncode()
                parsed = list(csv.DictReader(io.StringIO(result.stdout)))
                assert len(parsed) == 1, f"expected one report for {version}/{flow}/{quality}"
                row = {"version": version, **requested, **parsed[0]}
                assert row["flow"] == flow and row["quality"] == quality, f"wrong actual mode: {row}"
                assert row["scene"] == "journey", f"unexpected scene: {row}"
                assert (int(row["width"]), int(row["height"])) == SIZE
                assert int(row["measured_frames"]) == args.frames and int(row["warmup_frames"]) == WARMUP
                assert all(math.isfinite(float(row[key])) and float(row[key]) >= 0
                           for key in ("mean_ms", "p50_ms", "p95_ms", "bytes_per_frame"))
                for field, expected in (("intensity", INTENSITY), ("speed", SPEED),
                                        ("bleach", BLEACH), ("drop_size", DROP_SIZE)):
                    if field in row:
                        assert float(row[field]) == expected, f"reported {field} differs: {row}"
                with report.with_suffix(".samples.csv").open() as handle:
                    samples = list(csv.DictReader(handle))
                assert len(samples) == args.frames
                assert [int(sample["frame"]) for sample in samples] == list(range(args.frames))
                assert all(math.isfinite(float(sample["work_ms"])) and float(sample["work_ms"]) >= 0
                           for sample in samples)
                elapsed = [float(sample["elapsed_seconds"]) for sample in samples]
                assert all(math.isfinite(value) for value in elapsed)
                assert all(a < b for a, b in zip(elapsed, elapsed[1:])), "simulation clock must advance"
                for frame, actual in enumerate(elapsed):
                    expected = 10.0 + (WARMUP + frame + 1) * SPEED / 60.0
                    assert math.isclose(actual, expected, rel_tol=2e-6, abs_tol=5e-6), \
                        f"effective scene speed differs for {version}/{flow}: frame={frame}, time={actual}, expected={expected}"
                if reference_elapsed is None:
                    reference_elapsed = elapsed
                else:
                    assert elapsed == reference_elapsed, "cases sampled different scene times"
                measured_scene_time_ranges[f"{version}/{flow}/{quality}"] = [elapsed[0], elapsed[-1]]
                assert report.read_text() == result.stdout, "saved report differs from stdout"
                check_inputs()
                rows.append(row)
                with (args.output / "paired.csv").open("w", newline="") as handle:
                    fieldnames = list(dict.fromkeys(key for saved_row in rows for key in saved_row))
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"{version} {flow} {quality} {width}x{height}: {row['mean_ms']} ms; p95 {row['p95_ms']} ms", flush=True)
    assert len(rows) == metadata["expected_cases"]
    lines = ["# Whirlpool CPU comparison", "",
             f"Sequential native release runs at {width}×{height}: {WARMUP} warmup + {args.frames} measured frames. "
             "Every case explicitly uses speed 3×, intensity 2×, bleaching 1× and drop size 1×. "
             "The fixture uses fixed 60 Hz wall-time steps and 0.25× surface time. Simulation, rendering and ANSI encoding "
             "are measured; terminal transport and emulation are excluded. These are not interactive terminal FPS results.", "",
             "Matched existing modes, with identical scene timestamps and source schedules across versions.", "",
             "| Flow | Quality | Before mean ms | Before p95 ms | After mean ms | After p95 ms |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for flow in FLOWS["before"]:
        for quality in QUALITIES:
            cases = {row["version"]: row for row in rows if row["quality"] == quality and row["flow"] == flow}
            values = " | ".join(f"{float(cases[version][key]):.3f}" for version in FLOWS
                                for key in ("mean_ms", "p95_ms"))
            lines.append(f"| {flow} | {quality} | {values} |")
    lines += ["", "New default Drain, at the same speed and settings. The baseline has no Drain mode; "
              "these costs are reported separately and do not describe a before/after speedup.", "",
              "| Flow | Quality | After mean ms | After p95 ms |", "| --- | --- | ---: | ---: |"]
    for row in rows:
        if row["flow"] == "drain":
            lines.append(f"| drain | {row['quality']} | {float(row['mean_ms']):.3f} | {float(row['p95_ms']):.3f} |")
    reproduce = shlex.join(["uv", "run", "--script", "scripts/benchmark_whirlpool.py",
                            "--before", str(binaries["before"]), "--before-source", str(archives["before"]),
                            "--after", str(binaries["after"]), "--after-source", str(archives["after"]),
                            "--output", str(args.output), "--frames", str(args.frames)])
    lines += ["", "Both versions replay ten chronological seconds of water and pigment sources before warmup, "
              "then apply the same eight-impact-per-physical-second schedule. Actual per-frame scene times are checked "
              "against speed 3× and must match exactly across all 15 cases. Images and physics intentionally differ; "
              "this compares complete modes. Initialization precedes measurement.", "",
              "paired.csv and metadata.json record explicit requested controls and native reported modes. "
              "Every raw report and per-frame sample is retained. Both protected source archives, binaries, compiler "
              "and native build provenance are verified; source and binary hashes are rechecked before and after each case.", "",
              f"Reproduce: `{reproduce}`. Run only after protected native builds and without competing builds, captures or PTY tests.", ""]
    (args.output / "comparison.md").write_text("\n".join(lines))
    metadata["utc_completed"] = datetime.now(timezone.utc).isoformat()
    metadata["cases"] = len(rows)
    metadata["measured_scene_time_ranges"] = measured_scene_time_ranges
    save_metadata()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
