#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Compare fixed source-frame work separately from real PTY display cadence."""
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
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PTY_RUNNER = ROOT / "scripts/measure_native_pty.py"
SIZES = ((200, 60), (320, 100))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=ROOT / "benchmarks/whirlpool-after/acid-rain")
    parser.add_argument("--after", type=Path, default=ROOT / "benchmarks/clarity-after/acid-rain")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/clarity-perf-after/performance")
    parser.add_argument("--section", choices=("all", "pty", "cpu"), default="all")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--pty-seconds", type=float, default=20.0)
    parser.add_argument("--pty-warmup-seconds", type=float, default=6.0)
    parser.add_argument("--water", choices=("classic", "rapids", "drain"), default="drain")
    parser.add_argument("--quality", choices=("auto", "high", "balanced", "low"), default="auto")
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--intensity", type=float, default=2.0)
    parser.add_argument("--bleach", type=float, default=1.0)
    parser.add_argument("--rain", type=float, default=6.0)
    parser.add_argument("--drop-size", type=float, default=1.0)
    parser.add_argument("--rgb-shift", type=float, default=None)
    args = parser.parse_args()
    if not 1 <= args.frames <= 100_000 or not 1 <= args.pty_seconds <= 60 or not 0 <= args.pty_warmup_seconds <= 30:
        parser.error("invalid frame count or bounded PTY measurement duration")
    if not (.1 <= args.speed <= 3 and 0 <= args.intensity <= 4 and 0 <= args.bleach <= 4
            and 0 <= args.rain <= 60 and .25 <= args.drop_size <= 3
            and (args.rgb_shift is None or 0 <= args.rgb_shift <= 12)):
        parser.error("invalid water/optics controls")
    cpu_quality = "balanced" if args.quality == "auto" else args.quality
    optics_args = ["--water", args.water, "--scene", "journey", "--speed", str(args.speed),
                   "--intensity", str(args.intensity), "--bleach", str(args.bleach),
                   "--drop-size", str(args.drop_size)]
    if args.rgb_shift is not None:
        optics_args += ["--rgb-shift", str(args.rgb_shift)]
    requested_controls = {"water": args.water, "quality": args.quality, "speed": args.speed,
                          "intensity": args.intensity, "bleach": args.bleach, "rain": args.rain,
                          "drop_size": args.drop_size, "rgb_shift": args.rgb_shift,
                          "rgb_effects": "on (verified help)", "seed": 7, "scene": "journey"}
    binaries = {"before": args.before.resolve(), "after": args.after.resolve()}
    args.output = args.output.resolve()
    compiler = subprocess.check_output(["rustc", "-vV"], text=True)
    proofs = {}
    binary_hashes = {}
    archive_hashes = {}
    source_hashes = {}
    cargo_hashes = {}
    default_blending = {}
    for version, binary in binaries.items():
        proof = json.loads((binary.parent / "provenance.json").read_text())
        archive = binary.parent / "source.tar.gz"
        binary_hashes[version] = sha(binary)
        archive_hashes[version] = sha(archive)
        assert proof["rebuilt_native_binary_sha256"] == binary_hashes[version]
        assert proof["archive_sha256"] == archive_hashes[version]
        assert proof["rustflags"] == "-C target-cpu=native" and proof["rustc"] == compiler
        with tarfile.open(archive) as bundle:
            archived = {m.name: hashlib.sha256(bundle.extractfile(m).read()).hexdigest()
                        for m in bundle.getmembers() if m.isfile()}
            main_source = bundle.extractfile("src/main.rs").read().decode()
            if args.rgb_shift is not None:
                cli_source = bundle.extractfile("src/cli.rs").read().decode()
                assert re.search(r"\bpub\s+rgb_shift\s*:", cli_source), \
                    f"{version} cannot express the requested matched --rgb-shift setting"
            interpolation_default = re.search(r"let\s+mut\s+interpolation\s*=\s*(true|false)\s*;", main_source)
            default_blending[version] = ("on" if interpolation_default[1] == "true" else "off") \
                if interpolation_default else "unavailable in legacy source"
        source_hashes[version] = {name: value for name, value in archived.items() if name.endswith(".rs")}
        cargo_hashes[version] = {name: archived[name] for name in ("Cargo.toml", "Cargo.lock")}
        assert proof["source_sha256"] == source_hashes[version]
        assert proof["cargo_files_sha256"] == cargo_hashes[version]
        build = json.loads((binary.parent / "native-build.json").read_text())
        assert [item["argv"] for item in build] == [
            ["cargo", "check", "--locked"], ["cargo", "build", "--release", "--locked"]]
        assert all(item["returncode"] == 0 for item in build)
        assert build[-1]["env"]["RUSTFLAGS"] == proof["rustflags"]
        proofs[version] = proof

    def check_inputs() -> None:
        for version, binary in binaries.items():
            assert sha(binary) == binary_hashes[version], "protected executable changed"
            assert sha(binary.parent / "source.tar.gz") == archive_hashes[version], "source archive changed"
        assert source_hashes["after"] == {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src").glob("*.rs"))}, \
            "after source differs from the protected build"
        assert cargo_hashes["after"] == {name: sha(ROOT / name) for name in ("Cargo.toml", "Cargo.lock")}

    check_inputs()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "utc_started": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "binary_sha256": binary_hashes, "source_archive_sha256": archive_hashes,
        "build_provenance": proofs, "rustc": compiler, "section": args.section,
        "sizes": SIZES, "frames": args.frames, "warmup_frames": 30,
        "pty_seconds": args.pty_seconds, "pty_warmup_seconds": args.pty_warmup_seconds,
        "pty_requested_fps": 60,
        "source_verified_default_blending": default_blending,
        "baseline_blending_semantics": "baseline leaves i untouched and uses the frozen executable's default; it does not mean blending is disabled",
        "script_sha256": {"benchmark_clarity.py": sha(Path(__file__)), "measure_native_pty.py": sha(PTY_RUNNER)},
        "clock_semantics": {
            "pty": "SYNC marker timestamps measure displayed/output frames, including interpolated frames. They do not measure simulation or source-render FPS. No native terminal emulator is present.",
            "cpu": f"One fixed-60Hz fixture step advances scene time by speed/60 = {args.speed / 60:g}s and physical water by speed/240 = {args.speed / 240:g}s, then computes one source image and encodes ANSI synchronously. The headless water grid is uncapped at viewport dimensions in both versions; the interactive simulation cap is excluded, as are the presentation worker and blended intermediate frames. This does not measure live source-render FPS.",
            "window": "Every PTY version uses the same wall-time warmup and measurement duration; a faster presentation worker therefore contributes more displayed frames without shortening the physical observation window.",
        },
        "controls": requested_controls,
        "cpu_fixture": {"quality": cpu_quality, "rain_impacts_per_physical_second": 8,
                        "rain_note": "The immutable headless fixture ignores the interactive --rain option. Both versions retain the same deterministic 8/s schedule; only PTY measures the requested interactive rain rate."},
        "commands": [], "pty_results": [], "cpu_results": [],
    }

    def save() -> None:
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def run(command: list[str], label: str) -> subprocess.CompletedProcess:
        check_inputs()
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(command, capture_output=True, text=True)
        metadata["commands"].append({"case": label, "argv": command, "started_utc": started,
                                     "completed_utc": datetime.now(timezone.utc).isoformat(),
                                     "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        save()
        result.check_returncode()
        check_inputs()
        return result

    save()
    if args.section in ("all", "pty"):
        for width, height in SIZES:
            for version, blending in (("before", "baseline"), ("after", "on"), ("after", "off")):
                label = f"{version}-{blending}-{width}x{height}"
                folder = args.output / "pty" / label
                command = ["uv", "run", "--script", str(PTY_RUNNER), "--binary", str(binaries[version]),
                           "--output", str(folder), "--sizes", f"{width}x{height}", "--blending", blending,
                           "--seconds", str(args.pty_seconds), "--warmup-seconds", str(args.pty_warmup_seconds),
                           "--water", args.water, "--quality", args.quality, "--speed", str(args.speed),
                           "--intensity", str(args.intensity), "--bleach", str(args.bleach),
                           "--rain", str(args.rain), "--drop-size", str(args.drop_size)]
                if args.rgb_shift is not None:
                    command += ["--rgb-shift", str(args.rgb_shift)]
                result = run(command, label)
                report = json.loads((folder / "metadata.json").read_text())
                assert report["binary_sha256"] == binary_hashes[version]
                assert len(report["cases"]) == 1
                case = {"version": version, **report["cases"][0]}
                assert case["size"] == [width, height] and case["blending"] == blending
                expected_argv = [str(binaries[version]), "--fps", "60", "--quality", args.quality, "--seed", "7",
                                 "--water", args.water, "--scene", "journey", "--speed", str(args.speed),
                                 "--intensity", str(args.intensity), "--bleach", str(args.bleach),
                                 "--rain", str(args.rain), "--drop-size", str(args.drop_size)]
                if args.rgb_shift is not None:
                    expected_argv += ["--rgb-shift", str(args.rgb_shift)]
                assert case["argv"] == expected_argv
                assert case["requested_controls"] == requested_controls
                assert case["exit_code"] == 0 and case["terminal_restored"]
                assert case["simulation_fps"] is None and case["source_render_fps"] is None
                case["blending_label"] = f"baseline (default {default_blending[version]})" \
                    if blending == "baseline" else f"{blending} via i"
                if blending != "baseline":
                    assert case["controls"][-1]["state"] == blending and all(c["latency_ms"] <= 1000 for c in case["controls"])
                metadata["pty_results"].append(case)
                save()
                print(result.stdout, end="", flush=True)

    if args.section in ("all", "cpu"):
        reference_times = None
        for width, height in SIZES:
            for version, binary in binaries.items():
                label = f"{version}-{args.water}-{cpu_quality}-{width}x{height}"
                folder = args.output / "cpu"
                folder.mkdir(exist_ok=True)
                report = folder / f"{label}.csv"
                command = [str(binary), "--benchmark", "--quality", cpu_quality,
                           "--size", f"{width}x{height}", "--frames", str(args.frames),
                           *optics_args, "--report", str(report)]
                result = run(command, label)
                parsed = list(csv.DictReader(io.StringIO(result.stdout)))
                assert len(parsed) == 1
                row = {"version": version, **parsed[0]}
                assert row["flow"] == args.water and row["quality"] == cpu_quality and row["scene"] == "journey"
                assert (int(row["width"]), int(row["height"])) == (width, height)
                assert int(row["measured_frames"]) == args.frames and int(row["warmup_frames"]) == 30
                assert all(math.isfinite(float(row[key])) and float(row[key]) >= 0 for key in ("mean_ms", "p95_ms"))
                samples = list(csv.DictReader(report.with_suffix(".samples.csv").open()))
                assert len(samples) == args.frames and [int(s["frame"]) for s in samples] == list(range(args.frames))
                times = [float(s["elapsed_seconds"]) for s in samples]
                assert all(math.isclose(actual, 10 + (31 + i) * args.speed / 60, rel_tol=2e-6, abs_tol=5e-6) for i, actual in enumerate(times))
                assert all(math.isfinite(float(s["work_ms"])) and float(s["work_ms"]) >= 0 for s in samples)
                if reference_times is None:
                    reference_times = times
                else:
                    assert times == reference_times, "fixed source-frame scenes are not time-matched"
                assert report.read_text() == result.stdout
                metadata["cpu_results"].append(row)
                save()
                print(f"{label}: source-frame work {row['mean_ms']} ms, p95 {row['p95_ms']} ms", flush=True)
    lines = ["# Source-frame work and displayed-frame cadence", "",
             "Two clocks are reported separately. Intermediate blended frames count toward display cadence; they do not represent new physics or rendered source images. PTY output includes write backpressure but excludes native terminal-emulator rendering.", ""]
    if metadata["pty_results"]:
        lines += ["| Size | Version | Blending | Display FPS | Interval mean ms | p95 ms | Max ms |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
        for r in metadata["pty_results"]:
            interval = r["interval_ms"]
            lines.append(f"| {r['size'][0]}×{r['size'][1]} | {r['version']} | {r['blending_label']} | {r['display_fps']:.2f} | {interval['mean']:.3f} | {interval['p95']:.3f} | {interval['max']:.3f} |")
        lines += ["", f"Each PTY case uses {args.pty_warmup_seconds:g}s warmup and {args.pty_seconds:g}s measurement. Identical explicit controls, verified in help: {json.dumps(requested_controls, sort_keys=True)}. Before baseline leaves i untouched: its frozen source default is {default_blending['before']}. After on/off states are selected through i and confirmed by the HUD. Source-render and simulation FPS remain unmeasured by these display traces.", ""]
    if metadata["cpu_results"]:
        lines += ["| Size | Version | Source-frame mean ms | p95 ms |", "| --- | --- | ---: | ---: |"]
        for r in metadata["cpu_results"]:
            lines.append(f"| {r['width']}×{r['height']} | {r['version']} | {float(r['mean_ms']):.3f} | {float(r['p95_ms']):.3f} |")
        lines += ["", f"Fixed fixture: {cpu_quality}, 30 warmup + {args.frames} measured source frames; speed{args.speed:g} and identical scene timestamps. Cost includes physics, source rendering and synchronous ANSI encoding. Both headless versions use an uncapped water grid at viewport dimensions, so this excludes the interactive simulation cap and the presentation worker. The headless fixture retains its eight-impacts-per-physical-second rain schedule, while PTY uses the requested {args.rain:g}/s. These are complete-version comparisons; images intentionally differ.", ""]
    (args.output / "comparison.md").write_text("\n".join(lines))
    metadata["utc_completed"] = datetime.now(timezone.utc).isoformat()
    save()
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
