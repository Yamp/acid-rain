#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Measure the release renderer sequentially and save a baseline comparison."""
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

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/acid-rain")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks")
    parser.add_argument("--baseline", type=Path, default=ROOT / "benchmarks/baseline.csv")
    args = parser.parse_args()
    motion = dict(re.findall(r"pub const ([A-Z_]+): f32 = ([0-9.]+);", (ROOT / "src/motion.rs").read_text()))
    scene_speed = float(motion["DEFAULT_SPEED"])
    surface_scale = float(motion["SURFACE_TIME_SCALE"])
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    rows = []
    commands = []
    for quality in ("high", "balanced", "low"):
        for width, height in ((120, 40), (200, 60), (320, 100)):
            destination = args.output / f"current-{quality}-{width}x{height}.csv"
            command = [str(binary), "--benchmark", "--quality", quality,
                       "--size", f"{width}x{height}", "--frames", str(args.frames),
                       "--report", str(destination)]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            commands.append({"argv": command, "stdout": result.stdout, "stderr": result.stderr})
            row = next(csv.DictReader(io.StringIO(result.stdout)))
            rows.append(row)
            print(f"{width}x{height} {quality}: mean {row['mean_ms']} ms; p95 {row['p95_ms']} ms", flush=True)
    with (args.output / "current.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    baseline_path = args.baseline
    baseline = {}
    if baseline_path.exists():
        with baseline_path.open() as handle:
            baseline = {(r["width"], r["height"], r.get("quality")): r for r in csv.DictReader(handle)}
    lines = ["# Renderer benchmark comparison", "",
             "CPU frame preparation into a counting sink; terminal emulation and transport are excluded.",
             f"Both use a 60 Hz frame clock and matching dimensions/warmup/frame counts. Current scene time is {scene_speed:g}x wall time; surface time is {surface_scale:g}x scene time, with eight impacts per surface second. Earlier captures use their earlier default motion rates.",
             "Current scenes, water detail, optics and temporal reconstruction differ from the baseline image.",
             "Interactive mode caps its simulation grid; these benchmarks deliberately do not.", "",
             "| Cells | Quality | Before mean, ms | After mean, ms | After p95, ms | Speedup | Output bytes/frame |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        old = baseline.get((row["width"], row["height"], row["quality"])) or baseline.get((row["width"], row["height"], None))
        before = float(old["mean_ms"]) if old else None
        mean = float(row["mean_ms"])
        speedup = f"{before / mean:.2f}×" if before else "—"
        lines.append(f"| {row['width']}×{row['height']} | {row['quality']} | "
                     f"{before:.3f} | {mean:.3f} | {float(row['p95_ms']):.3f} | {speedup} | "
                     f"{float(row['bytes_per_frame']):,.0f} |" if before else
                     f"| {row['width']}×{row['height']} | {row['quality']} | — | {mean:.3f} | "
                     f"{float(row['p95_ms']):.3f} | — | {float(row['bytes_per_frame']):,.0f} |")
    lines += ["", "Raw per-frame samples are in `current-*.samples.csv`; metadata and commands are in `current-metadata.json`.",
              f"Compared with `{baseline_path.relative_to(ROOT) if baseline_path.is_relative_to(ROOT) else baseline_path}`. Source snapshots for prior deliveries are retained beside their baseline tables.",
              "", "Reproduce: `RUSTFLAGS='-C target-cpu=native' cargo build --release --locked`, then",
              f"`{shlex.join(['uv', 'run', '--script', 'scripts/benchmark.py', '--baseline', str(baseline_path)])}` with no competing builds or render jobs.", ""]
    report = "\n".join(lines)
    (args.output / "comparison.md").write_text(report)
    metadata = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "rustc": subprocess.check_output(["rustc", "-vV"], text=True),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree": "uncommitted implementation; source hashes below",
        "baseline_csv": str(baseline_path),
        "scene_time_scale": scene_speed,
        "surface_time_scale": surface_scale,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "src").glob("*.rs"))},
        "commands": commands,
        "note": "Native release build. Fixed high/balanced/low, no quality adaptation during benchmarks.",
    }
    (args.output / "current-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
