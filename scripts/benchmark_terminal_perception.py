#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Compare fixed source work and fast-reader PTY output in separate runs.

Run cargo check and build the two binaries before using this script. Examples:
  uv run --script scripts/benchmark_terminal_perception.py --before /path/before \
      --variant before --phase source --output benchmarks/terminal-perception
  uv run --script scripts/benchmark_terminal_perception.py --after /path/after \
      --variant after --phase all --output benchmarks/terminal-perception
  uv run --script scripts/benchmark_terminal_perception.py --phase report \
      --output benchmarks/terminal-perception

All cases execute sequentially. Run without competing builds or render jobs.
The baseline has no --lsd option; both use their default retinal model, with
subjective intensity 1 in the new implementation. No timing jobs run on import.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import select
import shlex
import statistics
import struct
import subprocess
import sys
import termios
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CASES = ((200, 60, "high"), (320, 100, "high"), (320, 100, "balanced"))
PTY_CASES = ((200, 60, "high"), (320, 100, "high"))
CONTROLS = ["--water", "classic", "--scene", "journey", "--speed", "2.4",
            "--intensity", "4", "--rgb-shift", "2", "--rain", "32",
            "--drop-size", "1.5", "--bleach", "1"]


class SyncMarkers:
    """Find split ANSI markers without counting their retained prefix twice."""

    def __init__(self) -> None:
        self.tail = b""

    def feed(self, chunk: bytes, stamp: int) -> list[tuple[int, str]]:
        combined = self.tail + chunk
        found = []
        for marker, label in ((b"\x1b[?2026h", "begin"), (b"\x1b[?2026l", "end")):
            start = 0
            while (index := combined.find(marker, start)) >= 0:
                if index + len(marker) > len(self.tail):
                    found.append((index, label))
                start = index + len(marker)
        self.tail = combined[-7:]
        return [(stamp, label) for _, label in sorted(found)]


def interval_summary(stamps: list[int]) -> dict:
    if len(stamps) < 2:
        raise ValueError("fewer than two synchronized display updates were captured")
    intervals = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]
    if min(intervals) < 0 or not any(intervals):
        raise ValueError("invalid or entirely coalesced output timestamps")
    ordered = sorted(intervals)
    mean = statistics.mean(intervals)
    return {"mean_ms": mean, "p50_ms": ordered[(len(ordered) - 1) // 2],
            "p95_ms": ordered[(len(ordered) - 1) * 95 // 100], "max_ms": max(ordered),
            "display_fps": 1000 / mean, "intervals": len(intervals),
            "coalesced_intervals": sum(value == 0 for value in intervals)}


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def case_name(case: tuple[int, int, str]) -> str:
    return f"{case[0]}x{case[1]}-{case[2]}"


def source_case(binary: Path, directory: Path, case: tuple[int, int, str], frames: int) -> dict:
    width, height, quality = case
    prefix = directory / case_name(case)
    report = prefix.with_suffix(".csv")
    capture = prefix.with_suffix(".ppm")
    command = [str(binary), "--benchmark", "--frames", str(frames), "--size",
               f"{width}x{height}", "--quality", quality, "--fps", "60", "--seed", "7",
               *CONTROLS, "--report", str(report), "--snapshot", str(capture)]
    print(f"Source: {shlex.join(command)}", flush=True)
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
    prefix.with_suffix(".stdout.txt").write_text(result.stdout)
    prefix.with_suffix(".stderr.txt").write_text(result.stderr)
    result.check_returncode()
    with report.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one summary row in {report}")
    row = rows[0]
    expected = {"width": str(width), "height": str(height), "quality": quality,
                "scene": "journey", "flow": "classic", "measured_frames": str(frames),
                "warmup_frames": "30"}
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"unexpected source fixture: {row}")
    numeric = {key: float(row[key]) for key in ("mean_ms", "p50_ms", "p95_ms", "bytes_per_frame")}
    if not all(math.isfinite(value) and value >= 0 for value in numeric.values()):
        raise ValueError("nonfinite or negative source measurements")
    samples = report.with_suffix(".samples.csv")
    with samples.open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != frames or any(int(row["frame"]) != i for i, row in enumerate(raw)):
        raise ValueError("source sample count or sequence does not match the requested fixture")
    header = f"P6\n{width} {height * 2}\n255\n".encode()
    image = capture.read_bytes()
    if not image.startswith(header) or len(image) != len(header) + width * height * 6:
        raise ValueError(f"invalid PPM capture: {capture}")
    return {"case": case_name(case), "argv": command, "command": shlex.join(command),
            "wall_seconds": time.monotonic() - started, "frames": frames, **numeric,
            "artifacts": {path.name: sha(path) for path in (report, samples, capture)}}


def load_pty_helper():
    path = ROOT / "scripts/pty_smoke.py"
    spec = importlib.util.spec_from_file_location("terminal_perception_pty", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pty_case(binary: Path, directory: Path, case: tuple[int, int, str],
             warmup: float, seconds: float) -> dict:
    helper = load_pty_helper()
    width, height, quality = case
    prefix = directory / case_name(case)
    command = [str(binary), "--fps", "60", "--quality", quality, "--seed", "7", *CONTROLS]
    print(f"PTY: {shlex.join(command)} [{width}x{height}, {warmup:g}s + {seconds:g}s]", flush=True)
    terminal = helper.TerminalProcess(binary, width, height, tuple(CONTROLS), quality=quality)
    events = []
    parser = SyncMarkers()
    captured = 0
    error_tail = b""
    runtime_error = False
    quit_at = None
    try:
        terminal.wait_for(lambda: terminal.frames >= 2, "initial display frames", timeout=30)
        offset = len(terminal.data)
        terminal.send(b"h")
        terminal.wait_for(lambda: b"ACID RAIN / CONTROLS" in terminal.data[offset:], "settings help")
        help_screen = bytes(terminal.data[offset:])
        # Assert the common baseline contract here: the current helper's full
        # startup assertion may acquire controls absent from the older binary.
        for expected in ("Quality: High", "Water: Classic", "Scene: journey", "Speed: 2.40x",
                         "Trip: 4.0", "RGB: on", "Interpolation: on", "Previous state: running",
                         "Rain: 32.0/s", "Drop size: 1.50x", "Bleach: 1.00x", "RGB shift: 2.00px"):
            if expected.encode() not in help_screen:
                raise ValueError(f"PTY settings help does not contain {expected!r}")
        prefix.with_suffix(".help.ansi").write_bytes(help_screen)
        terminal.send(b"h")
        terminal.wait_frames(2)
        terminal.drain(0.05)
        start = time.monotonic_ns()
        lower = start + round(warmup * 1e9)
        upper = lower + round(seconds * 1e9)
        # Keep only a tail in memory; raw timestamp CSV is the cadence evidence.
        # No compression or full ANSI dump competes with the measured reader.
        terminal.data[:] = terminal.data[-32768:]
        while True:
            now = time.monotonic_ns()
            if now >= upper and quit_at is None:
                terminal.send(b"q")
                quit_at = now
            if quit_at is not None and now - quit_at > 5_000_000_000:
                raise TimeoutError("PTY did not quit within five seconds")
            ready = select.select([terminal.master], [], [], 0.02)[0]
            if ready:
                try:
                    chunk = os.read(terminal.master, 262144)
                except BlockingIOError:
                    continue
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    chunk = b""
                stamp = time.monotonic_ns()
                if chunk:
                    events.extend(parser.feed(chunk, stamp))
                    captured += len(chunk)
                    combined = error_tail + chunk
                    runtime_error |= any(marker in combined.lower() for marker in
                                         (b"panicked at", b"fatal runtime error", b"error:"))
                    error_tail = combined[-32:]
                    terminal.frames += (terminal.marker_tail + chunk).count(helper.SYNC_BEGIN)
                    terminal.marker_tail = (terminal.marker_tail + chunk)[-(len(helper.SYNC_BEGIN) - 1):]
                    terminal.data.extend(chunk)
                    del terminal.data[:-32768]
            if terminal.poll() is not None and not ready:
                break
        if terminal.exit_code != 0 or quit_at is None or runtime_error:
            raise RuntimeError(f"PTY failed or exited early: status={terminal.exit_code}")
        if sys.platform == "darwin":
            fcntl.ioctl(terminal.slave, termios.FIONREAD, struct.pack("I", 0))
        if termios.tcgetattr(terminal.slave) != terminal.original_termios:
            raise RuntimeError("terminal attributes were not restored exactly")
        tail = bytes(terminal.data)
        last_draw = tail.rfind(helper.SYNC_BEGIN)
        if tail.rfind(helper.ALT_LEAVE) <= last_draw:
            raise RuntimeError("missing terminal restoration after the final drawing")
        for sequence in (b"\x1b[?25h", b"\x1b[?7h", b"\x1b[0m"):
            if sequence not in tail[max(last_draw, 0):]:
                raise RuntimeError(f"missing restoration sequence {sequence!r}")
    finally:
        prefix.with_suffix(".tail.ansi").write_bytes(bytes(terminal.data))
        with prefix.with_suffix(".timestamps.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("monotonic_ns", "event"))
            writer.writerows(events)
        terminal.close()
    stamps = [stamp for stamp, label in events if label == "begin" and lower <= stamp < upper]
    return {"case": case_name(case), "argv": command, "command": shlex.join(command),
            "size": [width, height], "warmup_seconds": warmup, "measured_seconds": seconds,
            "measurement_start_ns": lower, "measurement_end_ns": upper,
            "captured_bytes_including_warmup_and_shutdown": captured,
            "updates_in_measured_window": len(stamps),
            "window_updates_per_second": len(stamps) / seconds,
            "source_render_fps": None, "simulation_fps": None,
            "terminal_restored": True, **interval_summary(stamps)}


def metadata(binary: Path, variant: str, phase: str, args) -> dict:
    provenance_path = binary.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    return {"schema": 1, "variant": variant, "phase": phase,
            "started_utc": datetime.now(timezone.utc).isoformat(), "binary": str(binary),
            "binary_sha256": sha(binary), "provenance": provenance,
            "driver_sha256": sha(Path(__file__)),
            "pty_helper_sha256": sha(ROOT / "scripts/pty_smoke.py"),
            "python": sys.version, "platform": platform.platform(), "controls": CONTROLS,
            "lsd_argument": "omitted: baseline lacks it; new implementation default is 1",
            "frames": args.frames, "warmup_seconds": args.warmup_seconds,
            "measured_seconds": args.seconds, "status": "running", "cases": [],
            "scope": ("Counting sink: simulation + source shading + ANSI encoding; fixed 60 Hz "
                      "fixture, 30 warmup frames, fixed 8 impacts per physical second. --rain "
                      "does not alter this fixture. Grid is deliberately uncapped."
                      if phase == "source" else
                      "Fast PTY reader, no terminal emulator. Synchronized output includes "
                      "interpolated display frames. Markers in one read share a timestamp; "
                      "source-render and simulation FPS are not inferred. Interactive grid is capped.")}


def make_report(output: Path) -> str:
    lines = ["# Terminal perception comparison", "",
             "Source preparation and PTY output cadence are separate measurements.",
             "Fixed controls: Classic, Journey, speed 2.4, Trip 4, RGB radius 2, "
             "drop size 1.5, bleach 1, seed 7. LSD argument omitted in both versions.", ""]
    for phase, cases in (("source", SOURCE_CASES), ("pty", PTY_CASES)):
        records = {}
        for variant in ("before", "after"):
            path = output / variant / f"{phase}.json"
            if path.exists():
                records[variant] = json.loads(path.read_text())
        if phase == "source":
            lines += ["## Source work", "", "Milliseconds per prepared source frame; no PTY or emulator.", "",
                      "| Case | Before mean | After mean | Before p95 | After p95 | Speedup |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"]
        else:
            lines += ["", "## PTY output", "", "Display updates include interpolated frames. "
                      "Rain 32/s; fast reader without a terminal emulator.", "",
                      "| Case | Before updates/s | After updates/s | Before p95 ms | After p95 ms |",
                      "| --- | ---: | ---: | ---: | ---: |"]
        for case in cases:
            name = case_name(case)
            entries = {variant: next((row for row in data["cases"] if row["case"] == name), None)
                       for variant, data in records.items()}
            before, after = entries.get("before"), entries.get("after")
            metric = "mean_ms" if phase == "source" else "window_updates_per_second"
            format_value = lambda entry, key: "—" if entry is None else f"{entry[key]:.3f}"
            row = [name, format_value(before, metric), format_value(after, metric),
                   format_value(before, "p95_ms"), format_value(after, "p95_ms")]
            if phase == "source":
                matched = before and after and before["frames"] == after["frames"]
                row.append(f"{before['mean_ms'] / after['mean_ms']:.3f}×"
                           if matched and after["mean_ms"] > 0 else "—")
            lines.append("| " + " | ".join(row) + " |")
        for variant, data in records.items():
            lines += ["", f"{variant}: `{data['binary']}`; SHA-256 `{data['binary_sha256']}`; "
                      f"status `{data['status']}`. See `{variant}/{phase}.json` for exact commands and scope."]
    lines += ["", "Do not compare these results with concurrent builds/renderers or different "
              "hardware. Source speedup is omitted when measured frame counts differ.", ""]
    report = "\n".join(lines)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.md").write_text(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--variant", choices=("before", "after", "both"), default="both")
    parser.add_argument("--phase", choices=("source", "pty", "all", "report"), default="source")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/terminal-perception")
    parser.add_argument("--frames", type=int, choices=(90, 120), default=120)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not (0 <= args.warmup_seconds <= 30 and 1 <= args.seconds <= 60):
        parser.error("warmup must be 0..30 seconds and measurement 1..60 seconds")
    output = args.output.resolve()
    if args.phase == "report":
        print(make_report(output))
        return
    variants = ("before", "after") if args.variant == "both" else (args.variant,)
    binaries = {}
    for variant in variants:
        path = getattr(args, variant)
        if path is None or not path.is_file() or not os.access(path, os.X_OK):
            parser.error(f"--{variant} must name an executable binary")
        binaries[variant] = path.resolve()
    phases = ("source", "pty") if args.phase == "all" else (args.phase,)
    try:
        for variant in variants:
            binary = binaries[variant]
            for phase in phases:
                directory = output / variant / phase
                directory.mkdir(parents=True, exist_ok=True)
                record = metadata(binary, variant, phase, args)
                path = output / variant / f"{phase}.json"
                save_json(path, record)
                try:
                    cases = SOURCE_CASES if phase == "source" else PTY_CASES
                    for case in cases:
                        if sha(binary) != record["binary_sha256"]:
                            raise RuntimeError("binary changed during the benchmark")
                        result = (source_case(binary, directory, case, args.frames) if phase == "source"
                                  else pty_case(binary, directory, case, args.warmup_seconds, args.seconds))
                        record["cases"].append(result)
                        save_json(path, record)
                    if sha(binary) != record["binary_sha256"]:
                        raise RuntimeError("binary changed during the benchmark")
                    record["status"] = "complete"
                except BaseException as error:
                    record["status"] = "failed"
                    record["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
                    save_json(path, record)
    finally:
        print(make_report(output), flush=True)


def self_test() -> None:
    import tempfile

    parser = SyncMarkers()
    assert parser.feed(b"prefix\x1b[?20", 1) == []
    assert parser.feed(b"26himage\x1b[?2026l\x1b[?2026h", 2) == [
        (2, "begin"), (2, "end"), (2, "begin")]
    assert parser.feed(b"pixels", 3) == []
    result = interval_summary([0, 10_000_000, 30_000_000, 60_000_000])
    assert result["mean_ms"] == 20.0
    assert result["p50_ms"] == 20.0
    assert result["display_fps"] == 50.0
    try:
        interval_summary([1])
    except ValueError:
        pass
    else:
        raise AssertionError("insufficient cadence samples were accepted")
    with tempfile.TemporaryDirectory(prefix="terminal-perception-check-") as name:
        output = Path(name)
        before = {"binary": "/fixture/before", "binary_sha256": "test-only", "status": "complete",
                  "cases": [{"case": "200x60-high", "mean_ms": 20, "p95_ms": 24, "frames": 90}]}
        save_json(output / "before/source.json", before)
        report = make_report(output)
        assert "| 200x60-high | 20.000 | — | 24.000 | — | — |" in report
        after = {**before, "binary": "/fixture/after", "cases": [
            {"case": "200x60-high", "mean_ms": 10, "p95_ms": 12, "frames": 90}]}
        save_json(output / "after/source.json", after)
        assert "| 200x60-high | 20.000 | 10.000 | 24.000 | 12.000 | 2.000× |" in make_report(output)
        after["cases"][0]["frames"] = 120
        save_json(output / "after/source.json", after)
        assert "2.000×" not in make_report(output)
    print("PASS: split/coalesced markers, interval statistics, partial reports and unmatched fixtures")


if __name__ == "__main__":
    main()
