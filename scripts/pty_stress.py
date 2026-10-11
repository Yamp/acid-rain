#!/usr/bin/env python3
"""Stress the real terminal with deterministic water-mode, resize and rain changes.

Run after cargo check and a native release build:
    uv run --no-project python scripts/pty_stress.py --report benchmarks/pty-stress.md
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import termios
import time

from pty_smoke import ROOT, TerminalProcess, require


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/acid-rain")
    parser.add_argument("--report", type=Path, default=ROOT / "benchmarks/pty-stress.md")
    parser.add_argument("--operations", type=int, default=175)
    parser.add_argument("--seed", type=int, default=731)
    options = parser.parse_args()
    if not 7 <= options.operations <= 2000:
        parser.error("operations must be between 7 and 2000")
    binary = options.binary.resolve()
    require(binary.is_file() and os.access(binary, os.X_OK), f"build the release executable first: {binary}")
    options.report.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(options.seed)
    categories = ["resize", "quality", "drain", "rapids", "burst", "rain", "drop size"]
    operations = [categories[index % len(categories)] for index in range(options.operations)]
    rng.shuffle(operations)
    counts: Counter[str] = Counter()
    log: list[dict] = []
    evidence = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "seed": options.seed,
        "requested_operations": options.operations,
        "frame_semantics": "PTY display frames, including intermediate blended frames; not source-render or physics steps",
        "startup_args": ["--water", "drain", "--rain", "60", "--drop-size", "3",
                         "--speed", "3", "--intensity", "2", "--bleach", "1"],
        "startup_mode_transitions": [],
        "operations": log,
    }
    sizes = [(1, 1), (1, 80), (160, 1), (2, 2), (80, 24), (120, 40), (200, 60), (17, 7), (3, 53)]
    terminal = TerminalProcess(binary, 120, 40, tuple(evidence["startup_args"]))
    failed = False
    start_time = time.monotonic()
    water_mode = "drain"

    def change_water(key: bytes) -> dict:
        nonlocal water_mode
        previous = water_mode
        selected = "drain" if key == b"d" else "rapids"
        expected = "classic" if previous == selected else selected
        # A one-cell terminal intentionally cannot show the HUD. Give mode
        # transitions enough room so their actual state is checked as well.
        terminal.resize(120, 40)
        terminal.wait_frames(2)
        start = len(terminal.data)
        terminal.send(key)
        label = f"Water: {expected.title()}".encode("ascii")
        terminal.wait_for(lambda: label in terminal.data[start:], f"{key.decode()} switches to {expected}")
        water_mode = expected
        return {"input_hex": key.hex(), "water_before": previous, "water_after": expected,
                "verified_hud": label.decode("ascii")}

    try:
        terminal.wait_frames(2)
        start = len(terminal.data)
        terminal.send(b"h")
        terminal.wait_for(lambda: b"ACID RAIN / CONTROLS" in terminal.data[start:], "Drain startup help")
        require(b"Water: Drain" in terminal.data[start:], "--water drain did not start Drain")
        terminal.send(b"h")
        terminal.wait_frames(2)
        counts["drain startup"] += 1
        # Cover all directed transitions, independent of the shuffled sequence.
        for key in (b"d", b"f", b"d", b"f", b"f", b"d"):
            transition = change_water(key)
            terminal.wait_frames(2)
            evidence["startup_mode_transitions"].append(transition)
            counts["verified startup mode transitions"] += 1
        for index, category in enumerate(operations):
            entry = {"index": index, "category": category, "frames_before": terminal.frames}
            log.append(entry)
            if index % 25 == 0:
                terminal.resize(0, 0)
                terminal.drain(0.04)
                require(terminal.poll() is None, "process exited at zero-size terminal")
                terminal.resize(120, 40)
                terminal.wait_frames(2)
                counts["zero-size recovery"] += 1
                entry["zero_size_recovery"] = True
            if category == "resize":
                size = rng.choice(sizes)
                entry["size"] = size
                terminal.resize(*size)
            elif category in ("drain", "rapids"):
                entry.update(change_water(b"d" if category == "drain" else b"f"))
            else:
                key = {
                    "quality": lambda: rng.choice([b"0", b"1", b"2", b"3", b"k"]),
                    "burst": lambda: b"r" * rng.randint(1, 3),
                    "rain": lambda: rng.choice([b"\x1b[A" * 35, b"\x1b[B" * 35, b"R"]),
                    "drop size": lambda: rng.choice([b"." * 15, b"," * 15]),
                }[category]()
                entry["input_hex"] = key.hex()
                terminal.send(key)
            terminal.wait_frames(2)
            require(terminal.poll() is None, f"process exited after operation {index}: {category}")
            entry["frames_after"] = terminal.frames
            entry["result"] = "PASS"
            counts[category] += 1
        terminal.finish(b"q")
        counts["clean exit and terminal restoration"] += 1
    except (AssertionError, OSError, termios.error, UnicodeError, subprocess.SubprocessError) as error:
        failed = True
        evidence["error"] = str(error)
        evidence["terminal_tail"] = bytes(terminal.data[-8000:]).decode("utf-8", errors="replace")
    finally:
        evidence["elapsed_seconds"] = time.monotonic() - start_time
        evidence["frames"] = terminal.frames
        evidence["last_water_mode"] = water_mode
        evidence["result"] = "FAIL" if failed else "PASS"
        terminal.close()
        options.report.with_suffix(".json").write_text(json.dumps(evidence, indent=2) + "\n")
    table = ["| Check | Completed | Result |", "| --- | ---: | --- |"]
    table += [f"| {category} | {count} | PASS |" for category, count in sorted(counts.items())]
    report = ["# Deterministic PTY stress", "", f"Binary: {binary}",
              f"SHA256: {evidence['binary_sha256']}", f"Seed: {options.seed}",
              f"Startup: {' '.join(evidence['startup_args'])}", "", *table, "",
              f"Display frames: {terminal.frames}; elapsed: {evidence['elapsed_seconds']:.2f}s.",
              "Each operation waits for two display frames, including any blended intermediate frames; these counts are not source-render or physics steps. Zero-size recovery runs every 25 operations.",
              "Six startup transitions cover every directed Classic/Rapids/Drain switch. Every d/f transition verifies its HUD at 120x40.",
              "The JSON sidecar records every operation, resize, keyboard byte sequence and frame count.",
              f"Overall: {evidence['result']}"]
    if failed:
        report += ["", f"Failure: {evidence['error']}"]
    text = "\n".join(report) + "\n"
    options.report.write_text(text)
    print(text, end="")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
