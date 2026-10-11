#!/usr/bin/env python3
"""Exercise the real terminal CLI without a GUI or third-party dependencies.

Run after cargo check and cargo build --release:
    uv run --no-project python scripts/pty_smoke.py
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time


ROOT = Path(__file__).resolve().parents[1]
SYNC_BEGIN = b"\x1b[?2026h"
SYNC_END = b"\x1b[?2026l"
ALT_ENTER = b"\x1b[?1049h"
ALT_LEAVE = b"\x1b[?1049l"
# Keep the controlling session alive after the program exits. macOS revokes the
# slave when its session leader exits, preventing post-exit termios inspection.
SUPERVISOR = """
import os, subprocess, sys
status_fd, release_fd = map(int, sys.argv[1:3])
child = subprocess.Popen(sys.argv[3:])
os.write(status_fd, f"PID {child.pid}\\n".encode())
status = child.wait()
os.write(status_fd, f"EXIT {status}\\n".encode())
os.read(release_fd, 1)
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def controlling_terminal() -> None:
    """Called in the child after Popen creates its new session."""
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


def cpu_seconds(pid: int) -> float:
    """Read cumulative process CPU time; ps works on macOS and Linux."""
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "time="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    days = 0
    if "-" in result:
        day_text, result = result.split("-", 1)
        days = int(day_text)
    seconds = 0.0
    for component in result.split(":"):
        seconds = seconds * 60 + float(component)
    return days * 86400 + seconds


class TerminalProcess:
    def __init__(self, binary: Path, width: int, height: int, extra_args: tuple[str, ...] = (),
                 *, quality: str | None = None) -> None:
        require(quality in (None, "auto", "high", "balanced", "low"), "invalid terminal quality")
        quality_args = () if quality is None else ("--quality", quality)
        self.master, self.slave = pty.openpty()
        self.original_termios = termios.tcgetattr(self.slave)
        self.data = bytearray()
        self.frames = 0
        self.marker_tail = b""
        self.exit_code = None
        self.set_size(width, height)
        status_read, status_write = os.pipe()
        release_read, self.release_write = os.pipe()
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-c", SUPERVISOR, str(status_write), str(release_read),
                 str(binary), "--fps", "60", *quality_args, "--seed", "7", *extra_args],
                stdin=self.slave,
                stdout=self.slave,
                stderr=self.slave,
                start_new_session=True,
                preexec_fn=controlling_terminal,
                env={**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"},
                close_fds=True,
                pass_fds=(status_write, release_read),
            )
        except BaseException:
            os.close(self.master)
            os.close(self.slave)
            os.close(status_read)
            os.close(self.release_write)
            raise
        finally:
            os.close(status_write)
            os.close(release_read)
        self.status = os.fdopen(status_read, "rb", buffering=0)
        require(select.select([self.status], [], [], 5.0)[0], "PTY supervisor did not start the program")
        message = self.status.readline().split()
        require(len(message) == 2 and message[0] == b"PID", f"invalid supervisor response: {message!r}")
        self.child_pid = int(message[1])
        os.set_blocking(self.master, False)

    def poll(self) -> int | None:
        if self.exit_code is None and select.select([self.status], [], [], 0)[0]:
            message = self.status.readline().split()
            require(len(message) == 2 and message[0] == b"EXIT", f"invalid supervisor exit: {message!r}")
            self.exit_code = int(message[1])
        return self.exit_code

    def set_size(self, width: int, height: int) -> None:
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))

    def resize(self, width: int, height: int) -> None:
        self.set_size(width, height)
        os.kill(self.child_pid, signal.SIGWINCH)

    def send(self, keys: bytes) -> None:
        require(self.poll() is None, "process exited before keyboard input")
        require(os.write(self.master, keys) == len(keys), "short PTY input write")

    def drain(self, duration: float) -> bytes:
        start = len(self.data)
        deadline = time.monotonic() + duration
        while (remaining := deadline - time.monotonic()) > 0:
            ready, _, _ = select.select([self.master], [], [], min(remaining, 0.05))
            if not ready:
                continue
            try:
                chunk = os.read(self.master, 262144)
            except BlockingIOError:
                continue
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            self.frames += (self.marker_tail + chunk).count(SYNC_BEGIN)
            self.marker_tail = (self.marker_tail + chunk)[-(len(SYNC_BEGIN) - 1):]
            self.data.extend(chunk)
        return bytes(self.data[start:])

    def wait_for(self, predicate, description: str, timeout: float = 8.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            require(time.monotonic() < deadline, f"timed out waiting for {description}")
            require(
                self.poll() is None,
                f"process exited while waiting for {description}: {bytes(self.data[-400:])!r}",
            )
            self.drain(0.05)

    def wait_frames(self, count: int = 2) -> None:
        target = self.frames + count
        self.wait_for(lambda: self.frames >= target, f"{count} display frame(s)")

    def finish(self, key: bytes) -> None:
        self.send(key)
        deadline = time.monotonic() + 5.0
        while self.poll() is None:
            require(time.monotonic() < deadline, "process did not exit after quit input")
            self.drain(0.05)
        self.drain(0.1)
        require(self.exit_code == 0, f"exit status was {self.exit_code}")
        if sys.platform == "darwin":
            # Restoring ICANON sets macOS's transient PENDIN state. Let the
            # line discipline settle via a non-consuming readiness query before
            # comparing every saved attribute; do not mask any termios flags.
            fcntl.ioctl(self.slave, termios.FIONREAD, struct.pack("I", 0))
        require(termios.tcgetattr(self.slave) == self.original_termios, "terminal attributes were not restored exactly")
        output = bytes(self.data)
        output.decode("utf-8")
        require(ALT_ENTER in output and ALT_LEAVE in output, "alternate screen was not entered and restored")
        require(output.rfind(ALT_LEAVE) > output.rfind(SYNC_BEGIN), "drawing continued after alternate-screen restoration")
        for sequence in [b"\x1b[?25h", b"\x1b[?7h", b"\x1b[0m"]:
            require(sequence in output[output.rfind(SYNC_BEGIN):], f"missing terminal restoration sequence {sequence!r}")
        require(output.count(SYNC_END) == self.frames + 1, "unbalanced frame synchronization or missing restoration")
        lower = output.lower()
        for marker in [b"panicked at", b"fatal runtime error", b"error:"]:
            require(marker not in lower, f"runtime error appeared in PTY output: {marker!r}")

    def close(self) -> None:
        active_error = sys.exc_info()[1]
        try:
            if self.poll() is None:
                os.kill(self.child_pid, signal.SIGKILL)
            os.write(self.release_write, b"x")
            deadline = time.monotonic() + 5.0
            # macOS can wait for queued PTY output when the session closes.
            # Continue consuming it even after killing a failed test's child.
            while self.process.poll() is None:
                if time.monotonic() >= deadline:
                    self.process.kill()
                    self.process.wait(timeout=1)
                    raise TimeoutError("PTY supervisor did not exit during cleanup")
                self.drain(0.05)
        except Exception as error:
            if active_error is None:
                raise
            print(f"PTY cleanup also failed: {error}", file=sys.stderr)
        finally:
            self.status.close()
            os.close(self.release_write)
            os.close(self.master)
            os.close(self.slave)


def require_startup_settings(screen: bytes, water: str = "Classic", *, rain: float = 32.0,
                             drop_size: float = 1.5, bleach: float = 1.0, rgb_shift: float = 2.0,
                             lsd: float = 1.0) -> None:
    """Independent user-requested contract; never derive expected values from Rust constants."""
    screen = screen[screen.index(b"ACID RAIN / CONTROLS"):]
    for expected in ("Quality: High", f"Water: {water}", "Scene: journey", "Speed: 2.40x", "Trip: 4.0",
                     "RGB: on", "Animation held | Interpolation: on", "Previous state: running",
                     f"Rain: {rain:.1f}/s", f"Drop size: {drop_size:.2f}x", f"Bleach: {bleach:.2f}x",
                     f"RGB shift: {rgb_shift:.2f}px", f"LSD: {lsd:.1f}"):
        require(expected.encode() in screen, f"startup setting missing: {expected}")


def run_smoke(binary: Path, rows: list[tuple[str, str, str]], artifact_dir: Path) -> None:
    terminal = TerminalProcess(binary, 120, 40)
    try:
        terminal.wait_frames()
        attributes = termios.tcgetattr(terminal.slave)
        require(not attributes[3] & (termios.ICANON | termios.ECHO), "interactive session did not enter raw mode")
        rows.append(("startup 120x40", "PASS", "raw mode; synchronized RGB frames"))
        for width, height in [(1, 1), (80, 24), (160, 50)]:
            start = len(terminal.data)
            terminal.resize(width, height)
            terminal.wait_for(
                lambda: SYNC_BEGIN + b"\x1b[2J" in terminal.data[start:],
                f"resize clear at {width}x{height}",
            )
            terminal.wait_frames()
            rows.append((f"resize {width}x{height}", "PASS", "clear inside synchronized frame; animation continues"))

        start = len(terminal.data)
        terminal.send(b"h")
        terminal.wait_for(lambda: b"ACID RAIN / CONTROLS" in terminal.data[start:], "default intensity in help")
        require_startup_settings(bytes(terminal.data[start:]))
        require(b"Toggle intermediate-frame blending" in terminal.data[start:],
                "help is missing the intermediate-frame blending control")
        require(b"n / m" in terminal.data[start:]
                and b"Reset subjective LSD to 1.0" in terminal.data[start:],
                "help is missing n/m adjustment or N reset for subjective LSD")
        terminal.send(b"h")
        terminal.wait_frames(2)
        rows.append(("default intensity", "PASS", "help starts at Trip: 4.0; animation resumes"))
        rows.append(("default speed and bleaching", "PASS", "help starts at Speed: 2.40x and Bleach: 1.00x"))
        rows.append(("default Classic", "PASS", "help reports Water: Classic without a startup flag"))
        rows.append(("default quality and rain", "PASS", "High, Rain: 32.0/s, Drop size: 1.50x, running Journey"))
        rows.append(("default frame blending", "PASS", "help reports Interpolation: on and the i binding"))
        rows.append(("default RGB shift", "PASS", "help reports RGB shift: 2.00px independently of Trip: 4.0"))
        rows.append(("default LSD and help", "PASS", "LSD: 1.0 alongside Trip: 4.0; n/m adjustment and N reset documented"))

        def interpolation_key(value: str, paused: bool = False) -> None:
            start = len(terminal.data)
            frames_before = terminal.frames
            sent = time.monotonic()
            terminal.send(b"i")
            expected = f"Frame blending: {value} | i: toggle".encode("ascii")
            terminal.wait_for(lambda: expected in terminal.data[start:],
                              f"i frame blending {value}", timeout=1.0)
            latency_ms = (time.monotonic() - sent) * 1000.0
            if paused:
                terminal.drain(0.2)
                require(terminal.frames == frames_before + 1,
                        "paused interpolation toggle did not redraw exactly once")
                require(not terminal.drain(0.3), "paused interpolation toggle restarted output")
            else:
                terminal.wait_frames(2)
            rows.append((f"i blending {value}{' paused' if paused else ''}", "PASS",
                         f"{expected.decode()}; HUD latency {latency_ms:.1f}ms <= 1000ms; "
                         + ("one frame then idle" if paused else "display output continues")))

        interpolation_key("off")
        start = len(terminal.data)
        terminal.send(b"h")
        terminal.wait_for(lambda: b"Animation held | Interpolation: off" in terminal.data[start:],
                          "help reports disabled blending", timeout=1.0)
        terminal.drain(0.15)
        require(not terminal.drain(0.3), "help with blending disabled continued output")
        terminal.send(b"h")
        terminal.wait_frames(2)
        rows.append(("help with blending off", "PASS", "live off state; output held; animation resumes"))
        interpolation_key("on")

        for label, key in [("next scene", b"g"), ("journey", b"t"), ("rain burst", b"r")]:
            terminal.send(key)
            terminal.wait_frames()
            rows.append((f"key {key.decode()} ({label})", "PASS", "animation continues without errors"))

        def intensity_key(key: bytes, label: str, value: float, paused: bool = False) -> None:
            start = len(terminal.data)
            frames_before = terminal.frames
            terminal.send(key)
            expected = f"Intensity: {value:.1f}x | +/-: adjust (0-4) | e: RGB trails".encode("ascii")
            terminal.wait_for(lambda: expected in terminal.data[start:], f"intensity HUD after {label}")
            if paused:
                terminal.drain(0.2)
                require(terminal.frames == frames_before + 1, "paused intensity control did not redraw exactly once")
                require(not terminal.drain(0.3), "paused intensity control restarted streaming")
            else:
                terminal.wait_frames()
            rows.append((f"intensity {label}{' paused' if paused else ''}", "PASS",
                         expected.decode() + "; " + ("one frame then idle" if paused else "animation continues")))

        def intensity_controls(paused: bool = False) -> None:
            for key, label, value in [
                (b"+", "increase", 2.1),
                (b"-", "decrease", 2.0),
                (b"+" * 60, "upper bound", 4.0),
                (b"-" * 80, "lower bound", 0.0),
                (b"=" * 20, "restore 2.0 with =", 2.0),
            ]:
                intensity_key(key, label, value, paused)
                if paused and value == 4.0:
                    start = len(terminal.data)
                    terminal.send(b"h")
                    terminal.wait_for(lambda: b"ACID RAIN / CONTROLS" in terminal.data[start:],
                                      "help at paused intensity 4.0")
                    require(b"Trip: 4.0" in terminal.data[start:], "help did not show the live intensity 4.0")
                    require(b"Previous state: paused" in terminal.data[start:], "intensity help lost pause state")
                    terminal.drain(0.15)
                    frames_before = terminal.frames
                    terminal.send(b"h")
                    terminal.wait_frames(1)
                    terminal.drain(0.2)
                    require(terminal.frames == frames_before + 1, "intensity help exit redrew more than once")
                    require(not terminal.drain(0.3), "intensity help exit resumed animation")
                    rows.append(("intensity 4.0 help while paused", "PASS",
                                 "Trip: 4.0; one exit redraw then idle; subsequent controls keep working"))

        intensity_key(b"-" * 20, "prepare 2.0 for control sweep", 2.0)
        intensity_controls()

        def rgb_shift_key(key: bytes, label: str, value: float, paused: bool = False) -> None:
            start = len(terminal.data)
            frames_before = terminal.frames
            terminal.send(key)
            expected = f"RGB shift: {value:.2f}px | j/l: less/more | J: reset".encode("ascii")
            terminal.wait_for(lambda: expected in terminal.data[start:], f"RGB shift HUD after {label}")
            if paused:
                terminal.drain(0.2)
                require(terminal.frames == frames_before + 1, "paused RGB shift did not redraw exactly once")
                require(not terminal.drain(0.3), "paused RGB shift restarted streaming")
            else:
                terminal.wait_frames()
            rows.append((f"RGB shift {label}{' paused' if paused else ''}", "PASS",
                         expected.decode() + "; " + ("one frame then idle" if paused else "animation continues")))

        def rgb_shift_help(value: float, intensity: float, label: str, paused: bool = False) -> None:
            start = len(terminal.data)
            terminal.send(b"h")
            terminal.wait_for(lambda: b"ACID RAIN / CONTROLS" in terminal.data[start:], "RGB shift help state")
            screen = bytes(terminal.data[start:])
            screen = screen[screen.index(b"ACID RAIN / CONTROLS"):]
            expected = f"RGB shift: {value:.2f}px".encode("ascii")
            require(expected in screen, "help lost the independent RGB shift setting")
            require(f"Trip: {intensity:.1f}".encode("ascii") in screen, "RGB shift changed psychedelic intensity")
            require(b"RGB: on" in screen, "RGB shift zero disabled the separate RGB effects toggle")
            require(b"Previous state: " + (b"paused" if paused else b"running") in screen,
                    "RGB shift help lost the pause state")
            terminal.drain(0.15)
            frames_before = terminal.frames
            terminal.send(b"h")
            terminal.wait_frames(1 if paused else 2)
            if paused:
                terminal.drain(0.2)
                require(terminal.frames == frames_before + 1, "RGB shift help exit redrew more than once")
                require(not terminal.drain(0.3), "RGB shift help exit resumed paused animation")
            rows.append((f"RGB shift {label}{' paused' if paused else ''}", "PASS",
                         f"{expected.decode()}; Trip: {intensity:.1f}; RGB toggle remains on; pause state preserved"))

        def rgb_shift_controls(paused: bool = False) -> None:
            for key, label, value in [
                (b"j", "decrease", 1.75),
                (b"l", "increase", 2.00),
                (b"j" * 20, "zero bound", 0.0),
                (b"l" * 60, "upper bound", 12.0),
                (b"J", "reset", 2.00),
            ]:
                rgb_shift_key(key, label, value, paused)
                if value == 0.0:
                    rgb_shift_help(0.0, 2.0, "zero disables displacement setting", paused)

        rgb_shift_controls()
        rgb_shift_key(b"lll", "nondefault before intensity changes", 2.75)
        intensity_key(b"+" * 20, "4.0 with independent RGB shift", 4.0)
        rgb_shift_help(2.75, 4.0, "unchanged at intensity 4.0")
        intensity_key(b"-" * 30, "1.0 with independent RGB shift", 1.0)
        rgb_shift_help(2.75, 1.0, "unchanged at intensity 1.0")
        intensity_key(b"=" * 10, "restore 2.0 after RGB shift independence", 2.0)
        rgb_shift_key(b"J", "reset after intensity changes", 2.00)

        def lsd_key(key: bytes, label: str, value: float, paused: bool) -> None:
            start = len(terminal.data)
            frames_before = terminal.frames
            terminal.send(key)
            expected = f"LSD: {value:.1f}x | n/m: less/more (0-2) | N: reset".encode("ascii")
            terminal.wait_for(lambda: expected in terminal.data[start:], f"LSD HUD after {label}")
            if paused:
                terminal.drain(0.2)
                require(terminal.frames == frames_before + 1,
                        "paused LSD control did not redraw exactly once")
                require(not terminal.drain(0.3), "paused LSD control restarted streaming")
            else:
                terminal.wait_frames()

        def lsd_controls(paused: bool = False) -> None:
            suffix = " paused" if paused else " running"
            behavior = "one redraw per input, then idle" if paused else "animation continues"
            for key, label, value in [(b"n", "decrease", 0.9), (b"m", "increase", 1.0)]:
                lsd_key(key, label, value, paused)
            rows.append(("LSD n/m" + suffix, "PASS", "visible 0.9 then 1.0; " + behavior))
            for key, label, value in [(b"n" * 40, "lower clamp", 0.0),
                                      (b"m" * 40, "upper clamp", 2.0),
                                      (b"N", "reset", 1.0)]:
                lsd_key(key, label, value, paused)
            rows.append(("LSD bounds/reset" + suffix, "PASS", "visible 0.0, 2.0, N restores 1.0; " + behavior))
            for keys, value in [("т", 0.9), ("ьь", 1.1), ("Т", 1.0)]:
                lsd_key(keys.encode("utf-8"), "Russian " + keys, value, paused)
            rows.append(("LSD Russian aliases" + suffix, "PASS", "т lowers, ь raises, Т resets to 1.0; " + behavior))

        lsd_controls()

        def speed_key(key: bytes, label: str, value: str, paused: bool = False) -> None:
            start = len(terminal.data)
            frames_before = terminal.frames
            terminal.send(key)
            expected = f"Speed: {value}x".encode("ascii")
            terminal.wait_for(
                lambda: expected in terminal.data[start:],
                f"speed HUD {value} after {label}",
                timeout=3.0,
            )
            if paused:
                terminal.drain(0.2)
                require(terminal.frames == frames_before + 1, "paused speed change did not redraw exactly once")
                require(not terminal.drain(0.3), "paused speed change restarted streaming")
            else:
                terminal.wait_frames()
            rows.append((f"speed {label}{' paused' if paused else ''}", "PASS", f"{expected.decode()}; {'one frame then idle' if paused else 'animation continues'}"))

        for key, label, value in [
            (b"[", "[", "2.30"), (b"]", "]", "2.40"),
            (b"\x1b[D", "Left", "2.30"), (b"\x1b[C", "Right", "2.40"),
            (b"]" * 5, "before upper bound", "2.90"),
            (b"]]", "upper bound", "3.00"), (b"[", "[", "2.90"),
            (b"s", "reset", "2.40"),
        ]:
            speed_key(key, label, value)

        def rain_key(key: bytes, label: str, rate: float, size: float, paused: bool = False) -> None:
            start = len(terminal.data)
            frames_before = terminal.frames
            terminal.send(key)
            expected = f"Rain: {rate:.1f}/s | Drop size: {size:.2f}x".encode("ascii")
            terminal.wait_for(lambda: expected in terminal.data[start:], f"rain HUD after {label}")
            if paused:
                terminal.drain(0.2)
                require(terminal.frames == frames_before + 1, "paused rain control did not redraw exactly once")
                require(not terminal.drain(0.3), "paused rain control restarted streaming")
            else:
                terminal.wait_frames()
            rows.append((f"rain {label}{' paused' if paused else ''}", "PASS",
                         expected.decode() + "; " + ("one frame then idle" if paused else "animation continues")))

        def rain_controls(paused: bool = False) -> None:
            for key, label, rate, size in [
                (b"\x1b[A", "Up", 34.0, 1.5),
                (b"\x1b[B", "Down", 32.0, 1.5),
                (b".", "larger", 32.0, 1.75),
                (b",", "smaller", 32.0, 1.5),
                (b"\x1b[A" * 40, "rate upper bound", 60.0, 1.5),
                (b"\x1b[B" * 40, "rate zero bound", 0.0, 1.5),
                (b"." * 20, "diameter upper bound", 0.0, 3.0),
                (b"," * 20, "diameter lower bound", 0.0, 0.25),
                (b"R", "reset", 32.0, 1.5),
            ]:
                rain_key(key, label, rate, size, paused)

        rain_controls()

        def bleach_key(key: bytes, label: str, value: float, paused: bool = False) -> None:
            start = len(terminal.data)
            frames_before = terminal.frames
            terminal.send(key)
            expected = f"Bleach: {value:.2f}x | v/b: less/more | B: reset".encode("ascii")
            terminal.wait_for(lambda: expected in terminal.data[start:], f"bleach HUD after {label}")
            if paused:
                terminal.drain(0.2)
                require(terminal.frames == frames_before + 1, "paused bleach control did not redraw exactly once")
                require(not terminal.drain(0.3), "paused bleach control restarted streaming")
            else:
                terminal.wait_frames()
            rows.append((f"bleach {label}{' paused' if paused else ''}", "PASS",
                         expected.decode() + "; " + ("one frame then idle" if paused else "animation continues")))

        def bleach_controls(paused: bool = False) -> None:
            for key, label, value in [
                (b"b", "faster", 1.25),
                (b"v", "slower", 1.0),
                (b"v" * 20, "off lower bound", 0.0),
                (b"b" * 24, "upper bound", 4.0),
                (b"B", "reset", 1.0),
            ]:
                bleach_key(key, label, value, paused)

        bleach_controls()

        def quality_key(key: bytes, mode: str, paused: bool = False) -> None:
            start = len(terminal.data)
            frames_before = terminal.frames
            expected = mode.encode("ascii")
            terminal.send(key)
            terminal.wait_for(
                lambda: b"Quality: " + expected in terminal.data[start:],
                f"quality HUD for {mode}",
            )
            labels = re.findall(rb"Quality: (Auto|High|Balanced|Low)\b", terminal.data[start:])
            require(labels and labels[-1] == expected, f"quality label mismatch: expected {mode}, got {labels!r}")
            if paused:
                terminal.drain(0.2)
                require(terminal.frames == frames_before + 1, "paused quality selection did not redraw exactly once")
                require(not terminal.drain(0.3), "paused quality selection restarted streaming")
                evidence = f"Quality: {mode}; exactly one frame, then idle"
            else:
                terminal.wait_frames()
                evidence = f"Quality: {mode}; animation continues"
            rows.append((f"quality {key.decode()}{' paused' if paused else ''}", "PASS", evidence))

        for mode in ["Balanced", "Low", "Auto", "High"]:
            quality_key(b"k", mode)
        quality_key(b"0", "Auto")
        for key, mode in [(b"1", "High"), (b"2", "Balanced"), (b"3", "Low"), (b"0", "Auto")]:
            quality_key(key, mode)

        def flow_key(mode: str, paused: bool = False, key: bytes = b"f") -> None:
            start = len(terminal.data)
            frames_before = terminal.frames
            terminal.send(key)
            expected = f"Water: {mode}".encode("ascii")
            terminal.wait_for(lambda: expected in terminal.data[start:], f"water HUD for {mode}")
            terminal.drain(0.2)
            if paused:
                require(terminal.frames == frames_before + 1, "paused water toggle did not redraw exactly once")
                require(not terminal.drain(0.3), "paused water toggle restarted streaming")
            else:
                terminal.wait_frames()
            rows.append((f"{key.decode()} water {mode}{' paused' if paused else ''}", "PASS",
                         expected.decode() + "; " + ("one frame then idle" if paused else "animation continues")))

        flow_key("Drain", key=b"d")
        flow_key("Classic", key=b"d")
        flow_key("Rapids")
        flow_key("Drain", key=b"d")
        flow_key("Rapids")
        for width, height in [(1, 1), (160, 50)]:
            start = len(terminal.data)
            terminal.resize(width, height)
            terminal.wait_for(lambda: SYNC_BEGIN + b"\x1b[2J" in terminal.data[start:],
                              f"rapids resize clear at {width}x{height}")
            terminal.wait_frames()
            rows.append((f"rapids resize {width}x{height}", "PASS", "clear and flowing frames; process remains alive"))

        def open_help(key: bytes = b"h", paused: bool = False, flow: str = "Rapids") -> bytes:
            start = len(terminal.data)
            terminal.send(key)
            terminal.wait_for(lambda: b"ACID RAIN / CONTROLS" in terminal.data[start:], "help screen")
            terminal.drain(0.15)
            screen = bytes(terminal.data[start:])
            for binding in [b"g / Tab", b"0 / 1 / 2 / 3", b"Left / Right", b"RGB echoes", b"Ctrl-C",
                            b"Up / Down", b", / .", b"Reset rain rate and drop size",
                            b"j / l", b"Reset RGB shift to 2.00px", b"n / m",
                            b"Reset subjective LSD to 1.0"]:
                require(binding in screen, f"help is missing {binding!r}")
            require(b"Rain: 32.0/s | Drop size: 1.50x" in screen, "help lost current rain controls")
            require(b"Trip: 2.0" in screen, "help lost the selected intensity")
            require(b"LSD: 1.0" in screen, "help lost the independent subjective LSD value")
            require(b"Speed: 2.40x" in screen, "help lost the selected speed")
            require(b"Bleach: 1.00x" in screen, "help lost the selected bleaching")
            require(b"RGB shift: 2.00px" in screen, "help lost the selected RGB shift")
            require(b"Animation held | Interpolation: on" in screen,
                    "help lost the enabled interpolation state")
            require(re.search(rb"i\s+[^\r\n\x1b]*intermediate-frame blending", screen),
                    "help is missing the i interpolation binding")
            for binding in [b"v / b", b"Reset bleaching", b"8s half-life"]:
                require(binding in screen, f"help is missing {binding!r}")
            require(f"Water: {flow}".encode() in screen, "help lost the selected water mode")
            require(re.search(rb"f\s+[^\r\n\x1b]*(?:rapids|Rapids)", screen), "help is missing the f water binding")
            require(re.search(rb"d\s+[^\r\n\x1b]*drain", screen), "help is missing the d water binding")
            state = b"paused" if paused else b"running"
            require(b"Previous state: " + state in screen, "help lost the pause state")
            return screen

        screen = open_help()
        artifact = artifact_dir / "help.ansi"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(screen[screen.rfind(SYNC_BEGIN):])
        cpu_before = cpu_seconds(terminal.child_pid)
        require(not terminal.drain(1.1), "help continued streaming animation")
        cpu_delta = max(0.0, cpu_seconds(terminal.child_pid) - cpu_before)
        require(cpu_delta <= 0.04, "help consumed excessive idle CPU")
        rows.append(("h help and idle", "PASS", f"all controls/settings; 0 bytes; {cpu_delta:.3f}s CPU / 1.1s idle"))
        terminal.resize(80, 10)
        terminal.wait_frames(1)
        for key in (b"\x1b[B", b"\x1b[A"):
            start = len(terminal.data)
            terminal.send(key)
            terminal.wait_frames(1)
            require(b"Rain: 32.0/s | Drop size: 1.50x" in terminal.data[start:],
                    "help navigation changed rain settings")
        rows.append(("help Up/Down scroll only", "PASS", "both arrows redraw help without changing rain settings"))
        start = len(terminal.data)
        terminal.send(b"\x1b[F")
        terminal.wait_for(lambda: b"q / Ctrl-C" in terminal.data[start:], "last help page")
        rows.append(("help scroll End", "PASS", "last bindings accessible at 80x10"))
        for width, height in [(1, 1), (160, 50)]:
            terminal.resize(width, height)
            terminal.wait_frames(1)
            terminal.drain(0.1)
            require(not terminal.drain(0.1), "resized help restarted animation")
            rows.append((f"help resize {width}x{height}", "PASS", "single static redraw"))
        terminal.send(b"\x1b")
        terminal.wait_frames(2)
        require(terminal.poll() is None, "Esc closed the app instead of help")
        rows.append(("Esc closes help", "PASS", "animation resumes and process remains alive"))
        flow_key("Classic")

        def echo_key(value: str, paused: bool = False) -> None:
            start = len(terminal.data)
            frames_before = terminal.frames
            terminal.send(b"e")
            expected = f"RGB echoes: {value}".encode()
            terminal.wait_for(lambda: expected in terminal.data[start:], "RGB echo toggle")
            terminal.drain(0.15)
            if paused:
                require(terminal.frames == frames_before + 1, "paused echo toggle did not redraw once")
                require(not terminal.drain(0.3), "paused echo toggle restarted streaming")
            else:
                terminal.wait_frames()
            rows.append((f"e RGB {value}{' paused' if paused else ''}", "PASS", "visible state; " + ("one frame then idle" if paused else "animation continues")))

        echo_key("off")
        echo_key("on")

        terminal.send(b" ")
        terminal.drain(0.4)
        cpu_before = cpu_seconds(terminal.child_pid)
        paused_frames = terminal.frames
        start = time.monotonic()
        # Exceed the three-second HUD lifetime: pausing must not wake up merely
        # to expire the message or restart streaming.
        idle_output = terminal.drain(3.2)
        elapsed = time.monotonic() - start
        cpu_delta = max(0.0, cpu_seconds(terminal.child_pid) - cpu_before)
        require(not idle_output and terminal.frames == paused_frames, "paused animation continued emitting terminal output")
        require(cpu_delta <= 0.06, f"paused process consumed {cpu_delta:.3f}s CPU in {elapsed:.3f}s")
        rows.append(("space pauses", "PASS", f"0 bytes, 0 frames, {cpu_delta:.3f}s CPU / {elapsed:.3f}s idle"))

        interpolation_key("off", paused=True)
        interpolation_key("on", paused=True)
        intensity_controls(paused=True)
        rgb_shift_controls(paused=True)
        lsd_controls(paused=True)
        bleach_controls(paused=True)
        flow_key("Drain", paused=True, key=b"d")
        flow_key("Classic", paused=True, key=b"d")
        flow_key("Rapids", paused=True)
        open_help(b"?", paused=True)
        before = terminal.frames
        terminal.send(b"h")
        terminal.wait_frames(1)
        terminal.drain(0.15)
        require(terminal.frames == before + 1, "paused help exit redrew more than once")
        require(not terminal.drain(0.3), "closing help lost the paused state")
        rows.append(("? help while paused", "PASS", "h restores paused state; single redraw then idle"))
        echo_key("off", paused=True)
        echo_key("on", paused=True)
        flow_key("Drain", paused=True, key=b"d")
        flow_key("Rapids", paused=True)
        flow_key("Classic", paused=True)

        terminal.send(b"g")
        terminal.wait_frames(1)
        terminal.drain(0.2)
        require(not terminal.drain(0.3), "paused scene change restarted streaming")
        rows.append(("input while paused", "PASS", "scene redraws once and returns to idle"))
        quality_key(b"1", "High", paused=True)
        quality_key(b"0", "Auto", paused=True)
        speed_key(b"[", "[", "2.30", paused=True)
        speed_key(b"s", "reset", "2.40", paused=True)
        rain_controls(paused=True)
        terminal.send(b" ")
        terminal.wait_frames()
        rows.append(("space resumes", "PASS", "frames restart"))
        terminal.finish(b"q")
        rows.append(("q cleanup", "PASS", f"exit 0; exact termios restored; screen/cursor/wrap restored; {terminal.frames} frames"))
    finally:
        terminal.close()

    terminal = TerminalProcess(binary, 320, 100)
    try:
        terminal.wait_frames(3)
        for value in ("off", "on"):
            start = len(terminal.data)
            sent = time.monotonic()
            terminal.send(b"i")
            expected = f"Frame blending: {value} | i: toggle".encode("ascii")
            terminal.wait_for(lambda: expected in terminal.data[start:],
                              f"fullscreen i {value}", timeout=1.0)
            latency_ms = (time.monotonic() - sent) * 1000.0
            terminal.wait_frames(2)
            rows.append((f"fullscreen i {value}", "PASS", f"visible state within {latency_ms:.1f}ms <= 1000ms"))
        start = len(terminal.data)
        sent = time.monotonic()
        terminal.send(b"h")
        terminal.wait_for(lambda: b"ACID RAIN / CONTROLS" in terminal.data[start:],
                          "fullscreen help barrier", timeout=1.0)
        require_startup_settings(bytes(terminal.data[start:]))
        help_latency = (time.monotonic() - sent) * 1000.0
        terminal.drain(0.15)
        require(not terminal.drain(0.4), "fullscreen worker wrote animation after help")
        rows.append(("fullscreen help stops worker", "PASS", f"help latency {help_latency:.1f}ms; then 0 bytes"))
        terminal.send(b"h")
        terminal.wait_frames(2)
        terminal.send(b" ")
        terminal.drain(0.4)
        paused_frames = terminal.frames
        require(not terminal.drain(0.4) and terminal.frames == paused_frames,
                "fullscreen worker continued displaying frames while paused")
        rows.append(("fullscreen pause stops worker", "PASS", "in-flight output settles; then 0 bytes/frames"))
        terminal.resize(0, 0)
        terminal.drain(0.15)
        require(terminal.poll() is None, "fullscreen worker exited on zero-size resize")
        require(not terminal.drain(0.3), "zero-size resize kept old-sized frames streaming")
        start = len(terminal.data)
        terminal.resize(160, 50)
        terminal.wait_for(lambda: SYNC_BEGIN + b"\x1b[2J" in terminal.data[start:],
                          "paused resize restores the new viewport", timeout=1.0)
        terminal.drain(0.15)
        require(not terminal.drain(0.3), "paused resize restarted the presentation worker")
        rows.append(("fullscreen zero-size/resize barrier", "PASS", "old frame history cleared; new viewport redraws and remains paused"))
        terminal.send(b" ")
        terminal.wait_frames(2)
        terminal.finish(b"q")
        rows.append(("fullscreen worker exit", "PASS", "exit 0; worker joined before screen/cursor/wrap and exact termios restoration"))
    finally:
        terminal.close()

    for mode in ("drain", "classic", "rapids"):
        terminal = TerminalProcess(binary, 80, 24, ("--water", mode))
        try:
            terminal.wait_frames()
            start = len(terminal.data)
            terminal.send(b"h")
            expected = f"Water: {mode.title()}".encode("ascii")
            terminal.wait_for(lambda: expected in terminal.data[start:], f"--water {mode} startup help")
            require_startup_settings(bytes(terminal.data[start:]), mode.title())
            terminal.finish(b"q")
            rows.append((f"--water {mode} startup", "PASS", expected.decode() + "; clean exit and exact terminal restoration"))
        finally:
            terminal.close()

    terminal = TerminalProcess(binary, 80, 24, ("--rapids", "--rain", "14", "--drop-size", "1.75", "--intensity", "4", "--bleach", "2.5", "--rgb-shift", "3.25", "--lsd", "1.7"))
    try:
        terminal.wait_frames()
        start = len(terminal.data)
        terminal.send(b"h")
        terminal.wait_for(lambda: b"ACID RAIN / CONTROLS" in terminal.data[start:], "help before Ctrl-C")
        require_startup_settings(bytes(terminal.data[start:]), "Rapids", rain=14.0, drop_size=1.75,
                                 bleach=2.5, rgb_shift=3.25, lsd=1.7)
        require(b"Water: Rapids" in terminal.data[start:], "--rapids startup did not select rapids")
        rows.append(("--rapids startup", "PASS", "frames animate; help reports Water: Rapids"))
        require(b"Rain: 14.0/s | Drop size: 1.75x" in terminal.data[start:],
                "--rain and --drop-size startup values missing from help")
        rows.append(("--rain/--drop-size startup", "PASS", "help reports Rain: 14.0/s and Drop size: 1.75x"))
        require(b"Trip: 4.0" in terminal.data[start:], "--intensity 4 startup value missing from help")
        rows.append(("--intensity 4 startup", "PASS", "help reports Trip: 4.0"))
        require(b"LSD: 1.7" in terminal.data[start:], "--lsd startup value missing from help")
        rows.append(("--lsd 1.7 startup", "PASS", "help reports LSD: 1.7 alongside independent Trip: 4.0"))
        require(b"RGB shift: 3.25px" in terminal.data[start:], "--rgb-shift startup value missing from help")
        rows.append(("--rgb-shift startup", "PASS", "help reports RGB shift: 3.25px alongside independent Trip: 4.0"))
        require(b"Bleach: 2.50x" in terminal.data[start:], "--bleach startup value missing from help")
        rows.append(("--bleach startup", "PASS", "help reports Bleach: 2.50x"))
        terminal.finish(b"\x03")
        rows.append(("Ctrl-C cleanup", "PASS", "exit 0; exact termios restored; screen/cursor/wrap restored"))
    finally:
        terminal.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/acid-rain")
    parser.add_argument("--report", type=Path, default=ROOT / "benchmarks/pty-smoke.txt")
    options = parser.parse_args()
    binary = options.binary.resolve()
    rows: list[tuple[str, str, str]] = []
    header = [
        "acid-rain headless PTY smoke test",
        f"UTC: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"Platform: {platform.platform()}",
        f"Python: {platform.python_version()}",
        f"Binary: {binary}",
    ]
    failed = False
    try:
        require(binary.is_file() and os.access(binary, os.X_OK), f"build the release executable first: {binary}")
        header.append(f"Binary SHA256: {hashlib.sha256(binary.read_bytes()).hexdigest()}")
        run_smoke(binary, rows, options.report.parent)
    except (AssertionError, OSError, termios.error, UnicodeError, subprocess.SubprocessError) as error:
        rows.append(("smoke test", "FAIL", str(error)))
        failed = True
    report = "\n".join(header + ["", "| Check | Result | Evidence |", "| --- | --- | --- |"])
    report += "\n" + "\n".join(f"| {check} | {result} | {evidence} |" for check, result, evidence in rows)
    report += f"\n\nOverall: {'FAIL' if failed else 'PASS'}\n"
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(report)
    print(report, end="")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
