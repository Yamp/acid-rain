#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["pillow>=10"]
# ///
"""Inspect actual native PNG captures and write conservative diagnostics."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

from PIL import Image, ImageChops, ImageStat, __version__ as pillow_version


HUES = ("red", "yellow", "green", "cyan", "blue", "magenta")
LIMITS = {"black_max_channel": 4, "white_min_channel": 245,
          "gray_max_channel_range": 12, "saturated_min_hsv_s": 90,
          "saturated_min_value": 32, "movement_min_channel_delta": 2}


def decode_png(path: Path) -> tuple[Image.Image, dict]:
    """Decode every pixel; verify alone would only check the container."""
    with Image.open(path) as source:
        if source.format != "PNG":
            raise ValueError(f"expected PNG, found {source.format}")
        source.load()
        if min(source.size) < 1:
            raise ValueError("empty image extent")
        metadata = {"source_mode": source.mode, "icc_profile_present": bool(source.info.get("icc_profile")),
                    "srgb_chunk": source.info.get("srgb"), "gamma_chunk": source.info.get("gamma")}
        rgb = source.convert("RGB")
    return rgb, metadata


def threshold(channel: Image.Image, predicate) -> Image.Image:
    return channel.point([255 if predicate(i) else 0 for i in range(256)])


def fraction(mask: Image.Image) -> float:
    return mask.histogram()[255] / (mask.width * mask.height)


def analyze_image(path: Path, expected_size=None) -> dict:
    path = Path(path)
    result = {"path": str(path.resolve()), "status": "PASS", "diagnostics": [], "errors": []}
    try:
        with path.open("rb") as stream:
            result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        rgb, metadata = decode_png(path)
        result.update(metadata, size=list(rgb.size), decoded_pixels=rgb.width * rgb.height)
        if expected_size and rgb.size != expected_size:
            result["errors"].append(f"expected dimensions {expected_size}, decoded {rgb.size}")
        red, green, blue = rgb.split()
        highest = ImageChops.lighter(ImageChops.lighter(red, green), blue)
        lowest = ImageChops.darker(ImageChops.darker(red, green), blue)
        spread = ImageChops.subtract(highest, lowest)
        hue, saturation, _ = rgb.convert("HSV").split()
        gray = threshold(spread, lambda n: n <= LIMITS["gray_max_channel_range"])
        black = threshold(highest, lambda n: n <= LIMITS["black_max_channel"])
        white = ImageChops.multiply(gray, threshold(lowest, lambda n: n >= LIMITS["white_min_channel"]))
        saturated = ImageChops.multiply(
            threshold(saturation, lambda n: n >= LIMITS["saturated_min_hsv_s"]),
            threshold(highest, lambda n: n >= LIMITS["saturated_min_value"]))
        hue_histogram = hue.histogram(mask=saturated)
        coverage = dict.fromkeys(HUES, 0)
        for value, count in enumerate(hue_histogram):
            coverage[HUES[int(value * 6 / 256 + .5) % 6]] += count
        coverage = {name: count / result["decoded_pixels"] for name, count in coverage.items()}
        # Encoded RGB statistics diagnose output appearance, not physical luminance.
        encoded_luma = rgb.convert("L", (.2126, .7152, .0722, 0))
        metrics = {
            "mean_rgb_encoded": ImageStat.Stat(rgb).mean,
            "mean_rec709_encoded_luma": ImageStat.Stat(encoded_luma).mean[0] / 255,
            "mean_hsv_saturation": ImageStat.Stat(saturation).mean[0] / 255,
            "mean_channel_range": ImageStat.Stat(spread).mean[0] / 255,
            "black_fraction": fraction(black), "white_fraction": fraction(white),
            "gray_fraction": fraction(gray), "saturated_fraction": fraction(saturated),
            "hue_coverage": coverage,
            "hue_bins_at_least_one_percent": sum(v >= .01 for v in coverage.values()),
        }
        scalar_values = [value for value in metrics.values() if isinstance(value, (int, float))]
        if not all(math.isfinite(value) for value in scalar_values):
            result["errors"].append("nonfinite derived image statistics")
        result.update(metrics=metrics, decoded_finite=True)
        if highest.getextrema()[1] == 0:
            result["errors"].append("entire output is exactly black")
        if metrics["black_fraction"] > .98:
            result["diagnostics"].append("more than 98% near-black pixels; inspect exposure or capture timing")
        if metrics["white_fraction"] > .90:
            result["diagnostics"].append("more than 90% near-white pixels; inspect white washout")
        if metrics["gray_fraction"] > .95:
            result["diagnostics"].append("more than 95% gray/neutral pixels; inspect the intended scene")
        if metrics["saturated_fraction"] < .05:
            result["diagnostics"].append("fewer than 5% bright saturated pixels; inspect color coverage")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        result["errors"].append(f"cannot decode valid PNG: {error}")
    if result["errors"]:
        result["status"] = "FAIL"
    return result


def compare_images(first: Path, second: Path) -> dict:
    result = {"first": str(Path(first).resolve()), "second": str(Path(second).resolve())}
    try:
        a, _ = decode_png(first)
        b, _ = decode_png(second)
        if a.size != b.size:
            return result | {"status": "SKIP", "reason": "different dimensions; no resampling"}
        difference = ImageChops.difference(a, b)
        red, green, blue = difference.split()
        largest = ImageChops.lighter(ImageChops.lighter(red, green), blue)
        changed = threshold(largest, lambda n: n > LIMITS["movement_min_channel_delta"])
        result.update(status="MEASURED", size=list(a.size),
                      mean_absolute_channel_delta=sum(ImageStat.Stat(difference).mean) / 3,
                      maximum_channel_delta=largest.getextrema()[1],
                      changed_pixel_fraction=fraction(changed),
                      identical=largest.getextrema()[1] == 0)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        result.update(status="SKIP", reason=f"invalid input: {error}")
    return result


def markdown_report(report: dict) -> str:
    lines = ["# Native Metal capture diagnostics", "", f"Result: **{report['status']}**. "
             "Image statistics describe captured files, not displayed frame rate or subjective visual quality.", "",
             "PNG decoding proves finite encoded pixels; it cannot detect NaN/Inf already hidden by HDR-to-PNG conversion.", "",
             "| Capture | Size | Status | Near black | Near white | Gray | Saturated | Mean saturation | Hue bins ≥1% |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in report["images"]:
        if "metrics" not in item:
            lines.append(f"| {Path(item['path']).name} | — | FAIL | — | — | — | — | — | — |")
            continue
        m = item["metrics"]
        size = "×".join(map(str, item["size"]))
        lines.append(f"| {Path(item['path']).name} | {size} | {item['status']} | "
                     f"{m['black_fraction']:.2%} | {m['white_fraction']:.2%} | {m['gray_fraction']:.2%} | "
                     f"{m['saturated_fraction']:.2%} | {m['mean_hsv_saturation']:.3f} | {m['hue_bins_at_least_one_percent']} |")
    if report["comparisons"]:
        lines += ["", "Explicit frame pairs (no implicit temporal ordering or FPS inference):", "",
                  "| First → second | Status | Mean channel delta (8-bit) | Pixels changed >2/255 | Max channel delta |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for pair in report["comparisons"]:
            label = f"{Path(pair['first']).name} → {Path(pair['second']).name}"
            if pair["status"] == "MEASURED":
                lines.append(f"| {label} | MEASURED | {pair['mean_absolute_channel_delta']:.4f} | "
                             f"{pair['changed_pixel_fraction']:.2%} | {pair['maximum_channel_delta']} |")
            else:
                lines.append(f"| {label} | {pair['status']}: {pair['reason']} | — | — | — |")
    lines += ["", "Thresholds are output diagnostics, not aesthetic pass/fail gates. "
              "Near black: all channels ≤4; near white: all ≥245 and spread ≤12; "
              "gray: channel spread ≤12; saturated: HSV S≥90/255 and V≥32/255."]
    for item in report["images"]:
        for kind in ("errors", "diagnostics"):
            for message in item[kind]:
                lines.append(f"\n- {Path(item['path']).name} — {kind}: {message}")
    for message in report["errors"]:
        lines.append(f"\n- Error: {message}")
    return "\n".join(lines) + "\n"


def dimensions(value: str) -> tuple[int, int]:
    try:
        width, height = map(int, value.lower().split("x"))
        if width > 0 and height > 0:
            return width, height
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected positive WIDTHxHEIGHT")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", help="PNG paths or quoted glob patterns; default native/artifacts/*.png")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/image-checks")
    parser.add_argument("--expect-size", type=dimensions, help="optional exact expected drawable extent")
    parser.add_argument("--compare", nargs=2, action="append", default=[], metavar=("FIRST", "SECOND"),
                        help="explicit ordered frame pair; may repeat")
    args = parser.parse_args()
    patterns = args.images or [str(Path(__file__).resolve().parents[1] / "artifacts/*.png")]
    paths = []
    errors = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            errors.append(f"no files match {pattern}")
        paths.extend(Path(path).resolve() for path in matches)
    # Explicit pairs also receive validation, even if outside the glob.
    paths.extend(Path(path).resolve() for pair in args.compare for path in pair)
    paths = list(dict.fromkeys(paths))
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "pillow": pillow_version,
              "thresholds": LIMITS, "errors": errors,
              "images": [analyze_image(path, args.expect_size) for path in paths],
              "comparisons": [compare_images(Path(a), Path(b)) for a, b in args.compare],
              "limits": ["Encoded PNG checks cannot prove finite upstream HDR values.",
                         "Frame differences do not establish animation timing or displayed FPS.",
                         "Color thresholds are diagnostics; only missing/invalid PNGs, requested extent mismatch, and exactly black output fail."]}
    report["status"] = "FAIL" if errors or any(item["status"] == "FAIL" for item in report["images"]) else "PASS"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    markdown = markdown_report(report)
    (args.output / "report.md").write_text(markdown)
    print(markdown)
    print(f"Saved {args.output.resolve()}/report.json and report.md")
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    sys.exit(main())
