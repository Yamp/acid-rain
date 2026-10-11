#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]
# ///
"""Render deterministic scene previews and assemble labeled contact sheets."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
SCENES = ("acid", "aurora", "storm", "abyss", "lava", "prism")
PREVIEW_SIZE = (600, 360)
BACKGROUND = (12, 16, 24)


def label_font(size: int):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def contact_sheet(items: list[tuple[str, Path]], columns: int, title: str, path: Path):
    margin, label_height, title_height = 20, 44, 60
    width, height = PREVIEW_SIZE
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (margin + columns * (width + margin), title_height + rows * (height + label_height + margin)),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 18), title, fill=(232, 238, 250), font=label_font(25))
    for index, (name, source) in enumerate(items):
        x = margin + index % columns * (width + margin)
        y = title_height + index // columns * (height + label_height + margin)
        draw.text((x, y + 7), name, fill=(174, 190, 218), font=label_font(21))
        with Image.open(source) as frame:
            # A PPM pixel corresponds to one rendered half-cell, so nearest
            # scaling preserves the terminal's actual spatial detail.
            preview = frame.convert("RGB").resize(PREVIEW_SIZE, Image.Resampling.NEAREST)
        sheet.paste(preview, (x, y + label_height))
    sheet.save(path)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/acid-rain")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--rapids", action="store_true", help="Preview experimental river rapids")
    parser.add_argument("--skip-render", action="store_true", help="Assemble existing PPM snapshots")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    items = []
    for scene in SCENES:
        snapshot = args.output / f"scene-{scene}.ppm"
        if not args.skip_render:
            subprocess.run(
                [str(args.binary), "--snapshot", str(snapshot), "--scene", scene,
                 "--size", "200x60", "--quality", "high", "--frames", str(args.frames)]
                + (["--rapids"] if args.rapids else []),
                check=True,
            )
        items.append((scene.upper(), snapshot))
    title = "ACID RAIN / RIVER RAPIDS" if args.rapids else "ACID RAIN / SIX WORLDS"
    contact_sheet(items, 3, title, args.output / "scenes.png")
    baseline = args.output / "baseline-200x60.ppm"
    if baseline.exists():
        contact_sheet(
            [("BEFORE / ORIGINAL", baseline), ("AFTER / ACID", items[0][1])],
            2,
            "ACID RAIN / 200 x 60 TERMINAL CELLS",
            args.output / "before-after.png",
        )


if __name__ == "__main__":
    main()
